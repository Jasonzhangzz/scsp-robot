"""Coupled, regularized friction-pyramid LCP in the SCM/MJWarp pipeline.

0 <= p _|_ W p + g >= 0,
W = J M^-1 J.T + r I, g = J v_free + phi/h.

Unlike diagonal SCM, the LCP retains coupling between all contact rows.
Collision, smooth dynamics and Euler integration use the common GPU path.
The small single-world LCP is solved by the unmodified QuantEcon v0.11.2
Lemke library in float64 on CPU; this backend is
an accuracy reference, not a claim of GPU solver throughput.
"""

import numpy as np
import warp as wp

if __package__:
    from .scm_backend import ScmEngine
else:
    from scm_backend import ScmEngine


def _problem(matrix, offset, tolerance, max_iterations):
    W, g = np.asarray(matrix, dtype=np.float64), np.asarray(offset, dtype=np.float64)
    n = g.size
    if g.ndim != 1 or W.shape != (n, n) or not np.isfinite(W).all() or not np.isfinite(g).all():
        raise ValueError("LCP requires a finite square matrix and matching vector")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("LCP tolerance and iteration limit must be positive")
    if not np.allclose(W, W.T, atol=1e-10, rtol=1e-10):
        raise ValueError("LCP matrix must be symmetric positive definite")
    W = (W + W.T) / 2
    if n:
        np.linalg.cholesky(W)
    return W, g


def _certificate(W, g, p, iterations, tolerance, solver):
    if not np.isfinite(p).all():
        raise FloatingPointError("Nonfinite LCP impulse")
    w = W @ p + g
    primal = float(np.maximum(-p, 0).max(initial=0))
    dual = float(np.maximum(-w, 0).max(initial=0))
    complementarity = float(np.abs(p * w).max(initial=0))
    residual = max(primal, dual, complementarity)
    if residual > tolerance:
        raise FloatingPointError(f"LCP complementarity residual {residual:.3e} > {tolerance:.3e}")
    return {"residual": residual, "primal_violation": primal, "dual_violation": dual,
            "complementarity": complementarity, "iterations": iterations,
            "rows": g.size, "solver": solver}


def solve_lcp_active_set(matrix, offset, tolerance=1e-8, max_iterations=2000):
    """Independent feasible active-set QP solve for cross-validation."""
    W, g = _problem(matrix, offset, tolerance, max_iterations)
    n = g.size
    p, passive = np.zeros(n), np.zeros(n, dtype=bool)
    iterations = 0
    while iterations < max_iterations:
        w = W @ p + g
        violated = np.flatnonzero(~passive & (w < -tolerance))
        if not len(violated):
            return p, _certificate(W, g, p, iterations, tolerance, "active-set")
        passive[violated[np.argmin(w[violated])]] = True
        while True:
            iterations += 1
            if iterations > max_iterations:
                raise FloatingPointError("LCP active-set iteration limit exceeded")
            ids = np.flatnonzero(passive)
            candidate = np.zeros(n)
            candidate[ids] = np.linalg.solve(W[np.ix_(ids, ids)], -g[ids])
            if np.all(candidate[ids] > 0):
                p = candidate
                break
            bounded = ids[candidate[ids] <= 0]
            alpha = float(np.min(p[bounded] / (p[bounded] - candidate[bounded])))
            p += alpha * (candidate - p)
            released = passive & (p <= 1e-14)
            p[released], passive[released] = 0., False
    raise FloatingPointError("LCP active-set iteration limit exceeded")


def solve_lcp(matrix, offset, tolerance=1e-8, max_iterations=2000, solver="lemke"):
    """QuantEcon Lemke solve; independently certify the ORIGINAL unscaled LCP.

    Positive diagonal scaling improves pivot conditioning without changing
    complementarity: y=S*p, W_scaled=S^-1*W*S^-1, g_scaled=S^-1*g.
    A library success flag alone is insufficient: check all three residuals.
    The active-set option is explicit and is never a silent fallback.
    """
    if solver == "active-set":
        return solve_lcp_active_set(matrix, offset, tolerance, max_iterations)
    if solver != "lemke":
        raise ValueError(f"Unknown LCP solver: {solver}")
    W, g = _problem(matrix, offset, tolerance, max_iterations)
    if not g.size:
        p = np.zeros(0)
        return p, _certificate(W, g, p, 0, tolerance, "quantecon-lemke")
    if __package__:
        from .vendor.quantecon_lcp import lcp_lemke
        from .vendor.quantecon_lcp.linprog_simplex import PivOptions
    else:
        from vendor.quantecon_lcp import lcp_lemke
        from vendor.quantecon_lcp.linprog_simplex import PivOptions
    scale = np.sqrt(np.diag(W))
    result = lcp_lemke(np.ascontiguousarray(W / scale[:, None] / scale[None, :]),
                       np.ascontiguousarray(g / scale), max_iter=max_iterations,
                       piv_options=PivOptions(tol_piv=1e-12, tol_ratio_diff=1e-13))
    if not result.success or result.status != 0:
        reasons = {1: "iteration limit", 2: "secondary ray termination", 3: "numerical breakdown"}
        raise FloatingPointError(f"QuantEcon LCP failed: {reasons.get(result.status, result.status)}")
    p = result.z / scale
    if p.min(initial=0) < -tolerance:
        raise FloatingPointError("QuantEcon returned negative LCP impulses")
    p = np.maximum(p, 0)
    return p, _certificate(W, g, p, int(result.num_iter), tolerance, "quantecon-lemke")


class LcpEngine(ScmEngine):
    def __init__(self, model, initial_data, *, regularization=1e-8,
                 tolerance=1e-8, max_iterations=2000, solver="lemke"):
        super().__init__(model, initial_data, regularization=regularization,
                         motion="dynamic", friction="point")
        self.tolerance, self.max_iterations = tolerance, max_iterations
        self.solver = solver
        self.last_lcp = None

    def step(self, m, d):
        if d.nworld != 1:
            raise ValueError("LCP accuracy backend supports one world")
        self.prepare_step(m, d)
        count = int(d.nefc.numpy()[0])
        if count > d.efc.J.shape[1]:
            raise RuntimeError("LCP constraint buffer overflow; increase --njmax")
        J = d.efc.J.numpy()[0, :count, :m.nv].astype(np.float64)
        gap = d.efc.efc_dist.numpy()[0, :count].astype(np.float64)
        free = d.qvel_smooth_pred.numpy()[0, :m.nv].astype(np.float64)
        inverse = (np.diag(self.inverse_diagonal.numpy().astype(np.float64)) if self.use_diagonal
                   else self.inverse_dense.numpy().astype(np.float64))
        h = float(m.opt.timestep.numpy()[0])
        W = J @ inverse @ J.T + self.regularization * np.eye(count)
        impulse, self.last_lcp = solve_lcp(W, J @ free + gap / h,
                                          self.tolerance, self.max_iterations, self.solver)
        self.last_problem = {"W": W, "g": J @ free + gap / h, "J": J,
                             "inverse_mass": inverse, "free_velocity": free, "impulse": impulse}
        # Native force decoding uses the same pyramid-row layout as SCM.
        forces = np.zeros(d.efc.force.shape, dtype=np.float32)
        forces[0, :count] = impulse / h
        generalized = np.asarray((J.T @ impulse / h)[None], dtype=np.float32)
        wp.copy(d.efc.force, wp.array(forces, dtype=d.efc.force.dtype))
        wp.copy(d.qfrc_constraint, wp.array(generalized, dtype=d.qfrc_constraint.dtype))
        self.finish_step(m, d)
