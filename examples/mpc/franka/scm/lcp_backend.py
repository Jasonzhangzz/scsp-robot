"""Coupled contact LCPs in the SCM/MJWarp pipeline.

Two rigid contact laws share MJWarp collision, smooth dynamics and Euler
integration; only the contact impulse solve differs.

law="convex" (symmetric, the paper's eq. 21 target):
    0 <= p _|_ W p + g >= 0,  W = J M^-1 J.T + r I,  g = J v_free + d/h,
with p the impulses of the MJWarp friction-pyramid rows J = J_n +/- mu J_t.
This convex relaxation couples normal and tangential rows, so sliding
contacts can separate ("lift-off") even when the physical contact persists.

law="coulomb" (Stewart-Trinkle facet LCP, non-relaxed Coulomb friction):
    z = [c_n, beta, lambda],
    0 <= c_n    _|_ J_n v+ + d/h              >= 0   (non-penetration)
    0 <= beta_k _|_ (r_k - J_n) v+ + lambda_i >= 0   (maximum dissipation facets)
    0 <= lambda _|_ c_n - sum_k beta_k         >= 0   (pyramid friction bound)
    v+ = v_free + M^-1 (J_n.T c_n + sum_k (r_k - J_n).T beta_k),
where r_k are the MJWarp pyramid rows of a contact and J_n = (r_0 + r_1)/2.
The facet directions r_k - J_n = +/- mu J_t are exactly the MuJoCo pyramid
geometry, but normal and tangential complementarity are kept separate, so a
sliding contact stays in contact and Coulomb's law holds to solver tolerance.
The LCP matrix is non-symmetric and copositive-plus; it is solved by Lemke.

d is the contact gap used for the position term: d=phi (signed, Baumgarte-like
pop-out of residual penetration) or d=max(phi,0) (velocity-level
non-penetration, no pop-out), selected by gap_policy.

Each small single-world LCP is solved by the unmodified QuantEcon v0.11.2
Lemke library in float64 on CPU and certified on the original unscaled
problem; this backend is an accuracy reference, not a GPU solver claim.
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


def pyramid_contacts(dims, addresses, row_count):
    """Group MJWarp efc rows by contact: list of (dim, row ids).

    dim==1 contacts own one normal row; dim>=3 contacts own 2*(dim-1)
    pyramid rows in +/- pairs. Every efc row must belong to a contact
    (the reference bodies have no joint limits or equality constraints).
    """
    contacts, covered = [], np.zeros(row_count, dtype=bool)
    for dim, address in zip(np.asarray(dims, dtype=int), np.asarray(addresses, dtype=int)):
        if address < 0:
            continue
        rows = 1 if dim == 1 else 2 * (dim - 1)
        ids = np.arange(address, address + rows)
        if ids[-1] >= row_count or covered[ids].any():
            raise ValueError("contact efc rows exceed or overlap the constraint buffer")
        covered[ids] = True
        contacts.append((int(dim), ids))
    if not covered.all():
        raise ValueError("Coulomb reference requires every constraint row to be a contact row")
    return contacts


def coulomb_problem(J, gap, free, inverse, h, regularization, contacts, nonnegative_gap=True):
    """Stewart-Trinkle facet LCP (M, q) built from MJWarp pyramid rows.

    Returns the problem together with a decoder that maps the solution
    z=[c_n, beta, lambda] back to impulses of the native pyramid rows, so the
    generalized impulse J.T p is unchanged and the MJWarp force decoding
    (normal = sum of pair rows, tangential = mu*(p_+ - p_-)) still applies.
    """
    J, gap, free = (np.asarray(J, dtype=np.float64), np.asarray(gap, dtype=np.float64),
                    np.asarray(free, dtype=np.float64))
    normals, facets, owner, depth, layout = [], [], [], [], []
    for index, (dim, ids) in enumerate(contacts):
        rows = J[ids]
        if dim == 1:
            normal, directions = rows[0], np.zeros((0, J.shape[1]))
        else:
            pairs = rows.reshape(-1, 2, J.shape[1])
            sums = pairs.sum(axis=1)
            scale = max(1.0, float(np.abs(sums).max()))
            if np.abs(sums - sums[0]).max() > 1e-4 * scale:
                raise ValueError("pyramid pairs of one contact disagree on the normal row")
            normal = sums[0] / 2
            directions = rows - normal
        normals.append(normal)
        facets.extend(directions)
        owner.extend([index] * len(directions))
        depth.append(gap[ids[0]])
        layout.append((ids, len(directions)))
    C, F = len(normals), len(facets)
    Jn = np.asarray(normals, dtype=np.float64).reshape(C, J.shape[1])
    Jd = np.asarray(facets, dtype=np.float64).reshape(F, J.shape[1])
    owner = np.asarray(owner, dtype=int)
    A = np.vstack([Jn, Jd])
    W = A @ inverse @ A.T + regularization * np.eye(C + F)
    W = (W + W.T) / 2
    E = np.zeros((F, C))
    E[np.arange(F), owner] = 1.0
    n = 2 * C + F
    M = np.zeros((n, n))
    M[:C + F, :C + F] = W
    M[C:C + F, C + F:] = E
    M[C + F:, :C] = np.eye(C)
    M[C + F:, C:C + F] = -E.T
    d = np.asarray(depth, dtype=np.float64)
    if nonnegative_gap:
        d = np.maximum(d, 0.0)
    q = np.concatenate([Jn @ free + d / h, Jd @ free, np.zeros(C)])

    def decode(z):
        z = np.asarray(z, dtype=np.float64)
        c_n, beta = z[:C], z[C:C + F]
        p = np.zeros(J.shape[0])
        start = 0
        for index, (ids, count) in enumerate(layout):
            if count == 0:
                p[ids] = c_n[index]
            else:
                block = beta[start:start + count]
                slack = max(c_n[index] - block.sum(), 0.0)
                p[ids] = block + slack / count
            start += count
        return p

    return {"M": M, "q": q, "W": W, "Jn": Jn, "Jd": Jd, "owner": owner, "depth": d,
            "normals": C, "facets": F, "decode": decode}


def solve_coulomb_lcp(M, q, tolerance=1e-8, max_iterations=2000):
    """Lemke solve of the (non-symmetric) Stewart-Trinkle LCP with certificate.

    Positive diagonal scaling preserves complementarity. Rows of the W block
    are scaled by sqrt(W_ii); the friction-bound rows (zero diagonal) are
    scaled by 1/sqrt(W_nn,ii) so the identity coupling to c_n stays O(1).
    """
    M, q = np.asarray(M, dtype=np.float64), np.asarray(q, dtype=np.float64)
    n = q.size
    if q.ndim != 1 or M.shape != (n, n) or not np.isfinite(M).all() or not np.isfinite(q).all():
        raise ValueError("LCP requires a finite square matrix and matching vector")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("LCP tolerance and iteration limit must be positive")
    if not n:
        z = np.zeros(0)
        certificate = _certificate(M, q, z, 0, tolerance, "quantecon-lemke-coulomb")
        certificate.update({"attempt": "empty", "lemke_status": 0})
        return z, certificate
    diagonal = np.diag(M)
    bound_rows = diagonal <= 0
    C = int(bound_rows.sum())

    def scaling(matrix):
        scale = np.ones(n)
        scale[~bound_rows] = np.sqrt(np.diag(matrix)[~bound_rows])
        if C:
            # Friction-bound row i pairs with normal unknown i (M[C+F+i, i] = 1).
            scale[bound_rows] = 1.0 / scale[:C]
        return scale

    # Resting contacts are degenerate for Lemke (opposed facet pair impulses
    # cancel; hyperstatic normal distribution), so the W block is singular up
    # to r I. Pivot settings and the proximal stage only steer the search;
    # acceptance is decided solely by the complementarity certificate on the
    # ORIGINAL unscaled problem, so every accepted impulse solves the same
    # Coulomb LCP to tolerance.
    attempts = []
    for label, scale in (("scaled", scaling(M)), ("unscaled", np.ones(n))):
        for tol_piv in (1e-12, 1e-9):
            attempts.append((f"{label}; tol_piv={tol_piv:g}", scale, tol_piv, 0.0))
        for perturbation in (1e-10, 1e-9):
            attempts.append((f"{label}; q perturbed {perturbation:g}", scale, 1e-12, perturbation))
    best = None
    for label, scale, tol_piv, perturbation in attempts:
        q_run = q.copy()
        q_run[C:] += perturbation  # facet and friction-bound rows only
        z, status, iterations = _lemke(M, q_run, scale, tol_piv, max_iterations)
        if z is None:
            continue
        z = _polish(M, q, z)
        residual = _residual(M, q, z)
        if best is None or residual < best[0]:
            best = (residual, z, iterations, label, status)
        if residual <= tolerance:
            break
    if best is None or best[0] > tolerance:
        # Proximal point on the Delassus block: solve LCP(M + delta P, q - delta P z_k)
        # repeatedly; its fixed point solves the original LCP, each inner problem is
        # well conditioned, and the residual on the original problem is certified.
        prox_mask = np.zeros(n)
        prox_mask[~bound_rows] = 1.0
        for delta in (1e-4, 1e-3, 1e-2):
            shifted = M + delta * np.diag(prox_mask)
            scale = scaling(shifted)
            z = np.zeros(n)
            total = 0
            for iteration in range(1, 101):
                candidate, status, iterations = _lemke(shifted, q - delta * prox_mask * z, scale,
                                                       1e-12, max_iterations)
                if candidate is None:
                    break
                total += iterations
                z = _polish(M, q, candidate)
                residual = _residual(M, q, z)
                if best is None or residual < best[0]:
                    best = (residual, z, total, f"proximal delta={delta:g}; {iteration} outer", status)
                if residual <= tolerance:
                    break
            if best is not None and best[0] <= tolerance:
                break
    if best is None:
        raise FloatingPointError("QuantEcon Coulomb LCP produced no finite candidate")
    residual, z, iterations, label, status = best
    if residual > tolerance:
        raise FloatingPointError(f"Coulomb LCP residual {residual:.3e} > {tolerance:.3e} after all pivot settings")
    certificate = _certificate(M, q, z, iterations, tolerance, "quantecon-lemke-coulomb")
    certificate.update({"attempt": label, "lemke_status": status})
    return z, certificate


def _lemke(M, q, scale, tol_piv, max_iterations):
    """One QuantEcon Lemke run on the diagonally scaled problem; None on breakdown."""
    if __package__:
        from .vendor.quantecon_lcp import lcp_lemke
        from .vendor.quantecon_lcp.linprog_simplex import PivOptions
    else:
        from vendor.quantecon_lcp import lcp_lemke
        from vendor.quantecon_lcp.linprog_simplex import PivOptions
    try:
        result = lcp_lemke(np.ascontiguousarray(M / scale[:, None] / scale[None, :]),
                           np.ascontiguousarray(q / scale), max_iter=max_iterations,
                           piv_options=PivOptions(tol_piv=tol_piv, tol_ratio_diff=tol_piv / 10))
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return None, -1, 0
    z = np.asarray(result.z, dtype=np.float64) / scale
    if not np.isfinite(z).all():
        return None, int(result.status), int(result.num_iter)
    return np.maximum(z, 0), int(result.status), int(result.num_iter)


def _residual(M, q, z):
    w = M @ z + q
    return max(float(np.maximum(-z, 0).max(initial=0)), float(np.maximum(-w, 0).max(initial=0)),
               float(np.abs(z * w).max(initial=0)))


def _polish(M, q, z):
    """Re-solve the basic variables identified by Lemke on the unscaled problem.

    Lemke pivots to ~1e-12 on the scaled problem; the identified active set is
    exact, so one linear solve on M_BB z_B = -q_B removes the round-off. The
    polished point is used only when it is feasible and reduces the residual.
    """
    w = M @ z + q
    basic = z > w
    if not basic.any():
        return z
    ids = np.flatnonzero(basic)
    try:
        candidate = np.zeros_like(z)
        candidate[ids] = np.linalg.solve(M[np.ix_(ids, ids)], -q[ids])
    except np.linalg.LinAlgError:
        return z
    if not np.isfinite(candidate).all():
        return z
    candidate = np.where(np.abs(candidate) < 1e-13, 0.0, candidate)
    if candidate.min(initial=0) < 0:
        return z
    return candidate if _residual(M, q, candidate) < _residual(M, q, z) else z


class LcpEngine(ScmEngine):
    def __init__(self, model, initial_data, *, regularization=1e-8,
                 tolerance=1e-8, max_iterations=2000, solver="lemke",
                 law="convex", gap_policy="signed"):
        super().__init__(model, initial_data, regularization=regularization,
                         motion="dynamic", friction="point")
        if law not in ("convex", "coulomb"):
            raise ValueError(f"Unknown LCP contact law: {law}")
        if gap_policy not in ("signed", "nonnegative"):
            raise ValueError(f"Unknown LCP gap policy: {gap_policy}")
        if law == "coulomb" and solver != "lemke":
            raise ValueError("The Coulomb facet LCP is non-symmetric; only the Lemke solver applies")
        self.tolerance, self.max_iterations = tolerance, max_iterations
        self.solver, self.law, self.gap_policy = solver, law, gap_policy
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
        depth = np.maximum(gap, 0.0) if self.gap_policy == "nonnegative" else gap
        if self.law == "convex":
            W = J @ inverse @ J.T + self.regularization * np.eye(count)
            g = J @ free + depth / h
            impulse, self.last_lcp = solve_lcp(W, g, self.tolerance, self.max_iterations, self.solver)
            self.last_problem = {"W": W, "g": g, "J": J, "law": "convex",
                                 "inverse_mass": inverse, "free_velocity": free, "impulse": impulse}
        else:
            nacon = int(d.nacon.numpy()[0])
            contacts = pyramid_contacts(d.contact.dim.numpy()[:nacon],
                                        d.contact.efc_address.numpy()[:nacon, 0], count)
            problem = coulomb_problem(J, gap, free, inverse, h, self.regularization, contacts,
                                      nonnegative_gap=self.gap_policy == "nonnegative")
            z, self.last_lcp = solve_coulomb_lcp(problem["M"], problem["q"],
                                                 self.tolerance, self.max_iterations)
            impulse = problem["decode"](z)
            C, F = problem["normals"], problem["facets"]
            facet_sum = np.bincount(problem["owner"], weights=z[C:C + F], minlength=C) if F else np.zeros(C)
            self.last_lcp.update({"normals": C, "facets": F,
                                  "friction_bound_slack": float(np.min(z[:C] - facet_sum, initial=0.0))})
            self.last_problem = {"W": problem["W"], "M": problem["M"], "q": problem["q"], "z": z,
                                 "g": J @ free + depth / h, "J": J, "law": "coulomb",
                                 "Jn": problem["Jn"], "Jd": problem["Jd"], "owner": problem["owner"],
                                 "inverse_mass": inverse, "free_velocity": free, "impulse": impulse}
        # Native force decoding uses the same pyramid-row layout as SCM.
        forces = np.zeros(d.efc.force.shape, dtype=np.float32)
        forces[0, :count] = impulse / h
        generalized = np.asarray((J.T @ impulse / h)[None], dtype=np.float32)
        wp.copy(d.efc.force, wp.array(forces, dtype=d.efc.force.dtype))
        wp.copy(d.qfrc_constraint, wp.array(generalized, dtype=d.qfrc_constraint.dtype))
        self.finish_step(m, d)
