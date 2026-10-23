"""Two smooth replacements for SCM's global amplitude clip.

Both keep the diagonal Delassus impulse as the initial estimate and replace
the shared scale ``clip(alpha, 0, 1)`` with one smooth residual step on the
coupled Delassus operator

    (W beta)_i = (J M^{-1} (J^T beta))_i + epsilon beta_i.

``spectral`` is the minimal revision: step length ``1/lambda_max(W)``, with
``lambda_max`` from power iteration on that matvec.

``preconditioned`` is the Jacobi revision: step ``omega D^{-1}`` with

    omega = 1 / L_N,
    L_N = max_i sum_j |W_ij| / sqrt(D_ii D_jj),
    D = diag(W).

``L_N`` is a Gershgorin upper bound on ``lambda_max(D^{-1/2} W D^{-1/2})``
and does not need an eigendecomposition. ``D`` and the step length are fixed
for the contact geometry of the step and reused across ``correction_steps``
applications. The published model uses one application.

Impulses stay in the same units as the prediction benchmark:
``g = J v_projected + max(gap, 0) / h`` and ``beta = h * lambda``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

if __package__:
    from .scm_backend import ScmEngine
else:
    from scm_backend import ScmEngine


def softplus(values, temperature):
    """``T log(1 + exp(x/T))``, with the hard positive part at ``T = 0``."""
    values = np.asarray(values, dtype=np.float64)
    if temperature == 0:
        return np.maximum(values, 0.0)
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("softplus temperature must be finite and nonnegative")
    return temperature * np.logaddexp(0.0, values / temperature)


def apply_delassus(jacobian, inverse_mass, regularization, impulse):
    """``J M^{-1} (J^T beta) + epsilon beta`` without storing ``W``."""
    impulse = np.asarray(impulse, dtype=np.float64)
    if impulse.shape != (jacobian.shape[0],):
        raise ValueError("impulse length must match the constraint rows")
    generalized = inverse_mass @ (jacobian.T @ impulse)
    return jacobian @ generalized + regularization * impulse


def coupled_delassus(jacobian, inverse_mass, regularization):
    """Dense ``W`` for the Gershgorin bound. The impulse update does not need it."""
    rows = jacobian.shape[0]
    if rows == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return jacobian @ inverse_mass @ jacobian.T + regularization * np.eye(rows)


def diagonal_delassus(jacobian, inverse_mass, regularization):
    diagonal = np.einsum("ij,jk,ik->i", jacobian, inverse_mass, jacobian)
    return np.asarray(diagonal, dtype=np.float64) + regularization


def spectral_radius(jacobian, inverse_mass, regularization, *, max_iterations=200, tolerance=1e-12):
    """Power iteration for ``lambda_max(W)``.

    The start vector is the largest diagonal entry plus a full-support
    perturbation, so an eigenvector orthogonal to the all-ones vector is still
    visible. Returns ``(lambda_max, iterations, source)``. A dense
    decomposition is used only when the Rayleigh pair does not certify the
    largest eigenvalue.
    """
    rows = jacobian.shape[0]
    if rows == 0:
        return 0.0, 0, "empty"
    diagonal = diagonal_delassus(jacobian, inverse_mass, regularization)
    start = 0.01 * np.ones(rows)
    start[int(np.argmax(diagonal))] += 1.0
    value, iterations, residual = _power_iterate(
        jacobian, inverse_mass, regularization, start, max_iterations, tolerance)
    lower = float(np.max(diagonal))
    certified = (value + 1e-9 * max(1.0, lower) >= lower
                 and residual <= 1e-8 * max(1.0, abs(value)))
    if certified:
        return value, iterations, "power"
    dense = coupled_delassus(jacobian, inverse_mass, regularization)
    value = float(np.linalg.eigvalsh(dense)[-1])
    return value, iterations, "eigvalsh"


def _power_iterate(jacobian, inverse_mass, regularization, start, max_iterations, tolerance):
    vector = np.asarray(start, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("power iteration requires a nonzero start")
    vector /= norm
    rayleigh = 0.0
    residual = np.inf
    used = 0
    for used in range(1, max_iterations + 1):
        image = apply_delassus(jacobian, inverse_mass, regularization, vector)
        updated = float(vector @ image)
        image_norm = float(np.linalg.norm(image))
        residual = float(np.linalg.norm(image - updated * vector))
        converged = abs(updated - rayleigh) <= tolerance * max(1.0, abs(updated))
        eigen_residual = residual <= 1e-8 * max(1.0, abs(updated))
        if image_norm == 0 or (converged and eigen_residual):
            return updated, used, residual
        vector = image / image_norm
        rayleigh = updated
    image = apply_delassus(jacobian, inverse_mass, regularization, vector)
    updated = float(vector @ image)
    residual = float(np.linalg.norm(image - updated * vector))
    return updated, used, residual


def normalized_gershgorin(matrix, diagonal):
    """``max_i sum_j |W_ij| / sqrt(D_ii D_jj)``."""
    diagonal = np.asarray(diagonal, dtype=np.float64)
    if matrix.shape != (diagonal.size, diagonal.size):
        raise ValueError("Gershgorin matrix and diagonal disagree")
    if diagonal.size == 0:
        return 0.0
    if np.any(diagonal <= 0):
        raise ValueError("Delassus diagonal must be positive")
    scale = np.sqrt(diagonal)
    normalized = np.abs(matrix) / scale[:, None] / scale[None, :]
    return float(np.max(np.sum(normalized, axis=1)))


def environment_impulses(jacobian, inverse_mass, projected_velocity, gap, timestep,
                         regularization, temperature, correction, steps, *,
                         nonnegative_gap=True, max_iterations=200, tolerance=1e-12):
    """Diagonal softplus estimate followed by ``steps`` residual corrections.

    ``correction`` is ``spectral`` or ``preconditioned``. ``steps = 0`` returns
    the diagonal estimate and leaves the step length undefined.
    """
    if correction not in ("spectral", "preconditioned"):
        raise ValueError(f"unknown residual correction: {correction}")
    if steps < 0:
        raise ValueError("correction steps must be nonnegative")
    jacobian = np.asarray(jacobian, dtype=np.float64)
    inverse_mass = np.asarray(inverse_mass, dtype=np.float64)
    rows, dofs = jacobian.shape
    if inverse_mass.shape != (dofs, dofs):
        raise ValueError("inverse mass must match the Jacobian columns")
    gap = np.asarray(gap, dtype=np.float64)
    projected = np.asarray(projected_velocity, dtype=np.float64)
    if gap.shape != (rows,) or projected.shape != (dofs,):
        raise ValueError("gap and projected velocity do not match the Jacobian")
    if timestep <= 0 or regularization <= 0:
        raise ValueError("timestep and Delassus regularization must be positive")
    distance = np.maximum(gap, 0.0) if nonnegative_gap else gap
    info = {"power_iterations": 0, "spectral_source": None, "lipschitz": None, "step_size": None,
            "initial": np.zeros(rows), "diagonal": np.zeros(rows)}
    if rows == 0:
        return np.zeros(0), info
    linear = jacobian @ projected + distance / timestep
    info["linear"] = linear
    if correction == "preconditioned" and steps > 0:
        # D and L_N come from one dense W. Later products still use the matvec.
        diagonal = np.diag(coupled_delassus(jacobian, inverse_mass, regularization)).copy()
    else:
        diagonal = diagonal_delassus(jacobian, inverse_mass, regularization)
    beta = softplus(-linear / diagonal, temperature)
    info["initial"] = beta.copy()
    info["diagonal"] = diagonal
    if steps == 0:
        return beta, info
    if correction == "spectral":
        lipschitz, used, source = spectral_radius(
            jacobian, inverse_mass, regularization, max_iterations=max_iterations, tolerance=tolerance)
        info.update(power_iterations=used, spectral_source=source)
    else:
        lipschitz = normalized_gershgorin(
            coupled_delassus(jacobian, inverse_mass, regularization), diagonal)
        info["spectral_source"] = "gershgorin"
    if not np.isfinite(lipschitz) or lipschitz <= 0:
        raise FloatingPointError(f"nonpositive residual step scale: {lipschitz}")
    step = 1.0 / lipschitz
    for _ in range(steps):
        residual = apply_delassus(jacobian, inverse_mass, regularization, beta) + linear
        if correction == "spectral":
            trial = beta - step * residual
        else:
            trial = beta - step * residual / diagonal
        beta = softplus(trial, temperature)
    if not np.isfinite(beta).all():
        raise FloatingPointError("nonfinite residual impulse")
    info.update(lipschitz=lipschitz, step_size=step)
    return beta, info


class ResidualScmEngine(ScmEngine):
    """SCM prediction with a residual correction instead of the amplitude clip."""

    def __init__(self, model, initial_data, *, regularization=1e-8, motion="dynamic",
                 friction="point", softplus_temperature=1e-4, correction="spectral",
                 correction_steps=1, power_iterations=200):
        if correction not in ("spectral", "preconditioned"):
            raise ValueError(f"unknown residual correction: {correction}")
        if correction_steps < 0:
            raise ValueError("correction steps must be nonnegative")
        if not np.isfinite(softplus_temperature) or softplus_temperature <= 0:
            raise ValueError("residual SCM requires a positive finite softplus temperature (N s)")
        super().__init__(model, initial_data, regularization=regularization, motion=motion,
                         friction=friction, amplitude_correction=False, response_time=0.0,
                         damping_ratio=1.0, softplus_temperature=0.0)
        self.softplus_temperature = float(softplus_temperature)
        self.nonnegative_gap = True
        self.correction = correction
        self.correction_steps = int(correction_steps)
        self.power_iterations = int(power_iterations)
        self.last_lipschitz = None
        self.last_step_size = None
        self.last_power_iterations = 0
        self.last_spectral_source = None

    def inverse_mass(self):
        if self.use_diagonal:
            return np.diag(self.inverse_diagonal.numpy().astype(np.float64))
        return self.inverse_dense.numpy().astype(np.float64)

    def step(self, m, d):
        if d.nworld != 1:
            raise ValueError("residual SCM prediction supports one world")
        self.prepare_step(m, d)
        count = int(d.nefc.numpy()[0])
        if count > d.efc.J.shape[1]:
            raise RuntimeError("residual SCM constraint buffer overflow; increase --njmax")
        jacobian = d.efc.J.numpy()[0, :count, :m.nv].astype(np.float64)
        gap = d.efc.efc_dist.numpy()[0, :count].astype(np.float64)
        free = d.qvel_smooth_pred.numpy()[0, :m.nv].astype(np.float64)
        projected = free
        if self.motion == "increment":
            projected = free - d.qvel.numpy()[0, :m.nv].astype(np.float64)
        timestep = float(m.opt.timestep.numpy()[0])
        impulse, info = environment_impulses(
            jacobian, self.inverse_mass(), projected, gap, timestep, self.regularization,
            self.softplus_temperature, self.correction, self.correction_steps,
            nonnegative_gap=True, max_iterations=self.power_iterations)
        self.last_lipschitz = info["lipschitz"]
        self.last_step_size = info["step_size"]
        self.last_power_iterations = info["power_iterations"]
        self.last_spectral_source = info["spectral_source"]
        self.last_initial_impulse = info["initial"]
        forces = np.zeros(d.efc.force.shape, dtype=np.float32)
        if count:
            forces[0, :count] = impulse / timestep
        generalized = np.asarray((jacobian.T @ impulse / timestep)[None], dtype=np.float32)
        wp.copy(d.efc.force, wp.array(forces, dtype=d.efc.force.dtype))
        wp.copy(d.qfrc_constraint, wp.array(generalized, dtype=d.qfrc_constraint.dtype))
        self.finish_step(m, d)
