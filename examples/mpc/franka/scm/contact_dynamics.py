"""MuJoCo collision with independent contact laws (float64 accuracy backends).

DART equations are a NumPy port of v6.16.0 ContactConstraint and
PgsBoxedLcpSolver, with deterministic ordering, zero restitution/slip compliance.
ContactSDF's LSE projection follows asu-iris/ContactSDF commit 3a21edf8.
Its dynamic extension uses Q=M/h² and b=M*v/h+f_smooth, hence preserves
v_free=v+h*a_smooth. This is not the upstream learned quasi-dynamic model.
Neither backend calls MuJoCo's constraint solver to obtain reaction forces.
"""

import mujoco
import numpy as np

# DART portions: Copyright (c) 2011-2025, The DART development contributors.
# BSD license and source URLs: vendor/dart_contact/.

DART_ERP = .01
DART_MAX_ERV = .001
DART_CFM = 1e-5


def full_mass(model, data):
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    return mass


def contact_snapshot(model, data):
    """Active MuJoCo point contacts, with relative Jacobians in native frames."""
    ids = [i for i in range(data.ncon) if data.contact[i].efc_address >= 0]
    count = len(ids)
    native = {"gap": np.array([data.contact[i].dist for i in ids]),
              "position": np.array([data.contact[i].pos for i in ids]).reshape(count, 3),
              "frame": np.array([data.contact[i].frame for i in ids]).reshape(count, 3, 3),
              "geom": np.array([data.contact[i].geom for i in ids], dtype=int).reshape(count, 2),
              "friction": np.array([data.contact[i].friction for i in ids]).reshape(count, 5),
              "dim": np.full(count, 3), "force": np.zeros((count, 6))}
    jac = np.zeros((count, 3, model.nv))
    for k, i in enumerate(ids):
        con = data.contact[i]
        points = []
        for geom in con.geom:
            j = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, j, None, con.pos, int(model.geom_bodyid[geom]))
            points.append(j)
        jac[k] = native["frame"][k] @ (points[1] - points[0])
    return native, jac


def pyramid_rows(native, jac):
    # Opposite friction edges are adjacent, as in MuJoCo/ComFree.
    rows = []
    for i, (normal, t1, t2) in enumerate(jac):
        mu1, mu2 = native["friction"][i, :2]
        rows.extend((normal + mu1*t1, normal - mu1*t1,
                     normal + mu2*t2, normal - mu2*t2))
    return np.asarray(rows).reshape(-1, jac.shape[-1]), np.repeat(native["gap"], 4)


def dart_pgs_solve(matrix, rhs, lo, hi, findex, iterations=30, delta=1e-6,
                   relative=1e-3, epsilon=1e-9, return_info=False):
    """DART boxed PGS, zero warm start and randomizeConstraintOrder=False."""
    a, b = np.array(matrix, float, copy=True), np.array(rhs, float, copy=True)
    x = np.zeros(b.size)
    order = np.flatnonzero(np.diag(a) >= epsilon)
    done, sweep = True, -1
    for sweep in range(iterations):
        done = True
        for i in order:
            old = x[i]
            new = (b[i] - a[i] @ x + a[i, i]*x[i]) / a[i, i]
            parent = int(findex[i])
            upper = hi[i] if parent < 0 else hi[i]*x[parent]
            lower = lo[i] if parent < 0 else -upper
            x[i] = np.clip(new, lower, upper)
            if sweep == 0:
                done &= abs(x[i]-old) <= delta
            elif abs(x[i]) > epsilon:
                done &= abs((x[i]-old)/x[i]) <= relative
        if done:
            break
    if return_info:
        return x, {"converged": bool(done), "iterations": sweep+1,
                   "solver": "DART deterministic boxed PGS NumPy port"}
    return x


def dart_contact_impulse(jac, gap, friction, inverse, free, h, return_info=False):
    j = jac.reshape(-1, inverse.shape[0])
    matrix = j @ inverse @ j.T
    # ContactConstraint::getVelocityChange multiplies each diagonal by 1+CFM.
    diagonal = np.diag_indices_from(matrix)
    matrix[diagonal] *= 1 + DART_CFM
    rhs = -j @ free
    rhs[::3] += np.minimum(np.maximum(-gap, 0)*DART_ERP/h, DART_MAX_ERV)
    lo = np.zeros(len(rhs))
    hi = np.zeros(len(rhs))
    parent = np.full(len(rhs), -1)
    for i, mu in enumerate(friction[:, :2]):
        hi[3*i] = np.inf
        lo[3*i+1:3*i+3], hi[3*i+1:3*i+3] = -mu, mu
        parent[3*i+1:3*i+3] = 3*i
    impulse, info = dart_pgs_solve(matrix, rhs, lo, hi, parent, return_info=True)
    if return_info:
        return impulse, j, info
    return impulse, j


def dart_contact_basis(normal):
    """ContactConstraint::getTangentBasisMatrixODE, default first direction +z."""
    tangent = np.cross(normal, [0., 0., 1.])
    if tangent @ tangent < 1e-12:
        tangent = np.cross(normal, [1., 0., 0.])
    if tangent @ tangent < 1e-12:
        tangent = np.cross(normal, [0., 1., 0.])
    tangent /= np.linalg.norm(tangent)
    return np.stack((normal, np.cross(normal, tangent), tangent))


def lse_project(query, jacobian, gap, q_inverse_sqrt, h, sigma=120.):
    """Upstream D-SDFInit, including +1e-2 row norm and outer hard max.

    Returns projected z and nonnegative row multipliers. Inputs contain only
    real contacts; padding rows must be removed before logsumexp.
    """
    if h <= 0 or sigma <= 0 or not np.isfinite([h, sigma]).all():
        raise ValueError("ContactSDF h and sigma must be finite and positive")
    if not len(gap):
        return np.array(query, copy=True), np.zeros(0)
    matrix = jacobian @ q_inverse_sqrt / h
    scale = 1/np.sqrt(np.sum(matrix*matrix, axis=1) + 1e-2)
    matrix = scale[:, None]*matrix
    values = -scale*gap/h - matrix @ query
    logits = sigma*values
    maximum = logits.max()
    weights = np.exp(logits-maximum)
    total = weights.sum()
    weights /= total
    distance = max((maximum + np.log(total))/sigma, 0.)
    return query + distance*(matrix.T @ weights), distance*scale*weights


def contactsdf_velocity(mass, free, jacobian, gap, h, sigma=120.):
    """Dynamic extension of the upstream LSE projection; exact free motion."""
    eigenvalues, eigenvectors = np.linalg.eigh(mass)
    if np.min(eigenvalues) <= 0:
        raise ValueError("ContactSDF requires positive definite generalized mass")
    inverse_sqrt = (eigenvectors/np.sqrt(eigenvalues)) @ eigenvectors.T
    sqrt = (eigenvectors*np.sqrt(eigenvalues)) @ eigenvectors.T
    query = sqrt @ free
    z, impulse = lse_project(query, jacobian, gap, h*inverse_sqrt, h, sigma)
    return inverse_sqrt @ z, impulse


class CpuContactEngine:
    """MuJoCo geometry/smooth dynamics/integration, independent point-contact law."""

    def __init__(self, model, law, sigma=120., regularization=1e-8):
        if model.nu or model.neq or model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("Accuracy backends require unactuated Euler scenes without equalities")
        self.model, self.law, self.sigma = model, law, sigma
        self.regularization = regularization

    def step(self, data, h=None):
        model = self.model
        h = float(model.opt.timestep if h is None else h)
        mujoco.mj_forward(model, data)
        mass = full_mass(model, data)
        inverse = np.linalg.inv(mass)
        free = data.qvel + h*data.qacc_smooth
        native, jac = contact_snapshot(model, data)
        if self.law == "diagonal_contact":
            # The planner uses cyclic rays; native force diagnostics use n,t1,t2.
            from pathlib import Path
            import sys
            root = str(Path(__file__).resolve().parents[4])
            if root not in sys.path:
                sys.path.insert(0, root)
            from models.diagonal_contact import diagonal_contact_velocity_numpy
            if np.any(data.contact.dim[:data.ncon] > 3):
                raise ValueError('Diagonal contact backend implements point friction only (condim <= 3)')
            if not np.allclose(native['friction'][:, 0], native['friction'][:, 1]):
                raise ValueError('Diagonal contact backend requires equal tangent friction coefficients')
            cyclic, gap = pyramid_rows(native, jac)
            cyclic = cyclic.reshape(-1, 4, model.nv)[:, [0, 2, 1, 3]].reshape(-1, model.nv)
            velocity, impulses = diagonal_contact_velocity_numpy(free, cyclic, gap, inverse, h,
                self.regularization, return_impulses=True, friction=native['friction'][:, 0])
            impulse = np.array([np.r_[n, t] for n, t in impulses]).reshape(-1)
            rows = jac.reshape(-1, model.nv)
            native['force'][:, :3] = impulse.reshape(-1, 3)/h
        elif self.law == "dart":
            # DART creates constraints only for touching/penetrating contacts.
            keep = native["gap"] <= 0
            native = {k: v[keep] for k, v in native.items()}
            jac = jac[keep]
            change = np.array([dart_contact_basis(frame[0]) @ frame.T for frame in native["frame"]]).reshape(-1, 3, 3)
            dart_jac = change @ jac
            impulse, rows, self.last_lcp = dart_contact_impulse(
                dart_jac, native["gap"], native["friction"], inverse, free, h, return_info=True)
            velocity = free + inverse @ rows.T @ impulse
            native["force"][:, :3] = np.einsum("nji,nj->ni", change, impulse.reshape(-1, 3))/h
        elif self.law == "contactsdf":
            rows, gap = pyramid_rows(native, jac)
            velocity, impulse = contactsdf_velocity(mass, free, rows, gap, h, self.sigma)
            edges = impulse.reshape(-1, 4)/h
            native["force"][:, 0] = edges.sum(axis=1)
            native["force"][:, 1] = (edges[:, 0]-edges[:, 1])*native["friction"][:, 0]
            native["force"][:, 2] = (edges[:, 2]-edges[:, 3])*native["friction"][:, 1]
        else:
            raise ValueError(f"Unknown CPU contact law: {self.law}")
        self.last_contacts = native
        self.last_problem = {"mass": mass, "free_velocity": free, "J": rows, "impulse": impulse}
        data.qvel[:] = velocity
        mujoco.mj_integratePos(model, data.qpos, data.qvel, h)
        data.time += h
