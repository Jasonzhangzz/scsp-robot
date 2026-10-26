"""Smooth 3-row Coulomb-box SCM with K unrolled Gauss-Seidel sweeps.

Per environment contact i with rows (n_i, t1_i, t2_i), friction mu_i,
D_i = diag(J_i Minv J_i^T) + eps and g_n = n_i.v + max(phi_i, 0)/h:

    lambda_n  = softplus_T(lambda_n  - (n_i.v  + max(phi,0)/h + eps lambda_n)/D_n)
    lambda_t  = clip_T(lambda_t - (t_i.v + eps lambda_t)/D_t, mu_i lambda_n)
    v        += Minv J_i^T (lambda_new - lambda_old)

where clip_T(x, c) = x - softplus_T(x - c) + softplus_T(-x - c) is a C-infinity
box projection onto [-c, c] (exact in the interior up to T, saturating at +-c).
``sweeps`` sweeps are unrolled in a FIXED contact order, so the map
v_free -> v_plus is an explicit finite composition of smooth elementary maps:
closed form, differentiable by the chain rule (NumPy / CasADi below), and
evaluable by sampling (CEM) or gradient (SQP/IPOPT) solvers alike.
``sweeps=1`` is the planner's existing one-pass law with smooth projections.

Soft-contact alignment (MuJoCo solref/solimp): pass per-row ``reference``
velocities r_i = J_i v + h aref_i and per-row ``compliance`` R_i (MuJoCo's
regularizer, acceleration per unit force).  The row residual then becomes
J_i v+ - r_i + R_i lambda_i, i.e. MuJoCo's regularized constraint
J a = aref - R f written at the impulse level; the rigid law is the special
case r = (-max(phi,0)/h, 0, 0), R = eps.
"""
import numpy as np


def softplus(x, T):
    if T == 0:
        return np.maximum(x, 0.)
    return np.maximum(x, 0.) + T * np.log1p(np.exp(-np.abs(x) / T))


def soft_clip(x, cap, T):
    """Smooth projection of x onto [-cap, cap]; cap >= 0."""
    return x - softplus(x - cap, T) + softplus(-x - cap, T)


def soft_box_contact_velocity_numpy(free, jacobian, phi, inverse, h, friction, *,
                                    temperature=1e-4, regularization=1e-8, sweeps=3,
                                    return_impulses=False, reference=None, compliance=None):
    """``jacobian``: (n_c, 3, nv) rows (n, t1, t2); ``phi``: (n_c,) signed gaps.

    ``reference``: optional (n_c, 3) target row velocities (default rigid
    ``(-max(phi,0)/h, 0, 0)``); ``compliance``: optional (n_c, 3) per-row
    regularizer R_i added to the diagonal and to the residual (default
    ``regularization``).
    """
    jacobian = np.asarray(jacobian, dtype=float)
    count = jacobian.shape[0]
    velocity = np.array(free, dtype=float, copy=True)
    impulses = np.zeros((count, 3))
    if count == 0:
        return (velocity, impulses) if return_impulses else velocity
    directions = np.einsum('cij,jk->cik', jacobian, np.asarray(inverse).T)   # J_i Minv^T
    if compliance is None:
        compliance = np.full((count, 3), float(regularization))
    compliance = np.asarray(compliance, dtype=float).reshape(count, 3)
    if reference is None:
        reference = np.zeros((count, 3))
        reference[:, 0] = -np.maximum(np.asarray(phi, dtype=float), 0.) / h
    reference = np.asarray(reference, dtype=float).reshape(count, 3)
    diagonal = np.einsum('cij,cij->ci', jacobian, directions) + compliance
    for _ in range(int(sweeps)):
        for i in range(count):
            old = impulses[i].copy()
            trial = old - (jacobian[i] @ velocity - reference[i] + compliance[i] * old) / diagonal[i]
            new_n = softplus(trial[0], temperature)
            cap = float(friction[i]) * new_n
            new_t = soft_clip(trial[1:], cap, temperature)
            new = np.r_[new_n, new_t]
            velocity = velocity + (new - old) @ directions[i]
            impulses[i] = new
    return (velocity, impulses) if return_impulses else velocity


def soft_box_contact_velocity_cs(free, jacobian, phi, inverse, h, friction, *,
                                 temperature=1e-4, regularization=1e-8, sweeps=3,
                                 reference=None, compliance=None):
    """CasADi version. ``jacobian``: list of (3 x nv) MX/SX, ``phi``: (n_c,) vector.

    ``reference`` / ``compliance``: optional lists of (3,) per-contact row
    targets and regularizers, as in the NumPy version.
    """
    import casadi as cs

    def sp(x):
        # Keep exact derivatives at zero as well as overflow-safe evaluation.
        # The max/abs identity loses softplus curvature at zero under AD.
        values = [temperature * cs.logsumexp(cs.vertcat(0, x[j] / temperature))
                  for j in range(x.numel())]
        return cs.reshape(cs.vertcat(*values), x.size1(), x.size2())

    def clip(x, c):
        return x - sp(x - c) + sp(-x - c)

    velocity = free
    impulses = [cs.DM.zeros(3, 1) for _ in jacobian]
    for _ in range(int(sweeps)):
        for i, J in enumerate(jacobian):
            directions = inverse @ J.T                                  # nv x 3
            R = regularization if compliance is None else compliance[i]
            target = (cs.vertcat(-cs.fmax(phi[i], 0) / h, 0, 0) if reference is None
                      else reference[i])
            diagonal = cs.diag(J @ directions) + R
            old = impulses[i]
            trial = old - (J @ velocity - target + R * old) / diagonal
            new_n = sp(trial[0])
            cap = friction[i] * new_n
            new = cs.vertcat(new_n, clip(trial[1:3], cap))
            velocity = velocity + directions @ (new - old)
            impulses[i] = new
    return velocity, impulses
