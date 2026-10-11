"""Finite-time diagonal SCM contact response, in impulse units.

p = max(-D^-1 (a J v_free + b phi/h), 0),
D = diag(J M^-1 J.T) + r I, v_next = v_free + M^-1 J.T p.
The coefficients follow an implicitly discretized, decoupled spring-damper
row with response time tau and damping ratio zeta. This is not a coupled
implicit contact solve. Geometry and D are frozen within each prediction.
"""
import numpy as np


# Conservative starting values for the 20--50 ms planning steps. The 8 ms,
# zeta=.25 calibration from the 2 ms simulator benchmark is not an MPC fit.
SOFT_SCM_RESPONSE_TIME = 0.05
SOFT_SCM_DAMPING_RATIO = 1.0
SOFT_SCM_REGULARIZATION = 1e-8


def normalize_contact_model(name):
    name = str(name).strip().lower().replace('_', '-')
    if name == 'scm-soft':
        name = 'soft-scm'
    if name not in ('soft-scm', 'comfree'):
        raise ValueError("contact_model must be 'soft-scm' or 'comfree'")
    return name


def soft_scm_coefficients(h, response_time=SOFT_SCM_RESPONSE_TIME,
                          damping_ratio=SOFT_SCM_DAMPING_RATIO,
                          regularization=SOFT_SCM_REGULARIZATION):
    values = np.asarray([h, response_time, damping_ratio, regularization], dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('soft-SCM timestep, response time, damping ratio and regularization must be positive and finite')
    x = float(h) / float(response_time)
    denominator = 1.0 + 2.0 * damping_ratio * x + x*x
    return ((2.0 * damping_ratio * x + x*x) / denominator,
            x*x / denominator)


def inverse_diagonal_numpy(jacobian, inverse_mass, regularization):
    diagonal = np.sum((jacobian @ inverse_mass) * jacobian, axis=1)
    return np.where(np.any(jacobian != 0, axis=1), 1.0 / (diagonal + regularization), 0.0)


def inverse_diagonal_cs(jacobian, inverse_mass, regularization):
    import casadi as cs
    diagonal = cs.sum2((jacobian @ inverse_mass) * jacobian)
    return cs.if_else(cs.sum2(jacobian * jacobian) > 0,
                      1.0 / (diagonal + regularization), 0.0)


def inverse_diagonal_torch(jacobian, inverse_mass, regularization):
    import torch
    diagonal = torch.sum((jacobian @ inverse_mass) * jacobian, dim=-1)
    return torch.where(torch.any(jacobian != 0, dim=-1),
                       1.0 / (diagonal + regularization), 0.0)


def grouped_contact_velocity_cs(smooth, jacobian, phi, inverse, h, a, b, regularization):
    """One unilateral normal and one Coulomb disk per four-row contact.

    The rows encode n+mu*t1, n+mu*t2, n-mu*t1, n-mu*t2.
    Applying four independent normal reactions to this pyramid duplicates
    support. Recover the physical basis and project its tangential impulse
    smoothly into the disk bounded by the single normal impulse. Padding stays zero.
    """
    import casadi as cs
    velocity = smooth
    for start in range(0, int(jacobian.size1()), 4):
        j = jacobian[start:start+4, :]
        n = cs.sum1(j)/4
        t = cs.vertcat((j[0, :]-j[2, :])/2, (j[1, :]-j[3, :])/2)
        mu = cs.sqrt(cs.sumsqr(t[0, :3]) + 1e-20)
        t = t/cs.fmax(mu, 1e-10)
        normal = cs.fmax(-(a*(n@smooth) + b*cs.sum1(phi[start:start+4])/4/h)
                         / (n@inverse@n.T + regularization), 0)
        tangent = -a*(t@smooth)/(cs.sum2((t@inverse)*t) + regularization)
        cap = mu*normal
        tangent *= cap/cs.sqrt(cap*cap + cs.sumsqr(tangent) + 1e-12)
        velocity += inverse@(n.T*normal + t.T@tangent)
    return velocity


def grouped_contact_velocity_numpy(smooth, jacobian, phi, inverse, h, a, b, regularization):
    velocity = np.array(smooth, dtype=float, copy=True)
    for start in range(0, jacobian.shape[0], 4):
        j = jacobian[start:start+4]
        n = j.mean(axis=0)
        t = np.stack([(j[0]-j[2])/2, (j[1]-j[3])/2])
        mu = np.linalg.norm(t[0, :3])
        t /= max(mu, 1e-10)
        normal = max(-(a*np.dot(n, smooth) + b*np.mean(phi[start:start+4])/h)
                     / (n@inverse@n + regularization), 0.)
        tangent = -a*(smooth@t.T)/(np.sum((t@inverse)*t, axis=1) + regularization)
        cap = mu*np.asarray(normal)[..., None]
        tangent *= cap/np.sqrt(cap*cap + np.sum(tangent*tangent, axis=-1, keepdims=True) + 1e-12)
        velocity += (normal[..., None]*n if np.ndim(normal) else normal*n)@inverse.T + tangent@(t@inverse.T)
    return velocity


def grouped_contact_velocity_torch(smooth, jacobian, phi, inverse, h, a, b, regularization):
    import torch
    velocity = smooth
    for start in range(0, jacobian.shape[0], 4):
        j = jacobian[start:start+4]
        n = j.mean(dim=0)
        t = torch.stack([(j[0]-j[2])/2, (j[1]-j[3])/2])
        mu = torch.linalg.vector_norm(t[0, :3])
        t = t/mu.clamp_min(1e-10)
        normal = torch.clamp(-(a*(smooth@n) + b*phi[start:start+4].mean()/h)
                              / (n@inverse@n + regularization), min=0)
        tangent = -a*(smooth@t.T)/(torch.sum((t@inverse)*t, dim=1) + regularization)
        cap = mu*normal[..., None]
        tangent = tangent*cap/torch.sqrt(cap*cap + torch.sum(tangent*tangent, dim=-1, keepdim=True) + 1e-12)
        velocity = velocity + normal[..., None]*(inverse@n) + tangent@(t@inverse.T)
    return velocity
