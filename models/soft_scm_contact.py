"""Finite-time diagonal SCM contact response, in impulse units.

p = max(-D^-1 (a J v_free + b phi/h), 0),
D = diag(J M^-1 J.T) + r I, v_next = v_free + M^-1 J.T p.
The coefficients follow an implicitly discretized, decoupled spring-damper
row with response time tau and damping ratio zeta. Grouped contacts use
bounded block Gauss-Seidel sweeps so fingertip pressure and support reactions
are coupled. Geometry stays frozen within each prediction.
Grouped contacts use a full closing-speed bias for positive gaps, so
unreachable contact cannot transmit speculative support forces.
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


# Fixed sweeps keep the contact map differentiable for NLP/RTI backends.
# Subtract each block's old impulse before updating it: applying the isolated
# contact response repeatedly would incorrectly stiffen a single contact.
GROUPED_CONTACT_SWEEPS = 8
GROUPED_CONTACT_VERSION = 3


def grouped_contact_velocity_cs(smooth, jacobian, phi, inverse, h, a, b, regularization,
                                return_impulses=False):
    """Coupled unilateral normals and Coulomb disks, one per pyramid block."""
    import casadi as cs
    blocks = []
    for start in range(0, int(jacobian.size1()), 4):
        j = jacobian[start:start+4, :]
        n = cs.sum1(j)/4
        t = cs.vertcat((j[0, :]-j[2, :])/2, (j[1, :]-j[3, :])/2)
        mu = cs.sqrt(cs.sumsqr(t[0, :3]) + 1e-20)
        t = t/cs.fmax(mu, 1e-10)
        mn, mt = inverse@n.T, inverse@t.T
        dn = (n@mn + regularization)/a
        dt = (cs.sum2((t@inverse)*t) + regularization)/a
        gap = cs.sum1(phi[start:start+4])/4
        # A positive gap must close within this prediction step before the
        # contact can transmit force. Using b/a (<1) there let a separated
        # finger provide speculative support several steps before impact.
        bias = cs.if_else(gap > 0, gap, b/a*gap)/h
        blocks.append((n, t, mu, mn, mt, dn, dt, bias))
    velocity = smooth
    impulses = [(0, cs.DM.zeros(2)) for _ in blocks]
    for _ in range(GROUPED_CONTACT_SWEEPS):
        for i, (n,t,mu,mn,mt,dn,dt,bias) in enumerate(blocks):
            old_n, old_t = impulses[i]
            free = velocity - mn*old_n - mt@old_t
            normal = cs.fmax(-(n@free + bias)/dn, 0)
            supported = free + mn*normal
            tangent = -(t@supported)/dt
            cap = mu*normal
            tangent *= cap/cs.sqrt(cap*cap + cs.sumsqr(tangent) + 1e-12)
            velocity = supported + mt@tangent
            impulses[i] = (normal, tangent)
    return (velocity, impulses) if return_impulses else velocity


def grouped_contact_velocity_numpy(smooth, jacobian, phi, inverse, h, a, b, regularization):
    blocks = []
    for start in range(0, jacobian.shape[0], 4):
        j = jacobian[start:start+4]
        n = j.mean(axis=0)
        t = np.stack([(j[0]-j[2])/2, (j[1]-j[3])/2])
        mu = np.linalg.norm(t[0, :3])
        t /= max(mu, 1e-10)
        mn, mt = inverse@n, t@inverse.T
        dn = (n@mn + regularization)/a
        dt = (np.sum((t@inverse)*t, axis=1) + regularization)/a
        gap = np.mean(phi[start:start+4])
        bias = (gap if gap > 0 else b/a*gap)/h
        blocks.append((n,t,mu,mn,mt,dn,dt,bias))
    velocity = np.array(smooth, dtype=float, copy=True)
    impulses = [(np.zeros(velocity.shape[:-1]), np.zeros(velocity.shape[:-1]+(2,))) for _ in blocks]
    for _ in range(GROUPED_CONTACT_SWEEPS):
        for i, (n,t,mu,mn,mt,dn,dt,bias) in enumerate(blocks):
            old_n, old_t = impulses[i]
            free = velocity - old_n[...,None]*mn - old_t@mt
            normal = np.maximum(-(free@n + bias)/dn, 0.)
            supported = free + normal[...,None]*mn
            tangent = -(supported@t.T)/dt
            cap = mu*normal[...,None]
            tangent *= cap/np.sqrt(cap*cap + np.sum(tangent*tangent, axis=-1, keepdims=True) + 1e-12)
            velocity = supported + tangent@mt
            impulses[i] = (normal, tangent)
    return velocity


def grouped_contact_velocity_torch(smooth, jacobian, phi, inverse, h, a, b, regularization):
    import torch
    blocks = []
    for start in range(0, jacobian.shape[0], 4):
        j = jacobian[start:start+4]
        n = j.mean(dim=0)
        t = torch.stack([(j[0]-j[2])/2, (j[1]-j[3])/2])
        mu = torch.linalg.vector_norm(t[0, :3])
        t = t/mu.clamp_min(1e-10)
        mn, mt = inverse@n, t@inverse.T
        dn = (n@mn + regularization)/a
        dt = (torch.sum((t@inverse)*t, dim=1) + regularization)/a
        gap = phi[start:start+4].mean()
        bias = torch.where(gap > 0, gap, b/a*gap)/h
        blocks.append((n,t,mu,mn,mt,dn,dt,bias))
    velocity = smooth
    impulses = [(smooth.new_zeros(smooth.shape[:-1]), smooth.new_zeros(smooth.shape[:-1]+(2,))) for _ in blocks]
    for _ in range(GROUPED_CONTACT_SWEEPS):
        for i, (n,t,mu,mn,mt,dn,dt,bias) in enumerate(blocks):
            old_n, old_t = impulses[i]
            free = velocity - old_n[...,None]*mn - old_t@mt
            normal = torch.clamp(-(free@n + bias)/dn, min=0)
            supported = free + normal[...,None]*mn
            tangent = -(supported@t.T)/dt
            cap = mu*normal[...,None]
            tangent = tangent*cap/torch.sqrt(cap*cap + torch.sum(tangent*tangent, dim=-1, keepdim=True) + 1e-12)
            velocity = supported + tangent@mt
            impulses[i] = (normal, tangent)
    return velocity
