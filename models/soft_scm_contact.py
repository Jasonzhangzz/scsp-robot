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
