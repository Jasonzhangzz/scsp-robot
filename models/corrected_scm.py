"""softplus_nonnegative_gap_v2: diagonal impulse and global line correction.

The CSO benchmark's scm_corrected now denotes the new soft model. Touching and
penetrating rows use exactly zero distance; temperature is in N s.
"""
import numpy as np

LAW_VERSION = 'softplus_nonnegative_gap_v2'
TEMPERATURE = 1e-4


def corrected_scm_velocity(free, jacobian, gap, inverse, h, regularization=1e-8,
                           temperature=TEMPERATURE):
    free, jacobian, gap, inverse = map(np.asarray, (free, jacobian, gap, inverse))
    if min(h, regularization, temperature) <= 0:
        raise ValueError('h, regularization and temperature must be positive')
    diagonal = np.einsum('ij,ij->i', jacobian @ inverse, jacobian) + regularization
    g = free @ jacobian.T + np.maximum(gap, 0.) / h
    initial = temperature*np.logaddexp(0., -g/(diagonal*temperature))
    generalized = initial @ jacobian
    denominator = np.sum((generalized @ inverse) * generalized, axis=-1)
    denominator += regularization * np.sum(initial**2, axis=-1)
    numerator = -np.sum(g * initial, axis=-1)
    alpha = np.ones_like(denominator)
    np.divide(numerator, denominator, out=alpha, where=denominator > 0)
    alpha = np.clip(alpha, 0., 1.)
    impulse = initial * alpha[..., None]
    return free + impulse @ jacobian @ inverse.T, dict(
        initial_impulse=initial, impulse=impulse, alpha=alpha, law_version=LAW_VERSION)


def corrected_scm_velocity_cs(free, jacobian, gap, inverse, h, regularization=1e-8,
                              temperature=TEMPERATURE):
    import casadi as cs
    diagonal = cs.sum2((jacobian @ inverse) * jacobian) + regularization
    g = jacobian @ free + cs.fmax(gap, 0) / h
    trial = -g/diagonal
    initial = cs.fmax(trial, 0)+temperature*cs.log(1+cs.exp(-cs.fabs(trial)/temperature))
    generalized = jacobian.T @ initial
    denominator = cs.dot(generalized, inverse @ generalized)+regularization*cs.sumsqr(initial)
    alpha = cs.if_else(denominator > 0, cs.fmin(cs.fmax(-cs.dot(g, initial)/
                      cs.if_else(denominator > 0, denominator, 1), 0), 1), 1)
    return free + alpha*(inverse @ generalized)


def corrected_scm_velocity_torch(free, jacobian, gap, inverse, h, regularization=1e-8,
                                 temperature=TEMPERATURE):
    import torch
    diagonal = torch.sum((jacobian @ inverse) * jacobian, dim=-1)+regularization
    g = free @ jacobian.T + torch.clamp(gap, min=0)/h
    initial = temperature*torch.nn.functional.softplus(-g/(diagonal*temperature))
    generalized = initial @ jacobian
    denominator = torch.sum((generalized @ inverse)*generalized, dim=-1)
    denominator += regularization*torch.sum(initial**2, dim=-1)
    safe = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    alpha = torch.where(denominator > 0, -torch.sum(g*initial, dim=-1)/safe,
                        torch.ones_like(denominator)).clamp(0, 1)
    return free + alpha[..., None]*(generalized @ inverse.T)
