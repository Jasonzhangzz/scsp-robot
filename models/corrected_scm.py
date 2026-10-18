"""One diagonal-Delassus prediction followed by one global amplitude correction.

Impulses, not forces, are returned. This is ``scm_corrected`` in the shared
prediction benchmark: dynamic projection, signed gap, no soft-SCM coefficients.
"""
import numpy as np


def corrected_scm_velocity(free, jacobian, gap, inverse, h, regularization=1e-8):
    free, jacobian, gap, inverse = map(np.asarray, (free, jacobian, gap, inverse))
    if h <= 0 or regularization <= 0:
        raise ValueError('h and regularization must be positive')
    diagonal = np.einsum('ij,ij->i', jacobian @ inverse, jacobian) + regularization
    g = free @ jacobian.T + gap / h
    initial = np.maximum(-g / diagonal, 0.)
    generalized = initial @ jacobian
    denominator = np.sum((generalized @ inverse) * generalized, axis=-1)
    denominator += regularization * np.sum(initial**2, axis=-1)
    numerator = -np.sum(g * initial, axis=-1)
    alpha = np.ones_like(denominator)
    np.divide(numerator, denominator, out=alpha, where=denominator > 0)
    alpha = np.clip(alpha, 0., 1.)
    impulse = initial * alpha[..., None]
    return free + impulse @ jacobian @ inverse.T, dict(
        initial_impulse=initial, impulse=impulse, alpha=alpha)


def corrected_scm_velocity_cs(free, jacobian, gap, inverse, h, regularization=1e-8):
    """CasADi/SQP form of :func:`corrected_scm_velocity`."""
    import casadi as cs
    diagonal = cs.sum2((jacobian @ inverse) * jacobian) + regularization
    gap_rate = jacobian @ free + gap / h
    initial = cs.fmax(-gap_rate / diagonal, 0)
    generalized = cs.transpose(initial) @ jacobian
    denominator = (generalized @ inverse @ cs.transpose(generalized)
                   + regularization * cs.sumsqr(initial))
    numerator = -cs.dot(gap_rate, initial)
    alpha = cs.fmin(cs.fmax(numerator / (denominator + 1e-12), 0), 1)
    return free + alpha * (inverse @ cs.transpose(jacobian) @ initial)


def corrected_scm_velocity_torch(free, jacobian, gap, inverse, h, regularization=1e-8):
    """Torch form used by the velocity-state MPC fallback."""
    import torch
    diagonal = torch.sum((jacobian @ inverse) * jacobian, dim=-1) + regularization
    if free.ndim == 1:
        gap_rate = jacobian @ free + gap / h
        initial = torch.clamp(-gap_rate / diagonal, min=0.)
        generalized = initial @ jacobian
        denominator = generalized @ inverse @ generalized + regularization * torch.sum(initial**2)
        alpha = torch.clamp((-torch.sum(gap_rate * initial)) / (denominator + 1e-12), 0., 1.)
        return free + alpha * (inverse @ jacobian.T @ initial)
    gap_rate = free @ jacobian.T + gap / h
    initial = torch.clamp(-gap_rate / diagonal, min=0.)
    generalized = initial @ jacobian
    denominator = torch.einsum('bi,ij,bj->b', generalized, inverse, generalized) + regularization * torch.sum(initial**2, dim=-1)
    alpha = torch.clamp((-torch.sum(gap_rate * initial, dim=-1)) / (denominator + 1e-12), 0., 1.)
    return free + alpha[:, None] * (generalized @ inverse.T)
