"""Finite, diagonal-mass point-contact updates with no parameter learning.

Four input rays per point use cyclic order n+mu*t1, n+mu*t2,
n-mu*t1, n-mu*t2 (the planner's existing geometry interface).
We recover normal/tangent rows, retain D_i=J_i Minv J_i.T+r, and perform
one ordered pass by default. Later rows see THIS prediction's corrected
velocity. There is no history, fitted stiffness/damping, full Delassus, or
inner convergence test. Frozen geometry makes the finite map piecewise affine
in free velocity, including its box-friction projection.

Negative gaps are not actively recovered: no penetration-restoration stiffness
is introduced. Positive gaps use phi/h to prevent premature support. Friction
is the independent tangent box, matching DART's benchmark law rather than a
Euclidean Coulomb disk. These choices are explicit model limitations.
"""
import numpy as np

DIAGONAL_CONTACT_PASSES = 1


def _check(h, regularization, passes):
    if not np.isfinite([h, regularization]).all() or h <= 0 or regularization <= 0:
        raise ValueError('Diagonal contact h and regularization must be positive and finite')
    if isinstance(passes, bool) or int(passes) != passes or passes < 1:
        raise ValueError('Diagonal contact passes must be a positive integer')


def diagonal_contact_velocity_numpy(smooth, jacobian, phi, inverse, h,
                                    regularization=1e-8, passes=DIAGONAL_CONTACT_PASSES,
                                    return_impulses=False, friction=None):
    _check(h, regularization, passes)
    if jacobian.shape[0] % 4:
        raise ValueError('Four cyclic pyramid rays are required per contact')
    blocks = []
    for start in range(0, len(jacobian), 4):
        j = jacobian[start:start+4]
        normal = j.mean(axis=0)
        t = np.stack(((j[0]-j[2])/2, (j[1]-j[3])/2))
        mu = np.linalg.norm(t[0, :3]) if friction is None else float(friction[start//4])
        tangent = t/max(mu, 1e-10)
        rows = np.vstack((normal, tangent))
        directions = rows @ inverse.T
        diagonal = np.sum(rows*directions, axis=1)+regularization
        bias = max(float(np.mean(phi[start:start+4])), 0.)/h
        blocks.append((rows, directions, diagonal, mu, bias))
    velocity = np.array(smooth, dtype=float, copy=True)
    impulses = [np.zeros(velocity.shape[:-1]+(3,)) for _ in blocks]
    for _ in range(int(passes)):
        for i, (rows, directions, diagonal, mu, bias) in enumerate(blocks):
            for axis in range(3):
                old = impulses[i][..., axis].copy()
                trial = old-(velocity@rows[axis]+(bias if axis == 0 else 0.)+regularization*old)/diagonal[axis]
                if axis == 0:
                    updated = np.maximum(trial, 0.)
                else:
                    cap = mu*impulses[i][..., 0]
                    updated = np.clip(trial, -cap, cap)
                velocity += (updated-old)[..., None]*directions[axis]
                impulses[i][..., axis] = updated
    result = [(p[..., 0], p[..., 1:]) for p in impulses]
    return (velocity, result) if return_impulses else velocity


def diagonal_contact_velocity_cs(smooth, jacobian, phi, inverse, h,
                                 regularization=1e-8, passes=DIAGONAL_CONTACT_PASSES,
                                 return_impulses=False, friction=None):
    import casadi as cs
    _check(h, regularization, passes)
    if int(jacobian.size1()) % 4:
        raise ValueError('Four cyclic pyramid rays are required per contact')
    blocks = []
    for start in range(0, int(jacobian.size1()), 4):
        j = jacobian[start:start+4, :]
        n = cs.sum1(j)/4
        t = cs.vertcat((j[0, :]-j[2, :])/2, (j[1, :]-j[3, :])/2)
        mu = cs.sqrt(cs.sumsqr(t[0, :3])+1e-20) if friction is None else friction[start//4]
        rows = cs.vertcat(n, t/cs.fmax(mu, 1e-10))
        directions = inverse@rows.T
        diagonal = cs.sum2((rows@inverse)*rows)+regularization
        bias = cs.fmax(cs.sum1(phi[start:start+4])/4, 0)/h
        blocks.append((rows, directions, diagonal, mu, bias))
    velocity = smooth
    impulses = [[cs.DM(0) for _ in range(3)] for _ in blocks]
    for _ in range(int(passes)):
        for i, (rows, directions, diagonal, mu, bias) in enumerate(blocks):
            for axis in range(3):
                old = impulses[i][axis]
                trial = old-(rows[axis, :]@velocity+(bias if axis == 0 else 0)+regularization*old)/diagonal[axis]
                if axis == 0:
                    updated = cs.fmax(trial, 0)
                else:
                    cap = mu*impulses[i][0]
                    updated = cs.fmin(cs.fmax(trial, -cap), cap)
                velocity += directions[:, axis]*(updated-old)
                impulses[i][axis] = updated
    result = [(p[0], cs.vertcat(*p[1:])) for p in impulses]
    return (velocity, result) if return_impulses else velocity


def diagonal_contact_velocity_torch(smooth, jacobian, phi, inverse, h,
                                    regularization=1e-8, passes=DIAGONAL_CONTACT_PASSES,
                                    return_impulses=False, friction=None):
    import torch
    _check(h, regularization, passes)
    if jacobian.shape[0] % 4:
        raise ValueError('Four cyclic pyramid rays are required per contact')
    blocks = []
    for start in range(0, len(jacobian), 4):
        j = jacobian[start:start+4]
        n = j.mean(dim=0)
        t = torch.stack(((j[0]-j[2])/2, (j[1]-j[3])/2))
        mu = torch.linalg.vector_norm(t[0, :3]) if friction is None else friction[start//4]
        rows = torch.cat((n[None], t/mu.clamp_min(1e-10)))
        directions = rows@inverse.T
        diagonal = torch.sum(rows*directions, dim=1)+regularization
        bias = torch.clamp(phi[start:start+4].mean(), min=0)/h
        blocks.append((rows, directions, diagonal, mu, bias))
    velocity = smooth
    impulses = [[smooth.new_zeros(smooth.shape[:-1]) for _ in range(3)] for _ in blocks]
    for _ in range(int(passes)):
        for i, (rows, directions, diagonal, mu, bias) in enumerate(blocks):
            for axis in range(3):
                old = impulses[i][axis]
                trial = old-(velocity@rows[axis]+(bias if axis == 0 else 0)+regularization*old)/diagonal[axis]
                if axis == 0:
                    updated = torch.clamp(trial, min=0)
                else:
                    cap = mu*impulses[i][0]
                    updated = torch.minimum(torch.maximum(trial, -cap), cap)
                velocity = velocity+(updated-old)[..., None]*directions[axis]
                impulses[i][axis] = updated
    result = [(p[0], torch.stack(p[1:], dim=-1)) for p in impulses]
    return (velocity, result) if return_impulses else velocity
