"""Shared actuator and contact velocity response for CSO and Cartesian CPO.

All velocities, Jacobians and external wrenches must use the same frame.
The actuator input is a commanded wrench K*u, before velocity damping.
"""
import numpy as np
from models.soft_scm_contact import soft_scm_coefficients


def actuator_coefficients(h, tip_mass, damping):
    if min(h, tip_mass, damping) <= 0 or not np.isfinite([h, tip_mass, damping]).all():
        raise ValueError('actuator timestep, mass and damping must be positive')
    decay = np.exp(-h * damping / tip_mass)
    memory = tip_mass / (damping * h) * (1 - decay)
    return decay, memory


def contact_response_cs(object_velocity, tip_velocity, wrench, external,
                        jac, phi, inverse, mobility, *, h, tip_mass, damping,
                        response_time, rigid_response_time, damping_ratio,
                        regularization, blend_speed, variant):
    import casadi as cs
    from models.soft_scm_contact import grouped_contact_velocity_cs
    decay, memory = actuator_coefficients(h, tip_mass, damping)
    free_tip = memory * tip_velocity + (1 - memory) * wrench / damping
    end_tip = decay * tip_velocity + (1 - decay) * wrench / damping
    free = cs.vertcat(object_velocity + h * inverse[:6, :6] @ external, free_tip)
    a, b = soft_scm_coefficients(h, rigid_response_time, damping_ratio, regularization)
    velocity = grouped_contact_velocity_cs(free, jac, phi, inverse, h, a, b, regularization)
    if variant == 'legacy':
        a, b = soft_scm_coefficients(h, response_time, damping_ratio, regularization)
        supported_free = cs.vertcat(.5 * object_velocity + mobility[:6, :6] @ external / h, free_tip)
        supported = grouped_contact_velocity_cs(supported_free, jac, phi, mobility, h, a, b, regularization)
        loading = 0
        for start in range(0, int(jac.size1()), 4):
            normal = cs.sum1(jac[start:start + 4, :]) / 4
            gap = cs.sum1(phi[start:start + 4]) / 4
            loading = cs.fmax(loading, cs.if_else(cs.sumsqr(normal[6:9]) > 0,
                cs.fmax(-(normal @ supported_free) - gap / h, 0), 0))
        fraction = cs.fmin(1., cs.fmax(0., loading / blend_speed))
        weight = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
        velocity += weight * (supported - velocity)
    elif variant != 'physical':
        raise ValueError('contact dynamics variant must be legacy or physical')
    return velocity, end_tip + velocity[6:9] - free_tip


def contact_response_torch(object_velocity, tip_velocity, wrench, external,
                           jac, phi, inverse, mobility, **options):
    import torch
    from models.soft_scm_contact import grouped_contact_velocity_torch
    h = options['h']
    decay, memory = actuator_coefficients(h, options['tip_mass'], options['damping'])
    free_tip = memory * tip_velocity + (1 - memory) * wrench / options['damping']
    end_tip = decay * tip_velocity + (1 - decay) * wrench / options['damping']
    free = torch.cat([object_velocity + h * inverse[:6, :6] @ external, free_tip])
    coefficients = lambda time: soft_scm_coefficients(h, time, options['damping_ratio'], options['regularization'])
    a, b = coefficients(options['rigid_response_time'])
    velocity = grouped_contact_velocity_torch(free, jac, phi, inverse, h, a, b, options['regularization'])
    if options['variant'] == 'legacy':
        a, b = coefficients(options['response_time'])
        supported_free = torch.cat([.5 * object_velocity + mobility[:6, :6] @ external / h, free_tip])
        supported = grouped_contact_velocity_torch(supported_free, jac, phi, mobility, h, a, b, options['regularization'])
        normals, gaps = jac.reshape(-1, 4, 9).mean(dim=1), phi.reshape(-1, 4).mean(dim=1)
        loading = torch.where(normals[:, 6:].square().sum(dim=1) > 0,
            (-normals @ supported_free - gaps / h).clamp_min(0), torch.zeros_like(gaps)).max()
        fraction = (loading / options['blend_speed']).clamp(0, 1)
        weight = fraction**3 * (10 - 15 * fraction + 6 * fraction**2)
        velocity = velocity + weight * (supported - velocity)
    elif options['variant'] != 'physical':
        raise ValueError('contact dynamics variant must be legacy or physical')
    return velocity, end_tip + velocity[6:9] - free_tip
