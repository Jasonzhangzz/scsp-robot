"""Bounded, dimensionless soft-SCM calibration parameters.

Inspired by ContactSDF V-B (on-policy one-step prediction loss), implemented
independently. Q scales describe an effective quasi-static metric, not identified
physical mass. Gravity and robot stiffness fix the scale gauge. Four symmetric
friction-pyramid rows must belong to the same contact.
"""
import numpy as np
import casadi as cs

SCM_PARAMETER_NAMES = ('impedance_translation', 'impedance_rotation_x',
                       'impedance_rotation_y', 'impedance_rotation_z',
                       'response_time', 'friction')
SCM_PARAMETER_DIM = 6
SCM_LOG_LOWER = np.log([.1, .1, .1, .1, .1, .2])
SCM_LOG_UPPER = np.log([10., 10., 10., 10., 10., 2.])


def validate_theta(theta):
    theta = np.asarray(theta, dtype=float).reshape(6)
    if not np.isfinite(theta).all() or np.any(theta < SCM_LOG_LOWER-1e-10) or np.any(theta > SCM_LOG_UPPER+1e-10):
        raise ValueError('SCM log parameters are outside calibrated bounds')
    return theta.copy()


def scaled_inverse_numpy(base, theta):
    s = np.ones(len(base))
    s[:3] = np.exp(-.5*theta[0])
    s[3:6] = np.exp(-.5*np.asarray(theta)[1:4])
    return s[:, None] * base * s[None, :]


def scaled_inverse_cs(base, theta):
    n = base.shape[0]
    d = cs.diag(cs.vertcat(cs.repmat(cs.exp(-.5*theta[0]), 3, 1),
                          cs.exp(-.5*theta[1:4]), cs.DM.ones(n-6)))
    return d @ cs.DM(base) @ d


def friction_jacobian_numpy(jacobian, theta):
    blocks = np.asarray(jacobian).reshape(-1, 4, jacobian.shape[1])
    normal = blocks.mean(axis=1, keepdims=True)
    return (normal + np.exp(theta[5])*(blocks-normal)).reshape(jacobian.shape)


def friction_jacobian_cs(jacobian, theta):
    rows = []
    for i in range(jacobian.shape[0]//4):
        block = jacobian[4*i:4*i+4, :]
        normal = cs.repmat(cs.sum1(block)/4, 4, 1)
        rows.append(normal + cs.exp(theta[5])*(block-normal))
    return cs.vertcat(*rows)


def response_cs(h, tau, zeta, theta):
    x = h/(tau*cs.exp(theta[4]))
    den = 1+2*zeta*x+x*x
    return (2*zeta*x+x*x)/den, x*x/den
