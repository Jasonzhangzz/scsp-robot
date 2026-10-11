"""Withdrawal must lose flip progress in the MPC model itself."""
from types import SimpleNamespace

import casadi as cs
import numpy as np
from scipy.spatial.transform import Rotation

from models.contact_geometry import contact_geometry_cs
from models.explicit_model import ExplicitModel
from examples.mpc.franka.ik2.contact_frames import (
    contact_jacobian_angular_body_frame, support_patch_jacobians,
)
from planning.object_pose_cost import cost_config, object_cost_np, target_up_axis


def test_withdrawal_falls_back_and_costs_more_than_continued_pressure():
    angle = np.pi / 6
    rotation = Rotation.from_euler('y', angle).as_matrix()
    pose = np.r_[[0., 0., .04*(np.cos(angle)+np.sin(angle))],
                 [np.cos(angle/2), 0., np.sin(angle/2), 0.]]
    point = np.array([-.04, 0., .04])
    tip = pose[:3] + rotation @ (point + [-.01, 0., 0.])
    reference = np.r_[pose, tip]
    param = SimpleNamespace(
        h_=.02, n_qpos_=10, n_qvel_=9, n_cmd_=3, n_robot_qpos_=3,
        max_ncon_=3, Q=np.diag([50.]*3+[.05]*3+[600.]*3), obj_mass_=.01,
        gravity_=np.array([0., 0., -9.8, 0., 0., 0.]), robot_stiff_=600*np.eye(3),
        contact_dynamics='quasi-dynamic', grouped_contact_response_=True,
    )
    model = ExplicitModel(param)
    jac, phi = np.zeros((12, 9)), np.zeros(12)
    directions = np.array([[1., .5, 0.], [1., 0., .5],
                           [1., -.5, 0.], [1., 0., -.5]]) @ rotation.T
    lever = rotation @ point
    skew = np.array([[0., -lever[2], lever[1]], [lever[2], 0., -lever[0]],
                     [-lever[1], lever[0], 0.]])
    jac[:4] = directions @ np.c_[np.eye(3), -skew, -np.eye(3)]
    for i, y in enumerate([-.08, .08]):
        support = pose[:3] + rotation @ np.array([.04, y, -.04])
        jac[4+i*4:8+i*4], _ = support_patch_jacobians(pose[:3], rotation, support, 9, .5)
    jac = contact_jacobian_angular_body_frame(jac, rotation)
    x, ref, u = cs.SX.sym('x', 10), cs.SX.sym('ref', 10), cs.SX.sym('u', 3)
    gaps, rows = contact_geometry_cs(x, ref, cs.DM(phi), cs.DM(jac))
    step = cs.Function('flip_step', [x, u, ref], [model.prediction_fn(x, u, gaps, rows, 1.)])
    target = np.r_[[0., 0., .04], [2**-.5, 0., 2**-.5, 0.]]
    config = cost_config(500., 0., 20., target_up_axis(target[3:]))

    def rollout(command):
        state = reference.copy()
        costs = []
        for _ in range(5):
            state = np.asarray(step(state, command, reference)).ravel()
            costs.append(object_cost_np(state, target, config))
        return state, sum(costs)+10*costs[-1]

    held, held_cost = rollout(rotation @ np.array([1./600., 0., 0.]))
    # Withdraw from the actual patch. Geometry inside the dynamics disables
    # pressure; no contact rows are manually removed for this comparison.
    released, released_cost = rollout(-.004*rotation[:, 0])
    def pitch(state):
        return Rotation.from_quat(state[[4, 5, 6, 3]]).as_euler('xyz')[1]

    assert pitch(held) > pitch(reference) > pitch(released)
    assert released_cost > held_cost + 1.
    assert object_cost_np(released, target, config) > object_cost_np(reference, target, config)
