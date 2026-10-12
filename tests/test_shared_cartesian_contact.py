from types import SimpleNamespace
import numpy as np
import casadi as cs
import pytest
from scipy.spatial.transform import Rotation

from models.cartesian_rigid_contact import CartesianRigidContactModel
from planning.mlqp_point import LambdaContactControlOptimizer
from planning.object_pose_cost import object_cost_np, object_cost_batch_np, cost_config


@pytest.mark.parametrize('variant', ['legacy', 'physical'])
def test_cso_and_cpo_predict_same_contact_velocity_with_different_state_dimensions(variant):
    param = SimpleNamespace(h_=.02, max_ncon_=3, sim_obj_mass_=.01,
        sim_obj_inertia_diag_=np.array([1e-5, 2e-5, 3e-5]),
        gravity_=np.array([0., 0., -9.8, 0., 0., 0.]), obj_mass_=.01,
        Q=np.diag([50.]*3 + [.05]*3 + [600.]*3), robot_stiff_=600*np.eye(3),
        soft_scm_response_time=.05, soft_scm_damping_ratio=1., soft_scm_regularization=1e-8,
        contact_dynamics_variant_=variant)
    cpo = CartesianRigidContactModel(param)
    cso = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    cso.shared_response = dict(cpo.response_options, mass=(1/np.diag(cpo.base_inverse)).tolist())
    cso.robot_stiffness, cso.h, cso.base_mu_arm_obj = 600., .02, .9
    cso.soft_scm_response_time, cso.soft_scm_damping_ratio, cso.soft_scm_regularization = .05, 1., 1e-8
    point, frame = np.array([.03, 0., .02]), np.eye(3)
    # Environment contact plus padding, followed by the fingertip patch.
    env = np.zeros((8, 6)); env[:4, 2] = 1.
    env[0, 0], env[1, 1], env[2, 0], env[3, 1] = .5, .5, -.5, -.5
    velocity = np.array([.01, -.02, 0., .1, -.2, .05])
    tip_velocity, wrench = np.array([-.01, .02, 0.]), np.array([1., .1, -.1])
    external = param.obj_mass_*param.gravity_
    predicted = cso._arm_velocity_cs(cs.DM(velocity), cs.DM(external), cs.DM(wrench),
        cs.DM(point), cs.DM(frame), cs.DM(env), cs.DM.zeros(8),
        cs.DM(np.linalg.inv(param.Q[:6, :6])), cs.DM.zeros(6), tip_velocity=cs.DM(tip_velocity))
    skew = np.array([[0., -point[2], point[1]], [point[2], 0., -point[0]], [-point[1], point[0], 0.]])
    rel = np.c_[np.eye(3), -skew, -np.eye(3)]
    pyramid = np.array([[1., .9, 0.], [1., 0., .9], [1., -.9, 0.], [1., 0., -.9]])
    jac = np.r_[np.c_[env, np.zeros((8, 3))], pyramid@rel]
    state = np.r_[np.zeros(3), [1., 0., 0., 0.], point, velocity, tip_velocity]
    output = np.asarray(cpo.prediction_fn(state, wrench/600., np.zeros(12), jac, 7.)).ravel()
    cpo_velocity = np.r_[output[10:13], Rotation.from_quat(output[[4,5,6,3]]).apply(output[13:16])]
    np.testing.assert_allclose(np.asarray(predicted).ravel(), cpo_velocity, atol=1e-10)
    import torch
    actual = cpo.prediction_torch(torch.tensor(state), torch.tensor(wrench/600.),
                                 torch.zeros(12, dtype=torch.float64), torch.tensor(jac))
    np.testing.assert_allclose(actual.numpy(), output, atol=1e-10)
    # CSO works in the object's local frame; CPO mixes world linear and
    # body angular coordinates. Their agreement must survive a frame change.
    rotation = Rotation.from_euler('xyz', [20., 35., 60.], degrees=True)
    r = rotation.as_matrix()
    param.gravity_ = np.r_[r @ param.gravity_[:3], np.zeros(3)]
    rotated_model = CartesianRigidContactModel(param)
    transform = np.zeros((9, 9))
    transform[:3, :3], transform[3:6, 3:6], transform[6:, 6:] = r.T, np.eye(3), r.T
    rotated_state = rotated_model.observation_state(
        np.r_[np.zeros(3), rotation.as_quat()[[3,0,1,2]], r@point],
        r@velocity[:3], r@velocity[3:], r@tip_velocity)
    result = np.asarray(rotated_model.prediction_fn(rotated_state, r@wrench/600.,
                        np.zeros(12), jac@transform, 7.)).ravel()
    world_omega = Rotation.from_quat(result[[4,5,6,3]]).apply(result[13:16])
    np.testing.assert_allclose(np.r_[result[10:13], world_omega],
                              np.r_[r@cpo_velocity[:3], r@cpo_velocity[3:]], atol=1e-10)


def test_physical_response_is_independent_of_legacy_impedance_and_retains_incoming_velocity():
    from tests.test_bunny_contact_dynamics import rigid_model
    param = rigid_model().param_
    param.contact_dynamics_variant_ = 'physical'
    model = CartesianRigidContactModel(param)
    state = model.observation_state(np.r_[np.zeros(3), [1.,0.,0.,0.], [.1,0.,.1]],
                                    [.1,0.,0.], [0.,.2,0.])
    phi, jac = np.zeros(4), np.zeros((4,9)); jac[:, 2] = 1.
    expected = np.asarray(model.prediction_fn(state, np.zeros(3), phi, jac, 7.)).ravel()
    param.Q *= 1000
    other = CartesianRigidContactModel(param)
    np.testing.assert_allclose(other.prediction_fn(state, np.zeros(3), phi, jac, 7.), expected[:,None], atol=1e-12)
    assert expected[10] > .01


@pytest.mark.parametrize('region,barrier', [(0.,0.), (.08,0.), (0.,2000.)])
def test_batched_cost_preserves_scalar_ranking_with_invalid_candidates(region, barrier):
    rng = np.random.default_rng(12)
    poses = rng.normal(size=(70,7)); target = rng.normal(size=7)
    config = np.r_[cost_config(500.,20.,20.), region, .02, barrier]
    poses[3, 3:] = 0.; poses[7, 0] = np.nan
    expected = np.array([object_cost_np(p, target, config) for p in poses])
    actual = object_cost_batch_np(poses, target, config)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.argsort(actual), np.argsort(expected))


def test_mpc_cost_uses_continuous_contact_point_without_accessing_lambda_cost(monkeypatch):
    import json
    from pathlib import Path
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from planning.mpc_explicit import _build_mpc_planner_runtime
    record = json.loads((Path(__file__).parent/'fixtures/bunny_pose_contact.json').read_text())
    _, param, mpc, _ = _build_mpc_planner_runtime(record)
    pose = np.asarray(record['msg']['policy_q'])
    rotation = Rotation.from_quat(pose[[4,5,6,3]])
    samples = rotation.apply(param.lambda_optimizer.sample_point) + pose[:3]
    point = np.asarray(record['msg']['actual_contact']['point_world'])
    assert np.min(np.linalg.norm(samples-point, axis=1)) > 1e-5

    class ForbiddenLambdaAccess:
        def __getattribute__(self, name):
            raise AssertionError('MPC cost accessed lambda optimizer: '+name)

    param.lambda_optimizer = ForbiddenLambdaAccess()
    state = np.r_[pose, np.zeros(9)] if mpc.velocity_state else pose
    def evaluate(contact):
        cp = mpc._cost_params(target_p=param.target_p_, target_q=param.target_q_,
            phi_vec=record['msg']['phi_vec'], jac_mat=record['msg']['jac_mat'],
            verify_cost_param=1., virtual_point=pose[7:10], contact_point=contact,
            contact_reference=pose[:7])
        return float(mpc._path_cost(state, np.zeros(3), cp))
    original = evaluate(point)
    assert np.isfinite(original)
    assert evaluate(point+[.007, -.004, .002]) != pytest.approx(original, abs=1e-10)
