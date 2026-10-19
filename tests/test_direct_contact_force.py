"""Regressions for point-force execution against actual MuJoCo dynamics."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from planning.direct_contact_force import (
    environment_contacts_body, gravity_wrench_body, object_frame,
    object_mass_body, object_velocity_body, step_point_force,
    step_actuator_contact_force,
)


@pytest.fixture
def env():
    model = mujoco.MjModel.from_xml_string('''
      <mujoco>
        <option timestep="0.0001" gravity="0 0 0"/>
        <worldbody>
          <geom name="table" type="plane" size="1 1 .1" friction=".5"/>
          <body name="obj">
            <freejoint/>
            <inertial pos=".02 -.01 .03" quat=".9238795 0 0 .3826834"
                      mass="1" diaginertia=".01 .02 .025"/>
            <geom name="obj" type="sphere" size=".1" friction=".9"/>
          </body>
        </worldbody>
      </mujoco>''')
    data = mujoco.MjData(model)
    quat = Rotation.from_euler('xyz', [.4, -.8, .7]).as_quat()
    data.qpos[:7] = np.r_[0., 0., 1., quat[[3, 0, 1, 2]]]
    mujoco.mj_forward(model, data)
    return SimpleNamespace(model_=model, data_=data,
                           param_=SimpleNamespace(frame_skip_=3),
                           _sync_viewer=lambda: None,
                           get_state=lambda: data.qpos.copy())


def test_body_mass_and_velocity_preserve_kinetic_energy(env):
    """Detects double rotation of the free joint's angular components."""
    mass = np.zeros((6, 6))
    try:
        mujoco.mj_fullM(env.model_, env.data_, mass)
    except TypeError:
        mujoco.mj_fullM(env.model_, mass, env.data_.qM)
    env.data_.qvel[:] = [.1, -.4, .3, 1.5, -.8, 2.1]
    velocity = object_velocity_body(env)
    np.testing.assert_array_equal(velocity[3:], env.data_.qvel[3:])
    np.testing.assert_allclose(velocity @ object_mass_body(env) @ velocity,
                               env.data_.qvel @ mass @ env.data_.qvel, atol=1e-12)
    # This object's inertial axes differ from its body axes.
    assert abs(object_mass_body(env)[3, 4]) > 1e-4


def test_force_stays_attached_to_rotating_surface_and_clears_after_step(env, monkeypatch):
    point = np.array([.08, .02, -.01])
    force = np.array([.1, .7, -.3])
    env.data_.qvel[3:] = [0., 0., 50.]
    calls = []
    apply = mujoco.mj_applyFT

    def capture(model, data, f_world, torque_world, point_world, body, result):
        rot = object_frame(env)
        np.testing.assert_allclose(f_world, rot @ force, atol=1e-12)
        np.testing.assert_allclose(point_world, data.xpos[body]+rot @ point, atol=1e-12)
        apply(model, data, f_world, torque_world, point_world, body, result)
        # Generalized angular force is expressed about the joint origin.
        np.testing.assert_allclose(result[:3], f_world, atol=1e-12)
        np.testing.assert_allclose(result[3:], np.cross(point, force), atol=1e-12)
        calls.append(np.array(f_world))

    monkeypatch.setattr(mujoco, 'mj_applyFT', capture)
    before = env.data_.qpos.copy()
    info = step_point_force(env, point, force)
    assert len(calls) == 3
    assert np.linalg.norm(calls[-1]-calls[0]) > 1e-4
    assert not np.array_equal(env.data_.qpos, before)
    np.testing.assert_allclose(info['force_body'][3:],
                               np.cross(point-env.model_.body_ipos[1], force))
    np.testing.assert_allclose(info['actual_applied_impulse_body'],
                               info['force_body'] * .0003)
    np.testing.assert_array_equal(env.data_.qfrc_applied, 0.)
    np.testing.assert_array_equal(env.data_.xfrc_applied, 0.)


def test_table_contact_rows_match_physical_point_velocity(env):
    env.data_.qpos[2] = .0999
    env.data_.qvel[:] = [.2, -.1, -.3, .5, -.8, 1.]
    mujoco.mj_forward(env.model_, env.data_)
    rows, gaps = environment_contacts_body(env)
    assert rows.shape == (4, 6)
    con = env.data_.contact[0]
    jac = np.zeros((3, 6))
    mujoco.mj_jac(env.model_, env.data_, jac, None, con.pos, env.model_.body('obj').id)
    n, t1, t2 = con.frame.reshape(3, 3)
    if con.geom1 != env.model_.geom('table').id:
        n = -n
    mu = float(con.friction[0])
    rays = np.stack((n+mu*t1, n+mu*t2, n-mu*t1, n-mu*t2))
    np.testing.assert_allclose(rows @ object_velocity_body(env),
                               rays @ jac @ env.data_.qvel, atol=1e-12)
    np.testing.assert_allclose(gaps, con.dist)


def test_gravity_uses_actual_com_moment(env):
    env.model_.opt.gravity[:] = [0., 0., -9.81]
    force = object_frame(env).T @ env.model_.opt.gravity
    np.testing.assert_allclose(gravity_wrench_body(env),
                               np.r_[force, np.cross(env.model_.body_ipos[1], force)])


def test_optimized_force_pulse_preserves_force_and_integrated_impulse(env, monkeypatch):
    env.param_.lambda_optimizer = SimpleNamespace(h=.0003)
    env.force_args_ = SimpleNamespace(force_apply_duration=.0001)
    point = np.array([.08, .02, -.01])
    force = np.array([.7, .1, -.03])
    solved = dict(lam=force, p_local=point, n_arm=np.eye(3)[:, 0],
                  t1=np.eye(3)[:, 1], t2=np.eye(3)[:, 2])
    calls = []
    apply = mujoco.mj_applyFT

    def capture(model, data, f, torque, p, body, result):
        calls.append(object_frame(env).T @ f)
        apply(model, data, f, torque, p, body, result)

    monkeypatch.setattr(mujoco, 'mj_applyFT', capture)
    info = step_actuator_contact_force(env, solved)
    np.testing.assert_allclose(calls, [force, np.zeros(3), np.zeros(3)], atol=1e-12)
    wrench = np.r_[force, np.cross(point-env.model_.body_ipos[1], force)]
    np.testing.assert_allclose(info['applied_force_body'], wrench)
    np.testing.assert_allclose(info['actual_applied_impulse_body'], .0001*wrench)
    np.testing.assert_allclose(info['force_body'], wrench/3)
    assert info['active_substeps'] == 1
    assert not info['force_was_clipped']
    parked = step_actuator_contact_force(env, dict(solved, settling=True))
    assert parked['active_substeps'] == 0
    np.testing.assert_array_equal(parked['applied_force_body'], np.zeros(6))
    np.testing.assert_array_equal(parked['actual_applied_impulse_body'], np.zeros(6))


def test_direct_rollout_uses_corrected_scm_and_continuous_physical_force(monkeypatch):
    from examples.mpc.fingertips.test.test_scm_mujoco import build_parser, _run_rollout
    from planning.mlqp_point import LambdaContactControlOptimizer
    monkeypatch.setattr(LambdaContactControlOptimizer, '_build_acados_contact_solver',
                        lambda *a: pytest.fail('New benchmark must not build a legacy solver'))
    args = build_parser().parse_args([])
    rows = _run_rollout(args, 'scm_corrected', 'foam_brick', seed=0, steps=2500)
    final = rows[-1]
    assert final['success'] and final['settled']
    assert final['success_hold'] >= 10
    assert final['goal_pos'] < .02 and final['goal_quat'] < .015
    np.testing.assert_allclose(final['simulation_time_s'], .02*len(rows), atol=1e-9)
    assert not np.allclose(final['object_pose'], rows[0]['initial_object_pose'])
    for row in rows:
        assert row['optimizer_model'] == 'scm_corrected'
        if row['planning_updated']:
            assert row['planning_time_ms'] > 0
        else:
            assert row['planning_time_ms'] is None
        info = row['execution_info']
        assert info['pose_overwrites'] == 0
        assert not info['force_was_clipped']
        assert np.linalg.norm(row['optimal_force_contact']) <= 10.000001
        assert info['active_substeps'] in (0, 40)
        np.testing.assert_allclose(info['actual_applied_impulse_body'][:3],
            row['applied_force_body'] * .02, atol=1e-12)


@pytest.fixture(scope='module')
def shared_predictors():
    from examples.mpc.fingertips.test.test_0902 import build_parser
    from examples.mpc.fingertips.test.params import ExplicitMPCParams as MjParams
    from examples.mpc.franka.runtime.params import ExplicitMPCParams as IsaacParams, build_lambda_optimizer
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    from planning.direct_contact_force import configure_direct_force_optimizer
    from envs.fingertips_env import MjSimulator
    import os
    import argparse

    os.environ['MUJOCO_HEADLESS'] = '1'
    args = build_parser().parse_args([])
    args.rollout = True
    args.solver = 'acados'
    p = MjParams(args, rand_seed=0)
    simulator = MjSimulator(p)
    scm_opt = configure_direct_force_optimizer(p, simulator, args)
    b = simulator.model_.body('obj').id
    np.testing.assert_array_equal(simulator.model_.body_ipos[b], 0.)
    np.testing.assert_array_equal(simulator.model_.body_iquat[b], [1., 0., 0., 0.])
    np.testing.assert_allclose(simulator.model_.body_inertia[b], p.lambda_source_param_.sim_obj_inertia_diag_)
    for name in ('obj', 'table'):
        g = simulator.model_.geom(name).id
        assert simulator.model_.geom_condim[g] == 3
        np.testing.assert_array_equal(simulator.model_.geom_friction[g], [.7, 0., 0.])
    source_parser = argparse.ArgumentParser()
    _add_rollout_policy_args(source_parser)
    source_args = source_parser.parse_args([])
    source_args.solver = 'acados'
    isaac_p = IsaacParams(source_args, rand_seed=0, build_optimizer=False)
    isaac_opt = build_lambda_optimizer(isaac_p, source_args)
    # A tilted object with a genuine active support patch.
    simulator.data_.qpos[2] -= .001
    mujoco.mj_forward(simulator.model_, simulator.data_)
    jac, gaps = environment_contacts_body(simulator, isaac_p.mu_table_object_)
    assert len(gaps) > 0
    for opt in (scm_opt, isaac_opt):
        opt.update_Jacobian(jac, phi=gaps)
    return scm_opt, isaac_opt, simulator


def test_scm_and_isaac_factories_have_identical_parameters(shared_predictors):
    scm, isaac, _ = shared_predictors
    for name in ('h', 'contact_model', 'wrench_is_force', 'gravity_velocity_scale',
                 'wrench_velocity_scale', 'grouped_contact_response', 'robot_stiffness',
                 'soft_scm_response_time', 'soft_scm_damping_ratio', 'soft_scm_regularization',
                 'max_contacts', 'max_contact_force', 'friction_cone', 'mu_arm_obj',
                 'friction_reg_coef', 'force_reg_coef', 'shared_response', 'cso_guidance',
                 'pos_coef', 'ori_coef', 'top_k'):
        assert getattr(scm, name) == getattr(isaac, name), name
    for name in ('obj_inertia', 'Q_inv', 'dynamics_scale', 'sample_point', 'normal', 't1', 't2'):
        np.testing.assert_array_equal(getattr(scm, name), getattr(isaac, name), err_msg=name)
    assert scm.robot_stiffness == 600.
    assert scm.soft_scm_response_time == .05
    assert scm.friction_cone == 'pyramid'
    assert scm.max_contact_force == 3.


@pytest.mark.parametrize('speed', [0., .02, -.04])
def test_exported_scm_prediction_matches_isaac_nlp_and_numpy(shared_predictors, speed, monkeypatch):
    scm, isaac, _ = shared_predictors
    point = scm.sample_point[0]
    frame = np.column_stack((scm.normal[0], scm.t1[0], scm.t2[0]))
    gravity = np.array([.03, -.04, -.08, 0., 0., 0.])
    velocity = np.array([speed, -.5*speed, 0., .1, -.1, .2])
    target = np.array([.02, -.01, 0., 1., 0., 0., 0.])
    current = np.array([0., 0., 0., 1., 0., 0., 0.])
    force = np.array([.6, .05, -.02])
    reference = isaac._arm_velocity_numpy(velocity, gravity, frame@force, point,
                                           frame, isaac.J_tilde, isaac.env_phi)
    exported = scm.predict_contact_response(force, point, frame, gravity, velocity=velocity)
    np.testing.assert_allclose(exported['v_plus'], reference, atol=1e-12, rtol=1e-12)
    # The generated ACADOS candidate graph is also the authoritative NLP map.
    packed = np.r_[isaac.env_phi, target, velocity,
                    isaac.J_tilde.reshape(-1, order='F'),
                    isaac.compute_env_diag_inverse(isaac.J_tilde).reshape(-1, order='F'),
                    gravity, frame[:, 0], frame[:, 1], frame[:, 2], point,
                    isaac.scm_theta, isaac.object_cost_config(), isaac.max_contact_force]
    expected, _ = isaac._acados_candidate_fn(current, force, packed)
    np.testing.assert_allclose(exported['x_plus'], np.asarray(expected).reshape(7), atol=1e-12)
    # Drive and reaction are different quantities; the reaction must be
    # transmitted through the same nine-DoF contact graph.
    assert not np.allclose(exported['contact_force_body'], frame@force)


@pytest.mark.parametrize('mode', ['qp','lcp'])
def test_baseline_prediction_is_independent_and_does_not_change_plant(shared_predictors, mode, monkeypatch):
    from planning.point_contact_baselines import PointContactBaseline
    scm, _, env = shared_predictors
    env.force_args_ = SimpleNamespace(force_apply_duration=.0005)
    predictor = PointContactBaseline(env, mode)
    signature = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(env.model_, signature))
    mujoco.mj_getState(env.model_, env.data_, state, signature)
    monkeypatch.setattr(scm, 'predict_contact_response',
                        lambda *a, **k: pytest.fail('Baseline cannot use SCM prediction'))
    predictor.prepare()
    frame = np.column_stack((scm.normal[0],scm.t1[0],scm.t2[0]))
    command = np.array([.6,.05,-.02])
    out = predictor.predict(command,scm.sample_point[0],frame)
    repeated = predictor.predict(command,scm.sample_point[0],frame)
    np.testing.assert_allclose(out['x_plus'],repeated['x_plus'],atol=1e-12)
    after = np.zeros_like(state)
    mujoco.mj_getState(env.model_,env.data_,after,signature)
    np.testing.assert_array_equal(after,state)
    assert np.isfinite(out['x_plus']).all()
    if mode == 'lcp':
        assert out['residual'] <= 1e-6
    else:
        assert predictor.model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
        # Recompute the entire native force pipeline independently: skipping
        # cached position/velocity work must give the same actual QP result.
        m,d=predictor.model,predictor.data
        expected=d.qacc.copy()
        d.qacc_warmstart[:]=predictor.warmstart
        mujoco.mj_forward(m,d)
        np.testing.assert_allclose(d.qacc,expected,atol=1e-9)
