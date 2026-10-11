from types import SimpleNamespace

import casadi as cs
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from planning.object_pose_cost import (
    ObjectCostStages, cost_config, object_cost_np, object_cost_cs,
    object_cost_torch, rotate_axis_np, reset_objective_history,
)


def quat(angles):
    q = Rotation.from_euler('xyz', angles, degrees=True).as_quat()
    return q[[3, 0, 1, 2]]


def args(**kwargs):
    return SimpleNamespace(**dict(dict(pos_coef=500., ori_coef=20., two_stage_object_cost=True), **kwargs))


def obs(t, tilt=0, linear=0, angular=0):
    return dict(sim_time=t, policy_q=np.r_[np.zeros(3), quat([tilt, 0, 80])],
                object_linear_velocity=[linear, 0, 0], object_angular_velocity=[0, 0, angular])


def test_flip_ignores_position_yaw_and_quaternion_sign_but_not_inversion():
    c = cost_config(0, 0, 20)
    target = np.r_[np.zeros(3), quat([0, 0, -100])]
    for yaw in [-175, 0, 83]:
        q = quat([0, 0, yaw])
        for sign in [1, -1]:
            assert object_cost_np(np.r_[[8, -9, 4], sign * q], target, c) == pytest.approx(0, abs=1e-12)
    assert object_cost_np(np.r_[np.zeros(3), quat([180, 0, 0])], target, c) == pytest.approx(20)


def test_numpy_symbolic_torch_and_lambda_frame_agree():
    import torch
    current = quat([45, -20, 80])
    prediction = quat([32, -12, 95])
    target = np.r_[[.4, .1, .38], quat([0, 0, -50])]
    pose = np.r_[[.35, .2, .4], prediction]
    rot = Rotation.from_quat(current[[1, 2, 3, 0]])
    relative = (rot.inv() * Rotation.from_quat(prediction[[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
    up_local = rot.inv().apply([0, 0, 1])
    world_config = cost_config(0, 0, 20)
    local_config = cost_config(0, 0, 20, reference_up=up_local)
    expected = object_cost_np(pose, target, world_config)
    assert object_cost_np(np.r_[np.zeros(3), relative], target, local_config) == pytest.approx(expected)
    x, xd, c = cs.SX.sym('x', 7), cs.SX.sym('xd', 7), cs.SX.sym('c', 9)
    fn = cs.Function('object_cost_test', [x, xd, c], [object_cost_cs(x, xd, c)])
    for config in [world_config, cost_config(500, 20)]:
        expected = object_cost_np(pose, target, config)
        assert float(fn(pose, target, config)) == pytest.approx(expected, abs=1e-9)
        xt, xdt = torch.tensor(pose), torch.tensor(target)
        assert float(object_cost_torch(xt[:3], xt[3:7], xdt, config)) == pytest.approx(expected)
        batch = object_cost_torch(xt[:3].repeat(3, 1), xt[3:7].repeat(3, 1), xdt, config)
        np.testing.assert_allclose(batch.numpy(), expected)


def test_complete_rotation_retains_legacy_rti_hessian():
    x, target = cs.SX.sym('state', 7), cs.SX.sym('target', 7)
    config = cs.SX.sym('config', 9)
    cost = object_cost_cs(x, target, config)
    legacy = 500 * cs.sumsqr(x[:3] - target[:3]) + 20 * (1 - cs.dot(x[3:7], target[3:7])**2)
    difference = cost - legacy
    fn = cs.Function('legacy_derivatives', [x, target, config],
        [difference, cs.gradient(difference, x), cs.hessian(difference, x)[0]])
    value, gradient, hessian = fn(np.r_[[.3, .2, .4], quat([25, -10, 75])],
        np.r_[[.4, .1, .38], quat([0, 0, -50])], cost_config(500, 20))
    assert float(value) == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(gradient, 0, atol=1e-12)
    np.testing.assert_allclose(hessian, 0, atol=1e-12)


def test_stable_gate_duplicates_missing_frames_and_fallback():
    stages = ObjectCostStages(args())
    assert not stages.update(obs(0))
    assert not stages.update(obs(.2, angular=.3))
    assert not stages.update(obs(.3))
    assert not stages.update(obs(.3))
    assert not stages.update(obs(.5, linear=.03))
    assert not stages.update(obs(.6))
    assert not stages.update({'sim_time': .8})
    assert not stages.update(obs(.9))
    assert not stages.update(obs(1.1))
    assert stages.update(obs(1.2))
    assert stages.stage == 'pose' and stages.first_flip_time == pytest.approx(1.2)
    assert not stages.update(obs(1.3, tilt=26))
    assert not stages.update(obs(1.35, tilt=24))
    assert not stages.update(obs(1.4, tilt=30))
    assert stages.update(obs(1.5, tilt=30))
    assert stages.stage == 'flip' and stages.fallbacks == 1 and stages.switches == 2
    assert ObjectCostStages(args()).first_flip_time is None


def test_disabled_cost_legacy_monitor_and_validation():
    stages = ObjectCostStages(args(two_stage_object_cost=False))
    assert stages.stage == 'pose'
    np.testing.assert_allclose(stages.config(), cost_config(500, 20))
    assert not stages.update(obs(0))
    assert not stages.update(obs(.3))
    assert stages.first_flip_time == pytest.approx(.3)
    assert stages.switches == 0
    with pytest.raises(ValueError, match='ori_coef'):
        ObjectCostStages(args(ori_coef=0))
    with pytest.raises(ValueError, match='threshold'):
        ObjectCostStages(args(flip_exit_angle_deg=5))


def test_authored_standing_axis_matches_world_up():
    standing = quat([0, 90, 0])
    axis = Rotation.from_quat(standing[[1, 2, 3, 0]]).inv().apply([0, 0, 1])
    np.testing.assert_allclose(rotate_axis_np(standing, axis), [0, 0, 1], atol=1e-12)


def test_missing_or_nonfinite_physical_observation_cannot_be_stable():
    import json
    stages = ObjectCostStages(args())
    stages.update(obs(0))
    assert not stages.update(obs(.4, linear=float('nan')))
    json.dumps(stages.diagnostics(), allow_nan=False)
    assert not stages.update(obs(.5))
    assert not stages.update(obs(.7))
    assert stages.update(obs(.8))


def test_feedback_and_payload_use_active_object_cost(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from examples.mpc.franka.ik2.test_mppi_isaac import _dwell_payload, _plan_from_obs, _pred_reduction_from_policy
    config = cost_config(0, 0, 20)
    current = np.r_[[.3, .1, .4], quat([20, 0, 80]), [.2, .1, .4]]
    p = SimpleNamespace(target_p_=np.zeros(3), target_q_=quat([0, 0, -120]))
    policy = dict(object_cost_config=config, object_cost_stage='flip', object_progress_scale=20.,
                  x_plus_opt=None, p_arm_world=np.zeros(3))
    now, reduction = _pred_reduction_from_policy(p, current, policy)
    assert now == pytest.approx(20 * (1 - np.cos(np.deg2rad(20))) / 2)
    assert reduction is None
    dwell = _dwell_payload(None, None, None, policy, False, False, False, now, None,
        .01, 99., curr_q=current, param=p)
    assert dwell['c_after'] == pytest.approx(now)
    assert dwell['progress_cost'] == pytest.approx(now / 20)
    physical_obs = dict(obs(.5), table_ground=.35, actual_contact=dict(point_world=[.1,.2,.3],normal_world=[1,0,0]))
    payload, _ = _plan_from_obs(physical_obs, dwell)
    assert payload['sim_time'] == .5
    assert payload['actual_contact'] == physical_obs['actual_contact']
    assert payload['object_angular_velocity'] == [0, 0, 0]
    assert payload['dwell']['object_cost_stage'] == 'flip'


def test_confidence_scores_executed_mpc_prediction_instead_of_ideal_lambda(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from examples.mpc.franka.ik2.test_mppi_isaac import _pred_reduction_from_policy
    config = cost_config(500, 0, 20)
    current = np.r_[[.3, .1, .4], quat([60, 0, 0]), [.2, .1, .4]]
    target = np.r_[[.4, .1, .4], quat([0, 0, 0])]
    executed = np.r_[[.32, .1, .4], quat([50, 0, 0])]
    param = SimpleNamespace(target_p_=target[:3], target_q_=target[3:])
    policy = dict(object_cost_config=config, execution_object_pose_world=executed,
        x_plus_opt=np.array([0., 0., 0., 1., 0., 0., 0.]),
        info=dict(control_input=np.ones(3), solver_failed=False))
    now, reduction = _pred_reduction_from_policy(param, current, policy)
    assert now == pytest.approx(object_cost_np(current[:7], target, config))
    assert reduction == pytest.approx(now - object_cost_np(executed, target, config))


def test_mpc_runtime_parameters_and_disabled_signature():
    from examples.mpc.franka.ik2.params import ExplicitMPCParams
    from planning.mpc_costs import pack_cost_params
    p = ExplicitMPCParams.__new__(ExplicitMPCParams)
    p.__dict__.update(n_qpos_=10, n_qvel_=9, n_cmd_=3, max_ncon_=10,
        smooth_contact_detour=False, quadratic_contact_track=True, attract_coef=20.,
        reject_dis=.001, reject_coef=0., contact_coef=20., contact_cost_param=0., spline_escape_cost=False,
        pos_coef=500., ori_coef=20., runtime_object_cost_=True)
    path, final = p.init_cost_fns()
    pose = np.r_[[.3, .2, .4], quat([25, 0, 75])]
    target = np.r_[[.4, .1, .38], quat([0, 0, -50])]
    for c in [cost_config(0, 0, 20), cost_config(500, 20), cost_config(500, 20, 20)]:
        p.object_cost_config_ = c
        packed = pack_cost_params('param', p, path, target_p=target[:3], target_q=target[3:])
        np.testing.assert_allclose(packed[-12:], np.r_[c, [0.,0.,0.]])
        assert float(final(np.r_[pose, [.2, .2, .4]], packed)) == pytest.approx(10 * object_cost_np(pose, target, c), abs=1e-8)
    p.runtime_object_cost_ = False
    legacy_path, _ = p.init_cost_fns()
    assert path.size_in(2)[0] == legacy_path.size_in(2)[0] + 7 + 12


@pytest.mark.parametrize('backend', ['ipopt', 'acados', 'torch-lbfgs'])
def test_lambda_solver_switches_without_rebuild(tmp_path, backend, monkeypatch):
    import trimesh
    from planning.mlqp_point import LambdaContactControlOptimizer
    if backend == 'torch-lbfgs':
        import torch
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    mesh = tmp_path / 'box.stl'
    trimesh.creation.box(extents=[.08, .08, .08]).export(str(mesh))
    opt = LambdaContactControlOptimizer(str(mesh), solver=backend, sample_num=12,
        pos_coef=500, ori_coef=20, force_reg_coef=.01, friction_reg_coef=0., time_step=.05)
    if backend == 'acados':
        assert opt.acados_solver is not None, str(opt.acados_error)
    solver_id = id(opt.acados_solver)
    current, target = np.r_[np.zeros(3), [1., 0, 0, 0]], np.r_[[.03, 0, 0], quat([0, 30, 70])]
    opt.update_Jacobian(np.zeros((40, 6)))
    common = dict(x_d=target, current_x=current, v_last=np.zeros(6),
        J_tilde=opt.J_tilde, D_inv=opt.compute_env_diag_inverse(opt.J_tilde),
        tau_o_np=np.zeros(6), n_arm=[1., 0, 0], t1=[0., 1, 0], t2=[0., 0, 1],
        p_arm=[0., 0, .03], curr_ori_coef=1.)
    results = []
    for c in [cost_config(0, 0, 20), cost_config(500, 20), cost_config(500, 20, 20)]:
        opt.set_object_cost_config(c)
        result = opt._solve_optimization(**common)
        assert result is not None
        if backend == 'acados':
            assert opt.last_solver_status == 'acados'
        elif backend == 'torch-lbfgs':
            assert opt.last_solver_status == 'torch-lbfgs'
        force = np.asarray(result['lam_arm_opt']).reshape(3)
        expected = object_cost_np(np.asarray(result['x_plus_opt']).reshape(7), target, c) + .01 * force @ force
        assert float(result['cost']) == pytest.approx(expected, abs=2e-6 if backend == 'torch-lbfgs' else 1e-7)
        assert opt._pose_cost(current, target) == pytest.approx(object_cost_np(current, target, c))
        results.append(force)
    assert id(opt.acados_solver) == solver_id
    assert np.linalg.norm(results[1] - results[0]) > .001


@pytest.mark.parametrize('backend', ['ipopt', 'acados'])
def test_mpc_solver_uses_runtime_cost_without_rebuild(backend):
    import argparse
    from examples.mpc.fingertips.test.test_0902 import add_rollout_via_args
    from examples.mpc.franka.ik2.params import ExplicitMPCParams
    from planning.mpc_explicit import MPCExplicit
    parser = argparse.ArgumentParser()
    add_rollout_via_args(parser)
    a = parser.parse_args(['--obj', 'stanford_bunny2', '--sample_num', '12'])
    a.two_stage_object_cost, a.solver_backend = True, backend
    p = ExplicitMPCParams(a)
    p.field_cost_weight, p.quadratic_contact_track = 0., True
    p.object_cost_config_ = cost_config(0, 0, 20)
    mpc = MPCExplicit(p)
    if backend == 'acados':
        assert mpc.acados_solver_ is not None, str(mpc._acados_init_error)
    solver = mpc.acados_solver_
    current = np.r_[p.init_obj_qpos_, p.init_fingertip_pos_]
    phi, jac = np.ones(40), np.zeros((40, 9))
    values = []
    for config in [cost_config(0, 0, 20), cost_config(500, 20)]:
        p.object_cost_config_ = config
        result = mpc.plan_once(p.target_p_, p.target_q_, current, phi, jac,
            virtual_point=current[7:], contact_point=current[7:])
        assert result['solver_backend'] == backend
        if backend == 'ipopt':
            assert result['solve_status'] == 'Solve_Succeeded'
        params = mpc._cost_params(target_p=p.target_p_, target_q=p.target_q_, phi_vec=phi,
                                  jac_mat=jac, virtual_point=current[7:], contact_point=current[7:])
        trajectory = np.asarray(result['sol_guess']['x0']).reshape(5, 3+mpc.param_.n_qpos_)
        expected = float(mpc._terminal_cost(trajectory[-1, 3:], params))
        states = np.vstack([mpc.observation_state(current), trajectory[:-1, 3:]])
        expected += sum(float(mpc._path_cost(x, u, params)) for x, u in zip(states, trajectory[:, :3]))
        assert float(np.asarray(result['cost_opt']).reshape(-1)[0]) == pytest.approx(expected, abs=1e-6)
        values.append(expected)
    assert mpc.acados_solver_ is solver
    assert abs(values[0] - values[1]) > 1e-3
    assert mpc.acados_failure_count == 0


def test_switch_discards_cost_history_but_preserves_physics():
    class Tracker:
        def reset_contact(self):
            self.reset_called = True
        def reset(self):
            self.reset_called = True
    opt = SimpleNamespace(last_candidate_deltas=np.ones(3), last_selected_idx=3,
        last_executed_force=np.ones(3), _pending_selected_count=4, _blocked_contact_indices={3: 9},
        sample_point=np.ones((4, 3)), scm_theta=np.arange(6))
    class MPC:
        def _reset_acados_solver(self):
            self.reset_called = True
    mpc = MPC()
    trackers = dict(value_tracker=Tracker(), model_cost_conf=Tracker(), sol_guess={'old': 1})
    reset_objective_history(opt, mpc, trackers)
    assert opt.last_selected_idx is None and opt.last_candidate_deltas is None
    assert opt._pending_selected_count == 0 and opt._blocked_contact_indices == {}
    assert trackers['sol_guess'] is None and mpc.reset_called
    np.testing.assert_array_equal(opt.scm_theta, np.arange(6))


@pytest.mark.parametrize('target_angles', [[0, -90, 40], [0, 90, 15], [0, 0, 75], [35, 25, -60]])
def test_target_pose_defines_upright_family_independently_of_object_axis(target_angles):
    from planning.object_pose_cost import target_up_axis
    target_q = quat(target_angles)
    axis = target_up_axis(target_q)
    np.testing.assert_allclose(target_up_axis(-target_q), axis, atol=1e-12)
    c = cost_config(0, 0, 20, body_axis=axis)
    target = np.r_[np.zeros(3), target_q]
    np.testing.assert_allclose(rotate_axis_np(target_q, axis), [0., 0., 1.], atol=1e-12)
    # World yaw is free, including when the target's local X is up.
    for yaw in [-145, 50, 180]:
        pred = (Rotation.from_euler('z', yaw, degrees=True) * Rotation.from_quat(target_q[[1,2,3,0]])).as_quat()[[3,0,1,2]]
        assert object_cost_np(np.r_[[.4, -.3, .8], pred], target, c) == pytest.approx(0, abs=1e-12)
    # Lambda's relative rotation must use the same target-derived body axis.
    current = Rotation.from_euler('xyz', [40, -25, 70], degrees=True)
    prediction = Rotation.from_euler('x', 17, degrees=True) * Rotation.from_quat(target_q[[1,2,3,0]])
    world_q = prediction.as_quat()[[3,0,1,2]]
    local_q = (current.inv() * prediction).as_quat()[[3,0,1,2]]
    local_c = cost_config(0, 0, 20, body_axis=axis, reference_up=current.inv().apply([0,0,1]))
    world_cost = object_cost_np(np.r_[np.zeros(3), world_q], target, c)
    assert object_cost_np(np.r_[np.zeros(3), local_q], target, local_c) == pytest.approx(world_cost)
    import torch
    for q, config in [(world_q, c), (local_q, local_c)]:
        pose = np.r_[np.zeros(3), q]
        assert float(object_cost_cs(cs.DM(pose), cs.DM(target), cs.DM(config))) == pytest.approx(world_cost, abs=1e-9)
        assert float(object_cost_torch(torch.tensor(pose[:3]), torch.tensor(q), torch.tensor(target), config)) == pytest.approx(world_cost)
    stage = ObjectCostStages(args(), axis)
    state = dict(sim_time=0, policy_q=target, object_linear_velocity=[0,0,0], object_angular_velocity=[0,0,0])
    assert not stage.update(state)
    state['sim_time'] = .3
    assert stage.update(state) and stage.stage == 'pose'
    # A half turn around a world horizontal axis inverts the target-up axis.
    inverted = (Rotation.from_euler('x', 180, degrees=True) * Rotation.from_quat(target_q[[1,2,3,0]])).as_quat()[[3,0,1,2]]
    assert object_cost_np(np.r_[np.zeros(3), inverted], target, c) == pytest.approx(20)


def test_flip_stage_weights_position_and_up_then_restores_full_pose():
    stage = ObjectCostStages(args(obj='stanford_bunny2'))
    np.testing.assert_allclose(stage.config()[:3], [500., 0., 20.])
    target = np.r_[[.3, .1, .4], quat([0, 0, 65])]
    standing = np.r_[target[:3], quat([0, 0, -80])]
    displaced = standing.copy(); displaced[0] += .1
    assert object_cost_np(standing, target, stage.config()) == pytest.approx(0.)
    assert object_cost_np(displaced, target, stage.config()) == pytest.approx(.2)
    assert stage.progress_scale() == pytest.approx(20.2)
    stage.update(obs(0)); assert stage.update(obs(.3))
    np.testing.assert_allclose(stage.config()[:3], [500., 20., 20.])
    assert object_cost_np(standing, target, stage.config()) > 0.
    assert stage.progress_scale() == pytest.approx(.8)


@pytest.mark.parametrize('backend', ['ipopt', 'acados', 'torch-lbfgs'])
@pytest.mark.parametrize('grouped,drive_robot', [(False,False),(True,False),(True,True)])
def test_runtime_force_cap_and_quasi_prediction_match_mpc(tmp_path, backend, monkeypatch, grouped, drive_robot):
    import trimesh
    from planning.mlqp_point import LambdaContactControlOptimizer
    from models.explicit_model import ExplicitModel
    if backend == 'torch-lbfgs':
        import torch
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    mesh = tmp_path/'box.stl'
    trimesh.creation.box(extents=[.08]*3).export(str(mesh))
    opt = LambdaContactControlOptimizer(str(mesh), solver=backend, sample_num=8,
        pos_coef=500, ori_coef=20, time_step=.02, force_reg_coef=.01,
        max_contact_force=.8, contact_dynamics='quasi-dynamic', wrench_is_force=True,
        grouped_contact_response=grouped, robot_stiffness=600. if drive_robot else None)
    solver_id = id(opt.acados_solver)
    if backend == 'acados':
        assert opt.acados_solver is not None, opt.acados_error
    current = np.r_[np.zeros(3), [1., 0, 0, 0]]
    target = np.r_[[.2, 0, 0], [1., 0, 0, 0]]
    opt.set_object_cost_config(cost_config(500, 20))
    j = np.zeros((40, 6))
    tau = np.zeros(6)
    if grouped:
        from examples.mpc.franka.ik2.contact_frames import planar_table_jacobians
        _, j[:4], _ = planar_table_jacobians([0,0,.04], np.eye(3), 0., 6, .5)
        tau[2] = -.098
    opt.update_Jacobian(j)
    common = dict(x_d=target, current_x=current, v_last=np.zeros(6),
        J_tilde=j, D_inv=opt.compute_env_diag_inverse(j), tau_o_np=tau,
        n_arm=[1., 0, 0], t1=[0., 1, 0], t2=[0., 0, 1], p_arm=[0., 0, .02], curr_ori_coef=1.)
    results = []
    for limit in [.8, .25, .8]:
        opt.set_contact_force_limit(limit)
        result = opt._solve_optimization(**common)
        assert result is not None
        f = np.asarray(result['lam_arm_opt']).reshape(3)
        assert np.linalg.norm(f) <= limit + 1e-6
        # Same impulse/displacement convention as MPC, without a moving arm.
        wrench = tau.copy() if drive_robot else tau + np.r_[f, np.cross([0., 0, .02], f)]
        Q = np.eye(9); Q[:6, :6] = opt.obj_inertia
        full_j = np.c_[j, np.zeros((40,3))]
        command = np.zeros(3)
        if drive_robot:
            Q[6:,6:] = 600.*np.eye(3)
            command = f/600.
            Jc = np.array([[1,0,0,0,.02,0],[0,1,0,-.02,0,0],[0,0,1,0,0,0.]])
            mu = opt.mu_arm_obj
            pyramid = np.array([[1,mu,0],[1,0,mu],[1,-mu,0],[1,0,-mu]])
            full_j = np.r_[full_j, pyramid@np.c_[Jc,-np.eye(3)]]
        param = SimpleNamespace(h_=.02, n_qpos_=10, n_qvel_=9, n_robot_qpos_=3,
            n_cmd_=3, max_ncon_=full_j.shape[0]//4, obj_mass_=1., gravity_=wrench, Q=Q,
            robot_stiff_=(600. if drive_robot else 1.)*np.eye(3), contact_dynamics='quasi-dynamic', grouped_contact_response_=grouped)
        model = ExplicitModel(param)
        expected = np.asarray(model.step_once_fn(np.r_[current, np.zeros(3)],
            command, np.zeros(full_j.shape[0]), full_j, .2)).reshape(-1)[:7]
        np.testing.assert_allclose(np.asarray(result['x_plus_opt']).reshape(7), expected, atol=2e-6)
        cost = object_cost_np(expected, target, opt.object_cost_config()) + .01*np.dot(f, f)
        assert float(result['cost']) == pytest.approx(cost, abs=2e-5)
        assert id(opt.acados_solver) == solver_id
        results.append(expected[0])
    assert results[1] < results[0] and results[2] == pytest.approx(results[0], abs=1e-4)


def test_weighted_flip_cost_is_invariant_under_lambda_frame_transform():
    origin = np.array([.4, -.1, .3])
    r = Rotation.from_euler('xyz', [42, -15, 70], degrees=True)
    prediction = np.r_[[.35, .2, .4], quat([25, 12, 85])]
    target = np.r_[[.3, .1, .36], quat([0, -90, 10])]
    from planning.object_pose_cost import target_up_axis
    axis = target_up_axis(target[3:7])
    world = cost_config(500, 0, 20, body_axis=axis)
    local = cost_config(500, 0, 20, body_axis=axis, reference_up=r.inv().apply([0,0,1]))
    def transform(pose):
        q = (r.inv() * Rotation.from_quat(pose[[4,5,6,3]])).as_quat()[[3,0,1,2]]
        return np.r_[r.inv().apply(pose[:3] - origin), q]
    assert object_cost_np(transform(prediction), transform(target), local) == pytest.approx(
        object_cost_np(prediction, target, world))


def test_observed_contact_overrides_sample_geometry_without_corrupting_mesh():
    from planning.mlqp_point import LambdaContactControlOptimizer
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.sample_point = np.array([[.01, .02, .03]])
    opt.normal = np.array([[1., 0, 0]])
    opt.t1 = np.array([[0., 1, 0]])
    opt.t2 = np.array([[0., 0, 1]])
    opt.J_tilde = np.zeros((4,6))
    opt.compute_env_diag_inverse = lambda _: np.ones(4)
    captured = {}
    def solve(**kw):
        captured.update(kw)
        return dict(lam_arm_opt=np.array([0.,-.1,0.]), x_plus_opt=kw['current_x'], cost=2.)
    opt._solve_optimization = solve
    pose = np.r_[np.zeros(3), [1.,0,0,0]]
    point = np.array([.035,-.015,.025])
    actual = dict(contact_point_local=point, contact_normal_inward=[0,2,0])
    p, outward, _, cost, info = opt.optimize_control_input(pose, pose, np.zeros(6), sample_idx=0, **actual)
    np.testing.assert_allclose(p, point)
    np.testing.assert_allclose(outward, [0,-1,0])
    np.testing.assert_allclose(captured['p_arm'], point)
    np.testing.assert_allclose(captured['n_arm'], [0,1,0])
    frame = np.array([captured['n_arm'], captured['t1'], captured['t2']])
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(opt.sample_point, [[.01,.02,.03]])
    assert cost == 2. and not info['solver_failed'] and opt.last_executed_idx == 0
    with pytest.raises(ValueError, match='both'):
        opt.optimize_control_input(pose, pose, np.zeros(6), sample_idx=0, contact_point_local=point)
