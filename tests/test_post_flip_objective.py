"""Post-flip objectives, measured contact ranking, and Cartesian model parity."""
import argparse
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from planning.object_pose_cost import ObjectCostStages, cost_config, object_cost_np, target_up_axis
from planning.mlqp_point import LambdaContactControlOptimizer


@pytest.mark.parametrize('target_q', [[1., 0., 0., 0.], [2**-.5, 0., -2**-.5, 0.]])
def test_pose_retains_target_up_penalty_without_penalizing_yaw_twice(target_q):
    a = SimpleNamespace(pos_coef=500., ori_coef=20., two_stage_object_cost=True)
    stage = ObjectCostStages(a, target_up_axis(target_q))
    stage.stage = 'pose'
    target = np.r_[np.zeros(3), target_q]
    yaw = Rotation.from_euler('z', 80, degrees=True) * Rotation.from_quat(np.array(target_q)[[1,2,3,0]])
    tilted = Rotation.from_euler('x', 30, degrees=True) * yaw
    pose = np.r_[np.zeros(3), yaw.as_quat()[[3,0,1,2]]]
    fallen = np.r_[np.zeros(3), tilted.as_quat()[[3,0,1,2]]]
    legacy = cost_config(500, 20)
    assert object_cost_np(pose, target, stage.config()) == pytest.approx(object_cost_np(pose, target, legacy))
    assert object_cost_np(fallen, target, stage.config()) - object_cost_np(fallen, target, legacy) > 10 * (1 - np.cos(np.pi/6))
    moved = pose.copy(); moved[0] = .08
    assert object_cost_np(moved, target, stage.config()) - object_cost_np(pose, target, stage.config()) == pytest.approx(3.2)


@pytest.mark.parametrize('backend', ['portable', 'acados'])
def test_runtime_mpc_matches_lambda_osc_and_reduces_position_error(monkeypatch, backend):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from examples.mpc.franka.ik2.test_mpc_isaac import _add_rollout_policy_args, ISAAC_TASK_KP
    from planning.mpc_explicit import _build_mpc_planner_runtime
    a = _add_rollout_policy_args(argparse.ArgumentParser()).parse_args(
        ['--obj', 'foam_brick', '--sample_num', '12', '--solver-backend', backend, '--no-online-scm'])
    _, param, mpc, trackers = _build_mpc_planner_runtime(
        dict(args=vars(a), trial_count=0, gravity_=np.zeros(6)))
    np.testing.assert_allclose(param.robot_stiff_, ISAAC_TASK_KP * np.eye(3))
    assert param.lambda_optimizer.robot_stiffness == ISAAC_TASK_KP
    assert param.mpc_q_lb_[-1] == pytest.approx(param.table_height + a.table_clearance)
    assert 'upright_release' not in trackers
    assert not hasattr(param, 'contact_transition_predictor_')
    assert mpc.param_.n_qpos_ == 19 and param.n_qpos_ == 10
    # Isolate the path's object term by holding fingertip/command fixed.
    target = np.r_[param.target_p_, param.target_q_]
    state = np.r_[target, [.5, .1, .45]]
    param.object_cost_config_ = cost_config(500,20)
    packed = mpc._cost_params(target_p=target[:3], target_q=target[3:])
    moved = state.copy(); moved[0] += .08
    assert float(mpc.path_cost_fn(moved, np.zeros(3), packed) - mpc.path_cost_fn(state, np.zeros(3), packed)) == pytest.approx(3.2)

    # Isolate contact tracking: a co-moving fingertip retains zero error;
    # leaving it at the old world point must cost more after a rotation.
    param.object_cost_config_ = cost_config(0, 0)
    tracking = mpc._cost_params(target_p=target[:3], target_q=target[3:],
        contact_reference=state[:7], contact_point=state[7:],
        virtual_point=state[7:], verify_cost_param=1.)
    transformed = state.copy()
    rotation_delta = Rotation.from_euler('y', 25, degrees=True)
    transformed[3:7] = (rotation_delta * Rotation.from_quat(state[[4,5,6,3]])).as_quat()[[3,0,1,2]]
    transformed[7:] = state[:3]+rotation_delta.apply(state[7:]-state[:3])
    held_cost = float(mpc.path_cost_fn(transformed,np.zeros(3),tracking))
    stale = transformed.copy(); stale[7:] = state[7:]
    assert held_cost == pytest.approx(float(mpc.path_cost_fn(state,np.zeros(3),tracking)),abs=1e-10)
    assert float(mpc.path_cost_fn(stale,np.zeros(3),tracking)) > held_cost + 1e-5

    trackers['object_cost_stages'].stage = 'pose'
    param.object_cost_config_ = trackers['object_cost_stages'].config()
    current = state.copy(); current[0] -= .08
    jac = np.zeros((40,9))
    mu = param.mu_object_
    pyramid = np.array([[1,mu,0], [1,0,mu], [1,-mu,0], [1,0,-mu]])
    jac[:4] = pyramid @ np.c_[np.eye(3), np.zeros((3,3)), -np.eye(3)]
    phi = np.ones(40); phi[:4] = 0
    sol = mpc.plan_once(target[:3], target[3:], current, phi, jac,
        virtual_point=current[7:], contact_point=current[7:], verify_cost_param=1.)
    assert sol['solver_backend'] == ('acados' if backend == 'acados' else 'ipopt')
    assert sol['action'][0] > 0
    forces = np.asarray(sol['u_traj']) @ param.robot_stiff_.T
    assert np.max(np.linalg.norm(forces, axis=1)) <= param.contact_command_force_limit_ + 1e-6
    assert np.linalg.norm(sol['action']) <= a.mpc_step_limit + 1e-8
    predicted = np.asarray(sol['rollout_q'])[-1]
    assert np.linalg.norm(predicted[:3] - target[:3]) < np.linalg.norm(current[:3] - target[:3])
    assert object_cost_np(predicted, target, param.object_cost_config_) < object_cost_np(current, target, param.object_cost_config_)
    # A future finger 5 mm away must not retain the initial closed contact.
    withdrawn = current.copy(); withdrawn[7] -= .005
    command = np.array([.004,0.,0.])
    mp = mpc.model.runtime_parameters(param.model_params)
    augmented, reference = mpc.observation_state(withdrawn), mpc.observation_state(current)
    ghost = np.asarray(mpc.model.prediction_fn(augmented,command,phi,jac,mp)).ravel()
    reachable = np.asarray(mpc.stage_prediction_fn(augmented,command,phi,jac,mp,reference)).ravel()
    assert ghost[0] > withdrawn[0] + 1e-6
    assert reachable[0] == pytest.approx(withdrawn[0],abs=1e-10)


@pytest.mark.parametrize('stage', ['flip', 'pose'])
@pytest.mark.parametrize('confidence', [.5, 1.])
@pytest.mark.parametrize('separation', [.02, .1])
@pytest.mark.parametrize('occupied_progress,expected_idx', [(-.001, 1), (.001, 1), (.009, 1), (.0099, 0)])
def test_measured_one_step_gain_does_not_force_mpc_withdrawal(stage, confidence, separation, occupied_progress, expected_idx):
    from examples.mpc.fingertips.test.test_0902 import (
        compute_rollout_contact_via, ContactValueTracker, ModelCostConfidence, SmoothedApproachVia,
    )
    points = np.array([[separation/2, 0., .02], [-separation/2, 0., .02]])
    normals = np.array([[-1., 0., 0.], [1., 0., 0.]])
    current, target = np.r_[np.zeros(3), [1., 0., 0., 0.]], np.r_[[.08, 0., 0.], [1., 0., 0., 0.]]
    predictions = [np.r_[[occupied_progress, 0., 0.], current[3:]], np.r_[[.01, 0., 0.], current[3:]]]
    forces = [np.array([.1, 0., 0.])] * 2
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.__dict__.update(sample_point=points, normal=normals, top_k=2,
        sample_geodesic=np.array([[0., separation], [separation, 0.]]),
        last_global_idx=1, last_selected_idx=0, last_executed_idx=0,
        contact_switch_radius=.03, contact_switch_confidence=confidence, mu_arm_obj=.5,
        pos_coef=500., ori_coef=20., _object_cost_config=cost_config(500, 20, 20),
        update_Jacobian=lambda *a, **kw: None,
        get_availble_point_idx=lambda *a, **kw: np.array([0,1]),
        filter_rankable_indices=lambda ids: ids)
    # Use actual rescoring and top-k selection, with deterministic predicted poses.
    def rank(*a, **kw):
        assert opt.lock_contact_patch == (stage == 'flip' and confidence == 1.)
        if stage == 'flip' and separation < .03:
            assert kw['contact_anchor_local'] is not None
        else:
            assert kw['contact_anchor_local'] is None
        scores = opt._rescore_candidate_costs([0,1], [3., 3.], predictions, target, current)
        opt._store_candidate_buffers([0,1], scores, predictions, forces)
        opt.last_global_total_cost = float(scores[1])
        opt.last_best_x_plus, opt.last_best_force, opt.last_best_cost = predictions[1], forces[1], float(scores[1])
        return points[1], normals[1], float(scores[1]), 1., 1
    opt.choose_contact_points = rank
    raw = object_cost_np(predictions[0], target, cost_config(500,20,20)) + .01
    def solve(*a, **kw):
        opt.last_executed_idx = 0
        return points[0], -normals[0], predictions[0], raw, dict(control_input=forces[0], solver_failed=False)
    opt.resolve_executed_contact = solve
    opt.optimize_control_input = solve
    origin = np.array([.4, .1, .4])
    param = SimpleNamespace(lambda_optimizer=opt, target_p_=origin + target[:3], target_q_=target[3:],
        object_cost_stage_=stage, lambda_actual_contact_=dict(point_world=origin + points[0], normal_world=-normals[0]))
    a = SimpleNamespace(ground_height_threshold=.012, ideal_contact_surface_margin=-.0005)
    tip = origin + points[0] - .0095 * normals[0]
    tracker = ContactValueTracker()
    tracker.contact_active = tracker._holding_p_arm = True
    result = compute_rollout_contact_via(param, a, np.r_[origin, current[3:], tip], np.eye(3),
        np.zeros(6), np.zeros((40,6)), .01, tracker, ModelCostConfidence(), SmoothedApproachVia(),
        False, None, floor_ground=.362, floor_z=.35)
    assert result['best_contact_idx'] == expected_idx
    assert result['value_info']['contact_patch_locked'] == (stage == 'flip' and confidence == 1.)
    # A nearby guide keeps the measured patch even if its single-step gain is
    # negative. A distant destination is still allowed through the normal via.
    executed_idx = 0  # measured support survives a distant ranking winner
    np.testing.assert_allclose(result['p_arm_surface_world'], origin + points[executed_idx])
    assert not result['value_info']['switch_observed_contact']
    assert opt.last_executed_idx == 0  # execution remains the observed contact
    assert opt.last_executed_cost == pytest.approx(opt.normalize_pose_cost(raw - .01))


@pytest.mark.parametrize('backend', ['ipopt', 'acados', 'torch-lbfgs'])
def test_contact_solver_explores_pressure_across_disengaged_branch(tmp_path, monkeypatch, backend):
    import trimesh
    if backend == 'torch-lbfgs':
        import torch
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    from examples.mpc.franka.ik2.contact_frames import support_patch_jacobians
    mesh = tmp_path/'box.stl'
    trimesh.creation.box(extents=[.08]*3).export(str(mesh))
    opt = LambdaContactControlOptimizer(str(mesh), solver=backend, sample_num=8,
        max_contacts=1, pos_coef=0, ori_coef=20, time_step=.02,
        max_contact_force=3., contact_dynamics='quasi-dynamic', wrench_is_force=True,
        grouped_contact_response=True, robot_stiffness=600.)
    target = np.r_[np.zeros(3), [np.cos(np.pi/6), 0., -np.sin(np.pi/6), 0.]]
    current = np.array([0., 0., 0., 1., 0., 0., 0.])
    config = cost_config(0, 0, 20, target_up_axis(target[3:]))
    opt.set_object_cost_config(config)
    _, jac = support_patch_jacobians(np.zeros(3), np.eye(3), [.02, 0, -.04], 6, .5)
    opt.update_Jacobian(jac, phi=np.zeros(4))
    tau, point = np.array([0., 0., -.098, 0., 0., 0.]), np.array([-.02, 0., .04])
    frame = np.diag([1., -1., -1.])[:, [2, 0, 1]]
    zero_velocity = opt._arm_velocity_numpy(np.zeros(6), tau, np.zeros(3), point, frame, jac, np.zeros(4))
    zero_pose = np.asarray(opt.cs_qposInteg_(current, zero_velocity)).ravel()
    result = opt._solve_optimization(x_d=target, current_x=current, v_last=np.zeros(6),
        J_tilde=jac, D_inv=opt.compute_env_diag_inverse(jac), tau_o_np=tau,
        n_arm=frame[:, 0], t1=frame[:, 1], t2=frame[:, 2], p_arm=point, curr_ori_coef=1.)
    assert result is not None
    force = np.asarray(result['lam_arm_opt']).ravel()
    assert force[0] > 1.
    assert np.linalg.norm(force) <= 3. + 1e-6
    predicted = np.asarray(result['x_plus_opt']).ravel()
    assert object_cost_np(predicted, target, config) < object_cost_np(zero_pose, target, config) - 1e-3
