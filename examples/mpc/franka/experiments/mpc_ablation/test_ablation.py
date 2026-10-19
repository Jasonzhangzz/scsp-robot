"""Behavioral checks for the current MPC ablation interface (CPU only)."""
import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import casadi as cs
import numpy as np
import pytest

from .config import MODES, configure_ablation, selection_parser
from .params import ExplicitMPCParams, BaselineParams, build_lambda_optimizer
from . import policy
from .stages import make_object_cost_stages
from planning.mpc_costs import pack_cost_params
from planning.object_pose_cost import ObjectCostStages, cost_config


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from .mpc import _add_rollout_policy_args
    return _add_rollout_policy_args(argparse.ArgumentParser())


@pytest.mark.parametrize('mode', MODES)
def test_selection_and_exclusive_flags(mode):
    args = selection_parser().parse_args([] if mode == 'full' else ['--' + mode])
    assert configure_ablation(args).ablation_mode == mode
    with pytest.raises(SystemExit):
        selection_parser().parse_args(['--a1', '--a2'])


def test_baseline_defaults_and_execution_protocol_are_preserved(parser):
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args as add_baseline
    from .mpc import _apply_execution_mode
    original = add_baseline(argparse.ArgumentParser()).parse_args([])
    args = parser.parse_args([])
    for name, value in vars(original).items():
        assert getattr(args, name) == value
    assert not _apply_execution_mode(parser.parse_args(['--a1', '--sync-planner'])).async_planner
    assert _apply_execution_mode(parser.parse_args(['--a2', '--async-planner'])).async_planner
    assert not _apply_execution_mode(parser.parse_args(['--trial', '--sync-planner'])).async_planner


@pytest.mark.parametrize('beta,threshold,accepted', [(.1, .3, False), (.5, .5, True)])
def test_sensitivity_parameters_reach_the_live_verify_tracker(parser, monkeypatch, beta, threshold, accepted):
    from . import planner
    args = parser.parse_args(['--obj', 'elephant', '--verify_beta', str(beta),
                              '--verify-accept-threshold', str(threshold)])
    args.solver_backend, args.sample_num = 'torch-lbfgs', 6
    monkeypatch.setattr(planner, 'MPCExplicit', Mock(return_value=SimpleNamespace(tool_pose_control=False)))
    _, _, _, trackers = planner._build_mpc_planner_runtime(dict(args=vars(args), trial_count=0))
    tracker = trackers['value_tracker']
    value, _ = tracker.update_verify(tightness=.8)
    assert value == pytest.approx(beta * .8)
    info = tracker.update_values(0., 1., same_patch=True, near_arm=True, tightness=.4)
    assert info['accept_p_arm'] is accepted
    # Accepted contacts retain the baseline hysteresis until release tightness .65.
    if accepted:
        assert tracker.update_values(0., 1., same_patch=True, near_arm=True,
                                     tightness=.6)['accept_p_arm']
        assert not tracker.update_values(0., 1., same_patch=True, near_arm=True,
                                         tightness=.7)['accept_p_arm']


@pytest.mark.parametrize('obj', ['elephant', 'stanford_bunny2', 'mug'])
@pytest.mark.parametrize('mode', MODES)
def test_ablation_keeps_scene_and_initial_pose(parser, obj, mode):
    args = parser.parse_args(['--obj', obj])
    original = BaselineParams(args, rand_seed=3, build_optimizer=False)
    changed = ExplicitMPCParams(parser.parse_args(['--obj', obj, '--ablation-mode', mode]),
                                rand_seed=3, build_optimizer=False)
    for name in ('mesh_path_', 'visual_mesh_path_', 'collision_mesh_paths_',
                 'collision_geometry_', 'contact_sampling_', 'table_height',
                 'sim_obj_mass_', 'sim_obj_inertia_diag_', 'obj_inertia_',
                 'gravity_', 'h_', 'Q', 'init_obj_qpos_', 'target_p_', 'target_q_',
                 'object_vertices_', 'mu_table_object_', 'mu_fingertip_object_'):
        if hasattr(original, name):
            np.testing.assert_equal(getattr(changed, name), getattr(original, name), err_msg=name)
    assert changed.runtime_object_cost_ == original.runtime_object_cost_


@pytest.mark.parametrize('task', ['rotation', 'ground-rotation'])
@pytest.mark.parametrize('mode', MODES)
def test_all_modes_share_the_enlarged_baseline_translation(parser, task, mode):
    args = parser.parse_args(['--obj', 'elephant', '--target-type', task])
    baseline = BaselineParams(args, rand_seed=4, build_optimizer=False)
    args = parser.parse_args(['--obj', 'elephant', '--target-type', task,
                              '--ablation-mode', mode])
    ablated = ExplicitMPCParams(args, rand_seed=4, build_optimizer=False)
    np.testing.assert_array_equal(ablated.target_p_, baseline.target_p_)
    np.testing.assert_array_equal(ablated.init_obj_qpos_, baseline.init_obj_qpos_)
    lower, upper = (.20, .30) if task == 'rotation' else (.12, .20)
    assert lower <= np.linalg.norm(ablated.target_p_[:2] - ablated.init_obj_qpos_[:2]) <= upper


def params_for_cost(mode):
    param = ExplicitMPCParams.__new__(ExplicitMPCParams)
    param.__dict__.update(n_qpos_=10, n_qvel_=9, n_cmd_=3, max_ncon_=10,
        smooth_contact_detour=False, quadratic_contact_track=True,
        attract_coef=20., reject_dis=.001, reject_coef=100., contact_coef=20.,
        contact_cost_param=1., spline_escape_cost=False, field_cost_weight=0.,
        pos_coef=500., ori_coef=20., runtime_object_cost_=True,
        cpo_variant='full', rollout_press_patch=True, ablation_mode=mode,
        a6_center_weight=.05, object_geometry_center_local_=np.array([.03, .01, -.02]),
        object_cost_config_=cost_config(500., 20.))
    return param


def cost(param, obj, tip, contact, verify=1., virtual=(1., 2., 3.)):
    path, final = param.init_cost_fns()
    if param.ablation_mode == 'a6':
        from planning.object_pose_cost import extended_cost_config
        packed = np.r_[obj, extended_cost_config(param.object_cost_config_)]
        state = np.r_[obj, tip]
        return float(path(state, np.zeros(3), packed)), float(final(state, packed))
    packed = pack_cost_params('param', param, path,
        target_p=obj[:3], target_q=obj[3:7], verify_cost_param=verify,
        virtual_point=virtual, contact_point=contact, contact_reference=obj)
    state = np.r_[obj, tip]
    return float(path(state, np.zeros(3), packed)), float(final(state, packed))


@pytest.mark.parametrize('mode', ['a1', 'a2'])
def test_direct_cost_tracks_target_without_center_via_or_verify(mode):
    param = params_for_cost(mode)
    obj = np.r_[[.1, 0., .4], [1., 0., 0., 0.]]
    tip = np.array([.2, .1, .5])
    assert cost(param, obj, tip, tip)[0] == pytest.approx(0.)
    assert cost(param, obj, tip, tip, verify=0., virtual=(-8., 3., -5.))[0] == pytest.approx(0.)
    assert cost(param, obj, tip, tip + [.05, 0., 0.])[0] == pytest.approx(20 * .05**2)


def test_a6_cost_uses_center_and_task_without_verify_or_contact_references():
    ablated = params_for_cost('a6')
    obj = np.r_[[.1, -.1, .4], [np.sqrt(.5), 0., 0., np.sqrt(.5)]]
    tip = np.array([.2, .2, .5])
    center = obj[:3] + np.array([-.01, .03, -.02])
    path, final = ablated.init_cost_fns()
    from planning.object_pose_cost import extended_cost_config
    assert path.size1_in(2) == final.size1_in(1) == 19
    packed = np.r_[obj, extended_cost_config(ablated.object_cost_config_)]
    state = np.r_[obj, tip]
    assert float(path(state, np.zeros(3), packed)) == pytest.approx(20*.05*np.sum((tip-center)**2))
    assert float(final(state, packed)) == pytest.approx(0.)


def selector():
    opt = SimpleNamespace(sample_point=np.array([[.04, 0., 0.], [.2, 0., 0.]]),
        normal=np.array([[-1., 0., 0.], [-1., 0., 0.]]),
        last_global_idx=1, last_executed_idx=0, update_Jacobian=Mock(),
        get_availble_point_idx=Mock(return_value=np.array([0, 1])),
        filter_rankable_indices=lambda ids: ids,
        choose_contact_points=Mock(return_value=(np.array([.2, 0., 0.]),
            np.array([-1., 0., 0.]), 0., 1., 1)),
        choose_nearby_topk_idx=Mock(return_value=1),
        last_candidate_ids=np.array([0, 1]), last_candidate_costs=np.array([1., 0.]),
        last_candidate_x_plus=np.tile([0., 0., 0., 1., 0., 0., 0.], (2, 1)),
        last_candidate_forces=np.array([[1., 0., 0.], [2., 0., 0.]]))
    opt.resolve_executed_contact = Mock(return_value=(np.array([.04, 0., 0.]),
        np.array([1., 0., 0.]), None, .5, dict(control_input=np.zeros(3))))
    return opt


def run_policy(mode, opt):
    param = SimpleNamespace(ablation_mode=mode, lambda_optimizer=opt,
        target_p_=np.zeros(3), target_q_=np.array([1., 0., 0., 0.]))
    args = SimpleNamespace(ground_height_threshold=.012, ideal_contact_surface_margin=-.0005)
    curr = np.r_[np.zeros(3), [1., 0., 0., 0.], [.04, .01, 0.]]
    forbidden = Mock(side_effect=AssertionError('verification/via must not change direct target'))
    return policy.compute_ablation_contact_policy(param, args, curr, np.eye(3),
        np.zeros(6), np.zeros((40, 9)), .01, forbidden, forbidden, forbidden, False, None)


def test_a1_executes_best_and_never_asks_for_p_arm():
    opt = selector()
    opt.resolve_executed_contact.side_effect = AssertionError('p_arm must be ignored')
    result = run_policy('a1', opt)
    np.testing.assert_allclose(result['mpc_contact_point'], [.2095, 0., 0.])
    np.testing.assert_array_equal(result['mpc_contact_point'], result['best_contact_point_world'])
    assert opt.last_executed_idx == 1 and result['verify_cost'] == 1.


def test_a2_never_consults_best_and_cannot_replace_p_arm():
    opt = selector()
    opt.choose_contact_points.side_effect = AssertionError('best-contact ranking must be ignored')
    opt.choose_nearby_topk_idx.side_effect = AssertionError('best-contact guide must be ignored')
    first = run_policy('a2', opt)
    opt.sample_point[1] = [100., -10., 20.]
    opt.last_candidate_costs[:] = [-100., 1e9]
    second = run_policy('a2', opt)
    np.testing.assert_allclose(first['mpc_contact_point'], [.0495, 0., 0.])
    np.testing.assert_array_equal(first['mpc_contact_point'], second['mpc_contact_point'])
    assert first['value_info']['target_source'] == 'p_arm_world'


def test_a3_offline_obb_preserves_actor_frame_and_reuses_cache(tmp_path, monkeypatch):
    import trimesh
    from scipy.spatial.transform import Rotation
    from . import obb as obb_module
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [.3, -.2, .7]).as_matrix()
    transform[:3, 3] = [.03, -.02, .015]
    source = trimesh.creation.box(extents=[.03, .06, .1], transform=transform)
    path = tmp_path/'offset_box.obj'
    source.export(path)
    fitted = obb_module.cached_mesh_obb(path, tmp_path/'cache')
    np.testing.assert_allclose(sorted(fitted.extents), [.03, .06, .1], atol=1e-8)
    local = (source.vertices-fitted.box_to_actor[:3, 3]) @ fitted.box_to_actor[:3, :3]
    assert np.all(np.abs(local) <= .5*fitted.extents+1e-8)
    assert not np.allclose(fitted.box_to_actor[:3, 3], 0)
    # Both current and target OBBs are transported by the actor pose, without
    # replacing the actor's origin with the offset box center.
    for angles, position in (([.1, .4, -.2], [.4, .1, .5]), ([.6, -.3, .8], [.6, -.1, .4])):
        rotation = Rotation.from_euler('xyz', angles)
        q = rotation.as_quat()[[3, 0, 1, 2]]
        expected = fitted.vertices @ rotation.as_matrix().T+position
        np.testing.assert_allclose(fitted.world_vertices(np.r_[position, q]), expected, atol=1e-12)
    before = fitted.mesh_path.stat().st_mtime_ns
    monkeypatch.setattr(obb_module, '_fit_obb', Mock(side_effect=AssertionError('Cached OBB must not be refitted')))
    reused = obb_module.cached_mesh_obb(path, tmp_path/'cache')
    assert reused.mesh_path == fitted.mesh_path and reused.mesh_path.stat().st_mtime_ns == before


def test_a3_obb_cache_invalidates_when_mesh_changes(tmp_path):
    import trimesh
    from .obb import cached_mesh_obb
    path = tmp_path/'object.obj'
    trimesh.creation.box(extents=[.02, .03, .05]).export(path)
    first = cached_mesh_obb(path, tmp_path/'cache')
    trimesh.creation.box(extents=[.04, .06, .1]).export(path)
    second = cached_mesh_obb(path, tmp_path/'cache')
    assert first.mesh_path != second.mesh_path
    np.testing.assert_allclose(sorted(second.extents), 2*np.sort(first.extents))


def test_a3_lambda_samples_obb_faces_without_modifying_mpc_or_scene(parser, tmp_path, monkeypatch):
    from planning.mlqp_point import LambdaContactControlOptimizer
    args = configure_ablation(parser.parse_args(['--a3', '--obj', 'elephant',
        '--a3-obb-cache-dir', str(tmp_path)]))
    # Initialize the real Lambda geometry with its forward/Torch backend,
    # avoiding compiled solvers for this geometry isolation check.
    args.solver_backend, args.sample_num = 'torch-lbfgs', 24
    param = ExplicitMPCParams(args, build_optimizer=False)
    physical = {name: getattr(param, name) for name in ('mesh_path_', 'collision_mesh_paths_',
        'object_vertices_', 'Q', 'obj_inertia_', 'sim_obj_inertia_diag_', 'target_p_', 'target_q_')}
    forbidden = Mock(side_effect=AssertionError('Geometry checks must not build an NLP'))
    monkeypatch.setattr(LambdaContactControlOptimizer, '_build_acados_contact_solver', forbidden)
    monkeypatch.setattr(LambdaContactControlOptimizer, '_precompile_optimization_function', forbidden)
    opt = build_lambda_optimizer(param, args)
    assert opt.contact_geometry == 'obb' and opt.pp.surface_sampling
    assert opt.pp.mesh_path != param.mesh_path_ and not opt.pp.collision_parts
    assert len(opt.sample_point) >= 6
    obb = opt.obb_geometry
    points = (opt.sample_point-obb.box_to_actor[:3, 3]) @ obb.box_to_actor[:3, :3]
    assert np.all(np.abs(points) <= .5*obb.extents+1e-9)
    assert np.all(np.min(np.abs(np.abs(points)-.5*obb.extents), axis=1) < 1e-9)
    normals = opt.normal @ obb.box_to_actor[:3, :3]
    np.testing.assert_allclose(np.max(np.abs(normals), axis=1), 1., atol=1e-9)
    # Matching candidates have inward normals, never averaged corner normals.
    assert np.all(np.sum(points*normals, axis=1) < 0)
    for name, original in physical.items():
        np.testing.assert_equal(getattr(param, name), original, err_msg=name)
    baseline = params_for_cost('full')
    ablated = params_for_cost('a3')
    for verify in (0., .5, 1.):
        obj = np.r_[[.4, -.1, .5], [1., 0., 0., 0.]]
        np.testing.assert_allclose(cost(ablated, obj, [.6, .1, .4], [.45, -.1, .5], verify),
                                   cost(baseline, obj, [.6, .1, .4], [.45, -.1, .5], verify))


def test_a3_keeps_the_baseline_policy_and_measured_inputs(monkeypatch):
    unchanged = dict(mpc_contact_point=np.array([.1, .2, .3]))
    baseline = Mock(return_value=unchanged)
    monkeypatch.setattr(policy, '_baseline_policy', baseline)
    opt = selector()
    result = run_policy('a3', opt)
    assert result is unchanged
    assert baseline.call_args.args[0].lambda_optimizer is opt


def test_a3_mug_execution_path_keeps_real_collision_parts(parser, tmp_path, monkeypatch):
    import trimesh
    from . import planner
    from planning.compound_contact_path import CompoundApproachVia
    args = configure_ablation(parser.parse_args(['--a3', '--obj', 'mug',
        '--a3-obb-cache-dir', str(tmp_path)]))
    args.solver_backend, args.sample_num = 'torch-lbfgs', 12
    unchanged_mpc = Mock(return_value=SimpleNamespace(tool_pose_control=False))
    monkeypatch.setattr(planner, 'MPCExplicit', unchanged_mpc)
    _, param, _, trackers = planner._build_mpc_planner_runtime(dict(args=vars(args), trial_count=0))
    assert param.lambda_optimizer.contact_geometry == 'obb'
    assert not param.lambda_optimizer.pp.collision_parts
    path = trackers['approach_via']
    assert isinstance(path, CompoundApproachVia)
    assert len(path.parts) == len(param.collision_mesh_paths_) > 1
    for mesh, source in zip(path.parts, param.collision_mesh_paths_):
        np.testing.assert_allclose(mesh.vertices, trimesh.load_mesh(source).vertices)
    assert unchanged_mpc.call_args.args[0] is param


def test_a6_policy_does_not_access_lambda_or_verification():
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError('A6 must not consult Lambda: ' + name)
    with pytest.raises(ValueError, match='independent AdamW'):
        run_policy('a6', Forbidden())


@pytest.mark.parametrize('flag,value', [
    ('--a6-adamw-iterations', '0'), ('--a6-adamw-lr', 'nan'),
    ('--a6-adamw-weight-decay', '-1'), ('--a6-fd-epsilon', '0'),
])
def test_a6_rejects_invalid_optimizer_settings(flag, value):
    with pytest.raises(ValueError):
        configure_ablation(selection_parser().parse_args(['--a6', flag, value]))


def a6_runtime(parser, monkeypatch, extra=()):
    from .planner import _build_mpc_planner_runtime
    import planning.mlqp_point as lambda_module
    import planning.mpc_explicit as mpc_module
    forbidden = Mock(side_effect=AssertionError('A6 must not construct or solve an NLP'))
    monkeypatch.setattr(cs, 'nlpsol', forbidden)
    monkeypatch.setattr(mpc_module.MPCExplicit, '__init__', forbidden)
    monkeypatch.setattr(mpc_module.MPCExplicit, '_build_acados_solver', forbidden)
    monkeypatch.setattr(lambda_module.LambdaContactControlOptimizer, '_build_acados_contact_solver', forbidden)
    monkeypatch.setattr(lambda_module.LambdaContactControlOptimizer, '_solve_optimization', forbidden)
    from . import params as params_module
    monkeypatch.setattr(params_module, '_baseline_optimizer', forbidden)
    args = configure_ablation(parser.parse_args(['--a6', '--obj', 'elephant',
        '--a6-adamw-iterations', '8', '--mpc-horizon', '3', *extra]))
    args.sample_num = 6
    return _build_mpc_planner_runtime(dict(args=vars(args), trial_count=0))


def test_a6_actual_comfree_rollout_reduces_cost_without_an_nlp(parser, monkeypatch):
    args, param, controller, _ = a6_runtime(parser, monkeypatch)
    assert controller.model.contact_model == 'comfree'
    assert controller.planner_solver_ == 'adamw' and param.lambda_optimizer is None
    # Empty contacts make the fingertip obey its commanded increments exactly.
    state = np.r_[[.5, 0., .5], [1., 0., 0., 0.], [.7, .1, .55]]
    result = controller.plan_once(target_p=state[:3], target_q=state[3:7],
        curr_x=state, phi_vec=np.ones(40), jac_mat=np.zeros((40, 9)))
    assert result['cost_opt'][0] < result['initial_cost']
    assert result['gradient_method'] == 'central_difference'
    assert result['iterations'] == args.a6_adamw_iterations
    assert result['rollout_evaluations'] == 1 + args.a6_adamw_iterations*(1+2*3*3)
    assert np.dot(result['action'], state[:3]-state[7:10]) > 0
    assert np.isfinite(result['rollout_q']).all()
    assert not hasattr(controller, 'acados_solver_') and controller.adamw_solve_count == 1
    assert np.max(np.linalg.norm(result['u_traj'] @ param.robot_stiff_.T, axis=1)) <= controller.contact_force_limit+1e-8


def test_a6_request_runs_adamw_even_without_contacts_and_ignores_dwell(parser, monkeypatch):
    from .planner import handle_mpc_request
    args, param, controller, trackers = a6_runtime(parser, monkeypatch)
    state = np.r_[[.5, 0., .5], [1., 0., 0., 0.], [.7, .1, .55]]
    assert param.lambda_optimizer is None
    assert set(trackers) == {'object_cost_stages', 'sol_guess'}
    message = dict(policy_q=state, phi_vec=np.ones(40), jac_mat=np.zeros((40, 9)),
        jac_mat_env=np.zeros((40, 9)), table_ground=.012, if_contact=False,
        sim_time=0., executed_action=np.zeros(3), dwell={'invalid': 'must be ignored'})
    first = handle_mpc_request(args, param, controller, trackers, message)
    assert first['solver_backend'] == 'adamw' and first['physics_backend'] == 'isaac'
    assert first['mpc_solves'] == 1
    assert {'verify_cost', 'policy', 'best_contact_point_world', 'p_arm_world', 'lambda_solves'}.isdisjoint(first)
    assert first['robot_trajectory_world'].shape == (3, 3)
    message['sim_time'] = .02
    message['executed_action'] = first['action']
    second = handle_mpc_request(args, param, controller, trackers, message)
    assert second['mpc_solves'] == 2 and trackers['sol_guess'] is not None


def test_a6_isaac_command_executes_only_adamw_increment(parser, monkeypatch):
    from .adamw import trajectory_command
    from .isaac import apply_incoming
    action = np.array([.001, -.002, .0005])
    command = trajectory_command(dict(action=action, solve_time=.003, object_cost_stage='flip'), 5)
    assert command['kind'] == 'trajectory'
    assert {'press', 'via', 'policy', 'verify_cost', 'best_contact_point_world', 'p_arm_world'}.isdisjoint(command)
    state = np.r_[[.5, 0., .5], [1., 0., 0., 0.], [.6, .1, .55]]
    env = SimpleNamespace(get_policy_state=lambda: state, set_via_action=Mock())
    assert apply_incoming(env, command, 'via') == 'ok'
    positional, keywords = env.set_via_action.call_args
    np.testing.assert_allclose(positional[0], state[7:10]+action)
    np.testing.assert_array_equal(keywords['action'], action)
    assert 'press' not in keywords and 'path_blocked' not in keywords


def test_a6_rejects_mjwarp_before_loading_simulators(monkeypatch):
    import sys
    from examples.mpc.franka.experiments import test_mpc_ablation as entry
    execute = Mock()
    monkeypatch.setattr(entry.os, 'execv', execute)
    monkeypatch.setattr(sys, 'argv', ['test_mpc_ablation.py', '--a6', '--physics-backend', 'mjwarp'])
    with pytest.raises(SystemExit):
        entry.main()
    execute.assert_not_called()
    with pytest.raises(ValueError, match='Isaac Gym'):
        configure_ablation(SimpleNamespace(ablation_mode='a6', physics_backend='mjwarp'))


def test_a6_fd_differentiates_full_rollout_and_projection_respects_slew(parser, monkeypatch):
    _, param, controller, _ = a6_runtime(parser, monkeypatch)
    # A known coupled horizon cost detects transposed map packing and verifies
    # derivatives in every time/control coordinate without symbolic gradients.
    u = cs.MX.sym('u', 3, 3)
    target = np.arange(9).reshape(3, 3)*.001
    objective = cs.sumsqr(cs.cumsum(u, 1)-cs.DM(target.T))
    controller.batch_cost_fn = cs.Function('coupled_probe', [u], [objective]).map(18)
    controls = np.arange(9).reshape(3, 3)*.0001
    gradient = controller._numerical_gradient(controls, ())
    residual = np.cumsum(controls, axis=0)-target
    expected = 2*np.flip(np.cumsum(np.flip(residual, axis=0), axis=0), axis=0)
    np.testing.assert_allclose(gradient, expected, atol=1e-12)
    controller.contact_force_limit = .6
    controller.first_action_bounds = (np.array([.0002, -.0003, .0001]),
                                      np.array([.0007, .0003, .0008]))
    projected = controller._project(np.full((3, 3), .005), -.005, .005)
    assert np.all(projected[0] >= controller.first_action_bounds[0]-1e-12)
    assert np.all(projected[0] <= controller.first_action_bounds[1]+1e-12)
    assert np.all(np.linalg.norm(projected @ param.robot_stiff_.T, axis=1) <= .6+1e-10)


@pytest.mark.parametrize('extra', [(), ('--no-two-stage-object-cost',)])
def test_a6_contact_rollout_matches_executed_control_prediction(parser, monkeypatch, extra):
    _, param, controller, _ = a6_runtime(parser, monkeypatch, extra)
    from examples.mpc.franka.runtime.contact import _contact_jacobian_np, tangent_basis_from_normal
    state = np.r_[[.5, 0., .5], [1., 0., 0., 0.], [.56, 0., .5]]
    phi, jac = np.ones(40), np.zeros((40, 9))
    relative = np.zeros((3, 9))
    relative[:, :3], relative[:, 6:] = np.eye(3), -np.eye(3)
    jac[:4] = _contact_jacobian_np(*tangent_basis_from_normal([-1., 0., 0.]),
                                 relative, param.mu_fingertip_object_)
    phi[:4] = .0001
    result = controller.plan_once(target_p=state[:3]+[-.01, 0., 0.], target_q=state[3:7],
        curr_x=state, phi_vec=phi, jac_mat=jac)
    assert np.isfinite(result['cost_opt']).all()
    assert result['cost_opt'][0] <= result['initial_cost']
    assert controller.path_cost_fn.size1_in(2) == 19
    predicted = state.copy()
    for command, expected in zip(result['u_traj'], result['rollout_q']):
        predicted = np.asarray(controller.stage_prediction_fn(predicted, command, phi, jac,
            controller.model.runtime_parameters(param.model_params), state)).reshape(-1)
        np.testing.assert_allclose(predicted, expected, atol=1e-12)


def test_a8_keeps_exact_second_stage_cost_and_never_falls_back(parser):
    args = configure_ablation(parser.parse_args(['--a8']))
    stages = make_object_cost_stages(args, [0., 0., 1.])
    baseline = ObjectCostStages(args)
    baseline.stage = 'pose'
    np.testing.assert_array_equal(stages.config(), baseline.config())
    for t, quat in ((0., [1.,0.,0.,0.]), (1., [0.,1.,0.,0.]),
                    (2., [0.,1.,0.,0.]), (3., [1.,0.,0.,0.]), (4., [1.,0.,0.,0.])):
        assert not stages.update(dict(sim_time=t, policy_q=np.r_[np.zeros(3), quat],
            object_linear_velocity=np.zeros(3), object_angular_velocity=np.zeros(3)))
        assert stages.enabled and stages.stage == 'pose'
        np.testing.assert_array_equal(stages.config(), baseline.config())
    assert stages.switches == stages.fallbacks == 0
    assert stages.first_flip_time is not None


def test_a5_factory_and_cartesian_adaptation_keep_cma_separate_from_mpc(parser):
    from .mpc import adapt_param_for_cartesian_solver
    args = configure_ablation(parser.parse_args(['--a5', '--solver-backend', 'ipopt',
        '--a5-population-size', '6', '--a5-population-iterations', '2']))
    args.sample_num = 6
    param = ExplicitMPCParams(args, build_optimizer=False)
    opt = build_lambda_optimizer(param, args)
    param.lambda_optimizer = opt
    adapt_param_for_cartesian_solver(param, args)
    assert opt.solver == 'cma-es' and opt.population_search.backend == 'cma-es'
    assert opt.population_search.size == 6 and opt.population_search.iterations == 2
    assert param.planner_solver_ == 'ipopt' and args.solver_backend == 'ipopt'


def test_a4_native_qp_uses_observed_state_without_moving_the_execution_state(tmp_path, monkeypatch):
    import mujoco
    import trimesh
    from .cso import MujocoQPCSO
    from planning.mlqp_point import LambdaContactControlOptimizer
    from planning.object_pose_cost import extended_cost_config
    mesh = tmp_path/'box.stl'
    trimesh.creation.box(extents=[.1, .1, .1]).export(mesh)
    opt = SimpleNamespace(h=.02, friction_reg_coef=.03, force_reg_coef=.01,
        max_contact_force=1.2, mu_arm_obj=.8, friction_cone='pyramid',
        _friction_force_cap_factor=1., max_normal_force=1.2/np.sqrt(1+.8**2),
        object_cost_config=lambda: extended_cost_config(cost_config(500., 20.)))
    from types import MethodType
    opt._project_contact_force = MethodType(LambdaContactControlOptimizer._project_contact_force, opt)
    param = SimpleNamespace(lambda_optimizer=opt, gravity_=np.array([0.,0.,-9.81,0.,0.,0.]),
        mu_table_object_=.5, sim_obj_mass_=.1, sim_obj_inertia_diag_=np.full(3, .000167),
        table_height=.35, collision_mesh_paths_=[], mesh_path_=str(mesh))
    predictor = MujocoQPCSO(param)
    state = np.r_[[.6, 0., .6], [1.,0.,0.,0.], [.5,0.,.5]]
    predictor.prepare_observation(dict(policy_q=state, object_linear_velocity=np.array([.01,0,0]),
        object_angular_velocity=np.array([0.,0.,.2]), sim_time=1.))
    qpos = predictor.env.data_.qpos.copy()
    qvel = predictor.env.data_.qvel.copy()
    calls = []
    original = mujoco.mj_forwardSkip
    def native(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(mujoco, 'mj_forwardSkip', native)
    a = predictor.predict([.1,0.,0.], np.zeros(3), np.eye(3))
    b = predictor.predict([.1,0.,0.], np.zeros(3), np.eye(3))
    assert len(calls) == 2
    np.testing.assert_array_equal(predictor.env.data_.qpos, qpos)
    np.testing.assert_array_equal(predictor.env.data_.qvel, qvel)
    np.testing.assert_allclose(a['x_plus'], b['x_plus'], atol=1e-12)
    assert a['x_plus'][0] == pytest.approx(.01*.02 + .1/.1*.02**2)
    assert predictor.model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    # Refresh at table contact, then run the same independent force solver.
    state[2] = .40
    predictor.prepare_observation(dict(policy_q=state))
    assert predictor.env.data_.ncon > 0
    result = predictor.solve_candidate(p_arm=np.zeros(3), n_arm=[1.,0.,0.],
        t1=[0.,1.,0.], t2=[0.,0.,1.], x_d=[.005,0.,0.,1.,0.,0.,0.])
    assert np.isfinite(result['cost'])
    force = result['lam_arm_opt']
    assert np.linalg.norm(force) <= opt.max_contact_force+1e-10
    assert np.abs(force[1:]).sum() <= opt.mu_arm_obj*force[0]+1e-10
    assert predictor.diagnostics()['fallback_to_scm'] == 0


def test_a5_compiled_objective_preserves_scm_values_and_calls_cma():
    from planning.contact_force_sampling import ForcePopulationSearch
    from .cso import CMASoftSCMCSO
    force, theta, cap = cs.SX.sym('f', 3), cs.SX.sym('theta', 6), cs.SX.sym('cap')
    fn = cs.Function('test_a5_objective', [force, theta, cap],
        [cs.vertcat(force,1.,0.,0.,0.), cs.sumsqr(force-cs.DM([.6,.1,-.1]))],
        ['lam_arm','scm_theta','force_limit'], ['x_plus_opt','cost'])
    opt = SimpleNamespace(force_objective_fn=fn, solver='cma-es', contact_model='soft-scm',
        base_mu_arm_obj=1., friction_cone='pyramid', _friction_force_cap_factor=1.,
        population_search=ForcePopulationSearch('cma-es', size=12, iterations=6, seed=3))
    opt._solve_optimization = lambda **kwargs: opt.population_search.solve(opt, kwargs)
    backend = CMASoftSCMCSO(opt)
    kwargs = dict(scm_theta=np.zeros(6), force_limit=3.)
    before = fn(lam_arm=[.3,.1,0.], **kwargs)
    after = opt.force_objective_fn(lam_arm=[.3,.1,0.], **kwargs)
    for key in before:
        np.testing.assert_allclose(np.asarray(before[key]), np.asarray(after[key]), atol=1e-12)
    result = backend.solve_candidate(**kwargs)
    assert result['cost'] < float(fn(lam_arm=[0.,0.,0.], **kwargs)['cost'])
    assert backend.diagnostics()['solves'] == 1
    assert backend.diagnostics()['evaluations'] > 12
    assert backend.diagnostics()['failures'] == 0



@pytest.mark.parametrize('flags,script,head', [
    (['--a1', '--headless', '--sync-planner'], 'rollout.py', []),
    (['--a2', '--viewer', '--async-planner'], 'isaac.py', ['--planner', 'mpc']),
    (['--a4', '--physics-backend', 'mjwarp'], 'rollout.py', []),
])
def test_entry_dispatch_preserves_all_flags(monkeypatch, flags, script, head):
    import sys
    from examples.mpc.franka.experiments import test_mpc_ablation as entry
    captured = Mock()
    monkeypatch.setattr(sys, 'argv', ['test_mpc_ablation.py', *flags])
    monkeypatch.setattr(entry.os, 'execv', captured)
    entry.main()
    argv = captured.call_args[0][1]
    assert argv[2].endswith('/experiments/mpc_ablation/' + script)
    assert argv[3:] == head + flags


def test_a8_cannot_change_dynamics_when_two_stage_flag_disabled(parser):
    args = configure_ablation(parser.parse_args(['--a8', '--no-two-stage-object-cost']))
    assert args.two_stage_object_cost
    param = ExplicitMPCParams(args, build_optimizer=False)
    assert param.runtime_object_cost_
    assert make_object_cost_stages(args, [0., 0., 1.]).stage == 'pose'
