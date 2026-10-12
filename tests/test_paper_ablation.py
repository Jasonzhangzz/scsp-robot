"""Behavioral checks for the isolated paper ablations, without Isaac/GPU."""
import itertools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import casadi as cs
import numpy as np
import pytest

from examples.mpc.franka.ik2 import contact_policy_ablation as policy_module
from examples.mpc.franka.ik2.params_ablation import ExplicitMPCParams
from examples.mpc.franka.ik2.run_ablation import (
    experiment_argv, sensitivity_grid, summarize_grid_cell,
)
from planning.mpc_costs import pack_cost_params
from planning.object_pose_cost import cost_config


def selector(costs=(0.0, 1.0, 2.0)):
    opt = SimpleNamespace(
        sample_point=np.array([[0.0, 0., 0.], [0.1, 0., 0.], [0.2, 0., 0.]]),
        normal=np.tile([0., 0., -1.], (3, 1)),
        last_executed_idx=0, last_selected_idx=0,
        lock_contact_patch=True, _blocked_contact_indices={2: 99},
        m=0.01, last_candidate_ids=np.array([0, 1, 2]),
        last_candidate_costs=np.array(costs),
        update_Jacobian=Mock(),
        get_availble_point_idx=Mock(return_value=np.array([0, 1, 2])),
        filter_rankable_indices=lambda ids: ids,
        choose_contact_points=Mock(return_value=('discard',)),
    )
    return opt


def direct_policy(mode, opt):
    param = SimpleNamespace(ablation_mode=mode, lambda_optimizer=opt,
                            target_p_=np.zeros(3), target_q_=np.array([1., 0., 0., 0.]))
    args = SimpleNamespace(ground_height_threshold=.012, ideal_contact_surface_margin=-.0005)
    state = np.r_[np.zeros(3), [1., 0., 0., 0.], [.20, .01, .0095]]
    # These objects raise if either ranking verification or via is consulted.
    forbidden = Mock(side_effect=AssertionError('RS/via must be bypassed'))
    result = policy_module.compute_ablation_contact_policy(param, args, state, np.eye(3),
        np.zeros(6), np.zeros((40, 9)), .01, forbidden, forbidden, forbidden, True, 0)
    return param, args, state, result


def test_a1_nearest_is_independent_of_all_lambda_predictions():
    opt = selector()
    def prediction_a(*_a, **_kw):
        opt.last_selected_idx = 0
        opt.last_candidate_costs = np.array([-1e9, 2e9, np.inf])
        opt.last_best_force = np.array([1e8, 1e8, 1e8])
    opt.choose_contact_points.side_effect = prediction_a
    _, _, _, first = direct_policy('a1', opt)
    def prediction_b(*_a, **_kw):
        opt.last_selected_idx = 1
        opt.last_candidate_costs = None
        opt.last_best_force = None
    opt.choose_contact_points.side_effect = prediction_b
    _, _, _, second = direct_policy('a1', opt)
    np.testing.assert_allclose(first['mpc_contact_point'], [.2, 0., .0095])
    np.testing.assert_array_equal(first['mpc_contact_point'], second['mpc_contact_point'])
    np.testing.assert_array_equal(first['info']['control_input'], np.zeros(3))
    assert first['x_plus_opt'] is None and first['verify_cost'] == 1.0
    assert opt.choose_contact_points.call_count == 2


def test_a2_follows_each_finite_global_best_without_incumbent_or_rs():
    opt = selector((.4, .1, np.inf))
    _, _, _, first = direct_policy('a2', opt)
    assert first['best_contact_idx'] == 1
    np.testing.assert_array_equal(first['mpc_contact_point'], first['best_contact_point_world'])
    opt.last_candidate_costs = np.array([np.inf, .4, .01])
    _, _, _, second = direct_policy('a2', opt)
    assert second['best_contact_idx'] == 2
    opt.last_candidate_costs[:] = np.inf
    _, _, _, failed = direct_policy('a2', opt)
    assert failed['best_contact_idx'] == 2  # documented nearest fallback


def cost_param(mode='a3', runtime=True):
    param = ExplicitMPCParams.__new__(ExplicitMPCParams)
    param.__dict__.update(
        n_qpos_=10, n_qvel_=9, n_cmd_=3, max_ncon_=10,
        smooth_contact_detour=False, quadratic_contact_track=True,
        attract_coef=20., reject_dis=.001, reject_coef=100.,
        contact_coef=20., contact_cost_param=1., spline_escape_cost=False,
        field_cost_weight=10., pos_coef=500., ori_coef=20.,
        runtime_object_cost_=runtime, cpo_variant='full', rollout_press_patch=True,
        ablation_mode=mode, ablation_collision_weight=10000.,
        ablation_collision_radius=.01, ablation_collision_tolerance=.0005,
        object_vertices_=np.array(list(itertools.product((-.05, .05), repeat=3))),
        collision_mesh_paths_=[], object_cost_config_=cost_config(500., 20.),
    )
    return param


def evaluate_stage(param, obj, ee, contact, verify=0., virtual=(10., 10., 10.)):
    path, terminal = param.init_cost_fns()
    packed = pack_cost_params('param', param, path,
        target_p=obj[:3], target_q=obj[3:], verify_cost_param=verify,
        virtual_point=virtual, contact_point=contact, contact_reference=obj)
    state = np.r_[obj, ee]
    return float(path(state, np.zeros(3), packed)), float(terminal(state, packed))


@pytest.mark.parametrize('runtime', [False, True])
def test_a3_tracks_direct_target_and_ignores_verify_virtual_rejection_and_center(runtime):
    param = cost_param(runtime=runtime)
    obj = np.r_[np.zeros(3), [1., 0., 0., 0.]]
    ee = np.array([.1, 0., 0.])
    near, _ = evaluate_stage(param, obj, ee, ee)
    assert evaluate_stage(param, obj, ee, ee, 1., (-20., 5., 4.))[0] == pytest.approx(near)
    assert evaluate_stage(param, obj, ee, [.3, 0., 0.])[0] > near
    # Move object with the direct reference/EE in free space: no COM attraction.
    shifted = obj.copy(); shifted[1] += .02
    assert evaluate_stage(param, shifted, ee, ee)[0] == pytest.approx(near)
    param.ablation_mode = 'full'
    assert evaluate_stage(param, obj, ee, ee)[1] == pytest.approx(0.)


def test_full_keeps_baseline_symbolic_cost_and_a3_keeps_terminal_objective():
    from examples.mpc.franka.ik2.params import ExplicitMPCParams as BaselineParams
    full = cost_param('full')
    path, final = full.init_cost_fns()
    baseline_path, baseline_final = BaselineParams.init_cost_fns(full)
    assert path.serialize() == baseline_path.serialize()
    assert final.serialize() == baseline_final.serialize()
    direct = cost_param('a3')
    _, direct_final = direct.init_cost_fns()
    # A3 uses MX call boundaries for bounded code generation on CasADi 3.6.
    # The terminal values and derivatives still match the baseline objective.
    inputs = np.random.default_rng(1).normal(size=(3, final.size1_in(0)))
    packed = np.random.default_rng(2).normal(size=final.size1_in(1))
    for state in inputs:
        assert float(direct_final(state, packed)) == pytest.approx(float(final(state, packed)))


def test_a3_collision_cost_moves_and_rotates_with_object_and_allows_surface_contact():
    param = cost_param()
    q = cs.SX.sym('q', 10)
    fn = cs.Function('collision', [q], [param.object_collision_cost_cs(q)])
    obj = np.r_[np.zeros(3), [1., 0., 0., 0.]]
    legal = np.r_[obj, [.0595, 0., 0.]]
    outside = np.r_[obj, [.1, 0., 0.]]
    inside = np.r_[obj, [.03, 0., 0.]]
    assert float(fn(outside)) == pytest.approx(0.)
    assert float(fn(legal)) == pytest.approx(0., abs=1e-12)
    assert float(fn(inside)) > 0
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_euler('z', 40., degrees=True)
    translated_obj = np.r_[[1., 2., 3.], rot.as_quat()[[3, 0, 1, 2]]]
    moved = np.r_[translated_obj, rot.apply(inside[7:]) + translated_obj[:3]]
    assert float(fn(moved)) == pytest.approx(float(fn(inside)), abs=1e-10)


def test_a3_collision_uses_union_of_parts_so_a_cavity_remains_open(tmp_path):
    import trimesh
    param = cost_param()
    paths = []
    for offset in (-.08, .08):
        mesh = trimesh.creation.box(extents=(.04, .1, .1))
        mesh.apply_translation([offset, 0., 0.])
        path = tmp_path / ('part'+str(offset)+'.stl'); mesh.export(path)
        paths.append(str(path))
    param.collision_mesh_paths_ = paths
    q = cs.SX.sym('q', 10)
    fn = cs.Function('union_collision', [q], [param.object_collision_cost_cs(q)])
    obj = np.r_[np.zeros(3), [1., 0., 0., 0.]]
    assert float(fn(np.r_[obj, [0., 0., 0.]])) == pytest.approx(0.)
    assert float(fn(np.r_[obj, [.08, 0., 0.]])) > 0.


@pytest.mark.parametrize('mode', ['a1', 'a2', 'a3'])
def test_direct_modes_reach_mpc_without_target_clamp_via_or_lambda_seed(monkeypatch, mode):
    from planning.mpc_explicit_ablation import handle_mpc_request
    helper_module = ModuleType('planning.MPPIWarp')
    helper_module._apply_dwell_payload = Mock(side_effect=AssertionError('No Lambda feedback'))
    helper_module._opt_snapshot = lambda _opt: {}
    helper_module._pickle_safe = lambda policy: policy
    helper_module.clamp_via_to_tip = Mock(side_effect=AssertionError('Do not clamp direct target'))
    monkeypatch.setitem(sys.modules, 'planning.MPPIWarp', helper_module)
    opt = selector()
    param, args, state, policy = direct_policy('a1', opt)
    param.ablation_mode = mode
    param.__dict__.update(max_ncon_=10, n_qvel_=9, table_height=0.,
        lambda_obj_mass_=.01, gravity_=np.array([0., 0., -9.81, 0., 0., 0.]),
        runtime_object_cost_=False, mpc_u_lb_=-.005, mpc_u_ub_=.005,
        cpo_direct_trajectory=True)
    args.via_max_lead = .005
    monkeypatch.setattr(policy_module, 'compute_ablation_contact_policy', lambda *_a, **_kw: policy)
    conf = SimpleNamespace(tightness=lambda: 1., accum=0.)
    trackers = dict(value_tracker=None, model_cost_conf=conf, approach_via=None,
        arrived_hold=False, arrived_dest_idx=None, sol_guess=None, last_verify_cost=None)
    mpc = SimpleNamespace(velocity_state=False,
        plan_once=Mock(return_value=dict(action=np.zeros(3), sol_guess=None, cost_opt=0.)))
    msg = dict(policy_q=state, phi_vec=np.ones(40), jac_mat=np.zeros((40, 9)),
        jac_mat_env=np.zeros((40, 9)), table_ground=.012, sim_time=0.,
        if_contact=False, dwell=None)
    result = handle_mpc_request(args, param, mpc, trackers, msg)
    assert not result['mpc_skipped']
    assert mpc.plan_once.call_args.kwargs['verify_cost_param'] == 1.
    np.testing.assert_array_equal(mpc.plan_once.call_args.kwargs['contact_point'], policy['p_arm_world'])
    assert np.linalg.norm(policy['p_arm_world'] - state[7:10]) > args.via_max_lead
    helper_module.clamp_via_to_tip.assert_not_called()
    helper_module._apply_dwell_payload.assert_not_called()


def test_grid_is_cartesian_and_commands_preserve_trial_seeds():
    grid = sensitivity_grid(['verify_beta=.09,.18', 'contact_dwell_steps=2,4,8'])
    assert len(grid) == 6
    assert len(sensitivity_grid()) == 27
    flags = experiment_argv('a1', 'rotation', ['--trial-start', '10', '--trial_num', '2'])
    assert flags[-8:] == ['--cso-guidance', 'cso', '--cpo-variant', 'full',
                          '--ablation-mode', 'a1', '--ablation-variant', 'a1']
    assert flags[flags.index('--trial-start')+1] == '10'
    assert '--no_random_init_tilt' in flags
    with pytest.raises(ValueError):
        sensitivity_grid(['value_rho=.1,.2'])
    with pytest.raises(ValueError):
        sensitivity_grid(['verify_beta=nan'])


def test_sensitivity_aggregation_includes_failures_in_error_and_success_metrics():
    rows = [dict(trial=10, success=1, completion_time_s=2., completion_steps=100,
                 final_pos_err=.01, solve_dt_mean_ms=5.),
            dict(trial=11, success=0, completion_time_s=None, completion_steps=None,
                 final_pos_err=.1, solve_dt_mean_ms=15.)]
    summary = summarize_grid_cell(rows)
    assert summary['success_rate'] == .5
    assert summary['completion_time_mean_s'] == 2.
    assert summary['final_pos_err_mean'] == pytest.approx(.055)
    assert summary['solve_dt_mean_ms'] == 10.
    with pytest.raises(ValueError):
        summarize_grid_cell(rows+[rows[0]])


def test_sensitivity_saves_separate_results_with_matched_seed_ranges(tmp_path, monkeypatch):
    import json
    from examples.mpc.franka.ik2 import run_ablation as runner
    def simulate(command, stdout, stderr, check):
        destination = Path(command[command.index('--ranking-metrics-jsonl')+1])
        assert command[command.index('--trial-start')+1] == '10'
        destination.write_text(json.dumps(dict(trial=10, success=0, steps=2,
            final_pos_err=.1, solve_dt_mean_ms=5.))+'\n')
    monkeypatch.setattr(runner.subprocess, 'run', simulate)
    args = SimpleNamespace(sensitivity_axis=['verify_beta=.09,.18'],
                           sensitivity_output=str(tmp_path/'grid'), task='flipping', dry_run=False)
    assert runner.run_sensitivity(args, ['--trial-start', '10', '--trial_num', '1']) == 0
    results = json.loads((tmp_path/'grid/sensitivity_summary.json').read_text())
    assert len(results) == 2
    assert all(row['success_rate'] == 0. for row in results)
    assert (tmp_path/'grid/cell_000/trials.jsonl').exists()
    assert (tmp_path/'grid/cell_001/trials.jsonl').exists()
    with pytest.raises(ValueError, match='empty'):
        runner.run_sensitivity(args, [])


def test_factored_collision_has_same_value_gradient_and_hessian_as_inline_expression():
    from models.contact_geometry import rotation_cs
    param = cost_param()
    state = cs.SX.sym('collision_state', 10)
    tip = rotation_cs(state[3:7]).T @ (state[7:10]-state[:3])
    radius = param.ablation_collision_radius-param.ablation_collision_tolerance
    inline = cs.mmax(cs.vertcat(*[
        cs.fmax(radius-cs.mmax(cs.DM(n) @ tip-cs.DM(b)),0)**2
        for n,b in param._collision_planes()]))
    factored = param.object_collision_cost_cs(state)
    functions = [cs.Function(name,[state],[expression,cs.gradient(expression,state),cs.hessian(expression,state)[0]])
                 for name,expression in (('inline_collision',inline),('factored_collision',factored))]
    for ee in ([.03,.01,.0],[.1,.0,.0]):
        point=np.r_[[.02,.03,.04],[.8,.2,.1,.3],ee]
        expected, actual = (fn(point) for fn in functions)
        for a,b in zip(expected,actual):
            np.testing.assert_allclose(np.asarray(a),np.asarray(b),atol=1e-10,rtol=1e-10)


def test_many_plane_max_preserves_derivatives_and_has_bounded_codegen(tmp_path):
    param = cost_param()
    normals = np.random.default_rng(1).normal(size=(344, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    tip = cs.MX.sym('local_tip', 3)
    distances = cs.DM(normals) @ tip
    native = cs.fmax(.05-cs.mmax(distances), 0)**2
    compact = cs.fmax(.05-param._collision_max_function(344)(distances), 0)**2
    functions = [cs.Function(name, [tip], [value, cs.gradient(value, tip), cs.hessian(value, tip)[0]])
                 for name, value in (('native_max_cost', native), ('compact_max_cost', compact))]
    for point in ([.01, .0, .0], [.1, .05, .03], [0., 0., 0.]):
        expected, actual = (fn(point) for fn in functions)
        for a, b in zip(expected, actual):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-10, rtol=1e-10)
    generator = cs.CodeGenerator('compact_collision.c')
    generator.add(functions[1])
    generator.generate(str(tmp_path) + '/')
    # The original derivative graph produced >20 MB and multi-GB compiler RSS.
    assert (tmp_path/'compact_collision.c').stat().st_size < 1024 * 1024
