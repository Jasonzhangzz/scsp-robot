"""Counterfactual intervention drives the shared, real CPO contact policy."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('open3d')
from examples.mpc.fingertips.test.test_0902 import (
    compute_rollout_contact_via, ContactValueTracker, ModelCostConfidence,
    SmoothedApproachVia,
)


@pytest.mark.parametrize('symbol', ['SX', 'MX'])
def test_contact_jacobian_builds_for_both_solver_graph_types(symbol):
    import casadi as cs
    from planning.mlqp_point import LambdaContactControlOptimizer
    point = getattr(cs, symbol).sym('point', 3)
    jacobian = LambdaContactControlOptimizer.compute_contact_jacobian(point)
    fn = cs.Function('point_jacobian', [point], [jacobian])
    p, velocity, omega = np.array([.1, -.2, .3]), np.array([1., 2., 3.]), np.array([.4, .5, .6])
    np.testing.assert_allclose(np.asarray(fn(p)) @ np.r_[velocity, omega], velocity+np.cross(omega, p))


def test_rs_acados_failure_never_runs_ipopt(monkeypatch):
    import argparse
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    from planning.mpc_explicit import _build_mpc_planner_runtime
    args = _add_rollout_policy_args(argparse.ArgumentParser()).parse_args(
        ['--obj', 'foam_brick', '--solver-backend', 'acados'])
    args.rs_require_acados = True
    _, param, mpc, _ = _build_mpc_planner_runtime(dict(args=vars(args), trial_count=0))
    opt = param.lambda_optimizer
    assert opt.acados_max_fallbacks_per_cycle == 0
    assert mpc.param_.rs_require_acados_ and not mpc.param_.mpc_ipopt_fallback_
    monkeypatch.setattr(opt, '_solve_optimization_acados',
                        lambda **kw: (_ for _ in ()).throw(RuntimeError('test QP failure')))
    monkeypatch.setattr(opt, '_ipopt_optimization', lambda **kw: pytest.fail('contact IPOPT fallback'))
    assert opt._solve_optimization() is None
    assert opt.acados_fallback_count == 0
    monkeypatch.setattr(mpc, '_plan_once_acados', lambda *a, **kw: None)
    monkeypatch.setattr(mpc, '_plan_once_ipopt', lambda *a, **kw: pytest.fail('MPC IPOPT fallback'))
    mpc.last_acados_error = 'test QP failure'
    q = np.r_[param.target_p_, param.target_q_, param.target_p_+[.1,0,0]]
    with pytest.raises(RuntimeError, match='RS acados MPC solve failed'):
        mpc.plan_once(target_p=param.target_p_, target_q=param.target_q_, curr_x=q,
            phi_vec=np.ones(param.max_ncon_*4),
            jac_mat=np.zeros((param.max_ncon_*4, param.n_qvel_)),
            virtual_point=q[7:10], contact_point=q[7:10])


def policy(forced=None, *, prefer_near=True, capture=None):
    points = np.array([[.05, 0., .02], [-.05, 0., .02]])
    normals = np.array([[-1., 0., 0.], [1., 0., 0.]])
    predictions = [np.array([.01, 0, 0, 1, 0, 0, 0]), np.array([.02, 0, 0, 1, 0, 0, 0])]
    forces = [np.array([.1, 0, 0]), np.array([.2, 0, 0])]
    opt = SimpleNamespace(
        sample_point=points, normal=normals, sample_geodesic=np.array([[0.,.1],[.1,0.]]),
        last_global_idx=1, last_selected_idx=None, last_executed_idx=None,
        last_best_x_plus=predictions[1], last_best_force=forces[1], last_best_cost=0.,
        last_global_total_cost=0., last_candidate_ids=np.array([0,1]),
        last_candidate_costs=np.array([.01,0.]), last_candidate_x_plus=predictions,
        last_candidate_forces=forces, contact_switch_radius=.03,
        contact_switch_confidence=1., mu_arm_obj=.5,
        update_Jacobian=lambda *a, **kw: None,
        get_availble_point_idx=lambda *a, **kw: np.array([0,1]),
        filter_rankable_indices=lambda ids: ids,
        choose_contact_points=lambda *a, **kw: (points[1], normals[1], 0., .01, 1),
        choose_nearby_topk_idx=lambda _: 0 if prefer_near else 1,
        pose_delta_for_sample=lambda idx: (.099 if prefer_near else 0.) if idx==0 else .1)
    def resolve(*a, **kw):
        opt.last_executed_idx = 0
        return points[0], -normals[0], predictions[0], .01, {
            'control_input': forces[0], 'solver_failed': False}
    opt.resolve_executed_contact = resolve
    def optimize(*a, sample_idx, **kw):
        return points[sample_idx], -normals[sample_idx], predictions[sample_idx], .01, {
            'control_input': forces[sample_idx], 'solver_failed': False}
    opt.optimize_control_input = optimize
    opt.normalize_pose_cost = lambda cost: cost
    opt._pose_cost = lambda *a: 0.
    contexts = []
    def hook(context):
        contexts.append(context)
        return forced
    param = SimpleNamespace(lambda_optimizer=opt, target_p_=np.array([.4,.1,.4]),
                            target_q_=np.array([1.,0,0,0]), rs_decision_hook=hook)
    args = SimpleNamespace(ground_height_threshold=.012, ideal_contact_surface_margin=-.0005)
    tip = param.target_p_ + points[0] - .0095*normals[0]
    tracker = ContactValueTracker()
    if capture is not None:
        capture.update(param=param, args=args, tracker=tracker,
                       confidence=ModelCostConfidence(), via=SmoothedApproachVia(),
                       curr_q=np.r_[param.target_p_, param.target_q_, tip])
    result = compute_rollout_contact_via(param, args, np.r_[param.target_p_, param.target_q_, tip],
        np.eye(3), np.zeros(6), np.zeros((40,6)), .01, tracker,
        ModelCostConfidence(), SmoothedApproachVia(), False, None, floor_ground=.362, floor_z=.35)
    return result, contexts, tracker


def test_no_override_preserves_normal_near_selection():
    result, contexts, _ = policy()
    assert contexts[0]['eligible'] and contexts[0]['decision'] == 'stay'
    assert result['best_contact_idx'] == 0
    assert 'rs_forced_decision' not in result['value_info']


def test_force_switch_uses_optimal_cache_and_routes_cpo_to_new_target():
    result, contexts, tracker = policy('switch')
    assert contexts[0]['decision'] == 'stay'
    assert result['best_contact_idx'] == 1
    np.testing.assert_allclose(result['p_arm_surface_world'], [.35,.1,.42])
    np.testing.assert_allclose(result['info']['control_input'], [.2,0,0])
    assert result['value_info']['switch_observed_contact']
    assert not result['value_info']['accept_p_arm']
    assert not tracker._holding_p_arm
    assert tracker.verify == result['value_info']['verify'] == 1.


def test_force_stay_keeps_near_geometry_and_prediction():
    result, contexts, tracker = policy('stay', prefer_near=False)
    assert contexts[0]['decision'] == 'switch'
    assert result['best_contact_idx'] == 0
    np.testing.assert_allclose(result['p_arm_surface_world'], [.45,.1,.42])
    np.testing.assert_allclose(result['info']['control_input'], [.1,0,0])
    assert result['value_info']['rs_commitment_active']
    assert result['value_info']['accept_p_arm']
    assert tracker.verify == result['value_info']['verify'] == 0.


def test_commitment_survives_new_rankings_and_wrong_contact_then_resumes_verify():
    state = {}
    result, contexts, tracker = policy('switch', capture=state)
    param, q = state['param'], state['curr_q']
    committed = param.rs_contact_commitment_
    frozen = tracker.verify
    frozen_window = list(tracker._target_window)
    # A later CSO ranking chooses the other point and the object translates.
    param.lambda_optimizer.choose_nearby_topk_idx = lambda _: 0
    q[:3] += [.02, .01, 0]
    def tick():
        return compute_rollout_contact_via(param, state['args'], q, np.eye(3),
            np.zeros(6), np.zeros((40,6)), .01, tracker, state['confidence'],
            state['via'], False, None, floor_ground=.362, floor_z=.35)
    second = tick()
    np.testing.assert_allclose(second['p_arm_surface_world'], q[:3]+committed.local_point)
    assert tracker.verify == frozen and len(contexts) == 1
    tracker.verify = .9  # emulate a stage reset or unrelated tracker mutation
    tracker._target_window.append(.9)
    param.lambda_actual_contact_ = dict(point_world=q[:3]+[.05, 0, .02],
                                      normal_world=[1., 0, 0], dist=-.001)
    tick()
    assert param.rs_contact_commitment_ is committed
    assert tracker.verify == frozen and list(tracker._target_window) == frozen_window
    # Proximity without actual contact does not release the commitment.
    param.lambda_actual_contact_.update(point_world=q[:3]+committed.local_point, dist=.001)
    tick()
    assert param.rs_contact_commitment_ is committed
    param.rs_decision_hook = lambda context: None
    param.lambda_actual_contact_['dist'] = -.001
    tick()
    assert param.rs_contact_commitment_ is None and committed.record['reached']
    assert len(tracker._target_window) == len(frozen_window)+1
