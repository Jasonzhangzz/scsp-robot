"""Counterfactual intervention drives the shared, real CPO contact policy."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('open3d')
from examples.mpc.fingertips.test.test_0902 import (
    compute_rollout_contact_via, ContactValueTracker, ModelCostConfidence,
    SmoothedApproachVia,
)


def policy(forced=None, *, prefer_near=True):
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
    contexts = []
    def hook(context):
        contexts.append(context)
        return forced
    param = SimpleNamespace(lambda_optimizer=opt, target_p_=np.array([.4,.1,.4]),
                            target_q_=np.array([1.,0,0,0]), rs_decision_hook=hook)
    args = SimpleNamespace(ground_height_threshold=.012, ideal_contact_surface_margin=-.0005)
    tip = param.target_p_ + points[0] - .0095*normals[0]
    tracker = ContactValueTracker()
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


def test_force_stay_keeps_near_geometry_and_prediction():
    result, contexts, tracker = policy('stay', prefer_near=False)
    assert contexts[0]['decision'] == 'switch'
    assert result['best_contact_idx'] == 0
    np.testing.assert_allclose(result['p_arm_surface_world'], [.45,.1,.42])
    np.testing.assert_allclose(result['info']['control_input'], [.1,0,0])
    assert result['value_info']['accept_p_arm']
    assert not result['value_info']['switch_observed_contact']
    assert tracker._holding_p_arm
