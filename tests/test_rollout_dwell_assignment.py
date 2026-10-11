"""Rollout dwell must charge a stuck patch, not every sample the ball grazes."""

import numpy as np
from types import SimpleNamespace

from examples.mpc.fingertips.test.test_0902 import (
    ContactValueTracker,
    _rollout_dwell_assignment,
)
from planning.mlqp_point import LambdaContactControlOptimizer


def test_arrival_keeps_guided_patch_index_and_cached_prediction_together():
    from examples.mpc.fingertips.test.test_0902 import (
        compute_rollout_contact_via, ModelCostConfidence, SmoothedApproachVia,
    )
    points = np.array([[.05, 0., .02], [-.05, 0., .02]])
    normals = np.array([[-1., 0., 0.], [1., 0., 0.]])
    predictions = [np.array([.01, 0., 0., 1., 0., 0., 0.]),
                   np.array([.02, 0., 0., 1., 0., 0., 0.])]
    forces = [np.array([.1, 0., 0.]), np.array([.2, 0., 0.])]
    opt = SimpleNamespace(
        sample_point=points, normal=normals,
        sample_geodesic=np.array([[0., .1], [.1, 0.]]),
        last_global_idx=1, last_selected_idx=None, last_executed_idx=None,
        last_best_x_plus=predictions[1], last_best_force=forces[1], last_best_cost=0.,
        last_global_total_cost=0., last_candidate_ids=np.array([0, 1]),
        last_candidate_costs=np.array([.01, 0.]), last_candidate_x_plus=predictions,
        last_candidate_forces=forces, contact_switch_radius=.03,
        contact_switch_confidence=1., mu_arm_obj=.5,
        update_Jacobian=lambda *a, **kw: None,
        get_availble_point_idx=lambda *a, **kw: np.array([0, 1]),
        filter_rankable_indices=lambda ids: ids,
        choose_contact_points=lambda *a, **kw: (points[1], normals[1], 0., .01, 1),
        choose_nearby_topk_idx=lambda _: 0,
        pose_delta_for_sample=lambda idx: .099 if idx == 0 else .1,
    )
    def resolve(*a, **kw):
        opt.last_executed_idx = 0
        return points[0], -normals[0], predictions[0], .01, {
            'control_input': forces[0], 'solver_failed': False}
    opt.resolve_executed_contact = resolve
    param = SimpleNamespace(lambda_optimizer=opt, target_p_=np.array([.4, .1, .4]),
                            target_q_=np.array([1., 0., 0., 0.]))
    args = SimpleNamespace(ground_height_threshold=.012, ideal_contact_surface_margin=-.0005)
    tip = param.target_p_ + points[0] - .0095 * normals[0]
    result = compute_rollout_contact_via(param, args, np.r_[param.target_p_, param.target_q_, tip],
        np.eye(3), np.zeros(6), np.zeros((40, 6)), .01, ContactValueTracker(),
        ModelCostConfidence(), SmoothedApproachVia(), False, None, floor_ground=.362, floor_z=.35)
    assert result['arrived_at_best']
    assert result['best_contact_idx'] == opt.last_executed_idx == 0
    assert opt.last_global_idx == 1
    np.testing.assert_allclose(result['best_contact_world'], param.target_p_ + points[0])
    np.testing.assert_allclose(result['x_plus_opt'], predictions[0])
    np.testing.assert_allclose(opt.last_executed_x_plus, predictions[0])
    np.testing.assert_allclose(result['info']['control_input'], forces[0])


class _FakeOpt:
    def __init__(self):
        self.sample_point = np.array([
            [0.00, 0.00, 0.06],   # back
            [0.00, 0.00, -0.05],  # foot
            [0.00, 0.02, 0.055],  # back neighbour
        ], dtype=np.float64)
        self.sample_geodesic = np.array([
            [0.00, 0.12, 0.02],
            [0.12, 0.00, 0.13],
            [0.02, 0.13, 0.00],
        ], dtype=np.float64)
        self.contact_switch_radius = 0.03


def _assign(opt, tip, executed_world, physical, executed_idx=0, prev_dwell=None):
    obj = np.zeros(3)
    rot = np.eye(3)
    return _rollout_dwell_assignment(
        opt, tip, tip, obj, rot, executed_idx, executed_world,
        physical, prev_dwell=prev_dwell)


def test_foot_collision_charges_the_foot_not_the_back():
    opt = _FakeOpt()
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.00, 0.00, -0.04])
    progress_idx, active, dead, occupied, on_exec, _ = _assign(
        opt, tip, executed_world, True, prev_dwell=0)
    assert occupied == 1
    assert progress_idx == 1
    assert active and dead
    assert not on_exec


def test_stuck_on_the_foot_keeps_one_cluster_id():
    opt = _FakeOpt()
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.00, 0.00, -0.045])
    progress_idx, active, dead, occupied, on_exec, _ = _assign(
        opt, tip, executed_world, True, prev_dwell=1)
    assert occupied == 1
    assert progress_idx == 1
    assert active and dead
    assert not on_exec


def test_leaving_the_foot_stops_sticking_to_it():
    opt = _FakeOpt()
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.00, 0.00, 0.02])
    progress_idx, active, dead, _, on_exec, _ = _assign(
        opt, tip, executed_world, True, prev_dwell=1)
    assert progress_idx != 1
    assert not on_exec
    assert active and dead


def test_travel_away_from_both_patches_is_inactive():
    opt = _FakeOpt()
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.20, 0.00, 0.10])
    progress_idx, active, dead, _, on_exec, _ = _assign(
        opt, tip, executed_world, False)
    assert progress_idx == 0
    assert not active
    assert not dead
    assert not on_exec


def test_contact_on_the_executed_back_uses_the_stable_executed_idx():
    opt = _FakeOpt()
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.00, 0.015, 0.055])
    progress_idx, active, dead, occupied, on_exec, _ = _assign(
        opt, tip, executed_world, True, executed_idx=0)
    assert on_exec
    assert progress_idx == 0
    assert occupied == 2
    assert active
    assert not dead


def test_geodesically_near_but_far_in_world_is_not_on_target():
    opt = _FakeOpt()
    opt.sample_geodesic[0, 1] = 0.02
    opt.sample_geodesic[1, 0] = 0.02
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.00, 0.00, -0.04])
    progress_idx, active, dead, occupied, on_exec, _ = _assign(
        opt, tip, executed_world, True)
    assert occupied == 1
    assert progress_idx == 1
    assert dead
    assert not on_exec


def test_far_travel_does_not_charge_the_nearest_sample():
    opt = _FakeOpt()
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.20, 0.00, 0.20])
    _, active, dead, occupied, _, _ = _assign(
        opt, tip, executed_world, False)
    assert occupied == 0
    assert not active
    assert not dead


def test_bounce_without_contact_is_travel():
    opt = _FakeOpt()
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.00, 0.00, -0.08])
    progress_idx, active, dead, _, on_exec, _ = _assign(
        opt, tip, executed_world, False)
    assert progress_idx == 0
    assert not active
    assert not dead
    assert not on_exec


def test_approach_near_executed_without_contact_is_still_travel():
    opt = _FakeOpt()
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.00, 0.00, 0.10])
    progress_idx, active, dead, _, on_exec, _ = _assign(
        opt, tip, executed_world, False, executed_idx=0, prev_dwell=0)
    assert progress_idx == 0
    assert not active
    assert not dead
    assert not on_exec


def test_contact_dropout_still_charges_the_nearby_cluster():
    opt = _FakeOpt()
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.00, 0.00, -0.05])
    progress_idx, active, dead, _, on_exec, _ = _assign(
        opt, tip, executed_world, False, executed_idx=0, prev_dwell=1)
    assert progress_idx == 1
    assert active and dead
    assert not on_exec


def test_rank_switch_still_charges_the_occupied_foot():
    opt = _FakeOpt()
    executed_world = np.array([0.00, 0.00, 0.06])
    tip = np.array([0.00, 0.00, -0.05])
    progress_idx, active, dead, _, on_exec, _ = _assign(
        opt, tip, executed_world, True, executed_idx=0, prev_dwell=1)
    assert progress_idx == 1
    assert active and dead
    assert not on_exec


def test_verify_follows_tightness_even_off_target():
    tracker = ContactValueTracker(beta=1.0, exit_threshold=0.2, window_size=1)
    value, _ = tracker.update_verify(
        1.0, 0.08, physical_contact=True, on_target=False, tightness=0.0)
    assert value == 0.0
    value, _ = tracker.update_verify(
        1.0, 0.08, physical_contact=True, on_target=False, tightness=0.55)
    assert value == 0.55
    assert tracker.contact_active


def test_dead_increment_ignores_sliding_pose_noise():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.sample_point = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    opt.sample_geodesic = np.array([[0.0, 0.01], [0.01, 0.0]])
    opt.contact_switch_radius = 0.03
    opt.contact_switch_confidence = 1.0
    opt.contact_patch_max_block_cycles = 320
    opt._dwell_idx = None
    opt._dwell_steps = 0
    opt._dwell_best_cost = None
    opt._dwell_last_cost = None
    opt._dwell_was_active = False
    opt._dwell_blocked = False
    opt._blocked_contact_indices = {}
    opt._contact_patch_failures = {}
    opt.lock_contact_patch = True
    opt.note_contact_progress(
        0, 0.040, active=True, improve_eps=0.002, dead_increment=True,
        gamma=0.5, min_dwell_steps=2)
    opt.note_contact_progress(
        0, 0.037, active=True, improve_eps=0.002, dead_increment=True,
        gamma=0.5, min_dwell_steps=2)
    conf = opt.note_contact_progress(
        0, 0.036, active=True, improve_eps=0.002, dead_increment=True,
        gamma=0.5, min_dwell_steps=2)
    assert conf < 1.0
    assert opt._dwell_steps >= 2


def test_blocked_patch_does_not_restore_confidence():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.sample_point = np.array([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    opt.sample_geodesic = np.array([[0.0, 0.10], [0.10, 0.0]])
    opt.contact_switch_radius = 0.03
    opt.contact_switch_confidence = 0.039
    opt.contact_patch_max_block_cycles = 320
    opt._dwell_idx = 1
    opt._dwell_steps = 4
    opt._dwell_best_cost = 0.07
    opt._dwell_last_cost = 0.07
    opt._dwell_was_active = False
    opt._dwell_blocked = True
    opt._blocked_contact_indices = {0: 2}
    opt._contact_patch_failures = {0: 1}
    opt.lock_contact_patch = False
    conf = opt.note_contact_progress(
        0, 0.07, active=True, dead_increment=True, block_cycles=20,
        block_radius=0.03)
    assert conf == 0.039
    assert 0 in opt._blocked_contact_indices
    assert opt._blocked_contact_indices[0] >= 20
