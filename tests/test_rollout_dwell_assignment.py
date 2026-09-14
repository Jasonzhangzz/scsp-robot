"""Rollout dwell must charge a stuck patch, not every sample the ball grazes."""

import numpy as np

from examples.mpc.fingertips.test.test_0902 import (
    ContactValueTracker,
    _rollout_dwell_assignment,
)
from planning.mlqp_point import LambdaContactControlOptimizer


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


def test_verify_ignores_wrong_patch_contact():
    tracker = ContactValueTracker(window_size=5, confirm_steps=1,
                                  min_hold_steps=0, release_steps=2)
    for _ in range(8):
        value, _ = tracker.update_verify(
            1.0, 0.08, physical_contact=True, on_target=False)
        assert value == 0.0
        assert not tracker.contact_active


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
