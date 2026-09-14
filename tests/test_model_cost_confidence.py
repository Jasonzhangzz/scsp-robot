"""Model-vs-reality cost reduction tightens verify, it does not blacklist."""

import numpy as np
import pytest

from examples.mpc.fingertips.test.test_0902 import (
    ContactValueTracker,
    ModelCostConfidence,
    _arrived_at_best_contact,
    _cost_span_quality,
    _lambda_pose_cost,
    _protect_destination_dwell,
    _verify_cost_threshold,
)


def test_threshold_collapses_onto_min_error():
    assert _verify_cost_threshold(1.0, 5.0, tightness=0.0) == 5.0
    assert _verify_cost_threshold(1.0, 5.0, tightness=0.5) == 3.0
    assert _verify_cost_threshold(1.0, 5.0, tightness=1.0) == 1.0


def test_best_contact_quality_is_one():
    assert _cost_span_quality(1.0, 1.0, 5.0) == 1.0
    assert _cost_span_quality(5.0, 1.0, 5.0) == 0.0
    assert _cost_span_quality(3.0, 1.0, 5.0) == 0.5


def test_lambda_pose_cost_matches_identity_local_frame():
    # current = identity, target 2 cm along x, pos_coef=500, ori_coef=20
    cost = _lambda_pose_cost(
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        [0.02, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        500.0, 20.0)
    assert cost == pytest.approx(500.0 * 0.02 ** 2)


def test_underdelivery_is_capped_so_one_bounce_is_not_terminal():
    conf = ModelCostConfidence(threshold=6.0, min_steps=3)
    assert conf.tightness() == 0.0
    conf.observe(0.40, 0.40)
    assert conf.accum == 0.0
    conf.observe(2.40, -0.01)
    assert conf.accum == pytest.approx(2.0)
    assert conf.tightness() < 0.4
    conf.observe(2.40, 0.0)
    conf.observe(2.40, 0.0)
    assert conf.tightness() == 1.0


def test_new_geodesic_patch_resets_the_gate():
    conf = ModelCostConfidence(threshold=6.0, min_steps=3)
    conf.observe(2.40, 0.0)
    conf.observe(2.40, 0.0)
    assert conf.tightness() > 0.0

    class _Opt:
        sample_geodesic = np.array([[0.0, 0.01, 0.12],
                                    [0.01, 0.0, 0.11],
                                    [0.12, 0.11, 0.0]])

    opt = _Opt()
    conf.note_sample(opt, 0)
    conf.note_sample(opt, 1)
    assert conf.tightness() > 0.0
    conf.note_sample(opt, 2)
    assert conf.tightness() == 0.0
    assert conf.accum == 0.0


def test_span_gate_does_not_throw_away_on_low_confidence():
    tracker = ContactValueTracker()
    info = tracker.update_values(
        1.0, 1.0, solver_ok=True, same_patch=True, near_arm=True,
        is_best_sample=True, confidence=0.2,
        min_error=1.0, max_error=5.0, tightness=0.0)
    assert info['accept_p_arm']
    assert info['cost_ok']


def test_tightness_rejects_neighbour_but_keeps_min_error():
    tracker = ContactValueTracker()
    neighbour = tracker.update_values(
        1.0, 3.0, solver_ok=True, same_patch=True, near_arm=True,
        min_error=1.0, max_error=5.0, tightness=1.0)
    assert not neighbour['accept_p_arm']
    assert not neighbour['cost_ok']
    assert neighbour['cost_thresh'] == 1.0

    best = ContactValueTracker().update_values(
        1.0, 1.0, solver_ok=True, same_patch=True, near_arm=True,
        is_best_sample=True, min_error=1.0, max_error=5.0, tightness=1.0)
    assert best['accept_p_arm']
    assert best['cost_ok']
    assert best['quality'] == 1.0


def test_held_neighbour_is_released_when_gate_hits_min_error():
    tracker = ContactValueTracker()
    first = tracker.update_values(
        1.0, 3.0, solver_ok=True, same_patch=True, near_arm=True,
        min_error=1.0, max_error=5.0, tightness=0.0)
    assert first['accept_p_arm']
    released = tracker.update_values(
        1.0, 3.0, solver_ok=True, same_patch=True, near_arm=True,
        min_error=1.0, max_error=5.0, tightness=1.0)
    assert not released['accept_p_arm']
    assert not tracker._holding_p_arm


def test_verify_enter_at_min_error_survives_full_tightness():
    tracker = ContactValueTracker(window_size=1, confirm_steps=1,
                                  min_hold_steps=0, release_steps=2)
    value, _ = tracker.update_verify(
        1.0, 0.001, physical_contact=True, on_target=True, tightness=1.0)
    assert tracker.contact_active
    assert value > 0.0


def test_best_sample_is_accepted_even_when_ema_lags():
    tracker = ContactValueTracker()
    tracker.v_arm = 9.0
    info = tracker.update_values(
        1.0, 1.0, solver_ok=True, same_patch=True, near_arm=True,
        is_best_sample=True, min_error=1.0, max_error=5.0, tightness=1.0)
    assert info['accept_p_arm']
    assert info['cost_ok']


def test_approach_press_can_enter_without_physical_contact():
    tracker = ContactValueTracker(window_size=1, confirm_steps=1,
                                  min_hold_steps=0, release_steps=2)
    value, _ = tracker.update_verify(
        1.0, 0.001, physical_contact=False, on_target=True,
        allow_approach_press=True)
    assert tracker.contact_active
    assert value > 0.0


def test_arrival_uses_the_closer_of_surface_and_track():
    # Logged bunny tail: surface 2.4 cm, sphere-centre track 3.1 cm.
    tip = np.array([-0.040, 0.145, 0.081])
    surface = np.array([-0.0615, 0.1453, 0.0753])
    track = np.array([-0.070, 0.155, 0.090])
    assert float(np.linalg.norm(tip - surface)) < 0.03
    assert float(np.linalg.norm(tip - track)) > 0.03
    assert _arrived_at_best_contact(tip, surface, track)


def test_arrival_rejects_a_far_tip():
    tip = np.array([0.02, 0.03, 0.02])
    surface = np.array([-0.06, 0.14, 0.075])
    track = np.array([-0.07, 0.15, 0.082])
    assert not _arrived_at_best_contact(tip, surface, track)


def test_arrival_accepts_occupied_best_patch_within_5cm():
    class _Opt:
        sample_geodesic = np.array([[0.0, 0.02], [0.02, 0.0]])

    tip = np.array([-0.06, 0.14, 0.11])
    surface = np.array([-0.06, 0.14, 0.075])
    track = np.array([-0.07, 0.15, 0.082])
    assert float(np.linalg.norm(tip - surface)) > 0.03
    assert float(np.linalg.norm(tip - surface)) < 0.05
    assert _arrived_at_best_contact(
        tip, surface, track, occupied_idx=1, best_idx=0, optimizer=_Opt())
    assert not _arrived_at_best_contact(
        tip, surface, track, occupied_idx=1, best_idx=0, optimizer=_Opt(),
        radius=0.01)


def test_destination_graze_is_not_a_dead_dwell():
    class _Opt:
        sample_geodesic = np.array([[0.0, 0.02, 0.12],
                                    [0.02, 0.0, 0.11],
                                    [0.12, 0.11, 0.0]])

    opt = _Opt()
    active, dead = _protect_destination_dwell(opt, 1, 0, True, True)
    assert not active and not dead
    active, dead = _protect_destination_dwell(opt, 2, 0, True, True)
    assert active and dead


def test_clamped_arrival_distance_clears_tight_enter_gate():
    tracker = ContactValueTracker(window_size=1, confirm_steps=1,
                                  min_hold_steps=0, release_steps=2)
    value, q_dist = tracker.update_verify(
        1.0, 0.01, physical_contact=False, on_target=True,
        tightness=1.0, allow_approach_press=True)
    assert q_dist > 0.85
    assert tracker.contact_active
    assert value > 0.0
