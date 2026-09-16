"""Model-vs-reality cost reduction tightens verify, it does not blacklist."""

import numpy as np
import pytest

from examples.mpc.fingertips.test.test_0902 import (
    ContactValueTracker,
    ModelCostConfidence,
    SmoothedApproachVia,
    _arrived_at_best_contact,
    _blend_travel_to_press,
    _patch_press_point,
    _floor_slide_away_from_patch,
    _on_opposite_sides,
    _cost_span_quality,
    _lambda_pose_cost,
    _protect_destination_dwell,
    _rollout_verify_cost,
    _should_observe_model_cost,
    _orbit_xy,
    _press_approach_desired,
    _press_path_blocked,
    _travel_press_weight,
    _verify_cost_threshold,
    _verify_distance,
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


def test_tightness_rejects_neighbour_but_keeps_best():
    tracker = ContactValueTracker()
    neighbour = tracker.update_values(
        1.0, 3.0, solver_ok=True, same_patch=True, near_arm=True,
        min_error=1.0, max_error=5.0, tightness=1.0)
    assert not neighbour['accept_p_arm']
    assert neighbour['accept_scale'] == pytest.approx(0.0)

    best = ContactValueTracker().update_values(
        1.0, 1.0, solver_ok=True, same_patch=True, near_arm=True,
        is_best_sample=True, min_error=1.0, max_error=5.0, tightness=1.0)
    assert best['accept_p_arm']
    assert best['quality'] == 1.0


def test_held_neighbour_is_released_when_tightness_rises():
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


def test_neighbor_cost_jump_does_not_flip_accept():
    tracker = ContactValueTracker()
    cheap = tracker.update_values(
        1.0, 1.2, same_patch=True, near_arm=True, tightness=0.1)
    expensive = tracker.update_values(
        1.0, 9.0, same_patch=True, near_arm=True, tightness=0.1)
    assert cheap['accept_p_arm']
    assert expensive['accept_p_arm'] == cheap['accept_p_arm']


def test_verify_tracks_tightness_not_arm_cost_or_distance():
    tracker = ContactValueTracker(beta=1.0, exit_threshold=0.2)
    near, _ = tracker.update_verify(quality=1.0, dist_exec=0.001, tightness=0.6)
    far, _ = tracker.update_verify(quality=0.1, dist_exec=0.20, tightness=0.6)
    assert near == pytest.approx(0.6)
    assert far == pytest.approx(0.6)
    assert tracker.contact_active


def test_best_sample_is_accepted_even_when_ema_lags():
    tracker = ContactValueTracker()
    tracker.v_arm = 9.0
    info = tracker.update_values(
        1.0, 1.0, solver_ok=True, same_patch=True, near_arm=True,
        is_best_sample=True, min_error=1.0, max_error=5.0, tightness=1.0)
    assert info['accept_p_arm']
    assert info['cost_ok']


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


def test_arrival_rejects_an_occupied_patch_beyond_3cm():
    class _Opt:
        sample_geodesic = np.array([[0.0, 0.02], [0.02, 0.0]])

    tip = np.array([-0.06, 0.14, 0.11])
    surface = np.array([-0.06, 0.14, 0.075])
    track = np.array([-0.07, 0.15, 0.082])
    assert float(np.linalg.norm(tip - surface)) > 0.03
    assert not _arrived_at_best_contact(
        tip, surface, track, occupied_idx=1, best_idx=0, optimizer=_Opt())


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


def test_clamped_arrival_distance_is_logged_but_does_not_write_verify():
    tracker = ContactValueTracker(beta=1.0, exit_threshold=0.2)
    near = _verify_distance(0.031, 0.024, 0.031, arrived=True)
    assert near == pytest.approx(0.01)
    value, q_dist = tracker.update_verify(dist_exec=near, tightness=0.7)
    assert q_dist > 0.85
    assert value == pytest.approx(0.7)
    assert tracker.contact_active


def test_far_false_arrival_keeps_the_real_distance():
    # Logged bunny ear: arrived=1 while the tip was 6.5 cm from the patch.
    far = _verify_distance(0.074, 0.065, 0.070, arrived=True)
    assert far == pytest.approx(0.065)
    tracker = ContactValueTracker(beta=1.0, exit_threshold=0.2)
    value, q_dist = tracker.update_verify(dist_exec=far, tightness=0.7)
    assert q_dist < 0.01
    assert value == pytest.approx(0.7)


def test_tightness_or_verify_flip_does_not_snap_travel():
    via = np.array([0.0, 0.0, 0.20])
    press = np.array([0.0, 0.0, 0.05])
    far, escape = _blend_travel_to_press(via, press, tightness=0.0, use_via=True)
    assert escape
    assert np.allclose(far, via)
    mid, _ = _blend_travel_to_press(via, press, tightness=0.5, use_via=True)
    near, escape = _blend_travel_to_press(via, press, tightness=1.0, use_via=True)
    dropped, drop_escape = _blend_travel_to_press(
        via, press, tightness=1.0, use_via=False, via_phase='drop')
    # Same via in → same travel out.  tightness / use_via / verify_cost
    # may flip 0↔1; the via filter is what walks toward press.
    assert np.allclose(mid, via)
    assert np.allclose(near, via)
    assert np.allclose(dropped, via)
    assert escape
    assert not drop_escape
    assert _travel_press_weight(0.5) == pytest.approx(0.5)


def test_patch_press_stays_on_the_sphere_track():
    surface = np.array([0.04, 0.09, 0.015])
    outward = np.array([1.0, 0.0, 0.0])
    track = surface + 0.0105 * outward
    press = _patch_press_point(track, surface)
    expect = track - 0.0025 * outward
    expect[2] = max(float(expect[2]), float(surface[2]))
    assert np.allclose(press, expect)
    assert float(press[2]) >= float(surface[2])


def test_via_displacement_is_clipped():
    via = SmoothedApproachVia(rate=1.0, max_step=0.006)
    via.via = np.array([0.10, 0.0, 0.20])
    tip = np.array([0.10, 0.0, 0.20])
    obj = np.array([0.0, 0.0, 0.03])
    press = np.array([-0.08, 0.0, 0.04])
    start = via.via.copy()
    _, target, _ = via.update(tip, obj, press, press, 0.08, 0.06, False, press)
    step = float(np.linalg.norm(target - start))
    assert step == pytest.approx(0.006, abs=1e-9)
    assert not np.allclose(target, press)


def test_via_stays_within_a_short_lead_of_the_tip():
    via = SmoothedApproachVia(rate=1.0, max_step=0.02, max_lead=0.005)
    via.via = np.array([0.10, 0.0, 0.20])
    tip = np.array([0.10, 0.0, 0.08])
    obj = np.array([0.0, 0.0, 0.03])
    press = np.array([-0.08, 0.0, 0.04])
    _, target, _ = via.update(tip, obj, press, press, 0.08, 0.06, False, press)
    assert float(np.linalg.norm(target - tip)) == pytest.approx(0.005, abs=1e-9)
    assert not np.allclose(target, press)


def test_via_lerps_onto_press_instead_of_snapping():
    via = SmoothedApproachVia(rate=0.25)
    via.via = np.array([0.05, 0.0, 0.20])
    tip = np.array([0.05, 0.0, 0.20])
    obj = np.array([0.0, 0.0, 0.03])
    best = np.array([0.05, 0.0, 0.04])
    track = np.array([0.05, 0.0, 0.052])
    press = np.array([0.05, 0.0, 0.05])
    _, target, phase = via.update(tip, obj, best, track, 0.08, 0.06, True, press)
    assert phase == 'drop'
    assert np.allclose(target, 0.75 * np.array([0.05, 0.0, 0.20]) + 0.25 * press)
    assert not np.allclose(target, press)
    assert target[2] < 0.20


def test_lift_or_opposite_side_keeps_the_via_even_at_full_tightness():
    via = np.array([0.10, 0.0, 0.20])
    press = np.array([-0.05, 0.0, 0.05])
    travel, escape = _blend_travel_to_press(
        via, press, tightness=1.0, use_via=True, via_phase='lift')
    assert escape
    assert np.allclose(travel, via)
    travel, escape = _blend_travel_to_press(
        via, press, tightness=1.0, use_via=True, via_phase='cross')
    assert escape
    assert np.allclose(travel, via)
    travel, escape = _blend_travel_to_press(
        via, press, tightness=1.0, use_via=True, opposite=True)
    assert escape
    assert np.allclose(travel, via)
    assert _travel_press_weight(1.0, via_phase='lift') == 0.0
    assert _travel_press_weight(1.0, via_phase='above') == 0.0
    assert _travel_press_weight(1.0, opposite=True) == 0.0


def test_above_phase_keeps_the_hover_via_at_full_tightness():
    via = np.array([0.08, 0.0, 0.20])
    press = np.array([0.08, 0.0, 0.018])
    travel, escape = _blend_travel_to_press(
        via, press, tightness=1.0, use_via=True, via_phase='above')
    assert escape
    assert np.allclose(travel, via)
    assert travel[2] == pytest.approx(0.20)


def test_opposite_sides_uses_the_horizontal_dot():
    tip = np.array([0.08, 0.0, 0.04])
    obj = np.array([0.0, 0.0, 0.03])
    assert _on_opposite_sides(tip, obj, np.array([-0.06, 0.0, 0.04]))
    assert not _on_opposite_sides(tip, obj, np.array([0.07, 0.01, 0.04]))


def test_via_and_p_arm_do_not_override_verify_cost():
    assert _rollout_verify_cost(0.12) == pytest.approx(0.12)
    assert _rollout_verify_cost(0.0) == 0.0
    # The old via-on / abort-orbit path forced 0.0 or 0.85 here.
    assert _rollout_verify_cost(0.12) != 0.0
    assert _rollout_verify_cost(0.12) != 0.85


def test_observe_whenever_pose_cost_can_scale():
    assert _should_observe_model_cost(True)
    assert not _should_observe_model_cost(False)
    # A flat predicted-best band is still a valid comparison once C(now)
    # exists.  Real contact can sit below every optimistic sample ΔC.
    assert _should_observe_model_cost(False, pose_cost_now=5.0)
    assert not _should_observe_model_cost(True, pose_cost_now=0.0)


def test_airborne_underdelivery_raises_tightness():
    # Fraction of C(now): predicted close → +1, airborne actual=0 → 0.
    conf = ModelCostConfidence(threshold=6.0, min_steps=3)
    for _ in range(3):
        conf.observe(1.0, -1.0)
    assert conf.tightness() == 1.0


def test_confidence_falls_when_actual_beats_prediction():
    conf = ModelCostConfidence(threshold=6.0, min_steps=3)
    conf.observe(1.0, 0.0)
    conf.observe(1.0, 0.0)
    raised = conf.tightness()
    conf.observe(0.2, 1.0)
    assert conf.tightness() < raised


def _via_geom():
    tip = np.array([0.05, 0.0, 0.045])
    obj = np.array([0.0, 0.0, 0.03])
    best = np.array([0.05, 0.0, 0.04])
    track = np.array([0.05, 0.0, 0.052])
    press = np.array([0.05, 0.0, 0.046])
    return tip, obj, best, track, press


def test_floor_slide_is_a_table_graze_beside_a_raised_patch():
    tip = np.array([0.014, 0.070, 0.008])
    best = np.array([0.014, 0.095, 0.018])
    assert _floor_slide_away_from_patch(tip, best)
    assert not _floor_slide_away_from_patch(
        np.array([0.014, 0.095, 0.022]), best)
    # XY-close is the landing corridor, even if the sample is a bit higher.
    assert not _floor_slide_away_from_patch(
        np.array([0.014, 0.090, 0.010]), best)
    # A sample that is itself at table height is a real floor press.
    assert not _floor_slide_away_from_patch(
        np.array([0.014, 0.090, 0.010]),
        np.array([0.014, 0.095, 0.011]))
    assert not _arrived_at_best_contact(
        tip, best, best + np.array([0.0, 0.0, 0.01]))


def test_via_on_the_far_face_stays_on_the_keepout_and_high():
    via = SmoothedApproachVia(rate=1.0)
    obj = np.array([0.0, 0.0, 0.03])
    best = np.array([0.08, 0.0, 0.04])
    track = np.array([0.08, 0.0, 0.052])
    press = np.array([0.08, 0.0, 0.046])
    tip = np.array([-0.08, 0.0, 0.01])
    _, target, phase = via.update(tip, obj, best, track, 0.08, 0.06, False, press)
    assert via.blocked
    assert phase in ('lift', 'cross')
    assert target[2] > press[2] + 0.02
    assert float(np.linalg.norm(target[:2] - obj[:2])) >= 0.05


def test_same_side_clear_chord_desired_is_press():
    via = SmoothedApproachVia(rate=1.0)
    tip, obj, best, track, press = _via_geom()
    _, target, phase = via.update(tip, obj, best, track, 0.08, 0.06, False, press)
    assert not via.blocked
    assert phase == 'drop'
    assert np.allclose(target, press)


def test_same_side_mid_height_desired_z_falls_toward_press():
    via = SmoothedApproachVia(rate=1.0)
    obj = np.array([0.0, 0.0, 0.03])
    best = np.array([0.08, 0.0, 0.04])
    track = np.array([0.08, 0.0, 0.052])
    press = np.array([0.08, 0.0, 0.046])
    tip = np.array([0.05, 0.0, 0.055])
    _, target, _ = via.update(tip, obj, best, track, 0.08, 0.06, False, press)
    assert target[2] <= press[2] + 0.005
    assert float(np.linalg.norm(target - press)) < 0.02


def test_via_does_not_climb_while_descending_same_side():
    via = SmoothedApproachVia(rate=1.0)
    via.via = np.array([0.05, 0.0, 0.09])
    tip, obj, best, track, press = _via_geom()
    _, target, phase = via.update(tip, obj, best, track, 0.08, 0.06, False, press)
    assert phase != 'lift'
    assert target[2] <= 0.09


def test_via_stays_in_drop_after_leaving_the_hover():
    via = SmoothedApproachVia(rate=1.0)
    tip, obj, best, track, press = _via_geom()
    use_via, target, phase = via.update(
        tip, obj, best, track, 0.08, 0.06, False, press)
    assert phase == 'drop'
    assert not use_via
    assert np.allclose(target, press)


def test_open_same_side_via_goes_to_press_not_sky():
    via = SmoothedApproachVia(rate=1.0)
    obj = np.array([0.0, 0.0, 0.03])
    best = np.array([0.12, 0.0, 0.04])
    track = np.array([0.12, 0.0, 0.052])
    press = np.array([0.12, 0.0, 0.046])
    tip = np.array([0.10, 0.0, 0.098])
    _, target, phase = via.update(tip, obj, best, track, 0.08, 0.06, False, press)
    assert phase == 'drop'
    assert np.allclose(target, press)


def test_side_patch_clear_approach_desired_is_press():
    via = SmoothedApproachVia(rate=1.0)
    obj = np.array([0.0, 0.0, 0.03])
    best = np.array([0.06, 0.0, 0.018])
    track = np.array([0.075, 0.0, 0.018])
    press = np.array([0.072, 0.0, 0.020])
    tip = np.array([0.10, 0.0, 0.10])
    _, target, phase = via.update(tip, obj, best, track, 0.08, 0.06, False, press)
    assert phase == 'drop'
    assert np.allclose(target, press)


def test_via_drops_once_over_the_patch():
    via = SmoothedApproachVia(rate=1.0)
    obj = np.array([0.0, 0.0, 0.03])
    best = np.array([0.08, 0.0, 0.04])
    track = np.array([0.08, 0.0, 0.052])
    press = np.array([0.08, 0.0, 0.046])
    tip = np.array([0.08, 0.0, 0.098])
    _, target, phase = via.update(tip, obj, best, track, 0.08, 0.06, False, press)
    assert phase == 'drop'
    assert np.allclose(target, press)


def test_same_side_surface_hop_is_not_a_through_mesh_approach():
    obj = np.array([0.0, 0.0, 0.03])
    tip = np.array([0.05, 0.0, 0.045])
    press = np.array([0.048, 0.018, 0.043])
    keepout = 0.06
    assert float(np.linalg.norm(tip - press)) <= 0.03
    assert not _on_opposite_sides(tip, obj, press)
    assert not _press_path_blocked(tip, obj, press, keepout)
    via = SmoothedApproachVia(rate=1.0)
    _, target, phase = via.update(
        tip, obj, press, press + np.array([0.0, 0.0, 0.01]),
        0.08, keepout, False, press)
    assert phase == 'drop'
    assert not via.blocked
    assert np.allclose(target, press)


def test_keepout_local_through_body_still_orbits():
    obj = np.array([0.0, 0.0, 0.03])
    tip = np.array([0.11, 0.02, 0.03])
    press = np.array([0.00, 0.02, 0.03])
    keepout = 0.10
    assert float(np.linalg.norm(tip[:2] - obj[:2])) <= keepout + 0.012
    assert float(np.linalg.norm(tip - press)) > 0.03
    assert not _on_opposite_sides(tip, obj, press)
    assert _press_path_blocked(tip, obj, press, keepout)
    via = SmoothedApproachVia(rate=1.0)
    _, target, phase = via.update(
        tip, obj, press, press + np.array([0.0, 0.0, 0.01]),
        0.08, keepout, False, press)
    assert via.blocked
    assert phase in ('lift', 'cross')
    assert float(np.linalg.norm(target[:2] - obj[:2])) >= 0.09


def test_wide_com_azimuth_inside_keepout_may_drop():
    """Same-side contacts already inside the keep-out are a slide, not an orbit."""
    obj = np.array([0.0, 0.0, 0.03])
    tip = np.array([0.05, 0.035, 0.05])
    press = np.array([0.05, -0.035, 0.05])
    keepout = 0.09
    assert not _on_opposite_sides(tip, obj, press)
    assert float(np.linalg.norm(tip[:2] - obj[:2])) <= keepout
    assert not _press_path_blocked(tip, obj, press, keepout)


def test_orbit_helper_still_takes_the_long_way_around_the_back():
    obj = np.array([0.0, 0.0, 0.03])
    tip = np.array([0.06, 0.04, 0.04])
    press = np.array([0.06, -0.04, 0.04])
    keepout = 0.10
    nxt = _orbit_xy(tip[:2], obj[:2], press[:2], keepout)
    # Short arc would walk toward +x / the snout.  The long way goes +y.
    assert float(nxt[1]) > float(nxt[0])


def test_same_half_face_cheeks_may_drop():
    """Small COM azimuth on the same face is a slide onto press."""
    obj = np.array([0.0, 0.0, 0.03])
    tip = np.array([0.05, 0.018, 0.04])
    press = np.array([0.05, -0.018, 0.04])
    keepout = 0.09
    assert not _on_opposite_sides(tip, obj, press)
    assert float(np.linalg.norm(tip[:2] - obj[:2])) <= keepout
    assert not _press_path_blocked(tip, obj, press, keepout)
    desired, blocked = _press_approach_desired(
        tip, obj, press, keepout, 0.098)
    assert not blocked
    assert np.allclose(desired, press)


def test_aligned_on_keepout_rim_drops_onto_press():
    obj = np.array([0.0, 0.0, 0.03])
    keepout = 0.09
    press = np.array([0.05, 0.0, 0.04])
    tip = np.array([0.09, 0.0, 0.10])
    assert not _press_path_blocked(tip, obj, press, keepout)
    desired, blocked = _press_approach_desired(
        tip, obj, press, keepout, 0.098)
    assert not blocked
    assert np.allclose(desired, press)


def test_aligned_inside_keepout_keeps_dropping():
    """Do not bounce back to the rim after leaving it toward dest."""
    obj = np.array([0.0, 0.0, 0.03])
    keepout = 0.09
    press = np.array([0.04, 0.0, 0.04])
    tip = np.array([0.07, 0.01, 0.09])
    assert float(np.linalg.norm(tip[:2] - press[:2])) > 0.03
    assert float(np.linalg.norm(tip[:2] - obj[:2])) < keepout - 0.012
    assert not _press_path_blocked(tip, obj, press, keepout)
    desired, blocked = _press_approach_desired(
        tip, obj, press, keepout, 0.098)
    assert not blocked
    assert np.allclose(desired, press)


def test_far_same_side_radial_approach_may_drop():
    """A radial inbound chord misses the COM ball; do not force an orbit."""
    obj = np.array([0.0, 0.0, 0.03])
    tip = np.array([0.18, 0.0, 0.05])
    press = np.array([0.06, 0.0, 0.04])
    keepout = 0.09
    assert float(np.linalg.norm(tip[:2] - obj[:2])) > keepout
    assert float(np.linalg.norm(tip[:2] - press[:2])) > 0.03
    assert not _on_opposite_sides(tip, obj, press)
    assert not _press_path_blocked(tip, obj, press, keepout)
    via = SmoothedApproachVia(rate=1.0)
    _, target, phase = via.update(
        tip, obj, press, press + np.array([0.0, 0.0, 0.01]),
        0.08, keepout, False, press)
    assert not via.blocked
    assert phase == 'drop'
    assert np.allclose(target, press)


def test_far_inbound_chord_still_orbits():
    obj = np.array([0.0, 0.0, 0.03])
    tip = np.array([0.02, 0.12, 0.04])
    press = np.array([0.02, 0.0, 0.03])
    keepout = 0.06
    assert float(np.linalg.norm(tip[:2] - obj[:2])) > keepout
    assert not _on_opposite_sides(tip, obj, press)
    assert _press_path_blocked(tip, obj, press, keepout)
    via = SmoothedApproachVia(rate=1.0)
    _, target, phase = via.update(
        tip, obj, press, press + np.array([0.0, 0.0, 0.01]),
        0.08, keepout, False, press)
    assert via.blocked
    assert phase in ('lift', 'cross')
    assert float(target[2]) > float(press[2]) + 0.01


def test_opposite_face_still_orbits():
    obj = np.array([0.0, 0.0, 0.03])
    tip = np.array([0.08, 0.0, 0.06])
    press = np.array([-0.08, 0.0, 0.04])
    keepout = 0.09
    assert _on_opposite_sides(tip, obj, press)
    assert _press_path_blocked(tip, obj, press, keepout)
    desired, blocked = _press_approach_desired(
        tip, obj, press, keepout, 0.098)
    assert blocked
    assert float(np.linalg.norm(desired[:2] - obj[:2])) == pytest.approx(keepout)


def test_mpc_terminal_pose_matches_lambda_weights():
    from examples.mpc.fingertips.test.params import ExplicitMPCParams

    param = ExplicitMPCParams.__new__(ExplicitMPCParams)
    param.n_qpos_ = 10
    param.n_cmd_ = 3
    param.max_ncon_ = 1
    param.n_qvel_ = 9
    param.attract_coef = 20.0
    param.reject_coef = 0.0
    param.reject_dis = 0.02
    param.contact_coef = 20.0
    param.contact_cost_param = 0.0
    param.spline_escape_cost = True
    param.field_cost_weight = 0.0
    param.quadratic_contact_track = True
    param.rollout_press_patch = True
    param.pos_coef = 500.0
    param.ori_coef = 20.0
    param.smooth_contact_detour = False
    path_fn, final_fn = param.init_cost_fns()
    x = np.zeros(10)
    x[3] = 1.0
    n_phi = 4
    n_jac = 4 * 9
    p_off = np.concatenate([
        [0.10, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        np.zeros(n_phi), np.zeros(n_jac), [1.0],
        np.zeros(3), np.zeros(3),
    ])
    p_on = p_off.copy()
    p_on[:3] = 0.0
    # Path must stay attract/press only; object pose lives on the terminal.
    u = np.zeros(3)
    assert float(np.asarray(path_fn(x, u, p_off)).reshape(())) == pytest.approx(
        float(np.asarray(path_fn(x, u, p_on)).reshape(())))
    terminal = float(np.asarray(final_fn(x, p_off)).reshape(()))
    assert terminal == pytest.approx(10.0 * _lambda_pose_cost(
        x[:3], x[3:7], [0.10, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 500.0, 20.0))
