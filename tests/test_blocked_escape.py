"""Leaving a blacklisted contact must go around the mesh, not through it."""

import numpy as np

from examples.mpc.fingertips.test.test_0902 import (
    _blocked_escape_via,
    _chord_clears_object,
    _object_top_z_world,
    _should_escape_blocked,
    _should_track_best_via,
    _track_best_via,
)


class _FakeOpt:
    def __init__(self):
        self.sample_point = np.array([
            [0.00, 0.00, 0.06],
            [0.05, 0.00, 0.00],
        ], dtype=np.float64)
        self.normal = np.array([
            [0.00, 0.00, 1.00],
            [1.00, 0.00, 0.00],
        ], dtype=np.float64)
        self._blocked_contact_indices = {1: 40}
        self._dwell_idx = 1


def test_side_leg_via_climbs_above_the_mesh():
    tip = np.array([0.05, 0.00, 0.00])
    obj = np.zeros(3)
    goal = np.array([0.00, 0.00, 0.06])
    via = _blocked_escape_via(
        tip, obj, goal, np.array([1.0, 0.0, 0.0]), object_top_z=0.06)
    assert via[2] == 0.075
    assert abs(via[0] - obj[0]) <= 0.06


def test_via_height_does_not_chase_the_tip():
    tip = np.array([0.05, 0.00, 0.12])
    via = _blocked_escape_via(
        tip, np.zeros(3), np.array([0.00, 0.00, 0.02]),
        np.array([1.0, 0.0, 0.0]), object_top_z=0.06)
    assert via[2] == 0.075


def test_downward_foot_normal_does_not_drive_into_the_table():
    tip = np.array([0.00, 0.00, 0.02])
    via = _blocked_escape_via(
        tip, np.zeros(3), np.array([0.00, 0.00, 0.06]),
        np.array([0.0, 0.0, -1.0]), object_top_z=0.06)
    assert via[2] > tip[2]


def test_chord_through_the_object_is_blocked():
    tip = np.array([0.05, 0.00, 0.00])
    goal = np.array([-0.05, 0.00, 0.00])
    assert not _chord_clears_object(tip, goal, np.zeros(3))


def test_side_pass_clears_the_object():
    tip = np.array([0.20, 0.00, 0.10])
    goal = np.array([0.20, 0.00, 0.00])
    assert _chord_clears_object(tip, goal, np.zeros(3))


def test_escape_starts_on_the_blocked_leg():
    opt = _FakeOpt()
    obj = np.zeros(3)
    rot = np.eye(3)
    goal = np.array([0.00, 0.00, 0.06])
    on, idx = _should_escape_blocked(
        opt, 1, np.array([0.05, 0.00, 0.00]), obj, rot, goal, False)
    assert on and idx == 1
    on, _ = _should_escape_blocked(
        opt, 1, np.array([0.05, 0.00, 0.00]), obj, rot, goal, True)
    assert not on


def test_escape_hold_continues_while_near_the_blocked_cluster():
    opt = _FakeOpt()
    on, idx = _should_escape_blocked(
        opt, 0, np.array([0.06, 0.00, 0.01]), np.zeros(3), np.eye(3),
        np.array([-0.05, 0.00, 0.02]), False, hold_idx=1)
    assert on and idx == 1


def test_midair_hold_far_from_cluster_does_not_escape():
    opt = _FakeOpt()
    on, _ = _should_escape_blocked(
        opt, 0, np.array([0.20, 0.00, 0.08]), np.zeros(3), np.eye(3),
        np.array([0.00, 0.00, 0.02]), False, hold_idx=1,
        physical_contact=False)
    assert not on


def test_released_does_not_rearm_in_free_flight():
    opt = _FakeOpt()
    on, _ = _should_escape_blocked(
        opt, 1, np.array([0.05, 0.00, 0.04]), np.zeros(3), np.eye(3),
        np.array([0.00, 0.00, 0.02]), False, object_top_z=0.08,
        released=True, physical_contact=False)
    assert not on


def test_clear_of_the_mesh_top_stops_the_via():
    opt = _FakeOpt()
    on, _ = _should_escape_blocked(
        opt, 1, np.array([0.05, 0.00, 0.12]), np.zeros(3), np.eye(3),
        np.array([0.00, 0.00, 0.06]), False, object_top_z=0.08)
    assert not on


def test_object_top_follows_aabb():
    top = _object_top_z_world(
        np.zeros(3), np.eye(3), [-0.06, -0.04, -0.04], [0.06, 0.04, 0.06])
    assert top == 0.06


def test_track_best_via_starts_when_the_chord_hits_the_object():
    assert _should_track_best_via(
        np.array([0.05, 0.00, 0.00]), np.array([-0.05, 0.00, 0.00]),
        np.zeros(3), False, object_top_z=0.06)


def test_track_best_via_keeps_going_if_chord_still_hits():
    assert _should_track_best_via(
        np.array([0.05, 0.00, 0.12]), np.array([-0.05, 0.00, 0.00]),
        np.zeros(3), False, object_top_z=0.08)


def test_track_best_via_stops_when_over_the_object():
    assert not _should_track_best_via(
        np.array([0.01, 0.00, 0.12]), np.array([-0.05, 0.00, 0.00]),
        np.zeros(3), False, object_top_z=0.08)


def test_track_best_via_stops_when_hovering_on_the_via():
    # Via is top+1.5 cm.  A tip 1.0 cm above the top used to keep the
    # waypoint and never descend onto best_contact.
    assert not _should_track_best_via(
        np.array([0.02, 0.00, 0.090]), np.array([-0.05, 0.00, 0.02]),
        np.zeros(3), False, object_top_z=0.08)


def test_track_best_via_pulls_back_when_the_tip_has_drifted():
    assert _should_track_best_via(
        np.array([0.20, 0.00, 0.04]), np.array([-0.05, 0.00, 0.00]),
        np.zeros(3), False, object_top_z=0.08)


def test_track_best_via_stops_when_chord_clears():
    # Close enough that the far-drift pull-back does not apply.
    assert not _should_track_best_via(
        np.array([0.07, 0.00, 0.10]), np.array([0.07, 0.00, 0.00]),
        np.zeros(3), False, object_top_z=0.08)


def test_track_best_via_stops_on_a_tail_far_from_the_com():
    # Bunny tail: horiz to COM > 8 cm, but the tip is already on the goal.
    tip = np.array([-0.06, 0.14, 0.08])
    goal = np.array([-0.061, 0.145, 0.076])
    assert float(np.linalg.norm(tip[:2])) > 0.08
    assert not _should_track_best_via(tip, goal, np.zeros(3), False,
                                      object_top_z=0.08)


def test_track_best_via_is_anchored_to_the_object():
    tip = np.array([0.20, 0.00, 0.04])
    via = _track_best_via(np.zeros(3), np.array([-0.05, 0.00, 0.02]), 0.06)
    assert via[2] == 0.075
    assert float(np.linalg.norm(via[:2])) < 0.04
    assert float(np.linalg.norm(via[:2])) < float(np.linalg.norm(tip[:2]))
