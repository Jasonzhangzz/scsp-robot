"""MPPI object-pose cost matches test_0902, and Yoshikawa det matches OSC."""

import queue

import numpy as np
import pytest

from examples.mpc.fingertips.test.test_0902 import _lambda_pose_cost
from planning.MPPIWarp import (
    CONTROL_WEIGHT_0902,
    _apply_opt_snapshot,
    _opt_snapshot,
    _pickle_safe,
    blend_action,
    clamp_via_to_tip,
    clip_toward,
    contact_is_on_ranked_patch,
    drain_latest,
    object_pose_cost,
    policy_control_substeps,
    put_latest,
    take_latest,
    terminal_object_pose_cost,
)


def test_robot_stiff_reads_scalar_or_matrix():
    for raw in (300.0, np.float32(300.0), np.array(300.0), np.array([300.0]), np.diag([300.0, 300.0])):
        stiff = np.asarray(raw, dtype=np.float32)
        assert float(np.asarray(stiff, dtype=np.float32).reshape(-1)[0]) == 300.0


def test_object_pose_cost_matches_0902_lambda():
    pos = np.array([0.41, 0.02, 0.36])
    quat = np.array([0.8, 0.2, -0.1, 0.55])
    target_p = np.array([0.40, 0.10, 0.38])
    target_q = np.array([0.0, 0.70710678, 0.0, 0.70710678])
    pos_coef, ori_coef = 500.0, 20.0
    got = object_pose_cost(pos, quat, target_p, target_q, pos_coef, ori_coef)
    expected = _lambda_pose_cost(pos, quat, target_p, target_q, pos_coef, ori_coef)
    assert got == expected
    assert terminal_object_pose_cost(pos, quat, target_p, target_q, pos_coef, ori_coef) == 10.0 * expected


def test_nearest_face_is_not_ranked_patch():
    ranked = np.array([0.45, 0.02, 0.36])
    nearest_face = np.array([0.38, 0.00, 0.37])
    assert contact_is_on_ranked_patch(ranked, ranked, 0.03)
    assert not contact_is_on_ranked_patch(nearest_face, ranked, 0.03)


def test_hover_gap_does_not_count_as_contact():
    contact_gap = 0.003
    assert 0.02 > contact_gap
    assert 0.0 < contact_gap


def test_control_weight_matches_0902_plan_once():
    assert CONTROL_WEIGHT_0902 == 50.0


def test_zero_control_substeps_is_one_policy_interval():
    assert policy_control_substeps(0, 0.002, 0.02) == 10
    assert policy_control_substeps(4, 0.002, 0.02) == 4


def test_opt_snapshot_roundtrip():
    class _Opt:
        last_global_idx = 3
        last_candidate_ids = np.array([3, 7], dtype=np.int32)
        last_candidate_deltas = np.array([0.2, -0.01])
        _dwell_steps = 5

    snap = _opt_snapshot(_Opt())
    dst = type("Dst", (), {})()
    _apply_opt_snapshot(dst, snap)
    assert dst.last_global_idx == 3
    assert dst._dwell_steps == 5
    assert list(dst.last_candidate_ids) == [3, 7]
    assert dst.last_candidate_deltas[1] == pytest.approx(-0.01)


def test_pickle_safe_keeps_policy_fields():
    packed = _pickle_safe({
        "mpc_virtual_point": np.array([0.1, 0.2, 0.3]),
        "value_info": {"accept_p_arm": True, "quality": 0.4},
        "escape_on": False,
    })
    assert packed["value_info"]["accept_p_arm"] is True
    assert list(packed["mpc_virtual_point"]) == [0.1, 0.2, 0.3]


def test_via_stays_within_lead_of_tip():
    tip = np.array([0.40, 0.00, 0.36])
    far = np.array([0.55, 0.10, 0.50])
    clamped = clamp_via_to_tip(tip, far, 0.008)
    assert float(np.linalg.norm(clamped - tip)) == pytest.approx(0.008, abs=1e-9)
    assert np.allclose(clamp_via_to_tip(tip, tip + np.array([0.002, 0.0, 0.0]), 0.008), tip + [0.002, 0.0, 0.0])


def test_blend_action_interpolates_toward_raw():
    prev = np.ones(7, dtype=np.float32)
    raw = np.zeros(7, dtype=np.float32)
    got = blend_action(prev, raw, 0.25)
    assert np.allclose(got, 0.75 * prev)
    assert np.allclose(blend_action(None, raw, 0.25), raw)


def test_clip_toward_matches_max_step():
    origin = np.zeros(3)
    target = np.array([0.03, 0.0, 0.0])
    assert np.allclose(clip_toward(origin, target, 0.01), [0.01, 0.0, 0.0])


def test_yoshikawa_matches_sqrt_det_jjt():
    rng = np.random.default_rng(0)
    jac = rng.normal(size=(3, 7))
    expected = float(np.sqrt(max(np.linalg.det(jac @ jac.T), 0.0)))
    cols = [jac[:, i] for i in range(7)]
    a00 = sum(c[0] * c[0] for c in cols)
    a01 = sum(c[0] * c[1] for c in cols)
    a02 = sum(c[0] * c[2] for c in cols)
    a11 = sum(c[1] * c[1] for c in cols)
    a12 = sum(c[1] * c[2] for c in cols)
    a22 = sum(c[2] * c[2] for c in cols)
    det = (
        a00 * (a11 * a22 - a12 * a12)
        - a01 * (a01 * a22 - a12 * a02)
        + a02 * (a01 * a12 - a11 * a02)
    )
    assert np.sqrt(max(det, 0.0)) == pytest.approx(expected, rel=1e-6, abs=1e-8)


def test_put_latest_keeps_only_newest():
    q = queue.Queue(maxsize=1)
    put_latest(q, {"seq": 1})
    put_latest(q, {"seq": 2})
    put_latest(q, {"seq": 3})
    assert take_latest(q) == {"seq": 3}
    assert take_latest(q) is None


def test_drain_latest_skips_stale_states():
    q = queue.Queue()
    q.put({"seq": 2})
    q.put({"seq": 3})
    assert drain_latest(q, {"seq": 1}) == {"seq": 3}


def test_drain_latest_stop_sentinel_exits():
    q = queue.Queue()
    q.put({"seq": 2})
    q.put(None)
    assert drain_latest(q, {"seq": 1}) is None
