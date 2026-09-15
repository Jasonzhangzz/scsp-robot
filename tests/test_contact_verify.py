"""verify_cost follows tightness, not projected p_arm cost or contact flags."""

import pytest

from examples.mpc.fingertips.test.test_0902 import ContactValueTracker


def test_verify_stays_off_while_tightness_is_zero():
    tracker = ContactValueTracker(beta=1.0)
    for _ in range(12):
        value, _ = tracker.update_verify(1.0, 0.001, physical_contact=False)
        assert value == 0.0
        assert not tracker.contact_active


def test_verify_ema_tracks_tightness():
    tracker = ContactValueTracker(beta=1.0, exit_threshold=0.2)
    value, _ = tracker.update_verify(tightness=0.8)
    assert value == 0.8
    assert tracker.contact_active
    # Decay is slower than rise so a one-frame dip cannot slam verify.
    value, _ = tracker.update_verify(tightness=0.2)
    assert 0.2 < value < 0.8
    for _ in range(12):
        value, _ = tracker.update_verify(tightness=0.2)
    assert value == pytest.approx(0.2, abs=0.01)


def test_verify_ignores_projected_quality_and_distance():
    tracker = ContactValueTracker(beta=1.0, exit_threshold=0.2)
    value, _ = tracker.update_verify(0.40, 0.20, physical_contact=True, tightness=0.5)
    assert value == 0.5
    assert tracker.contact_active


def test_verify_holds_when_physical_contact_drops():
    tracker = ContactValueTracker(beta=1.0, exit_threshold=0.2)
    tracker.update_verify(tightness=0.7)
    before = float(tracker.verify)
    values = []
    for _ in range(3):
        value, _ = tracker.update_verify(1.0, 0.001, physical_contact=False, tightness=0.7)
        values.append(value)
    assert min(values) == before
    assert tracker.contact_active


def test_verify_survives_a_one_frame_p_arm_reject():
    tracker = ContactValueTracker(beta=1.0, exit_threshold=0.2)
    tracker.update_verify(tightness=0.6)
    before = float(tracker.verify)
    assert before > 0.0
    tracker.update_values(0.10, 0.50, solver_ok=True, same_patch=False)
    assert tracker.verify == before
    assert tracker.contact_active


def test_switch_discards_old_patch_evidence():
    tracker = ContactValueTracker(beta=1.0, exit_threshold=0.2)
    tracker.update_verify(tightness=0.9)
    assert tracker.verify > 0.0
    tracker.reset_contact()
    assert tracker.verify == 0.0
    assert not tracker.contact_active
    assert not tracker._target_window
