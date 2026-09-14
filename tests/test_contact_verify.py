"""Regression tests for physical contact evidence in the fingertip policy."""

from examples.mpc.fingertips.test.test_0902 import ContactValueTracker


def test_verify_does_not_enter_from_a_virtual_near_miss():
    tracker = ContactValueTracker(window_size=5, confirm_steps=5,
                                  min_hold_steps=30, release_steps=8)
    for _ in range(12):
        value, _ = tracker.update_verify(1.0, 0.001, physical_contact=False)
        assert value == 0.0
        assert not tracker.contact_active


def test_verify_requires_a_full_good_window_after_contact():
    tracker = ContactValueTracker(window_size=5, confirm_steps=5,
                                  min_hold_steps=30, release_steps=8)
    for _ in range(4):
        tracker.update_verify(1.0, 0.001, physical_contact=True)
        assert not tracker.contact_active
    # Five good windows, rather than five individual samples, are required.
    for _ in range(5):
        tracker.update_verify(1.0, 0.001, physical_contact=True)
    assert tracker.contact_active
    assert tracker.verify > 0.0


def test_verify_ignores_mediocre_nearest_quality():
    tracker = ContactValueTracker(window_size=5, confirm_steps=1,
                                  min_hold_steps=0, release_steps=2)
    for _ in range(8):
        value, _ = tracker.update_verify(0.40, 0.001, physical_contact=True)
        assert value == 0.0
        assert not tracker.contact_active


def test_verify_holds_through_brief_physical_dropouts():
    tracker = ContactValueTracker(window_size=5, confirm_steps=5,
                                  min_hold_steps=30, release_steps=8)
    for _ in range(10):
        tracker.update_verify(1.0, 0.001, physical_contact=True)
    assert tracker.contact_active
    for _ in range(12):
        tracker.update_verify(1.0, 0.001, physical_contact=True)
    before = float(tracker.verify)
    values = []
    for _ in range(3):
        value, _ = tracker.update_verify(1.0, 0.001, physical_contact=False)
        values.append(value)
    assert min(values) > 0.5 * before
    assert tracker.contact_active


def test_verify_survives_a_one_frame_p_arm_reject():
    tracker = ContactValueTracker(window_size=5, confirm_steps=1,
                                  min_hold_steps=0, release_steps=2)
    for _ in range(6):
        tracker.update_verify(1.0, 0.001, physical_contact=True)
    before = float(tracker.verify)
    assert before > 0.0
    tracker.update_values(0.10, 0.50, solver_ok=True, same_patch=False)
    assert tracker.verify == before
    assert tracker.contact_active


def test_switch_discards_old_patch_evidence():
    tracker = ContactValueTracker(window_size=5, confirm_steps=1,
                                  min_hold_steps=0, release_steps=2)
    for _ in range(6):
        tracker.update_verify(1.0, 0.001, physical_contact=True)
    assert tracker.verify > 0.0
    tracker.reset_contact()
    assert tracker.verify == 0.0
    assert not tracker.contact_active
    assert not tracker._target_window
