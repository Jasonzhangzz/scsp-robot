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
    # Beta=1 follows both directions immediately, including frame-to-frame jumps.
    for target in (.2, 1., 0., .8, .1):
        value, _ = tracker.update_verify(tightness=target)
        assert value == target


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


def test_zero_beta_freezes_the_score_until_explicit_reset():
    tracker = ContactValueTracker(beta=0.)
    tracker.verify = .4
    for target in (0., 1., .2):
        assert tracker.update_verify(tightness=target)[0] == .4
    tracker.reset_contact()
    assert tracker.update_verify(tightness=1.)[0] == 0.


@pytest.mark.parametrize('beta', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_beta_is_rejected(beta):
    with pytest.raises(ValueError, match='verify beta'):
        ContactValueTracker(beta=beta)


@pytest.mark.parametrize('beta', [.18, .5, .95, 1.])
def test_verify_rise_and_fall_have_the_same_response(beta):
    rising = ContactValueTracker(beta=beta)
    falling = ContactValueTracker(beta=beta)
    rising.verify, falling.verify = .2, .8
    up, _ = rising.update_verify(tightness=.8)
    down, _ = falling.update_verify(tightness=.2)
    assert (up-.2)/(.8-.2) == pytest.approx(beta)
    assert (.8-down)/(.8-.2) == pytest.approx(beta)
    assert up+down == pytest.approx(1.)
