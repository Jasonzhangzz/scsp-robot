"""Contact-patch recovery without compiling or invoking an acados solver."""

import numpy as np
import pytest

from planning.mlqp_point import LambdaContactControlOptimizer


@pytest.fixture
def optimizer():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.sample_point = np.array([
        [0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.10, 0.0, 0.0]])
    opt.sample_geodesic = np.linalg.norm(
        opt.sample_point[:, None] - opt.sample_point[None, :], axis=2)
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
    return opt


def test_neighbour_sample_jitter_exhausts_one_patch(optimizer):
    for step in range(30):
        optimizer.note_contact_progress(step % 2, 10.0, active=True)

    assert optimizer._dwell_idx == 0
    assert optimizer.contact_switch_confidence <= 0.05
    assert not optimizer.lock_contact_patch
    assert set(optimizer._blocked_contact_indices) == {0, 1}
    assert optimizer._contact_patch_failures == {0: 1, 1: 1}


def test_passive_progress_does_not_erase_failed_approaches(optimizer):
    def observe(cost, active=True):
        return optimizer.note_contact_progress(
            0, cost, active=active, gamma=0.5, min_dwell_steps=1)

    observe(10.0)
    assert observe(10.0) == pytest.approx(0.5)
    assert observe(9.0, active=False) == pytest.approx(0.5)
    assert observe(8.0, active=False) == pytest.approx(0.5)
    assert optimizer._dwell_steps == 1
    assert observe(8.0) == pytest.approx(0.25)
    assert optimizer._dwell_steps == 2
    # Progress during a new active attempt restores trust gradually.
    assert observe(7.9) == pytest.approx(0.37)
    assert optimizer._dwell_steps == 0


def test_time_decay_keeps_discount_when_sample_hops(optimizer):
    optimizer.note_contact_progress(0, 0.05, active=False, time_decay=True,
                                    gamma=0.5, min_dwell_steps=1)
    conf = optimizer.note_contact_progress(
        2, 0.05, active=False, time_decay=True, gamma=0.5, min_dwell_steps=1)
    assert conf == pytest.approx(0.5)
    assert optimizer._dwell_idx == 2
    assert optimizer._dwell_steps >= 1


def test_air_hover_time_decay_discounts_confidence(optimizer):
    optimizer.note_contact_progress(0, 0.05, active=False, time_decay=True,
                                    gamma=0.5, min_dwell_steps=2)
    assert optimizer.contact_switch_confidence == 1.0
    optimizer.note_contact_progress(0, 0.05, active=False, time_decay=True,
                                    gamma=0.5, min_dwell_steps=2)
    optimizer.note_contact_progress(0, 0.05, active=False, time_decay=True,
                                    gamma=0.5, min_dwell_steps=2)
    assert optimizer.contact_switch_confidence == pytest.approx(0.5)
    assert optimizer._dwell_steps >= 2


def test_new_geodesic_patch_gets_a_fresh_attempt(optimizer):
    optimizer.note_contact_progress(0, 10.0, active=True)
    optimizer.note_contact_progress(1, 10.0, active=True,
                                    gamma=0.5, min_dwell_steps=1)
    assert optimizer.contact_switch_confidence == pytest.approx(0.5)

    optimizer.note_contact_progress(2, 10.0, active=True)
    assert optimizer._dwell_idx == 2
    assert optimizer._dwell_steps == 0
    assert optimizer.contact_switch_confidence == pytest.approx(1.0)


def test_repeated_failed_neighbourhood_has_bounded_backoff(optimizer):
    cooldowns = []
    for visit in range(5):
        # Travel to a different patch; then simulate expiry before a revisit.
        optimizer.note_contact_progress(2, 10.0, active=False)
        optimizer._blocked_contact_indices.clear()
        idx = visit % 2
        for _ in range(3):
            optimizer.note_contact_progress(
                idx, 10.0, active=True, gamma=0.1, min_dwell_steps=1)
        cooldowns.append(optimizer._blocked_contact_indices[idx])
        assert optimizer._contact_patch_failures[0] == visit + 1
        assert optimizer._contact_patch_failures[1] == visit + 1

        # Continuing to return a blocked patch is one failed visit, not a
        # fresh failure on every control frame.
        optimizer.note_contact_progress(
            idx, 10.0, active=True, gamma=0.1, min_dwell_steps=1)
        assert optimizer._contact_patch_failures[idx] == visit + 1

    assert cooldowns == [40, 80, 160, 320, 320]


def test_tight_merge_radius_does_not_blend_foot_into_back(optimizer):
    back, foot = 2, 0
    optimizer.note_contact_progress(back, 0.04, active=False, merge_radius=0.01)
    for _ in range(8):
        optimizer.note_contact_progress(
            foot, 0.05, active=True, gamma=0.5, min_dwell_steps=2,
            improve_eps=0.002, dead_increment=True, merge_radius=0.01)
    assert optimizer._dwell_idx == foot
    assert foot in optimizer._blocked_contact_indices
    assert back not in optimizer._blocked_contact_indices


def test_position_noise_does_not_reset_foot_failures(optimizer):
    optimizer.note_contact_progress(0, 0.040, active=True, improve_eps=0.002)
    optimizer.note_contact_progress(0, 0.0395, active=True, improve_eps=0.002)
    assert optimizer._dwell_steps == 1
    assert optimizer.note_contact_progress(
        0, 0.037, active=True, improve_eps=0.002) == 1.0
    assert optimizer._dwell_steps == 0


def test_pose_improve_recovers_confidence_gradually(optimizer):
    optimizer.note_contact_progress(0, 10.0, active=True, gamma=0.5, min_dwell_steps=1)
    assert optimizer.note_contact_progress(
        0, 10.0, active=True, gamma=0.5, min_dwell_steps=1) == pytest.approx(0.5)
    assert optimizer.note_contact_progress(
        0, 9.0, active=True, gamma=0.5, min_dwell_steps=1) == pytest.approx(0.62)
    assert optimizer.contact_switch_confidence < 1.0


def test_wrong_patch_dead_visits_are_blocked(optimizer):
    """A foot that cannot improve pose must be blacklisted, not the back."""
    back, foot = 2, 0
    optimizer.note_contact_progress(back, 4.0, active=False)
    for _ in range(8):
        optimizer.note_contact_progress(
            foot, 5.0, active=True, gamma=0.5, min_dwell_steps=2,
            dead_increment=True)
    assert optimizer._dwell_idx == foot
    assert foot in optimizer._blocked_contact_indices
    assert back not in optimizer._blocked_contact_indices
