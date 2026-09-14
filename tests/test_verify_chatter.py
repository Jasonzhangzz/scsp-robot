"""Verify chatter wastes steps and eventually blacklists the patch."""

import numpy as np
import pytest

from examples.mpc.fingertips.test.test_0902 import _verify_is_chatter
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


def test_verify_jump_is_chatter():
    assert _verify_is_chatter(0.0, 0.4)
    assert _verify_is_chatter(0.8, 0.1)
    assert not _verify_is_chatter(0.4, 0.5)
    assert _verify_is_chatter(0.4, 0.45, prev_accept=True, accept_now=False)


def test_verify_chatter_blacklists_the_patch(optimizer):
    for _ in range(8):
        optimizer.note_contact_progress(
            0, 0.05, active=True, dead_increment=True, time_decay=True,
            gamma=0.5, min_dwell_steps=2, block_cycles=20, block_radius=0.03)
    assert optimizer.contact_switch_confidence <= 0.05
    assert 0 in optimizer._blocked_contact_indices
