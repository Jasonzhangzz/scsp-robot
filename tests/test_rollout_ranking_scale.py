"""Rollout ranking must keep the switch-scale impulse model.

A previous rollout alignment treated lambda as a 0.75 N force over the
20 ms control interval.  Against the historical Q=50 inertia every
sample then predicted a micrometre-scale x_plus, so ContactValueTracker
always accepted the nearest patch (a local optimum).
--ideal_contact_switch never did this because it ranked with a 10 N
impulse over 50 ms.
"""

import numpy as np

from examples.mpc.fingertips.test.test_0902 import ContactValueTracker


def _one_step_translation(impulse, h, mass_eff=50.0):
    return float(h) * float(impulse) / float(mass_eff)


def test_switch_scale_pose_step_is_informative():
    dx = _one_step_translation(impulse=10.0, h=0.05)
    assert dx > 5e-3


def test_physical_force_scale_collapses_pose_steps():
    physical_impulse = 0.02 * 0.75
    dx = _one_step_translation(impulse=physical_impulse, h=0.02)
    assert dx < 1e-5


def test_other_patch_is_rejected_even_when_costs_match():
    tracker = ContactValueTracker()
    info = tracker.update_values(0.10, 0.10 + 1e-6, solver_ok=True,
                                 candidate_costs=[0.10, 0.10 + 1e-6],
                                 same_patch=False)
    assert not info['accept_p_arm']


def test_same_patch_near_best_is_lazy_accepted():
    tracker = ContactValueTracker()
    info = tracker.update_values(0.10, 0.11, solver_ok=True, same_patch=True,
                                 near_arm=True)
    assert info['accept_p_arm']
    assert info['cost_ok']


def test_same_patch_best_sample_is_accepted_without_near_arm():
    tracker = ContactValueTracker()
    info = tracker.update_values(0.10, 0.10, solver_ok=True, same_patch=True,
                                 is_best_sample=True)
    assert info['accept_p_arm']


def test_same_patch_far_neighbour_is_not_a_lazy_hold():
    tracker = ContactValueTracker()
    info = tracker.update_values(0.10, 0.11, solver_ok=True, same_patch=True,
                                 near_arm=False, is_best_sample=False)
    assert not info['accept_p_arm']
    assert info['cost_ok']


def test_same_patch_much_worse_is_rejected():
    tracker = ContactValueTracker()
    info = tracker.update_values(0.10, 0.50, solver_ok=True,
                                 candidate_costs=[0.10, 0.28, 0.50],
                                 same_patch=True, near_arm=True)
    assert not info['accept_p_arm']
    assert not info['cost_ok']
    assert info['quality'] < 0.40


def test_lazy_hold_survives_cost_noise():
    tracker = ContactValueTracker()
    first = tracker.update_values(0.10, 0.11, solver_ok=True, same_patch=True,
                                  near_arm=True)
    assert first['accept_p_arm']
    # 25% worse is above the 20% enter margin but inside the 40% release.
    held = tracker.update_values(0.10, 0.13, solver_ok=True, same_patch=True,
                                 near_arm=True)
    assert held['accept_p_arm']


def test_stagnant_steps_tighten_p_arm_accept_margin():
    tracker = ContactValueTracker()
    loose = tracker.update_values(0.10, 0.11, solver_ok=True, same_patch=True,
                                  near_arm=True, stagnant_steps=0)
    assert loose['accept_p_arm']
    tight = ContactValueTracker().update_values(
        0.10, 0.11, solver_ok=True, same_patch=True, near_arm=True,
        stagnant_steps=20, margin_gamma=0.8)
    assert not tight['accept_p_arm']
    assert tight['accept_scale'] < 0.05


def test_decaying_confidence_abandons_p_arm():
    tracker = ContactValueTracker()
    first = tracker.update_values(0.10, 0.10, solver_ok=True, same_patch=True,
                                  near_arm=True, confidence=1.0)
    assert first['accept_p_arm']
    abandoned = tracker.update_values(0.10, 0.10, solver_ok=True, same_patch=True,
                                      near_arm=True, confidence=0.85)
    assert not abandoned['accept_p_arm']
    assert not tracker._holding_p_arm


def test_reset_arm_does_not_decay_verify():
    tracker = ContactValueTracker()
    tracker.verify = 0.7
    tracker.good_streak = 4
    tracker.reset_arm(3)
    tracker.reset_arm(7)
    assert tracker.verify == 0.7
    assert tracker.good_streak == 4
