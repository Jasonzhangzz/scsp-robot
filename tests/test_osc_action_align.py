"""OSC increment residuals: press drops lateral, orbit keeps it."""

import numpy as np

from examples.mpc.franka.ik2.contact_frames import remaining_along_action


def test_press_drops_lateral_drift():
    p0 = np.array([0.0, 0.0, 0.0])
    action = np.array([0.005, 0.0, 0.0])
    p_curr = np.array([0.001, 0.003, 0.0])
    remain = remaining_along_action(p_curr, p0, action)
    np.testing.assert_allclose(remain, [0.004, 0.0, 0.0], atol=1e-6)


def test_blocked_orbit_keeps_inward_lateral():
    p0 = np.array([0.06, 0.0, 0.04])
    action = np.array([0.0, 0.005, 0.0])
    p_curr = np.array([0.055, 0.001, 0.04])
    remain = remaining_along_action(p_curr, p0, action, keep_lateral=True)
    np.testing.assert_allclose(remain, [0.005, 0.004, 0.0], atol=1e-6)
    assert float(remain[0]) > 0.0
