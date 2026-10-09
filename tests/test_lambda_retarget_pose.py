"""Lambda object references obey contact-force bounds and free-joint frames."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from models.comfree_gs_torch import GaussianCloud
from planning.lambda_contact_warp import LambdaContactWarp


POINTS = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]])
NORMALS = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
INERTIA = np.diag([0.2, 0.2, 0.2, 0.003, 0.003, 0.003])


def _optimizer(force_limit=2.0):
    optimizer = LambdaContactWarp(
        cloud=GaussianCloud([[0.0, 0.0, 0.0]], radius=0.04),
        obj_mass=0.2, device="cpu", max_contact_force=force_limit,
    )
    optimizer.last_grasp_result = {
        "contact_points": POINTS.copy(), "contact_normals": NORMALS.copy(),
    }
    return optimizer


def _pose(rotation=None):
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    if rotation is not None:
        quaternion = rotation.as_quat()[[3, 0, 1, 2]]
    return np.r_[np.zeros(3), quaternion]


def test_lambda_reference_is_force_driven_and_bounded():
    optimizer = _optimizer()
    current = _pose()
    goal = current.copy()
    goal[2] = 0.1
    dt = 0.02
    points, normals, predicted, cost, info = optimizer.optimize_control_input(
        goal, current, p_arm=POINTS[::-1], prediction_dt=dt,
        object_inertia=INERTIA,
    )
    np.testing.assert_allclose(points, POINTS[::-1])
    np.testing.assert_allclose(normals, NORMALS[::-1])
    assert 0.0 < predicted[2] < goal[2]
    assert cost < (goal[2] - current[2]) ** 2
    assert np.linalg.norm(predicted[3:7]) == pytest.approx(1.0)
    coefficients = info["contact_forces_local"]
    assert np.all(coefficients[:, 0] >= 0.0)
    assert np.all(np.abs(coefficients[:, 1:]) <= optimizer.mu_arm_obj * coefficients[:, :1] + 1.0e-9)
    assert np.all(np.linalg.norm(coefficients, axis=1) <= optimizer.max_contact_force + 1.0e-9)
    expected_wrench = np.r_[info["contact_force_world"] + [0.0, 0.0, -0.2 * 9.81],
                            info["contact_torque_local"]]
    np.testing.assert_allclose(info["velocity_next"], dt * np.linalg.solve(INERTIA, expected_wrench), atol=1.0e-10)
    np.testing.assert_allclose(predicted[:3], current[:3] + dt * info["velocity_next"][:3])
    np.testing.assert_array_equal(optimizer.last_grasp_result["x_plus_opt"], predicted)


def test_lambda_zero_velocity_hold_balances_world_gravity_after_rotation():
    optimizer = _optimizer(force_limit=10.0)
    rotation = Rotation.from_euler("y", 90.0, degrees=True)
    current = _pose(rotation)
    _, _, predicted, _, info = optimizer.optimize_control_input(
        current, current, p_arm=POINTS, r_obj_to_world=rotation.as_matrix(),
        prediction_dt=0.02, object_inertia=INERTIA,
    )
    np.testing.assert_allclose(predicted, current, atol=2.0e-7)
    np.testing.assert_allclose(info["contact_force_world"], [0.0, 0.0, 0.2 * 9.81], atol=1.0e-4)
    np.testing.assert_allclose(rotation.as_matrix() @ info["contact_force_local"], info["contact_force_world"])


def test_lambda_uses_measured_velocity_and_integrates_body_angular_velocity():
    optimizer = _optimizer(force_limit=1.0e-6)
    rotation = Rotation.from_euler("xyz", [20.0, 30.0, 40.0], degrees=True)
    current = _pose(rotation)
    velocity = np.array([0.1, -0.2, 0.3, 0.3, -0.2, 0.1])
    dt = 0.01
    predicted = optimizer.optimize_control_input(
        current, current, tau_o=np.zeros(6), p_arm=POINTS, v_obj=velocity,
        prediction_dt=dt, object_inertia=INERTIA,
    )[2]
    np.testing.assert_allclose(predicted[:3], dt * velocity[:3], atol=1.0e-8)
    # q_next = q * [1, 0.5 dt omega_body], normalized.
    increment = np.r_[0.5 * dt * velocity[3:], 1.0]
    expected_xyzw = (rotation * Rotation.from_quat(increment)).as_quat()
    np.testing.assert_allclose(predicted[3:7], expected_xyzw[[3, 0, 1, 2]], atol=1.0e-8)


def test_lambda_rotational_target_generates_contact_torque():
    optimizer = _optimizer(force_limit=2.0)
    current = _pose()
    target = _pose(Rotation.from_euler("z", 20.0, degrees=True))
    result = optimizer.optimize_control_input(
        target, current, p_arm=POINTS, prediction_dt=0.02,
        object_inertia=INERTIA, return_world_torque=True,
    )
    _, _, predicted, _, info, torque_world = result
    assert predicted[6] > 0.0
    assert info["contact_torque_local"][2] > 0.0
    np.testing.assert_allclose(torque_world, info["contact_torque_world"])
    assert abs(float(predicted[3:7] @ target[3:7])) > abs(float(current[3:7] @ target[3:7]))


@pytest.mark.parametrize("dt,inertia,error", [
    (0.0, INERTIA, "prediction_dt"),
    (0.01, np.zeros((6, 6)), "object_inertia"),
])
def test_lambda_rejects_invalid_dynamics_parameters(dt, inertia, error):
    with pytest.raises(ValueError, match=error):
        _optimizer().optimize_control_input(
            _pose(), _pose(), p_arm=POINTS, prediction_dt=dt,
            object_inertia=inertia,
        )
