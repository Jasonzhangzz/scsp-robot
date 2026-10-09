"""Real CPU Warp contact-force LM, checked against independent rigid dynamics."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from planning.lambda_batched_lm import BatchedLambdaLM, ensure_kernels


def _pairs(count=2):
    points = np.tile([[[-.04, 0., 0.], [.04, 0., 0.]]], (count, 1, 1))
    if count > 1:
        points[1, :, 2] = .02
    normals = np.tile([[[1., 0., 0.], [-1., 0., 0.]]], (count, 1, 1))
    return points, normals


def _pose_cost(pose, target, pos_weight=1., ori_weight=.0005):
    pose, target = np.asarray(pose), np.asarray(target)
    q = pose[3:] / np.linalg.norm(pose[3:])
    qd = target[3:] / np.linalg.norm(target[3:])
    return pos_weight * np.sum((pose[:3] - target[:3])**2) + ori_weight * (1 - np.clip(q @ qd, -1, 1)**2)


def _qmul(a, b):
    return np.r_[a[0] * b[0] - a[1:] @ b[1:],
                 a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])]


def _predict(current, points, vectors_local, inertia, dt, velocity, external, bias=None,
             support_row=None, support_gap=0., support_stiffness=0.2):
    """Independent mixed world-linear/body-angular backward-Euler prediction."""
    q = current[3:] / np.linalg.norm(current[3:])
    rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    generalized = np.r_[external[:3], rotation.T @ external[3:]]
    generalized -= np.zeros(6) if bias is None else bias
    generalized += np.r_[rotation @ vectors_local.sum(axis=0),
                          np.cross(points, vectors_local).sum(axis=0)]
    next_velocity = velocity + dt * np.linalg.solve(inertia, generalized)
    if support_row is not None:
        if support_gap < 0:
            from models.comfree_contact import row_mass, row_acceleration
            inverse = np.linalg.inv(inertia)
            mass = row_mass(np.trace(inverse[:3, :3])/3, support_gap)
            force = max(float(mass*row_acceleration(support_row @ next_velocity, support_gap, dt,
                                                   support_stiffness, .001)), 0)
            next_velocity += dt*(inverse @ support_row)*force
    next_q = _qmul(q, np.r_[1., .5 * dt * next_velocity[3:]])
    next_q /= np.linalg.norm(next_q)
    return np.r_[current[:3] + dt * next_velocity[:3], next_q], next_velocity


def _inputs():
    return dict(current_x=np.array([0., 0., 0., 1., 0., 0., 0.]),
                x_d=np.array([0., 0., .05, 1., 0., 0., 0.]),
                prediction_dt=.12, object_inertia=np.diag([.2, .2, .2, .004, .004, .004]))


def test_lm_actually_improves_pose_and_selected_outputs_are_same_candidate():
    points, normals = _pairs(3)
    points[2, :, 1] = .015
    solver = BatchedLambdaLM(device="cpu", iterations=30, max_contact_force=3.)
    result = solver.solve(points, normals, **_inputs())
    batch, selected = result["batch"], result["selected_index"]
    assert result["solver_converged"]
    assert result["pose_cost"] < .25 * result["current_pose_cost"]
    assert result["baseline_reduction"] > 0
    assert selected == np.argmin(batch.costs.numpy())
    np.testing.assert_array_equal(result["points"], points[selected].astype(np.float32))
    np.testing.assert_array_equal(result["normals"], normals[selected].astype(np.float32))
    np.testing.assert_array_equal(result["control_input"].reshape(-1), batch.forces.numpy()[selected])
    np.testing.assert_array_equal(result["x_plus_opt"], batch.poses.numpy()[selected])
    np.testing.assert_array_equal(result["velocity_next"], batch.velocities.numpy()[selected])
    assert result["cost"] == pytest.approx(batch.costs.numpy()[selected])
    current_cost = _pose_cost(_inputs()["current_x"], _inputs()["x_d"])
    pose_costs = np.array([_pose_cost(p, _inputs()["x_d"]) for p in batch.poses.numpy()])
    np.testing.assert_allclose(batch.pose_costs.numpy(), pose_costs, rtol=2e-5, atol=1e-9)
    np.testing.assert_allclose(batch.reductions.numpy(), current_cost - pose_costs, rtol=2e-5, atol=1e-9)


def test_force_projection_and_lm_respect_friction_pyramid_and_euclidean_caps():
    points, normals = _pairs(2)
    mu, limit = .7, 1.2
    solver = BatchedLambdaLM(device="cpu", iterations=12, mu=mu, max_contact_force=limit)
    result = solver.solve(points, normals, initial_forces=np.array([
        [[1000., 1000., -1000.], [1000., -1000., 1000.]],
        [[-20., 15., 15.], [-20., 15., 15.]],
    ]), **_inputs())
    forces = result["batch"].forces.numpy().reshape(-1, 2, 3)
    normal = forces[:, :, 0]
    assert np.all(normal >= 0)
    assert np.all(normal <= limit / np.sqrt(1 + 2 * mu**2) + 1e-6)
    assert np.all(np.abs(forces[:, :, 1:]) <= mu * normal[:, :, None] + 1e-6)
    assert np.all(np.linalg.norm(forces, axis=2) <= limit + 1e-6)
    # Tangential LM proposals can grow compressive normal force from rest.
    lifting = BatchedLambdaLM(device="cpu", iterations=30, mu=.9, max_contact_force=3.)
    lifted = lifting.solve(points[:1], normals[:1], initial_forces=np.zeros((1, 2, 3)), **_inputs())
    assert lifted["pose_reduction"] > .001
    assert np.max(np.abs(lifted["control_input"][:, 1:])) > .1
    assert np.min(lifted["control_input"][:, 0]) > .1


def test_prediction_uses_full_coupled_inertia_and_body_angular_quaternion_update():
    points, normals = _pairs(1)
    points[0, 0, 1:] = [.02, .015]
    points[0, 1, 1:] = [-.01, .025]
    quat = Rotation.from_euler("xyz", [20., 30., 40.], degrees=True).as_quat()[[3, 0, 1, 2]]
    current = np.r_[[.1, -.2, .3], quat]
    inertia = np.diag([.2, .23, .26, .003, .004, .005])
    inertia[0, 4] = inertia[4, 0] = .006
    inertia[1, 3] = inertia[3, 1] = -.004
    inertia[2, 5] = inertia[5, 2] = .003
    assert np.linalg.eigvalsh(inertia).min() > 0
    dt, velocity = .08, np.array([.1, -.03, .02, .2, -.1, .15])
    external = np.array([.3, -.5, -1.7, .02, -.03, .01])
    bias = np.array([.02, .03, -.04, .001, .002, -.001])
    initial = np.array([[[1.3, 0., 0.], [.7, 0., 0.]]])
    target, _ = _predict(current, points[0], initial[0, :, :1] * normals[0],
                          inertia, dt, velocity, external, bias)
    solver = BatchedLambdaLM(device="cpu", iterations=8, mu=0.)
    result = solver.solve(points, normals, current_x=current, x_d=target, prediction_dt=dt,
                          object_inertia=inertia, v_obj=velocity, tau_o=external,
                          bias_wrench=bias, initial_forces=initial)
    expected_pose, expected_velocity = _predict(current, points[0], result["force_vectors_local"],
                                                inertia, dt, velocity, external, bias)
    np.testing.assert_allclose(result["velocity_next"], expected_velocity, rtol=3e-6, atol=2e-7)
    np.testing.assert_allclose(result["x_plus_opt"], expected_pose, rtol=3e-6, atol=2e-7)
    assert np.linalg.norm(result["x_plus_opt"][3:]) == pytest.approx(1., abs=1e-6)
    _, diagonal_velocity = _predict(current, points[0], result["force_vectors_local"],
                                    np.diag(np.diag(inertia)), dt, velocity, external, bias)
    assert np.linalg.norm(expected_velocity - diagonal_velocity) > .01
    rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    np.testing.assert_allclose(result["contact_force_world"], rotation @ result["force_vectors_local"].sum(0), atol=1e-6)
    np.testing.assert_allclose(result["contact_torque_world"],
                               rotation @ np.cross(points[0], result["force_vectors_local"]).sum(0), atol=1e-6)


def test_default_gravity_matches_offset_com_full_mujoco_inertia():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body pos="0 0 .5"><freejoint/>'
        '<inertial pos=".03 -.02 .01" mass=".3" diaginertia=".002 .003 .004"/>'
        '<geom type="sphere" size=".04" contype="0" conaffinity="0"/>'
        '</body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    data.qpos[3:] = Rotation.from_euler("xyz", [20., -25., 35.], degrees=True).as_quat()[[3, 0, 1, 2]]
    mujoco.mj_forward(model, data)
    inertia = np.empty((6, 6))
    try:
        mujoco.mj_fullM(model, data, inertia)
    except TypeError:
        mujoco.mj_fullM(model, inertia, data.qM)
    assert np.linalg.norm(inertia[:3, 3:]) > 1e-4
    points, normals = _pairs(1)
    dt, velocity = .03, np.array([-.04, .05, .1, 0., 0., 0.])
    current = data.qpos.copy()
    # Default gravity is derived from physical inertia, not the fallback mass.
    solver = BatchedLambdaLM(device="cpu", iterations=5, mass=.7)
    result = solver.solve(points, normals, current_x=current, x_d=current,
        object_inertia=inertia, prediction_dt=dt, v_obj=velocity)
    expected_position = current[:3] + dt * (velocity[:3] + dt * np.array([0., 0., -9.81]))
    np.testing.assert_allclose(result["baseline_pose"][:3], expected_position, atol=2e-7)
    np.testing.assert_allclose(result["baseline_pose"][3:], current[3:], atol=2e-7)
    # Force applied at the displaced free-joint origin is a different wrench.
    wrong = solver.solve(points, normals, current_x=current, x_d=current,
        object_inertia=inertia, prediction_dt=dt, v_obj=velocity,
        tau_o=np.array([0., 0., -.3 * 9.81, 0., 0., 0.]))
    assert np.linalg.norm(wrong["baseline_pose"][3:] - current[3:]) > 1e-3


def test_pose_quality_is_sign_invariant_and_excludes_force_regularization():
    points, normals = _pairs(1)
    solver = BatchedLambdaLM(device="cpu", iterations=20, force_reg_coef=.002,
                             friction_reg_coef=.003, ori_coef=.02)
    inputs = _inputs()
    inputs["x_d"][3:] = Rotation.from_euler("x", .2).as_quat()[[3, 0, 1, 2]]
    first = solver.solve(points, normals, **inputs)
    first_forces = first["control_input"].copy()
    first_cost = first["cost"]
    inputs["current_x"][3:] *= 2.
    inputs["x_d"][3:] *= -3.
    second = solver.solve(points, normals, **inputs)
    np.testing.assert_allclose(second["control_input"], first_forces, atol=5e-5)
    assert second["cost"] == pytest.approx(first_cost, rel=1e-5)
    pose_cost = _pose_cost(second["x_plus_opt"], inputs["x_d"], ori_weight=.02)
    force_cost = .002 * np.sum(second["control_input"]**2) + .003 * np.sum(second["control_input"][:, 1:]**2)
    assert second["cost"] == pytest.approx(pose_cost + force_cost, rel=2e-5)
    assert second["pose_reduction"] == pytest.approx(
        _pose_cost(inputs["current_x"], inputs["x_d"], ori_weight=.02) - pose_cost, abs=1e-8)
    assert second["cost"] >= second["pose_cost"]


def test_current_pose_reduction_and_free_drift_reduction_have_distinct_baselines():
    points, normals = _pairs(1)
    inputs = _inputs()
    inputs["x_d"] = inputs["current_x"].copy()
    result = BatchedLambdaLM(device="cpu", iterations=30).solve(points, normals, **inputs)
    assert result["current_pose_cost"] == pytest.approx(0.)
    assert result["pose_reduction"] <= 1e-9
    assert result["baseline_cost"] > .01
    assert result["baseline_reduction"] > .01
    assert result["baseline_reduction"] - result["pose_reduction"] == pytest.approx(result["baseline_cost"], rel=1e-5)


def test_unilateral_support_prediction_matches_same_free_physics():
    points, normals = _pairs(1)
    inputs = _inputs()
    current, inertia, dt = inputs["current_x"], inputs["object_inertia"], inputs["prediction_dt"]
    velocity = np.array([.1, .03, -.2, .0, .0, .0])
    external = np.array([0., 0., -1.962, 0., 0., 0.])
    support_row = np.array([0., 0., 1., .01, -.02, 0.])
    result = BatchedLambdaLM(device="cpu", iterations=5, force_reg_coef=100.).solve(
        points, normals, **inputs, v_obj=velocity, tau_o=external, support_row=support_row,
        support_gap=-.002, support_stiffness=.2)
    expected, expected_v = _predict(current, points[0], result["force_vectors_local"], inertia,
                                    dt, velocity, external, support_row=support_row, support_gap=-.002)
    baseline, _ = _predict(current, points[0], np.zeros((2, 3)), inertia, dt, velocity,
                           external, support_row=support_row, support_gap=-.002)
    np.testing.assert_allclose(result["x_plus_opt"], expected, atol=1e-6)
    np.testing.assert_allclose(result["velocity_next"], expected_v, atol=1e-6)
    np.testing.assert_allclose(result["baseline_pose"], baseline, atol=1e-6)
    assert baseline[2] > current[2] + dt * (velocity[2] - dt * 9.81)

    # The live selector computes this row and gap directly on Warp.
    wp = ensure_kernels().wp
    device_result = BatchedLambdaLM(device="cpu", iterations=5, force_reg_coef=100.).solve(
        points, normals, **inputs, v_obj=velocity, tau_o=external,
        support_row=wp.array(support_row, dtype=float, device="cpu"),
        support_gap=wp.array([-.002], dtype=float, device="cpu"), support_stiffness=.2)
    np.testing.assert_allclose(device_result["baseline_pose"], baseline, atol=1e-6)


def test_device_batch_masks_invalid_candidates_and_reset_discards_buffers():
    points, normals = _pairs(3)
    wp = ensure_kernels().wp
    solver = BatchedLambdaLM(device="cpu", iterations=10)
    batch = solver.solve_device(wp.array(points.reshape(-1, 3), dtype=wp.vec3, device="cpu"),
                                wp.array(normals.reshape(-1, 3), dtype=wp.vec3, device="cpu"),
                                valid_mask=wp.array([0, 1, 0], dtype=int, device="cpu"), **_inputs())
    assert int(batch.selected.numpy()[0]) == 1
    np.testing.assert_array_equal(batch.valid.numpy(), [0, 1, 0])
    assert np.isposinf(batch.costs.numpy()[[0, 2]]).all()
    assert np.isneginf(batch.reductions.numpy()[[0, 2]]).all()
    with pytest.raises(FloatingPointError, match="all lambda contact candidates"):
        solver.solve(points, normals, valid=[0, 0, 0], **_inputs())
    solver.reset()
    assert solver.buffers is None and solver.graph is None
    assert solver.solve(points, normals, **_inputs())["selected_index"] >= 0


def test_gather_policy_choice_returns_its_control_pose_cost_and_contact_ids():
    points, normals = _pairs(3)
    points[2, :, 1] = .03
    solver = BatchedLambdaLM(device="cpu", iterations=15)
    batch = solver.solve_device(points, normals, **_inputs())
    chosen = solver.gather_selected(2, batch=batch)
    assert chosen["selected_index"] == 2
    np.testing.assert_array_equal(chosen["points"], points[2].astype(np.float32))
    np.testing.assert_array_equal(chosen["control_input"].reshape(-1), batch.forces.numpy()[2])
    np.testing.assert_array_equal(chosen["x_plus_opt"], batch.poses.numpy()[2])
    assert chosen["cost"] == pytest.approx(batch.costs.numpy()[2])
    assert chosen["pose_reduction"] == pytest.approx(batch.reductions.numpy()[2])


@pytest.mark.parametrize("bad_inertia", [np.zeros((6, 6)), np.diag([.2, .2, .2, .004, .004, -.004])])
def test_non_positive_inertia_is_reported_as_physics_error(bad_inertia):
    points, normals = _pairs(1)
    inputs = _inputs()
    inputs["object_inertia"] = bad_inertia
    with pytest.raises(ValueError, match="positive definite"):
        BatchedLambdaLM(device="cpu", iterations=1).solve(points, normals, **inputs)
