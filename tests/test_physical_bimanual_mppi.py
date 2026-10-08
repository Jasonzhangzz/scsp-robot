"""Tests for the physical rollout MPPI boundary (without MJWarp/MuJoCo)."""

from pathlib import Path

import numpy as np

from planning.contact_pair import ContactPair
from planning.physical_bimanual_mppi import (
    CallablePhysicalBackend,
    PhysicalBimanualMPPI,
    PhysicalRollout,
    _lift_progress_signal,
    shared_approach_translation,
)


def _pair():
    return ContactPair(
        contact_points_local=np.array([[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]),
        normals_local=np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
        tangent_basis_local=np.zeros((0,)),
        witness_force_local=np.zeros((2, 3)),
        desired_force_local=np.zeros((2, 3)),
    )


def test_contact_pair_projection_has_orthogonal_tangent_frames():
    pair = _pair()
    projected = pair.project_world([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(projected["contact_points_world"], [[0.95, 2.0, 3.0], [1.05, 2.0, 3.0]])
    for normal, basis in zip(projected["normals_world"], projected["tangent_basis_world"]):
        np.testing.assert_allclose(np.linalg.norm(normal), 1.0)
        np.testing.assert_allclose(basis.T @ normal, np.zeros(2), atol=1.0e-7)
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1.0e-7)


def test_physical_mppi_uses_backend_object_state_and_saves_trajectory(tmp_path: Path):
    calls = []

    def rollout(controls, contact_pair, target_pos, target_quat):
        del contact_pair, target_pos, target_quat
        calls.append(controls.copy())
        horizon = controls.shape[0]
        # Deliberately make object state unrelated to EE deltas.  The planner
        # must return this physical trace rather than integrating EE controls.
        object_pose = np.tile(np.array([0.0, 0.0, 0.37, 0.0, 0.0, 0.0]), (horizon, 1))
        ee_pose = np.zeros((horizon, 12))
        contact = np.ones((horizon, 2), dtype=bool)
        force = np.ones((horizon, 2))
        drift = np.zeros((horizon, 2))
        stage = np.sum(controls ** 2, axis=1)
        return PhysicalRollout(object_pose, ee_pose, contact, force, drift, stage, 0.0, float(stage.sum()))

    planner = PhysicalBimanualMPPI(
        CallablePhysicalBackend(rollout),
        horizon=4,
        samples=6,
        iterations=2,
        noise_sigma=0.02,
        seed=4,
    )
    output_path = tmp_path / "trajectory.npz"
    result = planner.plan_once(_pair(), [0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0], trajectory_path=output_path)
    assert len(calls) == 12
    assert result["ee_delta_pose"].shape == (4, 12)
    assert result["object_pose_se3"].shape == (4, 6)
    np.testing.assert_allclose(result["object_pose_se3"][:, 2], 0.37)
    assert output_path.exists()
    saved = np.load(output_path)
    assert saved["ee_delta_pose"].shape == (4, 12)
    assert saved["object_pose_se3"].shape == (4, 6)
    assert saved["time"].shape == (4,)


def test_physical_mppi_shifts_warm_start_horizon():
    def rollout(controls, *_):
        horizon = controls.shape[0]
        zeros = np.zeros((horizon, 6))
        return PhysicalRollout(
            zeros,
            np.zeros((horizon, 12)),
            np.ones((horizon, 2), dtype=bool),
            np.ones((horizon, 2)),
            np.zeros((horizon, 2)),
            np.sum(controls ** 2, axis=1),
            0.0,
            float(np.sum(controls ** 2)),
        )

    planner = PhysicalBimanualMPPI(CallablePhysicalBackend(rollout), horizon=3, samples=3, iterations=1, seed=0)
    planner.plan_once(_pair(), [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert planner.u_mean.shape == (3, 12)
    np.testing.assert_allclose(planner.u_mean[-1], 0.0)


def test_shared_approach_finishes_both_arms_together():
    current = np.zeros((2, 3))
    targets = np.array([[0.40, 0.0, 0.0], [0.08, 0.0, 0.0]])
    delta = shared_approach_translation(current, targets, steps=4, translation_limit=0.05)
    np.testing.assert_allclose(np.linalg.norm(delta[0]), 0.05)
    left_steps = 0.40 / np.linalg.norm(delta[0])
    right_steps = 0.08 / np.linalg.norm(delta[1])
    np.testing.assert_allclose(left_steps, right_steps)

    near_targets = np.array([[0.04, 0.0, 0.0], [0.02, 0.0, 0.0]])
    near_delta = shared_approach_translation(current, near_targets, steps=4, translation_limit=0.05)
    np.testing.assert_allclose(near_delta, near_targets / 4.0)


def test_lift_progress_requires_bilateral_physical_contact():
    target = np.array([1.0, 1.0])
    unilateral = _lift_progress_signal(
        0.02, 0.0, 0.06, [True, False], [1.0, 1.0], target, 30.0
    )
    no_contact = _lift_progress_signal(
        0.02, 0.0, 0.06, [False, False], [1.0, 1.0], target, 30.0
    )
    bilateral = _lift_progress_signal(
        0.02, 0.0, 0.06, [True, True], [1.0, 1.0], target, 30.0
    )
    assert unilateral[0] == 0.0
    assert no_contact[0] == 0.0
    assert bilateral[0] > 0.0
    assert bilateral[1] == 1.0
