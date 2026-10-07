"""EE-pose MPPI regression tests (kept at the historical test path)."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from planning.bigrasp_ee_cost import (  # noqa: E402
    evaluate_bimanual_ee_cost,
    integrate_ee_pose,
    numpy_pose_state,
    project_contact_points_world,
    transform_force_vectors_world,
)
from planning.mppi_bigrasp_ee import BimanualEEMPPI  # noqa: E402


def _params():
    return SimpleNamespace(
        mpc_horizon_=4,
        mppi_samples_=24,
        mppi_iterations_=1,
        mppi_init_iterations_=1,
        mppi_lambda_=1.0,
        mppi_noise_sigma_=0.01,
        mppi_noise_decay_=0.9,
        mppi_elite_frac_=0.25,
        mppi_device_="cpu",
        planner_cmd_limit=0.02,
        planner_rotation_delta_limit_=0.1,
        h_=0.01,
        obj_mass_=0.1,
        contact_stiffness=12.5,
        arm_friction=0.9,
        planner_ee_position_weight_=80.0,
        planner_ee_orientation_weight_=2.0,
        planner_force_tracking_weight_=12.0,
        planner_object_target_weight_=30.0,
        planner_object_orientation_weight_=3.0,
        planner_synchronization_weight_=25.0,
        planner_action_weight_=2.0,
        planner_smooth_action_weight_=3.0,
        planner_workspace_weight_=50.0,
        planner_contact_gate_scale_=0.004,
        planner_workspace_lower_=(-1.0, -1.0, 0.0),
        planner_workspace_upper_=(2.0, 1.0, 2.0),
    )


def _state():
    return numpy_pose_state(
        [0.0, 0.0, 0.2],
        [1.0, 0.0, 0.0, 0.0],
        ([-0.1, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0]),
        ([0.1, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0]),
    )


def test_ee_state_and_control_dimensions():
    state = _state()
    assert state.shape == (21,)
    planner = BimanualEEMPPI(_params(), seed=3)
    result = planner.plan_once(
        state,
        [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [0.0, 0.0, 0.26],
        [1.0, 0.0, 0.0, 0.0],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        support_z=0.0,
        hold_mask=(True, False),
        approach_offset=0.02,
    )
    assert result["action"].shape == (12,)
    np.testing.assert_array_equal(result["action"][:6], np.zeros(6))
    assert result["rollout"].shape == (4, 21)
    assert np.all(np.linalg.norm(result["rollout"][:, 3:7], axis=1) > 0.999)
    assert np.all(np.linalg.norm(result["rollout"][:, 10:14], axis=1) > 0.999)
    assert np.all(np.linalg.norm(result["rollout"][:, 17:21], axis=1) > 0.999)


def test_quaternion_integration_and_contact_projection():
    state = torch.as_tensor(_state())
    action = torch.zeros(12)
    action[3] = np.pi / 2.0
    action[9] = -np.pi / 2.0
    integrated = integrate_ee_pose(state, action)
    assert torch.allclose(torch.linalg.vector_norm(integrated[10:14]), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(integrated[17:21]), torch.tensor(1.0), atol=1e-6)

    points = project_contact_points_world([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0], [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
    np.testing.assert_allclose(points.numpy(), [[1.1, 2.0, 3.0], [1.0, 2.2, 3.0]])
    forces = transform_force_vectors_world([1.0, 0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(forces.numpy(), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_unilateral_contact_does_not_drag_surrogate_object():
    planner = BimanualEEMPPI(_params(), seed=0)
    contact_points = [[0.05, 0.0, 0.35], [-0.05, 0.0, 0.35]]
    normals = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    state = numpy_pose_state(
        [0.0, 0.0, 0.35],
        [1.0, 0.0, 0.0, 0.0],
        ([0.036, 0.0, 0.35], [1.0, 0.0, 0.0, 0.0]),
        ([0.20, 0.0, 0.35], [1.0, 0.0, 0.0, 0.0]),
    )
    context = planner.build_context(
        contact_points,
        normals,
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [0.0, 0.0, 0.41],
        [1.0, 0.0, 0.0, 0.0],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        support_z=0.30,
        approach_offset=0.0,
    )
    rollout, _, diagnostics = planner.rollout(
        state,
        torch.zeros((1, 4, 12), dtype=torch.float32),
        context,
    )
    np.testing.assert_allclose(
        rollout[0, :, 0:2].cpu().numpy(),
        np.repeat(state[None, 0:2], 4, axis=0),
        atol=1e-6,
    )
    terminal_gate = diagnostics["terminal"]["bilateral_gate"].detach().cpu().numpy()
    assert float(np.max(terminal_gate)) < 1e-4


def test_contact_gate_requires_tangential_proximity():
    planner = BimanualEEMPPI(_params(), seed=0)
    context = planner.build_context(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        [0.0, 0.0, 0.2],
        [1.0, 0.0, 0.0, 0.0],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        support_z=0.0,
        approach_offset=0.0,
    )
    state = numpy_pose_state(
        [0.0, 0.0, 0.2],
        [1.0, 0.0, 0.0, 0.0],
        ([0.10, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0]),
        ([-0.10, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0]),
    )
    _, diagnostics = evaluate_bimanual_ee_cost(
        torch.as_tensor(state), torch.zeros(12), context
    )
    np.testing.assert_allclose(
        diagnostics["normal_gate"].detach().cpu().numpy(),
        np.ones(2) * 0.9241418,
        atol=1e-5,
    )
    assert float(torch.max(diagnostics["force_gate"])) < 1e-9
