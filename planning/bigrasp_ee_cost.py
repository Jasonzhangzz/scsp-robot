"""Torch cost and geometry helpers for the bimanual EE-pose MPPI planner.

The planner keeps the robot part of its state in a 14 dimensional pose space:
``[left_position, left_quaternion, right_position, right_quaternion]``.  The
control is a pair of 6D SE(3) increments.  This module deliberately contains
no MuJoCo or cuRobo code so the same cost can be evaluated by batched MPPI
rollouts and by small unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:  # Torch is optional for importing the MuJoCo example and its CLI help.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised only in minimal environments
    torch = None
    F = None


STATE_DIM = 21
EE_POSE_DIM = 14
ACTION_DIM = 12


def _as_tensor(value, device, dtype=None):
    if torch is None:
        raise ImportError("Torch is required for the bimanual EE cost")
    if dtype is None:
        dtype = torch.float32
    return torch.as_tensor(value, device=device, dtype=dtype)


def quat_normalize(q, eps=1.0e-8):
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(eps)


def quat_conjugate(q):
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_multiply(q0, q1):
    w0, x0, y0, z0 = q0.unbind(-1)
    w1, x1, y1, z1 = q1.unbind(-1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dim=-1,
    )


def quat_exp(rotvec):
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-7,
        torch.sin(half) / angle.clamp_min(1.0e-7),
        0.5 - angle * angle / 48.0,
    )
    return quat_normalize(torch.cat((torch.cos(half), scale * rotvec), dim=-1))


def quat_rotate(q, v):
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[..., 1:]


def quat_alignment_error(q, target_q):
    q = quat_normalize(q)
    target_q = quat_normalize(target_q)
    return 1.0 - torch.sum(q * target_q, dim=-1).square()


def split_state(state):
    if state.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected state dimension {STATE_DIM}, got {state.shape[-1]}")
    return (
        state[..., 0:3],
        quat_normalize(state[..., 3:7]),
        state[..., 7:10],
        quat_normalize(state[..., 10:14]),
        state[..., 14:17],
        quat_normalize(state[..., 17:21]),
    )


def project_contact_points_world(object_pos, object_quat, contact_points_local):
    """Project fixed object-frame contact points into the world frame."""
    if torch is None:
        raise ImportError("Torch is required for contact projection")
    if isinstance(object_quat, torch.Tensor):
        object_pos = torch.as_tensor(object_pos, device=object_quat.device, dtype=object_quat.dtype)
    else:
        object_pos = torch.as_tensor(object_pos, dtype=torch.float32)
    object_quat = quat_normalize(torch.as_tensor(object_quat, device=object_pos.device, dtype=object_pos.dtype))
    local = torch.as_tensor(contact_points_local, device=object_pos.device, dtype=object_pos.dtype).reshape(2, 3)
    return object_pos.reshape(3) + quat_rotate(object_quat.reshape(4), local)


def transform_force_vectors_world(object_quat, force_vectors_local):
    """Rotate object-frame desired contact forces to world coordinates."""
    if torch is None:
        raise ImportError("Torch is required for force transformation")
    quat = quat_normalize(torch.as_tensor(object_quat, dtype=torch.float32) if not isinstance(object_quat, torch.Tensor) else object_quat)
    local = torch.as_tensor(force_vectors_local, device=quat.device, dtype=quat.dtype).reshape(2, 3)
    return quat_rotate(quat.reshape(4), local)


def integrate_ee_pose(state, action):
    """Apply a world-frame SE(3) increment to both EE poses."""
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected action dimension {ACTION_DIM}, got {action.shape[-1]}")
    obj_p, obj_q, left_p, left_q, right_p, right_q = split_state(state)
    left_p = left_p + action[..., 0:3]
    left_q = quat_multiply(quat_exp(action[..., 3:6]), left_q)
    right_p = right_p + action[..., 6:9]
    right_q = quat_multiply(quat_exp(action[..., 9:12]), right_q)
    return torch.cat((obj_p, obj_q, left_p, quat_normalize(left_q), right_p, quat_normalize(right_q)), dim=-1)


@dataclass
class BimanualEECostConfig:
    dt: float = 0.01
    object_mass: float = 0.01
    object_inertia_pos: float = 40.0
    object_inertia_rot: float = 0.05
    contact_stiffness: float = 12.5
    contact_radius: float = 0.01
    arm_friction: float = 0.9
    ee_position_weight: float = 80.0
    ee_orientation_weight: float = 2.0
    force_tracking_weight: float = 12.0
    wrench_residual_weight: float = 12.0
    object_position_weight: float = 300.0
    object_lateral_position_weight: float = 300.0
    object_orientation_weight: float = 15.0
    synchronization_weight: float = 25.0
    contact_gate_sync_weight: float = 25.0
    action_weight: float = 2.0
    smooth_action_weight: float = 3.0
    workspace_weight: float = 50.0
    contact_gate_scale: float = 0.004
    contact_tangent_scale: float = 0.015
    object_initial_z: float = 0.0
    workspace_lower: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    workspace_upper: tuple[float, float, float] = (2.0, 1.0, 2.0)


def contact_geometry(state, contact_points_local, normals_local, contact_radius=0.01):
    obj_p, obj_q, left_p, left_q, right_p, right_q = split_state(state)
    del left_q, right_q
    contact_points_local = torch.as_tensor(contact_points_local, device=state.device, dtype=state.dtype)
    normals_local = torch.as_tensor(normals_local, device=state.device, dtype=state.dtype)
    contact_local = contact_points_local.reshape((1,) * (state.ndim - 1) + (2, 3))
    normals_local = normals_local.reshape((1,) * (state.ndim - 1) + (2, 3))
    contact_world = obj_p.unsqueeze(-2) + quat_rotate(obj_q.unsqueeze(-2), contact_local)
    inward_world = quat_rotate(obj_q.unsqueeze(-2), normals_local)
    inward_world = inward_world / torch.linalg.vector_norm(inward_world, dim=-1, keepdim=True).clamp_min(1.0e-7)
    outward_world = -inward_world
    ee_pos = torch.stack((left_p, right_p), dim=-2)
    delta = ee_pos - contact_world
    signed_gap = torch.sum(delta * outward_world, dim=-1) - float(contact_radius)
    normal_distance = torch.sum(delta * outward_world, dim=-1)
    tangential_delta = delta - normal_distance.unsqueeze(-1) * outward_world
    return contact_world, inward_world, outward_world, signed_gap, tangential_delta


def evaluate_bimanual_ee_cost(
    state,
    action,
    context,
    previous_action=None,
    terminal=False,
):
    """Evaluate one or many rollout states.

    ``context`` is a dictionary of torch tensors with shape ``(2, 3)`` for
    contact data and ``(3,)``/``(4,)`` for object targets.  The leading batch
    dimensions are taken from ``state``.
    """
    cfg = context["config"]
    if isinstance(cfg, dict):
        cfg = BimanualEECostConfig(**cfg)
    obj_p, obj_q, left_p, left_q, right_p, right_q = split_state(state)
    contact_world, inward_world, outward_world, signed_gap, tangent_delta = contact_geometry(
        state,
        context["contact_points_local"],
        context["normals_local"],
        cfg.contact_radius,
    )
    ee_pos = torch.stack((left_p, right_p), dim=-2)
    target_quat = torch.as_tensor(context["ee_target_quat"], device=state.device, dtype=state.dtype)
    target_quat = target_quat.reshape((1,) * (state.ndim - 1) + (2, 4))

    approach_offset = context.get("approach_offset", 0.0)
    approach_target = contact_world + outward_world * float(approach_offset)
    position_error = torch.sum((ee_pos - approach_target) ** 2, dim=-1)
    orientation_error = torch.stack(
        (
            quat_alignment_error(left_q, target_quat[..., 0, :]),
            quat_alignment_error(right_q, target_quat[..., 1, :]),
        ),
        dim=-1,
    )

    stiffness = max(float(cfg.contact_stiffness), 1.0e-6)
    beta = 1.0 / max(float(cfg.contact_gate_scale), 1.0e-6)
    normal_force = F.softplus(-signed_gap * beta) / beta * stiffness
    # A fingertip displacement tangent to the surface drags the object in the
    # same direction while the normal component supplies the squeeze.  This
    # lets a single fixed contact target produce an upward lift as both EEs
    # move upward under the shared object-target cost.
    tangential_force = tangent_delta * stiffness * 0.25
    tangential_norm = torch.linalg.vector_norm(tangential_force, dim=-1).clamp_min(1.0e-8)
    tangential_limit = float(cfg.arm_friction) * normal_force
    tangential_force = tangential_force * torch.minimum(
        torch.ones_like(tangential_norm), tangential_limit / tangential_norm
    ).unsqueeze(-1)
    predicted_force = inward_world * normal_force.unsqueeze(-1) + tangential_force

    desired_force_local = torch.as_tensor(context["desired_force_local"], device=state.device, dtype=state.dtype)
    desired_force_world = quat_rotate(obj_q.unsqueeze(-2), desired_force_local.reshape((1,) * (state.ndim - 1) + (2, 3)))
    normal_gate = torch.sigmoid(-signed_gap / max(float(cfg.contact_gate_scale), 1.0e-6))
    tangential_distance = torch.linalg.vector_norm(tangent_delta, dim=-1)
    tangent_scale = max(float(cfg.contact_tangent_scale), 1.0e-6)
    tangential_gate = torch.exp(-0.5 * (tangential_distance / tangent_scale).square())
    # A normal penetration by itself is not contact: the fingertip must also
    # be close to the selected point in the tangent plane.  Without this term
    # the surrogate reports force while an EE is centimetres away laterally,
    # so MPPI prefers trajectories that MuJoCo cannot realize.
    force_gate = normal_gate * tangential_gate
    force_error = torch.sum((predicted_force - desired_force_world) ** 2, dim=-1) * force_gate

    total_force = torch.sum(predicted_force, dim=-2)
    lever_arm = contact_world - obj_p.unsqueeze(-2)
    total_torque = torch.sum(torch.cross(lever_arm, predicted_force, dim=-1), dim=-2)
    # Lambda's pair is a force closure: the two desired forces cancel in the
    # horizontal plane.  A single fingertip cannot produce that net wrench, so
    # the residual is large as soon as one arm touches and small when both
    # match.  It is intentionally not multiplied by ``force_gate``; gating it
    # would make a missing contact free.
    desired_net_force = torch.sum(desired_force_world, dim=-2)
    desired_net_torque = torch.sum(torch.cross(lever_arm, desired_force_world, dim=-1), dim=-2)
    wrench_force_residual = torch.sum((total_force - desired_net_force) ** 2, dim=-1)
    wrench_torque_residual = torch.sum((total_torque - desired_net_torque) ** 2, dim=-1)

    target_p = torch.as_tensor(context["target_object_pos"], device=state.device, dtype=state.dtype)
    target_q = torch.as_tensor(context["target_object_quat"], device=state.device, dtype=state.dtype)
    target_p = target_p.reshape((1,) * (state.ndim - 1) + (3,))
    target_q = target_q.reshape((1,) * (state.ndim - 1) + (4,))
    bilateral_gate = torch.prod(force_gate, dim=-1)
    unilateral_gate = torch.sum(force_gate, dim=-1) - 2.0 * bilateral_gate
    contact_gate_sync_error = (force_gate[..., 0] - force_gate[..., 1]).square()
    object_position_error = torch.sum((obj_p - target_p) ** 2, dim=-1)
    object_orientation_error = quat_alignment_error(obj_q, target_q)
    object_lateral_error = torch.sum((obj_p[..., :2] - target_p[..., :2]) ** 2, dim=-1)

    distance_delta = torch.linalg.vector_norm(ee_pos - approach_target, dim=-1)
    synchronization_error = (distance_delta[..., 0] - distance_delta[..., 1]).square()
    action_cost = torch.sum(action.square(), dim=-1)
    if previous_action is None:
        smooth_cost = torch.zeros_like(action_cost)
    else:
        smooth_cost = torch.sum((action - previous_action) ** 2, dim=-1)

    lower = _as_tensor(cfg.workspace_lower, state.device).reshape((1,) * (state.ndim - 1) + (1, 3))
    upper = _as_tensor(cfg.workspace_upper, state.device).reshape((1,) * (state.ndim - 1) + (1, 3))
    ee_pos_violation = F.relu(lower - ee_pos) + F.relu(ee_pos - upper)
    workspace_cost = torch.sum(ee_pos_violation.square(), dim=(-1, -2))

    total = (
        float(cfg.ee_position_weight) * torch.sum(position_error, dim=-1)
        + float(cfg.ee_orientation_weight) * torch.sum(orientation_error, dim=-1)
        + float(cfg.force_tracking_weight) * torch.sum(force_error, dim=-1)
        + float(cfg.wrench_residual_weight) * (wrench_force_residual + wrench_torque_residual)
        + float(cfg.object_position_weight) * object_position_error
        + float(cfg.object_orientation_weight) * object_orientation_error
        + float(cfg.object_lateral_position_weight) * object_lateral_error
        + float(cfg.synchronization_weight) * synchronization_error
        + float(cfg.contact_gate_sync_weight) * contact_gate_sync_error
        + float(cfg.action_weight) * action_cost
        + float(cfg.smooth_action_weight) * smooth_cost
        + float(cfg.workspace_weight) * workspace_cost
    )
    if terminal:
        total = total + 5.0 * (
            float(cfg.object_position_weight) * object_position_error
            + float(cfg.object_orientation_weight) * object_orientation_error
            + float(cfg.object_lateral_position_weight) * object_lateral_error
            + float(cfg.ee_position_weight) * torch.sum(position_error, dim=-1)
        )
    return total, {
        "contact_world": contact_world,
        "predicted_force_world": predicted_force,
        "desired_force_world": desired_force_world,
        "object_force_world": total_force,
        "object_torque_world": total_torque,
        "wrench_force_residual": wrench_force_residual,
        "wrench_torque_residual": wrench_torque_residual,
        "signed_gap": signed_gap,
        "tangential_distance": tangential_distance,
        "normal_gate": normal_gate,
        "tangential_gate": tangential_gate,
        "force_gate": force_gate,
        "bilateral_gate": bilateral_gate,
        "unilateral_gate": unilateral_gate,
        "contact_gate_sync_error": contact_gate_sync_error,
        "position_error": position_error,
        "orientation_error": orientation_error,
        "object_position_error": object_position_error,
        "object_orientation_error": object_orientation_error,
        "object_lateral_error": object_lateral_error,
    }


def numpy_pose_state(object_pos, object_quat, left_pose, right_pose):
    """Build and validate the public 21D state used by MPPI."""
    values = np.hstack(
        [
            np.asarray(object_pos, dtype=np.float32).reshape(3),
            np.asarray(object_quat, dtype=np.float32).reshape(4),
            np.asarray(left_pose[0], dtype=np.float32).reshape(3),
            np.asarray(left_pose[1], dtype=np.float32).reshape(4),
            np.asarray(right_pose[0], dtype=np.float32).reshape(3),
            np.asarray(right_pose[1], dtype=np.float32).reshape(4),
        ]
    )
    values[3:7] /= max(np.linalg.norm(values[3:7]), 1.0e-8)
    values[10:14] /= max(np.linalg.norm(values[10:14]), 1.0e-8)
    values[17:21] /= max(np.linalg.norm(values[17:21]), 1.0e-8)
    return values
