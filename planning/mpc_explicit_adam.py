"""Adam solvers for BigRasp lambda contacts and fingertip / joint MPC.

``--mpc`` uses DexForge ``compile_fast_step`` for the rollout.  CPU unit tests
can still drive the torch ``ComfreeGSModel`` 6-D tip stepper.  Costs lock each
arm to its assigned contact and penalize arm-arm proximity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from models.comfree_gs_torch import (
    GS_SOFTMIN_TAU,
    TIP_QUERY_RADIUS,
    ComfreeGSModel,
    GaussianCloud,
    gs_sphere_features,
)


PLAN_ONCE_KEYS = (
    "action",
    "u_traj",
    "rollout_q",
    "ctrl",
    "cost",
    "cost_opt",
    "solve_time",
    "solver_backend",
    "sol_guess",
    "contact_mask",
    "normal_force",
)


def _device(name=None):
    # Default CPU: CUDA + MuJoCo's OpenGL viewer commonly SIGSEGV on first
    # backward of a long rollout.  Pass --mppi-device cuda:0 to override.
    if name:
        try:
            device = torch.device(str(name).strip())
            if device.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return device
        except (RuntimeError, ValueError, TypeError):
            pass
    return torch.device("cpu")


def _normalize(vec, eps=1.0e-8):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return vec / norm


@dataclass
class BigraspGSCostWeights:
    contact_attract: float = 160.0
    contact_depth: float = 40.0
    penetration: float = 120.0
    object_position: float = 0.0
    object_lateral: float = 0.0
    object_orientation: float = 0.0
    action: float = 2.0
    smooth: float = 3.0
    sync: float = 25.0
    force: float = 0.0
    swap: float = 120.0
    inter_arm: float = 16000.0
    tip_sep: float = 80.0
    tip_sep_min: float = 0.05
    capsule_radius: float = 0.045
    wrist_radius: float = 0.035
    tip_sphere_radius: float = 0.01
    capsule_margin: float = 0.02
    gate_scale: float = 0.04
    target_phi: float = 0.0
    penetration_limit: float = 0.002
    approach_offset: float = 0.0
    query_mm: float = 1000.0
    terminal: float = 2.0
    verify_1: float = 1.0
    verify_2: float = 1.0


def _contact_frames(obj_pos, obj_quat, context, weights):
    if context.get("contact_points_world") is not None:
        contact_world = context["contact_points_world"]
        outward = context.get("outward_world")
        if outward is None:
            outward = torch.zeros_like(contact_world)
            outward[..., 2] = 1.0
    else:
        contact_local = context["contact_points_local"]
        normals_local = context["normals_local"]
        contact_world = _quat_rotate(obj_quat.unsqueeze(-2), contact_local) + obj_pos.unsqueeze(-2)
        outward = -_quat_rotate(obj_quat.unsqueeze(-2), normals_local)
    attract_target = contact_world + outward * float(weights.approach_offset)
    return contact_world, outward, attract_target


def _segment_distance(a0, a1, b0, b1):
    """Distance between two capsules' axes (..., 3)."""
    u = a1 - a0
    v = b1 - b0
    w = a0 - b0
    uu = torch.sum(u * u, dim=-1).clamp_min(1.0e-8)
    vv = torch.sum(v * v, dim=-1).clamp_min(1.0e-8)
    uv = torch.sum(u * v, dim=-1)
    uw = torch.sum(u * w, dim=-1)
    vw = torch.sum(v * w, dim=-1)
    denom = (uu * vv - uv * uv).clamp_min(1.0e-8)
    s = torch.clamp((uv * vw - vv * uw) / denom, 0.0, 1.0)
    t = torch.clamp((uu * vw - uv * uw) / denom, 0.0, 1.0)
    diff = w + s.unsqueeze(-1) * u - t.unsqueeze(-1) * v
    return torch.linalg.vector_norm(diff, dim=-1)


def _pack_query_capsules(capsules):
    """``(..., 6, 3)`` forearm/wrist/tip per arm, or the older 4-point layout."""
    if capsules.shape[-2] >= 6:
        return capsules[..., :6, :]
    left_fa, left_tip = capsules[..., 0, :], capsules[..., 1, :]
    right_fa, right_tip = capsules[..., 2, :], capsules[..., 3, :]
    return torch.stack((left_fa, left_tip, left_tip, right_fa, right_tip, right_tip), dim=-2)


def _inter_arm_sphere_cost(capsules, weights):
    """Left/right query-sphere pairs plus the forearm-to-tip capsules, in mm^2."""
    pts = _pack_query_capsules(capsules)
    mm = float(weights.query_mm)
    margin = float(weights.capsule_margin)
    radii = (
        float(weights.capsule_radius),
        float(weights.wrist_radius),
        float(weights.tip_sphere_radius),
    )
    inter = torch.zeros(pts.shape[:-2], device=pts.device, dtype=pts.dtype)
    for left in range(3):
        for right in range(3):
            dist = torch.linalg.vector_norm(pts[..., left, :] - pts[..., right + 3, :], dim=-1)
            inter = inter + torch.clamp(radii[left] + radii[right] + margin - dist, min=0.0) ** 2
    gap = _segment_distance(pts[..., 0, :], pts[..., 2, :], pts[..., 3, :], pts[..., 5, :])
    inter = inter + torch.clamp(2.0 * radii[0] + margin - gap, min=0.0) ** 2
    return inter * (mm * mm)


def evaluate_bigrasp_gs_cost(states, extras, cmd_traj, context, weights):
    """Assigned-contact path cost.  Left stays on contact0, right on contact1."""
    obj_pos = states[..., 0:3]
    obj_quat = states[..., 3:7]
    left = states[..., 7:10]
    right = states[..., 10:13]
    contact_world, outward, attract_target = _contact_frames(obj_pos, obj_quat, context, weights)
    assigned = torch.stack(
        (
            torch.sum((left - attract_target[..., 0, :]) ** 2, dim=-1),
            torch.sum((right - attract_target[..., 1, :]) ** 2, dim=-1),
        ),
        dim=-1,
    )
    query_mm = float(weights.query_mm)
    attract = assigned.sum(dim=-1) * (query_mm * query_mm)
    swapped = (
        torch.sum((left - attract_target[..., 1, :]) ** 2, dim=-1)
        + torch.sum((right - attract_target[..., 0, :]) ** 2, dim=-1)
    ) * (query_mm * query_mm)
    swap = torch.clamp(attract - swapped, min=0.0) ** 2

    assigned_dist = torch.stack(
        (
            torch.linalg.vector_norm(left - attract_target[..., 0, :], dim=-1),
            torch.linalg.vector_norm(right - attract_target[..., 1, :], dim=-1),
        ),
        dim=-1,
    )
    gate_scale = max(float(weights.gate_scale), 1.0e-4)
    per_tip_gate = torch.sigmoid((gate_scale - assigned_dist) / gate_scale)
    both_gate = per_tip_gate.prod(dim=-1)

    phi = extras["phi"]
    depth = torch.clamp(phi - float(weights.target_phi), min=0.0) * per_tip_gate
    penetration = torch.clamp(-phi - float(weights.penetration_limit), min=0.0)
    contact_depth = torch.sum(depth * depth, dim=-1)
    penetration_cost = torch.sum(penetration * penetration, dim=-1)

    target_p = context.get("target_object_pos")
    target_q = context.get("target_object_quat")
    object_pos = obj_pos.new_zeros(obj_pos.shape[:-1])
    object_lat = obj_pos.new_zeros(obj_pos.shape[:-1])
    object_ori = obj_pos.new_zeros(obj_pos.shape[:-1])
    if target_p is not None and float(weights.object_position) != 0.0:
        object_pos = both_gate * torch.sum((obj_pos - target_p) ** 2, dim=-1)
    if target_p is not None and float(weights.object_lateral) != 0.0:
        object_lat = both_gate * torch.sum((obj_pos[..., :2] - target_p[..., :2]) ** 2, dim=-1)
    if target_q is not None and float(weights.object_orientation) != 0.0:
        object_ori = both_gate * (1.0 - torch.sum(obj_quat * target_q, dim=-1) ** 2)

    left_progress = torch.sum((left - attract_target[..., 0, :]) * (-outward[..., 0, :]), dim=-1)
    right_progress = torch.sum((right - attract_target[..., 1, :]) * (-outward[..., 1, :]), dim=-1)
    sync = (left_progress - right_progress) ** 2
    action = torch.sum(cmd_traj ** 2, dim=-1)
    if cmd_traj.shape[-2] > 1:
        smooth = torch.sum((cmd_traj[..., 1:, :] - cmd_traj[..., :-1, :]) ** 2, dim=-1)
        smooth = F.pad(smooth, (0, 1))
    else:
        smooth = torch.zeros_like(action)

    force_cost = torch.zeros_like(action)
    desired = context.get("desired_force_local")
    if desired is not None and float(weights.force) != 0.0:
        desired_world = _quat_rotate(obj_quat.unsqueeze(-2), desired)
        predicted = extras["normal"] * extras["contact_force"].unsqueeze(-1)
        # Do not multiply by the approach gate. With a zero predicted force that
        # product grows as the tips arrive and the best iterate stays away.
        force_cost = torch.sum((predicted - desired_world) ** 2, dim=(-1, -2))

    tip_sep = torch.clamp(
        float(weights.tip_sep_min) - torch.linalg.vector_norm(left - right, dim=-1), min=0.0
    ) * query_mm
    inter_arm = torch.zeros_like(action)
    capsules = extras.get("capsules")
    if capsules is not None:
        inter_arm = _inter_arm_sphere_cost(capsules, weights)

    path = (
        float(weights.contact_attract) * attract
        + float(weights.swap) * swap
        + float(weights.contact_depth) * contact_depth
        + float(weights.penetration) * penetration_cost
        + float(weights.object_position) * object_pos
        + float(weights.object_lateral) * object_lat
        + float(weights.object_orientation) * object_ori
        + float(weights.action) * action
        + float(weights.smooth) * smooth
        + float(weights.sync) * sync
        + float(weights.force) * force_cost
        + float(weights.inter_arm) * inter_arm
        + float(weights.tip_sep) * (tip_sep ** 2)
    )
    terminal = (
        float(weights.contact_attract) * attract[..., -1]
        + float(weights.swap) * swap[..., -1]
        + float(weights.object_position) * object_pos[..., -1]
        + float(weights.object_orientation) * object_ori[..., -1]
    )
    return path.sum() + float(weights.terminal) * terminal


def _quat_rotate(quat, vec):
    from models.comfree_gs_torch import quat_rotate_wxyz

    return quat_rotate_wxyz(quat, vec)


def _stack_horizon_extras(extras):
    stacked = {}
    keys = extras[0].keys()
    for key in keys:
        values = [item[key] for item in extras]
        if values[0] is None:
            stacked[key] = None
        elif torch.is_tensor(values[0]):
            stacked[key] = torch.stack(values, dim=0)
        else:
            stacked[key] = values
    return stacked


class LambdaContactAdamOptimizer:
    """Adam contact selector on a Gaussian cloud (replaces mlqp acados NLP)."""

    def __init__(
        self,
        mesh_path=None,
        obj_mass=0.01,
        arm_friction=0.9,
        contact_stiffness=12.5,
        time_step=0.01,
        sample_num=70,
        scale_factors=(1.0, 1.0, 1.0),
        cloud=None,
        tip_radius=TIP_QUERY_RADIUS,
        gs_tau=GS_SOFTMIN_TAU,
        adam_steps=40,
        adam_lr=0.02,
        min_pair_distance=0.03,
        device=None,
        **_unused,
    ):
        self.mesh_path = None if mesh_path is None else str(mesh_path)
        self.m = float(obj_mass)
        self.mu_arm_obj = float(arm_friction)
        self.K_contact = float(contact_stiffness)
        self.h = float(time_step)
        self.tip_radius = float(tip_radius)
        self.gs_tau = float(gs_tau)
        self.adam_steps = int(adam_steps)
        self.adam_lr = float(adam_lr)
        self.min_pair_distance = float(min_pair_distance)
        self.device = _device(device)
        self.pp = None
        self.top_region_pairs = 1
        self.enable_timing_prints = False
        self.last_grasp_result = None
        self.last_static_equilibrium_result = None
        self.precomputed_contact_search_cache = None
        self.support_surface_point = None
        self.support_surface_normal = None

        if cloud is not None:
            self.cloud = cloud if isinstance(cloud, GaussianCloud) else GaussianCloud(cloud)
            frames = {
                "points": self.cloud.centers.copy(),
                "normals": np.zeros_like(self.cloud.centers),
            }
            frames["normals"][:, 2] = -1.0
        else:
            if self.mesh_path is None:
                raise ValueError("mesh_path or cloud is required")
            self.cloud, self.pp, frames = GaussianCloud.from_mesh(
                self.mesh_path,
                scale=scale_factors,
                sample_num=int(sample_num),
                radius=max(float(tip_radius) * 0.4, 0.002),
            )
        self.sample_point = np.asarray(frames["points"], dtype=np.float64)
        normals = np.asarray(frames.get("normals", np.zeros_like(self.sample_point)), dtype=np.float64)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norms > 1.0e-8, normals / np.maximum(norms, 1.0e-8), np.array([[0.0, 0.0, -1.0]]))
        self.normal = normals
        self.outward_normal = -self.normal
        self.point_idx = np.arange(self.sample_point.shape[0], dtype=int)
        self.sample_num = int(self.sample_point.shape[0])
        self.spheres = torch.as_tensor(self.cloud.as_spheres(), dtype=torch.float32, device=self.device)

    def set_timing_print_enabled(self, enabled):
        self.enable_timing_prints = bool(enabled)

    def set_support_surface(self, support_surface_point=None, support_surface_normal=None, **_unused):
        self.support_surface_point = None if support_surface_point is None else np.asarray(support_surface_point, dtype=np.float64).reshape(3)
        self.support_surface_normal = None if support_surface_normal is None else _normalize(support_surface_normal)

    def get_contact_candidate_indices(self, visible_face_idx=None, **_unused):
        if visible_face_idx is None:
            return self.point_idx.copy()
        idx = np.asarray(visible_face_idx, dtype=int).reshape(-1)
        idx = idx[(idx >= 0) & (idx < self.sample_num)]
        return idx if idx.size else self.point_idx.copy()

    def _seed_pair(self, visible_idx):
        points = self.sample_point[visible_idx]
        normals = self.normal[visible_idx]
        if points.shape[0] < 2:
            raise RuntimeError("Need at least two Gaussian samples to seed contacts")
        best_i, best_j, best = 0, min(1, points.shape[0] - 1), -1.0e9
        for i in range(points.shape[0]):
            delta = points[i + 1:] - points[i]
            dist = np.linalg.norm(delta, axis=1)
            anti = -np.sum(normals[i] * normals[i + 1:], axis=1)
            score = anti + 4.0 * dist
            if score.size == 0:
                continue
            j_rel = int(np.argmax(score))
            if float(score[j_rel]) > best:
                best = float(score[j_rel])
                best_i = i
                best_j = i + 1 + j_rel
        return points[best_i], points[best_j]

    def choose_contact_set(
        self,
        visible_face_idx=None,
        object_pos=None,
        object_rot=None,
        **_unused,
    ):
        visible = self.get_contact_candidate_indices(visible_face_idx)
        seed_a, seed_b = self._seed_pair(visible)
        points = torch.tensor(
            np.stack((seed_a, seed_b), axis=0),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        gravity = torch.tensor([0.0, 0.0, -self.m * 9.81], dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam([points], lr=self.adam_lr)
        last = None
        for _ in range(self.adam_steps):
            opt.zero_grad(set_to_none=True)
            phi, position, inward = gs_sphere_features(
                points.unsqueeze(0),
                self.spheres,
                query_radius=self.tip_radius,
                tau=self.gs_tau,
            )
            phi = phi.squeeze(0)
            position = position.squeeze(0)
            inward = inward.squeeze(0)
            surface = torch.sum(phi * phi)
            antipodal = torch.sum((inward[0] + inward[1]) ** 2)
            pair = torch.linalg.vector_norm(position[0] - position[1])
            sep = F.softplus(self.min_pair_distance - pair)
            force = -inward * self.K_contact * F.softplus(-phi).unsqueeze(-1)
            net = force.sum(dim=0) - gravity
            torque = torch.linalg.cross(position, force).sum(dim=0)
            last = {
                "phi": phi,
                "position": position,
                "inward": inward,
                "force": force,
            }
            loss = 25.0 * surface + 8.0 * antipodal + 40.0 * sep + 0.4 * torch.sum(net ** 2) + 0.1 * torch.sum(torque ** 2)
            loss.backward()
            opt.step()
        position = last["position"].detach().cpu().numpy().astype(np.float64)
        inward = last["inward"].detach().cpu().numpy().astype(np.float64)
        force = last["force"].detach().cpu().numpy().astype(np.float64)
        phi = last["phi"].detach().cpu().numpy().astype(np.float64)
        antipodal_margin = float(-np.dot(inward[0], inward[1]))
        self.last_grasp_result = {
            "contact_indices": np.array([0, 1], dtype=int),
            "witness_contact_forces_local": force.copy(),
            "witness_force_vectors_local": force.copy(),
            "phi": phi,
        }
        return position, inward, float(np.sum(phi ** 2)), 0.0, antipodal_margin


class MPCExplicitAdam:
    """Horizon Adam MPC over 6-D fingertip increments."""

    solver_backend = "adam"

    def __init__(self, param, model=None, cloud=None, cost_weights=None):
        self.param_ = param
        device = getattr(param, "mppi_device_", None) or getattr(param, "device", None)
        self.device = _device(device)
        if model is not None:
            self.model = model.to(self.device)
        else:
            if cloud is None:
                raise ValueError("MPCExplicitAdam requires a ComfreeGSModel or a GaussianCloud")
            table_z = getattr(param, "table_height", None)
            self.model = ComfreeGSModel(
                cloud,
                obj_mass=float(getattr(param, "obj_mass_", 0.2)),
                gravity=np.asarray(getattr(param, "gravity_", (0.0, 0.0, -9.8)))[:3],
                obj_inertia_pos=float(getattr(param, "planner_object_inertia_pos", 50.0)),
                obj_inertia_rot=float(getattr(param, "planner_object_inertia_rot", 0.05)),
                robot_stiffness=float(np.mean(np.diag(np.asarray(getattr(param, "robot_stiff_", np.eye(6)))))),
                contact_stiffness=float(getattr(param, "contact_stiffness", getattr(param, "model_params", 12.5))),
                timestep=float(getattr(param, "h_", 0.01)),
                tip_radius=float(getattr(param, "tip_radius", TIP_QUERY_RADIUS)),
                table_z=table_z,
                device=self.device,
            )
        requested = int(getattr(param, "mpc_horizon_", 8))
        self.horizon = max(1, min(requested, int(getattr(param, "adam_horizon", 8))))
        self.iters = int(getattr(param, "adam_iters", getattr(param, "mppi_iterations_", 4)))
        self.lr = float(getattr(param, "adam_lr", 0.02))
        self.restarts = int(getattr(param, "adam_restarts", 1))
        self.weights = cost_weights or BigraspGSCostWeights(
            contact_attract=float(getattr(param, "planner_ee_position_weight_", 160.0)),
            object_position=float(getattr(param, "planner_object_target_weight_", 0.0)),
            object_lateral=float(getattr(param, "planner_object_lateral_weight_", 0.0)),
            object_orientation=float(getattr(param, "planner_object_orientation_weight_", 0.0)),
            action=float(getattr(param, "planner_action_weight_", 2.0)),
            smooth=float(getattr(param, "planner_smooth_action_weight_", 3.0)),
            sync=float(getattr(param, "planner_synchronization_weight_", 25.0)),
            force=float(getattr(param, "planner_force_tracking_weight_", 0.0)),
        )
        u_limit = float(getattr(param, "planner_cmd_limit", 0.05))
        self.u_limit = u_limit
        self.n_action = 6
        self.n_state = 13
        self._warm = None
        self.cost_kind = getattr(param, "mpc_cost_kind", "bigrasp_gs")
        self.warp_model = None

    def reset(self):
        self._warm = None

    def _pack_context(self, **kwargs):
        contact = kwargs.get("contact_points_local")
        normals = kwargs.get("normals_local")
        ctx = {}
        if contact is not None:
            if normals is None:
                normals = np.zeros((2, 3), dtype=np.float64)
                normals[:, 2] = -1.0
            ctx["contact_points_local"] = torch.as_tensor(contact, dtype=torch.float32, device=self.device).reshape(2, 3)
            ctx["normals_local"] = torch.as_tensor(normals, dtype=torch.float32, device=self.device).reshape(2, 3)
        else:
            c1 = kwargs.get("contact_point_1")
            c2 = kwargs.get("contact_point_2")
            if c1 is not None and c2 is not None:
                ctx["contact_points_world"] = torch.as_tensor(
                    np.stack((c1, c2), axis=0), dtype=torch.float32, device=self.device
                ).reshape(2, 3)
            else:
                ctx["contact_points_local"] = torch.zeros(2, 3, dtype=torch.float32, device=self.device)
                ctx["normals_local"] = torch.tensor(
                    [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32, device=self.device
                )
        desired = kwargs.get("desired_force_local", kwargs.get("desired_force_vectors_local"))
        if desired is not None:
            ctx["desired_force_local"] = torch.as_tensor(desired, dtype=torch.float32, device=self.device).reshape(2, 3)
        target_p = kwargs.get("target_p", kwargs.get("object_target_pos"))
        target_q = kwargs.get("target_q", kwargs.get("object_target_quat"))
        if target_p is not None:
            ctx["target_object_pos"] = torch.as_tensor(target_p, dtype=torch.float32, device=self.device).reshape(3)
        if target_q is not None:
            ctx["target_object_quat"] = torch.as_tensor(target_q, dtype=torch.float32, device=self.device).reshape(4)
        weights = BigraspGSCostWeights(**self.weights.__dict__)
        if kwargs.get("verify_cost_param_1") is not None:
            weights.verify_1 = float(kwargs["verify_cost_param_1"])
        if kwargs.get("verify_cost_param_2") is not None:
            weights.verify_2 = float(kwargs["verify_cost_param_2"])
        if kwargs.get("approach_offset") is not None:
            weights.approach_offset = float(kwargs["approach_offset"])
        if kwargs.get("curr_qd") is not None:
            ctx["planner_xd"] = np.asarray(kwargs["curr_qd"], dtype=np.float32)
        if kwargs.get("qvel0") is not None:
            ctx["qvel0"] = np.asarray(kwargs["qvel0"], dtype=np.float32)
        if kwargs.get("qpos0") is not None:
            ctx["qpos0"] = np.asarray(kwargs["qpos0"], dtype=np.float32)
        return ctx, weights

    def _solve_raw(self, state, ctx, weights, init):
        raw = init.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([raw], lr=self.lr)
        best_cost = None
        best_raw = raw.detach().clone()
        for _ in range(max(self.iters, 1)):
            opt.zero_grad(set_to_none=True)
            cmd = self.u_limit * torch.tanh(raw)
            states, extras = self._rollout(state, cmd, ctx)
            stacked = _stack_horizon_extras(extras)
            cost = evaluate_bigrasp_gs_cost(states, stacked, cmd, ctx, weights)
            value = cost.reshape(-1)[0]
            if not torch.isfinite(value):
                break
            try:
                value.backward()
            except RuntimeError:
                break
            opt.step()
            with torch.no_grad():
                raw.clamp_(-8.0, 8.0)
            score = float(value.detach().cpu())
            if best_cost is None or score < best_cost:
                best_cost = score
                best_raw = raw.detach().clone()
        with torch.no_grad():
            cmd = self.u_limit * torch.tanh(best_raw)
            states, extras = self._rollout(state.detach(), cmd, ctx)
        if best_cost is None:
            best_cost = float("inf")
        return cmd.detach(), states.detach(), extras, best_cost

    def _rollout(self, state, cmd, ctx=None):
        if self.warp_model is not None:
            return self.warp_model.rollout(
                state,
                cmd,
                planner_xd=None if ctx is None else ctx.get("planner_xd"),
                qpos0=None if ctx is None else ctx.get("qpos0"),
                qvel0=None if ctx is None else ctx.get("qvel0"),
            )
        return self.model.rollout(state, cmd)

    def plan_once(
        self,
        target_p=None,
        target_q=None,
        curr_x=None,
        phi_vec=None,
        jac_mat=None,
        sol_guess=None,
        **kwargs,
    ):
        del phi_vec, jac_mat
        t0 = time.perf_counter()
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        if curr_x is None:
            curr_x = kwargs.get("state")
        state = torch.as_tensor(curr_x, dtype=torch.float32, device=self.device).reshape(self.n_state)
        # #region agent log
        try:
            import json as _json
            import time as _time

            with open("/home/zz/scsp-robot/.cursor/debug-a4aad5.log", "a") as _fh:
                _fh.write(
                    _json.dumps(
                        {
                            "sessionId": "a4aad5",
                            "hypothesisId": "A",
                            "location": "mpc_explicit_adam.py:plan_once",
                            "message": "state tensor built",
                            "data": {
                                "state_device": str(state.device),
                                "planner_device": str(self.device),
                                "has_warp": self.warp_model is not None,
                                "warp_torch_device": str(getattr(self.warp_model, "torch_device", None)),
                                "n_state": int(self.n_state),
                                "n_action": int(self.n_action),
                            },
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass
        # #endregion
        if not torch.isfinite(state).all():
            raise ValueError("plan_once received a non-finite planner state")
        ctx, weights = self._pack_context(
            target_p=target_p,
            target_q=target_q,
            **kwargs,
        )
        inits = []
        if self._warm is not None:
            inits.append(self._warm.to(self.device))
        if sol_guess is not None and isinstance(sol_guess, dict) and "u_traj" in sol_guess:
            warm = torch.as_tensor(sol_guess["u_traj"], dtype=torch.float32, device=self.device)
            if warm.shape == (self.horizon, self.n_action):
                inits.append(torch.atanh(torch.clamp(warm / max(self.u_limit, 1.0e-6), -0.999, 0.999)))
        while len(inits) < max(self.restarts, 1):
            inits.append(0.15 * torch.randn(self.horizon, self.n_action, device=self.device))
        best = None
        for init in inits[: max(self.restarts, 1)]:
            cmd, states, extras, cost = self._solve_raw(state, ctx, weights, init)
            if best is None or cost < best[3]:
                best = (cmd, states, extras, cost)
        cmd, states, extras, cost = best
        self._warm = torch.atanh(torch.clamp(cmd / max(self.u_limit, 1.0e-6), -0.999, 0.999))
        if self._warm.shape[0] > 1:
            self._warm = torch.cat((self._warm[1:], self._warm[-1:]), dim=0)
        u_traj = cmd.cpu().numpy().astype(np.float64)
        rollout = states.cpu().numpy().astype(np.float64)
        # #region agent log
        try:
            import json as _json
            import time as _time

            last = rollout[-1]
            contacts = ctx.get("contact_points_local")
            assigned = None
            if contacts is not None:
                c = contacts.detach().cpu().numpy() if torch.is_tensor(contacts) else np.asarray(contacts)
                assigned = [
                    float(np.linalg.norm(last[7:10] - c[0])),
                    float(np.linalg.norm(last[10:13] - c[1])),
                ]
            with open("/home/zz/scsp-robot/.cursor/debug-a4aad5.log", "a") as _fh:
                _fh.write(
                    _json.dumps(
                        {
                            "sessionId": "a4aad5",
                            "hypothesisId": "F",
                            "location": "mpc_explicit_adam.py:plan_once",
                            "message": "adam plan result",
                            "data": {
                                "iters": int(self.iters),
                                "horizon": int(self.horizon),
                                "lr": float(self.lr),
                                "cost": float(cost),
                                "action": [float(x) for x in u_traj[0]],
                                "last_left": [float(x) for x in last[7:10]],
                                "last_right": [float(x) for x in last[10:13]],
                                "assigned_dist": assigned,
                                "solve_ms": float((time.perf_counter() - t0) * 1000.0),
                            },
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass
        # #endregion
        if extras and extras[0].get("contact_force") is not None:
            forces = torch.stack([step["contact_force"] for step in extras], dim=0).detach().cpu().numpy()
        else:
            forces = np.zeros((max(len(extras), 1), 2), dtype=np.float64)
        result = {
            "action": u_traj[0].copy(),
            "u_traj": u_traj.copy(),
            "ctrl": u_traj.copy(),
            "rollout_q": rollout.copy(),
            "cost": float(cost),
            "cost_opt": np.array([float(cost)], dtype=np.float64),
            "solve_time": float(time.perf_counter() - t0),
            "solver_backend": self.solver_backend,
            "sol_guess": {"u_traj": u_traj.copy()},
            "contact_mask": (forces > 1.0e-4),
            "normal_force": forces.astype(np.float64),
        }
        return result


class MPCExplicitEEAdam(MPCExplicitAdam):
    """``--mpc`` / ``mpc_ee`` entry point: DexForge Warp step, 14-D joint ctrl."""

    def __init__(
        self,
        param,
        model=None,
        cloud=None,
        cost_weights=None,
        warp_model=None,
        graph_solver=None,
    ):
        if warp_model is not None or graph_solver is not None:
            self.param_ = param
            device = getattr(param, "mppi_device_", None) or getattr(param, "device", None)
            self.device = _device(device)
            self.model = None
            self.warp_model = warp_model
            if cloud is not None and warp_model is not None and hasattr(warp_model, "set_cloud"):
                self.warp_model.set_cloud(cloud)
            requested = int(getattr(param, "mpc_horizon_", 5))
            self.horizon = max(1, int(getattr(param, "adam_horizon", requested)))
            self.iters = int(getattr(param, "adam_iters", 40))
            self.lr = float(getattr(param, "adam_lr", 0.05))
            self.restarts = int(getattr(param, "adam_restarts", 1))
            self.weights = cost_weights or BigraspGSCostWeights(
                contact_attract=float(getattr(param, "planner_ee_position_weight_", 160.0)),
                object_position=float(getattr(param, "planner_object_target_weight_", 40.0)),
                object_lateral=float(getattr(param, "planner_object_lateral_weight_", 40.0)),
                object_orientation=float(getattr(param, "planner_object_orientation_weight_", 5.0)),
                action=float(getattr(param, "planner_action_weight_", 2.0)),
                smooth=float(getattr(param, "planner_smooth_action_weight_", 3.0)),
                sync=float(getattr(param, "planner_synchronization_weight_", 25.0)),
                force=float(getattr(param, "planner_force_tracking_weight_", 0.0)),
                gate_scale=float(getattr(param, "planner_contact_gate_scale_", 0.02)),
                approach_offset=float(getattr(param, "adam_approach_offset_", 0.0)),
            )
            self.u_limit = float(getattr(param, "planner_joint_delta_limit", 0.15))
            self.knot_substeps = max(1, int(getattr(param, "adam_knot_substeps", 6)))
            self.n_action = 14
            self.n_state = 21
            self._warm = None
            self.cost_kind = "bigrasp_gs"
            self.graph_solver = graph_solver
            if self.graph_solver is None and warp_model is not None and hasattr(warp_model, "compiled"):
                from planning.mpc_explicit_warp import BigraspWarpMPC

                self.graph_solver = BigraspWarpMPC(
                    warp_model,
                    horizon=self.horizon,
                    iters=self.iters,
                    lr=self.lr,
                    u_limit=self.u_limit,
                    knot_substeps=self.knot_substeps,
                    weights=self.weights,
                    cloud=cloud,
                )
            return
        self.graph_solver = None
        super().__init__(param, model=model, cloud=cloud, cost_weights=cost_weights)

    def plan_once(
        self,
        target_p=None,
        target_q=None,
        curr_x=None,
        phi_vec=None,
        jac_mat=None,
        sol_guess=None,
        **kwargs,
    ):
        if getattr(self, "graph_solver", None) is None:
            return super().plan_once(
                target_p,
                target_q,
                curr_x,
                phi_vec,
                jac_mat,
                sol_guess,
                **kwargs,
            )
        del phi_vec, jac_mat
        t0 = time.perf_counter()
        if curr_x is None:
            curr_x = kwargs.get("state")
        model = self.graph_solver.model
        qpos0 = kwargs.get("qpos0")
        qvel0 = kwargs.get("qvel0")
        if qpos0 is None:
            qpos0 = model.fill_qpos(curr_x)
        else:
            qpos0 = model.fill_qpos(curr_x, base=qpos0)
        if qvel0 is None:
            qvel0 = model.fill_qvel(kwargs.get("curr_qd"))
        else:
            qvel0 = model.fill_qvel(kwargs.get("curr_qd"), base=qvel0)
        contacts = kwargs.get("contact_points_local")
        normals = kwargs.get("normals_local")
        if contacts is None:
            contacts = np.zeros((2, 3), dtype=np.float32)
            normals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
        offset = kwargs.get("approach_offset")
        if offset is not None:
            self.weights.approach_offset = float(offset)
            self.graph_solver.weights.approach_offset = float(offset)
        warm = None
        # The shifted knot commands are measured from the qpos that will be
        # executed. Reusing the previous raw stacks the same increments again.
        if self.graph_solver._warm is not None:
            cmd = np.asarray(self.graph_solver._warm, dtype=np.float32)
            if cmd.shape == (self.horizon, self.n_action):
                warm = np.arctanh(np.clip(cmd / max(self.u_limit, 1.0e-6), -0.999, 0.999)).reshape(1, -1)
        elif sol_guess is not None and isinstance(sol_guess, dict) and "raw" in sol_guess:
            warm = sol_guess["raw"]
        result = self.graph_solver.solve(
            qpos0,
            qvel0,
            contact_points_local=contacts,
            normals_local=normals,
            target_p=target_p,
            target_q=target_q,
            desired_force_local=kwargs.get("desired_force_local", kwargs.get("desired_force_vectors_local")),
            warm=warm,
        )
        result["solve_time"] = float(time.perf_counter() - t0)
        return result


def configure_adam_layout(param):
    param.n_robot_qpos_ = 14
    param.n_qpos_ = 21
    param.n_qvel_ = 20
    param.n_cmd_ = 14
    param.n_state_ = 21
    param.n_action_ = 14
    param.mpc_model = "dexforge_step"
    param.planner_solver_ = "adam"
    param.mpc_cost_kind = "bigrasp_gs"
    bound = np.full(14, float(getattr(param, "planner_joint_delta_limit", 0.15)), dtype=np.float32)
    param.mpc_u_lb_ = -bound
    param.mpc_u_ub_ = bound
    return param
