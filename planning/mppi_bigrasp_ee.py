"""Batched Torch MPPI planner for the single-target bimanual grasp task.

The state is ``[object pose, left EE pose, right EE pose]`` (21 values) and
the control is two world-frame SE(3) increments (12 values).  This module is
kept independent from MuJoCo so it can be exercised with a small synthetic
state in unit tests.  The caller executes only the first action and supplies
the measured state again on the next cycle.
"""

from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
try:  # Keep importing bigrasp.py possible when only MuJoCo is installed.
    import torch
except Exception:  # pragma: no cover - minimal runtime without Torch
    torch = None

from .bigrasp_ee_cost import (
    ACTION_DIM,
    STATE_DIM,
    BimanualEECostConfig,
    evaluate_bimanual_ee_cost,
    integrate_ee_pose,
    quat_exp,
    quat_multiply,
    quat_normalize,
    split_state,
)


def _get(params, name, default):
    return getattr(params, name, default)


def _select_batch_value(value, index, batch_size):
    """Select one rollout's diagnostics and convert it to CPU NumPy values."""
    if torch is not None and isinstance(value, torch.Tensor):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[index].detach().cpu().numpy()
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _select_batch_value(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_select_batch_value(item, index, batch_size) for item in value)
    return value


class BimanualEEMPPI:
    """Torch MPPI with warm start, horizon shift and bimanual hold masks."""

    def __init__(self, params=None, *, device=None, seed=None):
        if torch is None:
            raise ImportError("Torch is required for BimanualEEMPPI")
        self.params = params
        self.horizon = max(int(_get(params, "mpc_horizon_", 20)), 1)
        self.samples = max(int(_get(params, "mppi_samples_", 256)), 8)
        self.iterations = max(int(_get(params, "mppi_iterations_", 4)), 1)
        self.init_iterations = max(int(_get(params, "mppi_init_iterations_", self.iterations)), 1)
        self.temperature = max(float(_get(params, "mppi_lambda_", 1.0)), 1.0e-6)
        self.noise_sigma = max(float(_get(params, "mppi_noise_sigma_", 0.005)), 1.0e-6)
        self.noise_decay = float(np.clip(_get(params, "mppi_noise_decay_", 0.85), 0.05, 1.0))
        self.elite_frac = float(np.clip(_get(params, "mppi_elite_frac_", 0.1), 0.01, 1.0))
        requested_device = device or _get(params, "mppi_device_", None)
        if requested_device is None:
            requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.dtype = torch.float32
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(int(seed))
        self.u_mean = torch.zeros((self.horizon, ACTION_DIM), device=self.device, dtype=self.dtype)
        self.last_action = torch.zeros((ACTION_DIM,), device=self.device, dtype=self.dtype)
        self.last_result = None
        self._has_warm_start = False

        raw_translation_limit = getattr(params, "planner_cmd_limit", None)
        if raw_translation_limit is None:
            raw_bounds = np.asarray(_get(params, "mpc_u_ub_", 0.05), dtype=np.float64).reshape(-1)
            raw_translation_limit = float(np.max(np.abs(raw_bounds[:3]))) if raw_bounds.size else 0.05
        self.translation_limit = abs(float(raw_translation_limit))
        self.rotation_limit = abs(float(_get(params, "planner_rotation_delta_limit_", 0.12)))
        if self.translation_limit <= 0.0:
            self.translation_limit = 0.05
        if self.rotation_limit <= 0.0:
            self.rotation_limit = 0.12

    @property
    def mean_action(self):
        return self.u_mean

    def reset(self):
        self.u_mean.zero_()
        self.last_action.zero_()
        self.last_result = None
        self._has_warm_start = False

    def _action_mask(self, hold_mask):
        hold = torch.as_tensor(hold_mask, device=self.device, dtype=torch.bool).reshape(2)
        mask = torch.ones((ACTION_DIM,), device=self.device, dtype=self.dtype)
        mask[0:6] = (~hold[0]).to(self.dtype)
        mask[6:12] = (~hold[1]).to(self.dtype)
        return mask

    def _clip_actions(self, actions, mask=None):
        out = actions.clone()
        out[..., 0:3] = out[..., 0:3].clamp(-self.translation_limit, self.translation_limit)
        out[..., 6:9] = out[..., 6:9].clamp(-self.translation_limit, self.translation_limit)
        out[..., 3:6] = out[..., 3:6].clamp(-self.rotation_limit, self.rotation_limit)
        out[..., 9:12] = out[..., 9:12].clamp(-self.rotation_limit, self.rotation_limit)
        if mask is not None:
            out = out * mask
        return out

    def _context(
        self,
        contact_points_local,
        normals_local,
        desired_force_local,
        target_object_pos,
        target_object_quat,
        ee_target_quat,
        approach_offset,
        support_z,
    ):
        cfg = BimanualEECostConfig(
            dt=float(_get(self.params, "planner_dt", _get(self.params, "h_", 0.01))),
            object_mass=max(float(_get(self.params, "obj_mass_", _get(self.params, "obj_mass", 0.01))), 1.0e-5),
            object_inertia_pos=float(_get(self.params, "planner_object_inertia_pos", 40.0)),
            object_inertia_rot=float(_get(self.params, "planner_object_inertia_rot", 0.05)),
            contact_stiffness=float(_get(self.params, "contact_stiffness", 12.5)),
            contact_radius=float(_get(self.params, "contact_radius", 0.01)),
            arm_friction=float(_get(self.params, "arm_friction", 0.9)),
            ee_position_weight=float(_get(self.params, "planner_ee_position_weight_", 80.0)),
            ee_orientation_weight=float(_get(self.params, "planner_ee_orientation_weight_", 2.0)),
            force_tracking_weight=float(_get(self.params, "planner_force_tracking_weight_", 12.0)),
            object_position_weight=float(_get(self.params, "planner_object_target_weight_", 300.0)),
            object_lateral_position_weight=float(_get(self.params, "planner_object_lateral_weight_", 300.0)),
            object_orientation_weight=float(_get(self.params, "planner_object_orientation_weight_", 15.0)),
            synchronization_weight=float(_get(self.params, "planner_synchronization_weight_", 25.0)),
            contact_gate_sync_weight=float(_get(self.params, "planner_contact_gate_sync_weight_", 25.0)),
            action_weight=float(_get(self.params, "planner_action_weight_", 2.0)),
            smooth_action_weight=float(_get(self.params, "planner_smooth_action_weight_", 3.0)),
            workspace_weight=float(_get(self.params, "planner_workspace_weight_", 50.0)),
            contact_gate_scale=float(_get(self.params, "planner_contact_gate_scale_", 0.004)),
            contact_tangent_scale=float(_get(self.params, "planner_contact_tangent_scale_", 0.015)),
            object_initial_z=float(torch.as_tensor(target_object_pos).reshape(-1)[2].item()),
            workspace_lower=tuple(_get(self.params, "planner_workspace_lower_", (-1.0, -1.0, 0.0))),
            workspace_upper=tuple(_get(self.params, "planner_workspace_upper_", (2.0, 1.0, 2.0))),
        )
        return {
            "config": cfg,
            "contact_points_local": torch.as_tensor(contact_points_local, device=self.device, dtype=self.dtype).reshape(2, 3),
            "normals_local": torch.as_tensor(normals_local, device=self.device, dtype=self.dtype).reshape(2, 3),
            "desired_force_local": torch.as_tensor(desired_force_local, device=self.device, dtype=self.dtype).reshape(2, 3),
            "target_object_pos": torch.as_tensor(target_object_pos, device=self.device, dtype=self.dtype).reshape(3),
            "target_object_quat": quat_normalize(torch.as_tensor(target_object_quat, device=self.device, dtype=self.dtype).reshape(4)),
            "ee_target_quat": quat_normalize(torch.as_tensor(ee_target_quat, device=self.device, dtype=self.dtype).reshape(2, 4)),
            "approach_offset": float(approach_offset),
            "support_z": None if support_z is None else float(support_z),
        }

    def build_context(
        self,
        contact_points_local,
        normals_local,
        desired_force_local,
        target_object_pos,
        target_object_quat,
        ee_target_quat,
        support_z=None,
        approach_offset=0.0,
    ):
        """Create the device-resident cost context for diagnostics/tests."""
        return self._context(
            contact_points_local,
            normals_local,
            desired_force_local,
            target_object_pos,
            target_object_quat,
            ee_target_quat,
            approach_offset,
            support_z,
        )

    def _rollout_impl(self, state, actions, context):
        """Roll out ``actions`` with shape ``(batch, horizon, 12)``."""
        if state.shape[-1] != STATE_DIM:
            raise ValueError(f"Expected state dimension {STATE_DIM}, got {state.shape[-1]}")
        batch = actions.shape[0]
        current = state.reshape(1, STATE_DIM).expand(batch, -1).clone()
        cfg = context["config"]
        velocity = torch.zeros((batch, 3), device=self.device, dtype=self.dtype)
        angular_velocity = torch.zeros((batch, 3), device=self.device, dtype=self.dtype)
        states = []
        costs = []
        previous = self.last_action.reshape(1, ACTION_DIM).expand(batch, -1)
        force_history = []
        for t in range(actions.shape[1]):
            action = actions[:, t]
            next_ee = integrate_ee_pose(current, action)
            stage_cost, diagnostics = evaluate_bimanual_ee_cost(next_ee, action, context, previous_action=previous)
            bilateral_gate = diagnostics["bilateral_gate"].clamp(0.0, 1.0)
            # Do not let a single predicted fingertip contact drag the object
            # during the bimanual approach.  The measured MuJoCo state is fed
            # back on the next cycle, so real unilateral motion remains visible.
            force = diagnostics["object_force_world"] * bilateral_gate.unsqueeze(-1)
            torque = diagnostics["object_torque_world"] * bilateral_gate.unsqueeze(-1)
            gravity = torch.zeros_like(force)
            gravity[:, 2] = -cfg.object_mass * 9.81
            velocity = 0.96 * velocity + (force + gravity) / cfg.object_mass * cfg.dt
            angular_velocity = 0.96 * angular_velocity + torque / max(cfg.object_inertia_rot, 1.0e-5) * cfg.dt
            obj_p, obj_q, *_ = split_state(next_ee)
            obj_p = obj_p + velocity * cfg.dt
            if context.get("support_z") is not None:
                support_z = float(context["support_z"])
                below = (obj_p[:, 2] < support_z) & (velocity[:, 2] < 0.0)
                obj_p[:, 2] = torch.where(below, torch.as_tensor(support_z, device=self.device, dtype=self.dtype), obj_p[:, 2])
                velocity[:, 2] = torch.where(below, torch.zeros_like(velocity[:, 2]), velocity[:, 2])
            obj_q = quat_multiply(quat_exp(angular_velocity * cfg.dt), obj_q)
            obj_q = quat_normalize(obj_q)
            current = torch.cat((obj_p, obj_q, next_ee[..., 7:]), dim=-1)
            states.append(current)
            costs.append(stage_cost)
            force_history.append(force)
            previous = action
        states = torch.stack(states, dim=1)
        costs = torch.stack(costs, dim=1)
        terminal_cost, terminal_diag = evaluate_bimanual_ee_cost(
            states[:, -1], actions[:, -1], context, previous_action=actions[:, -1], terminal=True
        )
        total = costs.sum(dim=1) + terminal_cost
        return states, total, {"stage_cost": costs, "terminal": terminal_diag, "force": torch.stack(force_history, dim=1)}

    def _rollout(self, state, actions, context):
        with torch.no_grad():
            return self._rollout_impl(state, actions, context)

    def rollout(self, state, actions, context):
        """Public batched rollout hook used by tests and diagnostics."""
        state_t = torch.as_tensor(state, device=self.device, dtype=self.dtype).reshape(STATE_DIM)
        action_t = torch.as_tensor(actions, device=self.device, dtype=self.dtype)
        if action_t.ndim == 2:
            action_t = action_t.unsqueeze(0)
        return self._rollout(state_t, self._clip_actions(action_t), context)

    def plan_once(
        self,
        state,
        contact_points_local,
        normals_local,
        desired_force_local,
        target_object_pos,
        target_object_quat,
        ee_target_quat,
        support_z=None,
        hold_mask=(False, False),
        approach_offset=0.0,
        **_,
    ):
        solve_t0 = time.perf_counter()
        state_t = torch.as_tensor(state, device=self.device, dtype=self.dtype).reshape(STATE_DIM)
        context = self._context(
            contact_points_local,
            normals_local,
            desired_force_local,
            target_object_pos,
            target_object_quat,
            ee_target_quat,
            approach_offset,
            support_z,
        )
        mask = self._action_mask(hold_mask)
        hold_np = torch.as_tensor(hold_mask, dtype=torch.bool).reshape(2).cpu().numpy().copy()
        self.u_mean = self._clip_actions(self.u_mean, mask)
        iterations = self.iterations if self._has_warm_start else self.init_iterations
        best_states = None
        best_cost = None
        best_diag = None
        for iteration in range(iterations):
            sigma = self.noise_sigma * (self.noise_decay ** iteration)
            noise = torch.randn(
                (self.samples, self.horizon, ACTION_DIM),
                generator=self.rng,
                device=self.device,
                dtype=self.dtype,
            ) * sigma
            actions = self._clip_actions(self.u_mean.unsqueeze(0) + noise, mask)
            states, costs, diagnostics = self._rollout(state_t, actions, context)
            elite_count = max(1, min(self.samples, int(round(self.samples * self.elite_frac))))
            elite_cost, elite_idx = torch.topk(costs, elite_count, largest=False)
            weights = torch.softmax(-(elite_cost - elite_cost.min()) / self.temperature, dim=0)
            self.u_mean = torch.sum(actions[elite_idx] * weights[:, None, None], dim=0)
            self.u_mean = self._clip_actions(self.u_mean, mask)
            best_idx = int(torch.argmin(costs).item())
            best_states = states[best_idx]
            best_cost = costs[best_idx]
            best_diag = diagnostics
        action = self._clip_actions(self.u_mean[0], mask)
        self.last_action = action.detach().clone()
        shifted = torch.zeros_like(self.u_mean)
        if self.horizon > 1:
            shifted[:-1] = self.u_mean[1:]
        self.u_mean = shifted
        self._has_warm_start = True
        result = {
            "action": action.detach().cpu().numpy().astype(np.float64),
            "rollout": None if best_states is None else best_states.detach().cpu().numpy(),
            "rollout_q": None if best_states is None else best_states.detach().cpu().numpy(),
            "cost": None if best_cost is None else float(best_cost.detach().cpu()),
            "cost_opt": None if best_cost is None else float(best_cost.detach().cpu()),
            "sol_guess": {
                "u_mean": self.u_mean.detach().cpu().numpy().copy(),
                "hold_mask": hold_np.copy(),
                "context": {"config": asdict(context["config"])},
            },
            "solver_backend": "mppi_ee",
            "solve_status": "success",
            "hold_mask": hold_np,
            "diagnostics": best_diag,
            "best_diagnostics": (
                _select_batch_value(best_diag, int(best_idx), self.samples)
                if best_cost is not None
                else None
            ),
            "best_index": int(best_idx) if best_cost is not None else None,
            "solve_time": float(time.perf_counter() - solve_t0),
        }
        self.last_result = result
        return result


# A short alias is convenient for downstream scripts that spell MPPI first.
MPPIBigraspEE = BimanualEEMPPI
BigraspEEMPPI = BimanualEEMPPI
