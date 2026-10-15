"""SPIDER-style MPPI whose rollouts use closed-form Warp contact.

Two action spaces share the same ``ExplicitModelWarp`` physics:

* :class:`ExplicitJointMPPI` samples 14-D joint increments ``Δq``
* :class:`ExplicitEEMPPI` samples 12-D dual-arm EE SE(3) increments

The environment still executes only the first action.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from planning.spider_mppi import (
    NU,
    _quat_rotate_wxyz,
    _slerp_wxyz,
    compute_spider_weights,
    sample_ctrls,
)


EE_NU = 12

PLAN_ONCE_KEYS = (
    "action",
    "ctrl",
    "cost",
    "cost_opt",
    "solve_time",
    "solver_backend",
    "contact_mask",
    "normal_force",
)


def _actuator_joint_addresses(model, actuator_ids):
    actuator_ids = np.asarray(actuator_ids, dtype=np.int32).reshape(-1)
    joint_ids = np.asarray(model.actuator_trnid, dtype=np.int32)[actuator_ids, 0]
    qpos_adr = np.asarray(model.jnt_qposadr, dtype=np.int32)[joint_ids]
    dof_adr = np.asarray(model.jnt_dofadr, dtype=np.int32)[joint_ids]
    return qpos_adr, dof_adr


def _site_jacobian(model, data, site_id, dof_adr):
    import mujoco

    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, int(model.nv)), dtype=np.float64)
    jacr = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, int(site_id))
    dof_adr = np.asarray(dof_adr, dtype=np.int32).reshape(-1)
    return np.vstack((jacp[:, dof_adr], jacr[:, dof_adr]))


class _ExplicitWarpMPPIBase:
    """Shared Warp worlds, SPIDER weights, and object/contact reward."""

    action_dim = NU
    solver_backend = "mppi_joint"

    def __init__(
        self,
        model,
        data,
        *,
        left_actuator_ids,
        right_actuator_ids,
        obj_qpos_adr: int,
        left_site_id: int,
        right_site_id: int,
        left_qpos_adr=None,
        right_qpos_adr=None,
        left_dof_adr=None,
        right_dof_adr=None,
        num_samples: int = 128,
        horizon: int = 8,
        knot_steps: int = 2,
        iterations: int = 2,
        temperature: float = 0.1,
        beta_traj: float = 0.85,
        noise_scale: float = 0.1,
        substeps: int = 1,
        terminal_reward_scale: float = 5.0,
        object_position_weight: float = 20.0,
        object_lateral_weight: float = 40.0,
        object_orientation_weight: float = 2.0,
        contact_weight: float = 12.0,
        comfree_stiffness: float = 0.2,
        comfree_damping: float = 0.001,
        nconmax: int = 48,
        njmax: int = 80,
        device: str = "cuda:0",
    ):
        import warp as wp
        from models.explicit_model_warp import ExplicitModelWarp

        self.wp = wp
        self.model_cpu = model
        self.data_cpu = data
        self.left_actuator_ids = np.asarray(left_actuator_ids, dtype=np.int32).reshape(-1)
        self.right_actuator_ids = np.asarray(right_actuator_ids, dtype=np.int32).reshape(-1)
        if self.left_actuator_ids.size + self.right_actuator_ids.size != NU:
            raise ValueError(
                f"Expected {NU} arm actuators, got "
                f"{self.left_actuator_ids.size + self.right_actuator_ids.size}"
            )
        inferred_left_q, inferred_left_dof = _actuator_joint_addresses(model, self.left_actuator_ids)
        inferred_right_q, inferred_right_dof = _actuator_joint_addresses(model, self.right_actuator_ids)
        self.left_qpos_adr = np.asarray(left_qpos_adr if left_qpos_adr is not None else inferred_left_q, dtype=np.int32)
        self.right_qpos_adr = np.asarray(right_qpos_adr if right_qpos_adr is not None else inferred_right_q, dtype=np.int32)
        self.left_dof_adr = np.asarray(left_dof_adr if left_dof_adr is not None else inferred_left_dof, dtype=np.int32)
        self.right_dof_adr = np.asarray(right_dof_adr if right_dof_adr is not None else inferred_right_dof, dtype=np.int32)
        self.obj_qpos_adr = int(obj_qpos_adr)
        self.left_site_id = int(left_site_id)
        self.right_site_id = int(right_site_id)
        self.num_samples = max(int(num_samples), 2)
        self.knot_steps = max(int(knot_steps), 1)
        self.horizon = max(int(horizon), 1)
        if self.horizon % self.knot_steps != 0:
            self.horizon = self.knot_steps * max(1, int(round(self.horizon / self.knot_steps)))
        self.num_knots = max(1, self.horizon // self.knot_steps)
        self.iterations = max(int(iterations), 1)
        self.temperature = max(float(temperature), 1.0e-6)
        self.beta_traj = float(np.clip(beta_traj, 0.05, 1.0))
        self.noise_scale = max(float(noise_scale), 1.0e-6)
        self.substeps = max(int(substeps), 1)
        self.terminal_reward_scale = float(terminal_reward_scale)
        self.object_position_weight = float(object_position_weight)
        self.object_lateral_weight = float(object_lateral_weight)
        self.object_orientation_weight = float(object_orientation_weight)
        self.contact_weight = float(contact_weight)
        if not torch.cuda.is_available():
            raise RuntimeError("Explicit MJWP MPPI requires a CUDA device")
        self.device = torch.device(device)
        wp.init()
        wp.set_device(str(self.device))

        ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
        order = np.concatenate((self.left_actuator_ids, self.right_actuator_ids))
        self._ctrl_low = torch.as_tensor(ctrlrange[order, 0], device=self.device)
        self._ctrl_high = torch.as_tensor(ctrlrange[order, 1], device=self.device)
        self._actuator_order = order
        self._left_qpos_idx = torch.as_tensor(self.left_qpos_adr, device=self.device, dtype=torch.long)
        self._right_qpos_idx = torch.as_tensor(self.right_qpos_adr, device=self.device, dtype=torch.long)

        self.rollout_model = ExplicitModelWarp(
            model,
            data,
            nworld=self.num_samples,
            nconmax=nconmax,
            njmax=njmax,
            comfree_stiffness=comfree_stiffness,
            comfree_damping=comfree_damping,
            device=str(self.device),
        )
        self.model_wp = self.rollout_model.model_wp
        self.data_wp = self.rollout_model.data_wp
        self.data_wp_prev = self.rollout_model.make_data()
        self._ctrl_buf = torch.zeros((self.num_samples, int(model.nu)), dtype=torch.float32, device=self.device)
        self.graph = self.rollout_model.capture_step_graph(self.data_wp)
        self.ctrls = None

    def reset(self):
        """Drop the receding-horizon mean so the next episode starts cold."""
        self.ctrls = None

    def _copy_cpu_state(self):
        self.rollout_model.broadcast_state(
            self.data_cpu.qpos,
            self.data_cpu.qvel,
            self.data_cpu.ctrl,
            data=self.data_wp,
        )
        self._save_state()

    def _save_state(self):
        self.rollout_model.copy_state(self.data_wp, self.data_wp_prev)

    def _load_state(self):
        self.rollout_model.copy_state(self.data_wp_prev, self.data_wp)

    def _clip_joint_cmd(self, cmd: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(cmd, self._ctrl_high), self._ctrl_low)

    def _measured_q(self) -> torch.Tensor:
        qpos = np.asarray(self.data_cpu.qpos, dtype=np.float32).reshape(-1)
        ordered = np.concatenate((qpos[self.left_qpos_adr], qpos[self.right_qpos_adr]))
        return torch.as_tensor(ordered, dtype=torch.float32, device=self.device)

    def _world_arm_q(self) -> torch.Tensor:
        qpos = self.wp.to_torch(self.data_wp.qpos)
        left = qpos.index_select(1, self._left_qpos_idx)
        right = qpos.index_select(1, self._right_qpos_idx)
        return torch.cat((left, right), dim=-1)

    def _knot_noise_scale(self, global_scale: float) -> torch.Tensor:
        scale = self.noise_scale * float(global_scale)
        return torch.full(
            (self.num_samples, self.num_knots, self.action_dim),
            scale,
            dtype=torch.float32,
            device=self.device,
        )

    def _reference(self, target_pos, target_quat):
        start = np.asarray(self.data_cpu.qpos[self.obj_qpos_adr : self.obj_qpos_adr + 7], dtype=np.float64)
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_quat = np.asarray(target_quat, dtype=np.float64).reshape(4)
        positions = []
        quats = []
        for step in range(self.horizon):
            alpha = float(step + 1) / float(self.horizon)
            positions.append((1.0 - alpha) * start[:3] + alpha * target_pos)
            quats.append(_slerp_wxyz(start[3:7], target_quat, alpha))
        pos = torch.as_tensor(np.stack(positions), dtype=torch.float32, device=self.device)
        quat = torch.as_tensor(np.stack(quats), dtype=torch.float32, device=self.device)
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
        return pos, quat

    def _reward(self, ref_pos, ref_quat, contact_local, terminal: bool) -> torch.Tensor:
        qpos = self.wp.to_torch(self.data_wp.qpos)
        obj = qpos[:, self.obj_qpos_adr : self.obj_qpos_adr + 7]
        delta = obj[:, :3] - ref_pos.unsqueeze(0)
        lateral = torch.linalg.vector_norm(delta[:, :2], dim=-1)
        height = torch.abs(delta[:, 2])
        quat = obj[:, 3:7]
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)
        aligned = torch.abs(torch.sum(quat * ref_quat.unsqueeze(0), dim=-1)).clamp(0.0, 1.0)
        orientation = 1.0 - aligned
        sites = self.wp.to_torch(self.data_wp.site_xpos)
        tips = torch.stack((sites[:, self.left_site_id], sites[:, self.right_site_id]), dim=1)
        local = contact_local.reshape(1, 2, 3).expand(qpos.shape[0], -1, -1)
        contact_world = obj[:, None, :3] + _quat_rotate_wxyz(quat[:, None, :], local)
        contact = torch.linalg.vector_norm(tips - contact_world, dim=-1).sum(dim=-1)
        reward = (
            -self.object_lateral_weight * lateral
            - self.object_position_weight * height
            - self.object_orientation_weight * orientation
            - self.contact_weight * contact
        )
        if terminal:
            reward = reward * self.terminal_reward_scale
        return reward

    def _rollout(self, samples: torch.Tensor, ref_pos, ref_quat, contact_local) -> torch.Tensor:
        self._load_state()
        total = torch.zeros(self.num_samples, device=self.device)
        for step in range(self.horizon):
            self._apply_action(samples[:, step])
            total = total + self._reward(
                ref_pos[step],
                ref_quat[step],
                contact_local,
                terminal=step == self.horizon - 1,
            )
        return total / float(self.horizon)

    def _write_joint_setpoints(self, joint_cmd: torch.Tensor):
        wp = self.wp
        cmd = self._clip_joint_cmd(joint_cmd)
        self._ctrl_buf.zero_()
        self._ctrl_buf[:, self.left_actuator_ids] = cmd[:, :7]
        self._ctrl_buf[:, self.right_actuator_ids] = cmd[:, 7:]
        wp.copy(self.data_wp.ctrl, wp.from_torch(self._ctrl_buf))
        if self.graph is None:
            for _ in range(self.substeps):
                self.rollout_model.step(self.data_wp)
            return
        for _ in range(self.substeps):
            wp.capture_launch(self.graph)

    def _apply_action(self, action: torch.Tensor):
        raise NotImplementedError

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _init_mean(self, approach_q) -> torch.Tensor:
        raise NotImplementedError

    def plan_once(self, contact_points_local, target_object_pos, target_object_quat, approach_q=None):
        started = time.perf_counter()
        self._copy_cpu_state()
        if self.ctrls is None or self.ctrls.shape != (self.horizon, self.action_dim):
            self.ctrls = self._init_mean(approach_q)
        contact_local = torch.as_tensor(contact_points_local, dtype=torch.float32, device=self.device).reshape(2, 3)
        ref_pos, ref_quat = self._reference(target_object_pos, target_object_quat)
        best_reward = None
        for iteration in range(self.iterations):
            noise = self._knot_noise_scale(self.beta_traj ** iteration)
            samples = self._clip_actions(sample_ctrls(self.ctrls, noise, self.knot_steps, 1.0))
            rewards = self._rollout(samples, ref_pos, ref_quat, contact_local)
            weights = compute_spider_weights(rewards, self.temperature)
            self.ctrls = self._clip_actions(torch.sum(weights[:, None, None] * samples, dim=0))
            best_reward = rewards
        action = self.ctrls[0].detach().cpu().numpy().astype(np.float64)
        shifted = torch.zeros_like(self.ctrls)
        if self.horizon > 1:
            shifted[:-1] = self.ctrls[1:]
            shifted[-1] = self.ctrls[-1]
        else:
            shifted = self.ctrls
        self.ctrls = shifted
        reward = float(best_reward.max().detach().cpu()) if best_reward is not None else 0.0
        return {
            "action": action,
            "ctrl": self.ctrls.detach().cpu().numpy().astype(np.float64),
            "cost": -reward,
            "cost_opt": -reward,
            "solve_time": float(time.perf_counter() - started),
            "solver_backend": self.solver_backend,
            "contact_mask": np.zeros((self.horizon, 2), dtype=bool),
            "normal_force": np.zeros((self.horizon, 2), dtype=np.float64),
        }


class ExplicitJointMPPI(_ExplicitWarpMPPIBase):
    """SPIDER MPPI over 14-D joint increments ``Δq``."""

    action_dim = NU
    solver_backend = "mppi_joint"

    def __init__(self, *args, joint_delta_limit: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_delta_limit = max(float(joint_delta_limit), 1.0e-4)
        self._delta_limit = torch.full((NU,), self.joint_delta_limit, device=self.device)

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(actions, self._delta_limit), -self._delta_limit)

    def _init_mean(self, approach_q) -> torch.Tensor:
        if approach_q is None:
            return torch.zeros((self.horizon, NU), device=self.device)
        goal = self._clip_joint_cmd(torch.as_tensor(np.asarray(approach_q, dtype=np.float32).reshape(NU), device=self.device))
        remaining = goal - self._measured_q()
        return self._clip_actions(remaining.unsqueeze(0).expand(self.horizon, -1).contiguous() / float(self.horizon))

    def _apply_action(self, action: torch.Tensor):
        self._write_joint_setpoints(self._world_arm_q() + action)


class ExplicitEEMPPI(_ExplicitWarpMPPIBase):
    """SPIDER MPPI over 12-D dual-arm EE SE(3) increments."""

    action_dim = EE_NU
    solver_backend = "mppi_ee"

    def __init__(
        self,
        *args,
        translation_limit: float = 0.05,
        rotation_limit: float = 0.12,
        **kwargs,
    ):
        kwargs["noise_scale"] = min(abs(float(translation_limit)), 0.02)
        super().__init__(*args, **kwargs)
        self.translation_limit = max(float(translation_limit), 1.0e-4)
        self.rotation_limit = max(float(rotation_limit), 1.0e-4)
        self._left_j_pinv = torch.zeros((7, 6), device=self.device, dtype=torch.float32)
        self._right_j_pinv = torch.zeros((7, 6), device=self.device, dtype=torch.float32)
        self._refresh_linearization()

    def _refresh_linearization(self):
        left_j = _site_jacobian(self.model_cpu, self.data_cpu, self.left_site_id, self.left_dof_adr)
        right_j = _site_jacobian(self.model_cpu, self.data_cpu, self.right_site_id, self.right_dof_adr)
        self._left_j_pinv = torch.as_tensor(np.linalg.pinv(left_j), dtype=torch.float32, device=self.device)
        self._right_j_pinv = torch.as_tensor(np.linalg.pinv(right_j), dtype=torch.float32, device=self.device)

    def _copy_cpu_state(self):
        super()._copy_cpu_state()
        self._refresh_linearization()

    def _clip_actions(self, actions: torch.Tensor) -> torch.Tensor:
        out = actions.clone()
        out[..., 0:3] = out[..., 0:3].clamp(-self.translation_limit, self.translation_limit)
        out[..., 6:9] = out[..., 6:9].clamp(-self.translation_limit, self.translation_limit)
        out[..., 3:6] = out[..., 3:6].clamp(-self.rotation_limit, self.rotation_limit)
        out[..., 9:12] = out[..., 9:12].clamp(-self.rotation_limit, self.rotation_limit)
        return out

    def _init_mean(self, approach_q) -> torch.Tensor:
        del approach_q
        return torch.zeros((self.horizon, EE_NU), device=self.device)

    def _apply_action(self, action: torch.Tensor):
        left_dq = action[:, :6] @ self._left_j_pinv.T
        right_dq = action[:, 6:] @ self._right_j_pinv.T
        self._write_joint_setpoints(self._world_arm_q() + torch.cat((left_dq, right_dq), dim=-1))


ExplicitBimanualMPPI = ExplicitJointMPPI
