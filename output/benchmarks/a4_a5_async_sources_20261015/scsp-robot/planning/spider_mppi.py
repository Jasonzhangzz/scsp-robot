"""SPIDER-style MPPI on batched MuJoCo Warp worlds.

The sampling loop matches ``spider/optimizers/sampling.py``:

* knot noise is interpolated onto the control horizon
* every sample is rolled out in its own Warp world
* the top 10% of rewards are standardized and turned into a softmax
* the noise scale is annealed by ``beta_traj`` across iterations

Controls are the 14 Panda position-actuator setpoints.  Rewards track the
object lift reference and the two lambda contact points.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F


NU = 14


def interp_knots(src: torch.Tensor, upsample: int) -> torch.Tensor:
    """Linearly upsample knots from ``(N, K, D)`` to ``(N, K * upsample, D)``.

    This is the order-1 branch of SPIDER's ``spider.interp.interp``.
    """
    upsample = max(int(upsample), 1)
    if src.shape[1] <= 1 or upsample == 1:
        return src.repeat(1, upsample, 1) if src.shape[1] <= 1 else src
    permuted = src.permute(0, 2, 1)
    stretched = F.interpolate(permuted, size=src.shape[1] * upsample, mode="linear", align_corners=True)
    return stretched.permute(0, 2, 1)


def sample_ctrls(ctrls: torch.Tensor, noise_scale: torch.Tensor, knot_steps: int, global_noise_scale: float) -> torch.Tensor:
    """Sample controls the way SPIDER's ``sample_ctrls`` does.

    ``noise_scale`` has shape ``(num_samples, num_knots, nu)``.
    """
    knot_noise = torch.randn_like(noise_scale) * noise_scale * float(global_noise_scale)
    delta = interp_knots(knot_noise, knot_steps)
    return ctrls.unsqueeze(0) + delta


def compute_spider_weights(rewards: torch.Tensor, temperature: float) -> torch.Tensor:
    """Top-10% standardized softmax, matching SPIDER's ``_compute_weights_impl``."""
    rewards = rewards.reshape(-1)
    nan_mask = torch.isnan(rewards) | torch.isinf(rewards)
    if (~nan_mask).any():
        finite_min = rewards[~nan_mask].min()
    else:
        finite_min = torch.tensor(-1000.0, device=rewards.device, dtype=rewards.dtype)
    rewards = torch.where(nan_mask, finite_min, rewards)
    top_k = max(1, int(0.1 * rewards.shape[0]))
    top_index = torch.topk(rewards, k=top_k, largest=True).indices
    weights = torch.zeros_like(rewards)
    top_rewards = rewards[top_index]
    if top_rewards.numel() == 1:
        standardized = torch.zeros_like(top_rewards)
    else:
        standardized = (top_rewards - top_rewards.mean()) / (top_rewards.std() + 1.0e-2)
    weights[top_index] = F.softmax(standardized / max(float(temperature), 1.0e-6), dim=0)
    return weights


def _quat_rotate_wxyz(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    qvec = quat[..., 1:]
    uv = torch.cross(qvec, vector, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + uuv)


def _slerp_wxyz(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64).reshape(4)
    end = np.asarray(end, dtype=np.float64).reshape(4)
    start = start / max(np.linalg.norm(start), 1.0e-12)
    end = end / max(np.linalg.norm(end), 1.0e-12)
    if float(np.dot(start, end)) < 0.0:
        end = -end
    dot = float(np.clip(np.dot(start, end), -1.0, 1.0))
    if dot > 0.9995:
        mixed = start + alpha * (end - start)
        return mixed / max(np.linalg.norm(mixed), 1.0e-12)
    theta_0 = float(np.arccos(dot))
    theta = theta_0 * float(alpha)
    sin_theta_0 = float(np.sin(theta_0))
    s0 = float(np.sin(theta_0 - theta)) / max(sin_theta_0, 1.0e-9)
    s1 = float(np.sin(theta)) / max(sin_theta_0, 1.0e-9)
    mixed = s0 * start + s1 * end
    return mixed / max(np.linalg.norm(mixed), 1.0e-12)


class SpiderBimanualMPPI:
    """Receding-horizon SPIDER MPPI over GPU-parallel MuJoCo Warp worlds."""

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
        device: str = "cuda:0",
    ):
        import mujoco_warp as mjwarp
        import warp as wp

        self.mjwarp = mjwarp
        self.wp = wp
        self.model_cpu = model
        self.data_cpu = data
        self.left_actuator_ids = np.asarray(left_actuator_ids, dtype=np.int32).reshape(-1)
        self.right_actuator_ids = np.asarray(right_actuator_ids, dtype=np.int32).reshape(-1)
        if self.left_actuator_ids.size + self.right_actuator_ids.size != NU:
            raise ValueError(f"Expected {NU} arm actuators, got {self.left_actuator_ids.size + self.right_actuator_ids.size}")
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
            raise RuntimeError("Spider MJWP MPPI requires a CUDA device")
        self.device = torch.device(device)
        wp.init()
        wp.set_device(str(self.device))

        ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float32)
        order = np.concatenate((self.left_actuator_ids, self.right_actuator_ids))
        self._ctrl_low = torch.as_tensor(ctrlrange[order, 0], device=self.device)
        self._ctrl_high = torch.as_tensor(ctrlrange[order, 1], device=self.device)
        self._actuator_order = order

        # SPIDER uses 20 Newton / 50 linesearch iterations for hand scenes.
        # The MuJoCo default of 100 Newton steps is compiled into the Warp
        # graph and multiplies every parallel rollout.
        model.opt.iterations = min(int(model.opt.iterations), 20)
        model.opt.ls_iterations = min(int(model.opt.ls_iterations), 10)
        self.model_wp = mjwarp.put_model(model)
        self.data_wp = mjwarp.put_data(model, data, nworld=self.num_samples, nconmax=48, njmax=80)
        self.data_wp_prev = mjwarp.put_data(model, data, nworld=self.num_samples, nconmax=48, njmax=80)
        self._ctrl_buf = torch.zeros((self.num_samples, int(model.nu)), dtype=torch.float32, device=self.device)
        self.graph = self._capture_step_graph()
        self.ctrls = None
        self._knot_noise = None

    def _capture_step_graph(self):
        wp = self.wp
        mjwarp = self.mjwarp

        def _once():
            mjwarp.step(self.model_wp, self.data_wp)

        with wp.ScopedDevice(str(self.device)):
            for _ in range(max(self.substeps, 2)):
                _once()
            wp.synchronize()
            try:
                with wp.ScopedCapture() as capture:
                    _once()
                wp.synchronize()
                return capture.graph
            except Exception:
                return None

    def _copy_cpu_state(self):
        """Broadcast the measured MuJoCo state into every Warp world."""
        wp = self.wp
        qpos = torch.as_tensor(self.data_cpu.qpos, dtype=torch.float32, device=self.device)
        qvel = torch.as_tensor(self.data_cpu.qvel, dtype=torch.float32, device=self.device)
        ctrl = torch.as_tensor(self.data_cpu.ctrl, dtype=torch.float32, device=self.device)
        qpos = qpos.reshape(1, -1).expand(self.num_samples, -1).contiguous()
        qvel = qvel.reshape(1, -1).expand(self.num_samples, -1).contiguous()
        ctrl = ctrl.reshape(1, -1).expand(self.num_samples, -1).contiguous()
        wp.copy(self.data_wp.qpos, wp.from_torch(qpos))
        wp.copy(self.data_wp.qvel, wp.from_torch(qvel))
        wp.copy(self.data_wp.ctrl, wp.from_torch(ctrl))
        self.mjwarp.forward(self.model_wp, self.data_wp)
        self._save_state()

    def _save_state(self):
        self._copy_state(self.data_wp, self.data_wp_prev)

    def _load_state(self):
        self._copy_state(self.data_wp_prev, self.data_wp)

    def _copy_state(self, src, dst):
        wp = self.wp
        for name in ("qpos", "qvel", "act", "act_dot", "qacc_warmstart", "ctrl"):
            if hasattr(src, name) and hasattr(dst, name):
                wp.copy(getattr(dst, name), getattr(src, name))

    def _clip(self, ctrls: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(ctrls, self._ctrl_high), self._ctrl_low)

    def _measured_ctrl(self) -> torch.Tensor:
        ctrl = np.asarray(self.data_cpu.ctrl, dtype=np.float32).reshape(-1)
        ordered = ctrl[self._actuator_order]
        return torch.as_tensor(ordered, dtype=torch.float32, device=self.device)

    def _knot_noise_scale(self, global_scale: float) -> torch.Tensor:
        scale = self.noise_scale * float(global_scale)
        return torch.full(
            (self.num_samples, self.num_knots, NU),
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
            self._apply_ctrl(samples[:, step])
            total = total + self._reward(
                ref_pos[step],
                ref_quat[step],
                contact_local,
                terminal=step == self.horizon - 1,
            )
        return total / float(self.horizon)

    def _apply_ctrl(self, ctrl: torch.Tensor):
        wp = self.wp
        self._ctrl_buf.zero_()
        self._ctrl_buf[:, self.left_actuator_ids] = ctrl[:, :7]
        self._ctrl_buf[:, self.right_actuator_ids] = ctrl[:, 7:]
        wp.copy(self.data_wp.ctrl, wp.from_torch(self._ctrl_buf))
        if self.graph is None:
            for _ in range(self.substeps):
                self.mjwarp.step(self.model_wp, self.data_wp)
            return
        for _ in range(self.substeps):
            wp.capture_launch(self.graph)

    def _approach_ctrls(self, approach_q) -> torch.Tensor:
        current = self._clip(self._measured_ctrl())
        if approach_q is None:
            return current.reshape(1, NU).expand(self.horizon, NU).contiguous()
        goal = torch.as_tensor(np.asarray(approach_q, dtype=np.float32).reshape(NU), device=self.device)
        goal = self._clip(goal)
        alpha = torch.linspace(1.0 / self.horizon, 1.0, self.horizon, device=self.device)
        return self._clip(current.unsqueeze(0) + alpha.unsqueeze(1) * (goal - current).unsqueeze(0))

    def plan_once(self, contact_points_local, target_object_pos, target_object_quat, approach_q=None):
        started = time.perf_counter()
        self._copy_cpu_state()
        if self.ctrls is None or self.ctrls.shape != (self.horizon, NU):
            self.ctrls = self._approach_ctrls(approach_q)
        contact_local = torch.as_tensor(contact_points_local, dtype=torch.float32, device=self.device).reshape(2, 3)
        ref_pos, ref_quat = self._reference(target_object_pos, target_object_quat)
        best_reward = None
        for iteration in range(self.iterations):
            noise = self._knot_noise_scale(self.beta_traj ** iteration)
            samples = self._clip(sample_ctrls(self.ctrls, noise, self.knot_steps, 1.0))
            rewards = self._rollout(samples, ref_pos, ref_quat, contact_local)
            weights = compute_spider_weights(rewards, self.temperature)
            self.ctrls = self._clip(torch.sum(weights[:, None, None] * samples, dim=0))
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
            "solver_backend": "spider_mjwp",
            "contact_mask": np.zeros((self.horizon, 2), dtype=bool),
            "normal_force": np.zeros((self.horizon, 2), dtype=np.float64),
        }
