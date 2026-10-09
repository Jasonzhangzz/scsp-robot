"""Warp MPPI on the same DexForge fast step and contact cost as Adam MPC."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from planning.bigrasp_warp_loss import ensure_kernels as ensure_loss
from planning.bigrasp_warp_loss import weights_struct
from planning.mpc_explicit_adam import BigraspGSCostWeights
from planning.warp_adam import ensure_warp


def command_from_raw(raw, u_limit):
    """Joint increments ``u_limit * tanh(raw)``, matching the Adam decoder."""
    raw = np.asarray(raw, dtype=np.float64)
    return float(u_limit) * np.tanh(raw)


_KERNELS = None


def ensure_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    wp = ensure_warp()

    @wp.kernel(enable_backward=False)
    def decode_batch(
        raw: wp.array2d(dtype=float),
        qpos0: wp.array2d(dtype=float),
        ctrl: wp.array2d(dtype=float),
        qadr: wp.array(dtype=int),
        uadr: wp.array(dtype=int),
        ranges: wp.array2d(dtype=float),
        u_limit: float,
        step: int,
        n_action: int,
        knot_substeps: int,
    ):
        world, joint = wp.tid()
        knot = step // wp.max(knot_substeps, 1)
        acc = float(0.0)
        for index in range(knot + 1):
            acc += u_limit * wp.tanh(raw[world, index * n_action + joint])
        value = wp.clamp(qpos0[world, qadr[joint]] + acc, ranges[uadr[joint], 0], ranges[uadr[joint], 1])
        ctrl[world, uadr[joint]] = value

    _KERNELS = SimpleNamespace(wp=wp, decode_batch=decode_batch)
    return _KERNELS


class BigraspWarpMPPI:
    """Sampled 14-D joint knots rolled out with ``compile_fast_step``."""

    solver_backend = "mppi_dexforge"

    def __init__(
        self,
        fast_step,
        *,
        horizon=5,
        knot_substeps=6,
        num_samples=64,
        iterations=4,
        temperature=0.1,
        noise_scale=0.4,
        beta=0.85,
        u_limit=0.15,
        weights=None,
    ):
        if int(fast_step.nworld) != int(num_samples):
            raise ValueError(
                f"DexForgeFastStep nworld={fast_step.nworld} must equal MPPI samples={num_samples}"
            )
        self.model = fast_step
        self.horizon = max(1, int(horizon))
        self.knot_substeps = max(1, int(knot_substeps))
        self.dense_steps = self.horizon * self.knot_substeps
        self.num_samples = max(2, int(num_samples))
        self.iterations = max(1, int(iterations))
        self.temperature = max(float(temperature), 1.0e-4)
        self.noise_scale = max(float(noise_scale), 1.0e-4)
        self.beta = float(np.clip(beta, 0.05, 1.0))
        self.u_limit = float(u_limit)
        self.weights = weights or BigraspGSCostWeights()
        self.n_action = 14
        self.device = fast_step.device_name
        self.steps = fast_step.allocate_graph_chain(self.dense_steps)
        self.left_phi_row, self.right_phi_row = fast_step.tip_phi_rows()
        self._graph = None
        self._init_buffers()
        self._mean = np.zeros((self.horizon, self.n_action), dtype=np.float32)

    def reset(self):
        self._mean[:] = 0.0

    def _init_buffers(self):
        wp = ensure_warp()
        ensure_kernels()
        ensure_loss()
        device = self.device
        worlds = self.num_samples
        knots = self.horizon
        raw_dim = self.n_action * knots
        ids = self.model._ids
        qadr = np.concatenate((ids["left_qpos"], ids["right_qpos"])).astype(np.int32)
        uadr = np.concatenate((ids["left_act"], ids["right_act"])).astype(np.int32)
        ranges = np.asarray(self.model.cpu_model.actuator_ctrlrange, dtype=np.float32)
        self.buf = SimpleNamespace(
            raw=wp.zeros((worlds, raw_dim), dtype=float, device=device),
            cmd=wp.zeros((worlds, knots, self.n_action), dtype=float, device=device),
            qadr=wp.array(qadr, dtype=int, device=device),
            uadr=wp.array(uadr, dtype=int, device=device),
            ranges=wp.array(ranges, dtype=float, device=device),
            contact_local=wp.zeros(2, dtype=wp.vec3, device=device),
            normal_local=wp.zeros(2, dtype=wp.vec3, device=device),
            target_pos=wp.zeros(1, dtype=wp.vec3, device=device),
            target_quat=wp.zeros(1, dtype=wp.vec4, device=device),
            flags=wp.zeros(2, dtype=int, device=device),
            obj_pos=wp.zeros((worlds, knots), dtype=wp.vec3, device=device),
            obj_quat=wp.zeros((worlds, knots), dtype=wp.vec4, device=device),
            left=wp.zeros((worlds, knots), dtype=wp.vec3, device=device),
            right=wp.zeros((worlds, knots), dtype=wp.vec3, device=device),
            capsules=wp.zeros((worlds, knots, 6), dtype=wp.vec3, device=device),
            phi=wp.zeros((worlds, knots, 2), dtype=float, device=device),
            loss=wp.zeros(worlds, dtype=float, device=device),
        )
        self.buf.target_quat.assign(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    def _decode(self):
        k = ensure_kernels()
        for step, slot in enumerate(self.steps):
            k.wp.launch(
                k.decode_batch,
                dim=(self.num_samples, self.n_action),
                inputs=[
                    self.buf.raw,
                    self.model.qpos,
                    slot.ctrl,
                    self.buf.qadr,
                    self.buf.uadr,
                    self.buf.ranges,
                    self.u_limit,
                    step,
                    self.n_action,
                    self.knot_substeps,
                ],
            )

    def _forward_and_cost(self):
        from comfree_warp.native_adjoint.fast_step_adjoint import forward as step_forward

        loss_k = ensure_loss()
        wp = loss_k.wp
        self.buf.loss.zero_()
        self._decode()
        for slot in self.steps:
            step_forward(self.model.compiled, slot.inputs, slot.workspace)
        ids = self.model._ids
        left_off = np.asarray(ids["left_tip_offset"], dtype=np.float32).reshape(3)
        right_off = np.asarray(ids["right_tip_offset"], dtype=np.float32).reshape(3)
        for knot in range(self.horizon):
            result = self.steps[(knot + 1) * self.knot_substeps - 1].recorded.result
            job = loss_k.ExtractWorldJob()
            job.qpos = result.qpos
            job.body_pos = result.body_position
            job.body_mat = result.body_matrix
            job.contact_distance = result.contact_distance
            job.contact_frame = result.contact_frame
            job.obj_qpos = int(ids["obj_qpos"])
            job.left_tip = int(ids["left_tip"])
            job.right_tip = int(ids["right_tip"])
            job.left_forearm = int(ids["left_forearm"])
            job.right_forearm = int(ids["right_forearm"])
            job.left_wrist = int(ids["left_wrist"])
            job.right_wrist = int(ids["right_wrist"])
            job.left_off = wp.vec3(float(left_off[0]), float(left_off[1]), float(left_off[2]))
            job.right_off = wp.vec3(float(right_off[0]), float(right_off[1]), float(right_off[2]))
            job.left_row = int(self.left_phi_row)
            job.right_row = int(self.right_phi_row)
            job.step = int(knot)
            job.obj_pos = self.buf.obj_pos
            job.obj_quat = self.buf.obj_quat
            job.left = self.buf.left
            job.right = self.buf.right
            job.capsules = self.buf.capsules
            job.phi = self.buf.phi
            wp.launch(loss_k.extract_world, dim=self.num_samples, inputs=[job])
        batch = loss_k.BatchJob()
        batch.obj_pos = self.buf.obj_pos
        batch.obj_quat = self.buf.obj_quat
        batch.left = self.buf.left
        batch.right = self.buf.right
        batch.phi = self.buf.phi
        batch.capsules = self.buf.capsules
        batch.cmd = self.buf.cmd
        batch.contact_local = self.buf.contact_local
        batch.normal_local = self.buf.normal_local
        batch.target_pos = self.buf.target_pos
        batch.target_quat = self.buf.target_quat
        batch.flags = self.buf.flags
        batch.has_capsules = 1
        batch.cmd_dim = self.n_action
        batch.horizon = self.horizon
        batch.weights = weights_struct(self.weights)
        batch.loss = self.buf.loss
        wp.launch(loss_k.evaluate_batch, dim=(self.num_samples, self.horizon), inputs=[batch])

    def _ensure_graph(self):
        if self._graph is not None or not str(self.device).startswith("cuda"):
            return
        wp = ensure_warp()
        self._forward_and_cost()
        wp.synchronize()
        with wp.ScopedCapture(device=self.device) as capture:
            self._forward_and_cost()
        self._graph = capture.graph

    def _rollout(self, raw):
        raw = np.asarray(raw, dtype=np.float32).reshape(self.num_samples, self.horizon * self.n_action)
        cmd = command_from_raw(raw.reshape(self.num_samples, self.horizon, self.n_action), self.u_limit)
        self.buf.raw.assign(raw)
        self.buf.cmd.assign(cmd.astype(np.float32))
        wp = ensure_warp()
        self._ensure_graph()
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            self._forward_and_cost()
        wp.synchronize()
        return np.asarray(self.buf.loss.numpy(), dtype=np.float64).reshape(self.num_samples)

    def plan_once(self, target_p=None, target_q=None, curr_x=None, sol_guess=None, **kwargs):
        del curr_x, sol_guess
        from planning.spider_mppi import compute_spider_weights
        import torch

        started = time.perf_counter()
        offset = kwargs.get("approach_offset")
        if offset is not None:
            self.weights.approach_offset = float(offset)
        model = self.model
        qpos0 = kwargs.get("qpos0")
        qvel0 = kwargs.get("qvel0")
        if qpos0 is None:
            raise ValueError("BigraspWarpMPPI.plan_once requires qpos0")
        if qvel0 is None:
            qvel0 = np.zeros(model.nv, dtype=np.float32)
        model.broadcast_state(np.asarray(qpos0, dtype=np.float32).reshape(model.nq), qvel0)
        contacts = np.asarray(kwargs.get("contact_points_local"), dtype=np.float32).reshape(2, 3)
        normals = kwargs.get("normals_local")
        if normals is None:
            normals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32).reshape(2, 3)
        self.buf.contact_local.assign(contacts)
        self.buf.normal_local.assign(normals)
        flags = np.zeros(2, dtype=np.int32)
        if target_p is None:
            self.buf.target_pos.zero_()
            self.buf.target_quat.assign(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        else:
            flags[0] = 1
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) if target_q is None else np.asarray(target_q, dtype=np.float32).reshape(4)
            self.buf.target_pos.assign(np.asarray(target_p, dtype=np.float32).reshape(1, 3))
            self.buf.target_quat.assign(quat.reshape(1, 4))
        self.buf.flags.assign(flags)
        mean = self._mean.copy()
        last_cost = np.zeros(self.num_samples, dtype=np.float64)
        for iteration in range(self.iterations):
            scale = self.noise_scale * (self.beta ** iteration)
            samples = mean.reshape(1, self.horizon, self.n_action) + scale * np.random.randn(
                self.num_samples, self.horizon, self.n_action
            ).astype(np.float32)
            samples[0] = mean
            last_cost = self._rollout(samples)
            rewards = torch.as_tensor(-last_cost, dtype=torch.float32)
            weights = compute_spider_weights(rewards, self.temperature).detach().cpu().numpy()
            mean = np.sum(weights.reshape(-1, 1, 1) * samples, axis=0).astype(np.float32)
            np.clip(mean, -8.0, 8.0, out=mean)
        self._mean = mean
        cmd = command_from_raw(mean, self.u_limit)
        action = cmd[0].copy()
        if self.horizon > 1:
            shifted = np.concatenate((mean[1:], mean[-1:]), axis=0)
        else:
            shifted = mean
        self._mean = shifted.astype(np.float32)
        finite = last_cost[np.isfinite(last_cost)]
        best = float(finite.min()) if finite.size else float("inf")
        return {
            "action": action.astype(np.float64),
            "u_traj": cmd.astype(np.float64),
            "ctrl": cmd.astype(np.float64),
            "cost": best,
            "cost_opt": np.array([best], dtype=np.float64),
            "solve_time": float(time.perf_counter() - started),
            "solver_backend": self.solver_backend,
            "contact_mask": np.zeros((self.horizon, 2), dtype=bool),
            "normal_force": np.zeros((self.horizon, 2), dtype=np.float64),
            "sol_guess": {"raw": mean.reshape(1, -1).astype(np.float32)},
        }
