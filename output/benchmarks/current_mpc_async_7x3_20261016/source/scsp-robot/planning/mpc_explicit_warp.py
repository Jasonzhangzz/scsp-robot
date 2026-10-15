"""Warp CUDA-graph Adam MPC over DexForge ``compile_fast_step``."""

from types import SimpleNamespace

import numpy as np

from models.dexforge_fast_step import _as_host_numpy
from planning.bigrasp_warp_loss import ensure_kernels as ensure_loss, weights_struct
from planning.mpc_explicit_adam import BigraspGSCostWeights
from planning.warp_adam import DEXFORGE_GRAD_CLIP, LOSS_GRADIENT_SCALE
from planning.warp_adam import ensure_kernels as ensure_adam, ensure_warp, launch_adam


_KERNELS = None


def ensure_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    wp = ensure_warp()

    @wp.kernel(enable_backward=False)
    def decode_step(
        raw: wp.array2d(dtype=float),
        qpos0: wp.array2d(dtype=float),
        ctrl: wp.array2d(dtype=float),
        cmd: wp.array2d(dtype=float),
        qadr: wp.array(dtype=int),
        uadr: wp.array(dtype=int),
        ranges: wp.array2d(dtype=float),
        u_limit: float,
        step: int,
        n_action: int,
        knot_substeps: int,
    ):
        joint = wp.tid()
        knot = step // wp.max(knot_substeps, 1)
        acc = float(0.0)
        for index in range(knot + 1):
            acc += u_limit * wp.tanh(raw[0, index * n_action + joint])
        q = qadr[joint]
        u = uadr[joint]
        value = wp.clamp(qpos0[0, q] + acc, ranges[u, 0], ranges[u, 1])
        ctrl[0, u] = value
        if step - knot * knot_substeps == 0:
            cmd[knot, joint] = u_limit * wp.tanh(raw[0, knot * n_action + joint])

    @wp.kernel(enable_backward=False)
    def pack_ctrl_grad(
        source: wp.array2d(dtype=float),
        target: wp.array2d(dtype=float),
        step: int,
    ):
        index = wp.tid()
        target[step, index] = source[0, index]

    @wp.kernel(enable_backward=False)
    def map_raw_grad(
        raw: wp.array2d(dtype=float),
        g_raw: wp.array2d(dtype=float),
        g_cmd: wp.array2d(dtype=float),
        g_ctrl: wp.array2d(dtype=float),
        uadr: wp.array(dtype=int),
        u_limit: float,
        horizon: int,
        n_action: int,
        knot_substeps: int,
    ):
        joint = wp.tid()
        dense_steps = horizon * wp.max(knot_substeps, 1)
        for knot in range(horizon):
            tangent = wp.tanh(raw[0, knot * n_action + joint])
            deriv = u_limit * (1.0 - tangent * tangent)
            acc = g_cmd[knot, joint]
            for dense in range(knot * knot_substeps, dense_steps):
                acc += g_ctrl[dense, uadr[joint]]
            g_raw[0, knot * n_action + joint] = deriv * acc

    @wp.kernel(enable_backward=False)
    def zero_cotangent_q(
        qpos: wp.array2d(dtype=float),
        qvel: wp.array2d(dtype=float),
        qacc: wp.array2d(dtype=float),
        force: wp.array2d(dtype=float),
        time: wp.array(dtype=float),
    ):
        i, j = wp.tid()
        if j < qpos.shape[1]:
            qpos[i, j] = 0.0
        if j < qvel.shape[1]:
            qvel[i, j] = 0.0
        if j < qacc.shape[1]:
            qacc[i, j] = 0.0
        if j < force.shape[1]:
            force[i, j] = 0.0
        if i == 0 and j == 0:
            time[0] = 0.0

    _KERNELS = SimpleNamespace(
        wp=wp,
        decode_step=decode_step,
        pack_ctrl_grad=pack_ctrl_grad,
        map_raw_grad=map_raw_grad,
        zero_cotangent_q=zero_cotangent_q,
    )
    return _KERNELS


class BigraspWarpMPC:
    """One-world DexForge step chain + Warp Adam, captured as a CUDA graph."""

    solver_backend = "adam_warp"

    def __init__(
        self,
        fast_step,
        *,
        horizon=5,
        iters=40,
        lr=0.05,
        u_limit=0.15,
        knot_substeps=6,
        weights=None,
        cloud=None,
    ):
        self.model = fast_step
        self.horizon = max(1, int(horizon))
        self.iters = max(1, int(iters))
        self.lr = float(lr)
        self.lr_min = min(self.lr, 0.0015)
        self.u_limit = float(u_limit)
        self.knot_substeps = max(1, int(knot_substeps))
        self.dense_steps = self.horizon * self.knot_substeps
        self.weights = weights or BigraspGSCostWeights()
        self.device = fast_step.device_name
        self.n_action = 14
        self.steps = fast_step.allocate_graph_chain(self.dense_steps)
        self._graph = None
        self.left_phi_row, self.right_phi_row = fast_step.tip_phi_rows()
        self._init_buffers(cloud)
        self._warm = None

    def _init_buffers(self, cloud):
        wp = ensure_warp()
        ensure_kernels()
        ensure_adam()
        ensure_loss()
        device = self.device
        h = self.horizon
        raw_dim = self.n_action * h
        ids = self.model._ids
        left_q = np.asarray(ids["left_qpos"], dtype=np.int32)
        right_q = np.asarray(ids["right_qpos"], dtype=np.int32)
        left_u = np.asarray(ids["left_act"], dtype=np.int32)
        right_u = np.asarray(ids["right_act"], dtype=np.int32)
        ranges = np.asarray(self.model.cpu_model.actuator_ctrlrange, dtype=np.float32)
        del cloud
        self.buf = SimpleNamespace(
            raw=wp.zeros((1, raw_dim), dtype=float, device=device),
            gradient=wp.zeros((1, raw_dim), dtype=float, device=device),
            first=wp.zeros((1, raw_dim), dtype=float, device=device),
            second=wp.zeros((1, raw_dim), dtype=float, device=device),
            grad_norm=wp.zeros(1, dtype=float, device=device),
            valid=wp.zeros(1, dtype=int, device=device),
            iteration=wp.zeros(1, dtype=int, device=device),
            best_raw=wp.zeros((1, raw_dim), dtype=float, device=device),
            best_loss=wp.zeros(1, dtype=float, device=device),
            best_available=wp.zeros(1, dtype=int, device=device),
            loss=wp.zeros(1, dtype=float, device=device),
            cmd=wp.zeros((h, self.n_action), dtype=float, device=device),
            g_cmd=wp.zeros((h, self.n_action), dtype=float, device=device),
            g_ctrl=wp.zeros((self.dense_steps, self.model.nu), dtype=float, device=device),
            qadr=wp.array(np.concatenate((left_q, right_q)), dtype=int, device=device),
            uadr=wp.array(np.concatenate((left_u, right_u)), dtype=int, device=device),
            ranges=wp.array(ranges, dtype=float, device=device),
            target_pos=wp.zeros(1, dtype=wp.vec3, device=device),
            target_quat=wp.zeros(1, dtype=wp.vec4, device=device),
            flags=wp.zeros(2, dtype=int, device=device),
            contact_local=wp.zeros(2, dtype=wp.vec3, device=device),
            normal_local=wp.zeros(2, dtype=wp.vec3, device=device),
            desired=wp.zeros(2, dtype=wp.vec3, device=device),
            obj_pos=wp.zeros(h, dtype=wp.vec3, device=device),
            obj_quat=wp.zeros(h, dtype=wp.vec4, device=device),
            left=wp.zeros(h, dtype=wp.vec3, device=device),
            right=wp.zeros(h, dtype=wp.vec3, device=device),
            capsules=wp.zeros((h, 6), dtype=wp.vec3, device=device),
            phi=wp.zeros((h, 2), dtype=float, device=device),
            inward=wp.zeros((h, 2), dtype=wp.vec3, device=device),
            contact_force=wp.zeros((h, 2), dtype=float, device=device),
            g_obj_pos=wp.zeros(h, dtype=wp.vec3, device=device),
            g_obj_quat=wp.zeros(h, dtype=wp.vec4, device=device),
            g_left=wp.zeros(h, dtype=wp.vec3, device=device),
            g_right=wp.zeros(h, dtype=wp.vec3, device=device),
            g_phi=wp.zeros((h, 2), dtype=float, device=device),
            g_capsules=wp.zeros((h, 6), dtype=wp.vec3, device=device),
        )
        self.buf.valid.fill_(1)
        self.buf.target_quat.assign(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    def _zero_cotangents(self):
        k = ensure_kernels()
        loss_k = ensure_loss()
        wp = k.wp
        nbody = int(self.model.nbody)
        for slot in self.steps:
            cot = slot.cotangent
            wp.launch(
                k.zero_cotangent_q,
                dim=cot.qpos.shape,
                inputs=[cot.qpos, cot.qvel, cot.qacc, cot.constraint_force, cot.time],
            )
            wp.launch(loss_k.zero_vec3_2d, dim=cot.body_position.shape, inputs=[cot.body_position])
            wp.launch(loss_k.zero_mat33_2d, dim=cot.body_matrix.shape, inputs=[cot.body_matrix])
            if cot.contact_distance.size:
                cot.contact_distance.zero_()
            if cot.contact_position.size:
                wp.launch(
                    loss_k.zero_vec3_2d,
                    dim=cot.contact_position.shape,
                    inputs=[cot.contact_position],
                )
            if cot.contact_frame.size:
                wp.launch(
                    loss_k.zero_mat33_2d,
                    dim=cot.contact_frame.shape,
                    inputs=[cot.contact_frame],
                )
            nbody  # keep

    def _decode(self):
        k = ensure_kernels()
        b = self.buf
        for step, slot in enumerate(self.steps):
            slot.force.zero_()
            k.wp.launch(
                k.decode_step,
                dim=self.n_action,
                inputs=[
                    b.raw,
                    self.model.qpos,
                    slot.ctrl,
                    b.cmd,
                    b.qadr,
                    b.uadr,
                    b.ranges,
                    self.u_limit,
                    step,
                    self.n_action,
                    self.knot_substeps,
                ],
            )

    def _forward(self):
        from comfree_warp.native_adjoint.fast_step_adjoint import forward as step_forward

        for slot in self.steps:
            step_forward(self.model.compiled, slot.inputs, slot.workspace)

    def _extract_and_loss(self):
        loss_k = ensure_loss()
        wp = loss_k.wp
        b = self.buf
        ids = self.model._ids
        b.loss.zero_()
        for knot in range(self.horizon):
            slot = self.steps[(knot + 1) * self.knot_substeps - 1]
            result = slot.recorded.result
            job = loss_k.ExtractJob()
            job.qpos = result.qpos
            job.body_pos = result.body_position
            job.body_mat = result.body_matrix
            job.obj_qpos = int(ids["obj_qpos"])
            job.left_tip = int(ids["left_tip"])
            job.right_tip = int(ids["right_tip"])
            job.left_forearm = int(ids["left_forearm"])
            job.right_forearm = int(ids["right_forearm"])
            job.left_wrist = int(ids["left_wrist"])
            job.right_wrist = int(ids["right_wrist"])
            left_off = np.asarray(ids["left_tip_offset"], dtype=np.float32).reshape(3)
            right_off = np.asarray(ids["right_tip_offset"], dtype=np.float32).reshape(3)
            job.left_off = wp.vec3(float(left_off[0]), float(left_off[1]), float(left_off[2]))
            job.right_off = wp.vec3(float(right_off[0]), float(right_off[1]), float(right_off[2]))
            job.contact_distance = result.contact_distance
            job.contact_frame = result.contact_frame
            job.left_row = int(self.left_phi_row)
            job.right_row = int(self.right_phi_row)
            job.step = int(knot)
            job.obj_pos = b.obj_pos
            job.obj_quat = b.obj_quat
            job.left = b.left
            job.right = b.right
            job.capsules = b.capsules
            job.phi = b.phi
            job.inward = b.inward
            wp.launch(loss_k.extract_step, dim=1, inputs=[job])
        horizon_job = loss_k.HorizonJob()
        horizon_job.obj_pos = b.obj_pos
        horizon_job.obj_quat = b.obj_quat
        horizon_job.left = b.left
        horizon_job.right = b.right
        horizon_job.phi = b.phi
        horizon_job.normal = b.inward
        horizon_job.contact_force = b.contact_force
        horizon_job.capsules = b.capsules
        horizon_job.cmd = b.cmd
        horizon_job.contact_local = b.contact_local
        horizon_job.normal_local = b.normal_local
        horizon_job.desired_force = b.desired
        horizon_job.target_pos = b.target_pos
        horizon_job.target_quat = b.target_quat
        horizon_job.flags = b.flags
        horizon_job.has_capsules = 1
        horizon_job.cmd_dim = self.n_action
        horizon_job.horizon = self.horizon
        horizon_job.weights = weights_struct(self.weights)
        horizon_job.loss = b.loss
        horizon_job.g_obj_pos = b.g_obj_pos
        horizon_job.g_obj_quat = b.g_obj_quat
        horizon_job.g_left = b.g_left
        horizon_job.g_right = b.g_right
        horizon_job.g_phi = b.g_phi
        horizon_job.g_cmd = b.g_cmd
        horizon_job.g_capsules = b.g_capsules
        wp.launch(loss_k.evaluate_horizon, dim=self.horizon, inputs=[horizon_job])

    def _seed(self):
        loss_k = ensure_loss()
        wp = loss_k.wp
        b = self.buf
        ids = self.model._ids
        left_off = np.asarray(ids["left_tip_offset"], dtype=np.float32).reshape(3)
        right_off = np.asarray(ids["right_tip_offset"], dtype=np.float32).reshape(3)
        for knot in range(self.horizon):
            slot = self.steps[(knot + 1) * self.knot_substeps - 1]
            job = loss_k.SeedJob()
            job.step = int(knot)
            job.obj_qpos = int(ids["obj_qpos"])
            job.left_tip = int(ids["left_tip"])
            job.right_tip = int(ids["right_tip"])
            job.left_forearm = int(ids["left_forearm"])
            job.right_forearm = int(ids["right_forearm"])
            job.left_wrist = int(ids["left_wrist"])
            job.right_wrist = int(ids["right_wrist"])
            job.left_off = wp.vec3(float(left_off[0]), float(left_off[1]), float(left_off[2]))
            job.right_off = wp.vec3(float(right_off[0]), float(right_off[1]), float(right_off[2]))
            job.g_obj_pos = b.g_obj_pos
            job.g_obj_quat = b.g_obj_quat
            job.g_left = b.g_left
            job.g_right = b.g_right
            job.g_phi = b.g_phi
            job.g_capsules = b.g_capsules
            job.inward = b.inward
            job.contact_grad = slot.cotangent.contact_distance
            job.left_row = int(self.left_phi_row)
            job.right_row = int(self.right_phi_row)
            job.qpos_grad = slot.cotangent.qpos
            job.body_pos_grad = slot.cotangent.body_position
            job.body_mat_grad = slot.cotangent.body_matrix
            wp.launch(loss_k.seed_cotangent, dim=1, inputs=[job])

    def _backward(self):
        from comfree_warp.native_adjoint.fast_step_adjoint import BackwardCall
        from comfree_warp.native_adjoint.fast_step_adjoint import backward as step_backward

        k = ensure_kernels()
        adam = ensure_adam()
        b = self.buf
        for dense in reversed(range(self.dense_steps)):
            slot = self.steps[dense]
            grads = step_backward(
                BackwardCall(
                    self.model.compiled,
                    slot.inputs,
                    slot.workspace,
                    slot.recorded,
                    slot.cotangent,
                )
            )
            k.wp.launch(
                k.pack_ctrl_grad,
                dim=self.model.nu,
                inputs=[grads.ctrl, b.g_ctrl, dense],
            )
            if dense > 0:
                prev = self.steps[dense - 1].cotangent
                k.wp.launch(adam.add_float2d, dim=grads.qpos.shape, inputs=[grads.qpos, prev.qpos])
                k.wp.launch(adam.add_float2d, dim=grads.qvel.shape, inputs=[grads.qvel, prev.qvel])
        k.wp.launch(
            k.map_raw_grad,
            dim=self.n_action,
            inputs=[
                b.raw,
                b.gradient,
                b.g_cmd,
                b.g_ctrl,
                b.uadr,
                self.u_limit,
                self.horizon,
                self.n_action,
                self.knot_substeps,
            ],
        )

    def _iteration(self):
        adam = ensure_adam()
        b = self.buf
        b.gradient.zero_()
        b.g_ctrl.zero_()
        b.contact_force.zero_()
        self._zero_cotangents()
        self._decode()
        self._forward()
        self._extract_and_loss()
        self._seed()
        self._backward()
        adam.wp.launch(
            adam.copy_if_better,
            dim=b.raw.shape,
            inputs=[b.loss, b.raw, b.best_loss, b.best_raw, b.best_available],
        )
        launch_adam(
            b.raw,
            b.gradient,
            b.first,
            b.second,
            b.grad_norm,
            b.valid,
            b.iteration,
            iterations=self.iters,
            learning_rate=self.lr,
            final_learning_rate=self.lr_min,
            epsilon=1.0e-8 * LOSS_GRADIENT_SCALE,
            clip=DEXFORGE_GRAD_CLIP * LOSS_GRADIENT_SCALE,
            gradient_scale=LOSS_GRADIENT_SCALE,
        )

    def _ensure_graph(self):
        if self._graph is not None:
            return
        wp = ensure_warp()
        self._iteration()
        wp.synchronize()
        self.buf.iteration.zero_()
        if not str(self.device).startswith("cuda"):
            return
        with wp.ScopedCapture(device=self.device) as capture:
            self._iteration()
        self._graph = capture.graph

    def _prepare(self, qpos, qvel, contacts, normals, target_p, target_q, desired, warm):
        wp = ensure_warp()
        qpos = _as_host_numpy(qpos).reshape(self.model.nq)
        qvel = _as_host_numpy(qvel).reshape(self.model.nv)
        self.model.qpos.assign(qpos.reshape(1, self.model.nq))
        self.model.qvel.assign(qvel.reshape(1, self.model.nv))
        contacts = _as_host_numpy(contacts, np.float32).reshape(2, 3)
        normals = _as_host_numpy(normals, np.float32).reshape(2, 3)
        self.buf.contact_local.assign(contacts)
        self.buf.normal_local.assign(normals)
        flags = np.zeros(2, dtype=np.int32)
        if desired is None or float(self.weights.force) == 0.0:
            self.buf.desired.zero_()
        else:
            flags[1] = 1
            self.buf.desired.assign(_as_host_numpy(desired, np.float32).reshape(2, 3))
        if target_p is None:
            self.buf.target_pos.zero_()
            self.buf.target_quat.assign(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        else:
            flags[0] = 1
            p = _as_host_numpy(target_p, np.float32).reshape(1, 3)
            q = (
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
                if target_q is None
                else _as_host_numpy(target_q, np.float32).reshape(1, 4)
            )
            self.buf.target_pos.assign(p)
            self.buf.target_quat.assign(q)
        self.buf.flags.assign(flags)
        if warm is None:
            raw = np.zeros((1, self.n_action * self.horizon), dtype=np.float32)
        else:
            raw = _as_host_numpy(warm, np.float32).reshape(1, self.n_action * self.horizon)
        self.buf.raw.assign(raw)
        self.buf.best_raw.assign(raw)
        self.buf.best_loss.fill_(1.0e30)
        self.buf.best_available.zero_()
        self.buf.first.zero_()
        self.buf.second.zero_()
        self.buf.iteration.zero_()
        self.buf.gradient.zero_()

    def _read_plan(self):
        wp = ensure_warp()
        adam = ensure_adam()
        wp.launch(
            adam.restore_best,
            dim=self.buf.raw.shape,
            inputs=[self.buf.raw, self.buf.best_raw, self.buf.best_available],
        )
        self._decode()
        self._forward()
        self._extract_and_loss()
        wp.synchronize()
        raw = np.asarray(self.buf.raw.numpy(), dtype=np.float32).reshape(self.horizon, self.n_action)
        cmd = self.u_limit * np.tanh(raw)
        left = np.asarray(self.buf.left.numpy(), dtype=np.float32).reshape(self.horizon, 3)
        right = np.asarray(self.buf.right.numpy(), dtype=np.float32).reshape(self.horizon, 3)
        obj = np.asarray(self.buf.obj_pos.numpy(), dtype=np.float32).reshape(self.horizon, 3)
        quat = np.asarray(self.buf.obj_quat.numpy(), dtype=np.float32).reshape(self.horizon, 4)
        rollout = np.concatenate((obj, quat, left, right), axis=1)
        cost = float(np.asarray(self.buf.loss.numpy()).reshape(-1)[0])
        return cmd.astype(np.float64), rollout.astype(np.float64), cost, raw

    def solve(
        self,
        qpos,
        qvel,
        *,
        contact_points_local,
        normals_local,
        target_p=None,
        target_q=None,
        desired_force_local=None,
        warm=None,
    ):
        import time

        t0 = time.perf_counter()
        self._prepare(
            qpos,
            qvel,
            contact_points_local,
            normals_local,
            target_p,
            target_q,
            desired_force_local,
            warm,
        )
        wp = ensure_warp()
        self._ensure_graph()
        steps = self.iters + 1
        if self._graph is not None:
            for _ in range(steps):
                wp.capture_launch(self._graph)
        else:
            for _ in range(steps):
                self._iteration()
        cmd, rollout, cost, raw = self._read_plan()
        self._warm = np.concatenate((cmd[1:], cmd[-1:]), axis=0)
        return {
            "action": cmd[0].copy(),
            "u_traj": cmd.copy(),
            "ctrl": cmd.copy(),
            "rollout_q": rollout.copy(),
            "cost": float(cost),
            "cost_opt": np.array([float(cost)], dtype=np.float64),
            "solve_time": float(time.perf_counter() - t0),
            "solver_backend": self.solver_backend,
            "sol_guess": {"u_traj": cmd.copy(), "raw": np.asarray(raw, dtype=np.float32).reshape(1, -1)},
            "contact_mask": np.zeros((self.horizon, 2), dtype=bool),
            "normal_force": np.zeros((self.horizon, 2), dtype=np.float64),
        }
