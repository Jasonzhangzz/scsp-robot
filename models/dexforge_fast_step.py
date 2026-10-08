"""DexForge ``compile_fast_step`` wrapper with a torch autograd bridge."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch


_COMFREE_CANDIDATES = (
    os.environ.get("COMFREE_WARP_ROOT"),
    "/home/zz/GraspSONIC/thirdparty/DexForge/forceaware/third_party/comfree_warp",
)


def ensure_comfree_warp():
    """Put the bundled DexForge comfree_warp package on ``sys.path``."""
    for candidate in _COMFREE_CANDIDATES:
        if not candidate:
            continue
        root = Path(candidate)
        if not (root / "comfree_warp" / "__init__.py").is_file():
            continue
        resolved = str(root)
        os.environ.setdefault("COMFREE_WARP_ROOT", resolved)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
        return resolved
    raise ImportError(
        "comfree_warp not found. Set COMFREE_WARP_ROOT to "
        "DexForge/forceaware/third_party/comfree_warp"
    )


def _wp_to_torch(array, device):
    host = np.asarray(array.numpy(), dtype=np.float32)
    return torch.as_tensor(host, device=device, dtype=torch.float32)


def _as_host_numpy(value, dtype=np.float32):
    """DexForge-style host copy: Warp ``assign`` only accepts NumPy, never CUDA tensors."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _dbg(hypothesis_id, location, message, **data):
    # #region agent log
    try:
        import json
        import time

        with open("/home/zz/scsp-robot/.cursor/debug-a4aad5.log", "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "sessionId": "a4aad5",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except OSError:
        return
    # #endregion


class DexForgeFastStep:
    """One compiled DexForge native step plus FK extras for BigRasp costs."""

    n_planner_qpos = 21
    n_planner_qvel = 20
    n_cmd = 14

    def __init__(self, xml_path, device=None, contact_topk=1):
        ensure_comfree_warp()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(key, "1")

        import warp as wp
        from comfree_warp import CollisionConfig, configure_collision, load_model
        from comfree_warp.native_adjoint.dynamics import DynamicsInput
        from comfree_warp.native_adjoint.fast_step import compile_fast_step
        from comfree_warp.native_adjoint.fast_step_adjoint import allocate_workspace
        from comfree_warp.native_adjoint.runtime import StepInput

        self._wp = wp
        self._DynamicsInput = DynamicsInput
        self._StepInput = StepInput
        wp.init()
        if device is None:
            device = "cuda:0" if "cuda:0" in [str(d) for d in wp.get_devices()] else "cpu"
        self.device_name = str(device)
        self.torch_device = torch.device("cpu")
        self.device_model, self.cpu_model = load_model(str(xml_path), device=self.device_name)
        configure_collision(self.device_model, CollisionConfig(contact_topk=int(contact_topk)))
        self.compiled = compile_fast_step(self.cpu_model, self.device_model)
        self.workspace = allocate_workspace(self.compiled, 1)
        self.nq = int(self.cpu_model.nq)
        self.nv = int(self.cpu_model.nv)
        self.nu = int(self.cpu_model.nu)
        self.nbody = int(self.cpu_model.nbody)
        self._ids = self._resolve_ids()
        self.qpos = wp.zeros((1, self.nq), dtype=float, device=self.device_name, requires_grad=True)
        self.qvel = wp.zeros((1, self.nv), dtype=float, device=self.device_name, requires_grad=True)
        self.ctrl = wp.zeros((1, self.nu), dtype=float, device=self.device_name, requires_grad=True)
        self.force = wp.zeros((1, self.nv), dtype=float, device=self.device_name, requires_grad=True)
        self.time = wp.zeros(1, dtype=float, device=self.device_name, requires_grad=True)
        self.softness = wp.zeros(1, dtype=float, device=self.device_name, requires_grad=True)
        self._slots = []
        self._init_qpos = self.qpos
        self._init_qvel = self.qvel
        # #region agent log
        _dbg(
            "E",
            "dexforge_fast_step.py:__init__",
            "compile_fast_step finished",
            warp_device=self.device_name,
            torch_device=str(self.torch_device),
            nq=self.nq,
            nu=self.nu,
        )
        # #endregion

    def _resolve_ids(self):
        import mujoco

        def body(name):
            return int(mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_BODY, name))

        def joint(name):
            return int(mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_JOINT, name))

        def actuator(name):
            return int(mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))

        obj_jnt = joint("obj_freejoint")
        left_jnt = [joint(f"left_joint{i}") for i in range(1, 8)]
        right_jnt = [joint(f"right_joint{i}") for i in range(1, 8)]

        def site(name, default_body, default_offset):
            sid = int(mujoco.mj_name2id(self.cpu_model, mujoco.mjtObj.mjOBJ_SITE, name))
            if sid < 0:
                return default_body, np.asarray(default_offset, dtype=np.float32)
            return (
                int(self.cpu_model.site_bodyid[sid]),
                np.asarray(self.cpu_model.site_pos[sid], dtype=np.float32),
            )

        left_tip_body, left_tip_off = site("left_tip_center", body("left_attachment"), (0.0, 0.0, 0.06))
        right_tip_body, right_tip_off = site("right_tip_center", body("right_attachment"), (0.0, 0.0, 0.06))
        return {
            "obj_body": body("obj"),
            "left_tip": left_tip_body,
            "right_tip": right_tip_body,
            "left_tip_offset": left_tip_off,
            "right_tip_offset": right_tip_off,
            "left_forearm": body("left_link4"),
            "right_forearm": body("right_link4"),
            "left_wrist": body("left_link6"),
            "right_wrist": body("right_link6"),
            "obj_qpos": int(self.cpu_model.jnt_qposadr[obj_jnt]),
            "left_qpos": np.array([self.cpu_model.jnt_qposadr[j] for j in left_jnt], dtype=np.int32),
            "right_qpos": np.array([self.cpu_model.jnt_qposadr[j] for j in right_jnt], dtype=np.int32),
            "obj_dof": int(self.cpu_model.jnt_dofadr[obj_jnt]),
            "left_dof": np.array([self.cpu_model.jnt_dofadr[j] for j in left_jnt], dtype=np.int32),
            "right_dof": np.array([self.cpu_model.jnt_dofadr[j] for j in right_jnt], dtype=np.int32),
            "left_act": np.array([actuator(f"left_actuator{i}") for i in range(1, 8)], dtype=np.int32),
            "right_act": np.array([actuator(f"right_actuator{i}") for i in range(1, 8)], dtype=np.int32),
        }

    def planner_state_from_qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        ids = self._ids
        return np.concatenate(
            (
                qpos[ids["obj_qpos"] : ids["obj_qpos"] + 7],
                qpos[ids["left_qpos"]],
                qpos[ids["right_qpos"]],
            )
        )

    def tip_phi_rows(self):
        """Contact-distance rows for the left and right fingertip query geoms."""
        collision = getattr(getattr(self.compiled, "base", None), "collision", None)
        rows_src = getattr(collision, "output_contact_rows", None)
        bodies_src = getattr(collision, "output_body_ids", None)
        if rows_src is None or bodies_src is None:
            return -1, -1
        rows = np.asarray(rows_src.numpy(), dtype=np.int32).reshape(-1)
        bodies = np.asarray(bodies_src.numpy(), dtype=np.int32).reshape(-1)

        def pick(body):
            for index, owner in enumerate(bodies):
                if int(owner) == int(body) and int(rows[index]) >= 0:
                    return int(rows[index])
            return -1

        ids = self._ids
        return pick(ids["left_tip"]), pick(ids["right_tip"])

    def fill_qpos(self, planner_x, base=None):
        qpos = np.zeros(self.nq, dtype=np.float32) if base is None else _as_host_numpy(base).copy()
        planner_x = _as_host_numpy(planner_x).reshape(21)
        ids = self._ids
        qpos[ids["obj_qpos"] : ids["obj_qpos"] + 7] = planner_x[:7]
        qpos[ids["left_qpos"]] = planner_x[7:14]
        qpos[ids["right_qpos"]] = planner_x[14:21]
        return qpos

    def fill_qvel(self, planner_xd=None, base=None):
        qvel = np.zeros(self.nv, dtype=np.float32) if base is None else _as_host_numpy(base).copy()
        if planner_xd is None:
            return qvel
        planner_xd = _as_host_numpy(planner_xd).reshape(-1)
        ids = self._ids
        if planner_xd.size >= 20:
            qvel[ids["obj_dof"] : ids["obj_dof"] + 6] = planner_xd[:6]
            qvel[ids["left_dof"]] = planner_xd[6:13]
            qvel[ids["right_dof"]] = planner_xd[13:20]
        return qvel

    def fill_ctrl(self, qpos, joint_delta):
        ctrl = np.zeros(self.nu, dtype=np.float32)
        ids = self._ids
        delta = _as_host_numpy(joint_delta).reshape(14)
        left = _as_host_numpy(qpos)[ids["left_qpos"]] + delta[:7]
        right = _as_host_numpy(qpos)[ids["right_qpos"]] + delta[7:]
        ranges = np.asarray(self.cpu_model.actuator_ctrlrange, dtype=np.float32)
        left = np.clip(left, ranges[ids["left_act"], 0], ranges[ids["left_act"], 1])
        right = np.clip(right, ranges[ids["right_act"], 0], ranges[ids["right_act"], 1])
        ctrl[ids["left_act"]] = left
        ctrl[ids["right_act"]] = right
        return ctrl

    def _allocate_cotangent(self):
        from comfree_warp.native_adjoint.fast_step_adjoint import StepCotangent

        wp = self._wp
        base = self.compiled.base
        device = self.device_name
        return StepCotangent(
            wp.zeros((1, base.integration.position_count), dtype=float, device=device),
            wp.zeros((1, base.integration.velocity_count), dtype=float, device=device),
            wp.zeros((1, base.integration.velocity_count), dtype=float, device=device),
            wp.zeros(1, dtype=float, device=device),
            wp.zeros((1, base.integration.velocity_count), dtype=float, device=device),
            wp.zeros((1, base.kinematics.body_count), dtype=wp.vec3, device=device),
            wp.zeros((1, base.kinematics.body_count), dtype=wp.mat33, device=device),
            wp.zeros((1, base.collision.contact_count), dtype=float, device=device),
            wp.zeros((1, base.collision.contact_count), dtype=wp.vec3, device=device),
            wp.zeros((1, base.collision.contact_count), dtype=wp.mat33, device=device),
        )

    def allocate_graph_chain(self, horizon):
        """Record a DexForge pointer chain once; later graph iters only replay."""
        self._ensure_chain(int(horizon))
        return self._slots[: int(horizon)]

    def _step_input(self):
        dynamics = self._DynamicsInput()
        dynamics.qpos = self.qpos
        dynamics.qvel = self.qvel
        dynamics.ctrl = self.ctrl
        dynamics.qfrc_applied = self.force
        return self._StepInput(dynamics, self.time, self.softness)

    def _ensure_chain(self, horizon):
        """DexForge-style pointer chain: step[h+1].qpos is step[h].result.qpos."""
        if len(self._slots) >= int(horizon):
            return
        from types import SimpleNamespace

        from comfree_warp.collision_config import FREEZE_TANGENT_GAUGE_VJP
        from comfree_warp.native_adjoint.fast_step_adjoint import allocate_workspace
        from comfree_warp.native_adjoint.fast_step_adjoint import record

        wp = self._wp
        qpos = self._init_qpos
        qvel = self._init_qvel
        for _ in range(int(horizon) - len(self._slots)):
            ctrl = wp.zeros((1, self.nu), dtype=float, device=self.device_name, requires_grad=True)
            force = wp.zeros((1, self.nv), dtype=float, device=self.device_name, requires_grad=True)
            time = wp.zeros(1, dtype=float, device=self.device_name, requires_grad=True)
            softness = wp.zeros(1, dtype=float, device=self.device_name, requires_grad=True)
            dynamics = self._DynamicsInput()
            dynamics.qpos = qpos
            dynamics.qvel = qvel
            dynamics.ctrl = ctrl
            dynamics.qfrc_applied = force
            inputs = self._StepInput(dynamics, time, softness)
            workspace = allocate_workspace(self.compiled, 1)
            recorded = record(
                self.compiled, inputs, workspace, freeze_frame_vjp=FREEZE_TANGENT_GAUGE_VJP
            )
            cotangent = self._allocate_cotangent()
            self._slots.append(
                SimpleNamespace(
                    inputs=inputs,
                    workspace=workspace,
                    ctrl=ctrl,
                    force=force,
                    recorded=recorded,
                    cotangent=cotangent,
                )
            )
            qpos = recorded.result.qpos
            qvel = recorded.result.qvel
        # #region agent log
        _dbg("G", "dexforge_fast_step.py:_ensure_chain", "step chain ready", horizon=int(horizon), slots=len(self._slots))
        # #endregion

    def _assign_state(self, qpos, qvel, ctrl):
        self.qpos.assign(np.asarray(qpos, dtype=np.float32).reshape(1, self.nq))
        self.qvel.assign(np.asarray(qvel, dtype=np.float32).reshape(1, self.nv))
        self.ctrl.assign(np.asarray(ctrl, dtype=np.float32).reshape(1, self.nu))
        self.force.zero_()

    def extras_from_body(self, qpos, body_pos):
        ids = self._ids
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if body_pos.ndim == 3:
            body_pos = body_pos[0]
        obj = qpos[ids["obj_qpos"] : ids["obj_qpos"] + 7]
        left_tip = np.asarray(body_pos[ids["left_tip"]], dtype=np.float32)
        right_tip = np.asarray(body_pos[ids["right_tip"]], dtype=np.float32)
        planner = np.concatenate((obj, left_tip, right_tip)).astype(np.float32)
        capsules = np.stack(
            (
                np.asarray(body_pos[ids["left_forearm"]], dtype=np.float32),
                left_tip,
                np.asarray(body_pos[ids["right_forearm"]], dtype=np.float32),
                right_tip,
            ),
            axis=0,
        )
        return planner, capsules

    def step_torch(self, qpos, qvel, ctrl):
        model = self

        class _Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, qpos_t, qvel_t, ctrl_t):
                from comfree_warp.native_adjoint.fast_step_adjoint import record

                qpos_np = qpos_t.detach().float().cpu().numpy()
                qvel_np = qvel_t.detach().float().cpu().numpy()
                ctrl_np = ctrl_t.detach().float().cpu().numpy()
                model._assign_state(qpos_np, qvel_np, ctrl_np)
                recorded = record(model.compiled, model._step_input(), model.workspace)
                model._wp.synchronize()
                next_qpos = _wp_to_torch(recorded.result.qpos, qpos_t.device).reshape(model.nq)
                next_qvel = _wp_to_torch(recorded.result.qvel, qpos_t.device).reshape(model.nv)
                body = _wp_to_torch(recorded.result.body_position, qpos_t.device)
                if body.ndim == 3 and body.shape[0] == 1:
                    body = body[0]
                mat = _wp_to_torch(recorded.result.body_matrix, qpos_t.device)
                if mat.ndim == 4 and mat.shape[0] == 1:
                    mat = mat[0]
                dist = _wp_to_torch(recorded.result.contact_distance, qpos_t.device)
                ctx.recorded = recorded
                ctx.qpos_dev = qpos_t.device
                return next_qpos, next_qvel, body, mat, dist

            @staticmethod
            def backward(ctx, g_qpos, g_qvel, g_body, g_mat, g_dist):
                from comfree_warp.native_adjoint.fast_step_adjoint import (
                    BackwardCall,
                    StepCotangent,
                    backward,
                )

                wp = model._wp
                rec = ctx.recorded
                result = rec.result

                def _seed(target, grad):
                    if grad is None:
                        host = np.zeros(np.asarray(target.numpy()).shape, dtype=np.float32)
                    else:
                        host = grad.detach().float().cpu().numpy()
                    target.grad.assign(np.asarray(host, dtype=np.float32).reshape(target.shape))

                try:
                    _seed(result.qpos, g_qpos)
                    _seed(result.qvel, g_qvel)
                    _seed(result.body_position, g_body)
                    _seed(result.body_matrix, g_mat)
                    _seed(result.contact_distance, g_dist)
                    cotangent = StepCotangent(
                        result.qpos.grad,
                        result.qvel.grad,
                        result.qacc.grad,
                        result.time.grad,
                        result.constraint_force.grad,
                        result.body_position.grad,
                        result.body_matrix.grad,
                        result.contact_distance.grad,
                        result.contact_position.grad,
                        result.contact_frame.grad,
                    )
                    grads = backward(
                        BackwardCall(model.compiled, model._step_input(), model.workspace, rec, cotangent)
                    )
                    wp.synchronize()
                    return (
                        _wp_to_torch(grads.qpos, ctx.qpos_dev).reshape(model.nq),
                        _wp_to_torch(grads.qvel, ctx.qpos_dev).reshape(model.nv),
                        _wp_to_torch(grads.ctrl, ctx.qpos_dev).reshape(model.nu),
                    )
                except Exception:
                    zeros_q = torch.zeros(model.nq, device=ctx.qpos_dev, dtype=torch.float32)
                    zeros_v = torch.zeros(model.nv, device=ctx.qpos_dev, dtype=torch.float32)
                    zeros_u = torch.zeros(model.nu, device=ctx.qpos_dev, dtype=torch.float32)
                    return zeros_q, zeros_v, zeros_u

        return _Fn.apply(qpos, qvel, ctrl)

    def rollout(self, planner_x, cmd_traj, planner_xd=None, qpos0=None, qvel0=None):
        import time as _time

        t0 = _time.perf_counter()
        qpos_np = self.fill_qpos(planner_x, base=qpos0)
        qvel_np = self.fill_qvel(planner_xd, base=qvel0)
        if torch.is_tensor(cmd_traj):
            tape_device = cmd_traj.device
            cmd_traj = cmd_traj.to(dtype=torch.float32)
        else:
            tape_device = self.torch_device
            cmd_traj = torch.as_tensor(cmd_traj, dtype=torch.float32, device=tape_device)
        qpos = torch.as_tensor(qpos_np, dtype=torch.float32, device=tape_device)
        ids = self._ids
        idx_q_left = torch.as_tensor(ids["left_qpos"], dtype=torch.long, device=tape_device)
        idx_q_right = torch.as_tensor(ids["right_qpos"], dtype=torch.long, device=tape_device)
        idx_u_left = torch.as_tensor(ids["left_act"], dtype=torch.long, device=tape_device)
        idx_u_right = torch.as_tensor(ids["right_act"], dtype=torch.long, device=tape_device)
        ctrl_traj = []
        q_arm = qpos
        for step in range(int(cmd_traj.shape[0])):
            ctrl = torch.zeros(self.nu, dtype=torch.float32, device=tape_device)
            ctrl[idx_u_left] = q_arm[idx_q_left] + cmd_traj[step, :7]
            ctrl[idx_u_right] = q_arm[idx_q_right] + cmd_traj[step, 7:]
            ctrl_traj.append(ctrl)
            q_arm = q_arm.clone()
            q_arm[idx_q_left] = ctrl[idx_u_left]
            q_arm[idx_q_right] = ctrl[idx_u_right]
        ctrl_traj = torch.stack(ctrl_traj, dim=0)
        states, capsules = self._rollout_horizon(qpos_np, qvel_np, ctrl_traj)
        extras = []
        phi = self._tip_phi(states)
        for step in range(states.shape[0]):
            extras.append(
                {
                    "capsules": capsules[step],
                    "phi": phi[step],
                    "normal": states.new_zeros(2, 3),
                    "contact_force": states.new_zeros(2),
                }
            )
        # #region agent log
        _dbg(
            "G",
            "dexforge_fast_step.py:rollout",
            "horizon rollout done",
            ms=round(1000.0 * (_time.perf_counter() - t0), 2),
            horizon=int(cmd_traj.shape[0]),
            tape_device=str(tape_device),
        )
        # #endregion
        return states, extras

    def _rollout_horizon(self, qpos_np, qvel_np, ctrl_traj):
        model = self
        horizon = int(ctrl_traj.shape[0])
        self._ensure_chain(horizon)

        class _HorizonFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, ctrls):
                from comfree_warp.collision_config import FREEZE_TANGENT_GAUGE_VJP
                from comfree_warp.native_adjoint.fast_step_adjoint import record

                model._init_qpos.assign(np.asarray(qpos_np, dtype=np.float32).reshape(1, model.nq))
                model._init_qvel.assign(np.asarray(qvel_np, dtype=np.float32).reshape(1, model.nv))
                ctrl_np = ctrls.detach().float().cpu().numpy()
                recorded = []
                for step in range(horizon):
                    slot = model._slots[step]
                    slot.ctrl.assign(np.asarray(ctrl_np[step], dtype=np.float32).reshape(1, model.nu))
                    slot.force.zero_()
                    recorded.append(
                        record(
                            model.compiled,
                            slot.inputs,
                            slot.workspace,
                            freeze_frame_vjp=FREEZE_TANGENT_GAUGE_VJP,
                        )
                    )
                model._wp.synchronize()
                states = []
                capsules = []
                for rec in recorded:
                    qpos_t = _wp_to_torch(rec.result.qpos, ctrls.device).reshape(model.nq)
                    body = _wp_to_torch(rec.result.body_position, ctrls.device)
                    mat = _wp_to_torch(rec.result.body_matrix, ctrls.device)
                    planner, caps = model._torch_extras(qpos_t, body, mat)
                    states.append(planner)
                    capsules.append(caps)
                ctx.recorded = recorded
                ctx.device = ctrls.device
                return torch.stack(states, dim=0), torch.stack(capsules, dim=0)

            @staticmethod
            def backward(ctx, g_states, g_capsules):
                from comfree_warp.native_adjoint.fast_step_adjoint import BackwardCall
                from comfree_warp.native_adjoint.fast_step_adjoint import StepCotangent
                from comfree_warp.native_adjoint.fast_step_adjoint import backward

                g_ctrl = []
                g_qpos = None
                g_qvel = None
                try:
                    for step in reversed(range(horizon)):
                        rec = ctx.recorded[step]
                        result = rec.result
                        slot = model._slots[step]
                        if g_qpos is None:
                            host_q = np.zeros(model.nq, dtype=np.float32)
                            host_v = np.zeros(model.nv, dtype=np.float32)
                        else:
                            host_q = g_qpos.detach().float().cpu().numpy().reshape(-1)
                            host_v = g_qvel.detach().float().cpu().numpy().reshape(-1)
                        if g_states is not None:
                            obj = g_states[step, :7].detach().float().cpu().numpy()
                            adr = int(model._ids["obj_qpos"])
                            host_q[adr : adr + 7] += obj
                        result.qpos.grad.assign(host_q.reshape(result.qpos.shape))
                        result.qvel.grad.assign(host_v.reshape(result.qvel.shape))
                        model._seed_tip_body_grad(
                            result,
                            None if g_states is None else g_states[step],
                            None if g_capsules is None else g_capsules[step],
                        )
                        cotangent = StepCotangent(
                            result.qpos.grad,
                            result.qvel.grad,
                            result.qacc.grad,
                            result.time.grad,
                            result.constraint_force.grad,
                            result.body_position.grad,
                            result.body_matrix.grad,
                            result.contact_distance.grad,
                            result.contact_position.grad,
                            result.contact_frame.grad,
                        )
                        grads = backward(
                            BackwardCall(model.compiled, slot.inputs, slot.workspace, rec, cotangent)
                        )
                        g_ctrl.append(_wp_to_torch(grads.ctrl, ctx.device).reshape(model.nu))
                        g_qpos = _wp_to_torch(grads.qpos, ctx.device).reshape(model.nq)
                        g_qvel = _wp_to_torch(grads.qvel, ctx.device).reshape(model.nv)
                    model._wp.synchronize()
                    return torch.stack(list(reversed(g_ctrl)), dim=0)
                except Exception:
                    return torch.zeros(horizon, model.nu, device=ctx.device, dtype=torch.float32)

        return _HorizonFn.apply(ctrl_traj)

    def _torch_extras(self, qpos, body, mat=None):
        ids = self._ids
        obj = qpos[ids["obj_qpos"] : ids["obj_qpos"] + 7]
        if body.ndim == 3 and body.shape[0] == 1:
            body = body[0]
        if mat is not None and mat.ndim == 4 and mat.shape[0] == 1:
            mat = mat[0]
        left_tip = self._site_world(body, mat, ids["left_tip"], ids["left_tip_offset"])
        right_tip = self._site_world(body, mat, ids["right_tip"], ids["right_tip_offset"])
        planner = torch.cat((obj, left_tip, right_tip), dim=-1)
        capsules = torch.stack(
            (
                body[ids["left_forearm"]],
                left_tip,
                body[ids["right_forearm"]],
                right_tip,
            ),
            dim=0,
        )
        return planner, capsules

    def _seed_tip_body_grad(self, result, g_state, g_capsules):
        body_g = np.zeros(np.asarray(result.body_position.numpy()).shape, dtype=np.float32)
        mat_g = np.zeros(np.asarray(result.body_matrix.numpy()).shape, dtype=np.float32)

        def _add(body_id, offset, grad):
            if grad is None:
                return
            grad = np.asarray(grad, dtype=np.float32).reshape(3)
            offset = np.asarray(offset, dtype=np.float32).reshape(3)
            if body_g.ndim == 3:
                body_g[0, body_id] += grad
            else:
                body_g[body_id] += grad
            outer = np.outer(grad, offset)
            if mat_g.ndim == 4:
                mat_g[0, body_id] += outer
            else:
                mat_g[body_id] += outer

        ids = self._ids
        if g_state is not None:
            gs = g_state.detach().float().cpu().numpy()
            _add(ids["left_tip"], ids["left_tip_offset"], gs[7:10])
            _add(ids["right_tip"], ids["right_tip_offset"], gs[10:13])
        if g_capsules is not None:
            gc = g_capsules.detach().float().cpu().numpy()
            _add(ids["left_forearm"], (0.0, 0.0, 0.0), gc[0])
            _add(ids["left_tip"], ids["left_tip_offset"], gc[1])
            _add(ids["right_forearm"], (0.0, 0.0, 0.0), gc[2])
            _add(ids["right_tip"], ids["right_tip_offset"], gc[3])
        result.body_position.grad.assign(body_g)
        result.body_matrix.grad.assign(mat_g)

    def _site_world(self, body, mat, body_id, offset):
        pos = body[body_id]
        if mat is None:
            return pos
        rot = mat[body_id]
        off = torch.as_tensor(offset, dtype=pos.dtype, device=pos.device)
        return pos + rot @ off

    def _tip_phi(self, planner_state):
        from models.comfree_gs_torch import gs_sphere_features, quat_rotate_wxyz

        obj_pos = planner_state[..., 0:3]
        obj_quat = planner_state[..., 3:7]
        tips = torch.stack((planner_state[..., 7:10], planner_state[..., 10:13]), dim=-2)
        spheres = getattr(self, "spheres_local", None)
        if spheres is None:
            return planner_state.new_zeros(tips.shape[-2])
        spheres = spheres.to(device=planner_state.device, dtype=planner_state.dtype)
        world = quat_rotate_wxyz(obj_quat.unsqueeze(-2), spheres[..., :3]) + obj_pos.unsqueeze(-2)
        packed = torch.cat((world, spheres[..., 3:4]), dim=-1)
        phi, _, _ = gs_sphere_features(tips, packed, query_radius=0.01)
        return phi

    def set_cloud(self, cloud):
        spheres = np.concatenate((cloud.centers, cloud.radii[:, None]), axis=1).astype(np.float32)
        self.spheres_local = torch.as_tensor(spheres, dtype=torch.float32, device=self.torch_device)
        return self
