"""Warp kernels for the assigned-contact BigRasp MPC loss."""

from types import SimpleNamespace

import numpy as np

from planning.mpc_explicit_adam import BigraspGSCostWeights
from planning.warp_adam import ensure_warp


_KERNELS = None


def ensure_kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    wp = ensure_warp()

    @wp.func
    def _softplus(x: float) -> float:
        return wp.max(x, 0.0) + wp.log(1.0 + wp.exp(-wp.abs(x)))

    @wp.func
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + wp.exp(-x))

    @wp.func
    def _quat_rotate(q: wp.vec4, v: wp.vec3) -> wp.vec3:
        u = wp.vec3(q[1], q[2], q[3])
        t = wp.cross(u, v) * 2.0
        return v + t * q[0] + wp.cross(u, t)

    @wp.func
    def _segment_distance(a0: wp.vec3, a1: wp.vec3, b0: wp.vec3, b1: wp.vec3) -> float:
        u = a1 - a0
        v = b1 - b0
        w = a0 - b0
        uu = wp.max(wp.dot(u, u), 1.0e-8)
        vv = wp.max(wp.dot(v, v), 1.0e-8)
        uv = wp.dot(u, v)
        uw = wp.dot(u, w)
        vw = wp.dot(v, w)
        denom = wp.max(uu * vv - uv * uv, 1.0e-8)
        s = wp.clamp((uv * vw - vv * uw) / denom, 0.0, 1.0)
        t = wp.clamp((uu * vw - uv * uw) / denom, 0.0, 1.0)
        diff = w + u * s - v * t
        return wp.length(diff)

    @wp.struct
    class Weights:
        contact_attract: float
        contact_depth: float
        penetration: float
        object_position: float
        object_lateral: float
        object_orientation: float
        action: float
        smooth: float
        sync: float
        force: float
        swap: float
        inter_arm: float
        tip_sep: float
        tip_sep_min: float
        capsule_radius: float
        wrist_radius: float
        tip_sphere_radius: float
        capsule_margin: float
        gate_scale: float
        target_phi: float
        penetration_limit: float
        approach_offset: float
        query_mm: float
        terminal: float

    @wp.struct
    class HorizonJob:
        obj_pos: wp.array(dtype=wp.vec3)
        obj_quat: wp.array(dtype=wp.vec4)
        left: wp.array(dtype=wp.vec3)
        right: wp.array(dtype=wp.vec3)
        phi: wp.array2d(dtype=float)
        normal: wp.array(dtype=wp.vec3, ndim=2)
        contact_force: wp.array2d(dtype=float)
        capsules: wp.array(dtype=wp.vec3, ndim=2)
        cmd: wp.array2d(dtype=float)
        contact_local: wp.array(dtype=wp.vec3)
        normal_local: wp.array(dtype=wp.vec3)
        desired_force: wp.array(dtype=wp.vec3)
        target_pos: wp.array(dtype=wp.vec3)
        target_quat: wp.array(dtype=wp.vec4)
        flags: wp.array(dtype=int)
        has_capsules: int
        cmd_dim: int
        horizon: int
        weights: Weights
        loss: wp.array(dtype=float)
        g_obj_pos: wp.array(dtype=wp.vec3)
        g_obj_quat: wp.array(dtype=wp.vec4)
        g_left: wp.array(dtype=wp.vec3)
        g_right: wp.array(dtype=wp.vec3)
        g_phi: wp.array2d(dtype=float)
        g_cmd: wp.array2d(dtype=float)
        g_capsules: wp.array(dtype=wp.vec3, ndim=2)

    @wp.kernel(enable_backward=False)
    def evaluate_horizon(job: HorizonJob):
        step = wp.tid()
        w = job.weights
        obj_pos = job.obj_pos[step]
        obj_quat = job.obj_quat[step]
        left = job.left[step]
        right = job.right[step]
        c0 = _quat_rotate(obj_quat, job.contact_local[0]) + obj_pos
        c1 = _quat_rotate(obj_quat, job.contact_local[1]) + obj_pos
        n0 = _quat_rotate(obj_quat, job.normal_local[0])
        n1 = _quat_rotate(obj_quat, job.normal_local[1])
        out0 = -n0
        out1 = -n1
        t0 = c0 + out0 * w.approach_offset
        t1 = c1 + out1 * w.approach_offset
        mm = w.query_mm
        assigned0 = wp.length_sq((left - t0) * mm)
        assigned1 = wp.length_sq((right - t1) * mm)
        attract = assigned0 + assigned1
        swapped = wp.length_sq((left - t1) * mm) + wp.length_sq((right - t0) * mm)
        swap_gap = wp.max(attract - swapped, 0.0)
        swap = swap_gap * swap_gap
        d0 = wp.length(left - t0)
        d1 = wp.length(right - t1)
        gate_scale = wp.max(w.gate_scale, 1.0e-4)
        g0 = _sigmoid((gate_scale - d0) / gate_scale)
        g1 = _sigmoid((gate_scale - d1) / gate_scale)
        both = g0 * g1
        phi0 = job.phi[step, 0]
        phi1 = job.phi[step, 1]
        depth0 = wp.max(phi0 - w.target_phi, 0.0) * g0
        depth1 = wp.max(phi1 - w.target_phi, 0.0) * g1
        pen0 = wp.max(-phi0 - w.penetration_limit, 0.0)
        pen1 = wp.max(-phi1 - w.penetration_limit, 0.0)
        contact_depth = depth0 * depth0 + depth1 * depth1
        penetration = pen0 * pen0 + pen1 * pen1
        object_pos = float(0.0)
        object_lat = float(0.0)
        object_ori = float(0.0)
        target_pos = job.target_pos[0]
        target_quat = job.target_quat[0]
        if job.flags[0] == 1:
            object_pos = both * wp.length_sq(obj_pos - target_pos)
            dxy = wp.vec2(obj_pos[0] - target_pos[0], obj_pos[1] - target_pos[1])
            object_lat = both * wp.dot(dxy, dxy)
            object_ori = both * (1.0 - wp.dot(obj_quat, target_quat) * wp.dot(obj_quat, target_quat))
        left_progress = wp.dot(left - t0, -out0)
        right_progress = wp.dot(right - t1, -out1)
        sync = (left_progress - right_progress) * (left_progress - right_progress)
        action = float(0.0)
        for index in range(job.cmd_dim):
            action += job.cmd[step, index] * job.cmd[step, index]
        smooth = float(0.0)
        if step + 1 < job.horizon:
            for index in range(job.cmd_dim):
                diff = job.cmd[step + 1, index] - job.cmd[step, index]
                smooth += diff * diff
        force_cost = float(0.0)
        if job.flags[1] == 1:
            pred0 = job.normal[step, 0] * job.contact_force[step, 0]
            pred1 = job.normal[step, 1] * job.contact_force[step, 1]
            des0 = _quat_rotate(obj_quat, job.desired_force[0])
            des1 = _quat_rotate(obj_quat, job.desired_force[1])
            force_cost = wp.length_sq(pred0 - des0) + wp.length_sq(pred1 - des1)
        tip_over = wp.max(w.tip_sep_min - wp.length(left - right), 0.0) * mm
        tip_sep = tip_over
        inter_arm = float(0.0)
        g_c0 = wp.vec3(0.0, 0.0, 0.0)
        g_c1 = wp.vec3(0.0, 0.0, 0.0)
        g_c2 = wp.vec3(0.0, 0.0, 0.0)
        g_c3 = wp.vec3(0.0, 0.0, 0.0)
        g_c4 = wp.vec3(0.0, 0.0, 0.0)
        g_c5 = wp.vec3(0.0, 0.0, 0.0)
        if job.has_capsules == 1:
            r0 = w.capsule_radius
            r1 = w.wrist_radius
            r2 = w.tip_sphere_radius
            for left_i in range(3):
                a = job.capsules[step, left_i]
                ra = r0
                if left_i == 1:
                    ra = r1
                if left_i == 2:
                    ra = r2
                for right_i in range(3):
                    b = job.capsules[step, right_i + 3]
                    rb = r0
                    if right_i == 1:
                        rb = r1
                    if right_i == 2:
                        rb = r2
                    delta = a - b
                    dist = wp.max(wp.length(delta), 1.0e-8)
                    over = wp.max(ra + rb + w.capsule_margin - dist, 0.0) * mm
                    inter_arm += over * over
                    dpair = 2.0 * w.inter_arm * over * mm * (-1.0)
                    nrm = delta / dist
                    if left_i == 0:
                        g_c0 += nrm * dpair
                    elif left_i == 1:
                        g_c1 += nrm * dpair
                    else:
                        g_c2 += nrm * dpair
                    if right_i == 0:
                        g_c3 -= nrm * dpair
                    elif right_i == 1:
                        g_c4 -= nrm * dpair
                    else:
                        g_c5 -= nrm * dpair
            gap = _segment_distance(
                job.capsules[step, 0],
                job.capsules[step, 2],
                job.capsules[step, 3],
                job.capsules[step, 5],
            )
            cap_over = wp.max(2.0 * w.capsule_radius + w.capsule_margin - gap, 0.0) * mm
            inter_arm += cap_over * cap_over
        path = (
            w.contact_attract * attract
            + w.swap * swap
            + w.contact_depth * contact_depth
            + w.penetration * penetration
            + w.object_position * object_pos
            + w.object_lateral * object_lat
            + w.object_orientation * object_ori
            + w.action * action
            + w.smooth * smooth
            + w.sync * sync
            + w.force * force_cost
            + w.inter_arm * inter_arm
            + w.tip_sep * (tip_sep * tip_sep)
        )
        terminal = float(0.0)
        if step + 1 == job.horizon:
            terminal = w.terminal * (
                w.contact_attract * attract
                + w.swap * swap
                + w.object_position * object_pos
                + w.object_orientation * object_ori
            )
        wp.atomic_add(job.loss, 0, path + terminal)

        scale = 1.0
        if step + 1 == job.horizon:
            scale = 1.0 + w.terminal
        mm2 = mm * mm
        d_swap = w.swap * 2.0 * swap_gap
        if step + 1 == job.horizon:
            d_swap += w.terminal * w.swap * 2.0 * swap_gap
        d_attract = w.contact_attract * scale + d_swap
        d_swapped = -d_swap
        g_left = (left - t0) * (2.0 * d_attract * mm2) + (left - t1) * (2.0 * d_swapped * mm2)
        g_right = (right - t1) * (2.0 * d_attract * mm2) + (right - t0) * (2.0 * d_swapped * mm2)
        g_t0 = (t0 - left) * (2.0 * d_attract * mm2) + (t0 - right) * (2.0 * d_swapped * mm2)
        g_t1 = (t1 - right) * (2.0 * d_attract * mm2) + (t1 - left) * (2.0 * d_swapped * mm2)
        d_sync = 2.0 * w.sync * (left_progress - right_progress)
        g_left += (-out0) * d_sync
        g_right -= (-out1) * d_sync
        g_t0 -= (-out0) * d_sync
        g_t1 += (-out1) * d_sync
        sep_len = wp.max(wp.length(left - right), 1.0e-8)
        d_sep = 2.0 * w.tip_sep * tip_over * mm * (-1.0)
        g_left += ((left - right) / sep_len) * d_sep
        g_right -= ((left - right) / sep_len) * d_sep
        job.g_left[step] = g_left
        job.g_right[step] = g_right

        g_phi0 = 2.0 * w.contact_depth * depth0 * g0
        if phi0 - w.target_phi < 0.0:
            g_phi0 = 0.0
        g_phi1 = 2.0 * w.contact_depth * depth1 * g1
        if phi1 - w.target_phi < 0.0:
            g_phi1 = 0.0
        if -phi0 - w.penetration_limit > 0.0:
            g_phi0 += -2.0 * w.penetration * pen0
        if -phi1 - w.penetration_limit > 0.0:
            g_phi1 += -2.0 * w.penetration * pen1
        job.g_phi[step, 0] = g_phi0
        job.g_phi[step, 1] = g_phi1

        g_obj = g_t0 + g_t1
        if job.flags[0] == 1:
            g_obj += (obj_pos - target_pos) * (2.0 * w.object_position * both * scale)
            g_obj += wp.vec3(
                (obj_pos[0] - target_pos[0]) * (2.0 * w.object_lateral * both),
                (obj_pos[1] - target_pos[1]) * (2.0 * w.object_lateral * both),
                0.0,
            )
        job.g_obj_pos[step] = g_obj
        g_quat = wp.vec4(0.0, 0.0, 0.0, 0.0)
        if job.flags[0] == 1:
            align = wp.dot(obj_quat, target_quat)
            g_quat = target_quat * (-2.0 * align * w.object_orientation * both * scale)
        # Finite-difference rotate VJP for contact frames.
        eps = 1.0e-4
        for axis in range(4):
            q2 = obj_quat
            if axis == 0:
                q2 = wp.vec4(obj_quat[0] + eps, obj_quat[1], obj_quat[2], obj_quat[3])
            elif axis == 1:
                q2 = wp.vec4(obj_quat[0], obj_quat[1] + eps, obj_quat[2], obj_quat[3])
            elif axis == 2:
                q2 = wp.vec4(obj_quat[0], obj_quat[1], obj_quat[2] + eps, obj_quat[3])
            else:
                q2 = wp.vec4(obj_quat[0], obj_quat[1], obj_quat[2], obj_quat[3] + eps)
            c0p = _quat_rotate(q2, job.contact_local[0]) + obj_pos
            c1p = _quat_rotate(q2, job.contact_local[1]) + obj_pos
            out0p = -_quat_rotate(q2, job.normal_local[0])
            out1p = -_quat_rotate(q2, job.normal_local[1])
            t0p = c0p + out0p * w.approach_offset
            t1p = c1p + out1p * w.approach_offset
            deriv = wp.dot(g_t0, (t0p - t0) / eps) + wp.dot(g_t1, (t1p - t1) / eps)
            if axis == 0:
                g_quat += wp.vec4(deriv, 0.0, 0.0, 0.0)
            elif axis == 1:
                g_quat += wp.vec4(0.0, deriv, 0.0, 0.0)
            elif axis == 2:
                g_quat += wp.vec4(0.0, 0.0, deriv, 0.0)
            else:
                g_quat += wp.vec4(0.0, 0.0, 0.0, deriv)
        job.g_obj_quat[step] = g_quat

        for index in range(job.cmd_dim):
            g = 2.0 * w.action * job.cmd[step, index]
            if step + 1 < job.horizon:
                diff = job.cmd[step + 1, index] - job.cmd[step, index]
                g += -2.0 * w.smooth * diff
            if step > 0:
                diff = job.cmd[step, index] - job.cmd[step - 1, index]
                g += 2.0 * w.smooth * diff
            job.g_cmd[step, index] = g

        if job.has_capsules == 1:
            gap = _segment_distance(
                job.capsules[step, 0],
                job.capsules[step, 2],
                job.capsules[step, 3],
                job.capsules[step, 5],
            )
            cap_over = wp.max(2.0 * w.capsule_radius + w.capsule_margin - gap, 0.0) * mm
            d_gap = 2.0 * w.inter_arm * cap_over * mm * (-1.0)
            eps_c = 1.0e-4
            for slot in range(6):
                acc = wp.vec3(0.0, 0.0, 0.0)
                if slot == 0:
                    acc = g_c0
                elif slot == 1:
                    acc = g_c1
                elif slot == 2:
                    acc = g_c2
                elif slot == 3:
                    acc = g_c3
                elif slot == 4:
                    acc = g_c4
                else:
                    acc = g_c5
                if slot == 1 or slot == 4:
                    job.g_capsules[step, slot] = acc
                    continue
                for axis in range(3):
                    shift = wp.vec3(0.0, 0.0, 0.0)
                    if axis == 0:
                        shift = wp.vec3(eps_c, 0.0, 0.0)
                    elif axis == 1:
                        shift = wp.vec3(0.0, eps_c, 0.0)
                    else:
                        shift = wp.vec3(0.0, 0.0, eps_c)
                    a0 = job.capsules[step, 0]
                    a1 = job.capsules[step, 2]
                    b0 = job.capsules[step, 3]
                    b1 = job.capsules[step, 5]
                    if slot == 0:
                        a0 += shift
                    elif slot == 2:
                        a1 += shift
                    elif slot == 3:
                        b0 += shift
                    else:
                        b1 += shift
                    dval = (_segment_distance(a0, a1, b0, b1) - gap) / eps_c
                    if axis == 0:
                        acc += wp.vec3(d_gap * dval, 0.0, 0.0)
                    elif axis == 1:
                        acc += wp.vec3(0.0, d_gap * dval, 0.0)
                    else:
                        acc += wp.vec3(0.0, 0.0, d_gap * dval)
                job.g_capsules[step, slot] = acc
        else:
            for slot in range(6):
                job.g_capsules[step, slot] = wp.vec3(0.0, 0.0, 0.0)

    @wp.struct
    class ExtractJob:
        qpos: wp.array2d(dtype=float)
        body_pos: wp.array(dtype=wp.vec3, ndim=2)
        body_mat: wp.array(dtype=wp.mat33, ndim=2)
        obj_qpos: int
        left_tip: int
        right_tip: int
        left_forearm: int
        right_forearm: int
        left_wrist: int
        right_wrist: int
        left_off: wp.vec3
        right_off: wp.vec3
        contact_distance: wp.array2d(dtype=float)
        contact_frame: wp.array(dtype=wp.mat33, ndim=2)
        left_row: int
        right_row: int
        step: int
        obj_pos: wp.array(dtype=wp.vec3)
        obj_quat: wp.array(dtype=wp.vec4)
        left: wp.array(dtype=wp.vec3)
        right: wp.array(dtype=wp.vec3)
        capsules: wp.array(dtype=wp.vec3, ndim=2)
        phi: wp.array2d(dtype=float)
        inward: wp.array(dtype=wp.vec3, ndim=2)

    @wp.kernel(enable_backward=False)
    def extract_step(job: ExtractJob):
        obj_adr = job.obj_qpos
        obj_pos = wp.vec3(job.qpos[0, obj_adr], job.qpos[0, obj_adr + 1], job.qpos[0, obj_adr + 2])
        obj_quat = wp.vec4(
            job.qpos[0, obj_adr + 3],
            job.qpos[0, obj_adr + 4],
            job.qpos[0, obj_adr + 5],
            job.qpos[0, obj_adr + 6],
        )
        left_mat = job.body_mat[0, job.left_tip]
        right_mat = job.body_mat[0, job.right_tip]
        left = job.body_pos[0, job.left_tip] + left_mat * job.left_off
        right = job.body_pos[0, job.right_tip] + right_mat * job.right_off
        job.obj_pos[job.step] = obj_pos
        job.obj_quat[job.step] = obj_quat
        job.left[job.step] = left
        job.right[job.step] = right
        job.capsules[job.step, 0] = job.body_pos[0, job.left_forearm]
        job.capsules[job.step, 1] = job.body_pos[0, job.left_wrist]
        job.capsules[job.step, 2] = left
        job.capsules[job.step, 3] = job.body_pos[0, job.right_forearm]
        job.capsules[job.step, 4] = job.body_pos[0, job.right_wrist]
        job.capsules[job.step, 5] = right
        phi0 = float(1.0)
        phi1 = float(1.0)
        n0 = wp.vec3(0.0, 0.0, 0.0)
        n1 = wp.vec3(0.0, 0.0, 0.0)
        if job.left_row >= 0:
            phi0 = job.contact_distance[0, job.left_row]
            mat0 = job.contact_frame[0, job.left_row]
            n0 = wp.vec3(mat0[0, 0], mat0[0, 1], mat0[0, 2])
        if job.right_row >= 0:
            phi1 = job.contact_distance[0, job.right_row]
            mat1 = job.contact_frame[0, job.right_row]
            n1 = wp.vec3(mat1[0, 0], mat1[0, 1], mat1[0, 2])
        job.phi[job.step, 0] = phi0
        job.phi[job.step, 1] = phi1
        job.inward[job.step, 0] = n0
        job.inward[job.step, 1] = n1

    @wp.struct
    class SeedJob:
        step: int
        obj_qpos: int
        left_tip: int
        right_tip: int
        left_forearm: int
        right_forearm: int
        left_wrist: int
        right_wrist: int
        left_off: wp.vec3
        right_off: wp.vec3
        g_obj_pos: wp.array(dtype=wp.vec3)
        g_obj_quat: wp.array(dtype=wp.vec4)
        g_left: wp.array(dtype=wp.vec3)
        g_right: wp.array(dtype=wp.vec3)
        g_phi: wp.array2d(dtype=float)
        g_capsules: wp.array(dtype=wp.vec3, ndim=2)
        inward: wp.array(dtype=wp.vec3, ndim=2)
        contact_grad: wp.array2d(dtype=float)
        left_row: int
        right_row: int
        qpos_grad: wp.array2d(dtype=float)
        body_pos_grad: wp.array(dtype=wp.vec3, ndim=2)
        body_mat_grad: wp.array(dtype=wp.mat33, ndim=2)

    @wp.kernel(enable_backward=False)
    def seed_cotangent(job: SeedJob):
        adr = job.obj_qpos
        g_pos = job.g_obj_pos[job.step]
        g_quat = job.g_obj_quat[job.step]
        job.qpos_grad[0, adr + 0] = job.qpos_grad[0, adr + 0] + g_pos[0]
        job.qpos_grad[0, adr + 1] = job.qpos_grad[0, adr + 1] + g_pos[1]
        job.qpos_grad[0, adr + 2] = job.qpos_grad[0, adr + 2] + g_pos[2]
        job.qpos_grad[0, adr + 3] = job.qpos_grad[0, adr + 3] + g_quat[0]
        job.qpos_grad[0, adr + 4] = job.qpos_grad[0, adr + 4] + g_quat[1]
        job.qpos_grad[0, adr + 5] = job.qpos_grad[0, adr + 5] + g_quat[2]
        job.qpos_grad[0, adr + 6] = job.qpos_grad[0, adr + 6] + g_quat[3]
        g_left = job.g_left[job.step] + job.g_capsules[job.step, 2]
        g_right = job.g_right[job.step] + job.g_capsules[job.step, 5]
        # Positive g_phi wants a smaller gap. The fast-step adjoint carries
        # that into the fingertip query; without a contact row, move the
        # sphere center along the stored inward normal (d(phi)/d(query) = -n).
        if job.left_row >= 0:
            job.contact_grad[0, job.left_row] = job.contact_grad[0, job.left_row] + job.g_phi[job.step, 0]
        else:
            g_left = g_left + job.inward[job.step, 0] * (-job.g_phi[job.step, 0])
        if job.right_row >= 0:
            job.contact_grad[0, job.right_row] = job.contact_grad[0, job.right_row] + job.g_phi[job.step, 1]
        else:
            g_right = g_right + job.inward[job.step, 1] * (-job.g_phi[job.step, 1])
        job.body_pos_grad[0, job.left_tip] = job.body_pos_grad[0, job.left_tip] + g_left
        job.body_pos_grad[0, job.right_tip] = job.body_pos_grad[0, job.right_tip] + g_right
        job.body_pos_grad[0, job.left_forearm] = (
            job.body_pos_grad[0, job.left_forearm] + job.g_capsules[job.step, 0]
        )
        job.body_pos_grad[0, job.left_wrist] = (
            job.body_pos_grad[0, job.left_wrist] + job.g_capsules[job.step, 1]
        )
        job.body_pos_grad[0, job.right_forearm] = (
            job.body_pos_grad[0, job.right_forearm] + job.g_capsules[job.step, 3]
        )
        job.body_pos_grad[0, job.right_wrist] = (
            job.body_pos_grad[0, job.right_wrist] + job.g_capsules[job.step, 4]
        )
        job.body_mat_grad[0, job.left_tip] = job.body_mat_grad[0, job.left_tip] + wp.outer(
            g_left, job.left_off
        )
        job.body_mat_grad[0, job.right_tip] = job.body_mat_grad[0, job.right_tip] + wp.outer(
            g_right, job.right_off
        )

    @wp.kernel(enable_backward=False)
    def zero_vec3(target: wp.array(dtype=wp.vec3)):
        target[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)

    @wp.kernel(enable_backward=False)
    def zero_vec3_2d(target: wp.array(dtype=wp.vec3, ndim=2)):
        i, j = wp.tid()
        target[i, j] = wp.vec3(0.0, 0.0, 0.0)

    @wp.kernel(enable_backward=False)
    def zero_mat33_2d(target: wp.array(dtype=wp.mat33, ndim=2)):
        i, j = wp.tid()
        target[i, j] = wp.matrix_from_cols(
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
        )

    _KERNELS = SimpleNamespace(
        wp=wp,
        Weights=Weights,
        HorizonJob=HorizonJob,
        ExtractJob=ExtractJob,
        SeedJob=SeedJob,
        evaluate_horizon=evaluate_horizon,
        extract_step=extract_step,
        seed_cotangent=seed_cotangent,
        zero_vec3=zero_vec3,
        zero_vec3_2d=zero_vec3_2d,
        zero_mat33_2d=zero_mat33_2d,
    )
    return _KERNELS


def weights_struct(weights: BigraspGSCostWeights):
    k = ensure_kernels()
    packed = k.Weights()
    packed.contact_attract = float(weights.contact_attract)
    packed.contact_depth = float(weights.contact_depth)
    packed.penetration = float(weights.penetration)
    packed.object_position = float(weights.object_position)
    packed.object_lateral = float(weights.object_lateral)
    packed.object_orientation = float(weights.object_orientation)
    packed.action = float(weights.action)
    packed.smooth = float(weights.smooth)
    packed.sync = float(weights.sync)
    packed.force = float(weights.force)
    packed.swap = float(weights.swap)
    packed.inter_arm = float(weights.inter_arm)
    packed.tip_sep = float(weights.tip_sep)
    packed.tip_sep_min = float(weights.tip_sep_min)
    packed.capsule_radius = float(weights.capsule_radius)
    packed.wrist_radius = float(weights.wrist_radius)
    packed.tip_sphere_radius = float(weights.tip_sphere_radius)
    packed.capsule_margin = float(weights.capsule_margin)
    packed.gate_scale = float(weights.gate_scale)
    packed.target_phi = float(weights.target_phi)
    packed.penetration_limit = float(weights.penetration_limit)
    packed.approach_offset = float(weights.approach_offset)
    packed.query_mm = float(weights.query_mm)
    packed.terminal = float(weights.terminal)
    return packed


def _as_np(value, shape, default=0.0):
    if value is None:
        return np.full(shape, default, dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if shape[-1] == -1:
        return array.reshape(shape[0], -1)
    try:
        return array.reshape(shape)
    except ValueError:
        return array.reshape(-1)[: int(np.prod(shape))].reshape(shape)


def evaluate_bigrasp_gs_cost_wp(states, extras, cmd_traj, context, weights, device="cpu"):
    """Numeric Warp twin of ``evaluate_bigrasp_gs_cost`` for tests."""
    k = ensure_kernels()
    wp = k.wp
    if hasattr(states, "detach"):
        states = states.detach().cpu().numpy()
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states.reshape(1, -1)
    horizon = int(states.shape[0])
    if hasattr(cmd_traj, "detach"):
        cmd_traj = cmd_traj.detach().cpu().numpy()
    cmd = np.asarray(cmd_traj, dtype=np.float32)
    if cmd.ndim == 1:
        cmd = cmd.reshape(1, -1)
    if cmd.shape[0] != horizon:
        cmd = np.broadcast_to(cmd, (horizon, cmd.shape[-1])).copy()
    phi = _as_np(extras["phi"], (horizon, 2))
    contact = context.get("contact_points_local")
    normals = context.get("normals_local")
    if contact is None:
        contact = np.zeros((2, 3), dtype=np.float32)
        normals = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    contact = _as_np(contact, (2, 3))
    normals = _as_np(normals, (2, 3))
    capsules = extras.get("capsules")
    has_capsules = 0 if capsules is None else 1
    if capsules is None:
        capsules = np.zeros((horizon, 6, 3), dtype=np.float32)
    else:
        raw = np.asarray(capsules, dtype=np.float32)
        if raw.ndim >= 2 and raw.shape[-2] == 4:
            packed = np.zeros((horizon, 6, 3), dtype=np.float32)
            src = raw.reshape(horizon, 4, 3)
            packed[:, 0] = src[:, 0]
            packed[:, 1] = src[:, 1]
            packed[:, 2] = src[:, 1]
            packed[:, 3] = src[:, 2]
            packed[:, 4] = src[:, 3]
            packed[:, 5] = src[:, 3]
            capsules = packed
        else:
            capsules = _as_np(capsules, (horizon, 6, 3))
    normal = extras.get("normal")
    force = extras.get("contact_force")
    normal = _as_np(normal, (horizon, 2, 3))
    force = _as_np(force, (horizon, 2))
    desired = context.get("desired_force_local")
    has_desired = 0 if desired is None or float(weights.force) == 0.0 else 1
    if desired is None:
        desired = np.zeros((2, 3), dtype=np.float32)
    else:
        desired = _as_np(desired, (2, 3))
    target_p = context.get("target_object_pos")
    target_q = context.get("target_object_quat")
    has_target = 1 if target_p is not None else 0
    if target_p is None:
        target_p = np.zeros(3, dtype=np.float32)
        target_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        target_p = _as_np(target_p, (3,))
        target_q = _as_np(target_q, (4,))

    job = k.HorizonJob()
    job.obj_pos = wp.array(states[:, 0:3], dtype=wp.vec3, device=device)
    job.obj_quat = wp.array(states[:, 3:7], dtype=wp.vec4, device=device)
    job.left = wp.array(states[:, 7:10], dtype=wp.vec3, device=device)
    job.right = wp.array(states[:, 10:13], dtype=wp.vec3, device=device)
    job.phi = wp.array(phi, dtype=float, device=device)
    job.normal = wp.array(normal, dtype=wp.vec3, device=device)
    job.contact_force = wp.array(force, dtype=float, device=device)
    job.capsules = wp.array(capsules, dtype=wp.vec3, device=device)
    job.cmd = wp.array(cmd, dtype=float, device=device)
    job.contact_local = wp.array(contact, dtype=wp.vec3, device=device)
    job.normal_local = wp.array(normals, dtype=wp.vec3, device=device)
    job.desired_force = wp.array(desired, dtype=wp.vec3, device=device)
    job.target_pos = wp.array(target_p.reshape(1, 3), dtype=wp.vec3, device=device)
    job.target_quat = wp.array(target_q.reshape(1, 4), dtype=wp.vec4, device=device)
    job.flags = wp.array(np.array([has_target, has_desired], dtype=np.int32), dtype=int, device=device)
    job.has_capsules = int(has_capsules)
    job.cmd_dim = int(cmd.shape[1])
    job.horizon = horizon
    job.weights = weights_struct(weights)
    job.loss = wp.zeros(1, dtype=float, device=device)
    job.g_obj_pos = wp.zeros(horizon, dtype=wp.vec3, device=device)
    job.g_obj_quat = wp.zeros(horizon, dtype=wp.vec4, device=device)
    job.g_left = wp.zeros(horizon, dtype=wp.vec3, device=device)
    job.g_right = wp.zeros(horizon, dtype=wp.vec3, device=device)
    job.g_phi = wp.zeros((horizon, 2), dtype=float, device=device)
    job.g_cmd = wp.zeros(cmd.shape, dtype=float, device=device)
    job.g_capsules = wp.zeros((horizon, 6), dtype=wp.vec3, device=device)
    wp.launch(k.evaluate_horizon, dim=horizon, inputs=[job])
    wp.synchronize()
    return float(np.asarray(job.loss.numpy()).reshape(-1)[0])
