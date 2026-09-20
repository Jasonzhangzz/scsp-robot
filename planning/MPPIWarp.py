"""Warp MPPI is the 7-DoF version of ``test_0902.py`` ``mpc.plan_once``.

Each sample applies a joint delta, runs fingertip FK, evaluates ranked-patch
contact, then scores the same costs as the 3-DoF fingertip MPC:

    path:     (1-v) * attract ||tip-via||² + contact_coef * v * ||tip-press||²
              + 50 * ||u||²
    terminal: 10 * (pos_coef ||Δp||² + ori_coef (1 − (q·q*)²))

Nearest-face SDF hits do not move the object or earn pose-cost credit.
Leftover joint motion maximizes Yoshikawa manipulability in the null space.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time
import traceback

import numpy as np
import trimesh
import warp as wp

# 0902 ``init_cost_fns`` path term: ``control_weight * ||u||²``.
CONTROL_WEIGHT_0902 = 50.0
POLICY_INTERVAL = 0.02


PANDA_Q_LB = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
PANDA_Q_UB = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)

_PI = wp.constant(3.141592653589793)
_SQRT2_INV = wp.constant(0.7071067811865476)


@wp.func
def _quat_to_R(q: wp.vec4) -> wp.mat33:
    w, x, y, z = q[0], q[1], q[2], q[3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return wp.mat33(
        ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy),
        2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx),
        2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz,
    )


@wp.func
def _quat_normalize(q: wp.vec4) -> wp.vec4:
    return q / wp.max(wp.length(q), 1.0e-8)


@wp.func
def _quat_rotate(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    return _quat_to_R(q) * v


@wp.func
def _quat_rotate_inv(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    return wp.transpose(_quat_to_R(q)) * v


@wp.func
def _quat_integrate(q: wp.vec4, omega: wp.vec3, h: float) -> wp.vec4:
    w, x, y, z = q[0], q[1], q[2], q[3]
    hx = 0.5 * h
    dq = wp.vec4(
        hx * (-x * omega[0] - y * omega[1] - z * omega[2]),
        hx * (w * omega[0] + y * omega[2] - z * omega[1]),
        hx * (w * omega[1] - x * omega[2] + z * omega[0]),
        hx * (w * omega[2] + x * omega[1] - y * omega[0]),
    )
    return _quat_normalize(q + dq)


@wp.func
def _mdh(a: float, alpha: float, d: float, theta: float) -> wp.mat44:
    ct = wp.cos(theta)
    st = wp.sin(theta)
    ca = wp.cos(alpha)
    sa = wp.sin(alpha)
    return wp.mat44(
        ct, -st, 0.0, a,
        st * ca, ct * ca, -sa, -d * sa,
        st * sa, ct * sa, ca, d * ca,
        0.0, 0.0, 0.0, 1.0,
    )


@wp.func
def _mat44_mul(A: wp.mat44, B: wp.mat44) -> wp.mat44:
    return A * B


@wp.func
def _franka_fk_tip(q0: float, q1: float, q2: float, q3: float, q4: float, q5: float, q6: float) -> wp.vec3:
    T = _mdh(0.0, 0.0, 0.333, q0)
    T = _mat44_mul(T, _mdh(0.0, -_PI * 0.5, 0.0, q1))
    T = _mat44_mul(T, _mdh(0.0, _PI * 0.5, 0.316, q2))
    T = _mat44_mul(T, _mdh(0.0825, _PI * 0.5, 0.0, q3))
    T = _mat44_mul(T, _mdh(-0.0825, -_PI * 0.5, 0.384, q4))
    T = _mat44_mul(T, _mdh(0.0, _PI * 0.5, 0.0, q5))
    T = _mat44_mul(T, _mdh(0.088, _PI * 0.5, 0.0, q6))
    # attachment: pos [0,0,0.107], quat wxyz [0.3826834, 0, 0, 0.9238795]
    # R is a +135 deg rotation about Z (same as planning.MPPIExplicit).
    T_attach = wp.mat44(
        -_SQRT2_INV, -_SQRT2_INV, 0.0, 0.0,
        _SQRT2_INV, -_SQRT2_INV, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.107,
        0.0, 0.0, 0.0, 1.0,
    )
    T = _mat44_mul(T, T_attach)
    T_tip = wp.mat44(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.06,
        0.0, 0.0, 0.0, 1.0,
    )
    T = _mat44_mul(T, T_tip)
    return wp.vec3(T[0, 3], T[1, 3], T[2, 3])


@wp.func
def _tangent_basis(n: wp.vec3):
    n_u = n / wp.max(wp.length(n), 1.0e-6)
    ref = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(n_u[2]) > 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)
    t1 = wp.cross(n_u, ref)
    t1 = t1 / wp.max(wp.length(t1), 1.0e-6)
    t2 = wp.cross(n_u, t1)
    t2 = t2 / wp.max(wp.length(t2), 1.0e-6)
    return n_u, t1, t2


@wp.func
def _query_sdf_local(mesh_id: wp.uint64, p_local: wp.vec3, max_dist: float):
    query = wp.mesh_query_point_sign_normal(mesh_id, p_local, max_dist)
    closest = p_local
    sdf = max_dist
    n_out = wp.vec3(0.0, 0.0, 1.0)
    hit = int(0)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        delta = p_local - closest
        dist = wp.length(delta)
        sign = query.sign
        if sign < 0.0:
            sdf = -dist
            if dist > 1.0e-8:
                n_out = -delta / dist
            else:
                n_out = wp.mesh_eval_face_normal(mesh_id, query.face)
        else:
            sdf = dist
            if dist > 1.0e-8:
                n_out = delta / dist
            else:
                n_out = wp.mesh_eval_face_normal(mesh_id, query.face)
        hit = int(1)
    return sdf, closest, n_out, hit


@wp.func
def _clip(v: float, lo: float, hi: float) -> float:
    return wp.max(lo, wp.min(hi, v))


@wp.func
def _lambda_pose_cost(obj_p: wp.vec3, obj_q: wp.vec4, target_p: wp.vec3, target_q: wp.vec4, pos_coef: float, ori_coef: float) -> float:
    # Same units as examples.mpc.fingertips.test.test_0902._lambda_pose_cost.
    dpos = obj_p - target_p
    qn = _quat_normalize(target_q)
    oq = _quat_normalize(obj_q)
    align = wp.dot(oq, qn)
    ori = 1.0 - align * align
    return pos_coef * wp.dot(dpos, dpos) + ori_coef * ori


@wp.func
def _yoshikawa_w(jp0: wp.vec3, jp1: wp.vec3, jp2: wp.vec3, jp3: wp.vec3, jp4: wp.vec3, jp5: wp.vec3, jp6: wp.vec3) -> float:
    # w = sqrt(det(J J^T)) for the 3x7 fingertip position Jacobian.
    a00 = jp0[0] * jp0[0] + jp1[0] * jp1[0] + jp2[0] * jp2[0] + jp3[0] * jp3[0] + jp4[0] * jp4[0] + jp5[0] * jp5[0] + jp6[0] * jp6[0]
    a01 = jp0[0] * jp0[1] + jp1[0] * jp1[1] + jp2[0] * jp2[1] + jp3[0] * jp3[1] + jp4[0] * jp4[1] + jp5[0] * jp5[1] + jp6[0] * jp6[1]
    a02 = jp0[0] * jp0[2] + jp1[0] * jp1[2] + jp2[0] * jp2[2] + jp3[0] * jp3[2] + jp4[0] * jp4[2] + jp5[0] * jp5[2] + jp6[0] * jp6[2]
    a11 = jp0[1] * jp0[1] + jp1[1] * jp1[1] + jp2[1] * jp2[1] + jp3[1] * jp3[1] + jp4[1] * jp4[1] + jp5[1] * jp5[1] + jp6[1] * jp6[1]
    a12 = jp0[1] * jp0[2] + jp1[1] * jp1[2] + jp2[1] * jp2[2] + jp3[1] * jp3[2] + jp4[1] * jp4[2] + jp5[1] * jp5[2] + jp6[1] * jp6[2]
    a22 = jp0[2] * jp0[2] + jp1[2] * jp1[2] + jp2[2] * jp2[2] + jp3[2] * jp3[2] + jp4[2] * jp4[2] + jp5[2] * jp5[2] + jp6[2] * jp6[2]
    det = (
        a00 * (a11 * a22 - a12 * a12)
        - a01 * (a01 * a22 - a12 * a02)
        + a02 * (a01 * a12 - a11 * a02)
    )
    return wp.sqrt(wp.max(det, 0.0))


@wp.func
def _load_q(state: wp.array(dtype=float), i: int):
    obj_p = wp.vec3(state[i + 0], state[i + 1], state[i + 2])
    obj_q = wp.vec4(state[i + 3], state[i + 4], state[i + 5], state[i + 6])
    return obj_p, _quat_normalize(obj_q)


@wp.kernel
def _query_p_arm_kernel(
    state: wp.array(dtype=float),
    mesh_id: wp.uint64,
    r_tip: float,
    max_dist: float,
    tip_out: wp.array(dtype=wp.vec3),
    p_arm_out: wp.array(dtype=wp.vec3),
    normal_out: wp.array(dtype=wp.vec3),
    sdf_out: wp.array(dtype=float),
):
    obj_p, obj_q = _load_q(state, 0)
    tip = _franka_fk_tip(state[7], state[8], state[9], state[10], state[11], state[12], state[13])
    p_local = _quat_rotate_inv(obj_q, tip - obj_p)
    sdf, closest_local, n_local, hit = _query_sdf_local(mesh_id, p_local, max_dist)
    p_arm = tip
    n_world = wp.vec3(0.0, 0.0, 1.0)
    if hit == 1:
        p_arm = obj_p + _quat_rotate(obj_q, closest_local)
        n_world = _quat_rotate(obj_q, n_local)
    tip_out[0] = tip
    p_arm_out[0] = p_arm
    normal_out[0] = n_world
    sdf_out[0] = sdf - r_tip


@wp.kernel
def _sample_noise_kernel(
    seed: int,
    sigma: float,
    horizon: int,
    noise: wp.array3d(dtype=float),
):
    n, t, i = wp.tid()
    rng = wp.rand_init(seed, n * horizon * 7 + t * 7 + i)
    noise[n, t, i] = sigma * wp.randn(rng)


@wp.kernel
def _rollout_kernel(
    q0: wp.array(dtype=float),
    mean_u: wp.array2d(dtype=float),
    noise: wp.array3d(dtype=float),
    Qinv: wp.array2d(dtype=float),
    q_lb: wp.array(dtype=float),
    q_ub: wp.array(dtype=float),
    u_lim: float,
    h: float,
    obj_mass: float,
    k_contact: float,
    mu_object: float,
    robot_stiff: float,
    mesh_id: wp.uint64,
    r_tip: float,
    max_sdf: float,
    target_p: wp.vec3,
    target_q: wp.vec4,
    virtual_point: wp.vec3,
    contact_point: wp.vec3,
    verify: float,
    attract_coef: float,
    contact_coef: float,
    pos_coef: float,
    ori_coef: float,
    w_energy: float,
    w_nopen: float,
    w_joint_limit: float,
    w_manip: float,
    w_wrong_patch: float,
    nopen_margin: float,
    patch_radius: float,
    contact_gap: float,
    horizon: int,
    costs: wp.array(dtype=float),
    first_u: wp.array2d(dtype=float),
):
    n = wp.tid()
    obj_p = wp.vec3(q0[0], q0[1], q0[2])
    obj_q = _quat_normalize(wp.vec4(q0[3], q0[4], q0[5], q0[6]))
    rq0 = q0[7]
    rq1 = q0[8]
    rq2 = q0[9]
    rq3 = q0[10]
    rq4 = q0[11]
    rq5 = q0[12]
    rq6 = q0[13]
    z_lock = obj_p[2]
    cost = float(0.0)

    for t in range(horizon):
        u0 = _clip(mean_u[t, 0] + noise[n, t, 0], -u_lim, u_lim)
        u1 = _clip(mean_u[t, 1] + noise[n, t, 1], -u_lim, u_lim)
        u2 = _clip(mean_u[t, 2] + noise[n, t, 2], -u_lim, u_lim)
        u3 = _clip(mean_u[t, 3] + noise[n, t, 3], -u_lim, u_lim)
        u4 = _clip(mean_u[t, 4] + noise[n, t, 4], -u_lim, u_lim)
        u5 = _clip(mean_u[t, 5] + noise[n, t, 5], -u_lim, u_lim)
        u6 = _clip(mean_u[t, 6] + noise[n, t, 6], -u_lim, u_lim)
        if t == 0:
            first_u[n, 0] = u0
            first_u[n, 1] = u1
            first_u[n, 2] = u2
            first_u[n, 3] = u3
            first_u[n, 4] = u4
            first_u[n, 5] = u5
            first_u[n, 6] = u6

        # Predicted next joints under the position-controlled explicit model:
        # with diagonal joint stiffness K, v_r ≈ u / h so q+ = q + u.
        nq0 = _clip(rq0 + u0, q_lb[0], q_ub[0])
        nq1 = _clip(rq1 + u1, q_lb[1], q_ub[1])
        nq2 = _clip(rq2 + u2, q_lb[2], q_ub[2])
        nq3 = _clip(rq3 + u3, q_lb[3], q_ub[3])
        nq4 = _clip(rq4 + u4, q_lb[4], q_ub[4])
        nq5 = _clip(rq5 + u5, q_lb[5], q_ub[5])
        nq6 = _clip(rq6 + u6, q_lb[6], q_ub[6])

        tip = _franka_fk_tip(nq0, nq1, nq2, nq3, nq4, nq5, nq6)
        p_local = _quat_rotate_inv(obj_q, tip - obj_p)
        sdf, closest_local, n_local, hit = _query_sdf_local(mesh_id, p_local, max_sdf)
        p_arm = tip
        n_world = wp.vec3(0.0, 0.0, 1.0)
        if hit == 1:
            p_arm = obj_p + _quat_rotate(obj_q, closest_local)
            n_world = _quat_rotate(obj_q, n_local)
            n_world = n_world / wp.max(wp.length(n_world), 1.0e-6)

        gap = sdf - r_tip
        penetration = wp.max(0.0, -gap)
        nopen = wp.max(0.0, penetration - nopen_margin)

        # Finite-difference fingertip J is used both for contact and for the
        # OSC-style nullspace manipulability term.
        eps = 1.0e-4
        jp0 = (_franka_fk_tip(nq0 + eps, nq1, nq2, nq3, nq4, nq5, nq6) - _franka_fk_tip(nq0 - eps, nq1, nq2, nq3, nq4, nq5, nq6)) * (0.5 / eps)
        jp1 = (_franka_fk_tip(nq0, nq1 + eps, nq2, nq3, nq4, nq5, nq6) - _franka_fk_tip(nq0, nq1 - eps, nq2, nq3, nq4, nq5, nq6)) * (0.5 / eps)
        jp2 = (_franka_fk_tip(nq0, nq1, nq2 + eps, nq3, nq4, nq5, nq6) - _franka_fk_tip(nq0, nq1, nq2 - eps, nq3, nq4, nq5, nq6)) * (0.5 / eps)
        jp3 = (_franka_fk_tip(nq0, nq1, nq2, nq3 + eps, nq4, nq5, nq6) - _franka_fk_tip(nq0, nq1, nq2, nq3 - eps, nq4, nq5, nq6)) * (0.5 / eps)
        jp4 = (_franka_fk_tip(nq0, nq1, nq2, nq3, nq4 + eps, nq5, nq6) - _franka_fk_tip(nq0, nq1, nq2, nq3, nq4 - eps, nq5, nq6)) * (0.5 / eps)
        jp5 = (_franka_fk_tip(nq0, nq1, nq2, nq3, nq4, nq5 + eps, nq6) - _franka_fk_tip(nq0, nq1, nq2, nq3, nq4, nq5 - eps, nq6)) * (0.5 / eps)
        jp6 = (_franka_fk_tip(nq0, nq1, nq2, nq3, nq4, nq5, nq6 + eps) - _franka_fk_tip(nq0, nq1, nq2, nq3, nq4, nq5, nq6 - eps)) * (0.5 / eps)
        manip = _yoshikawa_w(jp0, jp1, jp2, jp3, jp4, jp5, jp6)

        # Object twist from a 4-row frictional complementarity at the SDF point.
        # Tabletop lock zeroes gravity / vertical translation so Isaac remains
        # the source of physical table support.
        # Only the ranked patch may move the object.  A 2 cm SDF band on the
        # nearest face was treating hover as contact and collapsing to the
        # closest surface.
        d_patch = p_arm - contact_point
        on_patch = wp.dot(d_patch, d_patch) <= patch_radius * patch_radius
        wrong_near = float(0.0)
        if hit == 1 and (not on_patch):
            keepout = wp.max(contact_gap, 0.02)
            wrong_near = wp.max(0.0, keepout - gap)
        if hit == 1 and gap < contact_gap and on_patch:
            r_obj = p_arm - obj_p
            n_u, t1, t2 = _tangent_basis(n_world)

            # b = [0_6, K u]; v_nc_obj = 0, v_r = u / h when Q_rr = K.
            # Contact residual uses J_rel [J_obj, -J_robot].
            JQb_n = -(wp.dot(n_u, jp0) * (robot_stiff * u0) + wp.dot(n_u, jp1) * (robot_stiff * u1) + wp.dot(n_u, jp2) * (robot_stiff * u2) + wp.dot(n_u, jp3) * (robot_stiff * u3) + wp.dot(n_u, jp4) * (robot_stiff * u4) + wp.dot(n_u, jp5) * (robot_stiff * u5) + wp.dot(n_u, jp6) * (robot_stiff * u6))
            JQb_t1 = -(wp.dot(t1, jp0) * (robot_stiff * u0) + wp.dot(t1, jp1) * (robot_stiff * u1) + wp.dot(t1, jp2) * (robot_stiff * u2) + wp.dot(t1, jp3) * (robot_stiff * u3) + wp.dot(t1, jp4) * (robot_stiff * u4) + wp.dot(t1, jp5) * (robot_stiff * u5) + wp.dot(t1, jp6) * (robot_stiff * u6))
            JQb_t2 = -(wp.dot(t2, jp0) * (robot_stiff * u0) + wp.dot(t2, jp1) * (robot_stiff * u1) + wp.dot(t2, jp2) * (robot_stiff * u2) + wp.dot(t2, jp3) * (robot_stiff * u3) + wp.dot(t2, jp4) * (robot_stiff * u4) + wp.dot(t2, jp5) * (robot_stiff * u5) + wp.dot(t2, jp6) * (robot_stiff * u6))
            phi = 0.5 * gap
            # 4-row pyramid: n + mu * {t1, t2, -t1, -t2} applied to JQb + phi
            c0 = JQb_n + mu_object * JQb_t1 + phi
            c1 = JQb_n + mu_object * JQb_t2 + phi
            c2 = JQb_n + mu_object * (-JQb_t1) + phi
            c3 = JQb_n + mu_object * (-JQb_t2) + phi
            f0 = wp.max(-k_contact * c0, 0.0)
            f1 = wp.max(-k_contact * c1, 0.0)
            f2 = wp.max(-k_contact * c2, 0.0)
            f3 = wp.max(-k_contact * c3, 0.0)
            # wrench on object from J_obj^T (n + mu t) f
            fn = f0 + f1 + f2 + f3
            ft1 = mu_object * (f0 - f2)
            ft2 = mu_object * (f1 - f3)
            force = fn * n_u + ft1 * t1 + ft2 * t2
            torque = wp.cross(r_obj, force)
            # v_obj = Qinv_oo @ wrench / h  (Qinv_oo = I / obj_inertia diag)
            inv_m = Qinv[0, 0]
            inv_i = Qinv[3, 3]
            dpos = (inv_m * force) * h
            domega = (inv_i * torque)
            obj_p = wp.vec3(obj_p[0] + dpos[0], obj_p[1] + dpos[1], z_lock)
            obj_q = _quat_integrate(obj_q, domega, h)

        rq0, rq1, rq2, rq3, rq4, rq5, rq6 = nq0, nq1, nq2, nq3, nq4, nq5, nq6

        # 0902 path: attract the tip to the ranked via, then press the patch.
        # This is a cost hint, not OSC servo.  Pose still comes from the
        # joint → FK → on-patch contact update.
        d_virt = tip - virtual_point
        d_con = tip - contact_point
        attract = attract_coef * wp.dot(d_virt, d_virt)
        press = wp.dot(d_con, d_con)
        base = (1.0 - verify) * attract + contact_coef * verify * press
        energy = u0 * u0 + u1 * u1 + u2 * u2 + u3 * u3 + u4 * u4 + u5 * u5 + u6 * u6
        jl = float(0.0)
        jl += wp.max(q_lb[0] - nq0, 0.0) * wp.max(q_lb[0] - nq0, 0.0) + wp.max(nq0 - q_ub[0], 0.0) * wp.max(nq0 - q_ub[0], 0.0)
        jl += wp.max(q_lb[1] - nq1, 0.0) * wp.max(q_lb[1] - nq1, 0.0) + wp.max(nq1 - q_ub[1], 0.0) * wp.max(nq1 - q_ub[1], 0.0)
        jl += wp.max(q_lb[2] - nq2, 0.0) * wp.max(q_lb[2] - nq2, 0.0) + wp.max(nq2 - q_ub[2], 0.0) * wp.max(nq2 - q_ub[2], 0.0)
        jl += wp.max(q_lb[3] - nq3, 0.0) * wp.max(q_lb[3] - nq3, 0.0) + wp.max(nq3 - q_ub[3], 0.0) * wp.max(nq3 - q_ub[3], 0.0)
        jl += wp.max(q_lb[4] - nq4, 0.0) * wp.max(q_lb[4] - nq4, 0.0) + wp.max(nq4 - q_ub[4], 0.0) * wp.max(nq4 - q_ub[4], 0.0)
        jl += wp.max(q_lb[5] - nq5, 0.0) * wp.max(q_lb[5] - nq5, 0.0) + wp.max(nq5 - q_ub[5], 0.0) * wp.max(nq5 - q_ub[5], 0.0)
        jl += wp.max(q_lb[6] - nq6, 0.0) * wp.max(q_lb[6] - nq6, 0.0) + wp.max(nq6 - q_ub[6], 0.0) * wp.max(nq6 - q_ub[6], 0.0)
        cost += (
            base
            + w_energy * energy
            + w_nopen * nopen * nopen
            + w_joint_limit * jl
            + w_wrong_patch * wrong_near * wrong_near
            - w_manip * manip
        )

    # 0902 final_cost_fn = 10 * (pos_coef||Δp||² + ori_coef·ori).
    cost += 10.0 * _lambda_pose_cost(obj_p, obj_q, target_p, target_q, pos_coef, ori_coef)
    costs[n] = cost


def policy_control_substeps(control_substeps, sim_dt, policy_interval=POLICY_INTERVAL):
    """Frames per planner action.  0 means one --rollout interval (20 ms)."""
    n = int(control_substeps)
    if n <= 0:
        return max(1, int(round(float(policy_interval) / max(float(sim_dt), 1e-6))))
    return n


def clip_toward(origin, target, max_step):
    origin = np.asarray(origin, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if origin.size != target.size:
        raise ValueError(f"clip_toward size mismatch: {origin.size} vs {target.size}")
    delta = target - origin
    dist = float(np.linalg.norm(delta))
    if dist <= float(max_step) or dist < 1e-9:
        return target.copy()
    return origin + delta * (float(max_step) / dist)


def blend_action(prev, raw, alpha):
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if prev is None:
        return raw.copy()
    prev = np.asarray(prev, dtype=np.float32).reshape(raw.shape)
    return ((1.0 - alpha) * prev + alpha * raw).astype(np.float32)


def clamp_via_to_tip(tip, via, max_lead):
    return clip_toward(np.asarray(tip, dtype=np.float64).reshape(3), via, max_lead)


def adapt_param_for_cartesian_ranking(param, args):
    """Same 3-DoF ranking dims as OSC / ``test_0902`` ``plan_once``."""
    from examples.mpc.franka.ik2.params import build_lambda_optimizer

    table_height = float(param.table_height)
    param.n_cmd_ = 3
    param.n_robot_qpos_ = 3
    param.n_qpos_ = 10
    param.n_qvel_ = 9
    param.robot_stiff_ = np.diag(3 * [float(args.cartesian_joint_stiffness)])
    q = np.zeros((param.n_qvel_, param.n_qvel_))
    q[:6, :6] = param.obj_inertia_
    q[6:, 6:] = param.robot_stiff_
    param.Q = q
    param.mpc_u_lb_ = -float(getattr(args, "mpc_step_limit", 0.005))
    param.mpc_u_ub_ = -param.mpc_u_lb_
    param.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), np.array([-10.0, -10.0, table_height - 0.01])))
    param.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), np.array([10.0, 10.0, table_height + 1.0])))
    args.solver = "acados"
    param.torch_solver = "acados"
    param.planner_solver_ = "acados"
    if getattr(param, "lambda_optimizer", None) is None:
        param.lambda_optimizer = build_lambda_optimizer(param, args)
    param.lambda_optimizer.solver = "acados"
    param.sol_guess_ = None
    return param


def configure_mppi_rollout_param(param, args):
    param.rollout_press_patch = True
    param.quadratic_contact_track = True
    param.attract_coef = max(float(param.attract_coef), 20.0)
    param.field_cost_weight = 0.0
    param.contact_coef = max(float(param.contact_coef), float(param.attract_coef))
    param.lambda_optimizer.lock_contact_patch = False
    param.lambda_optimizer.contact_switch_confirm_steps = max(
        1, int(args.contact_switch_confirm_steps)
    )
    return param


def adapt_param_for_joint_mppi(param, args):
    """7-DoF MPPI model.  Does not overwrite cartesian ranking dims."""
    from examples.mpc.franka.ik2.params import build_lambda_optimizer

    param.mppi_tabletop_lock_ = True
    param.fingertip_radius_ = 0.01
    joint_stiffness = float(args.joint_model_stiffness)
    param.mppi_robot_stiff_ = joint_stiffness
    q_metric = np.zeros((13, 13), dtype=np.float32)
    q_metric[:6, :6] = np.asarray(param.obj_inertia_, dtype=np.float32)
    q_metric[6:, 6:] = np.diag(np.full(7, joint_stiffness, dtype=np.float32))
    param.mppi_Q_ = q_metric
    param.mppi_u_lim_ = float(args.joint_step)

    for name in (
        "Nsample", "Hsample", "Ndiffuse", "Ndiffuse_init",
        "temp_sample", "sigma_scale", "traj_diffuse_factor",
    ):
        setattr(param, f"mppi_{name}_", getattr(args, f"mppi_{name}"))
    param.mppi_seed_ = int(args.mppi_seed)
    param.mppi_w_energy_ = float(getattr(args, "mppi_w_energy", CONTROL_WEIGHT_0902))
    param.mppi_w_nopen_ = float(args.mppi_w_nopen)
    param.mppi_w_joint_limit_ = float(args.mppi_w_joint_limit)
    param.mppi_nopen_margin_ = float(args.mppi_nopen_margin)
    param.mppi_w_manip_ = float(getattr(args, "nullspace_stiffness", 10.0))
    param.nullspace_stiffness_ = param.mppi_w_manip_

    if getattr(param, "lambda_optimizer", None) is None:
        param.lambda_optimizer = build_lambda_optimizer(param, args)
    param.lambda_optimizer.solver = "acados"
    param.sol_guess_ = None
    return param


def contact_is_on_ranked_patch(p_arm, contact_point, patch_radius=0.03):
    """True when the SDF hit is the ranked lambda patch, not the nearest face."""
    p_arm = np.asarray(p_arm, dtype=np.float64).reshape(3)
    contact_point = np.asarray(contact_point, dtype=np.float64).reshape(3)
    return float(np.dot(p_arm - contact_point, p_arm - contact_point)) <= float(patch_radius) ** 2


def object_pose_cost(pos, quat, target_p, target_q, pos_coef, ori_coef):
    """Same lambda pose term as ``test_0902._lambda_pose_cost``."""
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    target_p = np.asarray(target_p, dtype=np.float64).reshape(3)
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    target_q = np.asarray(target_q, dtype=np.float64).reshape(4)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-9)
    target_q = target_q / max(float(np.linalg.norm(target_q)), 1e-9)
    dpos = pos - target_p
    ori = 1.0 - float(np.clip(np.dot(quat, target_q), -1.0, 1.0)) ** 2
    return float(pos_coef) * float(np.dot(dpos, dpos)) + float(ori_coef) * ori


def terminal_object_pose_cost(pos, quat, target_p, target_q, pos_coef, ori_coef):
    """0902 ``final_cost_fn``: ``10 * (pos_coef||Δp||² + ori_coef·ori)``."""
    return 10.0 * object_pose_cost(pos, quat, target_p, target_q, pos_coef, ori_coef)


def _joint_space_q_metric(param, joint_stiffness):
    q_metric = np.zeros((13, 13), dtype=np.float32)
    inertia = np.asarray(getattr(param, "obj_inertia_", np.eye(6, dtype=np.float32)), dtype=np.float32)
    if inertia.shape != (6, 6):
        inertia = np.eye(6, dtype=np.float32)
    q_metric[:6, :6] = inertia
    q_metric[6:, 6:] = np.diag(np.full(7, float(joint_stiffness), dtype=np.float32))
    return q_metric


class MPPIWarp:
    """Sample joint-delta sequences on the GPU and return the first action."""

    def __init__(self, param, mesh_path=None, device=None):
        wp.init()
        if device is None:
            device = "cuda:0" if wp.is_cuda_available() else "cpu"
        self.device = device
        self.param_ = param

        self.nsample = int(getattr(param, "mppi_Nsample_", 256))
        self.horizon = int(getattr(param, "mppi_Hsample_", 16))
        self.ndiffuse = int(getattr(param, "mppi_Ndiffuse_", 2))
        self.ndiffuse_init = int(getattr(param, "mppi_Ndiffuse_init_", 3))
        self.temp = float(getattr(param, "mppi_temp_sample_", 0.5))
        self.sigma0 = float(getattr(param, "mppi_sigma_scale_", 0.6))
        self.traj_diffuse = float(getattr(param, "mppi_traj_diffuse_factor_", 0.5))
        self.seed = int(getattr(param, "mppi_seed_", 0))
        self._seed_counter = self.seed

        self.u_lim = float(getattr(param, "mppi_u_lim_", np.max(np.abs(np.asarray(param.mpc_u_ub_, dtype=np.float32)))))
        self.h = float(param.h_)
        self.obj_mass = float(getattr(param, "lambda_obj_mass_", getattr(param, "obj_mass_", 0.01)))
        self.k_contact = float(param.model_params)
        self.mu_object = float(param.mu_object_)
        stiff = np.asarray(
            getattr(param, "mppi_robot_stiff_", getattr(param, "robot_stiff_", 300.0)),
            dtype=np.float32,
        )
        self.robot_stiff = float(np.asarray(stiff, dtype=np.float32).reshape(-1)[0])
        self.r_tip = float(getattr(param, "fingertip_radius_", 0.01))
        self.max_sdf = 1.5
        self.nopen_margin = float(getattr(param, "mppi_nopen_margin_", 0.002))

        opt = getattr(param, "lambda_optimizer", None)
        self.pos_coef = float(getattr(opt, "pos_coef", getattr(param, "pos_coef", 500.0)))
        self.ori_coef = float(getattr(opt, "ori_coef", getattr(param, "ori_coef", 20.0)))
        self.attract_coef = float(getattr(param, "attract_coef", 20.0))
        self.contact_coef = float(getattr(param, "contact_coef", 0.7))
        self.w_energy = float(getattr(param, "mppi_w_energy_", CONTROL_WEIGHT_0902))
        self.w_nopen = float(getattr(param, "mppi_w_nopen_", 80.0))
        self.w_joint_limit = float(getattr(param, "mppi_w_joint_limit_", 100.0))
        self.w_manip = float(getattr(param, "mppi_w_manip_", getattr(param, "nullspace_stiffness_", 10.0)))
        self.w_wrong_patch = float(getattr(param, "mppi_w_wrong_patch_", 80.0))
        self.patch_radius = float(getattr(param, "mppi_patch_radius_", 0.03))
        self.contact_gap = float(getattr(param, "mppi_contact_gap_", 0.003))

        q_metric = np.asarray(getattr(param, "mppi_Q_", param.Q), dtype=np.float32)
        if q_metric.shape != (13, 13):
            q_metric = _joint_space_q_metric(param, self.robot_stiff)
        self.Qinv_np = np.linalg.inv(q_metric).astype(np.float32)

        mesh_path = mesh_path or param.mesh_path_
        self._mesh = self._build_warp_mesh(mesh_path, device)
        self.mesh_id = self._mesh.id

        with wp.ScopedDevice(device):
            self.q0_wp = wp.zeros(14, dtype=float)
            self.mean_u_wp = wp.zeros((self.horizon, 7), dtype=float)
            self.noise_wp = wp.zeros((self.nsample, self.horizon, 7), dtype=float)
            self.costs_wp = wp.zeros(self.nsample, dtype=float)
            self.first_u_wp = wp.zeros((self.nsample, 7), dtype=float)
            self.Qinv_wp = wp.array(self.Qinv_np, dtype=float)
            self.q_lb_wp = wp.array(PANDA_Q_LB, dtype=float)
            self.q_ub_wp = wp.array(PANDA_Q_UB, dtype=float)
            self.tip_wp = wp.zeros(1, dtype=wp.vec3)
            self.p_arm_wp = wp.zeros(1, dtype=wp.vec3)
            self.normal_wp = wp.zeros(1, dtype=wp.vec3)
            self.sdf_wp = wp.zeros(1, dtype=float)

        self.mean_u = np.zeros((self.horizon, 7), dtype=np.float32)
        self._warmed = False

    @staticmethod
    def _build_warp_mesh(mesh_path, device):
        if mesh_path is None or not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"MPPIWarp mesh not found: {mesh_path}")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        if not mesh.is_watertight:
            try:
                mesh = mesh.convex_hull
            except Exception:
                pass
        vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(mesh.faces, dtype=np.int32).reshape(-1)
        with wp.ScopedDevice(device):
            return wp.Mesh(
                points=wp.array(vertices, dtype=wp.vec3),
                indices=wp.array(faces, dtype=wp.int32),
            )

    def query_p_arm_world(self, curr_x):
        """FK fingertip → object-local SDF closest point, returned in world."""
        q = np.asarray(curr_x, dtype=np.float32).reshape(-1)
        if q.size != 14:
            raise ValueError(f"query_p_arm_world expects 14-D state, got {q.size}")
        self.q0_wp.assign(q)
        wp.launch(
            _query_p_arm_kernel,
            dim=1,
            inputs=[self.q0_wp, self.mesh_id, self.r_tip, self.max_sdf],
            outputs=[self.tip_wp, self.p_arm_wp, self.normal_wp, self.sdf_wp],
            device=self.device,
        )
        wp.synchronize()
        tip = np.array(self.tip_wp.numpy()[0], dtype=np.float32)
        p_arm = np.array(self.p_arm_wp.numpy()[0], dtype=np.float32)
        normal = np.array(self.normal_wp.numpy()[0], dtype=np.float32)
        sdf = float(self.sdf_wp.numpy()[0])
        return {
            "tip_world": tip,
            "p_arm_world": p_arm,
            "normal_world": normal,
            "sdf": sdf,
        }

    def _maybe_apply_sol_guess(self, sol_guess):
        if isinstance(sol_guess, dict) and "mean_u" in sol_guess:
            candidate = np.asarray(sol_guess["mean_u"], dtype=np.float32)
            if candidate.shape == self.mean_u.shape:
                self.mean_u = np.clip(candidate, -self.u_lim, self.u_lim)
                self._warmed = True

    def plan_once(
        self,
        target_p,
        target_q,
        curr_x,
        phi_vec=None,
        jac_mat=None,
        verify_cost_param=0.0,
        virtual_point=None,
        contact_point=None,
        curr_ori_coef=1.0,
        sol_guess=None,
    ):
        del phi_vec, jac_mat, curr_ori_coef
        curr_x = np.asarray(curr_x, dtype=np.float32).reshape(-1)
        if curr_x.size != 14:
            raise ValueError(f"MPPIWarp curr_x must have 14 values, got {curr_x.size}")
        self._maybe_apply_sol_guess(sol_guess)

        target_p = np.asarray(target_p, dtype=np.float32).reshape(3)
        target_q = np.asarray(target_q, dtype=np.float32).reshape(4)
        if virtual_point is None:
            virtual_point = curr_x[0:3].copy()
        if contact_point is None:
            contact_point = self.query_p_arm_world(curr_x)["p_arm_world"]
        virtual_point = np.asarray(virtual_point, dtype=np.float32).reshape(3)
        contact_point = np.asarray(contact_point, dtype=np.float32).reshape(3)
        verify = float(np.clip(verify_cost_param, 0.0, 1.0))

        n_diffuse = self.ndiffuse_init if not self._warmed else self.ndiffuse
        self.q0_wp.assign(curr_x)

        best_cost = float("inf")
        for iteration in range(n_diffuse):
            sigma = self.sigma0 * self.u_lim * (self.traj_diffuse ** iteration)
            self._seed_counter += 1
            wp.launch(
                _sample_noise_kernel,
                dim=(self.nsample, self.horizon, 7),
                inputs=[int(self._seed_counter), float(sigma), int(self.horizon)],
                outputs=[self.noise_wp],
                device=self.device,
            )
            self.mean_u_wp.assign(self.mean_u)
            wp.launch(
                _rollout_kernel,
                dim=self.nsample,
                inputs=[
                    self.q0_wp,
                    self.mean_u_wp,
                    self.noise_wp,
                    self.Qinv_wp,
                    self.q_lb_wp,
                    self.q_ub_wp,
                    float(self.u_lim),
                    float(self.h),
                    float(self.obj_mass),
                    float(self.k_contact),
                    float(self.mu_object),
                    float(self.robot_stiff),
                    self.mesh_id,
                    float(self.r_tip),
                    float(self.max_sdf),
                    wp.vec3(float(target_p[0]), float(target_p[1]), float(target_p[2])),
                    wp.vec4(float(target_q[0]), float(target_q[1]), float(target_q[2]), float(target_q[3])),
                    wp.vec3(float(virtual_point[0]), float(virtual_point[1]), float(virtual_point[2])),
                    wp.vec3(float(contact_point[0]), float(contact_point[1]), float(contact_point[2])),
                    float(verify),
                    float(self.attract_coef),
                    float(self.contact_coef),
                    float(self.pos_coef),
                    float(self.ori_coef),
                    float(self.w_energy),
                    float(self.w_nopen),
                    float(self.w_joint_limit),
                    float(self.w_manip),
                    float(self.w_wrong_patch),
                    float(self.nopen_margin),
                    float(self.patch_radius),
                    float(self.contact_gap),
                    int(self.horizon),
                ],
                outputs=[self.costs_wp, self.first_u_wp],
                device=self.device,
            )
            wp.synchronize()
            costs = np.array(self.costs_wp.numpy(), dtype=np.float64)
            noise = np.array(self.noise_wp.numpy(), dtype=np.float32)
            first_u = np.array(self.first_u_wp.numpy(), dtype=np.float32)
            finite = np.isfinite(costs)
            if not np.any(finite):
                continue
            costs = np.where(finite, costs, np.max(costs[finite]) + 1.0e6)
            c0 = float(np.min(costs))
            weights = np.exp(-(costs - c0) / max(self.temp, 1.0e-6))
            weights = weights / max(float(np.sum(weights)), 1.0e-12)
            # Update the mean trajectory: u ← u + Σ w ε
            self.mean_u = np.clip(
                self.mean_u + np.einsum("n,nth->th", weights.astype(np.float32), noise),
                -self.u_lim,
                self.u_lim,
            )
            best_cost = c0
            self._last_weights = weights
            self._last_first_u = first_u

        action = np.clip(self.mean_u[0].copy(), -self.u_lim, self.u_lim).astype(np.float32)
        shifted = np.zeros_like(self.mean_u)
        if self.horizon > 1:
            shifted[:-1] = self.mean_u[1:]
            shifted[-1] = self.mean_u[-1]
        self.mean_u = shifted
        self._warmed = True
        return {
            "action": action,
            "sol_guess": {"mean_u": self.mean_u.copy(), "warm_start": True},
            "cost_opt": float(best_cost),
            "solve_status": "mppi_warp",
            "rollout_q": None,
        }


_OPT_SNAPSHOT_KEYS = (
    "last_selected_idx",
    "last_best_idx",
    "last_executed_idx",
    "last_global_idx",
    "last_global_total_cost",
    "last_topk_ids",
    "last_candidate_ids",
    "last_candidate_raw_costs",
    "last_candidate_deltas",
    "last_candidate_pose_costs",
    "last_best_delta",
    "last_pose_cost_now",
    "lock_contact_patch",
    "contact_switch_confidence",
    "_dwell_steps",
    "_dwell_idx",
    "point_curvature",
    "region_max_point_curvature",
)


def _pickle_safe(obj, depth=0):
    if depth > 8:
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return np.asarray(obj)
    if isinstance(obj, (list, tuple)):
        return [_pickle_safe(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _pickle_safe(v, depth + 1) for k, v in obj.items()}
    try:
        return float(obj)
    except Exception:
        return str(obj)


def _opt_snapshot(opt):
    snap = {}
    for key in _OPT_SNAPSHOT_KEYS:
        if hasattr(opt, key):
            snap[key] = getattr(opt, key)
    return snap


def _apply_opt_snapshot(opt, snap):
    if not snap:
        return
    for key, value in snap.items():
        setattr(opt, key, value)


def _apply_dwell_payload(param, args, dwell):
    from examples.mpc.fingertips.test.test_0902 import (
        _protect_destination_dwell,
        _rollout_dwell_assignment,
        _same_contact_patch,
        _should_observe_model_cost,
    )

    if not dwell:
        return
    opt = param.lambda_optimizer
    model_cost_conf = dwell.get("model_cost_conf")
    c_now_cost = dwell.get("c_now_cost")
    pred_reduction = dwell.get("pred_reduction")
    c_after = dwell.get("c_after")
    if model_cost_conf is not None and c_now_cost is not None and dwell.get("post_physical"):
        if _should_observe_model_cost(opt.has_delta_span(), getattr(opt, "last_pose_cost_now", None)):
            pred_delta = pred_reduction
            if pred_delta is None or not np.isfinite(float(pred_delta)):
                pred_delta = getattr(opt, "last_best_delta", None)
            if pred_delta is None or not np.isfinite(float(pred_delta)):
                finite = np.asarray(getattr(opt, "last_candidate_deltas", []), dtype=np.float64).reshape(-1)
                finite = finite[np.isfinite(finite)]
                pred_delta = float(np.max(finite)) if finite.size else None
            if pred_delta is None:
                model_cost_conf.observe_unusable_prediction()
            else:
                act_reduction = float(c_now_cost) - float(c_after)
                pred_n = opt.normalize_cost_delta(pred_delta)
                act_n = opt.normalize_cost_delta(act_reduction)
                if pred_n is None or act_n is None:
                    model_cost_conf.observe_unusable_prediction()
                else:
                    model_cost_conf.observe(pred_n, act_n)

    tip_now = np.asarray(dwell["tip_now"], dtype=float)
    obj_pos = np.asarray(dwell["obj_pos"], dtype=float)
    r_now = np.asarray(dwell["r_now"], dtype=float).reshape(3, 3)
    tip_local_now = r_now.T @ (tip_now - obj_pos)
    p_arm_world = np.asarray(dwell["p_arm_world"], dtype=float)
    post_physical = bool(dwell.get("post_physical", False))
    last_accept_p_arm = bool(dwell.get("last_accept_p_arm", False))
    escape_on = bool(dwell.get("escape_on", False))
    verify_chatter = bool(dwell.get("verify_chatter", False))
    pos_err_now = float(dwell["pos_err_now"])

    (progress_idx, dwell_active, dwell_dead, occupied_for_log,
     on_exec_patch, _dist_exec_now) = _rollout_dwell_assignment(
        opt, tip_local_now, tip_now, obj_pos, r_now,
        getattr(opt, "last_executed_idx", None),
        p_arm_world, post_physical,
        prev_dwell=getattr(opt, "_dwell_idx", None),
    )
    if dwell_active and not on_exec_patch:
        dwell_dead = True
    elif dwell_active and on_exec_patch:
        dwell_dead = False
        dwell_active = False
    if (not last_accept_p_arm) and not on_exec_patch and escape_on and not post_physical:
        dwell_dead = False
        dwell_active = False
    if progress_idx is None:
        progress_idx = getattr(opt, "last_global_idx", None) or getattr(opt, "last_executed_idx", None)
    if verify_chatter:
        progress_idx = occupied_for_log or progress_idx
        dwell_active = True
        dwell_dead = True
    dest_idx = getattr(opt, "last_global_idx", None)
    dest_protected = bool(
        dest_idx is not None and progress_idx is not None
        and _same_contact_patch(opt, progress_idx, dest_idx, radius=0.03)
    )
    dwell_active, dwell_dead = _protect_destination_dwell(
        opt, progress_idx, dest_idx, dwell_active, dwell_dead
    )
    opt.note_contact_progress(
        progress_idx,
        pos_err_now,
        active=dwell_active,
        gamma=float(args.contact_dwell_gamma),
        min_dwell_steps=int(args.contact_dwell_steps),
        improve_eps=0.002,
        dead_increment=dwell_dead,
        merge_radius=0.03,
        block_radius=0.03,
        block_cycles=80,
        time_decay=bool(last_accept_p_arm and not dest_protected),
    )


def _build_planner_runtime(init):
    from planning.acados_env import ensure_acados_env
    ensure_acados_env()
    from examples.mpc.franka.ik2.params import ExplicitMPCParams
    from examples.mpc.fingertips.test.test_0902 import (
        ContactValueTracker,
        ModelCostConfidence,
        SmoothedApproachVia,
    )

    args = argparse.Namespace(**init["args"])
    args.solver = "acados"
    args.rollout = True
    trial_count = int(init["trial_count"])
    param = ExplicitMPCParams(
        args,
        rand_seed=trial_count,
        target_type=getattr(args, "target_type", "ground-rotation"),
        mpc_model="explicit",
    )
    param = adapt_param_for_cartesian_ranking(param, args)
    param = configure_mppi_rollout_param(param, args)
    param = adapt_param_for_joint_mppi(param, args)
    for key in (
        "target_p_", "target_q_", "mesh_path_", "table_height",
        "lambda_obj_mass_", "gravity_", "attract_coef", "contact_coef",
    ):
        if key in init:
            setattr(param, key, init[key])

    mpc = MPPIWarp(param, mesh_path=param.mesh_path_, device=init.get("device", "cuda:0"))
    trackers = {
        "value_tracker": ContactValueTracker(
            tau=float(args.value_tau),
            rel_scale=float(args.value_rel_scale),
            rho=float(args.value_rho),
            alpha=float(args.value_alpha),
            beta=float(args.verify_beta),
            window_size=int(getattr(args, "verify_window_size", 5)),
            confirm_steps=int(getattr(args, "verify_enter_steps", 5)),
            min_hold_steps=int(getattr(args, "verify_hold_steps", 30)),
            release_steps=int(getattr(args, "verify_release_steps", 8)),
            accept_margin_ratio=0.05,
            accept_margin_abs=0.02,
        ),
        "model_cost_conf": ModelCostConfidence(
            threshold=float(getattr(args, "model_cost_error_threshold", 6.0)),
            eps=float(getattr(args, "model_cost_error_eps", 1e-6)),
            min_steps=int(getattr(args, "model_cost_error_min_steps", 3)),
        ),
        "approach_via": SmoothedApproachVia(
            rate=float(getattr(args, "via_smooth_rate", 0.05)),
            max_step=max(1e-4, float(getattr(args, "mpc_step_limit", 0.005))),
            max_lead=max(1e-4, float(getattr(args, "via_max_lead", getattr(args, "mpc_step_limit", 0.005)))),
        ),
        "arrived_hold": False,
        "arrived_dest_idx": None,
        "sol_guess": None,
        "last_verify_cost": None,
    }
    return args, param, mpc, trackers


def handle_planner_request(args, param, mpc, trackers, msg):
    from examples.mpc.fingertips.test.test_0902 import (
        compute_rollout_contact_via,
        _verify_is_chatter,
    )
    from scipy.spatial.transform import Rotation

    dwell = msg.get("dwell")
    if dwell is not None:
        dwell = dict(dwell)
        dwell["model_cost_conf"] = trackers["model_cost_conf"]
        _apply_dwell_payload(param, args, dwell)

    curr_q = np.asarray(msg["policy_q"], dtype=np.float32)
    full_q = np.asarray(msg.get("full_q", curr_q), dtype=np.float32)
    if msg.get("jac_mat_env") is not None:
        jac_mat_env = np.asarray(msg["jac_mat_env"])
    else:
        from examples.mpc.franka.ik2.contact_frames import table_jac_mat_env
        jac_mat_env = table_jac_mat_env(
            curr_q[:3], curr_q[3:7], float(param.table_height),
            nv=int(getattr(param, "n_qvel_", 9)),
            mu=float(getattr(param, "mu_object_", 0.5)),
            max_ncon=int(getattr(param, "max_ncon_", 10)),
        )
    fingertip_radius = 0.01
    table_ground = float(msg["table_ground"])
    floor_z = float(msg["floor_z"]) if msg.get("floor_z") is not None else float(param.table_height)
    support_point = msg.get("support_point")
    support_normal = msg.get("support_normal")
    if support_point is not None:
        support_point = np.asarray(support_point, dtype=np.float64).reshape(3)
    if support_normal is not None:
        support_normal = np.asarray(support_normal, dtype=np.float64).reshape(3)
    r_obj_to_world = Rotation.from_quat([curr_q[4], curr_q[5], curr_q[6], curr_q[3]]).as_matrix()
    ranking_mass = float(getattr(param, "lambda_obj_mass_", param.lambda_optimizer.m))
    gravity = np.hstack([
        r_obj_to_world.T @ param.gravity_[:3] * ranking_mass,
        np.zeros(3),
    ])
    t0 = time.perf_counter()
    policy = compute_rollout_contact_via(
        param, args, curr_q, r_obj_to_world, gravity, jac_mat_env,
        fingertip_radius, trackers["value_tracker"], trackers["model_cost_conf"],
        trackers["approach_via"], trackers["arrived_hold"], trackers["arrived_dest_idx"],
        floor_ground=table_ground,
        floor_z=floor_z,
        support_point=support_point,
        support_normal=support_normal,
    )
    rank_dt = time.perf_counter() - t0
    trackers["arrived_hold"] = policy["arrived_hold"]
    trackers["arrived_dest_idx"] = policy["arrived_dest_idx"]
    verify_cost = policy["verify_cost"]
    verify_chatter = _verify_is_chatter(trackers["last_verify_cost"], verify_cost)
    trackers["last_verify_cost"] = float(verify_cost)

    tip = np.asarray(curr_q[7:10], dtype=np.float64)
    max_lead = max(1e-4, float(getattr(args, "via_max_lead", getattr(args, "mpc_step_limit", 0.005))))
    policy["mpc_virtual_point"] = clamp_via_to_tip(tip, policy["mpc_virtual_point"], max_lead)
    policy["mpc_contact_point"] = clamp_via_to_tip(tip, policy["mpc_contact_point"], max_lead)

    t1 = time.perf_counter()
    sol = mpc.plan_once(
        param.target_p_,
        param.target_q_,
        full_q,
        verify_cost_param=verify_cost,
        virtual_point=policy["mpc_virtual_point"],
        contact_point=policy["mpc_contact_point"],
        sol_guess=trackers["sol_guess"],
    )
    mppi_dt = time.perf_counter() - t1
    trackers["sol_guess"] = sol["sol_guess"]
    policy["choose_dt"] = rank_dt
    return {
        "action": np.asarray(sol["action"], dtype=np.float32).reshape(7),
        "sol_guess": sol["sol_guess"],
        "cost_opt": sol["cost_opt"],
        "policy": _pickle_safe(policy),
        "verify_cost": float(verify_cost),
        "verify_chatter": bool(verify_chatter),
        "if_contact": bool(msg.get("if_contact", False)),
        "opt_snapshot": _opt_snapshot(param.lambda_optimizer),
        "model_tightness": float(trackers["model_cost_conf"].tightness()),
        "model_accum": float(trackers["model_cost_conf"].accum),
        "rank_dt": float(rank_dt),
        "mppi_dt": float(mppi_dt),
    }


def put_latest(q, item):
    """Keep only the newest message, like a ROS topic queue of size 1."""
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        return False


def take_latest(q):
    """Non-blocking: return the newest queued item, or None."""
    item = None
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return item


def drain_latest(q, first):
    """Drop stale messages so the planner always uses the newest state."""
    msg = first
    while True:
        try:
            newer = q.get_nowait()
        except queue.Empty:
            return msg
        if newer is None:
            return None
        msg = newer


def mppi_planner_worker(state_q, action_q, ready_q, init):
    """Independent planner loop.  Must not import isaacgym."""
    try:
        args, param, mpc, trackers = _build_planner_runtime(init)
        ready_q.put({"ok": True})
    except Exception:
        ready_q.put({"ok": False, "error": traceback.format_exc()})
        return
    while True:
        try:
            msg = state_q.get()
        except (EOFError, KeyboardInterrupt):
            break
        if msg is None or (isinstance(msg, dict) and msg.get("cmd") == "stop"):
            break
        msg = drain_latest(state_q, msg)
        if msg is None:
            break
        try:
            put_latest(action_q, handle_planner_request(args, param, mpc, trackers, msg))
        except Exception:
            put_latest(action_q, {"ok": False, "error": traceback.format_exc()})


def planner_init_payload(args, param, trial_count, device):
    return {
        "args": {k: v for k, v in vars(args).items() if not callable(v)},
        "trial_count": int(trial_count),
        "device": device,
        "target_p_": np.asarray(param.target_p_, dtype=np.float64),
        "target_q_": np.asarray(param.target_q_, dtype=np.float64),
        "mesh_path_": param.mesh_path_,
        "table_height": float(param.table_height),
        "lambda_obj_mass_": float(getattr(param, "lambda_obj_mass_", param.obj_mass_)),
        "gravity_": np.asarray(param.gravity_, dtype=np.float64),
        "attract_coef": float(param.attract_coef),
        "contact_coef": float(param.contact_coef),
    }


class AsyncMppiPlanner:
    """ROS-style latest-only state/action bus.  Isaac never waits for a plan."""

    def __init__(self, state_q, action_q, proc):
        self.state_q = state_q
        self.action_q = action_q
        self.proc = proc

    @classmethod
    def start(cls, init, timeout=180.0):
        ctx = mp.get_context("spawn")
        state_q = ctx.Queue(maxsize=1)
        action_q = ctx.Queue(maxsize=1)
        ready_q = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=mppi_planner_worker,
            args=(state_q, action_q, ready_q, init),
            daemon=True,
        )
        proc.start()
        ready = ready_q.get(timeout=timeout)
        if not ready.get("ok", False):
            proc.join(timeout=2.0)
            raise RuntimeError(ready.get("error", "MPPI planner process failed to start"))
        return cls(state_q, action_q, proc)

    def publish_state(self, payload):
        return put_latest(self.state_q, payload)

    def take_action(self):
        out = take_latest(self.action_q)
        if isinstance(out, dict) and out.get("ok") is False:
            raise RuntimeError(out.get("error", "MPPI planner process failed"))
        return out

    def close(self):
        put_latest(self.state_q, None)
        if self.proc is not None:
            self.proc.join(timeout=5.0)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2.0)
