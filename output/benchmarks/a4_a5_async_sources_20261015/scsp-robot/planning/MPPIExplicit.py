import dataclasses
from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp

try:
    from dial_mpc.core.dial_core import MBDPI
except ModuleNotFoundError:
    from planning.dial_mpc.core.dial_core import MBDPI


@dataclasses.dataclass
class DialLikeConfig:
    # diffusion / MPPI
    Nsample: int = 128
    Hsample: int = 10
    Hnode: int = 4
    Ndiffuse: int = 1
    Ndiffuse_init: int = 2
    temp_sample: float = 0.06
    horizon_diffuse_factor: float = 0.9
    traj_diffuse_factor: float = 0.5
    update_method: str = "mppi"
    sigma_scale: float = 1.0


def _explicit_step_jax(qpos, cmd, phi_vec, jac_mat, model_params, params):
    h = params["h"]
    Q_inv = params["Q_inv"]
    obj_mass = params["obj_mass"]
    gravity = params["gravity"]
    robot_stiff = params["robot_stiff"]
    n_robot_qpos = params["n_robot_qpos"]

    # b vector
    b_o = obj_mass * gravity
    b_r = robot_stiff @ cmd
    b = jnp.concatenate([b_o, b_r])

    # non-contact term
    v_non_contact = Q_inv @ b / h
    JQb = jac_mat @ Q_inv @ b
    # Match the reference explicit MPC contact model.  The previous extra
    # JQb/h term doubled the command contribution at h=.01 and made MPPI use
    # different dynamics from the MPC it is intended to replace.
    contact_term = JQb + phi_vec
    raw = -model_params * contact_term

    # hard max
    contact_force = jnp.maximum(raw, 0.0)
    v_contact = Q_inv @ jac_mat.T @ contact_force / h

    v = v_non_contact + v_contact

    # integrate qpos
    qvel = v
    next_obj_pos = qpos[0:3] + h * qvel[0:3]
    if params["tabletop_lock"]:
        # The real object is supported by the PhysX table.  Keeping the
        # vertical coordinate fixed avoids duplicating each table manifold
        # point as an independent full-weight support constraint.
        next_obj_pos = next_obj_pos.at[2].set(qpos[2])

    quat = qpos[3:7]
    H = jnp.array([
        [-quat[1], quat[0], quat[3], -quat[2]],
        [-quat[2], -quat[3], quat[0], quat[1]],
        [-quat[3], quat[2], -quat[1], quat[0]],
    ]).T
    next_obj_quat = qpos[3:7] + 0.5 * h * (H @ qvel[3:6])
    # The explicit integrator is used for many consecutive MPPI rollout steps.
    # Without normalization the object quaternion slowly leaves S^3 and makes
    # both the terminal orientation cost and contact geometry inconsistent.
    next_obj_quat = next_obj_quat / (jnp.linalg.norm(next_obj_quat) + 1e-8)

    next_robot_qpos = qpos[-n_robot_qpos:] + h * qvel[-n_robot_qpos:]
    next_qpos = jnp.concatenate([next_obj_pos, next_obj_quat, next_robot_qpos])
    return next_qpos


def _quat_to_rot_jax(quat):
    w, x, y, z = quat
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z
    return jnp.array([
        [ww + xx - yy - zz, 2.0 * (xy - wz),     2.0 * (xz + wy)],
        [2.0 * (xy + wz),     ww - xx + yy - zz, 2.0 * (yz - wx)],
        [2.0 * (xz - wy),     2.0 * (yz + wx),   ww - xx - yy + zz],
    ])


def _franka_fk_T_jax(q):
    dh = jnp.array([
        [0.0,      0.0,       0.333],
        [0.0,     -jnp.pi / 2, 0.0],
        [0.0,      jnp.pi / 2, 0.316],
        [0.0825,   jnp.pi / 2, 0.0],
        [-0.0825, -jnp.pi / 2, 0.384],
        [0.0,      jnp.pi / 2, 0.0],
        [0.088,    jnp.pi / 2, 0.0],
    ])

    def _mdh(a, alpha, d, theta):
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        ca = jnp.cos(alpha)
        sa = jnp.sin(alpha)
        return jnp.array([
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ])

    T = jnp.eye(4)
    for i in range(7):
        a, alpha, d = dh[i]
        T = T @ _mdh(a, alpha, d, q[i])

    # attachment body in panda_nohand.xml
    attach_pos = jnp.array([0.0, 0.0, 0.107])
    attach_quat = jnp.array([0.3826834, 0.0, 0.0, 0.9238795])  # w, x, y, z
    R_attach = _quat_to_rot_jax(attach_quat)
    T_attach = jnp.eye(4)
    T_attach = T_attach.at[:3, :3].set(R_attach)
    T_attach = T_attach.at[:3, 3].set(attach_pos)

    # fingertip geom in attachment frame
    tip_pos = jnp.array([0.0, 0.0, 0.06])
    T_tip = jnp.eye(4)
    T_tip = T_tip.at[:3, 3].set(tip_pos)

    T_tip_world = T @ T_attach @ T_tip
    return T_tip_world


def _franka_fk_jax(q):
    return _franka_fk_T_jax(q)[:3, 3]


def _rot_to_quat_jax(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    def _case1():
        s = jnp.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
        return jnp.array([w, x, y, z])
    def _case2():
        cond1 = (R[0, 0] > R[1, 1]) & (R[0, 0] > R[2, 2])
        cond2 = R[1, 1] > R[2, 2]
        def _c1():
            s = jnp.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
            return jnp.array([w, x, y, z])
        def _c2():
            s = jnp.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
            return jnp.array([w, x, y, z])
        def _c3():
            s = jnp.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
            return jnp.array([w, x, y, z])
        return jax.lax.cond(cond1, _c1, lambda: jax.lax.cond(cond2, _c2, _c3))
    return jax.lax.cond(trace > 0.0, _case1, _case2)


def _franka_jacobian_pos_jax(q, eps=1e-4):
    eye = jnp.eye(q.shape[0], dtype=q.dtype)

    def fk_shift(e):
        return _franka_fk_jax(q + eps * e) - _franka_fk_jax(q - eps * e)

    J = jax.vmap(fk_shift)(eye) / (2.0 * eps)  # (7,3)
    return J.T  # (3,7)


def _tangent_basis_from_normal(n):
    n = n / (jnp.linalg.norm(n) + 1e-6)
    ref = jnp.where(jnp.abs(n[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]), jnp.array([0.0, 1.0, 0.0]))
    t1 = jnp.cross(n, ref)
    t1 = t1 / (jnp.linalg.norm(t1) + 1e-6)
    t2 = jnp.cross(n, t1)
    t2 = t2 / (jnp.linalg.norm(t2) + 1e-6)
    return n, t1, t2


def _contact_jacobian(n, t1, t2, J_rel, mu):
    con_frame = jnp.stack([n, t1, t2], axis=1)  # (3,3)
    con_frame_pmd = jnp.concatenate([con_frame, -con_frame[:, 1:]], axis=1)  # (3,5)
    con_jacp = con_frame_pmd.T @ J_rel  # (5, nv)
    con_jac = con_jacp[0] + mu * con_jacp[1:]  # (4, nv)
    return con_jac


class _PipeX(NamedTuple):
    pos: jnp.ndarray


class _PipelineState(NamedTuple):
    q: jnp.ndarray
    qd: jnp.ndarray
    x: _PipeX


class _State(NamedTuple):
    reward: jnp.ndarray
    pipeline_state: _PipelineState
    t: jnp.ndarray
    target_p: jnp.ndarray
    target_q: jnp.ndarray
    verify_cost: jnp.ndarray
    virtual_point: jnp.ndarray
    contact_point: jnp.ndarray
    curr_ori_coef: jnp.ndarray
    phi_vec: jnp.ndarray
    jac_mat: jnp.ndarray


class JaxExplicitEnv:
    def __init__(self, param, horizon):
        self.param_ = param
        self.action_size = param.n_cmd_
        self.horizon = int(horizon)
        self.lightweight_rollout = True

        tabletop_lock = bool(getattr(param, "mppi_tabletop_lock_", False))
        model_gravity = np.asarray(param.gravity_, dtype=np.float32).copy()
        if tabletop_lock:
            # Normal gravity is balanced by the table support in this reduced
            # planar rollout.  Real gravity/friction remain active in Isaac.
            model_gravity[2] = 0.0
        self._params = {
            "h": jnp.array(param.h_, dtype=jnp.float32),
            "Q_inv": jnp.array(jnp.linalg.inv(param.Q), dtype=jnp.float32),
            "obj_mass": jnp.array(param.obj_mass_, dtype=jnp.float32),
            "gravity": jnp.array(model_gravity, dtype=jnp.float32),
            "robot_stiff": jnp.array(param.robot_stiff_, dtype=jnp.float32),
            "n_robot_qpos": int(param.n_robot_qpos_),
            "tabletop_lock": tabletop_lock,
        }
        self._table_height = float(param.table_height)
        self._mu_object = float(param.mu_object_)
        self._contact_radius = float(getattr(param, "contact_radius_", 0.02))
        self._use_jax_contact = bool(getattr(param, "use_jax_contact_", True))

        self._target_p = None
        self._target_q = None
        self._verify_cost = None
        self._virtual_point = None
        self._contact_point = None
        self._curr_ori_coef = None

        self._phi_vec = None
        self._jac_mat = None
        self._model_params = jnp.array(param.model_params, dtype=jnp.float32)

        self._reject_dis = float(param.reject_dis)
        self._attract_coef = float(param.attract_coef)
        self._reject_coef = float(param.reject_coef)
        self._contact_coef = float(param.contact_coef)
        self._contact_cost_param = float(param.contact_cost_param)
        self._final_cost_mode = str(getattr(param, "final_cost_mode_", "object_pose"))
        self._drawer_open_axis = int(getattr(param, "drawer_open_axis_", 0))
        self._drawer_open_weight = float(getattr(param, "drawer_open_weight_", 500.0))
        self._drawer_lateral_weight = float(getattr(param, "drawer_lateral_weight_", 25.0))
        self._drawer_quat_weight = float(getattr(param, "drawer_quat_weight_", 0.0))

        u_lb = np.array(param.mpc_u_lb_, dtype=float)
        u_ub = np.array(param.mpc_u_ub_, dtype=float)
        if u_lb.ndim == 0:
            u_lb = np.full((self.action_size,), float(u_lb))
        if u_ub.ndim == 0:
            u_ub = np.full((self.action_size,), float(u_ub))
        self._u_lb = jnp.array(u_lb, dtype=jnp.float32)
        self._u_ub = jnp.array(u_ub, dtype=jnp.float32)

        # cost weights and references (optional)
        self._w_ee_ori = float(getattr(param, "mppi_w_ee_ori_", 1.0))
        self._w_joint_limit = float(getattr(param, "mppi_w_joint_limit_", 5.0))
        self._w_manip = float(getattr(param, "mppi_w_manip_", 0.1))
        self._w_cond = float(getattr(param, "mppi_w_cond_", 0.0))
        self._w_energy = float(getattr(param, "mppi_w_energy_", 0.00))
        self._w_vel = float(getattr(param, "mppi_w_vel_", 0.00))
        self._w_acc = float(getattr(param, "mppi_w_acc_", 0.000))
        self._w_base = float(getattr(param, "mppi_w_base_", 1.0))
        # p_arm_world keeps the fingertip on the nearest mesh point. A
        # separate contact-stage running term tells MPPI which tangential
        # direction should push the object; the terminal-only position cost
        # is too sparse for the short planning horizon. Existing
        # attract/contact costs remain unchanged.
        self._w_object_pos = float(getattr(param, "mppi_w_object_pos_", 50.0))

        q_ref = getattr(param, "init_robot_qpos_", None)
        if q_ref is None:
            q_ref = np.zeros((param.n_robot_qpos_,), dtype=np.float32)
        self._q_ref = jnp.array(q_ref, dtype=jnp.float32)
        T_ee_ref = _franka_fk_T_jax(self._q_ref)
        self._ee_ori_ref = _rot_to_quat_jax(T_ee_ref[:3, :3])

        q_lb = np.array(getattr(param, "mpc_q_lb_", -1e7 * np.ones((param.n_qpos_,))), dtype=np.float32)
        q_ub = np.array(getattr(param, "mpc_q_ub_", 1e7 * np.ones((param.n_qpos_,))), dtype=np.float32)
        self._q_lb = jnp.array(q_lb[-param.n_robot_qpos_:], dtype=jnp.float32)
        self._q_ub = jnp.array(q_ub[-param.n_robot_qpos_:], dtype=jnp.float32)

    def set_contacts(self, phi_vec, jac_mat):
        self._phi_vec = jnp.array(phi_vec, dtype=jnp.float32)
        self._jac_mat = jnp.array(jac_mat, dtype=jnp.float32)

    def set_contacts_from_state(self, q):
        phi_vec, jac_mat = self._compute_contacts(q)
        self._phi_vec = phi_vec
        self._jac_mat = jac_mat

    def set_cost_params(self, target_p, target_q, verify_cost_param, virtual_point, contact_point, curr_ori_coef):
        self._target_p = jnp.array(target_p, dtype=jnp.float32)
        self._target_q = jnp.array(target_q, dtype=jnp.float32)
        self._verify_cost = jnp.array(verify_cost_param, dtype=jnp.float32).reshape(())
        self._virtual_point = jnp.array(virtual_point, dtype=jnp.float32)
        self._contact_point = jnp.array(contact_point, dtype=jnp.float32)
        self._curr_ori_coef = jnp.array(curr_ori_coef, dtype=jnp.float32).reshape(())

    def _scale_action(self, u):
        # Map [-1, 1] to [lb, ub]
        # Quadratic interpolation between clipped nodes may overshoot, so the
        # interpolated controls must be clipped as well.
        u = jnp.clip(u, -1.0, 1.0)
        return 0.5 * (u + 1.0) * (self._u_ub - self._u_lb) + self._u_lb

    def _compute_contacts(self, q):
        nv = self.param_.n_qvel_
        max_ncon = self.param_.max_ncon_
        phi_vec = jnp.ones((max_ncon * 4,), dtype=jnp.float32)
        jac_mat = jnp.zeros((max_ncon * 4, nv), dtype=jnp.float32)

        obj_pos = q[0:3]
        ee_pos = _franka_fk_jax(q[-self.param_.n_robot_qpos_:])
        J_pos = _franka_jacobian_pos_jax(q[-self.param_.n_robot_qpos_:])
        J_rel = jnp.concatenate([jnp.eye(3), jnp.zeros((3, 3)), -J_pos], axis=1)  # (3, nv)

        # object-EE contact
        d = ee_pos - obj_pos
        dist = jnp.linalg.norm(d) - self._contact_radius
        n, t1, t2 = _tangent_basis_from_normal(d)
        con_jac = _contact_jacobian(n, t1, t2, J_rel, self._mu_object)
        contact_mask = dist < 0.0
        phi_vec = phi_vec.at[0:4].set(jnp.where(contact_mask, dist, 1.0))
        jac_mat = jac_mat.at[0:4].set(jnp.where(contact_mask, con_jac, 0.0))

        # object-table contact
        dist_table = obj_pos[2] - self._table_height
        n_t = jnp.array([0.0, 0.0, 1.0])
        t1_t = jnp.array([1.0, 0.0, 0.0])
        t2_t = jnp.array([0.0, 1.0, 0.0])
        J_rel_table = jnp.concatenate([jnp.eye(3), jnp.zeros((3, 3)), jnp.zeros((3, self.param_.n_robot_qpos_))], axis=1)
        con_jac_table = _contact_jacobian(n_t, t1_t, t2_t, J_rel_table, self._mu_object)
        table_mask = dist_table < 0.0
        phi_vec = phi_vec.at[4:8].set(jnp.where(table_mask, dist_table, 1.0))
        jac_mat = jac_mat.at[4:8].set(jnp.where(table_mask, con_jac_table, 0.0))

        return phi_vec, jac_mat

    def reset(self, q0):
        q0 = jnp.array(q0, dtype=jnp.float32)
        if q0.ndim != 1 or q0.shape[0] != self.param_.n_qpos_:
            raise ValueError(
                f"MPPI state has shape {q0.shape}, expected "
                f"({self.param_.n_qpos_},). Check n_robot_qpos_/n_qpos_."
            )
        # qd is represented in the same coordinates as the integrated state
        # (including the four quaternion coefficients), not generalized nv.
        qd0 = jnp.zeros_like(q0)
        ee_pos = _franka_fk_jax(q0[-self.param_.n_robot_qpos_:])[None, :]
        pipe = _PipelineState(q=q0, qd=qd0, x=_PipeX(pos=ee_pos))
        return _State(
            reward=jnp.array(0.0, dtype=jnp.float32),
            pipeline_state=pipe,
            t=jnp.array(0, dtype=jnp.int32),
            target_p=self._target_p,
            target_q=self._target_q,
            verify_cost=self._verify_cost,
            virtual_point=self._virtual_point,
            contact_point=self._contact_point,
            curr_ori_coef=self._curr_ori_coef,
            phi_vec=self._phi_vec,
            jac_mat=self._jac_mat,
        )

    def step(self, state, u):
        q = state.pipeline_state.q
        qd_prev = state.pipeline_state.qd
        if self._use_jax_contact:
            phi_vec, jac_mat = self._compute_contacts(q)
        else:
            phi_vec, jac_mat = state.phi_vec, state.jac_mat
        # u_scaled = self._scale_action(u)
        u_scaled = self._scale_action(u)
        next_q = _explicit_step_jax(
            q, u_scaled, phi_vec, jac_mat, self._model_params, self._params
        )
        qd = (next_q - q) / self._params["h"]
        qdd = (qd - qd_prev) / self._params["h"] if self._w_acc != 0.0 else jnp.zeros_like(qd)

        ee_pos = _franka_fk_jax(next_q[-self.param_.n_robot_qpos_:])[None, :]
        pipe = _PipelineState(q=next_q, qd=qd, x=_PipeX(pos=ee_pos))

        t = state.t
        cost = self._path_cost(next_q, u_scaled, qd, qdd, state)
        cost = cost + jnp.where(t == self.horizon - 1, self._final_cost(next_q, state), 0.0)
        reward = -cost
        return _State(
            reward=reward,
            pipeline_state=pipe,
            t=t + 1,
            target_p=state.target_p,
            target_q=state.target_q,
            verify_cost=state.verify_cost,
            virtual_point=state.virtual_point,
            contact_point=state.contact_point,
            curr_ori_coef=state.curr_ori_coef,
            phi_vec=phi_vec,
            jac_mat=jac_mat,
        )

    def _log_barrier(self, point, virtual_point):
        diff = point - virtual_point
        # A plain log(d^2 + eps) is negative near the goal.  Summing that over
        # the horizon rewards lingering there and can overwhelm the object
        # terminal cost.  log1p keeps the same minimum while remaining >= 0.
        squared_norm = jnp.dot(diff, diff)
        return jnp.log1p(squared_norm / 1e-3)

    def _path_cost(self, x, u, qd, qdd, context):
        q_robot = x[-self.param_.n_robot_qpos_:]
        T_ee = _franka_fk_T_jax(q_robot)
        ee_pos = T_ee[:3, 3]
        contact_cost = jnp.sum((x[0:3] - ee_pos) ** 2)
        control_cost = jnp.sum(u ** 2)
        # jax.debug.print("ee_pos: {}", ee_pos)

        # Use runtime target point from plan_once instead of a hard-coded constant.
        # For contact-only IK tests, pass the desired EE target via `contact_point`.
        # p = self._contact_point
        # contact_cost = 100.0 * jnp.sum((p - ee_pos) ** 2)
        virtual_point_cost = self._log_barrier(ee_pos, context.virtual_point)
        reject_distance_sq = jnp.sum((ee_pos - context.contact_point) ** 2)
        reject_radius_sq = self._reject_dis ** 2
        obstacle_cost = jnp.where(
            reject_distance_sq < reject_radius_sq,
            -jnp.log((reject_distance_sq + 1e-6) / (reject_radius_sq + 1e-6)),
            0.0,
        )
        attract_cost = self._attract_coef * virtual_point_cost + self._reject_coef * obstacle_cost

        contact_point_cost = self._log_barrier(ee_pos, context.contact_point)
        base_cost = (
            (1.0 - context.verify_cost) * attract_cost
            + self._contact_coef * context.verify_cost
            * (self._contact_cost_param * contact_cost + (1.0 - self._contact_cost_param) * contact_point_cost)
        )
        # The tabletop model only permits planar object motion, so track the
        # target in XY during contact and do not ask MPPI to fight gravity or
        # the physical table in Z.
        object_position_cost = jnp.sum((x[0:2] - context.target_p[0:2]) ** 2)
        object_position_tracking_cost = (
            self._w_object_pos * context.verify_cost * object_position_cost
        )
        # base_cost = contact_cost

        if self._w_ee_ori != 0.0:
            ee_quat = _rot_to_quat_jax(T_ee[:3, :3])
            ee_ori_cost = 1.0 - (jnp.dot(ee_quat, self._ee_ori_ref) ** 2)
        else:
            ee_ori_cost = jnp.array(0.0, dtype=x.dtype)

        # joint limits (soft)
        lower_violation = jnp.maximum(self._q_lb - q_robot, 0.0)
        upper_violation = jnp.maximum(q_robot - self._q_ub, 0.0)
        joint_limit_cost = jnp.sum((lower_violation + upper_violation) ** 2)

        # manipulability / condition number penalty
        if self._w_manip != 0.0 or self._w_cond != 0.0:
            J_pos = _franka_jacobian_pos_jax(q_robot)
            JJt = J_pos @ J_pos.T
            manip = jnp.sqrt(jnp.maximum(jnp.linalg.det(JJt), 0.0))
            manip_cost = 1.0 / (manip + 1e-6)
            s = jnp.linalg.svd(J_pos, compute_uv=False)
            cond_cost = s[0] / (s[-1] + 1e-6)
        else:
            manip_cost = jnp.array(0.0, dtype=x.dtype)
            cond_cost = jnp.array(0.0, dtype=x.dtype)

        # energy / velocity / acceleration penalties
        vel_cost = (
            jnp.sum(qd[-self.param_.n_robot_qpos_:] ** 2)
            if self._w_vel != 0.0 else jnp.array(0.0, dtype=x.dtype)
        )
        acc_cost = (
            jnp.sum(qdd[-self.param_.n_robot_qpos_:] ** 2)
            if self._w_acc != 0.0 else jnp.array(0.0, dtype=x.dtype)
        )

        return (
            self._w_base * base_cost
            + object_position_tracking_cost
            + self._w_energy * control_cost
            + self._w_ee_ori * context.curr_ori_coef * ee_ori_cost
            + self._w_joint_limit * joint_limit_cost
            + self._w_manip * manip_cost
            + self._w_cond * cond_cost
            + self._w_vel * vel_cost
            + self._w_acc * acc_cost
        )
        # return base_cost
    def _final_cost(self, x, context):
        if self._final_cost_mode == "drawer_open":
            axis = self._drawer_open_axis
            axis_cost = (x[axis] - context.target_p[axis]) ** 2
            pos_vec = x[0:3] - context.target_p
            lateral_cost = jnp.sum(pos_vec ** 2) - axis_cost
            quaternion_cost = 1.0 - (jnp.dot(x[3:7], context.target_q) ** 2)
            final_cost = (
                self._drawer_open_weight * axis_cost
                + self._drawer_lateral_weight * lateral_cost
                + self._drawer_quat_weight * quaternion_cost
            )
            return 10.0 * final_cost
        position_cost = jnp.sum((x[0:3] - context.target_p) ** 2)
        quaternion_cost = 1.0 - (jnp.dot(x[3:7], context.target_q) ** 2)
        final_cost = 500.0 * position_cost * context.curr_ori_coef + 5.0 * quaternion_cost
        return 10.0 * final_cost * context.verify_cost


class MPPIExplicit:
    def __init__(self, param):
        self.param_ = param

        cfg = DialLikeConfig(
            Nsample=getattr(param, "mppi_Nsample_", 128),
            Hsample=getattr(param, "mppi_Hsample_", 16),
            Hnode=getattr(param, "mppi_Hnode_", 4),
            Ndiffuse=getattr(param, "mppi_Ndiffuse_", 1),
            Ndiffuse_init=getattr(param, "mppi_Ndiffuse_init_", 2),
            temp_sample=getattr(param, "mppi_temp_sample_", 0.06),
            horizon_diffuse_factor=getattr(param, "mppi_horizon_diffuse_factor_", 0.9),
            traj_diffuse_factor=getattr(param, "mppi_traj_diffuse_factor_", 0.5),
            update_method="mppi",
            sigma_scale=getattr(param, "mppi_sigma_scale_", 1.0),
        )

        if param.n_cmd_ != param.n_robot_qpos_:
            raise ValueError(
                "MPPIExplicit joint-space mode requires n_cmd_ == "
                f"n_robot_qpos_, got {param.n_cmd_} and {param.n_robot_qpos_}."
            )
        expected_nq = 7 + int(param.n_robot_qpos_)
        expected_nv = 6 + int(param.n_robot_qpos_)
        if param.n_qpos_ != expected_nq or param.n_qvel_ != expected_nv:
            raise ValueError(
                "Inconsistent explicit-model dimensions: expected "
                f"n_qpos_={expected_nq}, n_qvel_={expected_nv} for "
                f"n_robot_qpos_={param.n_robot_qpos_}, got "
                f"{param.n_qpos_} and {param.n_qvel_}."
            )

        # MBDPI's node interpolation produces Hsample + 1 controls.
        self.env = JaxExplicitEnv(param, horizon=cfg.Hsample + 1)
        self.mbdpi = MBDPI(cfg, self.env)
        self.rng = jax.random.PRNGKey(getattr(param, "mppi_seed_", 0))
        self.Y0 = jnp.zeros([cfg.Hnode + 1, self.env.action_size])
        self._cfg = cfg
        self._finalize_jit = jax.jit(self._finalize)

    def _finalize(self, Y0, rews):
        """Fuse action extraction, horizon shift, and score reduction."""
        us = self.mbdpi.node2u_vmap(Y0)
        action = self.env._scale_action(us[0])
        shifted_Y0 = self.mbdpi.shift(Y0)
        best_cost = -jnp.max(rews)
        return action, shifted_Y0, best_cost

    def _maybe_apply_sol_guess(self, sol_guess):
        if isinstance(sol_guess, dict) and "Y0" in sol_guess:
            candidate = jnp.array(sol_guess["Y0"], dtype=jnp.float32)
            if candidate.shape == self.Y0.shape:
                self.Y0 = jnp.clip(candidate, -1.0, 1.0)

    def plan_once(
        self,
        target_p,
        target_q,
        curr_x,
        phi_vec,
        jac_mat,
        verify_cost_param,
        virtual_point,
        contact_point,
        curr_ori_coef,
        sol_guess=None,
    ):
        curr_x = np.asarray(curr_x, dtype=np.float32).reshape(-1)
        if curr_x.size != self.param_.n_qpos_:
            raise ValueError(
                f"curr_x has {curr_x.size} values, but MPPI was configured for "
                f"n_qpos_={self.param_.n_qpos_}."
            )
        self._maybe_apply_sol_guess(sol_guess)
        self.env.set_cost_params(
            target_p, target_q, verify_cost_param, virtual_point, contact_point, curr_ori_coef
        )
        if getattr(self.param_, "use_jax_contact_", True):
            
            self.env.set_contacts_from_state(jnp.array(curr_x, dtype=jnp.float32))
        else:
            self.env.set_contacts(phi_vec, jac_mat)

        state = self.env.reset(curr_x)
        n_diffuse = self._cfg.Ndiffuse
        if sol_guess is None:
            n_diffuse = self._cfg.Ndiffuse_init

        info = None
        for iteration in range(n_diffuse):
            factor = self.mbdpi.sigma_control * (self._cfg.traj_diffuse_factor ** iteration)
            self.rng, self.Y0, info = self.mbdpi.reverse_once(
                state, self.rng, self.Y0, factor
            )

        selected_Y0 = self.Y0
        action_device, self.Y0, best_cost_device = self._finalize_jit(selected_Y0, info["rews"])
        action, best_cost = jax.device_get((action_device, best_cost_device))
        action = np.asarray(action, dtype=np.float32)
        rollout_q = None
        if bool(getattr(self.param_, "mppi_return_rollout_", False)):
            us = self.mbdpi.node2u_vmap(selected_Y0)
            _, rollout_pipeline = self.mbdpi.rollout_us(state, us)
            rollout_q = np.array(rollout_pipeline.q, dtype=np.float32)

        sol_guess_out = dict(warm_start=True)
        if bool(getattr(self.param_, "mppi_export_warm_start_", False)):
            sol_guess_out["Y0"] = np.array(self.Y0)

        return dict(
            action=action,
            rollout_q=rollout_q,
            sol_guess=sol_guess_out,
            cost_opt=float(best_cost),
            solve_status="mppi_dial",
        )
