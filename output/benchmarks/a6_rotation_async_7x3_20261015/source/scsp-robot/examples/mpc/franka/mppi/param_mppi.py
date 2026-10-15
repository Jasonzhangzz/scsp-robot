from typing import Any, Dict, Optional, Sequence

import casadi as cs
import numpy as np


def _quad_form(weight, vec):
    return cs.mtimes([vec.T, cs.DM(weight), vec])


def _coerce_trajectory(traj, steps, dim, name):
    if traj is None:
        raise ValueError(f"{name} cannot be None.")

    if isinstance(traj, (list, tuple)):
        if len(traj) != steps:
            raise ValueError(f"{name} must have {steps} steps.")
        return np.column_stack(
            [np.asarray(item, dtype=np.float64).reshape(dim) for item in traj]
        )

    arr = np.asarray(traj, dtype=np.float64)
    if arr.shape == (dim, steps):
        return arr.copy()
    if arr.shape == (steps, dim):
        return arr.T.copy()
    if arr.size == dim:
        return np.repeat(arr.reshape(dim, 1), steps, axis=1)

    raise ValueError(
        f"{name} must have shape ({dim}, {steps}), ({steps}, {dim}), "
        f"or be a list of {steps} vectors of length {dim}."
    )


def _flatten_cols(mat):
    return np.asarray(mat, dtype=np.float64).reshape((-1,), order="F")


def _extract_vec3(value):
    if (
        isinstance(value, np.void)
        and getattr(value, "dtype", None) is not None
        and value.dtype.names is not None
    ):
        return np.array(
            [float(value["x"]), float(value["y"]), float(value["z"])],
            dtype=np.float64,
        )

    arr = np.asarray(value)
    if arr.dtype.names is not None:
        return np.array(
            [float(arr["x"]), float(arr["y"]), float(arr["z"])], dtype=np.float64
        )

    arr = arr.reshape(-1)
    return np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=np.float64)


def _skew(vec):
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -vec[2], vec[1]],
            [vec[2], 0.0, -vec[0]],
            [-vec[1], vec[0], 0.0],
        ],
        dtype=np.float64,
    )


def quat_conjugate_casadi(q):
    return cs.vertcat(q[0], -q[1], -q[2], -q[3])


def quat_multiply_casadi(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return cs.vertcat(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_normalize_casadi(q, eps=1e-8):
    return q / cs.sqrt(cs.sumsqr(q) + eps)


def quat_normalize_numpy(q, eps=1e-8):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q / np.sqrt(np.dot(q, q) + eps)


def quat_conjugate_numpy(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_multiply_numpy(q1, q2):
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_rot_numpy(q):
    w, x, y, z = quat_normalize_numpy(q)
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
    return np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def rpy_to_quaternion_numpy(angles):
    yaw, pitch, roll = np.asarray(angles, dtype=np.float64).reshape(3)
    qx = (
        np.sin(roll / 2.0) * np.cos(pitch / 2.0) * np.cos(yaw / 2.0)
        - np.cos(roll / 2.0) * np.sin(pitch / 2.0) * np.sin(yaw / 2.0)
    )
    qy = (
        np.cos(roll / 2.0) * np.sin(pitch / 2.0) * np.cos(yaw / 2.0)
        + np.sin(roll / 2.0) * np.cos(pitch / 2.0) * np.sin(yaw / 2.0)
    )
    qz = (
        np.cos(roll / 2.0) * np.cos(pitch / 2.0) * np.sin(yaw / 2.0)
        - np.sin(roll / 2.0) * np.sin(pitch / 2.0) * np.cos(yaw / 2.0)
    )
    qw = (
        np.cos(roll / 2.0) * np.cos(pitch / 2.0) * np.cos(yaw / 2.0)
        + np.sin(roll / 2.0) * np.sin(pitch / 2.0) * np.sin(yaw / 2.0)
    )
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _normalize_vec(vec, default=None, eps=1e-8):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm > eps:
        return vec / norm
    if default is None:
        default = np.zeros_like(vec)
    return np.asarray(default, dtype=np.float64).reshape(vec.shape)


def _tangent_basis_from_normal_numpy(normal):
    normal = _normalize_vec(normal, default=np.array([0.0, 0.0, 1.0]))
    ref = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(normal[2]) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    t1 = np.cross(normal, ref)
    t1 = _normalize_vec(t1, default=np.array([1.0, 0.0, 0.0]))
    t2 = np.cross(normal, t1)
    t2 = _normalize_vec(t2, default=np.array([0.0, 1.0, 0.0]))
    return normal, t1, t2


def _contact_jacobian_numpy(normal, t1, t2, J_rel, mu):
    con_frame = np.stack([normal, t1, t2], axis=1)
    con_frame_pmd = np.concatenate([con_frame, -con_frame[:, 1:]], axis=1)
    con_jacp = con_frame_pmd.T @ np.asarray(J_rel, dtype=np.float64)
    return con_jacp[0] + float(mu) * con_jacp[1:]


class IDTOMPPIParams:
    """
    CasADi helper for a q-only trajectory optimization model in the style of
    optimizer/trajectory_optimizer.cc.

    Decision variable:
        q_traj = [q_0, ..., q_T], flattened column-major.

    Per-step generalized positions:
        q_t = [obj_pos(3), obj_quat(4), robot_q(7)]  -> 14 dims

    Derived generalized velocities:
        v_t = [obj_linear_vel(3), obj_angular_vel(3), robot_v(7)] -> 13 dims

    Contact inputs per contact:
        [signed_distance(1), normal_W(3), contact_point_W(3), rel_velocity_W(3)]

    Notes:
    - This file no longer optimizes u directly; it computes the generalized
      force τ(q) needed to realize the trajectory and uses that in the cost.
    - `signed_distance` is an explicit CasADi input through the contact schedule.
    - Because CasADi cannot query Drake geometry, contact normals, contact
      points, and relative contact velocities are also passed in as symbolic
      inputs.
    """

    def __init__(
        self,
        horizon=20,
        dt=0.05,
        max_contacts=1,
        args=None,
        rand_seed=1,
        target_type="rotation",
        mpc_model="q_trajectory",
    ):
        self.h_ = float(dt)
        self.sim_dt_ = float(getattr(args, "sim_dt_", min(self.h_, 0.01)))
        self.frame_skip_ = max(int(round(self.h_ / max(self.sim_dt_, 1e-8))), 1)
        self.mpc_model = str(mpc_model)
        self.mpc_horizon_ = int(horizon)
        self.ipopt_max_iter_ = 100

        self.n_robot_qpos_ = 7
        self.n_obj_qpos_ = 7
        self.n_qpos_ = self.n_obj_qpos_ + self.n_robot_qpos_

        self.n_obj_qvel_ = 6
        self.n_robot_qvel_ = 7
        self.n_qvel_ = self.n_obj_qvel_ + self.n_robot_qvel_
        self.n_state_ = self.n_qpos_ + self.n_qvel_
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_

        # In q-trajectory mode, the direct decision variable is the flattened
        # trajectory, not a per-step actuation vector.
        self.n_decision_ = self.n_qpos_ * (self.mpc_horizon_ + 1)
        self.n_cmd_ = self.n_decision_

        self.max_contacts_ = int(max_contacts)
        self.max_ncon_ = max(
            int(getattr(args, "max_ncon_", self.max_contacts_)), self.max_contacts_
        )
        self.max_env_contacts_ = int(getattr(args, "max_env_contacts_", 4))
        self.contact_param_dim_per_contact_ = 10
        self.contact_param_dim_per_step_ = (
            self.max_contacts_ * self.contact_param_dim_per_contact_
        )
        self.contact_schedule_dim_ = (
            self.mpc_horizon_ * self.contact_param_dim_per_step_
        )

        self.cost_param_dim_ = (
            (self.mpc_horizon_ + 1) * (self.n_qpos_ + self.n_qvel_)
        )

        self.unactuated_dofs_ = np.arange(self.n_obj_qvel_, dtype=int)
        self.num_equality_constraints_ = (
            len(self.unactuated_dofs_) * self.mpc_horizon_
        )

        self.obj_q_slice = slice(0, 7)
        self.obj_pos_slice = slice(0, 3)
        self.obj_quat_slice = slice(3, 7)
        self.robot_q_slice = slice(7, 14)

        self.obj_linear_v_slice = slice(0, 3)
        self.obj_angular_v_slice = slice(3, 6)
        self.robot_v_slice = slice(6, 13)

        self.contact_cost_param = float(getattr(args, "contact_cost_param", 0.0))
        self.attract_coef = float(getattr(args, "attract_coef", 0.5))
        self.reject_coef = float(getattr(args, "reject_coef", 0.001))
        self.contact_coef = float(getattr(args, "contact_coef", 0.5))
        self.reject_dis = float(getattr(args, "reject_dis", 0.01))
        self.contact_point_fallback_offset_ = float(
            getattr(args, "contact_point_fallback_offset_", 0.01)
        )
        self.table_contact_distance_threshold_ = float(
            getattr(args, "table_contact_distance_threshold_", 0.02)
        )
        self.if_contact_separation_threshold_ = float(
            getattr(args, "if_contact_separation_threshold_", 0.0)
        )
        self.table_height = float(getattr(args, "table_height", 0.35))
        self.ground_height_threshold_ = float(
            getattr(args, "ground_height_threshold", 0.012)
        )

        self.model_path_ = ""
        self.mesh_path_ = None
        self.object_names_ = ["obj"]
        self.robot_name_ = "franka"
        self.ee_body_name_ = "fingertip"
        self.fingertip_body_names_ = ["fingertip", "attachment", "panda_link7"]
        self.fingertip_geoms = list(self.fingertip_body_names_)
        self.jc_kp_ = float(getattr(args, "jc_kp_", 200.0))
        self.jc_damping_ = float(getattr(args, "jc_damping_", 10.0))
        self.proximity_threshold_ = float(getattr(args, "proximity_threshold_", 0.1))

        self.obj_mass_ = 0.10
        self.obj_inertia_ = np.diag([0.0025, 0.0025, 0.0025])
        self.obj_linear_damping_ = np.diag([1.5, 1.5, 1.5])
        self.obj_angular_damping_ = np.diag([0.05, 0.05, 0.05])
        self.robot_inertia_ = np.diag(np.ones(self.n_robot_qvel_))
        self.robot_damping_ = np.diag(0.10 * np.ones(self.n_robot_qvel_))
        self.gravity_ = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.obj_spatial_inertia_ = np.zeros((6, 6), dtype=np.float64)
        self.obj_spatial_inertia_[:3, :3] = self.obj_mass_ * np.eye(3)
        self.obj_spatial_inertia_[3:, 3:] = self.obj_inertia_
        self.robot_stiff_ = np.diag(300.0 * np.ones(self.n_robot_qvel_))
        self.Q = np.zeros((self.n_qvel_, self.n_qvel_), dtype=np.float64)
        self.Q[:6, :6] = self.obj_spatial_inertia_
        self.Q[6:, 6:] = self.robot_stiff_

        # Contact parameters mirror trajectory_optimizer.cc.
        self.model_params = float(getattr(args, "model_param", 100.0))
        self.contact_stiffness_ = float(self.model_params)
        self.smoothing_factor_ = 0.01
        self.dissipation_velocity_ = 0.10
        self.stiction_velocity_ = 0.05
        self.friction_coefficient_ = float(getattr(args, "mu_object_", 0.50))
        self.mu_object_ = self.friction_coefficient_

        self.Qq = np.diag(
            np.hstack(
                [
                    100.0 * np.ones(3),
                    20.0 * np.ones(4),
                    1.0 * np.ones(self.n_robot_qpos_),
                ]
            )
        )
        self.Qv = np.diag(
            np.hstack(
                [
                    5.0 * np.ones(3),
                    1.0 * np.ones(3),
                    0.1 * np.ones(self.n_robot_qvel_),
                ]
            )
        )
        self.Qf_q = np.diag(
            np.hstack(
                [
                    500.0 * np.ones(3),
                    100.0 * np.ones(4),
                    5.0 * np.ones(self.n_robot_qpos_),
                ]
            )
        )
        self.Qf_v = np.diag(
            np.hstack(
                [
                    20.0 * np.ones(3),
                    5.0 * np.ones(3),
                    0.5 * np.ones(self.n_robot_qvel_),
                ]
            )
        )
        self.R = np.diag(
            np.hstack([np.zeros(self.n_obj_qvel_), 0.01 * np.ones(self.n_robot_qvel_)])
        )

        obj_pos_lb = np.array([-10.0, -10.0, -10.0], dtype=np.float64)
        obj_pos_ub = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        quat_lb = -1e7 * np.ones(4)
        quat_ub = 1e7 * np.ones(4)
        robot_lb = -1e2 * np.ones(self.n_robot_qpos_)
        robot_ub = 1e2 * np.ones(self.n_robot_qpos_)
        self.mpc_q_lb_ = np.hstack([obj_pos_lb, quat_lb, robot_lb])
        self.mpc_q_ub_ = np.hstack([obj_pos_ub, quat_ub, robot_ub])

        self._configure_scene(args=args, rand_seed=rand_seed, target_type=target_type)
        self.set_initial_state(
            np.hstack([self.init_obj_qpos_, self.init_robot_qpos_]),
            np.zeros(self.n_qvel_, dtype=np.float64),
            update_nominal=False,
            update_robot_goal=False,
        )
        self.set_target_pose(
            self.target_p_,
            self.target_q_,
            robot_goal_q=self.init_robot_qpos_,
            update_nominal=False,
        )
        self._refresh_decision_bounds()

        self.q_nom_traj_ = self.default_q_nom_trajectory()
        self.v_nom_traj_ = self.default_v_nom_trajectory(self.q_nom_traj_)
        self.last_isaac_observation_ = None

    def _configure_scene(self, args=None, rand_seed=1, target_type="rotation"):
        obj_name = getattr(args, "obj", "obj")
        self.object_names_ = [obj_name]
        self.model_path_ = f"./envs/xmls/env_fingertips_{obj_name}.xml"
        self.mesh_path_ = f"envs/assets/objects/{obj_name}.stl"

        self.init_robot_qpos_ = np.array(
            [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64
        )

        if args is None:
            self.init_obj_qpos_ = np.array(
                [0.45, 0.0, 0.40, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            self.target_p_ = np.array([0.60, 0.0, 0.40], dtype=np.float64)
            self.target_q_ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            return

        np.random.seed(100 + int(rand_seed))
        init_height = 0.05 + self.table_height
        init_xy_rand = 0.1 * np.random.rand(2)
        init_xy_rand[0] += 0.3
        yaw_angle = -np.pi * np.random.rand(1) + np.pi / 2.0
        init_obj_quat_rand = rpy_to_quaternion_numpy(
            np.hstack([yaw_angle, 0.0, 0.0])
        )
        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))

        if target_type != "rotation":
            raise ValueError(f"Target type {target_type} not supported")

        target_xy_rand = 0.1 * np.random.rand(2)
        target_xy_rand[0] += 0.35
        self.target_p_ = np.hstack([target_xy_rand, init_height - 0.02])
        target_yaw = np.pi * np.random.rand(1) - np.pi / 2.0
        self.target_q_ = rpy_to_quaternion_numpy(np.hstack([target_yaw, 0.0, 0.0]))

    def _update_goal_q(self):
        self.target_q_ = quat_normalize_numpy(self.target_q_)
        self.target_pose_ = np.hstack([self.target_p_, self.target_q_])
        self.goal_q_ = np.hstack([self.target_pose_, self.robot_goal_q_])

    def _refresh_decision_bounds(self):
        self.decision_lb_ = np.tile(self.mpc_q_lb_, self.mpc_horizon_ + 1)
        self.decision_ub_ = np.tile(self.mpc_q_ub_, self.mpc_horizon_ + 1)
        self.decision_lb_[: self.n_qpos_] = self.q_init_
        self.decision_ub_[: self.n_qpos_] = self.q_init_

    def set_target_pose(
        self,
        target_position,
        target_quaternion,
        robot_goal_q=None,
        update_nominal=True,
    ):
        self.target_p_ = np.asarray(target_position, dtype=np.float64).reshape(3)
        self.target_q_ = quat_normalize_numpy(target_quaternion)
        if robot_goal_q is None:
            if hasattr(self, "robot_goal_q_"):
                robot_goal_q = self.robot_goal_q_
            else:
                robot_goal_q = self.q_init_[self.robot_q_slice]
        self.robot_goal_q_ = np.asarray(robot_goal_q, dtype=np.float64).reshape(
            self.n_robot_qpos_
        )
        self._update_goal_q()
        if update_nominal and hasattr(self, "decision_lb_"):
            self.q_nom_traj_ = self.default_q_nom_trajectory()
            self.v_nom_traj_ = self.default_v_nom_trajectory(self.q_nom_traj_)

    def set_initial_state(
        self,
        q_init,
        v_init=None,
        update_nominal=True,
        update_robot_goal=True,
    ):
        self.q_init_ = self._normalized_q_numpy(q_init)
        if v_init is None:
            v_init = np.zeros(self.n_qvel_, dtype=np.float64)
        self.v_init_ = np.asarray(v_init, dtype=np.float64).reshape(self.n_qvel_)
        self.init_obj_qpos_ = self.q_init_[self.obj_q_slice].copy()
        self.init_robot_qpos_ = self.q_init_[self.robot_q_slice].copy()
        self.init_state_ = np.hstack([self.q_init_, self.v_init_])

        if update_robot_goal or not hasattr(self, "robot_goal_q_"):
            self.robot_goal_q_ = self.init_robot_qpos_.copy()
        if hasattr(self, "target_p_") and hasattr(self, "target_q_"):
            self._update_goal_q()
        if hasattr(self, "decision_lb_"):
            self._refresh_decision_bounds()
        if update_nominal and hasattr(self, "goal_q_"):
            self.q_nom_traj_ = self.default_q_nom_trajectory()
            self.v_nom_traj_ = self.default_v_nom_trajectory(self.q_nom_traj_)

    def _normalized_q_expr(self, q):
        quat = quat_normalize_casadi(q[self.obj_quat_slice])
        return cs.vertcat(q[self.obj_pos_slice], quat, q[self.robot_q_slice])

    def _normalized_q_numpy(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(self.n_qpos_).copy()
        q[self.obj_quat_slice] = quat_normalize_numpy(q[self.obj_quat_slice])
        return q

    def _angular_velocity_from_quats_expr(self, quat_prev, quat_curr):
        quat_prev = quat_normalize_casadi(quat_prev)
        quat_curr = quat_normalize_casadi(quat_curr)
        q_delta = quat_multiply_casadi(quat_curr, quat_conjugate_casadi(quat_prev))
        sign = cs.if_else(q_delta[0] >= 0, 1.0, -1.0)
        q_delta = sign * q_delta
        return (2.0 / self.h_) * q_delta[1:4]

    def _angular_velocity_from_quats_numpy(self, quat_prev, quat_curr):
        quat_prev = quat_normalize_numpy(quat_prev)
        quat_curr = quat_normalize_numpy(quat_curr)
        w1, x1, y1, z1 = quat_curr
        w2, x2, y2, z2 = np.array(
            [quat_prev[0], -quat_prev[1], -quat_prev[2], -quat_prev[3]]
        )
        q_delta = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )
        if q_delta[0] < 0.0:
            q_delta *= -1.0
        return (2.0 / self.h_) * q_delta[1:4]

    def _velocity_from_positions_expr(self, q_prev, q_curr):
        q_prev = self._normalized_q_expr(q_prev)
        q_curr = self._normalized_q_expr(q_curr)
        obj_linear_v = (q_curr[self.obj_pos_slice] - q_prev[self.obj_pos_slice]) / self.h_
        obj_angular_v = self._angular_velocity_from_quats_expr(
            q_prev[self.obj_quat_slice], q_curr[self.obj_quat_slice]
        )
        robot_v = (q_curr[self.robot_q_slice] - q_prev[self.robot_q_slice]) / self.h_
        return cs.vertcat(obj_linear_v, obj_angular_v, robot_v)

    def _velocity_from_positions_numpy(self, q_prev, q_curr):
        q_prev = self._normalized_q_numpy(q_prev)
        q_curr = self._normalized_q_numpy(q_curr)
        obj_linear_v = (q_curr[self.obj_pos_slice] - q_prev[self.obj_pos_slice]) / self.h_
        obj_angular_v = self._angular_velocity_from_quats_numpy(
            q_prev[self.obj_quat_slice], q_curr[self.obj_quat_slice]
        )
        robot_v = (q_curr[self.robot_q_slice] - q_prev[self.robot_q_slice]) / self.h_
        return np.hstack([obj_linear_v, obj_angular_v, robot_v])

    def _contact_wrench_expr(self, q_next, contact_step_param):
        q_next = self._normalized_q_expr(q_next)
        obj_pos = q_next[self.obj_pos_slice]

        total_force = cs.SX.zeros(3, 1)
        total_torque = cs.SX.zeros(3, 1)

        for i in range(self.max_contacts_):
            base = i * self.contact_param_dim_per_contact_
            signed_distance = contact_step_param[base]
            normal_W = contact_step_param[base + 1 : base + 4]
            point_W = contact_step_param[base + 4 : base + 7]
            relative_velocity_W = contact_step_param[base + 7 : base + 10]

            normal_W = normal_W / cs.sqrt(cs.sumsqr(normal_W) + 1e-8)
            vn = cs.dot(normal_W, relative_velocity_W)
            vt = relative_velocity_W - vn * normal_W

            s = vn / self.dissipation_velocity_
            dissipation_factor = cs.if_else(
                s < 0.0,
                1.0 - s,
                cs.if_else(s < 2.0, ((s - 2.0) * (s - 2.0)) / 4.0, 0.0),
            )

            exponent = -signed_distance / self.smoothing_factor_
            compliant_fn = cs.if_else(
                exponent >= 37.0,
                -self.contact_stiffness_ * signed_distance,
                self.smoothing_factor_
                * self.contact_stiffness_
                * cs.log(1.0 + cs.exp(exponent)),
            )
            fn = compliant_fn * dissipation_factor

            tangential_dir = -vt / cs.sqrt(
                self.stiction_velocity_ * self.stiction_velocity_ + cs.sumsqr(vt)
            )
            ft = tangential_dir * self.friction_coefficient_ * fn
            force_W = normal_W * fn + ft
            torque_W = cs.cross(point_W - obj_pos, force_W)

            total_force += force_W
            total_torque += torque_W

        return cs.vertcat(total_force, total_torque)

    def _inverse_dynamics_expr(self, q_next, v_next, a_curr, contact_step_param):
        wrench_obj = self._contact_wrench_expr(q_next, contact_step_param)
        force_obj = wrench_obj[:3]
        torque_obj = wrench_obj[3:]

        a_obj_linear = a_curr[self.obj_linear_v_slice]
        a_obj_angular = a_curr[self.obj_angular_v_slice]
        a_robot = a_curr[self.robot_v_slice]

        v_obj_linear = v_next[self.obj_linear_v_slice]
        v_obj_angular = v_next[self.obj_angular_v_slice]
        v_robot = v_next[self.robot_v_slice]

        tau_obj_linear = (
            self.obj_mass_ * a_obj_linear
            - self.obj_mass_ * cs.DM(self.gravity_)
            + cs.DM(self.obj_linear_damping_) @ v_obj_linear
            - force_obj
        )
        tau_obj_angular = (
            cs.DM(self.obj_inertia_) @ a_obj_angular
            + cs.DM(self.obj_angular_damping_) @ v_obj_angular
            - torque_obj
        )
        tau_robot = (
            cs.DM(self.robot_inertia_) @ a_robot
            + cs.DM(self.robot_damping_) @ v_robot
        )
        return cs.vertcat(tau_obj_linear, tau_obj_angular, tau_robot)

    def _rollout_symbolic(self, q_traj_flat, contact_schedule):
        q_raw = cs.reshape(q_traj_flat, self.n_qpos_, self.mpc_horizon_ + 1)
        q_list = [self._normalized_q_expr(q_raw[:, t]) for t in range(self.mpc_horizon_ + 1)]
        q_mat = cs.hcat(q_list)

        v_list = [cs.DM(self.v_init_)]
        for t in range(1, self.mpc_horizon_ + 1):
            v_list.append(self._velocity_from_positions_expr(q_list[t - 1], q_list[t]))
        v_mat = cs.hcat(v_list)

        contact_mat = cs.reshape(
            contact_schedule, self.contact_param_dim_per_step_, self.mpc_horizon_
        )

        a_list = []
        tau_list = []
        wrench_list = []
        h_list = []
        for t in range(self.mpc_horizon_):
            a_curr = (v_list[t + 1] - v_list[t]) / self.h_
            tau_curr = self._inverse_dynamics_expr(
                q_list[t + 1], v_list[t + 1], a_curr, contact_mat[:, t]
            )
            wrench_curr = self._contact_wrench_expr(q_list[t + 1], contact_mat[:, t])

            a_list.append(a_curr)
            tau_list.append(tau_curr)
            wrench_list.append(wrench_curr)
            h_list.append(tau_curr[: self.n_obj_qvel_])

        a_mat = (
            cs.hcat(a_list) if a_list else cs.SX.zeros(self.n_qvel_, 0)
        )
        tau_mat = (
            cs.hcat(tau_list) if tau_list else cs.SX.zeros(self.n_qvel_, 0)
        )
        wrench_mat = (
            cs.hcat(wrench_list) if wrench_list else cs.SX.zeros(6, 0)
        )
        h = cs.vertcat(*h_list) if h_list else cs.SX.zeros(0, 1)

        return {
            "q": q_mat,
            "v": v_mat,
            "a": a_mat,
            "tau": tau_mat,
            "wrench": wrench_mat,
            "h": h,
        }

    def default_q_nom_trajectory(self):
        alpha = np.linspace(0.0, 1.0, self.mpc_horizon_ + 1)
        q_traj = np.zeros((self.n_qpos_, self.mpc_horizon_ + 1), dtype=np.float64)
        for idx, s in enumerate(alpha):
            q_t = (1.0 - s) * self.q_init_ + s * self.goal_q_
            q_t[self.obj_quat_slice] = quat_normalize_numpy(q_t[self.obj_quat_slice])
            q_traj[:, idx] = q_t
        q_traj[:, 0] = self.q_init_
        return q_traj

    def default_v_nom_trajectory(self, q_nom_traj=None):
        if q_nom_traj is None:
            q_nom_traj = self.q_nom_traj_
        q_nom_traj = _coerce_trajectory(
            q_nom_traj, self.mpc_horizon_ + 1, self.n_qpos_, "q_nom_traj"
        )
        v_nom = np.zeros((self.n_qvel_, self.mpc_horizon_ + 1), dtype=np.float64)
        v_nom[:, 0] = self.v_init_
        for t in range(1, self.mpc_horizon_ + 1):
            v_nom[:, t] = self._velocity_from_positions_numpy(
                q_nom_traj[:, t - 1], q_nom_traj[:, t]
            )
        return v_nom

    def default_initial_guess(self, q_goal=None):
        if q_goal is None:
            q_goal = self.goal_q_
        q_goal = np.asarray(q_goal, dtype=np.float64).reshape(self.n_qpos_)
        alpha = np.linspace(0.0, 1.0, self.mpc_horizon_ + 1)
        q_guess = np.zeros((self.n_qpos_, self.mpc_horizon_ + 1), dtype=np.float64)
        for idx, s in enumerate(alpha):
            q_t = (1.0 - s) * self.q_init_ + s * q_goal
            q_t[self.obj_quat_slice] = quat_normalize_numpy(q_t[self.obj_quat_slice])
            q_guess[:, idx] = q_t
        q_guess[:, 0] = self.q_init_
        return _flatten_cols(q_guess)

    def pack_cost_param(self, q_nom_traj=None, v_nom_traj=None):
        if q_nom_traj is None:
            q_nom_traj = self.q_nom_traj_
        q_nom_traj = _coerce_trajectory(
            q_nom_traj, self.mpc_horizon_ + 1, self.n_qpos_, "q_nom_traj"
        )
        q_nom_traj[self.obj_quat_slice, :] = np.apply_along_axis(
            quat_normalize_numpy, 0, q_nom_traj[self.obj_quat_slice, :]
        )

        if v_nom_traj is None:
            v_nom_traj = self.default_v_nom_trajectory(q_nom_traj)
        v_nom_traj = _coerce_trajectory(
            v_nom_traj, self.mpc_horizon_ + 1, self.n_qvel_, "v_nom_traj"
        )
        return np.hstack([_flatten_cols(q_nom_traj), _flatten_cols(v_nom_traj)])

    def default_cost_param(self):
        return self.pack_cost_param(self.q_nom_traj_, self.v_nom_traj_)

    def _get_object_velocity_world_from_isaac(self, simulator):
        if hasattr(simulator, "get_object_velocity_world"):
            linear_velocity, angular_velocity = simulator.get_object_velocity_world()
            return (
                np.asarray(linear_velocity, dtype=np.float64).reshape(3),
                np.asarray(angular_velocity, dtype=np.float64).reshape(3),
            )
        return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    def _get_robot_joint_velocity_from_isaac(self, simulator):
        if hasattr(simulator, "get_current_joint_velocity"):
            return np.asarray(
                simulator.get_current_joint_velocity(), dtype=np.float64
            ).reshape(self.n_robot_qvel_)
        return np.zeros(self.n_robot_qvel_, dtype=np.float64)

    def _object_point_velocity_world(self, obj_linear_velocity, obj_angular_velocity, point_world, obj_pos):
        obj_linear_velocity = np.asarray(obj_linear_velocity, dtype=np.float64).reshape(3)
        obj_angular_velocity = np.asarray(obj_angular_velocity, dtype=np.float64).reshape(3)
        point_world = np.asarray(point_world, dtype=np.float64).reshape(3)
        obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
        return obj_linear_velocity + np.cross(obj_angular_velocity, point_world - obj_pos)

    def get_state_from_isaac(self, simulator):
        q_curr = np.asarray(simulator.get_state(), dtype=np.float64).reshape(self.n_qpos_)
        q_curr = self._normalized_q_numpy(q_curr)
        obj_linear_velocity, obj_angular_velocity = self._get_object_velocity_world_from_isaac(
            simulator
        )
        robot_velocity = self._get_robot_joint_velocity_from_isaac(simulator)
        v_curr = np.hstack([obj_linear_velocity, obj_angular_velocity, robot_velocity])
        return q_curr, v_curr, np.hstack([q_curr, v_curr])

    def sync_initial_state_from_isaac(
        self,
        simulator,
        update_nominal=True,
        update_robot_goal=True,
    ):
        q_curr, v_curr, init_state = self.get_state_from_isaac(simulator)
        self.set_initial_state(
            q_curr,
            v_curr,
            update_nominal=update_nominal,
            update_robot_goal=update_robot_goal,
        )
        return q_curr, v_curr, init_state

    def extract_isaac_contact_observation(self, simulator):
        q_curr, v_curr, _ = self.get_state_from_isaac(simulator)
        obj_pos = q_curr[self.obj_pos_slice]
        obj_quat = q_curr[self.obj_quat_slice]
        obj_linear_velocity = v_curr[self.obj_linear_v_slice]
        obj_angular_velocity = v_curr[self.obj_angular_v_slice]
        robot_velocity = v_curr[self.robot_v_slice]

        phi_vec = np.ones((self.max_ncon_ * 4,), dtype=np.float64)
        jac_mat = np.zeros((self.max_ncon_ * 4, self.n_qvel_), dtype=np.float64)
        jac_mat_env = np.zeros((self.max_ncon_ * 4, self.n_qvel_), dtype=np.float64)
        signed_distances = 1e3 * np.ones(self.max_contacts_, dtype=np.float64)
        normals_W = np.zeros((self.max_contacts_, 3), dtype=np.float64)
        contact_points_W = np.zeros((self.max_contacts_, 3), dtype=np.float64)
        relative_velocities_W = np.zeros((self.max_contacts_, 3), dtype=np.float64)
        contact_points_obj = []

        row_idx = 0
        row_env_idx = 0
        contact_idx = 0
        if_contact = False
        has_table_contact = False

        contacts = simulator.get_physx_contacts() if hasattr(simulator, "get_physx_contacts") else []
        obj_body_idx = getattr(simulator, "obj_body_idx", None)
        table_body_idx = getattr(simulator, "table_body_idx", None)
        franka_body_indices = set(getattr(simulator, "franka_body_indices", []))

        for contact in contacts:
            body0 = int(contact.get("body0", -1))
            body1 = int(contact.get("body1", -1))
            if obj_body_idx is None or (body0 != obj_body_idx and body1 != obj_body_idx):
                continue

            signed_distance = float(contact.get("separation", 0.0))
            normal_world = np.asarray(
                contact.get("normal", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
                dtype=np.float64,
            ).reshape(3)
            if body1 == obj_body_idx:
                normal_world = -normal_world
            normal_world = _normalize_vec(
                normal_world, default=np.array([0.0, 0.0, 1.0], dtype=np.float64)
            )

            contact_point_world = contact.get("pos", None)
            if contact_point_world is None:
                contact_point_world = obj_pos + self.contact_point_fallback_offset_ * normal_world
            else:
                contact_point_world = np.asarray(contact_point_world, dtype=np.float64).reshape(3)

            other_sim_idx = body1 if body0 == obj_body_idx else body0
            other_is_franka = other_sim_idx in franka_body_indices
            other_is_table = table_body_idx is not None and other_sim_idx == table_body_idx

            if other_is_franka and signed_distance <= self.if_contact_separation_threshold_:
                if_contact = True
            if other_is_table:
                has_table_contact = True

            obj_point_velocity = self._object_point_velocity_world(
                obj_linear_velocity, obj_angular_velocity, contact_point_world, obj_pos
            )
            other_point_velocity = np.zeros(3, dtype=np.float64)
            if other_is_franka and hasattr(simulator, "get_body_point_jacobian"):
                point_jacobian = np.asarray(
                    simulator.get_body_point_jacobian(other_sim_idx, contact_point_world),
                    dtype=np.float64,
                )
                other_point_velocity = point_jacobian @ robot_velocity
            relative_velocity_world = obj_point_velocity - other_point_velocity

            r_obj = contact_point_world - obj_pos
            j_obj = np.zeros((3, self.n_qvel_), dtype=np.float64)
            j_obj[:, 0:3] = np.eye(3, dtype=np.float64)
            j_obj[:, 3:6] = -_skew(r_obj)
            j_other = np.zeros((3, self.n_qvel_), dtype=np.float64)
            if other_is_franka and hasattr(simulator, "get_body_point_jacobian"):
                j_other[:, self.n_obj_qvel_ :] = np.asarray(
                    simulator.get_body_point_jacobian(other_sim_idx, contact_point_world),
                    dtype=np.float64,
                )
            j_rel_point = j_obj - j_other
            normal_basis, tangent_1, tangent_2 = _tangent_basis_from_normal_numpy(normal_world)
            con_jac = _contact_jacobian_numpy(
                normal_basis, tangent_1, tangent_2, j_rel_point, self.mu_object_
            )

            if (other_is_franka or other_is_table) and row_idx < self.max_ncon_:
                phi_vec[4 * row_idx : 4 * row_idx + 4] = 0.5 * signed_distance
                jac_mat[4 * row_idx : 4 * row_idx + 4, :] = con_jac
                row_idx += 1

            if other_is_table and row_env_idx < self.max_ncon_:
                jac_mat_env[4 * row_env_idx : 4 * row_env_idx + 4, :] = con_jac
                row_env_idx += 1
                contact_points_obj.append(
                    quat_to_rot_numpy(quat_conjugate_numpy(obj_quat))
                    @ (contact_point_world - obj_pos)
                )

            if contact_idx < self.max_contacts_:
                signed_distances[contact_idx] = signed_distance
                normals_W[contact_idx] = normal_world
                contact_points_W[contact_idx] = contact_point_world
                relative_velocities_W[contact_idx] = relative_velocity_world
                contact_idx += 1

        if not has_table_contact:
            dist_table = float(obj_pos[2] - self.table_height)
            if dist_table < self.table_contact_distance_threshold_:
                normal_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                contact_point_world = obj_pos + np.array([0.0, 0.0, -0.025], dtype=np.float64)
                relative_velocity_world = self._object_point_velocity_world(
                    obj_linear_velocity, obj_angular_velocity, contact_point_world, obj_pos
                )

                j_rel_table = np.zeros((3, self.n_qvel_), dtype=np.float64)
                j_rel_table[:, 0:3] = np.eye(3, dtype=np.float64)
                normal_basis, tangent_1, tangent_2 = _tangent_basis_from_normal_numpy(normal_world)
                con_jac_table = _contact_jacobian_numpy(
                    normal_basis, tangent_1, tangent_2, j_rel_table, self.mu_object_
                )
                if row_env_idx < self.max_ncon_:
                    jac_mat_env[4 * row_env_idx : 4 * row_env_idx + 4, :] = con_jac_table
                contact_points_obj.append(np.array([0.0, 0.0, -0.025], dtype=np.float64))

                if contact_idx < self.max_contacts_:
                    signed_distances[contact_idx] = dist_table
                    normals_W[contact_idx] = normal_world
                    contact_points_W[contact_idx] = contact_point_world
                    relative_velocities_W[contact_idx] = relative_velocity_world

        contact_step = self.pack_contact_step(
            signed_distances=signed_distances,
            normals_W=normals_W,
            contact_points_W=contact_points_W,
            relative_velocities_W=relative_velocities_W,
        )
        return {
            "q_curr": q_curr,
            "v_curr": v_curr,
            "phi_vec": phi_vec,
            "jac_mat": jac_mat,
            "jac_mat_env": jac_mat_env,
            "signed_distances": signed_distances,
            "normals_W": normals_W,
            "contact_points_W": contact_points_W,
            "relative_velocities_W": relative_velocities_W,
            "contact_points_obj": contact_points_obj,
            "if_contact": if_contact,
            "contact_step": contact_step,
        }

    def extract_isaac_contact_step(self, simulator):
        return self.extract_isaac_contact_observation(simulator)["contact_step"]

    def extract_isaac_contact_schedule(
        self, simulator, repeat=True, contact_steps=None, contact_step=None
    ):
        if contact_steps is None:
            if contact_step is None:
                contact_step = self.extract_isaac_contact_step(simulator)
            if repeat:
                contact_steps = [contact_step.copy() for _ in range(self.mpc_horizon_)]
            else:
                contact_steps = [contact_step] + [
                    self.zero_contact_step() for _ in range(self.mpc_horizon_ - 1)
                ]
        return self.pack_contact_schedule(contact_steps)

    def build_isaac_inputs(
        self,
        simulator,
        q_nom_traj=None,
        v_nom_traj=None,
        sync_initial_state=True,
        repeat_contact_schedule=True,
    ):
        if sync_initial_state:
            q_curr, v_curr, init_state = self.sync_initial_state_from_isaac(
                simulator,
                update_nominal=q_nom_traj is None and v_nom_traj is None,
                update_robot_goal=True,
            )
        else:
            q_curr, v_curr, init_state = self.get_state_from_isaac(simulator)

        if q_nom_traj is None:
            q_nom_traj = self.q_nom_traj_
        if v_nom_traj is None:
            v_nom_traj = self.default_v_nom_trajectory(q_nom_traj)
            v_nom_traj[:, 0] = v_curr

        contact_obs = self.extract_isaac_contact_observation(simulator)
        contact_schedule = self.extract_isaac_contact_schedule(
            simulator,
            repeat=repeat_contact_schedule,
            contact_step=contact_obs["contact_step"],
        )
        cost_param = self.pack_cost_param(q_nom_traj=q_nom_traj, v_nom_traj=v_nom_traj)

        self.last_isaac_observation_ = {
            "q_curr": q_curr,
            "v_curr": v_curr,
            "init_state": init_state,
            "q_nom_traj": q_nom_traj,
            "v_nom_traj": v_nom_traj,
            "cost_param": cost_param,
            "contact_schedule": contact_schedule,
            **contact_obs,
        }
        return self.last_isaac_observation_

    def pack_contact_step(
        self,
        signed_distances=None,
        normals_W=None,
        contact_points_W=None,
        relative_velocities_W=None,
    ):
        if signed_distances is None:
            signed_distances = 1e3 * np.ones(self.max_contacts_, dtype=np.float64)
        if normals_W is None:
            normals_W = np.zeros((self.max_contacts_, 3), dtype=np.float64)
        if contact_points_W is None:
            contact_points_W = np.zeros((self.max_contacts_, 3), dtype=np.float64)
        if relative_velocities_W is None:
            relative_velocities_W = np.zeros((self.max_contacts_, 3), dtype=np.float64)

        signed_distances = np.asarray(signed_distances, dtype=np.float64).reshape(
            self.max_contacts_
        )
        normals_W = np.asarray(normals_W, dtype=np.float64).reshape(
            self.max_contacts_, 3
        )
        contact_points_W = np.asarray(contact_points_W, dtype=np.float64).reshape(
            self.max_contacts_, 3
        )
        relative_velocities_W = np.asarray(
            relative_velocities_W, dtype=np.float64
        ).reshape(self.max_contacts_, 3)

        contact_step = np.zeros(self.contact_param_dim_per_step_, dtype=np.float64)
        for i in range(self.max_contacts_):
            base = i * self.contact_param_dim_per_contact_
            contact_step[base] = signed_distances[i]
            contact_step[base + 1 : base + 4] = normals_W[i]
            contact_step[base + 4 : base + 7] = contact_points_W[i]
            contact_step[base + 7 : base + 10] = relative_velocities_W[i]
        return contact_step

    def zero_contact_step(self):
        return self.pack_contact_step()

    def pack_contact_schedule(self, contact_steps=None):
        if contact_steps is None:
            contact_steps = [
                self.zero_contact_step() for _ in range(self.mpc_horizon_)
            ]
        contact_mat = _coerce_trajectory(
            contact_steps,
            self.mpc_horizon_,
            self.contact_param_dim_per_step_,
            "contact_steps",
        )
        return _flatten_cols(contact_mat)

    def zero_contact_schedule(self):
        return self.pack_contact_schedule()

    def init_model_fns(self):
        q_prev = cs.SX.sym("q_prev", self.n_qpos_)
        q_curr = cs.SX.sym("q_curr", self.n_qpos_)
        q_next = cs.SX.sym("q_next", self.n_qpos_)
        contact_step = cs.SX.sym("contact_step", self.contact_param_dim_per_step_)

        v_curr = self._velocity_from_positions_expr(q_prev, q_curr)
        v_next = self._velocity_from_positions_expr(q_curr, q_next)
        a_curr = (v_next - v_curr) / self.h_
        wrench = self._contact_wrench_expr(q_next, contact_step)
        tau = self._inverse_dynamics_expr(q_next, v_next, a_curr, contact_step)
        h = tau[: self.n_obj_qvel_]

        q_traj_flat = cs.SX.sym("q_traj_flat", self.n_decision_)
        contact_schedule = cs.SX.sym("contact_schedule", self.contact_schedule_dim_)
        rollout = self._rollout_symbolic(q_traj_flat, contact_schedule)

        velocity_fn = cs.Function(
            "velocity_from_q_fn",
            [q_prev, q_curr],
            [v_curr],
            ["q_prev", "q_curr"],
            ["v_curr"],
        )
        contact_wrench_fn = cs.Function(
            "contact_wrench_fn",
            [q_next, contact_step],
            [wrench],
            ["q_next", "contact_step"],
            ["wrench_obj"],
        )
        inverse_dynamics_fn = cs.Function(
            "inverse_dynamics_fn",
            [q_prev, q_curr, q_next, contact_step],
            [tau],
            ["q_prev", "q_curr", "q_next", "contact_step"],
            ["tau"],
        )
        unactuated_violation_fn = cs.Function(
            "unactuated_violation_fn",
            [q_prev, q_curr, q_next, contact_step],
            [h],
            ["q_prev", "q_curr", "q_next", "contact_step"],
            ["h"],
        )
        rollout_fn = cs.Function(
            "rollout_fn",
            [q_traj_flat, contact_schedule],
            [
                rollout["q"],
                rollout["v"],
                rollout["a"],
                rollout["tau"],
                rollout["wrench"],
                rollout["h"],
            ],
            ["q_traj_flat", "contact_schedule"],
            ["q", "v", "a", "tau", "wrench", "h"],
        )

        return (
            velocity_fn,
            contact_wrench_fn,
            inverse_dynamics_fn,
            unactuated_violation_fn,
            rollout_fn,
        )

    def init_cost_fns(self):
        q_traj_flat = cs.SX.sym("q_traj_flat", self.n_decision_)
        cost_param = cs.SX.sym("cost_param", self.cost_param_dim_)
        contact_schedule = cs.SX.sym("contact_schedule", self.contact_schedule_dim_)

        rollout = self._rollout_symbolic(q_traj_flat, contact_schedule)
        q_mat = rollout["q"]
        v_mat = rollout["v"]
        tau_mat = rollout["tau"]
        h = rollout["h"]

        q_nom_size = self.n_qpos_ * (self.mpc_horizon_ + 1)
        q_nom_mat = cs.reshape(
            cost_param[:q_nom_size], self.n_qpos_, self.mpc_horizon_ + 1
        )
        v_nom_mat = cs.reshape(
            cost_param[q_nom_size:], self.n_qvel_, self.mpc_horizon_ + 1
        )

        total_cost = 0
        for t in range(self.mpc_horizon_):
            q_err = q_mat[:, t] - q_nom_mat[:, t]
            v_err = v_mat[:, t] - v_nom_mat[:, t]
            total_cost += self.h_ * (
                _quad_form(self.Qq, q_err)
                + _quad_form(self.Qv, v_err)
                + _quad_form(self.R, tau_mat[:, t])
            )

        q_err_T = q_mat[:, self.mpc_horizon_] - q_nom_mat[:, self.mpc_horizon_]
        v_err_T = v_mat[:, self.mpc_horizon_] - v_nom_mat[:, self.mpc_horizon_]
        total_cost += _quad_form(self.Qf_q, q_err_T) + _quad_form(
            self.Qf_v, v_err_T
        )

        total_cost_fn = cs.Function(
            "total_cost_fn",
            [q_traj_flat, cost_param, contact_schedule],
            [total_cost],
            ["q_traj_flat", "cost_param", "contact_schedule"],
            ["total_cost"],
        )
        constraint_fn = cs.Function(
            "constraint_fn",
            [q_traj_flat, contact_schedule],
            [h],
            ["q_traj_flat", "contact_schedule"],
            ["h"],
        )
        objective_and_constraints_fn = cs.Function(
            "objective_and_constraints_fn",
            [q_traj_flat, cost_param, contact_schedule],
            [total_cost, h],
            ["q_traj_flat", "cost_param", "contact_schedule"],
            ["total_cost", "h"],
        )
        return total_cost_fn, constraint_fn, objective_and_constraints_fn


ExplicitMPCParams = IDTOMPPIParams
