import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from curobo.rollout.rollout_base import Goal
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

    _HAS_CUROBO = True
    _CUROBO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    Goal = None
    Pose = None
    JointState = None
    MpcSolver = None
    MpcSolverConfig = None
    _HAS_CUROBO = False
    _CUROBO_IMPORT_ERROR = exc

try:
    import jax.numpy as jnp

    from planning.MPPIExplicit import JaxExplicitEnv, _explicit_step_jax

    _HAS_EXPLICIT_CONTACT_EVAL = True
    _EXPLICIT_CONTACT_EVAL_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    jnp = None
    JaxExplicitEnv = None
    _explicit_step_jax = None
    _HAS_EXPLICIT_CONTACT_EVAL = False
    _EXPLICIT_CONTACT_EVAL_IMPORT_ERROR = exc


MUJOCO_ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
_FK_DH = np.array(
    [
        (0.0, 0.0, 0.333),
        (0.0, -np.pi / 2.0, 0.0),
        (0.0, np.pi / 2.0, 0.316),
        (0.0825, np.pi / 2.0, 0.0),
        (-0.0825, -np.pi / 2.0, 0.384),
        (0.0, np.pi / 2.0, 0.0),
        (0.088, np.pi / 2.0, 0.0),
    ],
    dtype=np.float64,
)
_ATTACH_QUAT_WXYZ = np.array([0.3826834, 0.0, 0.0, 0.9238795], dtype=np.float64)
_ATTACH_POS = np.array([0.0, 0.0, 0.107], dtype=np.float64)
_TIP_POS = np.array([0.0, 0.0, 0.06], dtype=np.float64)
# cuRobo's stock Franka model uses panda_hand as the end-effector. The task
# in this repo targets the custom fingertip added on top of panda_link7, so we
# convert fingertip goals into the equivalent panda_hand pose before solving.
_HAND_TO_TIP_POS = np.array([0.0, 0.0, 0.06], dtype=np.float64)
_HAND_TO_TIP_QUAT_WXYZ = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
_EPS = 1e-6


def _joint_index(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    if match is None:
        raise ValueError(f"Unable to extract joint index from joint name: {name}")
    return int(match.group(1))


def _tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_joint_vector(joint_state_like) -> np.ndarray:
    if hasattr(joint_state_like, "position"):
        joint_state_like = joint_state_like.position
    return _tensor_to_numpy(joint_state_like).reshape(-1).astype(np.float64)


def _safe_metric_scalar(metrics_obj, name: str, default: float = np.nan) -> float:
    if metrics_obj is None or not hasattr(metrics_obj, name):
        return float(default)
    value = getattr(metrics_obj, name)
    array = _tensor_to_numpy(value).reshape(-1)
    if array.size == 0:
        return float(default)
    return float(array[0])


def _extract_curobo_motion_cost(result) -> float:
    candidates = (
        result,
        getattr(result, "metrics", None),
        getattr(result, "debug", None),
        getattr(result, "info", None),
    )
    metric_names = (
        "cost",
        "total_cost",
        "rollout_cost",
        "trajectory_cost",
        "goal_cost",
        "pose_cost",
    )
    for metrics_obj in candidates:
        if metrics_obj is None:
            continue
        for name in metric_names:
            value = _safe_metric_scalar(metrics_obj, name, default=np.nan)
            if np.isfinite(value):
                return float(value)
    return float(np.nan)


def _normalize_quat_wxyz(quat_wxyz: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < _EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_dot_cost(q0_wxyz: Sequence[float], q1_wxyz: Sequence[float]) -> float:
    q0 = _normalize_quat_wxyz(q0_wxyz)
    q1 = _normalize_quat_wxyz(q1_wxyz)
    return float(1.0 - np.dot(q0, q1) ** 2)


def _quat_multiply_wxyz(q0_wxyz: Sequence[float], q1_wxyz: Sequence[float]) -> np.ndarray:
    q0 = _normalize_quat_wxyz(q0_wxyz)
    q1 = _normalize_quat_wxyz(q1_wxyz)
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return _normalize_quat_wxyz(
        np.array(
            [
                w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            ],
            dtype=np.float64,
        )
    )


def _quat_to_rot_wxyz(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = _normalize_quat_wxyz(quat_wxyz)
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


def _rot_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
            ],
            dtype=np.float64,
        )
        return _normalize_quat_wxyz(quat)

    if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        quat = np.array(
            [
                (rot[2, 1] - rot[1, 2]) / s,
                0.25 * s,
                (rot[0, 1] + rot[1, 0]) / s,
                (rot[0, 2] + rot[2, 0]) / s,
            ],
            dtype=np.float64,
        )
        return _normalize_quat_wxyz(quat)

    if rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        quat = np.array(
            [
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[0, 1] + rot[1, 0]) / s,
                0.25 * s,
                (rot[1, 2] + rot[2, 1]) / s,
            ],
            dtype=np.float64,
        )
        return _normalize_quat_wxyz(quat)

    s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
    quat = np.array(
        [
            (rot[1, 0] - rot[0, 1]) / s,
            (rot[0, 2] + rot[2, 0]) / s,
            (rot[1, 2] + rot[2, 1]) / s,
            0.25 * s,
        ],
        dtype=np.float64,
    )
    return _normalize_quat_wxyz(quat)


def _fingertip_goal_to_curobo_pose(
    goal_pos_tip: Sequence[float],
    goal_quat_tip: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    goal_pos_tip = np.asarray(goal_pos_tip, dtype=np.float64).reshape(3)
    goal_quat_tip = _normalize_quat_wxyz(goal_quat_tip)

    rot_world_tip = _quat_to_rot_wxyz(goal_quat_tip)
    rot_hand_tip = _quat_to_rot_wxyz(_HAND_TO_TIP_QUAT_WXYZ)
    rot_world_hand = rot_world_tip @ rot_hand_tip.T
    goal_pos_hand = goal_pos_tip - rot_world_hand @ _HAND_TO_TIP_POS
    goal_quat_hand = _quat_multiply_wxyz(goal_quat_tip, [0.0, 0.0, 0.0, -1.0])
    return goal_pos_hand.astype(np.float64), goal_quat_hand.astype(np.float64)


def _mdh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _franka_fk_transform(q_robot: Sequence[float]) -> np.ndarray:
    q_robot = np.asarray(q_robot, dtype=np.float64).reshape(7)
    transform = np.eye(4, dtype=np.float64)
    for i in range(7):
        a, alpha, d = _FK_DH[i]
        transform = transform @ _mdh_transform(float(a), float(alpha), float(d), float(q_robot[i]))

    attach = np.eye(4, dtype=np.float64)
    attach[:3, :3] = _quat_to_rot_wxyz(_ATTACH_QUAT_WXYZ)
    attach[:3, 3] = _ATTACH_POS

    tip = np.eye(4, dtype=np.float64)
    tip[:3, 3] = _TIP_POS
    return transform @ attach @ tip


def _franka_fk_pose(q_robot: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    transform = _franka_fk_transform(q_robot)
    position = transform[:3, 3].copy()
    quat_wxyz = _rot_to_quat_wxyz(transform[:3, :3])
    return position.astype(np.float64), quat_wxyz.astype(np.float64)


def _expand_bound(bound: Sequence[float], ndim: int) -> np.ndarray:
    array = np.asarray(bound, dtype=np.float64)
    if array.ndim == 0:
        array = np.full((ndim,), float(array), dtype=np.float64)
    return np.broadcast_to(array, (ndim,)).astype(np.float64)


def build_curobo_state(
    arm_q_mj: Sequence[float],
    curobo_joint_names: Sequence[str],
    default_joint_vector: Sequence[float],
) -> np.ndarray:
    arm_q_mj = np.asarray(arm_q_mj, dtype=np.float64).reshape(len(MUJOCO_ARM_JOINT_NAMES))
    joint_vec = np.asarray(default_joint_vector, dtype=np.float64).reshape(len(curobo_joint_names)).copy()
    arm_map = {_joint_index(name): value for name, value in zip(MUJOCO_ARM_JOINT_NAMES, arm_q_mj)}

    for idx, joint_name in enumerate(curobo_joint_names):
        if "finger" in str(joint_name):
            continue
        joint_vec[idx] = arm_map[_joint_index(str(joint_name))]

    return joint_vec


def extract_arm_configuration(
    curobo_joint_vector: Sequence[float],
    curobo_joint_names: Sequence[str],
) -> np.ndarray:
    curobo_joint_vector = np.asarray(curobo_joint_vector, dtype=np.float64).reshape(len(curobo_joint_names))
    arm_map = {
        _joint_index(str(name)): value
        for name, value in zip(curobo_joint_names, curobo_joint_vector)
        if "finger" not in str(name)
    }
    return np.array(
        [arm_map[_joint_index(name)] for name in MUJOCO_ARM_JOINT_NAMES],
        dtype=np.float64,
    )


def make_joint_state(mpc: MpcSolver, joint_vector: Sequence[float]) -> JointState:
    tensor = torch.tensor(
        np.asarray(joint_vector, dtype=np.float32).reshape(1, -1),
        device=mpc.tensor_args.device,
        dtype=mpc.tensor_args.dtype,
    )
    return JointState.from_position(tensor, joint_names=mpc.joint_names)


def make_pose(mpc: MpcSolver, position: Sequence[float], quat_wxyz: Sequence[float]) -> Pose:
    pos_t = torch.tensor(
        np.asarray(position, dtype=np.float32).reshape(1, 3),
        device=mpc.tensor_args.device,
        dtype=mpc.tensor_args.dtype,
    )
    quat_t = torch.tensor(
        np.asarray(quat_wxyz, dtype=np.float32).reshape(1, 4),
        device=mpc.tensor_args.device,
        dtype=mpc.tensor_args.dtype,
    )
    return Pose(position=pos_t, quaternion=quat_t)


class CuroboSolver:
    def __init__(self, param):
        if not _HAS_CUROBO:
            raise ImportError(
                "curobo is required for examples/mpc/franka/mppi2/Curobosolver.py"
            ) from _CUROBO_IMPORT_ERROR

        self.param_ = param
        self.n_robot_qpos_ = int(self.param_.n_robot_qpos_)
        self.h_ = float(self.param_.h_)
        self.u_lb_ = _expand_bound(self.param_.mpc_u_lb_, self.n_robot_qpos_)
        self.u_ub_ = _expand_bound(self.param_.mpc_u_ub_, self.n_robot_qpos_)

        self.max_attempts_ = int(getattr(self.param_, "curobo_max_attempts_", 1))
        self.use_goal_orientation_ = bool(getattr(self.param_, "curobo_use_goal_orientation_", False))
        self.use_midpoint_goal_ = bool(getattr(self.param_, "curobo_use_midpoint_goal_", True))
        self.preview_steps_ = max(int(getattr(self.param_, "curobo_preview_steps_", 8)), 2)
        self.contact_rollout_horizon_ = max(int(getattr(self.param_, "mpc_horizon_", 5)), 1)

        robot_cfg = getattr(self.param_, "curobo_robot_cfg_", "franka.yml")
        world_cfg = getattr(self.param_, "curobo_world_cfg_", {})
        step_dt = float(getattr(self.param_, "curobo_step_dt_", self.h_))
        store_rollouts = bool(getattr(self.param_, "curobo_store_rollouts_", True))

        mpc_config = MpcSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            store_rollouts=store_rollouts,
            step_dt=step_dt,
        )
        self.mpc = MpcSolver(mpc_config)
        self.mpc.enable_pose_cost(enable=True)
        self.mpc.enable_cspace_cost(enable=False)

        self.joint_names_ = list(self.mpc.joint_names)
        retract_cfg = _extract_joint_vector(self.mpc.rollout_fn.dynamics_model.retract_config)
        self.default_joint_vector_ = np.asarray(retract_cfg, dtype=np.float64).reshape(len(self.joint_names_))
        self._goal_buffer = None
        self.prev_dq_ = np.zeros((self.n_robot_qpos_,), dtype=np.float64)
        self._explicit_contact_evaluator = None
        if _HAS_EXPLICIT_CONTACT_EVAL:
            self._explicit_contact_evaluator = JaxExplicitEnv(
                self.param_,
                horizon=self.contact_rollout_horizon_,
            )

    @staticmethod
    def _sanitize_goal_point(point: Optional[Sequence[float]]) -> Optional[np.ndarray]:
        if point is None:
            return None
        point_array = np.asarray(point, dtype=np.float64).reshape(-1)
        if point_array.size < 3 or not np.all(np.isfinite(point_array[:3])):
            return None
        return point_array[:3].copy()

    def _select_goal_position(
        self,
        curr_x: Sequence[float],
        verify_cost_param: float,
        virtual_point: Optional[Sequence[float]],
        contact_point: Optional[Sequence[float]],
    ) -> Tuple[np.ndarray, str]:
        verify_enabled = float(verify_cost_param) > 0.5
        primary_label = "contact_point" if verify_enabled else "virtual_point"
        secondary_label = "virtual_point" if verify_enabled else "contact_point"

        primary = self._sanitize_goal_point(contact_point if verify_enabled else virtual_point)
        if primary is not None:
            return primary, primary_label

        secondary = self._sanitize_goal_point(virtual_point if verify_enabled else contact_point)
        if secondary is not None:
            return secondary, secondary_label

        if self.use_midpoint_goal_ and virtual_point is not None and contact_point is not None:
            vp = self._sanitize_goal_point(virtual_point)
            cp = self._sanitize_goal_point(contact_point)
            if vp is not None and cp is not None:
                return 0.5 * (vp + cp), "midpoint_fallback"

        q_robot = np.asarray(curr_x, dtype=np.float64).reshape(-1)[-self.n_robot_qpos_ :]
        ee_pos, _ = _franka_fk_pose(q_robot)
        return ee_pos.copy(), "ee_fallback"

    def _normalize_explicit_action(self, action: Sequence[float]) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(self.n_robot_qpos_)
        span = self.u_ub_ - self.u_lb_
        safe_span = np.where(np.abs(span) < _EPS, 1.0, span)
        normalized = 2.0 * (action - self.u_lb_) / safe_span - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _goal_orientation(self, curr_q_robot: Sequence[float], target_q: Sequence[float]) -> np.ndarray:
        if self.use_goal_orientation_:
            target_q = np.asarray(target_q, dtype=np.float64).reshape(-1)
            if target_q.size >= 4 and np.all(np.isfinite(target_q[:4])):
                return _normalize_quat_wxyz(target_q[:4])
        _, ee_quat = _franka_fk_pose(curr_q_robot)
        return ee_quat

    def _update_goal(self, current_state: JointState, goal_pose: Pose) -> None:
        goal = Goal(current_state=current_state, goal_state=current_state, goal_pose=goal_pose)
        if self._goal_buffer is None:
            self._goal_buffer = self.mpc.setup_solve_single(goal, 1)
            return

        if hasattr(self._goal_buffer, "goal_pose"):
            self._goal_buffer.goal_pose.copy_(goal_pose)
        if hasattr(self._goal_buffer, "goal_state"):
            self._goal_buffer.goal_state.copy_(current_state)
        if hasattr(self._goal_buffer, "current_state"):
            self._goal_buffer.current_state.copy_(current_state)

    def _compute_pose_tracking_cost(
        self,
        q_robot: Sequence[float],
        goal_pos: Sequence[float],
        goal_quat: Sequence[float],
    ) -> Dict[str, float]:
        ee_pos, ee_quat = _franka_fk_pose(q_robot)
        pos_error = float(np.sum((ee_pos - np.asarray(goal_pos, dtype=np.float64).reshape(3)) ** 2))
        ori_error = _quat_dot_cost(ee_quat, goal_quat)
        return {
            "ee_pos_error_cost": 100.0 * pos_error,
            "ee_ori_error_cost": 10.0 * ori_error,
            "ee_pos_error": float(np.sqrt(max(pos_error, 0.0))),
            "ee_ori_error": float(ori_error),
        }

    def _compute_final_cost(
        self,
        target_p: Sequence[float],
        target_q: Sequence[float],
        curr_x: Sequence[float],
    ) -> Dict[str, float]:
        curr_x = np.asarray(curr_x, dtype=np.float64).reshape(-1)

        obj_pos = curr_x[0:3]
        obj_quat = curr_x[3:7]

        final_position_cost = float(np.sum((obj_pos - np.asarray(target_p, dtype=np.float64).reshape(3)) ** 2))
        final_quaternion_cost = _quat_dot_cost(obj_quat, target_q)
        final_cost = 10.0 * (500.0 * final_position_cost + 5.0 * final_quaternion_cost * 4.0)

        return {
            "final_position_cost": float(final_position_cost),
            "final_quaternion_cost": float(final_quaternion_cost),
            "final_cost": float(final_cost),
        }

    def _compute_contact_rollout_costs(
        self,
        target_p: Sequence[float],
        target_q: Sequence[float],
        curr_x: Sequence[float],
        action: Sequence[float],
        phi_vec: Sequence[float],
        jac_mat: Sequence[float],
        verify_cost_param: float,
        virtual_point: Sequence[float],
        contact_point: Sequence[float],
        curr_ori_coef: Optional[Sequence[float]],
    ) -> Dict[str, float]:
        if self._explicit_contact_evaluator is None:
            final_costs = self._compute_final_cost(target_p=target_p, target_q=target_q, curr_x=curr_x)
            return {
                "contact_path_cost": float(np.nan),
                "contact_final_cost": float(final_costs["final_cost"]),
                "contact_total_cost": float(final_costs["final_cost"]),
                "final_cost": float(final_costs["final_cost"]),
                "final_position_cost": float(final_costs["final_position_cost"]),
                "final_quaternion_cost": float(final_costs["final_quaternion_cost"]),
                "predicted_obj_position_cost": float(final_costs["final_position_cost"]),
                "predicted_obj_quaternion_cost": float(final_costs["final_quaternion_cost"]),
                "contact_rollout_available": 0.0,
            }

        evaluator = self._explicit_contact_evaluator
        curr_x = np.asarray(curr_x, dtype=np.float32).reshape(-1)
        action_norm = jnp.array(self._normalize_explicit_action(action), dtype=jnp.float32)
        curr_ori_value = (
            float(np.asarray(curr_ori_coef, dtype=np.float32).reshape(-1)[0])
            if curr_ori_coef is not None
            else 1.0
        )

        evaluator.set_cost_params(
            target_p=np.asarray(target_p, dtype=np.float32).reshape(3),
            target_q=np.asarray(target_q, dtype=np.float32).reshape(4),
            verify_cost_param=float(verify_cost_param),
            virtual_point=np.asarray(virtual_point, dtype=np.float32).reshape(3),
            contact_point=np.asarray(contact_point, dtype=np.float32).reshape(3),
            curr_ori_coef=curr_ori_value,
        )

        if bool(getattr(self.param_, "use_jax_contact_", True)):
            evaluator.set_contacts_from_state(jnp.array(curr_x, dtype=jnp.float32))
        else:
            evaluator.set_contacts(phi_vec, jac_mat)

        q = jnp.array(curr_x, dtype=jnp.float32)
        qd = jnp.zeros((self.param_.n_qpos_,), dtype=jnp.float32)
        contact_path_cost = 0.0

        for _ in range(self.contact_rollout_horizon_):
            if bool(getattr(self.param_, "use_jax_contact_", True)):
                evaluator.set_contacts_from_state(q)
            u_scaled = evaluator._scale_action(action_norm)
            next_q = _explicit_step_jax(
                q,
                u_scaled,
                evaluator._phi_vec,
                evaluator._jac_mat,
                evaluator._model_params,
                evaluator._params,
            )
            qd_next = (next_q - q) / evaluator._params["h"]
            qdd = (qd_next - qd) / evaluator._params["h"]
            step_path_cost = evaluator._path_cost(next_q, u_scaled, qd_next, qdd)
            contact_path_cost += float(np.asarray(step_path_cost, dtype=np.float64))
            q = next_q
            qd = qd_next

        q_terminal = np.asarray(q, dtype=np.float32).reshape(-1)
        final_cost = float(np.asarray(evaluator._final_cost(q), dtype=np.float64))
        predicted_position_cost = float(
            np.sum((q_terminal[0:3] - np.asarray(target_p, dtype=np.float32).reshape(3)) ** 2)
        )
        predicted_quaternion_cost = float(_quat_dot_cost(q_terminal[3:7], target_q))
        return {
            "contact_path_cost": float(contact_path_cost),
            "contact_final_cost": float(final_cost),
            "contact_total_cost": float(contact_path_cost + final_cost),
            "final_cost": float(final_cost),
            "final_position_cost": predicted_position_cost,
            "final_quaternion_cost": predicted_quaternion_cost,
            "predicted_obj_position_cost": predicted_position_cost,
            "predicted_obj_quaternion_cost": predicted_quaternion_cost,
            "contact_rollout_available": 1.0,
        }

    def _build_rollout_preview(
        self,
        curr_x: Sequence[float],
        q_goal: Sequence[float],
    ) -> np.ndarray:
        curr_x = np.asarray(curr_x, dtype=np.float64).reshape(-1)
        q_start = curr_x[-self.n_robot_qpos_ :]
        q_goal = np.asarray(q_goal, dtype=np.float64).reshape(self.n_robot_qpos_)
        obj_state = curr_x[:7].copy()

        rollout = []
        for alpha in np.linspace(0.0, 1.0, self.preview_steps_):
            q_interp = (1.0 - alpha) * q_start + alpha * q_goal
            rollout.append(np.hstack([obj_state, q_interp]))
        return np.asarray(rollout, dtype=np.float32)

    def plan_once(
        self,
        target_p,
        target_q,
        curr_x,
        phi_vec,
        jac_mat,
        verify_cost_param=0.0,
        virtual_point=None,
        contact_point=None,
        curr_ori_coef=None,
        sol_guess=None,
        ref_joint_traj=None,
        ref_ctrl_traj=None,
    ):
        del sol_guess, ref_joint_traj, ref_ctrl_traj

        curr_x = np.asarray(curr_x, dtype=np.float64).reshape(-1)
        expected_state_dim = 7 + len(MUJOCO_ARM_JOINT_NAMES)
        if curr_x.size < expected_state_dim:
            raise ValueError(
                f"Expected curr_x to contain object pose (7) + Franka joints (7), "
                f"but got shape {curr_x.shape} with size {curr_x.size}."
            )
        curr_q_robot = curr_x[-self.n_robot_qpos_ :]
        verify_cost_param = float(verify_cost_param)

        if virtual_point is None:
            ee_pos, _ = _franka_fk_pose(curr_q_robot)
            virtual_point = ee_pos.copy()
        if contact_point is None:
            contact_point = np.asarray(virtual_point, dtype=np.float64).reshape(3).copy()

        curr_q_curobo = build_curobo_state(
            curr_q_robot,
            self.joint_names_,
            self.default_joint_vector_,
        )
        current_state = make_joint_state(self.mpc, curr_q_curobo)

        goal_quat_tip = self._goal_orientation(curr_q_robot, target_q)
        goal_pos, goal_source = self._select_goal_position(
            curr_x,
            verify_cost_param,
            virtual_point,
            contact_point,
        )
        goal_pos = np.asarray(goal_pos, dtype=np.float64).reshape(3)
        goal_pos_curobo, goal_quat_curobo = _fingertip_goal_to_curobo_pose(goal_pos, goal_quat_tip)
        goal_pose = make_pose(self.mpc, goal_pos_curobo, goal_quat_curobo)
        self._update_goal(current_state, goal_pose)
        self.mpc.update_goal(self._goal_buffer)

        result = self.mpc.step(current_state, max_attempts=self.max_attempts_)
        q_des_curobo = _extract_joint_vector(result.action)
        q_des_robot = extract_arm_configuration(q_des_curobo, self.joint_names_)
        dq_robot = np.clip(q_des_robot - curr_q_robot, self.u_lb_, self.u_ub_)
        q_cmd_robot = curr_q_robot + dq_robot

        pose_costs = self._compute_pose_tracking_cost(q_cmd_robot, goal_pos, goal_quat_tip)
        curobo_pose_error = _safe_metric_scalar(getattr(result, "metrics", None), "pose_error", default=np.nan)
        curobo_motion_cost = _extract_curobo_motion_cost(result)
        fallback_pose_cost = pose_costs["ee_pos_error_cost"] + pose_costs["ee_ori_error_cost"]
        base_cost = float(curobo_motion_cost if np.isfinite(curobo_motion_cost) else fallback_pose_cost)
        contact_costs = self._compute_contact_rollout_costs(
            target_p=target_p,
            target_q=target_q,
            curr_x=curr_x,
            action=dq_robot,
            phi_vec=phi_vec,
            jac_mat=jac_mat,
            verify_cost_param=verify_cost_param,
            virtual_point=np.asarray(virtual_point, dtype=np.float64).reshape(3),
            contact_point=np.asarray(contact_point, dtype=np.float64).reshape(3),
            curr_ori_coef=curr_ori_coef,
        )
        combined_cost = base_cost + float(contact_costs["contact_final_cost"])

        best_candidate = {
            "goal_pos": np.asarray(goal_pos, dtype=np.float32),
            "goal_quat": np.asarray(goal_quat_tip, dtype=np.float32),
            "goal_source": goal_source,
            "curobo_goal_pos": np.asarray(goal_pos_curobo, dtype=np.float32),
            "curobo_goal_quat": np.asarray(goal_quat_curobo, dtype=np.float32),
            "q_cmd_robot": np.asarray(q_cmd_robot, dtype=np.float32),
            "dq_robot": np.asarray(dq_robot, dtype=np.float32),
            "base_cost": float(base_cost),
            "combined_cost": float(combined_cost),
            "curobo_pose_error": float(curobo_pose_error),
            "curobo_motion_cost": float(curobo_motion_cost),
            "pose_costs": pose_costs,
            "contact_costs": contact_costs,
            "result": result,
        }

        self.prev_dq_ = np.asarray(best_candidate["dq_robot"], dtype=np.float64).copy()
        rollout_q = self._build_rollout_preview(curr_x, best_candidate["q_cmd_robot"])

        sol_guess_out = {
            "prev_dq": self.prev_dq_.astype(np.float32).copy(),
            "goal_pos": np.asarray(best_candidate["goal_pos"], dtype=np.float32).copy(),
            "goal_quat": np.asarray(best_candidate["goal_quat"], dtype=np.float32).copy(),
            "opt_cost": float(best_candidate["combined_cost"]),
        }

        cost_breakdown = dict(best_candidate["pose_costs"])
        cost_breakdown.update(best_candidate["contact_costs"])
        cost_breakdown["curobo_pose_error"] = float(best_candidate["curobo_pose_error"])
        cost_breakdown["curobo_motion_cost"] = float(best_candidate["curobo_motion_cost"])
        cost_breakdown["base_cost"] = float(best_candidate["base_cost"])
        cost_breakdown["path_cost"] = float(best_candidate["base_cost"])
        cost_breakdown["combined_cost"] = float(best_candidate["combined_cost"])

        return {
            "action": np.asarray(best_candidate["dq_robot"], dtype=np.float32),
            "joint_target": np.asarray(best_candidate["q_cmd_robot"], dtype=np.float32),
            "goal_pos": np.asarray(best_candidate["goal_pos"], dtype=np.float32),
            "goal_quat": np.asarray(best_candidate["goal_quat"], dtype=np.float32),
            "goal_source": best_candidate["goal_source"],
            "rollout_q": rollout_q,
            "sol_guess": sol_guess_out,
            "cost_opt": np.array([best_candidate["combined_cost"]], dtype=np.float32),
            "solve_status": "curobo_mppi2",
            "pose_error": float(best_candidate["pose_costs"]["ee_pos_error"]),
            "cost_breakdown": cost_breakdown,
            "candidate_costs": [
                {
                    "goal_pos": best_candidate["goal_pos"].tolist(),
                    "goal_source": best_candidate["goal_source"],
                    "base_cost": float(best_candidate["base_cost"]),
                    "final_cost": float(best_candidate["contact_costs"]["contact_final_cost"]),
                    "combined_cost": float(best_candidate["combined_cost"]),
                    "curobo_motion_cost": float(best_candidate["curobo_motion_cost"]),
                }
            ],
        }
