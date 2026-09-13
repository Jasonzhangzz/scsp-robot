"""Explicit-MPC cost definitions.

The solver in ``planning/mpc_explicit.py`` only consumes CasADi
``path_cost_fn(x, u, p)`` / ``final_cost_fn(x, p)`` plus a packed parameter
vector.  Variant-specific objectives live here.
"""
import casadi as cs
import numpy as np


COST_KINDS = ("param", "fingertip", "bigrasp", "isaac", "tilted_push")


def log_barrier(point, target, epsilon=1e-3):
    diff = point - target
    return cs.log(cs.dot(diff, diff) + epsilon)


def _normalize_quaternion_wxyz(quat_wxyz):
    quat_norm = cs.sqrt(cs.dot(quat_wxyz, quat_wxyz) + 1e-12)
    return quat_wxyz / quat_norm


def _quat_wxyz_to_z_axis_vector(quat_wxyz):
    quat_wxyz = _normalize_quaternion_wxyz(quat_wxyz)
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    return cs.vertcat(
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    )


def _quat_wxyz_to_x_axis_vector(quat_wxyz):
    quat_wxyz = _normalize_quaternion_wxyz(quat_wxyz)
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    return cs.vertcat(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
    )


def _normalize_vector(vec):
    return vec / cs.sqrt(cs.dot(vec, vec) + 1e-12)


def _normalize_vector_np(vec):
    if vec is None:
        return None
    vec = np.asarray(vec, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return (vec / norm).astype(np.float32)


def _quaternion_alignment_cost(curr_quat_wxyz, target_quat_wxyz):
    curr_quat_wxyz = _normalize_quaternion_wxyz(curr_quat_wxyz)
    target_quat_wxyz = _normalize_quaternion_wxyz(target_quat_wxyz)
    return 1.0 - cs.dot(curr_quat_wxyz, target_quat_wxyz) ** 2


def infer_cost_kind(param, explicit=None):
    if explicit:
        return str(explicit).strip().lower()
    kind = getattr(param, "mpc_cost_kind", None)
    if kind:
        return str(kind).strip().lower()
    if getattr(param, "project_mpc_action_to_support_tangent_", False):
        return "tilted_push"
    if hasattr(param, "planner_force_tracking_weight_") or hasattr(param, "planner_solver_"):
        n_qpos = int(getattr(param, "n_qpos_", 0))
        if n_qpos >= 13:
            return "bigrasp"
    if hasattr(param, "init_cost_fns"):
        return "param"
    raise ValueError("Unable to infer an explicit-MPC cost kind from param.")


def build_cost_fns(param, kind=None):
    kind = infer_cost_kind(param, kind)
    if kind not in COST_KINDS:
        raise ValueError(f"Unsupported cost kind '{kind}'. Expected one of {COST_KINDS}.")
    if kind in {"param", "fingertip"}:
        if not hasattr(param, "init_cost_fns"):
            raise ValueError(f"Cost kind '{kind}' requires param.init_cost_fns().")
        return param.init_cost_fns()
    if kind == "bigrasp":
        return build_bigrasp_cost_fns(param)
    if kind == "isaac":
        return build_isaac_cost_fns(param)
    return build_tilted_push_cost_fns(param)


def stage_cost_on_next_state(kind):
    return kind in {"isaac", "tilted_push"}


def uses_isaac_model(kind):
    return kind in {"isaac", "tilted_push"}


def acados_solver_profile(kind):
    if kind == "bigrasp":
        return {
            "qp_solver": "FULL_CONDENSING_HPIPM",
            "nlp_solver_type": "SQP",
            "globalization": "MERIT_BACKTRACKING",
            "regularize_method": "MIRROR",
            "nlp_solver_max_iter": 20,
            "qp_solver_iter_max": 400,
        }
    return {
        "qp_solver": "PARTIAL_CONDENSING_HPIPM",
        "nlp_solver_type": "SQP_RTI",
        "regularize_method": "PROJECT",
    }


def _as_vec(value, size, dtype=np.float64):
    if value is None:
        return np.zeros(size, dtype=dtype)
    arr = np.asarray(value, dtype=dtype).reshape(-1)
    if arr.size == 1 and size != 1:
        return np.full(size, arr.item(), dtype=dtype)
    return arr.reshape(size)


def _as_map(value, rows, cols, dtype=np.float64):
    if value is None:
        return np.zeros((rows, cols), dtype=dtype)
    return np.asarray(value, dtype=dtype).reshape(rows, cols)


def _fit_cost_vector(value, expected_dim):
    value = np.asarray(cs.DM(value), dtype=np.float64).reshape(-1)
    if value.size > expected_dim:
        return value[:expected_dim]
    if value.size < expected_dim:
        return np.concatenate([value, np.zeros(expected_dim - value.size, dtype=np.float64)])
    return value


def pack_cost_params(kind, param, path_cost_fn, **kwargs):
    kind = infer_cost_kind(param, kind)
    expected = int(np.prod(path_cost_fn.size_in(2)))
    if kwargs.get("cost_params") is not None:
        return _fit_cost_vector(kwargs["cost_params"], expected)

    if kind == "bigrasp":
        packed = _pack_bigrasp_cost_params(param, **kwargs)
    elif kind == "isaac":
        packed = _pack_isaac_cost_params(param, **kwargs)
    elif kind == "tilted_push":
        packed = _pack_tilted_push_cost_params(param, **kwargs)
    else:
        packed = _pack_fingertip_cost_params(param, **kwargs)
        compact = _pack_compact_pose_cost_params(param, **kwargs)
        if compact.size == expected:
            packed = compact
    return _fit_cost_vector(packed, expected)


def _pack_compact_pose_cost_params(param, target_p=None, target_q=None, phi_vec=None, jac_mat=None, **_unused):
    n_phi = int(param.max_ncon_ * 4)
    n_qvel = int(param.n_qvel_)
    return np.concatenate(
        [
            _as_vec(target_p, 3),
            _as_vec(target_q, 4),
            _as_vec(phi_vec, n_phi),
            _as_map(jac_mat, n_phi, n_qvel).reshape(-1, order="F"),
        ]
    )


def _pack_fingertip_cost_params(
    param,
    target_p=None,
    target_q=None,
    phi_vec=None,
    jac_mat=None,
    verify_cost_param=0,
    virtual_point=None,
    contact_point=None,
    **_unused,
):
    n_phi = int(param.max_ncon_ * 4)
    n_qvel = int(param.n_qvel_)
    return np.concatenate(
        [
            _as_vec(target_p, 3),
            _as_vec(target_q, 4),
            _as_vec(phi_vec, n_phi),
            _as_map(jac_mat, n_phi, n_qvel).reshape(-1, order="F"),
            np.asarray([float(verify_cost_param)], dtype=np.float64),
            _as_vec(virtual_point, 3),
            _as_vec(contact_point, 3),
        ]
    )


def build_bigrasp_cost_fns(param):
    x = cs.SX.sym("x", param.n_qpos_)
    u = cs.SX.sym("u", param.n_cmd_)

    target_position = cs.SX.sym("target_position", 3)
    target_quaternion = cs.SX.sym("target_quaternion", 4)
    phi_vec = cs.SX.sym("phi_vec", param.max_ncon_ * 4)
    jac_mat = cs.SX.sym("jac_mat", param.max_ncon_ * 4, param.n_qvel_)
    verify_cost_param_1 = cs.SX.sym("verify_cost_1", 1)
    verify_cost_param_2 = cs.SX.sym("verify_cost_2", 1)
    virtual_point_1 = cs.SX.sym("virtual_point_1", 3)
    virtual_point_2 = cs.SX.sym("virtual_point_2", 3)
    contact_point_1 = cs.SX.sym("contact_point_1", 3)
    contact_point_2 = cs.SX.sym("contact_point_2", 3)
    curr_ori_coef_1 = cs.SX.sym("curr_ori_coef_1", 1)
    curr_ori_coef_2 = cs.SX.sym("curr_ori_coef_2", 1)
    desired_force_world = cs.SX.sym("desired_force_world", 3)
    desired_torque_world = cs.SX.sym("desired_torque_world", 3)
    robot_contact_force_map_flat = cs.SX.sym("robot_contact_force_map_flat", 3 * param.max_ncon_ * 4)
    robot_contact_torque_map_flat = cs.SX.sym("robot_contact_torque_map_flat", 3 * param.max_ncon_ * 4)
    execute_desired_wrench = cs.SX.sym("execute_desired_wrench", 1)

    cost_params = cs.vvcat(
        [
            target_position,
            target_quaternion,
            phi_vec,
            jac_mat,
            verify_cost_param_1,
            verify_cost_param_2,
            virtual_point_1,
            virtual_point_2,
            contact_point_1,
            contact_point_2,
            curr_ori_coef_1,
            curr_ori_coef_2,
            desired_force_world,
            desired_torque_world,
            robot_contact_force_map_flat,
            robot_contact_torque_map_flat,
            execute_desired_wrench,
        ]
    )

    obj_pos = x[0:3]
    left_tip = x[7:10]
    right_tip = x[10:13]
    control_cost = cs.sumsqr(u)

    contact_cost_1 = cs.sumsqr(obj_pos - left_tip)
    contact_cost_2 = cs.sumsqr(obj_pos - right_tip)
    contact_point_cost_1 = log_barrier(left_tip, contact_point_1)
    contact_point_cost_2 = log_barrier(right_tip, contact_point_2)
    virtual_point_cost_1 = log_barrier(left_tip, virtual_point_1)
    virtual_point_cost_2 = log_barrier(right_tip, virtual_point_2)

    reject_cost_1 = cs.if_else(
        cs.sumsqr(left_tip - contact_point_1) < param.reject_dis,
        -contact_point_cost_1,
        0.0,
    )
    reject_cost_2 = cs.if_else(
        cs.sumsqr(right_tip - contact_point_2) < param.reject_dis,
        -contact_point_cost_2,
        0.0,
    )
    attract_cost_1 = param.attract_coef * virtual_point_cost_1 + param.reject_coef * reject_cost_1
    attract_cost_2 = param.attract_coef * virtual_point_cost_2 + param.reject_coef * reject_cost_2
    base_cost_1 = (1.0 - verify_cost_param_1) * attract_cost_1 + param.contact_coef * verify_cost_param_1 * (
        param.contact_cost_param * contact_cost_1 + (1.0 - param.contact_cost_param) * contact_point_cost_1
    )
    base_cost_2 = (1.0 - verify_cost_param_2) * attract_cost_2 + param.contact_coef * verify_cost_param_2 * (
        param.contact_cost_param * contact_cost_2 + (1.0 - param.contact_cost_param) * contact_point_cost_2
    )

    q_inv = np.linalg.inv(param.Q)
    b = cs.vertcat(cs.DM(param.obj_mass_ * param.gravity_), cs.DM(param.robot_stiff_) @ u)
    contact_force = cs.fmax(-param.model_params * (jac_mat @ q_inv @ b + phi_vec), 0)
    robot_contact_force_map = cs.reshape(robot_contact_force_map_flat, 3, param.max_ncon_ * 4)
    robot_contact_torque_map = cs.reshape(robot_contact_torque_map_flat, 3, param.max_ncon_ * 4)
    desired_wrench_gate = execute_desired_wrench * verify_cost_param_1 * verify_cost_param_2
    force_tracking_cost = (
        float(getattr(param, "planner_force_tracking_weight_", 1.0))
        * desired_wrench_gate
        * cs.sumsqr(robot_contact_force_map @ contact_force - desired_force_world)
    )
    torque_tracking_cost = (
        float(getattr(param, "planner_torque_tracking_weight_", 1.0))
        * desired_wrench_gate
        * cs.sumsqr(robot_contact_torque_map @ contact_force - desired_torque_world)
    )
    path_cost = base_cost_1 + base_cost_2 + 50.0 * control_cost + force_tracking_cost + torque_tracking_cost

    final_cost_1 = (1.0 - verify_cost_param_1) * cs.sumsqr(left_tip - virtual_point_1) + verify_cost_param_1 * cs.sumsqr(
        left_tip - contact_point_1
    )
    final_cost_2 = (1.0 - verify_cost_param_2) * cs.sumsqr(right_tip - virtual_point_2) + verify_cost_param_2 * cs.sumsqr(
        right_tip - contact_point_2
    )
    final_cost = 10.0 * 500.0 * (final_cost_1 + final_cost_2)
    _ = (curr_ori_coef_1, curr_ori_coef_2, target_position, target_quaternion)
    return (
        cs.Function("path_cost_fn_bigrasp", [x, u, cost_params], [path_cost]),
        cs.Function("final_cost_fn_bigrasp", [x, cost_params], [final_cost]),
    )


def _pack_bigrasp_cost_params(
    param,
    target_p=None,
    target_q=None,
    phi_vec=None,
    jac_mat=None,
    verify_cost_param_1=None,
    verify_cost_param_2=None,
    virtual_point_1=None,
    virtual_point_2=None,
    contact_point_1=None,
    contact_point_2=None,
    curr_ori_coef_1=None,
    curr_ori_coef_2=None,
    desired_force_world=None,
    desired_torque_world=None,
    robot_contact_force_map=None,
    robot_contact_torque_map=None,
    execute_desired_wrench=False,
    verify_cost_param=None,
    virtual_point=None,
    contact_point=None,
    curr_ori_coef=None,
    **_unused,
):
    if verify_cost_param_1 is None:
        verify_cost_param_1 = 0.0 if verify_cost_param is None else verify_cost_param
    if verify_cost_param_2 is None:
        verify_cost_param_2 = 0.0 if verify_cost_param is None else verify_cost_param
    if virtual_point_1 is None:
        virtual_point_1 = virtual_point
    if virtual_point_2 is None:
        virtual_point_2 = virtual_point
    if contact_point_1 is None:
        contact_point_1 = contact_point
    if contact_point_2 is None:
        contact_point_2 = contact_point
    if curr_ori_coef_1 is None:
        curr_ori_coef_1 = 0.0 if curr_ori_coef is None else curr_ori_coef
    if curr_ori_coef_2 is None:
        curr_ori_coef_2 = 0.0 if curr_ori_coef is None else curr_ori_coef
    if verify_cost_param_1 is None or verify_cost_param_2 is None:
        raise ValueError("bigrasp cost requires verify_cost_param_1/2 (or verify_cost_param).")

    n_phi = int(param.max_ncon_ * 4)
    n_qvel = int(param.n_qvel_)
    return np.concatenate(
        [
            _as_vec(target_p, 3),
            _as_vec(target_q, 4),
            _as_vec(phi_vec, n_phi),
            _as_map(jac_mat, n_phi, n_qvel).reshape(-1, order="F"),
            np.asarray([float(verify_cost_param_1)], dtype=np.float64),
            np.asarray([float(verify_cost_param_2)], dtype=np.float64),
            _as_vec(virtual_point_1, 3),
            _as_vec(virtual_point_2, 3),
            _as_vec(contact_point_1, 3),
            _as_vec(contact_point_2, 3),
            np.asarray([0.0 if curr_ori_coef_1 is None else float(curr_ori_coef_1)], dtype=np.float64),
            np.asarray([0.0 if curr_ori_coef_2 is None else float(curr_ori_coef_2)], dtype=np.float64),
            _as_vec(desired_force_world, 3),
            _as_vec(desired_torque_world, 3),
            _as_map(robot_contact_force_map, 3, n_phi).reshape(-1, order="F"),
            _as_map(robot_contact_torque_map, 3, n_phi).reshape(-1, order="F"),
            np.asarray([float(bool(execute_desired_wrench))], dtype=np.float64),
        ]
    )


def build_isaac_cost_fns(param):
    x = cs.SX.sym("x", param.n_qpos_)
    u = cs.SX.sym("u", param.n_cmd_)
    target_position = cs.SX.sym("target_position", 3)
    target_quaternion = cs.SX.sym("target_quaternion", 4)
    verify_cost_param = cs.SX.sym("verify_cost_param", 1)
    virtual_point = cs.SX.sym("virtual_point", 3)
    contact_point = cs.SX.sym("contact_point", 3)
    curr_ori_coef = cs.SX.sym("curr_ori_coef", 1)
    phi_vec = cs.SX.sym("phi_vec", param.max_ncon_ * 4)
    jac_mat = cs.SX.sym("jac_mat", param.max_ncon_ * 4, param.n_qvel_)
    cost_param = cs.vvcat(
        [
            target_position,
            target_quaternion,
            phi_vec,
            jac_mat,
            verify_cost_param,
            virtual_point,
            contact_point,
            curr_ori_coef,
        ]
    )

    position_cost = cs.sumsqr(x[0:3] - target_position)
    quaternion_cost = 1 - cs.dot(x[3:7], target_quaternion) ** 2
    contact_cost = cs.sumsqr(x[0:3] - x[7:10])
    control_cost = cs.sumsqr(u)
    virtual_point_cost = cs.log(cs.dot(x[7:10] - virtual_point, x[7:10] - virtual_point) + 1e-3)
    contact_point_cost = cs.log(cs.dot(x[7:10] - contact_point, x[7:10] - contact_point) + 1e-3)
    reject_cost = cs.if_else(cs.sumsqr(x[7:10] - contact_point) < param.reject_dis, -contact_point_cost, 0.0)
    attract_cost = param.attract_coef * virtual_point_cost + param.reject_coef * reject_cost
    base_cost = (1 - verify_cost_param) * attract_cost + param.contact_coef * verify_cost_param * (
        param.contact_cost_param * contact_cost + (1 - param.contact_cost_param) * contact_point_cost
    )
    final_cost = 500 * position_cost * curr_ori_coef + 5.0 * quaternion_cost
    return (
        cs.Function("path_cost_fn_cart_isaac", [x, u, cost_param], [base_cost + 50 * control_cost * verify_cost_param]),
        cs.Function("final_cost_fn_cart_isaac", [x, cost_param], [10 * final_cost * verify_cost_param]),
    )


def _pack_isaac_cost_params(
    param,
    target_p=None,
    target_q=None,
    phi_vec=None,
    jac_mat=None,
    verify_cost_param=0,
    virtual_point=None,
    contact_point=None,
    curr_ori_coef=None,
    **_unused,
):
    n_phi = int(param.max_ncon_ * 4)
    n_qvel = int(param.n_qvel_)
    return np.concatenate(
        [
            _as_vec(target_p, 3),
            _as_vec(target_q, 4),
            _as_vec(phi_vec, n_phi),
            _as_map(jac_mat, n_phi, n_qvel).reshape(-1, order="F"),
            np.asarray([float(verify_cost_param)], dtype=np.float64),
            _as_vec(virtual_point, 3),
            _as_vec(contact_point, 3),
            np.asarray([1.0 if curr_ori_coef is None else float(curr_ori_coef)], dtype=np.float64),
        ]
    )


def build_tilted_push_cost_fns(param):
    x = cs.SX.sym("x", param.n_qpos_)
    u = cs.SX.sym("u", param.n_cmd_)
    target_position = cs.SX.sym("target_position", 3)
    target_quaternion = cs.SX.sym("target_quaternion", 4)
    verify_cost_param = cs.SX.sym("verify_cost_param", 1)
    virtual_point = cs.SX.sym("virtual_point", 3)
    contact_point = cs.SX.sym("contact_point", 3)
    curr_ori_coef = cs.SX.sym("curr_ori_coef", 1)
    use_full_pose_terminal_cost = cs.SX.sym("use_full_pose_terminal_cost", 1)
    desired_force_world = cs.SX.sym("desired_force_world", 3)
    robot_contact_force_map_flat = cs.SX.sym("robot_contact_force_map_flat", 3 * param.max_ncon_ * 4)
    execute_best_force = cs.SX.sym("execute_best_force", 1)
    phi_vec = cs.SX.sym("phi_vec", param.max_ncon_ * 4)
    jac_mat = cs.SX.sym("jac_mat", param.max_ncon_ * 4, param.n_qvel_)

    position_cost = cs.sumsqr(x[0:3] - target_position)
    quaternion_cost = _quaternion_alignment_cost(x[3:7], target_quaternion)
    contact_cost = cs.sumsqr(x[0:3] - x[7:10])
    control_cost = cs.sumsqr(u)
    robot_contact_force_map = cs.reshape(robot_contact_force_map_flat, 3, param.max_ncon_ * 4)
    refine_stage_gate = cs.fmax(verify_cost_param, use_full_pose_terminal_cost)
    virtual_point_cost = cs.log(cs.dot(x[7:10] - virtual_point, x[7:10] - virtual_point) + 1e-3)
    contact_point_cost = cs.log(cs.dot(x[7:10] - contact_point, x[7:10] - contact_point) + 1e-3)

    q_inv = np.linalg.inv(param.Q)
    b = cs.vertcat(cs.DM(param.obj_mass_ * param.gravity_), cs.DM(param.robot_stiff_) @ u)
    contact_force = cs.fmax(-param.model_params * (jac_mat @ q_inv @ b + phi_vec), 0)
    force_tracking_cost = execute_best_force * verify_cost_param * cs.sumsqr(
        robot_contact_force_map @ contact_force - desired_force_world
    )
    cost_param = cs.vvcat(
        [
            target_position,
            target_quaternion,
            phi_vec,
            jac_mat,
            verify_cost_param,
            virtual_point,
            contact_point,
            curr_ori_coef,
            use_full_pose_terminal_cost,
            desired_force_world,
            robot_contact_force_map_flat,
            execute_best_force,
        ]
    )
    reject_cost = cs.if_else(cs.sumsqr(x[7:10] - contact_point) < param.reject_dis, -contact_point_cost, 0.0)
    attract_cost = param.attract_coef * virtual_point_cost + param.reject_coef * reject_cost
    base_cost = (1 - verify_cost_param) * attract_cost + param.contact_coef * verify_cost_param * (
        param.contact_cost_param * contact_cost + (1 - param.contact_cost_param) * contact_point_cost
    )
    path_cost = base_cost + 50 * control_cost * verify_cost_param + force_tracking_cost
    final_cost = (500 * position_cost * curr_ori_coef + 5.0 * quaternion_cost) * refine_stage_gate
    return (
        cs.Function("path_cost_fn_cart_isaac", [x, u, cost_param], [path_cost]),
        cs.Function("final_cost_fn_cart_isaac", [x, cost_param], [final_cost]),
    )


def _pack_tilted_push_cost_params(
    param,
    target_p=None,
    target_q=None,
    phi_vec=None,
    jac_mat=None,
    verify_cost_param=0,
    virtual_point=None,
    contact_point=None,
    curr_ori_coef=None,
    use_full_pose_terminal_cost=False,
    desired_force_world=None,
    robot_contact_force_map=None,
    execute_best_force=False,
    **_unused,
):
    n_phi = int(param.max_ncon_ * 4)
    n_qvel = int(param.n_qvel_)
    return np.concatenate(
        [
            _as_vec(target_p, 3),
            _as_vec(target_q, 4),
            _as_vec(phi_vec, n_phi),
            _as_map(jac_mat, n_phi, n_qvel).reshape(-1, order="F"),
            np.asarray([float(verify_cost_param)], dtype=np.float64),
            _as_vec(virtual_point, 3),
            _as_vec(contact_point, 3),
            np.asarray([1.0 if curr_ori_coef is None else float(curr_ori_coef)], dtype=np.float64),
            np.asarray([float(bool(use_full_pose_terminal_cost))], dtype=np.float64),
            _as_vec(desired_force_world, 3),
            _as_map(robot_contact_force_map, 3, n_phi).reshape(-1, order="F"),
            np.asarray([float(bool(execute_best_force))], dtype=np.float64),
        ]
    )


def postprocess_tilted_push_action(param, action, u_lb, u_ub):
    action = np.asarray(action, dtype=np.float32).reshape(param.n_cmd_).copy()
    if not bool(getattr(param, "project_mpc_action_to_support_tangent_", False)):
        return action
    if param.n_cmd_ != 3:
        return action
    support_normal = _normalize_vector_np(
        getattr(param, "mpc_action_support_normal_", getattr(param, "support_surface_normal_", None))
    )
    if support_normal is None:
        return action
    action = action - float(np.dot(action, support_normal)) * support_normal
    return np.clip(action, u_lb, u_ub).astype(np.float32)
