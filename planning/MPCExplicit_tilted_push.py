import casadi as cs
import numpy as np

_STAGE1_TWIST_REGULARIZATION = 0.2


def _log_barrier_function(x, target, epsilon=1e-3):
    diff = x - target
    squared_norm = cs.dot(diff, diff) + epsilon
    return cs.log(squared_norm)

def _normalize_quaternion_wxyz(quat_wxyz):
    quat_norm = cs.sqrt(cs.dot(quat_wxyz, quat_wxyz) + 1e-12)
    return quat_wxyz / quat_norm


def _quat_wxyz_to_z_axis_vector(quat_wxyz):
    quat_wxyz = _normalize_quaternion_wxyz(quat_wxyz)
    w = quat_wxyz[0]
    x = quat_wxyz[1]
    y = quat_wxyz[2]
    z = quat_wxyz[3]

    # R(q) e_z: the object's local z-axis expressed in the reference frame.
    # Matching only this vector means x/y axes are free to spin around z.
    return cs.vertcat(
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    )


def _quat_wxyz_to_x_axis_vector(quat_wxyz):
    quat_wxyz = _normalize_quaternion_wxyz(quat_wxyz)
    w = quat_wxyz[0]
    x = quat_wxyz[1]
    y = quat_wxyz[2]
    z = quat_wxyz[3]

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


def _plane_projected_x_axis_cost(curr_quat_wxyz, target_quat_wxyz):
    target_z_axis = _quat_wxyz_to_z_axis_vector(target_quat_wxyz)
    curr_x_axis = _quat_wxyz_to_x_axis_vector(curr_quat_wxyz)
    target_x_axis = _quat_wxyz_to_x_axis_vector(target_quat_wxyz)

    curr_x_axis_tangent = curr_x_axis - cs.dot(curr_x_axis, target_z_axis) * target_z_axis
    target_x_axis_tangent = target_x_axis - cs.dot(target_x_axis, target_z_axis) * target_z_axis
    curr_x_axis_tangent = _normalize_vector(curr_x_axis_tangent)
    target_x_axis_tangent = _normalize_vector(target_x_axis_tangent)

    axis_alignment = cs.dot(curr_x_axis_tangent, target_x_axis_tangent)
    axis_alignment = cs.fmax(cs.fmin(axis_alignment, 1.0), -1.0)
    return 1.0 - axis_alignment


def _z_axis_alignment_cost(curr_quat_wxyz, target_quat_wxyz):
    curr_z_axis = _quat_wxyz_to_z_axis_vector(curr_quat_wxyz)
    target_z_axis = _quat_wxyz_to_z_axis_vector(target_quat_wxyz)
    axis_alignment = cs.dot(curr_z_axis, target_z_axis)
    axis_alignment = cs.fmax(cs.fmin(axis_alignment, 1.0), -1.0)
    return 1.0 - axis_alignment


def _quaternion_alignment_cost(curr_quat_wxyz, target_quat_wxyz):
    curr_quat_wxyz = _normalize_quaternion_wxyz(curr_quat_wxyz)
    target_quat_wxyz = _normalize_quaternion_wxyz(target_quat_wxyz)
    return 1.0 - cs.dot(curr_quat_wxyz, target_quat_wxyz) ** 2


def build_isaac_cost_fns(param):
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

    position_cost = cs.sumsqr(x[0:3] - target_position)
    quaternion_cost = _quaternion_alignment_cost(x[3:7], target_quaternion)
    contact_cost = cs.sumsqr(x[0:3] - x[7:10])
    control_cost = cs.sumsqr(u)
    robot_contact_force_map = cs.reshape(robot_contact_force_map_flat, 3, param.max_ncon_ * 4)
    refine_stage_gate = cs.fmax(verify_cost_param, use_full_pose_terminal_cost)

    # Keep the same world-frame cost semantics as MuJoCo:
    # x[0:3] is object position in world, x[3:7] is object quaternion in wxyz,
    # and x[7:10] is EE position in world.
    virtual_point_cost = cs.log(cs.dot(x[7:10] - virtual_point, x[7:10] - virtual_point) + 1e-3)
    contact_point_cost = cs.log(cs.dot(x[7:10] - contact_point, x[7:10] - contact_point) + 1e-3)

    phi_vec = cs.SX.sym("phi_vec", param.max_ncon_ * 4)
    jac_mat = cs.SX.sym("jac_mat", param.max_ncon_ * 4, param.n_qvel_)
    q_inv = np.linalg.inv(param.Q)
    b_o = cs.DM(param.obj_mass_ * param.gravity_)
    b_r = cs.DM(param.robot_stiff_) @ u
    b = cs.vertcat(b_o, b_r)
    raw_contact_force = -param.model_params * (jac_mat @ q_inv @ b + phi_vec)
    contact_force = cs.fmax(raw_contact_force, 0)
    predicted_contact_force_world = robot_contact_force_map @ contact_force
    force_tracking_cost = execute_best_force * verify_cost_param * cs.sumsqr(
        predicted_contact_force_world - desired_force_world
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

    reject_cost = cs.if_else(
        cs.sumsqr(x[7:10] - contact_point) < param.reject_dis,
        -contact_point_cost,
        0.0,
    )
    attract_cost = param.attract_coef * virtual_point_cost + param.reject_coef * reject_cost

    base_cost = (1 - verify_cost_param) * attract_cost + param.contact_coef * verify_cost_param * (
        param.contact_cost_param * contact_cost + (1 - param.contact_cost_param) * contact_point_cost
    )
    path_cost = base_cost + 50 * control_cost * verify_cost_param + force_tracking_cost
    final_cost = (500 * position_cost * curr_ori_coef + 5.0 * quaternion_cost) * refine_stage_gate

    path_cost_fn = cs.Function(
        "path_cost_fn_cart_isaac",
        [x, u, cost_param],
        [path_cost],
    )
    final_cost_fn = cs.Function(
        "final_cost_fn_cart_isaac",
        [x, cost_param],
        [final_cost],
    )
    return path_cost_fn, final_cost_fn


class CartesianExplicitIsaacModel:
    def __init__(self, param):
        self.param_ = param
        self._init_utils()
        self._init_model()

    def _init_utils(self):
        quat = cs.SX.sym("quat", 4)
        h_q_body = cs.vertcat(
            cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
            cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
            cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]),
        )
        self.cs_qmat_body_fn_ = cs.Function("cs_qmat_body_fn_cart_isaac", [quat], [h_q_body.T])

        qvel = cs.SX.sym("qvel", self.param_.n_qvel_)
        qpos = cs.SX.sym("qpos", self.param_.n_qpos_)
        next_obj_pos = qpos[0:3] + self.param_.h_ * qvel[0:3]
        next_ee_pos = qpos[7:10] + self.param_.h_ * qvel[6:9]
        next_obj_quat = qpos[3:7] + 0.5 * self.param_.h_ * self.cs_qmat_body_fn_(qpos[3:7]) @ qvel[3:6]
        next_qpos = cs.vertcat(next_obj_pos, next_obj_quat, next_ee_pos)
        self.cs_qposInteg_ = cs.Function("cs_qposInteg_cart_isaac", [qpos, qvel], [next_qpos])

    def _init_model(self):
        curr_q = cs.SX.sym("curr_q", self.param_.n_qpos_)
        cmd = cs.SX.sym("cmd", self.param_.n_cmd_)
        phi_vec = cs.SX.sym("phi_vec", self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym("jac_mat", self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        model_params = cs.SX.sym("sigma", 1)

        b_o = cs.DM(self.param_.obj_mass_ * self.param_.gravity_)
        b_r = cs.DM(self.param_.robot_stiff_) @ cmd
        b = cs.vertcat(b_o, b_r)

        q_inv = np.linalg.inv(self.param_.Q)
        v_non_contact = q_inv @ b / self.param_.h_

        raw_contact_force = -model_params @ (jac_mat @ q_inv @ b + phi_vec)
        contact_force = cs.fmax(raw_contact_force, 0)
        v_contact = q_inv @ jac_mat.T @ contact_force / self.param_.h_

        v = v_non_contact + v_contact
        next_qpos = self.cs_qposInteg_(curr_q, v)
        self.step_once_fn = cs.Function(
            "step_once_cart_isaac",
            [curr_q, cmd, phi_vec, jac_mat, model_params],
            [next_qpos],
        )


class MPCExplicitIsaac:
    def __init__(self, param):
        self.param_ = param
        self.path_cost_fn, self.final_cost_fn = build_isaac_cost_fns(param)
        self.mpc_model = param.mpc_model
        if self.mpc_model != "explicit":
            raise ValueError(f"Invalid model type: {self.mpc_model}")
        self.model = CartesianExplicitIsaacModel(param)
        self.init_MPC()

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
        use_full_pose_terminal_cost=False,
        sol_guess=None,
        prev_u=None,
        desired_force_world=None,
        robot_contact_force_map=None,
        execute_best_force=False,
        u_lb=None,
        u_ub=None,
    ):
        if sol_guess is None:
            sol_guess = dict(x0=self.nlp_w0_, lam_x0=self.nlp_lam_x0_, lam_g0=self.nlp_lam_g0_)

        if desired_force_world is None:
            desired_force_world = np.zeros(3, dtype=np.float32)
        else:
            desired_force_world = np.asarray(desired_force_world, dtype=np.float32).reshape(3)

        if robot_contact_force_map is None:
            robot_contact_force_map = np.zeros((3, self.param_.max_ncon_ * 4), dtype=np.float32)
        else:
            robot_contact_force_map = np.asarray(robot_contact_force_map, dtype=np.float32).reshape(
                3,
                self.param_.max_ncon_ * 4,
            )

        execute_best_force = float(bool(execute_best_force))
        use_full_pose_terminal_cost = float(bool(use_full_pose_terminal_cost))
        if u_lb is None:
            u_lb = self.param_.mpc_u_lb_
        if u_ub is None:
            u_ub = self.param_.mpc_u_ub_

        u_lb = np.asarray(u_lb, dtype=np.float32)
        u_ub = np.asarray(u_ub, dtype=np.float32)
        if u_lb.ndim == 0:
            u_lb = np.full((self.param_.n_cmd_,), float(u_lb), dtype=np.float32)
        else:
            u_lb = u_lb.reshape(self.param_.n_cmd_)
        if u_ub.ndim == 0:
            u_ub = np.full((self.param_.n_cmd_,), float(u_ub), dtype=np.float32)
        else:
            u_ub = u_ub.reshape(self.param_.n_cmd_)

        cost_params = cs.vvcat(
            [
                target_p,
                target_q,
                phi_vec,
                jac_mat,
                verify_cost_param,
                virtual_point,
                contact_point,
                curr_ori_coef,
                np.array([use_full_pose_terminal_cost], dtype=np.float32),
                desired_force_world,
                np.reshape(robot_contact_force_map, (-1,), order="F"),
                np.array([execute_best_force], dtype=np.float32),
            ]
        )
        nlp_param = self.nlp_params_fn_(curr_x, phi_vec, jac_mat, cost_params, self.param_.model_params)
        nlp_lbw, nlp_ubw = self.nlp_bounds_fn_(
            u_lb,
            u_ub,
            self.param_.mpc_q_lb_,
            self.param_.mpc_q_ub_,
        )

        raw_sol = self.ipopt_solver(
            x0=sol_guess["x0"],
            lam_x0=sol_guess["lam_x0"],
            lam_g0=sol_guess["lam_g0"],
            lbx=nlp_lbw,
            ubx=nlp_ubw,
            lbg=0.0,
            ubg=0.0,
            p=nlp_param,
        )

        w_opt = raw_sol["x"].full().flatten()
        cost_opt = raw_sol["f"].full().flatten()
        sol_traj = np.reshape(w_opt, (self.param_.mpc_horizon_, -1))
        opt_u_traj = sol_traj[:, 0 : self.param_.n_cmd_]
        action = self._postprocess_action(
            opt_u_traj[0, :],
            target_p=target_p,
            curr_x=curr_x,
            u_lb=u_lb,
            u_ub=u_ub,
        )

        return dict(
            action=action,
            sol_guess=dict(
                x0=w_opt,
                lam_x0=raw_sol["lam_x"],
                lam_g0=raw_sol["lam_g"],
                opt_cost=raw_sol["f"].full().item(),
            ),
            cost_opt=cost_opt,
            solve_status=self.ipopt_solver.stats()["return_status"],
        )

    def _postprocess_action(self, action, target_p, curr_x, u_lb, u_ub):
        action = np.asarray(action, dtype=np.float32).reshape(self.param_.n_cmd_).copy()
        if not bool(getattr(self.param_, "project_mpc_action_to_support_tangent_", False)):
            return action
        if self.param_.n_cmd_ != 3:
            return action

        support_normal = _normalize_vector_np(
            getattr(self.param_, "mpc_action_support_normal_", getattr(self.param_, "support_surface_normal_", None))
        )
        if support_normal is None:
            return action

        action = action - float(np.dot(action, support_normal)) * support_normal
        return np.clip(action, u_lb, u_ub).astype(np.float32)

    def init_MPC(self):
        model_params = cs.SX.sym("model_param", 1)
        phi_vec = cs.SX.sym("phi_vec", self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym("jac_mat", self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        cost_params = cs.SX.sym("cost_params", self.path_cost_fn.size_in(2))

        lbu = cs.SX.sym("lbu", self.param_.n_cmd_)
        ubu = cs.SX.sym("ubu", self.param_.n_cmd_)
        lbq = cs.SX.sym("lbq", self.param_.n_qpos_)
        ubq = cs.SX.sym("ubq", self.param_.n_qpos_)

        w, w0, lbw, ubw, g = [], [], [], [], []
        q0 = cs.SX.sym("q", self.param_.n_qpos_)
        qk = q0
        j = 0.0
        for k in range(self.param_.mpc_horizon_):
            uk = cs.SX.sym(f"u{k}", self.param_.n_cmd_)
            w += [uk]
            lbw += [lbu]
            ubw += [ubu]
            w0 += [cs.DM.zeros(self.param_.n_cmd_)]

            pred_q = self.model.step_once_fn(qk, uk, phi_vec, jac_mat, model_params)
            # Evaluate the stage cost on the predicted next state so the first control
            # directly optimizes where the EE will land after applying uk.
            j += self.path_cost_fn(pred_q, uk, cost_params)

            qk = cs.SX.sym(f"q{k + 1}", self.param_.n_qpos_)
            w += [qk]
            w0 += [cs.DM.zeros(self.param_.n_qpos_)]
            lbw += [lbq]
            ubw += [ubq]
            g += [pred_q - qk]

        j += self.final_cost_fn(qk, cost_params)

        nlp_params = cs.vvcat([q0, phi_vec, jac_mat, cost_params, model_params])
        nlp_prog = {"f": j, "x": cs.vcat(w), "g": cs.vcat(g), "p": nlp_params}
        nlp_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": self.param_.ipopt_max_iter_,
            "ipopt.tol": 1e-4,
            "ipopt.linear_solver": "mumps",
        }
        self.ipopt_solver = cs.nlpsol("solver", "ipopt", nlp_prog, nlp_opts)

        self.nlp_w0_ = cs.vcat(w0)
        self.nlp_lam_x0_ = cs.DM.zeros(self.nlp_w0_.shape)
        self.nlp_lam_g0_ = cs.DM.zeros(cs.vcat(g).shape)
        self.nlp_bounds_fn_ = cs.Function(
            "nlp_bounds_fn_cart_isaac",
            [lbu, ubu, lbq, ubq],
            [cs.vcat(lbw), cs.vvcat(ubw)],
        )
        self.nlp_params_fn_ = cs.Function(
            "nlp_params_fn_cart_isaac",
            [q0, phi_vec, jac_mat, cost_params, model_params],
            [nlp_params],
        )
