import casadi as cs
import numpy as np


def _log_barrier_function(x, target, epsilon=1e-3):
    diff = x - target
    squared_norm = cs.dot(diff, diff) + epsilon
    return cs.log(squared_norm)


def build_isaac_cost_fns(param):
    x = cs.SX.sym("x", param.n_qpos_)
    u = cs.SX.sym("u", param.n_cmd_)

    target_position = cs.SX.sym("target_position", 3)
    target_quaternion = cs.SX.sym("target_quaternion", 4)
    verify_cost_param = cs.SX.sym("verify_cost_param", 1)
    virtual_point = cs.SX.sym("virtual_point", 3)
    contact_point = cs.SX.sym("contact_point", 3)
    curr_ori_coef = cs.SX.sym("curr_ori_coef", 1)

    position_cost = cs.sumsqr(x[0:3] - target_position)
    quaternion_cost = 1 - cs.dot(x[3:7], target_quaternion) ** 2
    contact_cost = cs.sumsqr(x[0:3] - x[7:10])
    control_cost = cs.sumsqr(u)

    # Keep the same world-frame cost semantics as MuJoCo:
    # x[0:3] is object position in world, x[3:7] is object quaternion in wxyz,
    # and x[7:10] is EE position in world. test_mpc_isaac.py already converts
    # Isaac's native xyzw quaternion/state APIs into this convention before
    # calling plan_once(), so the cost can match params.init_cost_fns() exactly.
    virtual_point_cost = cs.log(cs.dot(x[7:10] - virtual_point, x[7:10] - virtual_point) + 1e-3)
    contact_point_cost = cs.log(cs.dot(x[7:10] - contact_point, x[7:10] - contact_point) + 1e-3)

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

    reject_cost = cs.if_else(
        cs.sumsqr(x[7:10] - contact_point) < param.reject_dis,
        -contact_point_cost,
        0.0,
    )
    attract_cost = param.attract_coef * virtual_point_cost + param.reject_coef * reject_cost

    base_cost = (1 - verify_cost_param) * attract_cost + param.contact_coef * verify_cost_param * (
        param.contact_cost_param * contact_cost + (1 - param.contact_cost_param) * contact_point_cost
    )
    final_cost = 500 * position_cost * curr_ori_coef + 5.0 * quaternion_cost

    path_cost_fn = cs.Function(
        "path_cost_fn_cart_isaac",
        [x, u, cost_param],
        [base_cost + 50 * control_cost * verify_cost_param],
    )
    final_cost_fn = cs.Function(
        "final_cost_fn_cart_isaac",
        [x, cost_param],
        [10 * final_cost * verify_cost_param],
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
        sol_guess=None,
        prev_u=None,
    ):
        if sol_guess is None:
            sol_guess = dict(x0=self.nlp_w0_, lam_x0=self.nlp_lam_x0_, lam_g0=self.nlp_lam_g0_)

        cost_params = cs.vvcat(
            [target_p, target_q, phi_vec, jac_mat, verify_cost_param, virtual_point, contact_point, curr_ori_coef]
        )
        nlp_param = self.nlp_params_fn_(curr_x, phi_vec, jac_mat, cost_params, self.param_.model_params)
        nlp_lbw, nlp_ubw = self.nlp_bounds_fn_(
            self.param_.mpc_u_lb_,
            self.param_.mpc_u_ub_,
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

        return dict(
            action=opt_u_traj[0, :],
            sol_guess=dict(
                x0=w_opt,
                lam_x0=raw_sol["lam_x"],
                lam_g0=raw_sol["lam_g"],
                opt_cost=raw_sol["f"].full().item(),
            ),
            cost_opt=cost_opt,
            solve_status=self.ipopt_solver.stats()["return_status"],
        )

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
