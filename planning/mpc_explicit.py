"""Unified explicit MPC solver: acados first, IPOPT as fallback.

Cost definitions live in ``planning/mpc_costs.py``.  This module only
builds and solves the receding-horizon NLP.
"""
import os
import warnings

import casadi as cs
import numpy as np

from models.explicit_model import ExplicitModel
from planning.mpc_costs import (
    acados_solver_profile,
    build_cost_fns,
    infer_cost_kind,
    pack_cost_params,
    postprocess_tilted_push_action,
    stage_cost_on_next_state,
    uses_isaac_model,
)

try:
    import torch
except ImportError:
    torch = None


_ACADOS_STATUS_LABELS = {
    -1: "ACADOS_UNKNOWN",
    0: "ACADOS_SUCCESS",
    1: "ACADOS_NAN_DETECTED",
    2: "ACADOS_MAXITER",
    3: "ACADOS_MINSTEP",
    4: "ACADOS_QP_FAILURE",
    5: "ACADOS_READY",
    6: "ACADOS_UNBOUNDED",
    7: "ACADOS_TIMEOUT",
    8: "ACADOS_QPSCALING_BOUNDS_NOT_SATISFIED",
    9: "ACADOS_INFEASIBLE",
}
_ACADOS_IPOPT_FALLBACK_WARNED = False


def _as_vector(value, size):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.full((size,), arr.item(), dtype=np.float64)
    if arr.size != size:
        raise ValueError(f"Expected size {size}, got {arr.size}.")
    return arr


def _normalize_planner_solver_name(solver_name, default="acados"):
    if solver_name is None:
        solver_name = default
    solver_name = str(solver_name).strip().lower()
    if solver_name in {"snopt", "acados"}:
        return "acados"
    if solver_name in {"ipopt", "torch-lbfgs", "torch-gn"}:
        return solver_name
    raise ValueError(
        f"Unsupported planner solver '{solver_name}'. Expected 'acados' or 'ipopt'."
    )


def _warn_acados_ipopt_fallback(reason):
    global _ACADOS_IPOPT_FALLBACK_WARNED
    if _ACADOS_IPOPT_FALLBACK_WARNED:
        return
    _ACADOS_IPOPT_FALLBACK_WARNED = True
    warnings.warn(
        "Falling back from the acados generated planner solver to IPOPT. "
        f"Reason: {reason}",
        RuntimeWarning,
    )


def _format_acados_status(status_code, sqp_iter=None):
    label = _ACADOS_STATUS_LABELS.get(int(status_code), f"ACADOS_STATUS_{int(status_code)}")
    if sqp_iter is None:
        return label
    return f"{label} (sqp_iter={int(sqp_iter)})"


def _load_acados():
    from planning.acados_env import ensure_acados_env
    ensure_acados_env()
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    return AcadosModel, AcadosOcp, AcadosOcpSolver


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

        b = cs.vertcat(cs.DM(self.param_.obj_mass_ * self.param_.gravity_), cs.DM(self.param_.robot_stiff_) @ cmd)
        q_inv = np.linalg.inv(self.param_.Q)
        raw_contact_force = -model_params @ (jac_mat @ q_inv @ b + phi_vec)
        contact_force = cs.fmax(raw_contact_force, 0)
        v = (q_inv @ b + q_inv @ jac_mat.T @ contact_force) / self.param_.h_
        next_qpos = self.cs_qposInteg_(curr_q, v)
        self.step_once_fn = cs.Function(
            "step_once_cart_isaac",
            [curr_q, cmd, phi_vec, jac_mat, model_params],
            [next_qpos],
        )


class MPCExplicit:
    """Single-horizon explicit MPC. Prefers acados; IPOPT is the fallback."""

    def __init__(self, param, cost_kind=None, model=None):
        self.param_ = param
        self.cost_kind = infer_cost_kind(param, cost_kind)
        self.param_.mpc_cost_kind = self.cost_kind
        self.path_cost_fn, self.final_cost_fn = build_cost_fns(param, self.cost_kind)
        self.stage_cost_on_next = stage_cost_on_next_state(self.cost_kind)
        if getattr(param, "smooth_contact_detour", False):
            # Cost the predicted fingertip, otherwise the first stage only
            # sees 50||u||^2 and the ball freezes after a lift.
            self.stage_cost_on_next = True

        self.mpc_model = getattr(param, "mpc_model", "explicit")
        if self.mpc_model != "explicit":
            raise ValueError(f"Invalid model type: {self.mpc_model}")
        if model is not None:
            self.model = model
        elif uses_isaac_model(self.cost_kind):
            self.model = CartesianExplicitIsaacModel(param)
        else:
            self.model = ExplicitModel(param)

        self.ipopt_max_iter_ = int(getattr(param, "ipopt_max_iter_", 500))
        self.param_.mpc_u_lb_ = _as_vector(self.param_.mpc_u_lb_, self.param_.n_cmd_)
        self.param_.mpc_u_ub_ = _as_vector(self.param_.mpc_u_ub_, self.param_.n_cmd_)
        if not hasattr(self.param_, "mpc_q_lb_"):
            self.param_.mpc_q_lb_ = -1e7 * np.ones(self.param_.n_qpos_, dtype=np.float64)
        if not hasattr(self.param_, "mpc_q_ub_"):
            self.param_.mpc_q_ub_ = 1e7 * np.ones(self.param_.n_qpos_, dtype=np.float64)
        self.param_.mpc_q_lb_ = _as_vector(self.param_.mpc_q_lb_, self.param_.n_qpos_)
        self.param_.mpc_q_ub_ = _as_vector(self.param_.mpc_q_ub_, self.param_.n_qpos_)

        requested = getattr(param, "planner_solver_", None)
        if requested is None:
            requested = getattr(param, "torch_solver", None)
        self.planner_solver_ = _normalize_planner_solver_name(requested, default="acados")
        self.param_.planner_solver_ = self.planner_solver_

        self.acados_solver_ = None
        self._acados_init_error = None
        self.acados_solve_count = 0
        self.acados_failure_count = 0
        self.acados_fallback_count = 0
        self.init_MPC()

    def _cost_params(self, **kwargs):
        return pack_cost_params(self.cost_kind, self.param_, self.path_cost_fn, **kwargs)

    def plan_once(
        self,
        target_p=None,
        target_q=None,
        curr_x=None,
        phi_vec=None,
        jac_mat=None,
        verify_cost_param=0,
        virtual_point=None,
        contact_point=None,
        curr_ori_coef=None,
        sol_guess=None,
        **kwargs,
    ):
        curr_x = np.asarray(curr_x, dtype=np.float64).reshape(self.param_.n_qpos_)
        phi_vec = np.asarray(phi_vec, dtype=np.float64).reshape(self.param_.max_ncon_ * 4)
        jac_mat = np.asarray(jac_mat, dtype=np.float64).reshape(self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        cost_params = self._cost_params(
            target_p=target_p,
            target_q=target_q,
            phi_vec=phi_vec,
            jac_mat=jac_mat,
            verify_cost_param=verify_cost_param,
            virtual_point=virtual_point,
            contact_point=contact_point,
            curr_ori_coef=curr_ori_coef,
            **kwargs,
        )
        u_lb = kwargs.get("u_lb", self.param_.mpc_u_lb_)
        u_ub = kwargs.get("u_ub", self.param_.mpc_u_ub_)
        u_lb = _as_vector(u_lb, self.param_.n_cmd_)
        u_ub = _as_vector(u_ub, self.param_.n_cmd_)

        requested = _normalize_planner_solver_name(
            kwargs.get("solver_name", self.planner_solver_),
            default=self.planner_solver_,
        )
        if requested in {"torch-lbfgs", "torch-gn"}:
            try:
                return self._plan_torch(curr_x, phi_vec, jac_mat, cost_params, sol_guess)
            except Exception as exc:
                print(f"Torch MPC failed ({type(exc).__name__}: {exc}); falling back to acados/IPOPT")

        if requested == "acados":
            result = self._plan_once_acados(curr_x, phi_vec, jac_mat, cost_params, sol_guess, u_lb, u_ub)
            if result is None and sol_guess is not None:
                # Status 4 is almost always a stale RTI iterate, not a
                # missing solver.  Cold-start acados before paying IPOPT.
                self._reset_acados_solver()
                result = self._plan_once_acados(
                    curr_x, phi_vec, jac_mat, cost_params, None, u_lb, u_ub
                )
            if result is not None:
                return self._finalize_result(result, curr_x, u_lb, u_ub)
            self.acados_fallback_count += 1
            result = self._plan_once_ipopt(
                curr_x, phi_vec, jac_mat, cost_params, sol_guess, u_lb, u_ub,
                requested_solver="acados",
                fallback_reason="acados_unavailable_or_failed",
            )
            _warn_acados_ipopt_fallback(result.get("fallback_reason", "acados_unavailable_or_failed"))
            return self._finalize_result(result, curr_x, u_lb, u_ub)

        result = self._plan_once_ipopt(
            curr_x, phi_vec, jac_mat, cost_params, sol_guess, u_lb, u_ub,
            requested_solver=requested,
        )
        return self._finalize_result(result, curr_x, u_lb, u_ub)

    def _finalize_result(self, result, curr_x, u_lb, u_ub):
        action = np.asarray(result["action"], dtype=np.float64).reshape(self.param_.n_cmd_)
        if self.cost_kind == "tilted_push" or bool(
            getattr(self.param_, "project_mpc_action_to_support_tangent_", False)
        ):
            action = postprocess_tilted_push_action(self.param_, action, u_lb, u_ub)
        else:
            action = np.clip(action, u_lb, u_ub)
        result["action"] = action
        return result

    def _reset_acados_solver(self):
        solver = getattr(self, "acados_solver_", None)
        if solver is None:
            return
        reset = getattr(solver, "reset", None)
        if not callable(reset):
            return
        try:
            reset()
        except Exception:
            pass

    def _prepare_sol_guess(self, sol_guess):
        if sol_guess is None:
            return dict(x0=self.nlp_w0_, lam_x0=self.nlp_lam_x0_, lam_g0=self.nlp_lam_g0_)
        x0 = sol_guess.get("x0", self.nlp_w0_)
        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0.size != int(np.asarray(self.nlp_w0_).reshape(-1).size):
            return dict(x0=self.nlp_w0_, lam_x0=self.nlp_lam_x0_, lam_g0=self.nlp_lam_g0_)
        return dict(
            x0=x0,
            lam_x0=sol_guess.get("lam_x0", self.nlp_lam_x0_),
            lam_g0=sol_guess.get("lam_g0", self.nlp_lam_g0_),
        )

    def _plan_once_ipopt(self, curr_x, phi_vec, jac_mat, cost_params, sol_guess, u_lb, u_ub,
                         requested_solver="ipopt", fallback_reason=None):
        warm_start = self._prepare_sol_guess(sol_guess)
        nlp_param = self.nlp_params_fn_(curr_x, phi_vec, jac_mat, cost_params, self.param_.model_params)
        nlp_lbw, nlp_ubw = self.nlp_bounds_fn_(u_lb, u_ub, self.param_.mpc_q_lb_, self.param_.mpc_q_ub_)
        raw_sol = self.ipopt_solver(
            x0=warm_start["x0"],
            lam_x0=warm_start["lam_x0"],
            lam_g0=warm_start["lam_g0"],
            lbx=nlp_lbw,
            ubx=nlp_ubw,
            lbg=0.0,
            ubg=0.0,
            p=nlp_param,
        )
        w_opt = raw_sol["x"].full().flatten()
        cost_opt = raw_sol["f"].full().flatten()
        sol_traj = np.reshape(w_opt, (self.param_.mpc_horizon_, self.param_.n_cmd_ + self.param_.n_qpos_))
        opt_u_traj = sol_traj[:, : self.param_.n_cmd_]
        rollout_q = sol_traj[:, self.param_.n_cmd_ :]
        solve_status = self.ipopt_solver.stats()["return_status"]
        result = dict(
            action=opt_u_traj[0, :],
            u_traj=opt_u_traj,
            rollout_q=rollout_q,
            sol_guess=dict(
                x0=w_opt,
                lam_x0=raw_sol["lam_x"],
                lam_g0=raw_sol["lam_g"],
                opt_cost=raw_sol["f"].full().item(),
                solver_backend="ipopt",
                solve_status=solve_status,
            ),
            cost_opt=cost_opt,
            solve_status=solve_status,
            solver_backend="ipopt",
            requested_solver=str(requested_solver),
        )
        if fallback_reason is not None:
            result["fallback_reason"] = str(fallback_reason)
        return result

    def _plan_once_acados(self, curr_x, phi_vec, jac_mat, cost_params, sol_guess, u_lb, u_ub):
        if self.acados_solver_ is None:
            return None
        n, nu, nx = self.param_.mpc_horizon_, self.param_.n_cmd_, self.param_.n_qpos_
        stage_p = np.concatenate(
            [
                np.asarray(phi_vec, dtype=np.float64).reshape(-1),
                np.asarray(jac_mat, dtype=np.float64).reshape(-1, order="F"),
                np.asarray(cost_params, dtype=np.float64).reshape(-1),
                np.asarray([float(self.param_.model_params)], dtype=np.float64),
            ]
        )
        u_guess = np.zeros((n, nu), dtype=np.float64)
        x_guess = np.tile(curr_x.reshape(1, -1), (n + 1, 1))
        if sol_guess is not None:
            stacked = sol_guess.get("x0")
            if stacked is not None:
                stacked = np.asarray(stacked, dtype=np.float64).reshape(-1)
                expected = n * (nu + nx)
                if stacked.size == expected:
                    traj = stacked.reshape(n, nu + nx)
                    u_guess = traj[:, :nu]
                    x_guess[1:] = traj[:, nu:]
            if sol_guess.get("u_traj") is not None:
                u_guess = np.asarray(sol_guess["u_traj"], dtype=np.float64).reshape(n, nu)
            if sol_guess.get("x_traj") is not None:
                x_guess = np.asarray(sol_guess["x_traj"], dtype=np.float64).reshape(n + 1, nx)

        try:
            self.acados_solve_count += 1
            for k in range(n):
                self.acados_solver_.set(k, "x", x_guess[k])
                self.acados_solver_.set(k, "u", np.clip(u_guess[k], u_lb, u_ub))
                self.acados_solver_.set(k, "p", stage_p)
            self.acados_solver_.set(n, "x", x_guess[n])
            self.acados_solver_.set(n, "p", stage_p)
            self.acados_solver_.set(0, "lbx", curr_x)
            self.acados_solver_.set(0, "ubx", curr_x)
            self.acados_solver_.constraints_set(0, "lbu", u_lb)
            self.acados_solver_.constraints_set(0, "ubu", u_ub)
            for k in range(n):
                self.acados_solver_.constraints_set(k, "lbu", u_lb)
                self.acados_solver_.constraints_set(k, "ubu", u_ub)
            from planning.acados_env import quiet_acados_stderr
            with quiet_acados_stderr():
                status = int(self.acados_solver_.solve())
            if status != 0:
                raise RuntimeError(f"acados status {status}")
            u_traj = np.asarray([self.acados_solver_.get(k, "u") for k in range(n)], dtype=np.float64)
            x_traj = np.asarray([self.acados_solver_.get(k, "x") for k in range(n + 1)], dtype=np.float64)
            if not np.isfinite(u_traj).all() or not np.isfinite(x_traj).all():
                raise RuntimeError("acados returned non-finite values")
            try:
                sqp_iter = int(self.acados_solver_.get_stats("sqp_iter"))
            except Exception:
                sqp_iter = None
        except Exception:
            self.acados_failure_count += 1
            return None

        solve_status = _format_acados_status(0, sqp_iter=sqp_iter)
        stacked = np.column_stack([u_traj, x_traj[1:]]).reshape(-1)
        return dict(
            action=u_traj[0, :],
            u_traj=u_traj,
            rollout_q=x_traj[1:],
            sol_guess=dict(
                x0=stacked,
                u_traj=u_traj,
                x_traj=x_traj,
                lam_x0=np.zeros(n * (nu + nx)),
                lam_g0=np.zeros(n * nx),
                solver_backend="acados",
                solve_status=solve_status,
            ),
            cost_opt=np.asarray([0.0], dtype=np.float64),
            solve_status=solve_status,
            solver_backend="acados",
            requested_solver="acados",
        )

    def _plan_torch(self, curr_x, phi_vec, jac_mat, cost_params, sol_guess):
        if torch is None:
            raise RuntimeError("Torch is not installed")
        if self.cost_kind not in {"fingertip", "param"}:
            raise RuntimeError("Torch MPC is only implemented for fingertip costs")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        p = torch.as_tensor(np.asarray(cost_params).reshape(-1), device=dev, dtype=dtype)
        phi = torch.as_tensor(np.asarray(phi_vec).reshape(-1), device=dev, dtype=dtype)
        jac = torch.as_tensor(np.asarray(jac_mat), device=dev, dtype=dtype)
        q0 = torch.as_tensor(np.asarray(curr_x).reshape(-1), device=dev, dtype=dtype)
        n, nu, nq = self.param_.mpc_horizon_, self.param_.n_cmd_, self.param_.n_qpos_
        qlb = torch.as_tensor(self.param_.mpc_q_lb_, device=dev, dtype=dtype)
        qub = torch.as_tensor(self.param_.mpc_q_ub_, device=dev, dtype=dtype)
        ulb = torch.as_tensor(self.param_.mpc_u_lb_, device=dev, dtype=dtype)
        uub = torch.as_tensor(self.param_.mpc_u_ub_, device=dev, dtype=dtype)
        z0 = np.asarray(self._prepare_sol_guess(sol_guess)["x0"]).reshape(-1)
        if z0.size != n * (nu + nq):
            raise ValueError(f"invalid Torch MPC warm start size {z0.size}, expected {n * (nu + nq)}")
        if not np.isfinite(z0).all() or np.linalg.norm(z0) < 1e-12:
            for k in range(n):
                z0[k * (nu + nq) + nu:(k + 1) * (nu + nq)] = np.asarray(curr_x).reshape(-1)
        z = torch.nn.Parameter(torch.as_tensor(z0, device=dev, dtype=dtype))
        opt = torch.optim.LBFGS(
            [z],
            max_iter=int(self.param_.ipopt_max_iter_),
            tolerance_grad=1e-5,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )
        Qinv = torch.as_tensor(np.linalg.inv(self.param_.Q), device=dev, dtype=dtype)
        h = float(self.param_.h_)
        sigma = float(self.param_.model_params)
        target, quat = p[:3], p[3:7]
        verify = p[-7]
        virt, cp = p[-6:-3], p[-3:]

        def closure():
            opt.zero_grad()
            q = q0
            J = torch.zeros((), device=dev)
            off = 0
            for _k in range(n):
                u = torch.clamp(z[off:off + nu], ulb, uub)
                off += nu
                qnext_var = z[off:off + nq]
                off += nq
                b = torch.cat((
                    torch.as_tensor(self.param_.obj_mass_ * self.param_.gravity_, device=dev, dtype=dtype),
                    torch.as_tensor(self.param_.robot_stiff_, device=dev, dtype=dtype) @ u,
                ))
                cf = torch.clamp(-sigma * (jac @ Qinv @ b + phi), min=0)
                v = (Qinv @ b + Qinv @ jac.T @ cf) / h
                qpred = torch.cat((
                    q[:3] + h * v[:3],
                    q[3:7] + 0.5 * h * torch.stack((
                        -q[4] * v[3] - q[5] * v[4] - q[6] * v[5],
                        q[3] * v[3] + q[6] * v[4] - q[5] * v[5],
                        q[3] * v[4] - q[6] * v[3] + q[4] * v[5],
                        q[3] * v[5] + q[5] * v[3] - q[4] * v[4],
                    )),
                    q[-self.param_.n_robot_qpos_:] + h * v[-self.param_.n_robot_qpos_:],
                ))
                q = torch.clamp(qnext_var, qlb, qub)
                J = J + 1e4 * torch.sum((q - qpred) ** 2) + (1 - verify) * (
                    self.param_.attract_coef * torch.log(torch.sum((q[7:10] - virt) ** 2) + 1e-3)
                    + self.param_.reject_coef * torch.where(
                        torch.sum((q[:2] - q[7:9]) ** 2) + 1e-3 < self.param_.reject_dis,
                        1 / (torch.sum((q[:2] - q[7:9]) ** 2) + 1e-3),
                        torch.zeros((), device=dev),
                    )
                ) + self.param_.contact_coef * verify * (
                    self.param_.contact_cost_param * torch.sum((q[:3] - q[7:10]) ** 2)
                    + (1 - self.param_.contact_cost_param) * torch.log(torch.sum((q[7:10] - cp) ** 2) + 1e-3)
                ) + 50 * torch.sum(u * u)
            J = J + 10 * (500 * torch.sum((q[:3] - target) ** 2) + 5 * (1 - torch.dot(q[3:7], quat) ** 2))
            J.backward()
            return J

        opt.step(closure)
        out = z.detach().cpu().numpy()
        traj = out.reshape(n, -1)
        traj[:, :nu] = np.clip(traj[:, :nu], np.asarray(self.param_.mpc_u_lb_), np.asarray(self.param_.mpc_u_ub_))
        traj[:, nu:] = np.clip(traj[:, nu:], np.asarray(self.param_.mpc_q_lb_), np.asarray(self.param_.mpc_q_ub_))
        if not np.isfinite(traj).all():
            raise FloatingPointError("non-finite Torch MPC trajectory")
        out = traj.reshape(-1)
        final_cost = float(closure().detach().cpu())
        return dict(
            action=traj[0, :nu].copy(),
            u_traj=traj[:, :nu].copy(),
            rollout_q=traj[:, nu:].copy(),
            sol_guess=dict(x0=out, lam_x0=np.zeros_like(out), lam_g0=np.zeros(n * self.param_.n_qpos_), opt_cost=final_cost),
            cost_opt=np.array([final_cost]),
            solve_status="torch-lbfgs",
            solver_backend="torch-lbfgs",
        )

    def _build_acados_solver(self):
        AcadosModel, AcadosOcp, AcadosOcpSolver = _load_acados()
        n = int(self.param_.mpc_horizon_)
        nx, nu = int(self.param_.n_qpos_), int(self.param_.n_cmd_)
        phi = cs.SX.sym("phi", self.param_.max_ncon_ * 4)
        jac = cs.SX.sym("jac", self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        cp = cs.SX.sym("cp", self.path_cost_fn.size_in(2))
        mp = cs.SX.sym("model_param", 1)
        x = cs.SX.sym("x", nx)
        u = cs.SX.sym("u", nu)
        p = cs.vvcat([phi, jac, cp, mp])
        pred = self.model.step_once_fn(x, u, phi, jac, mp)

        model = AcadosModel()
        cost_tag = "detour4" if getattr(self.param_, "smooth_contact_detour", False) else "v1"
        model.name = f"mpc_explicit_{self.cost_kind}_{cost_tag}_h{n}_c{self.param_.max_ncon_}_x{nx}_u{nu}"
        model.x, model.u, model.p = x, u, p
        model.disc_dyn_expr = pred
        if self.stage_cost_on_next:
            model.cost_expr_ext_cost = self.path_cost_fn(pred, u, cp)
        else:
            model.cost_expr_ext_cost = self.path_cost_fn(x, u, cp)
        model.cost_expr_ext_cost_e = self.final_cost_fn(x, cp)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros(int(p.size1()))
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.constraints.idxbu = np.arange(nu, dtype=np.int64)
        ocp.constraints.lbu = np.asarray(self.param_.mpc_u_lb_, dtype=float).reshape(nu)
        ocp.constraints.ubu = np.asarray(self.param_.mpc_u_ub_, dtype=float).reshape(nu)
        idx = np.arange(nx, dtype=np.int64)
        ocp.constraints.idxbx = idx
        ocp.constraints.lbx = np.asarray(self.param_.mpc_q_lb_, dtype=float).reshape(nx)
        ocp.constraints.ubx = np.asarray(self.param_.mpc_q_ub_, dtype=float).reshape(nx)
        ocp.constraints.idxbx_e = idx
        ocp.constraints.lbx_e = ocp.constraints.lbx.copy()
        ocp.constraints.ubx_e = ocp.constraints.ubx.copy()
        ocp.constraints.idxbx_0 = idx
        ocp.constraints.lbx_0 = np.zeros(nx)
        ocp.constraints.ubx_0 = np.zeros(nx)
        ocp.solver_options.N_horizon = n
        ocp.solver_options.tf = float(self.param_.h_ * n)
        profile = acados_solver_profile(self.cost_kind)
        for key, value in profile.items():
            setattr(ocp.solver_options, key, value)
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.print_level = 0

        code_dir = os.path.join("/tmp", model.name + "_codegen")
        os.makedirs(code_dir, exist_ok=True)
        ocp.code_gen_opts.code_export_directory = code_dir
        json_file = os.path.join(code_dir, model.name + ".json")
        shared = os.path.join(code_dir, "libacados_ocp_solver_" + model.name + ".so")
        if os.path.isfile(json_file) and os.path.isfile(shared):
            return AcadosOcpSolver(
                ocp, json_file=json_file, generate=False, build=False,
                check_reuse_possible=False, verbose=False,
            )
        return AcadosOcpSolver(
            ocp, json_file=json_file, generate=True, build=True,
            check_reuse_possible=True, verbose=False,
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
        j = 0.0
        q0 = cs.SX.sym("q", self.param_.n_qpos_)
        qk = q0
        for k in range(self.param_.mpc_horizon_):
            uk = cs.SX.sym(f"u{k}", self.param_.n_cmd_)
            w += [uk]
            lbw += [lbu]
            ubw += [ubu]
            w0 += [cs.DM.zeros(self.param_.n_cmd_)]
            pred_q = self.model.step_once_fn(qk, uk, phi_vec, jac_mat, model_params)
            if self.stage_cost_on_next:
                j += self.path_cost_fn(pred_q, uk, cost_params)
            else:
                j += self.path_cost_fn(qk, uk, cost_params)
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
            "ipopt.max_iter": self.ipopt_max_iter_,
            "ipopt.tol": 1e-4,
            "ipopt.linear_solver": "mumps",
        }
        self.ipopt_solver = cs.nlpsol("solver", "ipopt", nlp_prog, nlp_opts)
        self.nlp_w0_ = cs.vcat(w0)
        self.nlp_lam_x0_ = cs.DM.zeros(self.nlp_w0_.shape)
        self.nlp_lam_g0_ = cs.DM.zeros(cs.vcat(g).shape)
        self.nlp_bounds_fn_ = cs.Function("nlp_bounds_fn", [lbu, ubu, lbq, ubq], [cs.vcat(lbw), cs.vvcat(ubw)])
        self.nlp_params_fn_ = cs.Function(
            "nlp_params_fn",
            [q0, phi_vec, jac_mat, cost_params, model_params],
            [nlp_params],
        )

        if self.planner_solver_ != "ipopt":
            try:
                self.acados_solver_ = self._build_acados_solver()
            except Exception as exc:
                self._acados_init_error = exc
                _warn_acados_ipopt_fallback(str(exc))
                print(f"acados initialization failed; IPOPT fallback remains available: {exc}")


class MPCExplicitAcados(MPCExplicit):
    """Compatibility alias used by the fingertip acados example."""


class MPCExplicitIsaac(MPCExplicit):
    def __init__(self, param):
        super().__init__(param, cost_kind="isaac")


class MPCExplicitTiltedPush(MPCExplicit):
    def __init__(self, param):
        super().__init__(param, cost_kind="tilted_push")


def planner_init_payload(args, param, trial_count, device=None):
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


def _build_mpc_planner_runtime(init):
    import argparse
    from planning.acados_env import ensure_acados_env
    ensure_acados_env()
    from examples.mpc.franka.ik2.params import ExplicitMPCParams
    from examples.mpc.fingertips.test.test_0902 import (
        ContactValueTracker,
        ModelCostConfidence,
        SmoothedApproachVia,
    )
    from planning.MPPIWarp import (
        adapt_param_for_cartesian_ranking,
        configure_mppi_rollout_param,
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
    for key in (
        "target_p_", "target_q_", "mesh_path_", "table_height",
        "lambda_obj_mass_", "gravity_", "attract_coef", "contact_coef",
    ):
        if key in init:
            setattr(param, key, init[key])
    mpc = MPCExplicit(param)
    mpc_step = max(1e-4, float(getattr(args, "mpc_step_limit", 0.005)))
    via_step = getattr(args, "via_max_step", None)
    via_step = mpc_step if via_step is None else max(1e-4, float(via_step))
    via_lead = max(1e-4, float(getattr(args, "via_max_lead", via_step)))
    trackers = {
        "value_tracker": ContactValueTracker(
            tau=float(args.value_tau),
            rel_scale=float(args.value_rel_scale),
            rho=float(args.value_rho),
            alpha=float(args.value_alpha),
            beta=float(args.verify_beta),
            window_size=int(getattr(args, "verify_window_size", 20)),
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
            max_step=via_step,
            max_lead=via_lead,
        ),
        "arrived_hold": False,
        "arrived_dest_idx": None,
        "sol_guess": None,
        "last_verify_cost": None,
    }
    return args, param, mpc, trackers


def _supported_call_kwargs(fn, kwargs):
    """Drop kwargs the callee does not accept.

    Isaac / tilted-push payloads always carry support_point.  Older
    ``compute_rollout_contact_via`` copies raise TypeError on that name.
    """
    import inspect
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


def handle_mpc_request(args, param, mpc, trackers, msg):
    import time
    from examples.mpc.fingertips.test.test_0902 import (
        compute_rollout_contact_via,
        _verify_is_chatter,
    )
    from examples.mpc.franka.ik2.contact_frames import gravity_wrench_object_frame
    from planning.MPPIWarp import (
        _apply_dwell_payload,
        _opt_snapshot,
        _pickle_safe,
        clamp_via_to_tip,
    )
    from scipy.spatial.transform import Rotation

    dwell = msg.get("dwell")
    if dwell is not None:
        dwell = dict(dwell)
        dwell["model_cost_conf"] = trackers["model_cost_conf"]
        _apply_dwell_payload(param, args, dwell)

    curr_q = np.asarray(msg["policy_q"], dtype=np.float32)
    max_ncon = int(getattr(param, "max_ncon_", 10))
    nv = int(getattr(param, "n_qvel_", 9))
    phi_vec = (
        np.asarray(msg["phi_vec"])
        if msg.get("phi_vec") is not None
        else np.ones((max_ncon * 4,), dtype=np.float64)
    )
    jac_mat = (
        np.asarray(msg["jac_mat"])
        if msg.get("jac_mat") is not None
        else np.zeros((max_ncon * 4, nv), dtype=np.float64)
    )
    if msg.get("jac_mat_env") is not None:
        jac_mat_env = np.asarray(msg["jac_mat_env"])
    else:
        from examples.mpc.franka.ik2.contact_frames import table_jac_mat_env
        jac_mat_env = table_jac_mat_env(
            curr_q[:3], curr_q[3:7], float(param.table_height),
            nv=nv, mu=float(getattr(param, "mu_object_", 0.5)), max_ncon=max_ncon,
        )
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
    gravity = gravity_wrench_object_frame(
        param.gravity_[:3], ranking_mass, r_obj_to_world, support_normal,
    )
    t0 = time.perf_counter()
    policy = compute_rollout_contact_via(
        param, args, curr_q, r_obj_to_world, gravity, jac_mat_env,
        0.01, trackers["value_tracker"], trackers["model_cost_conf"],
        trackers["approach_via"], trackers["arrived_hold"], trackers["arrived_dest_idx"],
        **_supported_call_kwargs(
            compute_rollout_contact_via,
            dict(
                floor_ground=table_ground,
                floor_z=floor_z,
                support_point=support_point,
                support_normal=support_normal,
            ),
        ),
    )
    rank_dt = time.perf_counter() - t0
    trackers["arrived_hold"] = policy["arrived_hold"]
    trackers["arrived_dest_idx"] = policy["arrived_dest_idx"]
    verify_cost = policy["verify_cost"]
    verify_chatter = _verify_is_chatter(trackers["last_verify_cost"], verify_cost)
    trackers["last_verify_cost"] = float(verify_cost)

    tip = np.asarray(curr_q[7:10], dtype=np.float64)
    max_lead = max(1e-4, float(getattr(args, "via_max_lead", getattr(args, "via_max_step", getattr(args, "mpc_step_limit", 0.005)))))
    value_info = policy.get("value_info") or {}
    if bool(value_info.get("path_blocked", False)):
        trackers["sol_guess"] = None
    # A blocked orbit already walks via at max_step.  Clamping it back
    # to a 3 mm tip lead whenever the tip crosses the keep-out is what
    # made the rim hop.
    if not bool(value_info.get("path_blocked", False)):
        policy["mpc_virtual_point"] = clamp_via_to_tip(tip, policy["mpc_virtual_point"], max_lead)
        policy["mpc_contact_point"] = clamp_via_to_tip(tip, policy["mpc_contact_point"], max_lead)

    t1 = time.perf_counter()
    sol = mpc.plan_once(
        param.target_p_,
        param.target_q_,
        curr_q,
        phi_vec,
        jac_mat,
        verify_cost_param=verify_cost,
        virtual_point=policy["mpc_virtual_point"],
        contact_point=policy["mpc_contact_point"],
        sol_guess=trackers["sol_guess"],
    )
    plan_dt = time.perf_counter() - t1
    trackers["sol_guess"] = sol["sol_guess"]
    policy["choose_dt"] = rank_dt
    return {
        "action": np.asarray(sol["action"], dtype=np.float64).reshape(3),
        "sol_guess": sol["sol_guess"],
        "cost_opt": sol.get("cost_opt"),
        "policy": _pickle_safe(policy),
        "verify_cost": float(verify_cost),
        "verify_chatter": bool(verify_chatter),
        "if_contact": bool(msg.get("if_contact", False)),
        "opt_snapshot": _opt_snapshot(param.lambda_optimizer),
        "model_tightness": float(trackers["model_cost_conf"].tightness()),
        "model_accum": float(trackers["model_cost_conf"].accum),
        "rank_dt": float(rank_dt),
        "plan_dt": float(plan_dt),
        "mppi_dt": float(plan_dt),
        "lambda_failures": int(getattr(param.lambda_optimizer, "acados_failure_count", 0)),
        "mpc_failures": int(getattr(mpc, "acados_failure_count", 0)),
        "lambda_solves": int(getattr(param.lambda_optimizer, "acados_solve_count", 0)),
        "mpc_solves": int(getattr(mpc, "acados_solve_count", 0)),
        "mpc_init_error": str(getattr(mpc, "_acados_init_error", "") or ""),
    }


def mpc_planner_worker(state_q, action_q, ready_q, init):
    """Independent planner loop.  Must not import isaacgym."""
    import traceback
    from planning.MPPIWarp import drain_latest, put_latest

    try:
        args, param, mpc, trackers = _build_mpc_planner_runtime(init)
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
            put_latest(action_q, handle_mpc_request(args, param, mpc, trackers, msg))
        except Exception:
            put_latest(action_q, {"ok": False, "error": traceback.format_exc()})


class AsyncMpcPlanner:
    """ROS-style latest-only state/action bus.  Isaac never waits for a plan."""

    def __init__(self, state_q, action_q, proc):
        self.state_q = state_q
        self.action_q = action_q
        self.proc = proc

    @classmethod
    def start(cls, init, timeout=180.0):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        state_q = ctx.Queue(maxsize=1)
        action_q = ctx.Queue(maxsize=1)
        ready_q = ctx.Queue(maxsize=1)
        proc = ctx.Process(
            target=mpc_planner_worker,
            args=(state_q, action_q, ready_q, init),
            daemon=True,
        )
        proc.start()
        ready = ready_q.get(timeout=timeout)
        if not ready.get("ok", False):
            proc.join(timeout=2.0)
            raise RuntimeError(ready.get("error", "MPC planner process failed to start"))
        return cls(state_q, action_q, proc)

    def publish_state(self, payload):
        from planning.MPPIWarp import put_latest

        return put_latest(self.state_q, payload)

    def take_action(self):
        from planning.MPPIWarp import take_latest

        out = take_latest(self.action_q)
        if isinstance(out, dict) and out.get("ok") is False:
            raise RuntimeError(out.get("error", "MPC planner process failed"))
        return out

    def close(self):
        from planning.MPPIWarp import put_latest

        put_latest(self.state_q, None)
        if self.proc is not None:
            self.proc.join(timeout=5.0)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2.0)
