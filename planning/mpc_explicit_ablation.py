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


def _set_acados_code_export_directory(ocp, directory):
    """Support legacy and current acados-template codegen APIs."""
    options = getattr(ocp, "code_gen_opts", None)
    if options is not None:
        options.code_export_directory = directory
    elif hasattr(ocp, "code_export_directory"):
        ocp.code_export_directory = directory
    else:
        raise AttributeError(
            "AcadosOcp exposes neither code_gen_opts nor code_export_directory"
        )


def _acados_codegen_cache_is_current(json_file, shared_file):
    if not (os.path.isfile(json_file) and os.path.isfile(shared_file)):
        return False
    try:
        import json
        with open(json_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return all(key in payload for key in ("acados_lib_path",
                                              "code_export_directory"))
    except (OSError, ValueError, TypeError):
        return False


def _load_acados():
    from planning.acados_env import ensure_acados_env
    ensure_acados_env()
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    return AcadosModel, AcadosOcp, AcadosOcpSolver


class CartesianExplicitIsaacModel(ExplicitModel):
    """Cartesian compatibility name; uses the configured contact predictor."""


class MPCExplicit:
    """Single-horizon explicit MPC. Prefers acados; IPOPT is the fallback."""

    def __init__(self, param, cost_kind=None, model=None):
        self.param_ = param
        self.contact_force_limit = (float(getattr(param, 'contact_command_force_limit_', param.contact_force_limit_))
            if getattr(param, 'runtime_object_cost_', False) and hasattr(param, 'contact_force_limit_') else None)
        self.cost_kind = infer_cost_kind(param, cost_kind)
        self.param_.mpc_cost_kind = self.cost_kind
        self.path_cost_fn, self.final_cost_fn = build_cost_fns(param, self.cost_kind)
        self.stage_cost_on_next = stage_cost_on_next_state(self.cost_kind)

        if getattr(param, "smooth_contact_detour", False):
            # Cost the predicted fingertip, otherwise the first stage only
            # sees 50||u||^2 and the ball freezes after a lift.
            self.stage_cost_on_next = True

        self.source_param_ = param
        self.velocity_state = bool(getattr(param, 'runtime_object_cost_', False)
            and getattr(param, 'contact_model', '') == 'soft-scm'
            and not getattr(param, 'online_scm_', False) and model is None
            and param.n_qpos_ == 10 and param.n_cmd_ == 3
            and getattr(param, 'mpc_carry_velocity_', True))
        self.recoverable_floor = self.velocity_state
        if self.velocity_state:
            import copy
            param = copy.copy(param)
            self.param_ = param
            param.n_qpos_ = 19
            param.mpc_q_lb_ = np.r_[param.mpc_q_lb_, np.full(9, -100.)]
            param.mpc_q_ub_ = np.r_[param.mpc_q_ub_, np.full(9, 100.)]
        self.mpc_model = getattr(param, "mpc_model", "explicit")
        if self.mpc_model != "explicit":
            raise ValueError(f"Invalid model type: {self.mpc_model}")
        if self.velocity_state and model is None:
            from models.cartesian_rigid_contact import CartesianRigidContactModel
            self.model = CartesianRigidContactModel(param)
        elif model is not None:
            self.model = model
        elif uses_isaac_model(self.cost_kind):
            self.model = CartesianExplicitIsaacModel(param)
        else:
            self.model = ExplicitModel(param)

        self.support_bounds = {}
        self.support_plane = {}
        if hasattr(param, 'support_surface_half_extents_'):
            self.support_bounds = dict(support_point=param.support_surface_point_,
                support_tangents=param.support_surface_tangents_,
                support_half_extents=param.support_surface_half_extents_)
            self.support_plane = dict(self.support_bounds,
                support_normal=param.support_surface_normal_,
                support_edges=getattr(param, 'support_edges_', None))
        self.support_vertices = getattr(param, 'support_vertices_', None)
        if self.support_vertices is None and self.velocity_state and hasattr(param, 'mesh_path_'):
            import trimesh
            self.support_vertices = np.asarray(trimesh.load_mesh(
                param.mesh_path_, process=False).convex_hull.vertices)
        self.update_contact_gaps = bool(getattr(param, 'runtime_object_cost_', False))
        state = cs.SX.sym('stage_state',param.n_qpos_)
        reference = cs.SX.sym('contact_reference',param.n_qpos_)
        cmd = cs.SX.sym('stage_cmd',param.n_cmd_)
        phi = cs.SX.sym('stage_phi',4*param.max_ncon_)
        jac = cs.SX.sym('stage_jac',4*param.max_ncon_,param.n_qvel_)
        mp = cs.SX.sym('stage_parameters',self.model.model_parameter_dim)
        gaps, stage_jac = phi, jac
        if self.update_contact_gaps:
            from models.contact_geometry import contact_geometry_cs
            gaps, stage_jac = contact_geometry_cs(state[:10], reference[:10], phi, jac, **self.support_bounds)
            self.contact_geometry_fn = cs.Function('transport_contact_geometry',
                [state, reference, phi, jac], [gaps, stage_jac])
        self.stage_prediction_fn = cs.Function('mpc_stage_prediction',
            [state,cmd,phi,jac,mp,reference], [self.model.prediction_fn(state,cmd,gaps,stage_jac,mp)])

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

        self.first_action_bounds = None
        self.acados_solver_ = None
        self._acados_init_error = None
        self.acados_solve_count = 0
        self.acados_failure_count = 0
        self.acados_fallback_count = 0
        self.init_MPC()

    def reset(self):
        """Cold-start acados / IPOPT iterates between episodes."""
        self._reset_acados_solver()
        if hasattr(self, "nlp_w0_"):
            self.nlp_w0_ = cs.DM.zeros(self.nlp_w0_.shape)
        if hasattr(self, "nlp_lam_x0_"):
            self.nlp_lam_x0_ = cs.DM.zeros(self.nlp_lam_x0_.shape)
        if hasattr(self, "nlp_lam_g0_"):
            self.nlp_lam_g0_ = cs.DM.zeros(self.nlp_lam_g0_.shape)
        self.acados_solve_count = 0
        self.acados_failure_count = 0
        self.acados_fallback_count = 0

    def _cost_params(self, **kwargs):
        return pack_cost_params(self.cost_kind, self.source_param_, self.path_cost_fn, **kwargs)

    def _path_cost(self, state, command, params):
        size = self.path_cost_fn.size1_in(0) if hasattr(self.path_cost_fn, 'size1_in') else len(state)
        return self.path_cost_fn(state[:size], command, params)

    def _terminal_cost(self, state, params):
        size = self.final_cost_fn.size1_in(0) if hasattr(self.final_cost_fn, 'size1_in') else len(state)
        cost = self.final_cost_fn(state[:size], params)
        if getattr(self, 'velocity_state', False):
            # A low terminal pose cost while still rotating is not a stable
            # flip. Price residual motion, especially during final pose work.
            angular_weight = .002 + .02*params[-11]/max(float(self.source_param_.ori_coef), 1e-12)
            cost += .5*cs.sumsqr(state[10:13]) + angular_weight*cs.sumsqr(state[13:16])
        return cost

    def observation_state(self, pose, observation=None, fingertip_velocity=None):
        if not self.velocity_state:
            return np.asarray(pose, dtype=float)
        observation = observation or {}
        return self.model.observation_state(pose, observation.get('object_linear_velocity'),
            observation.get('object_angular_velocity'), fingertip_velocity)

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
        curr_x = np.asarray(curr_x, dtype=np.float64).reshape(-1)
        if self.velocity_state and curr_x.size == 10:
            curr_x = self.model.observation_state(curr_x, kwargs.pop('object_linear_velocity', None),
                kwargs.pop('object_angular_velocity', None), kwargs.pop('fingertip_velocity', None))
        curr_x = curr_x.reshape(self.param_.n_qpos_)
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
            contact_reference=curr_x[:7],
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
                return self._plan_torch(curr_x, phi_vec, jac_mat, cost_params, sol_guess, u_lb, u_ub)
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
            # An IPOPT iterate is not a valid acados iterate (and vice versa).
            # Drop it when the backend changes so a failed/slow solver cannot
            # turn stale multipliers into an oversized first action.
            result = self._plan_once_ipopt(
                curr_x, phi_vec, jac_mat, cost_params, None, u_lb, u_ub,
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
        result = dict(x0=x0)
        for key, default in (("lam_x0", self.nlp_lam_x0_), ("lam_g0", self.nlp_lam_g0_)):
            value = np.asarray(sol_guess.get(key, default), dtype=np.float64).reshape(-1)
            result[key] = value if value.size == int(default.numel()) else default
        return result

    def _plan_once_ipopt(self, curr_x, phi_vec, jac_mat, cost_params, sol_guess, u_lb, u_ub,
                         requested_solver="ipopt", fallback_reason=None):
        if self.ipopt_solver is None:
            self.ipopt_solver = cs.nlpsol("solver", "ipopt", *self._ipopt_definition)
        warm_start = self._prepare_sol_guess(sol_guess)
        if sol_guess is None and getattr(self.param_, 'runtime_object_cost_', False):
            # Zero quaternions in IPOPT's historical default iterate produce
            # NaN dynamics derivatives. Start the staged objective from a
            # feasible zero-action rollout, including normalized orientations.
            state = np.asarray(curr_x, dtype=float)
            rows = []
            for k in range(self.param_.mpc_horizon_):
                command = np.clip(np.zeros(self.param_.n_cmd_), u_lb, u_ub)
                if k == 0 and getattr(self, 'first_action_bounds', None) is not None:
                    command = np.clip(command, *self.first_action_bounds)
                state = np.asarray(self.stage_prediction_fn(state, command, phi_vec, jac_mat,
                    self.model.runtime_parameters(self.param_.model_params), curr_x)).reshape(-1)
                rows.append(np.r_[command, state])
            warm_start = dict(warm_start, x0=np.concatenate(rows))
        nlp_param = self.nlp_params_fn_(curr_x, phi_vec, jac_mat, cost_params, self.model.runtime_parameters(self.param_.model_params))
        lower_q, upper_q = self.param_.mpc_q_lb_.copy(), self.param_.mpc_q_ub_.copy()
        if self.recoverable_floor:
            lower_q[9], upper_q[9] = -1e7, 1e7
        nlp_lbw, nlp_ubw = self.nlp_bounds_fn_(u_lb, u_ub, lower_q, upper_q)
        if getattr(self, 'first_action_bounds', None) is not None:
            nlp_lbw[:self.param_.n_cmd_], nlp_ubw[:self.param_.n_cmd_] = self.first_action_bounds
        raw_sol = self.ipopt_solver(
            x0=warm_start["x0"],
            lam_x0=warm_start["lam_x0"],
            lam_g0=warm_start["lam_g0"],
            lbx=nlp_lbw,
            ubx=nlp_ubw,
            lbg=self.nlp_g_lb_,
            ubg=self.nlp_g_ub_,
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
        if sol_guess is None:
            # Replacing x/u does not clear acados' dual/QP memory. This also
            # covers re-entry after approach and objective/contact changes.
            self._reset_acados_solver()
        n, nu, nx = self.param_.mpc_horizon_, self.param_.n_cmd_, self.param_.n_qpos_
        stage_p = np.concatenate(
            [
                np.asarray(phi_vec, dtype=np.float64).reshape(-1),
                np.asarray(jac_mat, dtype=np.float64).reshape(-1, order="F"),
                np.asarray(cost_params, dtype=np.float64).reshape(-1),
                np.asarray(self.model.runtime_parameters(self.param_.model_params), dtype=np.float64).reshape(-1),
            ] + ([np.asarray(curr_x,dtype=np.float64)] if self.update_contact_gaps else [])
        )
        stage_geometry = [(phi_vec, jac_mat, curr_x)] * n
        stage_parameters = [stage_p] * n
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

        if self.update_contact_gaps:
            mp = self.model.runtime_parameters(self.param_.model_params)
            u_guess = np.clip(u_guess, u_lb, u_ub)
            if self.contact_force_limit is not None:
                norms = np.linalg.norm(u_guess @ np.asarray(self.param_.robot_stiff_).T, axis=1)
                u_guess *= np.minimum(1., self.contact_force_limit/np.maximum(norms, 1e-12))[:, None]
            if getattr(self, 'first_action_bounds', None) is not None:
                u_guess[0] = np.clip(u_guess[0], *self.first_action_bounds)
            x_guess[0] = curr_x
            for k in range(n):
                gaps, rows, reference = phi_vec, jac_mat, curr_x
                if k and self.support_vertices is not None:
                    from models.contact_geometry import refresh_table_support
                    gaps, rows = self.contact_geometry_fn(x_guess[k], curr_x, phi_vec, jac_mat)
                    gaps, rows = refresh_table_support(self.support_vertices, x_guess[k],
                        np.asarray(gaps).ravel(), np.asarray(rows),
                        float(self.source_param_.table_height),
                        float(getattr(self.param_, 'support_friction_', self.param_.mu_object_)),
                        **getattr(self, 'support_plane', {}))
                    reference = x_guess[k].copy()
                stage_geometry[k] = (gaps, rows, reference)
                stage_parameters[k] = np.r_[np.asarray(gaps).ravel(),
                    np.asarray(rows).reshape(-1, order='F'), cost_params,
                    np.asarray(mp).ravel(), reference]
                x_guess[k+1] = np.asarray(self.stage_prediction_fn(
                    x_guess[k], u_guess[k], gaps, rows, mp, reference)).reshape(-1)

        try:
            self.acados_solve_count += 1
            for k in range(n):
                self.acados_solver_.set(k, "x", x_guess[k])
                self.acados_solver_.set(k, "u", np.clip(u_guess[k], u_lb, u_ub))
                self.acados_solver_.set(k, "p", stage_parameters[k])
            self.acados_solver_.set(n, "x", x_guess[n])
            self.acados_solver_.set(n, "p", stage_parameters[-1])
            self.acados_solver_.set(0, "lbx", curr_x)
            self.acados_solver_.set(0, "ubx", curr_x)
            self.acados_solver_.constraints_set(0, "lbu", u_lb)
            self.acados_solver_.constraints_set(0, "ubu", u_ub)
            for k in range(n):
                first_bounds = getattr(self, 'first_action_bounds', None) if k == 0 else None
                self.acados_solver_.constraints_set(k, "lbu", u_lb if first_bounds is None else first_bounds[0])
                self.acados_solver_.constraints_set(k, "ubu", u_ub if first_bounds is None else first_bounds[1])
            from planning.acados_env import quiet_acados_stderr
            # One RTI update per observation. Mode changes are handled in the
            # dynamics and feasible warm start, not by nonlinear rollout gates.
            with quiet_acados_stderr():
                self.last_acados_rti_iterations = 1
                status = int(self.acados_solver_.solve())
            if status != 0:
                detail = ''
                try:
                    qp_status = int(np.asarray(self.acados_solver_.get_stats('qp_stat')).reshape(-1)[-1])
                    detail = ' QP=' + _format_acados_status(qp_status)
                except Exception:
                    pass
                raise RuntimeError(f"acados status {status}{detail}")
            u_traj = np.asarray([self.acados_solver_.get(k, "u") for k in range(n)], dtype=np.float64)
            u_traj = np.clip(u_traj, u_lb, u_ub)
            if getattr(self, 'first_action_bounds', None) is not None:
                u_traj[0] = np.clip(u_traj[0], *self.first_action_bounds)
            if self.contact_force_limit is not None:
                # A single RTI linearization of the force ball may overshoot.
                # Project commands directly, as in the Torch backend; do not
                # spend ten solves (or IPOPT) enforcing its numerical residual.
                norms = np.linalg.norm(u_traj @ np.asarray(self.param_.robot_stiff_).T, axis=1)
                u_traj *= np.minimum(1., self.contact_force_limit/np.maximum(norms, 1e-12))[:, None]
            x_traj = np.asarray([self.acados_solver_.get(k, "x") for k in range(n + 1)], dtype=np.float64)
            if self.update_contact_gaps:
                # Publish the prediction of the executed command and warm-start
                # from it next time. This rollout never rejects an RTI action.
                mp = self.model.runtime_parameters(self.param_.model_params)
                x_traj[0] = curr_x
                for k in range(n):
                    gaps, rows, reference = stage_geometry[k]
                    x_traj[k+1] = np.asarray(self.stage_prediction_fn(
                        x_traj[k], u_traj[k], gaps, rows, mp, reference)).reshape(-1)
            if not np.isfinite(u_traj).all() or not np.isfinite(x_traj).all():
                raise RuntimeError("acados returned non-finite values")
            try:
                sqp_iter = int(self.acados_solver_.get_stats("sqp_iter"))
            except Exception:
                sqp_iter = None
        except Exception as exc:
            self.last_acados_error = str(exc)
            self.acados_failure_count += 1
            return None

        self.last_acados_error = ""
        solve_status = _format_acados_status(0, sqp_iter=sqp_iter)
        stacked = np.column_stack([u_traj, x_traj[1:]]).reshape(-1)
        reported_cost = 0.0
        if getattr(self.param_, 'runtime_object_cost_', False):
            reported_cost = float(self._terminal_cost(x_traj[-1], cost_params))
            for k in range(n):
                state = x_traj[k + 1] if self.stage_cost_on_next else x_traj[k]
                reported_cost += float(self._path_cost(state, u_traj[k], cost_params))
        return dict(
            action=u_traj[0, :],
            u_traj=u_traj,
            rollout_q=x_traj[1:],
            sol_guess=dict(
                x0=stacked,
                u_traj=u_traj,
                x_traj=x_traj,
                lam_x0=np.zeros(n * (nu + nx)),
                lam_g0=np.zeros(int(self.nlp_lam_g0_.numel())),
                solver_backend="acados",
                solve_status=solve_status,
            ),
            cost_opt=np.asarray([reported_cost], dtype=np.float64),
            solve_status=solve_status,
            solver_backend="acados",
            requested_solver="acados",
        )

    def _plan_torch(self, curr_x, phi_vec, jac_mat, cost_params, sol_guess, u_lb=None, u_ub=None):
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
        ulb = torch.as_tensor(self.param_.mpc_u_lb_ if u_lb is None else u_lb, device=dev, dtype=dtype)
        uub = torch.as_tensor(self.param_.mpc_u_ub_ if u_ub is None else u_ub, device=dev, dtype=dtype)
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
        Qinv = torch.as_tensor(self.model.current_inverse(), device=dev, dtype=dtype)
        h = float(self.param_.h_)
        sigma = float(self.param_.model_params)
        target, quat = p[:3], p[3:7]
        contact_params = p[:-19] if getattr(self.param_, 'runtime_object_cost_', False) else p
        verify = contact_params[-7]
        virt, cp = contact_params[-6:-3], contact_params[-3:]
        from planning.mpc_costs import cpo_stage_weights
        center_w, ref_w, phased = cpo_stage_weights(
            getattr(self.param_, 'contact_cost_param', 0.0),
            getattr(self.param_, 'cpo_variant', 'full'))

        def closure():
            opt.zero_grad()
            q = q0
            J = torch.zeros((), device=dev)
            off = 0
            for _k in range(n):
                low, high = ulb, uub
                if _k == 0 and self.first_action_bounds is not None:
                    low, high = [torch.as_tensor(b,device=dev,dtype=dtype) for b in self.first_action_bounds]
                u = torch.clamp(z[off:off + nu], low, high)
                off += nu
                if self.contact_force_limit is not None:
                    stiff = torch.as_tensor(self.param_.robot_stiff_, device=dev, dtype=dtype)
                    cap = self.contact_force_limit
                    u = u * torch.clamp(cap / torch.linalg.vector_norm(stiff @ u).clamp_min(1e-12), max=1.)
                qnext_var = z[off:off + nq]
                off += nq
                stage_phi, stage_jac = phi, jac
                if self.update_contact_gaps:
                    from models.contact_geometry import contact_geometry_torch
                    stage_phi, stage_jac = contact_geometry_torch(q[:10], q0[:10], phi, jac, **self.support_bounds)
                if self.velocity_state:
                    qpred = self.model.prediction_torch(q,u,stage_phi,stage_jac)
                else:
                    b = torch.cat((
                        torch.as_tensor(self.param_.obj_mass_ * self.param_.gravity_, device=dev, dtype=dtype),
                        torch.as_tensor(self.param_.robot_stiff_, device=dev, dtype=dtype) @ u,
                    ))
                    smooth = self.model.force_velocity_scale * (Qinv @ b)
                    stage_phi, stage_jac = phi, jac
                    if self.update_contact_gaps:
                        from models.contact_geometry import contact_geometry_torch
                        stage_phi, stage_jac = contact_geometry_torch(q, q0, phi, jac, **self.support_bounds)
                    v = self.model.contact_velocity_torch(smooth, stage_jac, stage_phi, Qinv, sigma)
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
                    qpred = torch.cat((qpred[:3], qpred[3:7]/torch.linalg.vector_norm(qpred[3:7]).clamp_min(1e-8), qpred[7:]))
                if getattr(self.param_, 'runtime_object_cost_', False):
                    from planning.object_pose_cost import object_cost_torch
                    object_state = torch.clamp(qnext_var, qlb, qub) if self.stage_cost_on_next else q
                    J = J + object_cost_torch(object_state[:3], object_state[3:7], p[:7], p[-12:])
                q = torch.clamp(qnext_var, qlb, qub)
                stage_cp = cp
                if getattr(self.param_, 'runtime_object_cost_', False):
                    from models.contact_geometry import attached_point_torch
                    stage_cp = attached_point_torch(q, p[-19:-12], cp)
                center_term = torch.sum((q[:3] - q[7:10]) ** 2)
                ref_term = torch.log(torch.sum((q[7:10] - stage_cp) ** 2) + 1e-3)
                place = self.param_.contact_coef * (center_w * center_term + ref_w * ref_term)
                lift = (
                    self.param_.attract_coef * torch.log(torch.sum((q[7:10] - virt) ** 2) + 1e-3)
                    + self.param_.reject_coef * torch.where(
                        torch.sum((q[:2] - q[7:9]) ** 2) + 1e-3 < self.param_.reject_dis,
                        1 / (torch.sum((q[:2] - q[7:9]) ** 2) + 1e-3),
                        torch.zeros((), device=dev),
                    )
                )
                tracking = (1 - verify) * lift + verify * place if phased else place
                if self.param_.ablation_mode in ('a1', 'a2', 'a3'):
                    tracking = self.param_.contact_coef * (q[7:10] - stage_cp).square().sum()
                    if self.param_.ablation_mode == 'a3':
                        tracking += self.param_.ablation_collision_weight * self.param_.object_collision_cost_torch(q)
                J = J + 1e4 * torch.sum((q - qpred) ** 2) + tracking + 50 * torch.sum(u * u)
            if getattr(self.param_, 'runtime_object_cost_', False):
                from planning.object_pose_cost import object_cost_torch
                J = J + 10 * object_cost_torch(q[:3], q[3:7], p[:7], p[-12:])
                if self.velocity_state:
                    angular_weight = .002 + .02*p[-11]/max(float(self.source_param_.ori_coef),1e-12)
                    J = J + .5*q[10:13].square().sum()+angular_weight*q[13:16].square().sum()
            else:
                J = J + 10 * (500 * torch.sum((q[:3] - target) ** 2) + 5 * (1 - torch.dot(q[3:7], quat) ** 2))
            J.backward()
            return J

        opt.step(closure)
        out = z.detach().cpu().numpy()
        traj = out.reshape(n, -1)
        traj[:, :nu] = np.clip(traj[:, :nu], np.asarray(ulb.cpu()), np.asarray(uub.cpu()))
        if self.first_action_bounds is not None:
            traj[0,:nu] = np.clip(traj[0,:nu],*self.first_action_bounds)
        traj[:, nu:] = np.clip(traj[:, nu:], np.asarray(self.param_.mpc_q_lb_), np.asarray(self.param_.mpc_q_ub_))
        if self.contact_force_limit is not None:
            norms = np.linalg.norm(traj[:, :nu] @ np.asarray(self.param_.robot_stiff_).T, axis=1)
            traj[:, :nu] *= np.minimum(1., self.contact_force_limit / np.maximum(norms, 1e-12))[:, None]
        if not np.isfinite(traj).all():
            raise FloatingPointError("non-finite Torch MPC trajectory")
        out = traj.reshape(-1)
        final_cost = float(closure().detach().cpu())
        return dict(
            action=traj[0, :nu].copy(),
            u_traj=traj[:, :nu].copy(),
            rollout_q=traj[:, nu:].copy(),
            sol_guess=dict(x0=out, lam_x0=np.zeros_like(out), lam_g0=np.zeros(int(self.nlp_lam_g0_.numel())), opt_cost=final_cost),
            cost_opt=np.array([final_cost]),
            solve_status="torch-lbfgs",
            solver_backend="torch-lbfgs",
        )

    def _build_acados_solver(self):
        graph_type = cs.MX if getattr(self.param_, 'ablation_mode', 'full') == 'a3' else cs.SX
        AcadosModel, AcadosOcp, AcadosOcpSolver = _load_acados()
        n = int(self.param_.mpc_horizon_)
        nx, nu = int(self.param_.n_qpos_), int(self.param_.n_cmd_)
        phi = graph_type.sym("phi", self.param_.max_ncon_ * 4)
        jac = graph_type.sym("jac", self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        cp = graph_type.sym("cp", self.path_cost_fn.size_in(2))
        mp = graph_type.sym("model_param", self.model.model_parameter_dim)
        x = graph_type.sym("x", nx)
        u = graph_type.sym("u", nu)
        reference = graph_type.sym('contact_reference',nx) if self.update_contact_gaps else x
        p = cs.vvcat([phi,jac,cp,mp] + ([reference] if self.update_contact_gaps else []))
        pred = self.stage_prediction_fn(x,u,phi,jac,mp,reference)

        model = AcadosModel()
        cost_tag = "staged2" if getattr(self.param_, "runtime_object_cost_", False) else "detour4" if getattr(self.param_, "smooth_contact_detour", False) else "v1"
        import hashlib
        if getattr(self.param_, 'runtime_object_cost_', False):
            cost_key = hashlib.sha1((self.path_cost_fn.serialize() + self.final_cost_fn.serialize()).encode()).hexdigest()[:10]
            cost_tag += '_' + cost_key + ('_next' if self.stage_cost_on_next else '_current')
            if self.velocity_state:
                cost_tag += '_terminal_motion1_lm2'
        dynamics_fn = self.stage_prediction_fn if self.update_contact_gaps else self.model.prediction_fn
        dynamics_key = hashlib.sha1(dynamics_fn.serialize().encode()).hexdigest()[:10]
        contact_tag = getattr(self.model, 'contact_model', 'explicit').replace('-', '_')
        if getattr(self.param_, 'ablation_mode', 'full') != 'full':
            cost_tag += '_ablation_' + self.param_.ablation_mode + '_' + hashlib.sha1(
                self.path_cost_fn.serialize().encode()).hexdigest()[:10]
        model.name = f"mpc_{contact_tag}_v3_{dynamics_key}_{self.cost_kind}_{cost_tag}_h{n}_c{self.param_.max_ncon_}_x{nx}_u{nu}"
        self.recoverable_floor = bool(getattr(self, 'velocity_state', False))
        if self.recoverable_floor:
            model.name += '_floor_recovery_v1'
        if getattr(self, 'velocity_state', False) and self.update_contact_gaps and not self.stage_cost_on_next:
            # Keep one directional derivative subroutine, called once per
            # Jacobian column. Expanding all columns as SX produces ~15 MB
            # of C and spends ~30 s compiling on a first object load. This
            # computes the same analytic Jacobian without duplicating its
            # scalar instruction graph in the generated source.
            direction = graph_type.sym('dynamics_direction', nx + nu)
            dynamics = cs.Function('compact_dynamics', [x, u, p], [pred], {'cse': True})
            jvp = cs.Function('compact_dynamics_jvp', [x, u, p, direction],
                [cs.jtimes(pred, cs.vertcat(u, x), direction)], {'cse': True})
            cp_start, cp_size, cp_shape = phi.numel() + jac.numel(), cp.numel(), cp.shape
            x, u, p = cs.MX.sym('x', nx), cs.MX.sym('u', nu), cs.MX.sym('p', p.numel())
            cp = cs.reshape(p[cp_start:cp_start + cp_size], *cp_shape)
            pred = dynamics(x, u, p)
            directions = cs.DM.eye(nx + nu)
            model.disc_dyn_custom_jac_ux_expr = cs.horzcat(*[
                jvp(x, u, p, directions[:, i]) for i in range(nx + nu)])
            model.name += '_jvp_v1'
        if self.contact_force_limit is not None:
            # Version the force configuration so the cache cannot reuse a
            # solver whose initial stage lacks the nonlinear force bound.
            model.name += '_force0_v2_' + str(int(self.contact_force_limit*1e6))
            model.con_h_expr = cs.sumsqr(cs.DM(self.param_.robot_stiff_) @ u)
            model.con_h_expr_0 = model.con_h_expr
        model.x, model.u, model.p = x, u, p
        model.disc_dyn_expr = pred
        if self.stage_cost_on_next:
            model.cost_expr_ext_cost = self._path_cost(pred, u, cp)
        else:
            model.cost_expr_ext_cost = self._path_cost(x, u, cp)
        model.cost_expr_ext_cost_e = self._terminal_cost(x, cp)
        if self.update_contact_gaps:
            # RTI linearizes the contact dynamics. Keep the analytic objective
            # Hessian, but avoid second derivatives of the coupled unilateral
            # map: those grow enormously and are discontinuous at mode changes.
            model.cost_expr_ext_cost_0 = model.cost_expr_ext_cost
            model.cost_expr_ext_cost_custom_hess = cs.hessian(
                model.cost_expr_ext_cost, cs.vertcat(u, x))[0]
            model.cost_expr_ext_cost_custom_hess_0 = model.cost_expr_ext_cost_custom_hess
            model.cost_expr_ext_cost_custom_hess_e = cs.hessian(
                model.cost_expr_ext_cost_e, x)[0]

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros(int(p.size1()))
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        if self.contact_force_limit is not None:
            ocp.constraints.lh = np.array([0.])
            ocp.constraints.uh = np.array([self.contact_force_limit**2])
            ocp.constraints.lh_0 = ocp.constraints.lh.copy()
            ocp.constraints.uh_0 = ocp.constraints.uh.copy()
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
        if self.recoverable_floor:
            # An observed downward tip velocity cannot instantaneously obey
            # a future hard clearance bound under a bounded wrench. Penalize
            # temporary clearance violations; do not alter the measured x0,
            # force cap, or executor's floor clipping/braking.
            ocp.constraints.idxsbx = np.array([9], dtype=np.int64)
            ocp.constraints.idxsbx_e = np.array([9], dtype=np.int64)
            ocp.cost.Zl = ocp.cost.Zu = np.array([1e6])
            ocp.cost.zl = ocp.cost.zu = np.array([1e3])
            ocp.cost.Zl_e = ocp.cost.Zu_e = np.array([1e6])
            ocp.cost.zl_e = ocp.cost.zu_e = np.array([1e3])
            ocp.cost.Zl_0 = ocp.cost.Zu_0 = np.zeros(0)
            ocp.cost.zl_0 = ocp.cost.zu_0 = np.zeros(0)
        ocp.solver_options.N_horizon = n
        ocp.solver_options.tf = float(self.param_.h_ * n)
        profile = acados_solver_profile(self.cost_kind)
        for key, value in profile.items():
            setattr(ocp.solver_options, key, value)
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON" if self.update_contact_gaps else "EXACT"
        if getattr(self, 'velocity_state', False):
            ocp.solver_options.levenberg_marquardt = .01
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.tol = 1e-4
        if self.contact_force_limit is not None:
            # Keep the validated QP feasibility tolerance. Post-solve force
            # projection handles any remaining RTI force-ball overshoot.
            ocp.solver_options.qp_solver_tol_ineq = 1e-8
        ocp.solver_options.print_level = 0

        code_dir = os.path.join("/tmp", model.name + "_codegen")
        os.makedirs(code_dir, exist_ok=True)
        _set_acados_code_export_directory(ocp, code_dir)
        json_file = os.path.join(code_dir, model.name + ".json")
        shared = os.path.join(code_dir, "libacados_ocp_solver_" + model.name + ".so")
        from planning.acados_env import build_cached_solver
        return build_cached_solver(ocp, json_file, shared)

    def init_MPC(self):
        graph_type = cs.MX if getattr(self.param_, 'ablation_mode', 'full') == 'a3' else cs.SX
        model_params = graph_type.sym("model_param", self.model.model_parameter_dim)
        phi_vec = graph_type.sym("phi_vec", self.param_.max_ncon_ * 4)
        jac_mat = graph_type.sym("jac_mat", self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        cost_params = graph_type.sym("cost_params", self.path_cost_fn.size_in(2))
        lbu = graph_type.sym("lbu", self.param_.n_cmd_)
        ubu = graph_type.sym("ubu", self.param_.n_cmd_)
        lbq = graph_type.sym("lbq", self.param_.n_qpos_)
        ubq = graph_type.sym("ubq", self.param_.n_qpos_)

        w, w0, lbw, ubw, g = [], [], [], [], []
        lower_g, upper_g = [], []
        j = 0.0
        q0 = graph_type.sym("q", self.param_.n_qpos_)
        qk = q0
        for k in range(self.param_.mpc_horizon_):
            uk = graph_type.sym(f"u{k}", self.param_.n_cmd_)
            w += [uk]
            lbw += [lbu]
            ubw += [ubu]
            w0 += [cs.DM.zeros(self.param_.n_cmd_)]
            pred_q = self.stage_prediction_fn(qk, uk, phi_vec, jac_mat, model_params, q0)
            if self.stage_cost_on_next:
                j += self._path_cost(pred_q, uk, cost_params)
            else:
                j += self._path_cost(qk, uk, cost_params)
            qk = graph_type.sym(f"q{k + 1}", self.param_.n_qpos_)
            w += [qk]
            w0 += [cs.DM.zeros(self.param_.n_qpos_)]
            lbw += [lbq]
            ubw += [ubq]
            g += [pred_q - qk]
            if self.recoverable_floor:
                low = cs.fmax(float(self.param_.mpc_q_lb_[9]) - qk[9], 0.)
                high = cs.fmax(qk[9] - float(self.param_.mpc_q_ub_[9]), 0.)
                j += 1e3*(low+high) + .5e6*(low**2+high**2)
            lower_g.extend([0.] * self.param_.n_qpos_)
            upper_g.extend([0.] * self.param_.n_qpos_)
            if self.contact_force_limit is not None:
                g += [cs.sumsqr(cs.DM(self.param_.robot_stiff_) @ uk)]
                lower_g.append(0.)
                upper_g.append(self.contact_force_limit**2)
        j += self._terminal_cost(qk, cost_params)

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
        self._ipopt_definition = (nlp_prog, nlp_opts)
        self.ipopt_solver = (cs.nlpsol("solver", "ipopt", nlp_prog, nlp_opts)
                             if self.planner_solver_ == 'ipopt' else None)
        self.nlp_w0_ = cs.vcat(w0)
        self.nlp_lam_x0_ = cs.DM.zeros(self.nlp_w0_.shape)
        self.nlp_lam_g0_ = cs.DM.zeros(cs.vcat(g).shape)
        self.nlp_g_lb_, self.nlp_g_ub_ = np.array(lower_g), np.array(upper_g)
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
                # Initialization failure is a backend change.  Publish the
                # concrete backend and start with a cold IPOPT iterate rather
                # than carrying acados multipliers into the fallback.
                self.planner_solver_ = "ipopt"
                self.param_.planner_solver_ = "ipopt"
                self.param_.torch_solver = "ipopt"


class MPCExplicitAcados(MPCExplicit):
    """Compatibility alias used by the fingertip acados example."""


class MPCExplicitIsaac(MPCExplicit):
    def __init__(self, param):
        super().__init__(param, cost_kind="isaac")


class MPCExplicitTiltedPush(MPCExplicit):
    def __init__(self, param):
        super().__init__(param, cost_kind="tilted_push")


def _configure_explicit_layout(param, *, n_qpos, n_qvel, n_cmd, n_robot_qpos, cost_kind):
    param.n_qpos_ = int(n_qpos)
    param.n_qvel_ = int(n_qvel)
    param.n_cmd_ = int(n_cmd)
    param.n_robot_qpos_ = int(n_robot_qpos)
    param.mpc_model = "explicit"
    param.mpc_cost_kind = cost_kind
    if getattr(param, "planner_solver_", None) not in {"acados", "ipopt"}:
        param.planner_solver_ = "acados"
    q = np.asarray(getattr(param, "Q", np.zeros((0, 0))), dtype=np.float32)
    if q.shape != (param.n_qvel_, param.n_qvel_):
        inertia = np.asarray(getattr(param, "obj_inertia_", np.identity(6)), dtype=np.float32).reshape(6, 6)
        stiff = float(np.mean(np.diag(np.asarray(getattr(param, "robot_stiff_", np.eye(1))))))
        param.robot_stiff_ = np.diag(param.n_cmd_ * [stiff]).astype(np.float32)
        param.Q = np.zeros((param.n_qvel_, param.n_qvel_), dtype=np.float32)
        param.Q[:6, :6] = inertia
        param.Q[6:, 6:] = param.robot_stiff_
    cmd_limit = float(getattr(param, "planner_cmd_limit", 0.05))
    joint_limit = float(getattr(param, "planner_joint_delta_limit", 0.15))
    if param.n_cmd_ == 6:
        bound = np.full(6, cmd_limit, dtype=np.float32)
    elif param.n_cmd_ == 14:
        bound = np.full(14, joint_limit, dtype=np.float32)
    else:
        bound = np.full(param.n_cmd_, cmd_limit, dtype=np.float32)
    param.mpc_u_lb_ = -bound
    param.mpc_u_ub_ = bound


class MPCExplicitEE(MPCExplicit):
    """Dual-arm 3D fingertip-increment MPC (state 13, command 6)."""

    def __init__(self, param, **kwargs):
        _configure_explicit_layout(
            param, n_qpos=13, n_qvel=12, n_cmd=6, n_robot_qpos=6, cost_kind="bigrasp"
        )
        kwargs.setdefault("cost_kind", "bigrasp")
        super().__init__(param, **kwargs)


class MPCExplicitJoint(MPCExplicit):
    """Dual-arm 14-D joint-increment MPC (state obj+q, command Δq)."""

    def __init__(self, param, **kwargs):
        _configure_explicit_layout(
            param, n_qpos=21, n_qvel=20, n_cmd=14, n_robot_qpos=14, cost_kind="bigrasp_joint"
        )
        kwargs.setdefault("cost_kind", "bigrasp_joint")
        super().__init__(param, **kwargs)


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


def _build_mpc_planner_runtime(init, param=None):
    import argparse
    from planning.acados_env import ensure_acados_env
    ensure_acados_env()
    from examples.mpc.franka.ik2.params_ablation import ExplicitMPCParams
    from examples.mpc.fingertips.test.test_0902 import (
        ContactValueTracker,
        ModelCostConfidence,
        SmoothedApproachVia,
    )
    from examples.mpc.franka.ik2.test_mpc_isaac_ablation import (
        adapt_param_for_cartesian_solver, _configure_rollout_param,
    )

    args = argparse.Namespace(**init["args"])
    from planning.runtime_compat import resolve_solver_backend
    args.solver_backend = getattr(args, "solver_backend", "auto")
    args.solver = resolve_solver_backend(args.solver_backend)
    args.rollout = True
    trial_count = int(init["trial_count"])
    if param is None:
        param = ExplicitMPCParams(
            args,
            rand_seed=trial_count,
            target_type=getattr(args, "target_type", "ground-rotation"),
            mpc_model="explicit",
        )
    param = adapt_param_for_cartesian_solver(param, args)
    param = _configure_rollout_param(param, args)
    for key in (
        "target_p_", "target_q_", "mesh_path_", "table_height",
        "lambda_obj_mass_", "gravity_", "attract_coef", "contact_coef",
    ):
        if key in init:
            setattr(param, key, init[key])
    from planning.object_pose_cost import ObjectCostStages, target_up_axis
    # Payload may override the generated target; derive the axis afterwards.
    param.object_up_axis_ = target_up_axis(param.target_q_)
    stages = ObjectCostStages(args, getattr(param, 'object_up_axis_', (0., 0., 1.)))
    if stages.enabled:
        param.object_cost_config_ = stages.config()
    param.mpc_carry_velocity_ = bool(getattr(args, 'mpc_carry_velocity', True))
    param.mpc_tip_effective_mass_ = float(getattr(args, 'mpc_tip_effective_mass', 2.5))
    param.mpc_tip_damping_ = float(getattr(args, 'mpc_tip_damping', 40.))
    mpc = MPCExplicit(param)
    mpc_step = max(1e-4, float(getattr(args, "mpc_step_limit", 0.005)))
    via_step = getattr(args, "via_max_step", None)
    via_step = mpc_step if via_step is None else max(1e-4, float(via_step))
    via_lead = max(1e-4, float(getattr(args, "via_max_lead", via_step)))
    from planning.franka_contact_motion import FrankaApproachVia, SurfaceNormalApproachVia
    approach_class = (SurfaceNormalApproachVia if getattr(param, 'collision_mesh_paths_', [])
                      else FrankaApproachVia)
    approach_options = (dict(floor_z=float(param.table_height)+float(args.table_clearance))
                        if approach_class is SurfaceNormalApproachVia else {})
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
        "approach_via": approach_class(
            rate=float(getattr(args, "via_smooth_rate", 0.05)),
            max_step=via_step,
            max_lead=via_lead,
            **approach_options,
        ),
        "arrived_hold": False,
        "arrived_dest_idx": None,
        "sol_guess": None,
        "last_verify_cost": None,
    }
    trackers['object_cost_stages'] = stages
    if param.contact_model == 'soft-scm' and (getattr(args, 'online_scm', False) or getattr(args, 'scm_record_data', False) or getattr(args, 'scm_parameters', '')):
        from models.scm_online import OnlineSCMIdentifier
        from pathlib import Path
        import datetime
        out = getattr(args, 'scm_learning_dir', '')
        if not out:
            out = 'examples/mpc/franka/scm/outputs/online_scm_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log = Path(out) / (str(args.obj)+'_trial_%03d'%trial_count) / 'learning.jsonl'
        trackers['scm_identifier'] = OnlineSCMIdentifier(mpc.model,
            min_samples=int(getattr(args,'scm_learning_min_samples',32)),
            update_every=int(getattr(args,'scm_learning_update_every',32)), log_path=log)
        identifier = trackers['scm_identifier']
        checkpoint = getattr(args,'scm_parameters','')
        if checkpoint:
            # Mesh identity comes from the object-specific output directory;
            # numeric convention/configuration compatibility is checked in load.
            if Path(checkpoint).parent.name.split('_trial_')[0] != str(args.obj):
                raise ValueError('SCM checkpoint belongs to a different object')
            identifier.load(checkpoint)
            mpc.model.set_scm_parameters(identifier.theta)
            param.lambda_optimizer.set_scm_parameters(identifier.theta)
        print('[online-scm] enabled=%s data=%s' % (bool(getattr(args,'online_scm',False)), log.parent))
    return args, param, mpc, trackers


def _supported_call_kwargs(fn, kwargs):
    """Drop kwargs the live callee does not accept.

    Isaac / tilted-push payloads always carry support_point.  An older
    ``compute_rollout_contact_via`` on disk raises TypeError on that name
    before ranking starts.  New signatures still receive the plane.
    """
    import inspect
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


def _call_rollout_contact_via(fn, *args, **kwargs):
    return fn(*args, **_supported_call_kwargs(fn, kwargs))


def handle_mpc_request(args, param, mpc, trackers, msg):
    import time
    request_started = time.perf_counter()
    from examples.mpc.fingertips.test.test_0902 import (
        _verify_is_chatter,
    )
    from examples.mpc.franka.ik2.contact_policy_ablation import (
        compute_ablation_contact_policy as compute_rollout_contact_via,
    )
    from examples.mpc.franka.ik2.contact_frames import gravity_wrench_object_frame
    from planning.MPPIWarp import (
        _apply_dwell_payload,
        _opt_snapshot,
        _pickle_safe,
        clamp_via_to_tip,
    )
    from scipy.spatial.transform import Rotation

    online_started = time.perf_counter()
    identifier = trackers.get('scm_identifier')
    if identifier is not None:
        update = identifier.observe(msg.get('scm_transitions', []), learn=bool(getattr(args,'online_scm',False)))
        if update is not None and update['accepted']:
            mpc.model.set_scm_parameters(identifier.theta)
            param.lambda_optimizer.set_scm_parameters(identifier.theta)
        if update is not None:
            print('[online-scm] samples=%d accepted=%s loss=%.4g->%.4g time=%.2fms scales=%s' %
                  (update['samples'], update['accepted'], update['validation_loss_before'],
                   update['validation_loss_after'], update['update_ms'], update['scales']))

    online_dt = time.perf_counter() - online_started
    from planning.object_pose_cost import reset_objective_history
    stages = trackers.get('object_cost_stages')
    changed = stages.update(msg) if stages is not None else False
    if changed:
        reset_objective_history(param.lambda_optimizer, mpc, trackers)
        print('[object-cost] transition:', stages.diagnostics(), flush=True)
    if stages is not None and stages.enabled:
        # CPO keeps the staged terminal objective. Pose-only CSO ranks with
        # ell_pose; the other CSO modes keep ell_cso.
        param.object_cost_stage_ = stages.stage
        param.object_cost_config_ = stages.config()
        q = np.asarray(msg['policy_q'])
        r = Rotation.from_quat([q[4], q[5], q[6], q[3]]).as_matrix()
        from planning.object_pose_cost import pose_only_config
        cso_config = stages.config(r)
        if str(getattr(param.lambda_optimizer, 'cso_guidance', 'cso')) == 'pose':
            cso_config = pose_only_config(
                stages.pos_weight, stages.ori_weight, stages.body_axis,
                r.T @ np.array([0., 0., 1.]))
        param.lambda_optimizer.set_object_cost_config(cso_config)
        from planning.contact_limits import contact_command_force_limit
        param.lambda_optimizer.set_contact_force_limit(float(getattr(param, 'contact_command_force_limit_', contact_command_force_limit(args, 600.))))
    elif stages is not None:
        param.object_cost_stage_ = stages.stage
    elif getattr(param, 'object_cost_config_', None) is not None:
        # Ramp pushing has a fixed support-relative objective, without a
        # world-up flip/pose stage machine. Lambda works in the observed body.
        q = np.asarray(msg['policy_q'])
        r = Rotation.from_quat(q[[4, 5, 6, 3]]).as_matrix()
        config = np.asarray(param.object_cost_config_, dtype=float).copy()
        config[6:9] = r.T @ config[6:9]
        param.lambda_optimizer.set_object_cost_config(config)

    ablation_direct = param.ablation_mode in ('a1', 'a2', 'a3')
    ablation_no_rs = param.ablation_mode in ('a1', 'a2')
    dwell = None if (ablation_no_rs or msg.get("osc_upright_guard_active", False)) else msg.get("dwell")
    if msg.get("osc_upright_guard_active", False):
        trackers["sol_guess"] = None
    if (dwell is not None and not changed and
            (stages is None or not stages.enabled or dwell.get('object_cost_stage') == stages.stage)):
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
    if getattr(param.lambda_optimizer, 'shared_response', None) is not None:
        param.lambda_optimizer.rank_object_velocity = np.r_[
            r_obj_to_world.T @ np.asarray(msg.get('object_linear_velocity', np.zeros(3))),
            r_obj_to_world.T @ np.asarray(msg.get('object_angular_velocity', np.zeros(3)))]
        param.lambda_optimizer.rank_tip_velocity = r_obj_to_world.T @ np.asarray(
            msg.get('fingertip_linear_velocity', np.zeros(3)))
    param.lambda_env_phi_ = msg.get('env_phi')
    # PhysX reports speculative patches before their signed gap reaches zero.
    # Keep that measured geometry: the dynamics must decide whether a bounded
    # command closes the gap, rather than switching on the binary impulse flag.
    measured_contact = msg.get('actual_contact')
    measured_gap = float(msg.get('contact_distance', float('inf')))
    if measured_contact is not None:
        measured_gap = float(measured_contact.get('dist', measured_gap))
    reachable_contact = bool(msg.get('if_contact', False) or (
        measured_contact is not None and np.isfinite(measured_gap)
        and measured_gap <= float(np.max(np.abs(param.mpc_u_ub_)))))
    param.lambda_actual_contact_ = (measured_contact if
        ((stages is not None and stages.enabled) or getattr(param, 'runtime_object_cost_', False))
        and reachable_contact else None)
    # Finite-difference fingertip motion is in world coordinates. Object
    # angular velocity is transformed to body coordinates by the model.
    previous = trackers.get('motion_observation')
    now = float(msg.get('sim_time', 0.))
    tip_velocity = np.zeros(3)
    if previous is not None and now > previous[0]:
        tip_velocity = (curr_q[7:10]-previous[1])/(now-previous[0])
    if msg.get('fingertip_linear_velocity') is not None:
        tip_velocity = np.asarray(msg['fingertip_linear_velocity'], dtype=float)
    trackers['motion_observation'] = (now, curr_q[7:10].copy())
    planner_state = (mpc.observation_state(curr_q, msg, tip_velocity)
        if hasattr(mpc, 'observation_state') else curr_q)
    predictor = getattr(param, 'contact_transition_predictor_', None)
    if predictor is not None and hasattr(predictor, 'set_observation'):
        predictor.set_observation(planner_state,param.lambda_optimizer)
    t0 = time.perf_counter()
    policy = _call_rollout_contact_via(
        compute_rollout_contact_via,
        param, args, curr_q, r_obj_to_world, gravity, jac_mat_env,
        0.01, trackers["value_tracker"], trackers["model_cost_conf"],
        trackers["approach_via"], trackers["arrived_hold"], trackers["arrived_dest_idx"],
        floor_ground=table_ground,
        floor_z=floor_z,
        support_point=support_point,
        support_normal=support_normal,
    )
    rank_dt = time.perf_counter() - t0
    trackers["arrived_hold"] = policy["arrived_hold"]
    trackers["arrived_dest_idx"] = policy["arrived_dest_idx"]
    # Travel bypasses MPC, so it does not need to overwrite the critic EMA.
    # Preserve its continuous value across lift/drop and impulse flag changes.
    verify_cost = float((policy.get("value_info") or {}).get("verify", policy["verify_cost"]))
    policy["verify_cost"] = verify_cost
    verify_chatter = _verify_is_chatter(trackers["last_verify_cost"], verify_cost)
    trackers["last_verify_cost"] = float(verify_cost)

    tip = np.asarray(curr_q[7:10], dtype=np.float64)
    max_lead = max(1e-4, float(getattr(args, "via_max_lead", getattr(args, "via_max_step", getattr(args, "mpc_step_limit", 0.005)))))
    value_info = policy.get("value_info") or {}
    contact_id = getattr(param.lambda_optimizer, 'last_executed_idx', None)
    context = (contact_id, bool(reachable_contact),
               tuple(np.flatnonzero(np.linalg.norm(jac_mat.reshape(-1, 4, nv), axis=(1, 2)) > 1e-10)))
    previous_context = trackers.get('mpc_contact_context')
    if previous_context is not None and context != previous_context:
        trackers['sol_guess'] = None
        if hasattr(mpc, '_reset_acados_solver'):
            mpc._reset_acados_solver()
        trackers['mpc_contact_resets'] = trackers.get('mpc_contact_resets', 0) + 1
    trackers['mpc_contact_context'] = context
    if bool(value_info.get("path_blocked", False)) and not reachable_contact:
        trackers["sol_guess"] = None
    # A blocked orbit already walks via at max_step.  Clamping it back
    # to a 3 mm tip lead whenever the tip crosses the keep-out is what
    # made the rim hop.
    if not ablation_direct and not bool(value_info.get("path_blocked", False)):
        policy["mpc_virtual_point"] = clamp_via_to_tip(tip, policy["mpc_virtual_point"], max_lead)
        policy["mpc_contact_point"] = clamp_via_to_tip(tip, policy["mpc_contact_point"], max_lead)

    from planning.franka_contact_motion import bounded_step
    pose_stage = bool(stages is not None and stages.enabled and stages.stage == 'pose')
    flip_stage = bool(stages is not None and stages.enabled and stages.stage == 'flip')
    # OSC clips endpoint commands at the table in both stages. Predicting
    # only the finite-response fingertip state above the floor is insufficient:
    # a downward increment can still be replaced by an upward executor command.
    align_executor = bool(
        (pose_stage and getattr(args, 'pose_executor_constraints', False))
        or (flip_stage and getattr(args, 'flip_executor_constraints', True)))
    # Keep the developer Cartesian step (mpc_u_ub_, 5 mm). Dividing the
    # contact-force cap by K_task boxes every increment under 1 mm and the
    # arm crawls. The object cost and its force limit are unchanged.
    contact_bound = float(np.max(np.abs(getattr(param, "mpc_u_ub_", max_lead))))
    approach_action = (bool(value_info.get('switch_observed_contact', False))
        or (not reachable_contact and (bool(value_info.get('path_blocked', False))
            or (stages is not None and stages.enabled)
            or np.linalg.norm(tip - np.asarray(policy.get('p_arm_world', tip))) > .025)))
    if getattr(param, 'approach_requires_contact_', False) and not reachable_contact:
        # Without a reachable measured finger row the object objective cannot
        # drive an approach. Use the bounded via until contact geometry exists.
        approach_action = True
    if bool(getattr(param, 'cpo_direct_trajectory', False)):
        # Direct trajectory optimization solves the dynamics-constrained
        # problem. The geometric lift/place path is the phase potential
        # this variant removes.
        approach_action = False
    execution_mode = 'approach' if approach_action else 'contact'
    mpc.first_action_bounds = None
    trackers['mpc_slew_rebased'] = False
    if (getattr(mpc, 'velocity_state', False) and msg.get('executed_action') is not None
            and not approach_action):
        previous_command = np.clip(np.asarray(msg['executed_action'], dtype=float).reshape(3),
                                   param.mpc_u_lb_, param.mpc_u_ub_)
        force_cap = getattr(mpc, 'contact_force_limit', None)
        if force_cap is not None:
            force = np.linalg.norm(np.asarray(param.robot_stiff_) @ previous_command)
            if force > force_cap:
                # A free-space command can exceed the contact cap. Centering
                # the slew box on it can make its intersection with the force
                # ball empty. Rebase on a feasible command at contact entry.
                previous_command *= force_cap / force
                trackers['mpc_slew_rebased'] = True
        slew = float(getattr(param, 'isaac_action_slew_', .0025))/np.sqrt(3.)
        mpc.first_action_bounds = (np.maximum(param.mpc_u_lb_, previous_command-slew),
                                   np.minimum(param.mpc_u_ub_, previous_command+slew))
    command_lower = getattr(param, 'mpc_u_lb_', -contact_bound)
    command_upper = getattr(param, 'mpc_u_ub_', contact_bound)
    trackers['mpc_executor_edge'] = False
    if (align_executor
            and getattr(mpc, 'velocity_state', False) and not approach_action):
        from planning.execution_constraints import table_edge_command_bounds
        from examples.mpc.franka.ik2.test_mpc_isaac_ablation import (
            _table_edge_gap_and_normal, ISAAC_TABLE_EDGE_GUARD_DISTANCE)
        gap, outward = _table_edge_gap_and_normal(curr_q, param)
        guard_active = bool(msg.get('if_contact', False) or (
            not value_info.get('path_blocked', False)
            and np.linalg.norm(tip - policy['p_arm_world']) <= .015))
        command_lower, command_upper, edge_applied = table_edge_command_bounds(
            command_lower, command_upper, gap, outward, ISAAC_TABLE_EDGE_GUARD_DISTANCE, guard_active)
        trackers['mpc_executor_edge'] = edge_applied
    trackers['mpc_executor_floor'] = dict(applied=False, rebased=False, emergency=False)
    if (align_executor
            and getattr(mpc, 'velocity_state', False) and not approach_action):
        from planning.execution_constraints import floor_command_bounds
        force_cap = getattr(mpc, 'contact_force_limit', None)
        radius = (None if force_cap is None else force_cap / np.linalg.norm(param.robot_stiff_, 2))
        previous_command = msg.get('executed_action')
        bounds, floor_info = floor_command_bounds(tip[2], param.mpc_q_lb_[9],
            command_lower, command_upper, first_bounds=mpc.first_action_bounds,
            previous=previous_command,
            slew=float(getattr(param, 'isaac_action_slew_', .0025))/np.sqrt(3.), max_norm=radius)
        mpc.first_action_bounds = bounds
        trackers['mpc_executor_floor'] = floor_info
    t1 = time.perf_counter()
    if approach_action:
        # The geometric approach replaces MPC's command. Solving that discarded
        # horizon can trigger a slow IPOPT fallback at contact-mode changes.
        action = bounded_step(tip, policy['mpc_virtual_point'], max_lead)
        sol = dict(sol_guess=None, cost_opt=None, solver_backend='approach')
    else:
        if (not ablation_no_rs and trackers["sol_guess"] is None and getattr(param, 'runtime_object_cost_', False)
                and param.lambda_actual_contact_ is not None and value_info.get('accept_p_arm', False)):
            # Start on the measured Lambda solution's active contact branch.
            # MPC still optimizes the same objective and all command constraints.
            inward = -r_obj_to_world.T @ np.asarray(param.lambda_actual_contact_['normal_world'], dtype=float)
            inward /= max(np.linalg.norm(inward), 1e-12)
            ref = np.array([0., 0., 1.]) if abs(inward[2]) < .9 else np.array([0., 1., 0.])
            tangent = np.cross(inward, ref); tangent /= np.linalg.norm(tangent)
            frame = np.column_stack([inward, tangent, np.cross(inward, tangent)])
            force = np.asarray(policy['info']['control_input'], dtype=float).reshape(3)
            command = np.linalg.solve(param.robot_stiff_, r_obj_to_world @ frame @ force)
            state = np.asarray(planner_state, dtype=float)
            states, rows = [state], []
            for _ in range(param.mpc_horizon_):
                state = np.asarray(mpc.stage_prediction_fn(state, command, phi_vec, jac_mat,
                    mpc.model.runtime_parameters(param.model_params), planner_state)).reshape(-1)
                states.append(state); rows.append(np.r_[command, state])
            trackers['sol_guess'] = dict(x0=np.concatenate(rows),
                u_traj=np.tile(command, (param.mpc_horizon_, 1)), x_traj=np.array(states))
        sol = mpc.plan_once(
            param.target_p_,
            param.target_q_,
            planner_state,
            phi_vec,
            jac_mat,
            verify_cost_param=(1.0 if ablation_direct else (
                float(not value_info.get('path_blocked', False))
                if getattr(param, 'runtime_object_cost_', False) and reachable_contact else verify_cost)),
            virtual_point=policy["mpc_virtual_point"],
            contact_point=policy["mpc_contact_point"],
            sol_guess=trackers["sol_guess"],
            **(dict(u_lb=command_lower, u_ub=command_upper)
               if trackers['mpc_executor_edge'] else {}),
        )
        action = np.asarray(sol['action'], dtype=float).reshape(3)
    plan_dt = time.perf_counter() - t1
    trackers["sol_guess"] = sol["sol_guess"]
    if execution_mode != 'contact' or msg.get('osc_upright_guard_active', False):
        trackers['sol_guess'] = None
        policy['x_plus_opt'] = None
        # Candidate force predictions no longer describe the executed action.
        # Do not feed their gains into confidence/dwell on the next frame.
        policy['execution_override'] = True
    if getattr(param, 'runtime_object_cost_', False):
        # Confidence must assess the action MPC actually sent, rather than
        # Lambda's separately optimized wrench at a hypothetical mesh sample.
        predicted = np.asarray(mpc.stage_prediction_fn(planner_state, action, phi_vec, jac_mat,
            mpc.model.runtime_parameters(param.model_params), planner_state)).reshape(-1)
        policy['execution_object_pose_world'] = predicted[:7]
        if stages is not None and stages.enabled:
            from planning.object_pose_cost import object_cost_np
            target = np.r_[param.target_p_, param.target_q_]
            policy['pred_dcost'] = object_cost_np(curr_q[:7], target, stages.config()) - object_cost_np(predicted[:7], target, stages.config())
            policy['pred_dcost_prev'] = None if dwell is None else dwell.get('pred_reduction')
            policy['act_dcost_prev'] = (None if dwell is None else float(dwell['c_now_cost'])-float(dwell['c_after']))
            policy['object_cost_now'] = object_cost_np(curr_q[:7], target, stages.config())
    policy['preserve_upright'] = False
    policy['release_pressure'] = False
    policy['execution_mode'] = execution_mode
    if getattr(args, 'rollout_trace_jsonl', '') and getattr(param, 'runtime_object_cost_', False):
        from planning.contact_prediction_diagnostics import contact_prediction_diagnostics
        policy['contact_prediction_diagnostics'] = contact_prediction_diagnostics(
            param, mpc, curr_q, policy, sol)
    policy['execution_path_blocked'] = bool(value_info.get('path_blocked', False) and approach_action)
    policy["choose_dt"] = rank_dt
    # #region agent log
    try:
        import json as _json
        import time as _time
        _payload = {
            "sessionId": "252d5a",
            "hypothesisId": "B",
            "location": "mpc_explicit.py:handle_mpc_request",
            "message": "plan_action",
            "timestamp": int(_time.time() * 1000),
            "data": {
                "mode": execution_mode,
                "action_norm_mm": round(float(np.linalg.norm(action)) * 1000.0, 3),
                "contact_bound_mm": round(float(contact_bound) * 1000.0, 3),
                "max_lead_mm": round(float(max_lead) * 1000.0, 3),
                "stages_enabled": bool(stages is not None and stages.enabled),
                "stage": None if stages is None else stages.stage,
                "pose_stage": bool(pose_stage),
                "if_contact": bool(msg.get("if_contact", False)),
                "path_blocked": bool(value_info.get("path_blocked", False)),
                "plan_dt_ms": round(float(plan_dt) * 1000.0, 1),
                "rank_dt_ms": round(float(rank_dt) * 1000.0, 1),
                "release": bool(policy.get("release_pressure", False)),
                "preserve": bool(policy.get("preserve_upright", False)),
            },
        }
        with open("/home/zz/scsp-robot/.cursor/debug-252d5a.log", "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_payload) + "\n")
    except Exception:
        pass
    # #endregion
    if stages is not None:
        policy['object_cost_stage'] = stages.stage
        policy['object_cost_config'] = stages.config() if stages.enabled else None
        policy['object_progress_scale'] = stages.progress_scale() if stages.enabled else None
        policy['object_feedback_scale'] = max(.05, float(getattr(stages, 'ori_weight', 20.))*.025) if stages.enabled else None
    elif getattr(param, 'object_cost_config_', None) is not None:
        policy['object_cost_config'] = np.asarray(param.object_cost_config_).copy()
    return {
        "object_cost_diagnostics": dict({} if stages is None else stages.diagnostics(),
            pressure_releases=0, upright_action_rejections=0),
        "seq": None if msg.get("seq") is None else int(msg["seq"]),
        "action": np.asarray(action, dtype=np.float64).reshape(3),
        "sol_guess": sol["sol_guess"],
        "cost_opt": sol.get("cost_opt"),
        "policy": _pickle_safe(policy),
        "verify_cost": float(verify_cost),
        "verify_chatter": bool(verify_chatter),
        "if_contact": bool(msg.get("if_contact", False)),
        "opt_snapshot": _opt_snapshot(param.lambda_optimizer),
        "scm_identification": None if identifier is None else identifier.diagnostics(),
        "model_tightness": float(trackers["model_cost_conf"].tightness()),
        "model_accum": float(trackers["model_cost_conf"].accum),
        "rank_dt": float(rank_dt),
        "online_dt": float(online_dt),
        "total_dt": float(time.perf_counter() - request_started),
        "mpc_skipped": bool(approach_action),
        "mpc_executor_floor": dict(trackers['mpc_executor_floor']),
        "mpc_executor_edge": bool(trackers['mpc_executor_edge']),
        "mpc_fallbacks": int(getattr(mpc, "acados_fallback_count", 0)),
        "mpc_contact_resets": int(trackers.get('mpc_contact_resets', 0)),
        "mpc_slew_rebased": bool(trackers.get('mpc_slew_rebased', False)),
        "mpc_floor_recovery_m": max(0., float(param.mpc_q_lb_[9]) - float(np.min(sol['rollout_q'][:, 9])))
            if getattr(mpc, 'recoverable_floor', False) and sol.get('rollout_q') is not None else 0.,
        "mpc_last_error": str(getattr(mpc, "last_acados_error", "") or ""),
        "mpc_rti_iterations": 0 if approach_action else int(getattr(mpc, "last_acados_rti_iterations", 0)),
        "plan_dt": float(plan_dt),
        "mppi_dt": float(plan_dt),
        "solver_backend": str(sol.get("solver_backend", getattr(mpc, "planner_solver_", "ipopt"))),
        "lambda_failures": int(getattr(param.lambda_optimizer, "acados_failure_count", 0)),
        "mpc_failures": int(getattr(mpc, "acados_failure_count", 0)),
        "lambda_solves": int(getattr(param.lambda_optimizer, "acados_solve_count", 0)),
        "mpc_solves": int(getattr(mpc, "acados_solve_count", 0)),
        "mpc_init_error": str(getattr(mpc, "_acados_init_error", "") or ""),
    }


def mpc_planner_worker(state_q, action_q, ready_q, init):
    """Independent planner loop.  Must not import isaacgym."""
    import traceback
    import time
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
            action = handle_mpc_request(args, param, mpc, trackers, msg)
            # Both processes use the same monotonic clock.  Carrying this
            # timestamp lets Isaac report queue/planner latency without using
            # wall-clock time or restarting the active policy interval.
            action["submitted_at"] = time.monotonic()
            action["planner_dt"] = float(action.get("total_dt", action.get("plan_dt", action.get("mppi_dt", 0.0))))
            put_latest(action_q, action)
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
