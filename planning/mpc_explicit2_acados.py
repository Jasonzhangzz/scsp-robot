"""acados backend for the single-fingertip explicit MPC example."""
import os
import sys
import ctypes
import numpy as np
import casadi as cs

from planning.mpc_explicit2 import MPCExplicit as _IPOPT_MPCExplicit


def _load_acados():
    """Load the repository's acados_template without requiring installation."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.environ.get("ACADOS_SOURCE_DIR", os.path.join(repo, "..", "thirdparty", "acados"))
    interface = os.path.join(root, "interfaces", "acados_template")
    if interface not in sys.path:
        sys.path.insert(0, interface)
    os.environ.setdefault("ACADOS_SOURCE_DIR", root)
    lib = os.path.join(root, "lib")
    if os.path.isdir(lib):
        old = os.environ.get("LD_LIBRARY_PATH", "").split(":") if os.environ.get("LD_LIBRARY_PATH") else []
        if lib not in old:
            os.environ["LD_LIBRARY_PATH"] = ":".join([lib] + old)
        # The dynamic loader reads LD_LIBRARY_PATH at process start.  Preload
        # acados dependencies explicitly so generated solvers can find HPIPM,
        # BLASFEO and qpOASES in an already-running Python process.
        for name in ("libblasfeo.so.0", "libhpipm.so", "libqpOASES_e.so", "libacados.so"):
            path = os.path.join(lib, name)
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                except OSError:
                    pass
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    return AcadosModel, AcadosOcp, AcadosOcpSolver


class MPCExplicitAcados(_IPOPT_MPCExplicit):
    """Single-contact MPC using acados by default, with IPOPT fallback."""

    def __init__(self, param):
        # Parent builds the exact CasADi objective and IPOPT fallback.
        super().__init__(param)
        self.acados_solver_ = None
        self._acados_init_error = None
        self.acados_solve_count = 0
        self.acados_failure_count = 0
        self.acados_fallback_count = 0
        try:
            self.acados_solver_ = self._build_acados_solver()
        except Exception as exc:
            self._acados_init_error = exc
            print(f"acados initialization failed; IPOPT fallback remains available: {exc}")

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

        model = AcadosModel()
        # Bump the generated-solver name so cached /tmp binaries do not reuse
        # the old point-attractor objective.
        # Include the objective coefficient in the cache key.  The
        # ideal-best diagnostic scales virtual-point tracking; reusing a
        # solver generated for the old coefficient silently discards that
        # change and makes tracking appear ineffective.
        tracking_key = int(round(float(self.param_.attract_coef) * 100.0))
        field_key = int(round(float(getattr(self.param_, 'field_cost_weight', 0.05)) * 100.0))
        quad_key = int(bool(getattr(self.param_, 'quadratic_contact_track', False)))
        step_key = int(round(abs(float(np.asarray(self.param_.mpc_u_ub_).reshape(-1)[0])) * 1000.0))
        model.name = (f"fingertip_mpc_explicit2_v12_h{n}_c{self.param_.max_ncon_}"
                      f"_a{tracking_key}_f{field_key}_q{quad_key}_u{step_key}")
        model.x, model.u, model.p = x, u, p
        model.disc_dyn_expr = self.model.step_once_fn(x, u, phi, jac, mp)
        model.cost_expr_ext_cost = self.path_cost_fn(x, u, cp)
        model.cost_expr_ext_cost_e = self.final_cost_fn(x, cp)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros(int(p.size1()))
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.constraints.idxbu = np.arange(nu, dtype=np.int64)
        def _bound(value, size):
            arr = np.asarray(value, dtype=float).reshape(-1)
            return np.full(size, arr.item()) if arr.size == 1 else arr.reshape(size)
        ocp.constraints.lbu = _bound(self.param_.mpc_u_lb_, nu)
        ocp.constraints.ubu = _bound(self.param_.mpc_u_ub_, nu)
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
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "EXACT"
        ocp.solver_options.integrator_type = "DISCRETE"
        # RTI solves one condensed QP per call and is the intended low-latency
        # mode for this receding-horizon loop.  IPOPT remains the fallback for
        # the occasional failed QP.
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.regularize_method = "PROJECT"
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.print_level = 0
        code_dir = os.path.join("/tmp", model.name + "_codegen")
        os.makedirs(code_dir, exist_ok=True)
        ocp.code_gen_opts.code_export_directory = code_dir
        json_file = os.path.join(code_dir, model.name + ".json")
        shared = os.path.join(code_dir, "libacados_ocp_solver_" + model.name + ".so")
        if os.path.isfile(json_file) and os.path.isfile(shared):
            return AcadosOcpSolver(ocp, json_file=json_file, generate=False, build=False,
                                   check_reuse_possible=False, verbose=False)
        return AcadosOcpSolver(ocp, json_file=json_file, generate=True, build=True,
                               check_reuse_possible=True, verbose=False)

    def _cost_params(self, target_p, target_q, phi_vec, jac_mat, verify, virtual, contact, ori=None):
        # Match ExplicitMPCParams.init_cost_fns from the reference script;
        # ``ori`` is retained only for call-site compatibility.
        value = cs.vvcat([target_p, target_q, phi_vec, jac_mat, verify, virtual, contact])
        expected = int(np.prod(self.path_cost_fn.size_in(2)))
        actual = int(np.prod(value.shape))
        if actual > expected:
            value = value[:expected]
        elif actual < expected:
            value = cs.vertcat(value, cs.DM.zeros(expected - actual, 1))
        return np.asarray(value, dtype=float).reshape(-1)

    def plan_once(self, target_p, target_q, curr_x, phi_vec, jac_mat,
                  verify_cost_param, virtual_point, contact_point, curr_ori_coef=None,
                  sol_guess=None):
        if self.acados_solver_ is None:
            self.param_.torch_solver = "ipopt"
            return super().plan_once(target_p, target_q, curr_x, phi_vec, jac_mat,
                                     verify_cost_param, virtual_point, contact_point,
                                     curr_ori_coef, sol_guess)
        cost = self._cost_params(target_p, target_q, phi_vec, jac_mat,
                                 verify_cost_param, virtual_point, contact_point, curr_ori_coef)
        phi = np.asarray(phi_vec, dtype=float).reshape(-1)
        jac = np.asarray(jac_mat, dtype=float)
        stage_p = np.concatenate([phi, jac.reshape(-1, order="F"), cost,
                                  [float(self.param_.model_params)]])
        curr = np.asarray(curr_x, dtype=float).reshape(-1)
        n, nu, nx = self.param_.mpc_horizon_, self.param_.n_cmd_, self.param_.n_qpos_
        # Use a measured-state warm start; this avoids the invalid all-zero quaternion.
        for k in range(n):
            self.acados_solver_.set(k, "x", curr)
            self.acados_solver_.set(k, "u", np.zeros(nu))
            self.acados_solver_.set(k, "p", stage_p)
        self.acados_solver_.set(n, "x", curr)
        self.acados_solver_.set(n, "p", stage_p)
        self.acados_solver_.set(0, "lbx", curr)
        self.acados_solver_.set(0, "ubx", curr)
        try:
            self.acados_solve_count += 1
            status = int(self.acados_solver_.solve())
            if status != 0:
                raise RuntimeError(f"acados status {status}")
            u_traj = np.asarray([self.acados_solver_.get(k, "u") for k in range(n)])
            x_traj = np.asarray([self.acados_solver_.get(k, "x") for k in range(n + 1)])
            if not np.isfinite(u_traj).all() or not np.isfinite(x_traj).all():
                raise RuntimeError("acados returned non-finite values")
        except Exception as exc:
            self.acados_failure_count += 1
            self.acados_fallback_count += 1
            self.param_.torch_solver = "ipopt"
            result = super().plan_once(target_p, target_q, curr_x, phi_vec, jac_mat,
                                       verify_cost_param, virtual_point, contact_point,
                                       curr_ori_coef, sol_guess)
            result["fallback_reason"] = str(exc)
            return result
        action = np.clip(u_traj[0], self.param_.mpc_u_lb_, self.param_.mpc_u_ub_)
        return dict(action=action, rollout_q=x_traj[1:],
                    sol_guess=dict(x0=np.concatenate([np.column_stack([u_traj, x_traj[1:]]).reshape(-1)]),
                                   lam_x0=np.zeros(n * (nu + nx)),
                                   lam_g0=np.zeros(n * nx), solver_backend="acados",
                                   solve_status="ACADOS_SUCCESS"),
                    cost_opt=np.asarray([0.0]), solve_status="ACADOS_SUCCESS",
                    solver_backend="acados", requested_solver="acados")
