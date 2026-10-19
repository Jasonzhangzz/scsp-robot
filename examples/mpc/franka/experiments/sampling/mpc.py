"""Contact-implicit MPC with acados replacing Drake's SNOPT NLP.

The decision variables are the fingertip force and the normal contact force.
The gap is the nonlinear point-contact gap, so the contact direction follows
the fingertip instead of a frozen dairlib linearization. Complementarity is a
smooth penalty: penetration is penalized, and contact force is penalized while
the gap is open. acados SQP is the solver.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from examples.mpc.franka.experiments.sampling.lcs import EE, EE_V, NX, OBJ, OBJ_V


class ContactMPC:
    def __init__(self, horizon=5, dt=0.05, ee_mass=0.4, object_mass=0.2,
                 radius=0.05, force_limit=50.0, complementarity_weight=20.0,
                 penetration_weight=200.0):
        self.horizon = int(horizon)
        self.dt = float(dt)
        self.ee_mass = float(ee_mass)
        self.object_mass = float(object_mass)
        self.radius = float(radius)
        self.force_limit = float(force_limit)
        self.complementarity_weight = float(complementarity_weight)
        self.penetration_weight = float(penetration_weight)
        self._solver = None

    def solve(self, x0, x_desired, **_ignored):
        solver = self._solver_instance()
        x0 = np.asarray(x0, dtype=float).reshape(NX)
        desired = np.asarray(x_desired, dtype=float).reshape(-1)
        if desired.size != NX:
            desired = np.asarray(x_desired[0], dtype=float).reshape(NX)
        parameter = desired
        for knot in range(self.horizon):
            solver.set(knot, "p", parameter)
            solver.set(knot, "x", x0)
            solver.set(knot, "u", np.zeros(4))
        solver.set(self.horizon, "p", parameter)
        solver.set(self.horizon, "x", x0)
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        status = int(solver.solve())
        states = np.zeros((self.horizon + 1, NX))
        controls = np.zeros((self.horizon, 4))
        for knot in range(self.horizon):
            states[knot] = np.asarray(solver.get(knot, "x"), dtype=float).reshape(NX)
            controls[knot] = np.asarray(solver.get(knot, "u"), dtype=float).reshape(4)
        states[-1] = np.asarray(solver.get(self.horizon, "x"), dtype=float).reshape(NX)
        if status not in (0, 2) or not np.isfinite(states).all() or not np.isfinite(controls).all():
            raise RuntimeError("acados contact MPC failed with status %d" % status)
        return {
            "x": states,
            "u": controls[:, :3],
            "lambda": controls[:, 3:4],
            "status": status,
            "solver": "acados",
        }

    def _solver_instance(self):
        if self._solver is not None:
            return self._solver
        import casadi as cs
        from planning.acados_env import build_cached_solver, ensure_acados_env

        ensure_acados_env()
        from acados_template import AcadosModel, AcadosOcp

        x = cs.SX.sym("x", NX)
        u = cs.SX.sym("u", 4)
        p = cs.SX.sym("x_desired", NX)
        nxt, gap = _discrete(
            x, u, self.dt, self.ee_mass, self.object_mass, self.radius, cs
        )
        # Gauss-Newton residuals. The last two are the smooth stand-in for
        # SNOPT's complementarity constraints.
        path = cs.vertcat(
            _weighted_state(x - p),
            np.sqrt(0.05) * u[:3],
            np.sqrt(0.01) * u[3],
            np.sqrt(self.penetration_weight) * cs.fmin(gap, 0),
            np.sqrt(self.complementarity_weight) * u[3] * cs.fmax(gap, 0),
        )
        terminal_gap = _opening(x, self.radius, cs)
        terminal = cs.vertcat(
            _weighted_state(x - p),
            np.sqrt(self.penetration_weight) * cs.fmin(terminal_gap, 0),
        )
        model = AcadosModel()
        model.name = "sampling_contact_mpc_h%d_dt%d" % (
            self.horizon, int(round(self.dt * 10000))
        )
        model.x = x
        model.u = u
        model.p = p
        model.disc_dyn_expr = nxt
        model.cost_y_expr = path
        model.cost_y_expr_e = terminal

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros(NX)
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        path_dim = int(path.numel())
        terminal_dim = int(terminal.numel())
        ocp.cost.W = np.eye(path_dim)
        ocp.cost.W_e = np.eye(terminal_dim)
        ocp.cost.yref = np.zeros(path_dim)
        ocp.cost.yref_e = np.zeros(terminal_dim)
        ocp.constraints.idxbu = np.arange(4)
        ocp.constraints.lbu = np.array(
            [-self.force_limit, -self.force_limit, -self.force_limit, 0.0]
        )
        ocp.constraints.ubu = np.full(4, self.force_limit)
        ocp.constraints.x0 = np.zeros(NX)
        ocp.solver_options.N_horizon = self.horizon
        ocp.solver_options.tf = self.horizon * self.dt
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        ocp.solver_options.nlp_solver_max_iter = 40
        ocp.solver_options.qp_solver_iter_max = 200
        ocp.solver_options.tol = 1e-4
        ocp.solver_options.print_level = 0
        if hasattr(ocp.solver_options, "regularize_method"):
            ocp.solver_options.regularize_method = "PROJECT"

        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        directory = cache / "scsp-robot" / "acados" / model.name
        directory.mkdir(parents=True, exist_ok=True)
        _export_directory(ocp, str(directory))
        json_file = directory / (model.name + ".json")
        shared = directory / ("libacados_ocp_solver_" + model.name + ".so")
        self._solver = build_cached_solver(ocp, json_file, shared)
        return self._solver


def _weighted_state(error):
    import casadi as cs

    weights = np.ones(NX)
    weights[EE] = np.sqrt(0.5)
    weights[OBJ] = np.sqrt(200.0)
    weights[EE_V] = 1.0
    weights[OBJ_V] = 1.0
    return cs.diag(weights) @ error


def _opening(state, radius, cs):
    delta = state[EE] - state[OBJ]
    normal = delta / cs.fmax(cs.norm_2(delta), 1e-4)
    return cs.dot(normal, delta) - radius


def _discrete(state, control, dt, ee_mass, object_mass, radius, cs):
    normal = (state[EE] - state[OBJ]) / cs.fmax(cs.norm_2(state[EE] - state[OBJ]), 1e-4)
    force = control[:3]
    contact = control[3]
    ee_velocity = state[EE_V] + dt * (force + contact * normal) / ee_mass
    object_velocity = state[OBJ_V] - dt * contact * normal / object_mass
    ee_position = state[EE] + dt * ee_velocity
    object_position = state[OBJ] + dt * object_velocity
    nxt = cs.vertcat(ee_position, object_position, ee_velocity, object_velocity)
    gap = cs.dot(normal, ee_position - object_position) - radius
    return nxt, gap


def _export_directory(ocp, directory):
    options = getattr(ocp, "code_gen_opts", None)
    if options is not None and hasattr(options, "code_export_directory"):
        options.code_export_directory = directory
        return
    if hasattr(ocp, "code_export_directory"):
        ocp.code_export_directory = directory
        return
    raise AttributeError("this acados version has no code export directory setting")
