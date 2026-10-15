import time
from typing import Dict, Optional

import casadi as cs
import numpy as np


def _as_flat_float64(value, expected_size=None, name="value"):
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if expected_size is not None and arr.size != expected_size:
        raise ValueError(
            f"{name} must have size {expected_size}, got {arr.size}."
        )
    return arr


def _reshape_cols(vec, rows, cols):
    return np.asarray(vec, dtype=np.float64).reshape((rows, cols), order="F")


class CITOIPOPT:
    def __init__(self, param):
        self.param_ = param
        (
            self.total_cost_fn,
            self.constraint_fn,
            self.objective_and_constraints_fn,
        ) = self.param_.init_cost_fns()
        self.rollout_fn = self.param_.init_model_fns()[-1]
        self._init_solver()

    def _init_solver(self):
        q_traj_flat = cs.SX.sym("q_traj_flat", self.param_.n_decision_)
        cost_param = cs.SX.sym("cost_param", self.param_.cost_param_dim_)
        contact_schedule = cs.SX.sym(
            "contact_schedule", self.param_.contact_schedule_dim_
        )

        total_cost, constraint = self.objective_and_constraints_fn(
            q_traj_flat, cost_param, contact_schedule
        )
        nlp_param = cs.vcat([cost_param, contact_schedule])

        nlp_prog = {
            "f": total_cost,
            "x": q_traj_flat,
            "g": constraint,
            "p": nlp_param,
        }
        nlp_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": int(self.param_.ipopt_max_iter_),
            "ipopt.tol": 1e-4,
            "ipopt.linear_solver": "mumps",
        }
        self.ipopt_solver = cs.nlpsol("cito_q_traj_solver", "ipopt", nlp_prog, nlp_opts)

        self.nlp_param_dim_ = int(
            self.param_.cost_param_dim_ + self.param_.contact_schedule_dim_
        )
        self.nlp_lam_x0_ = cs.DM.zeros((self.param_.n_decision_, 1))
        self.nlp_lam_g0_ = cs.DM.zeros((self.param_.num_equality_constraints_, 1))

    def _default_sol_guess(self):
        return {
            "x0": self.param_.default_initial_guess(),
            "lam_x0": self.nlp_lam_x0_,
            "lam_g0": self.nlp_lam_g0_,
        }

    def _shift_solution_guess(self, q_traj_opt):
        q_traj_opt = np.asarray(q_traj_opt, dtype=np.float64)
        shifted = np.zeros_like(q_traj_opt)
        shifted[:, 0] = self.param_.q_init_
        if q_traj_opt.shape[1] > 2:
            shifted[:, 1:-1] = q_traj_opt[:, 2:]
        shifted[:, -1] = q_traj_opt[:, -1]
        return shifted.reshape(-1, order="F")

    def _normalize_sol_guess(self, sol_guess):
        if sol_guess is None:
            return self._default_sol_guess()

        x0 = sol_guess.get("x0", self.param_.default_initial_guess())
        lam_x0 = sol_guess.get("lam_x0", self.nlp_lam_x0_)
        lam_g0 = sol_guess.get("lam_g0", self.nlp_lam_g0_)
        return {
            "x0": _as_flat_float64(x0, self.param_.n_decision_, name="x0"),
            "lam_x0": lam_x0,
            "lam_g0": lam_g0,
        }

    def _extract_rollout(self, q_traj_flat, contact_schedule):
        q_mat, v_mat, a_mat, tau_mat, wrench_mat, h = self.rollout_fn(
            q_traj_flat, contact_schedule
        )
        q_traj = np.asarray(q_mat, dtype=np.float64)
        v_traj = np.asarray(v_mat, dtype=np.float64)
        a_raw = np.asarray(a_mat, dtype=np.float64)
        a_traj = np.zeros((self.param_.n_qvel_, self.param_.mpc_horizon_ + 1), dtype=np.float64)
        if a_raw.size:
            a_traj[:, 1:] = (
                a_raw
                if a_raw.shape[1] == self.param_.mpc_horizon_
                else a_raw[:, : self.param_.mpc_horizon_]
            )
        return {
            "q": q_traj,
            "v": v_traj,
            "a": a_traj,
            "tau": np.asarray(tau_mat, dtype=np.float64),
            "wrench": np.asarray(wrench_mat, dtype=np.float64),
            "h": np.asarray(h, dtype=np.float64).reshape(-1),
        }

    def plan_once(
        self,
        cost_param,
        contact_schedule,
        sol_guess: Optional[Dict[str, np.ndarray]] = None,
    ):
        cost_param = _as_flat_float64(
            cost_param, self.param_.cost_param_dim_, name="cost_param"
        )
        contact_schedule = _as_flat_float64(
            contact_schedule,
            self.param_.contact_schedule_dim_,
            name="contact_schedule",
        )
        guess = self._normalize_sol_guess(sol_guess)
        nlp_param = np.hstack([cost_param, contact_schedule])

        solve_start = time.time()
        raw_sol = self.ipopt_solver(
            x0=guess["x0"],
            lam_x0=guess["lam_x0"],
            lam_g0=guess["lam_g0"],
            lbx=self.param_.decision_lb_,
            ubx=self.param_.decision_ub_,
            lbg=np.zeros((self.param_.num_equality_constraints_,), dtype=np.float64),
            ubg=np.zeros((self.param_.num_equality_constraints_,), dtype=np.float64),
            p=nlp_param,
        )
        solve_time = time.time() - solve_start

        q_traj_flat_opt = np.asarray(raw_sol["x"].full(), dtype=np.float64).reshape(-1)
        q_traj_opt = _reshape_cols(
            q_traj_flat_opt, self.param_.n_qpos_, self.param_.mpc_horizon_ + 1
        )
        rollout = self._extract_rollout(q_traj_flat_opt, contact_schedule)
        q_traj = rollout["q"]
        v_traj = rollout["v"]
        a_traj = rollout["a"]

        q_robot_curr = self.param_.q_init_[self.param_.robot_q_slice].copy()
        q_robot_target = q_traj[self.param_.robot_q_slice, 1].copy()
        dq_robot_target = q_robot_target - q_robot_curr

        q_guess_shifted = self._shift_solution_guess(q_traj_opt)
        solve_status = self.ipopt_solver.stats()["return_status"]
        return {
            "q_traj": q_traj,
            "v_traj": v_traj,
            "a_traj": a_traj,
            "q_traj_flat": q_traj_flat_opt.copy(),
            "rollout_q": q_traj.T.copy(),
            "rollout": rollout,
            "q_target": q_traj[:, 1].copy(),
            "v_target": v_traj[:, 1].copy(),
            "a_target": a_traj[:, 1].copy(),
            "robot_q_target": q_robot_target,
            "robot_dq_target": dq_robot_target,
            "cost_opt": float(raw_sol["f"].full().item()),
            "solve_time": float(solve_time),
            "solve_status": solve_status,
            "sol_guess": {
                "x0": q_guess_shifted,
                "lam_x0": raw_sol["lam_x"],
                "lam_g0": raw_sol["lam_g"],
                "opt_cost": float(raw_sol["f"].full().item()),
            },
        }

    def plan_from_isaac(
        self,
        simulator,
        sol_guess: Optional[Dict[str, np.ndarray]] = None,
        repeat_contact_schedule=True,
    ):
        observation = self.param_.build_isaac_inputs(
            simulator,
            sync_initial_state=True,
            repeat_contact_schedule=repeat_contact_schedule,
        )
        solution = self.plan_once(
            observation["cost_param"],
            observation["contact_schedule"],
            sol_guess=sol_guess,
        )
        solution["observation"] = observation
        return solution


MPCSolver = CITOIPOPT
