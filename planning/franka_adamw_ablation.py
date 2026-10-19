"""A6 direct shooting: ComFree forward costs, numerical gradients and AdamW.

CasADi is used solely as a forward evaluator (including contact transport).
No Jacobian, adjoint, SQP, NLP solver or Lambda contact solution is used.
"""
import time

import casadi as cs
import numpy as np

from models.explicit_model import ExplicitModel
from models.contact_geometry import contact_geometry_cs, rotation_cs
from planning.object_pose_cost import cost_config, extended_cost_config, object_cost_cs


def build_a6_cost_fns(param):
    """Only target pose and task weights enter A6's cost parameter vector."""
    x, u, p = cs.SX.sym('state', 10), cs.SX.sym('control', 3), cs.SX.sym('task', 19)
    center = x[:3] + rotation_cs(x[3:7]) @ cs.DM(param.object_geometry_center_local_)
    task = object_cost_cs(x[:7], p[:7], p[7:])
    path = (task + 50*cs.sumsqr(u)
            + param.contact_coef*param.a6_center_weight*cs.sumsqr(x[7:10]-center))
    return (cs.Function('a6_path_cost', [x, u, p], [path]),
            cs.Function('a6_terminal_cost', [x, p], [10*task]))


class FrankaComFreeAdamW:
    def __init__(self, param, args):
        self.iterations = int(args.a6_adamw_iterations)
        self.learning_rate = float(args.a6_adamw_lr)
        self.weight_decay = float(args.a6_adamw_weight_decay)
        self.fd_epsilon = float(args.a6_fd_epsilon)
        if (self.iterations < 1 or not np.isfinite(
                [self.learning_rate, self.weight_decay, self.fd_epsilon]).all()
                or self.learning_rate <= 0 or self.weight_decay < 0
                or self.fd_epsilon <= 0 or self.learning_rate*self.weight_decay >= 1):
            raise ValueError('Invalid A6 AdamW settings')
        if param.contact_model != 'comfree':
            raise ValueError('A6 requires contact_model=comfree')
        self.param_ = param
        for name, size in (('mpc_u_lb_', 3), ('mpc_u_ub_', 3),
                           ('mpc_q_lb_', 10), ('mpc_q_ub_', 10)):
            setattr(param, name, np.broadcast_to(np.asarray(getattr(param, name), dtype=float), (size,)).copy())
        self.model = ExplicitModel(param)
        self.path_cost_fn, self.final_cost_fn = build_a6_cost_fns(param)
        self.first_action_bounds = None
        self.contact_force_limit = getattr(param, 'contact_command_force_limit_', None)
        self.support_bounds = {}
        if hasattr(param, 'support_surface_half_extents_'):
            self.support_bounds = dict(support_point=param.support_surface_point_,
                support_tangents=param.support_surface_tangents_,
                support_half_extents=param.support_surface_half_extents_)
        self.adamw_solve_count = 0
        self.init_MPC()

    def init_MPC(self):
        """Construct only a forward shooting function and its batched map."""
        self.planner_solver_ = self.param_.planner_solver_ = 'adamw'
        self.horizon = int(self.param_.mpc_horizon_)
        nx, nu = self.param_.n_qpos_, self.param_.n_cmd_
        if self.horizon < 1 or nu != 3 or nx != 10:
            raise ValueError('A6 requires a positive horizon and Cartesian 10-state/3-control layout')
        x = cs.SX.sym('state', nx)
        ref = cs.SX.sym('reference', nx)
        command = cs.SX.sym('command', nu)
        gaps = cs.SX.sym('gaps', 4*self.param_.max_ncon_)
        rows = cs.SX.sym('rows', 4*self.param_.max_ncon_, self.param_.n_qvel_)
        parameters = cs.SX.sym('parameters', self.model.model_parameter_dim)
        moved_gaps, moved_rows = contact_geometry_cs(x, ref, gaps, rows, **self.support_bounds)
        self.stage_prediction_fn = cs.Function('a6_stage_prediction',
            [x, command, gaps, rows, parameters, ref],
            [self.model.prediction_fn(x, command, moved_gaps, moved_rows, parameters)])
        x0 = cs.MX.sym('x0', nx)
        controls = cs.MX.sym('controls', nu, self.horizon)
        phi = cs.MX.sym('phi', 4*self.param_.max_ncon_)
        jac = cs.MX.sym('jac', 4*self.param_.max_ncon_, self.param_.n_qvel_)
        mp = cs.MX.sym('mp', self.model.model_parameter_dim)
        cp = cs.MX.sym('cp', self.path_cost_fn.size1_in(2))
        state, cost, states = x0, 0, []
        for k in range(self.horizon):
            state = self.stage_prediction_fn(state, controls[:, k], phi, jac, mp, x0)
            states.append(state)
            cost += self.path_cost_fn(state, controls[:, k], cp)
            # Direct shooting replaces the NLP's state inequalities with the
            # same L1 + quadratic recovery penalty used for the table floor.
            low = cs.fmax(cs.DM(self.param_.mpc_q_lb_) - state, 0)
            high = cs.fmax(state - cs.DM(self.param_.mpc_q_ub_), 0)
            cost += 1e3*cs.sum1(low+high) + .5e6*(cs.sumsqr(low)+cs.sumsqr(high))
        cost += self.final_cost_fn(state, cp)
        inputs = [controls, x0, phi, jac, mp, cp]
        self.rollout_fn = cs.Function('a6_comfree_rollout', inputs,
            [cost, cs.horzcat(*states)])
        # Cost-only probes avoid materializing all perturbed state trajectories.
        cost_fn = cs.Function('a6_comfree_cost', inputs, [cost])
        self.batch_cost_fn = cost_fn.map(2*self.horizon*nu)

    def _project(self, controls, lower, upper):
        """Project onto command/slew boxes intersected with the force ball."""
        lo = np.broadcast_to(lower, controls.shape).copy()
        hi = np.broadcast_to(upper, controls.shape).copy()
        if self.first_action_bounds is not None:
            lo[0] = np.maximum(lo[0], self.first_action_bounds[0])
            hi[0] = np.minimum(hi[0], self.first_action_bounds[1])
        if np.any(lo > hi):
            raise ValueError('A6 command bounds have an empty intersection')
        projected = np.clip(controls, lo, hi)
        stiffness = np.asarray(self.param_.robot_stiff_, dtype=float)
        diagonal = np.diag(stiffness)
        if not np.allclose(stiffness, np.diag(diagonal)):
            raise ValueError('A6 force projection requires diagonal Cartesian stiffness')
        # The OSC clips increment norm as well as its components. Enforce it
        # inside the same projection so the executed first action matches.
        step = float(np.max(np.abs(self.param_.mpc_u_ub_)))
        diagonal = (np.full(3, 1/step) if self.contact_force_limit is None else
                    np.maximum(np.abs(diagonal)/float(self.contact_force_limit), 1/step))
        cap = 1.
        force = lambda value: np.linalg.norm(value*diagonal, axis=-1)
        if np.any(force(np.clip(np.zeros_like(controls), lo, hi)) > cap+1e-10):
            raise ValueError('A6 command box and contact force cap are infeasible')
        # KKT solution clip(u/(1+alpha*K^2), lo, hi). Unlike radial
        # clipping, this also respects a nonzero lower bound after floor/slew.
        active = force(projected) > cap
        if np.any(active):
            left, right = np.zeros(self.horizon), np.ones(self.horizon)
            def at(alpha):
                return np.clip(controls/(1+alpha[:, None]*diagonal**2), lo, hi)
            for _ in range(64):
                outside = force(at(right)) > cap
                if not np.any(outside & active):
                    break
                right[outside & active] *= 2
            for _ in range(60):
                mid = .5*(left+right)
                outside = force(at(mid)) > cap
                left = np.where(outside, mid, left)
                right = np.where(outside, right, mid)
            projected[active] = at(right)[active]
        return projected

    def _numerical_gradient(self, controls, fixed):
        """Central differences of the complete horizon, including contact motion.

        Probe the smooth extension outside the command box; project AdamW's
        updated iterate afterwards. Clipping probes would halve derivatives
        at a bound and silently change the shooting objective.
        """
        size = controls.size
        offsets = np.eye(size).reshape(size, *controls.shape)*self.fd_epsilon
        probes = np.concatenate((controls+offsets, controls-offsets), axis=0)
        packed = probes.transpose(2, 0, 1).reshape(controls.shape[1], -1)
        values = np.asarray(self.batch_cost_fn(packed, *fixed)).reshape(-1)
        return ((values[:size]-values[size:])/(2*self.fd_epsilon)).reshape(controls.shape)

    def plan_once(self, target_p, target_q, curr_x, phi_vec, jac_mat,
                  sol_guess=None, u_lb=None, u_ub=None):
        started = time.perf_counter()
        state = np.asarray(curr_x, dtype=float).reshape(self.param_.n_qpos_)
        phi, jac = np.asarray(phi_vec), np.asarray(jac_mat)
        lower = np.asarray(self.param_.mpc_u_lb_ if u_lb is None else u_lb, dtype=float)
        upper = np.asarray(self.param_.mpc_u_ub_ if u_ub is None else u_ub, dtype=float)
        config = getattr(self.param_, 'object_cost_config_',
                         cost_config(self.param_.pos_coef, self.param_.ori_coef))
        cp = np.r_[np.asarray(target_p).reshape(3), np.asarray(target_q).reshape(4),
                   extended_cost_config(config)]
        fixed = (state, phi, jac, self.model.runtime_parameters(self.param_.model_params), cp)
        controls = np.zeros((self.horizon, self.param_.n_cmd_))
        if sol_guess is not None and sol_guess.get('u_traj') is not None:
            previous = np.asarray(sol_guess['u_traj'], dtype=float).reshape(controls.shape)
            if np.isfinite(previous).all():
                controls = np.vstack((previous[1:], previous[-1:]))
        controls = self._project(controls, lower, upper)
        # Balanced coordinates make lr=.1 mean 10% of a command bound. Moment
        # estimates start fresh for each measured-state MPC problem.
        scale = np.maximum(np.maximum(np.abs(self.param_.mpc_u_lb_),
                                      np.abs(self.param_.mpc_u_ub_)), 1e-12)
        moment, variance = np.zeros_like(controls), np.zeros_like(controls)
        best, best_cost, evaluations, updates = None, np.inf, 0, 0
        initial_cost = None
        for iteration in range(self.iterations+1):
            cost, trajectory = self.rollout_fn(controls.T, *fixed)
            cost, trajectory = float(cost), np.asarray(trajectory).T
            evaluations += 1
            if initial_cost is None:
                initial_cost = cost
            if np.isfinite(cost) and np.isfinite(trajectory).all() and cost < best_cost:
                best, best_cost = (controls.copy(), trajectory.copy()), cost
            if iteration == self.iterations or not np.isfinite(cost):
                break
            gradient = self._numerical_gradient(controls, fixed)*scale
            evaluations += 2*controls.size
            if not np.isfinite(gradient).all():
                break
            updates += 1
            moment = .9*moment + .1*gradient
            variance = .999*variance + .001*gradient**2
            corrected_m = moment/(1-.9**updates)
            corrected_v = variance/(1-.999**updates)
            normalized = controls/scale
            normalized *= 1-self.learning_rate*self.weight_decay
            normalized -= self.learning_rate*corrected_m/(np.sqrt(corrected_v)+1e-8)
            controls = self._project(normalized*scale, lower, upper)
        if best is None:
            raise FloatingPointError('No finite A6 ComFree rollout')
        controls, trajectory = best
        self.adamw_solve_count += 1
        return dict(action=controls[0].copy(), u_traj=controls.copy(), rollout_q=trajectory,
            sol_guess=dict(u_traj=controls.copy()), cost_opt=np.array([best_cost]),
            initial_cost=initial_cost, solver_backend='adamw',
            solve_status='adamw' if updates == self.iterations else 'adamw-nonfinite-probe',
            gradient_method='central_difference', iterations=updates,
            rollout_evaluations=evaluations, solve_time=time.perf_counter()-started)
