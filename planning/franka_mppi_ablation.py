"""A7 MPPI: sample Cartesian contact trajectories with the existing MPC model.

The CasADi rollout reuses MPC's contact transport, dynamics and cost functions.
No NLP solver or CSO contact-selection result enters the sampling update.
"""
import tempfile
import time

import casadi as cs
import numpy as np

from planning.mpc_explicit_ablation import MPCExplicit


class FrankaCenterMPPI(MPCExplicit):
    def __init__(self, param, args, model=None):
        self.samples = int(args.a7_mppi_samples)
        self.iterations = int(args.a7_mppi_iterations)
        self.temperature = float(args.a7_mppi_temperature)
        self.noise = float(args.a7_mppi_noise)
        if (self.samples < 2 or self.iterations < 1
                or not np.isfinite([self.temperature, self.noise]).all()
                or min(self.temperature, self.noise) <= 0):
            raise ValueError('A7 requires samples >= 2, iterations >= 1 and positive temperature/noise')
        param.mpc_horizon_ = int(args.a7_mppi_horizon)
        if param.mpc_horizon_ < 1:
            raise ValueError('A7 horizon must be positive')
        self.rng = np.random.default_rng(int(param.trial_seed_))
        self.mppi_solve_count = 0
        super().__init__(param, model=model)

    def init_MPC(self):
        """Compile forward rollouts only; deliberately do not construct an NLP."""
        self.planner_solver_ = self.param_.planner_solver_ = 'mppi'
        self.stage_cost_on_next = True
        self._jit_directory = tempfile.TemporaryDirectory(prefix='franka_a7_mppi_')
        self.horizon = self.param_.mpc_horizon_
        nx, nu = self.param_.n_qpos_, self.param_.n_cmd_
        x0 = cs.MX.sym('x0', nx)
        controls = cs.MX.sym('controls', nu, self.horizon)
        phi = cs.MX.sym('phi', 4*self.param_.max_ncon_)
        jac = cs.MX.sym('jac', 4*self.param_.max_ncon_, self.param_.n_qvel_)
        mp = cs.MX.sym('mp', self.model.model_parameter_dim)
        cp = cs.MX.sym('cp', self.path_cost_fn.size1_in(2))
        x, cost, states = x0, 0, []
        for k in range(self.horizon):
            x = self.stage_prediction_fn(x, controls[:, k], phi, jac, mp, x0)
            states.append(x)
            cost += self._path_cost(x, controls[:, k], cp)
            # Match the existing MPC's recoverable floor constraint penalty.
            low = cs.fmax(float(self.param_.mpc_q_lb_[9])-x[9], 0)
            high = cs.fmax(x[9]-float(self.param_.mpc_q_ub_[9]), 0)
            cost += 1e3*(low+high) + .5e6*(low**2+high**2)
        # A7's task objective has only center attraction and ell_cso; retain
        # the same terminal ell_cso weight, without adding a motion objective.
        cost += self.final_cost_fn(x[:self.final_cost_fn.size1_in(0)], cp)
        self.rollout_fn = cs.Function('a7_contact_rollout', [controls, x0, phi, jac, mp, cp],
            [cost, cs.horzcat(*states)], dict(jit=True, jit_options=dict(
                flags='-O3', directory=self._jit_directory.name+'/')))
        self.batch_fn = self.rollout_fn.map(self.samples)
        self.last_effective_samples = None

    def _project(self, controls, lower, upper):
        controls = np.clip(controls, lower, upper)
        if self.first_action_bounds is not None:
            controls[..., 0, :] = np.clip(controls[..., 0, :], *self.first_action_bounds)
        if self.contact_force_limit is not None:
            force = np.linalg.norm(controls @ np.asarray(self.param_.robot_stiff_).T, axis=-1)
            controls *= np.minimum(1., self.contact_force_limit/np.maximum(force, 1e-12))[..., None]
        return controls

    def __del__(self):
        # Release CasADi's compiled function before removing its temporary files.
        self.batch_fn = None
        self.rollout_fn = None

    def plan_once(self, target_p=None, target_q=None, curr_x=None, phi_vec=None,
                  jac_mat=None, verify_cost_param=0, virtual_point=None,
                  contact_point=None, sol_guess=None, **kwargs):
        started = time.perf_counter()
        state = np.asarray(curr_x, dtype=float).reshape(-1)
        if self.velocity_state and state.size == 10:
            state = self.observation_state(state)
        phi, jac = np.asarray(phi_vec), np.asarray(jac_mat)
        lower = np.asarray(kwargs.pop('u_lb', self.param_.mpc_u_lb_))
        upper = np.asarray(kwargs.pop('u_ub', self.param_.mpc_u_ub_))
        cp = self._cost_params(target_p=target_p, target_q=target_q, phi_vec=phi,
            jac_mat=jac, verify_cost_param=verify_cost_param,
            virtual_point=virtual_point, contact_point=contact_point,
            contact_reference=state[:7], **kwargs)
        fixed = (state, phi, jac, self.model.runtime_parameters(self.param_.model_params), cp)
        mean = np.zeros((self.horizon, self.param_.n_cmd_))
        if sol_guess is not None and sol_guess.get('u_traj') is not None:
            previous = np.asarray(sol_guess['u_traj'])
            mean = np.vstack((previous[1:], previous[-1:]))
        mean = self._project(mean, lower, upper)
        for _ in range(self.iterations):
            noise = self.rng.normal(0, self.noise, (self.samples, *mean.shape))
            noise[0] = 0
            candidates = self._project(mean[None, :, :] + noise, lower, upper)
            costs, _states = self.batch_fn(candidates.transpose(2, 0, 1).reshape(
                self.param_.n_cmd_, -1), *fixed)
            costs = np.asarray(costs).reshape(-1)
            # Gaussian proposal likelihood correction for the shifted controls.
            correction = self.temperature*np.sum(
                mean[None, :, :]*(candidates-mean)/self.noise**2, axis=(1, 2))
            scores = costs+correction
            finite = np.isfinite(scores)
            if not finite.any():
                raise FloatingPointError('All A7 MPPI rollout costs are nonfinite')
            weights = np.zeros(self.samples)
            weights[finite] = np.exp(-(scores[finite]-scores[finite].min())/self.temperature)
            weights /= weights.sum()
            self.last_effective_samples = float(1/np.sum(weights**2))
            mean = self._project(np.einsum('s,shd->hd', weights, candidates), lower, upper)
        cost, trajectory = self.rollout_fn(mean.T, *fixed)
        trajectory = np.asarray(trajectory).T
        if not np.isfinite(trajectory).all() or not np.isfinite(float(cost)):
            raise FloatingPointError('Nonfinite A7 MPPI mean trajectory')
        self.mppi_solve_count += 1
        return dict(action=mean[0].copy(), u_traj=mean.copy(), rollout_q=trajectory,
            sol_guess=dict(u_traj=mean.copy()), cost_opt=np.array([float(cost)]),
            solver_backend='mppi', solve_time=time.perf_counter()-started,
            effective_samples=self.last_effective_samples)
