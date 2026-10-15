"""Goal-cost online MPC using the same acados/IPOPT solver as Isaac MPC.

Lambda supplies contact anchors and optional force targets, never a trajectory.
Reported x_plus_opt is the re-prediction of the bounded action sent to OSC.
"""

import copy
import time

import casadi as cs
import numpy as np

from models.bimanual_rigid_contact import BimanualRigidContactModel
from models.contact_geometry import rotation_cs
from planning.mpc_explicit import MPCExplicit


class BimanualOnlineMPC(MPCExplicit):
    def __init__(self, param):
        param = copy.copy(param)
        physical = ("h_", "planner_cmd_limit", "online_task_stiffness", "contact_command_force_limit_",
                    "mpc_tip_effective_mass_", "mpc_tip_damping_", "mpc_patch_radius_", "tip_radius",
                    "online_contact_offset")
        if any(not np.isfinite(float(getattr(param, name))) or float(getattr(param, name)) <= 0 for name in physical):
            raise ValueError("Online MPC physical parameters must be finite and positive")
        param.n_qpos_, param.n_qvel_, param.n_cmd_ = 25, 12, 6
        param.n_robot_qpos_ = 6
        param.max_ncon_ = 6  # two pad patches plus four table support patches
        param.mpc_model = "explicit"
        param.runtime_object_cost_ = True
        param.mpc_carry_velocity_ = False
        param.mpc_cost_kind = "param"
        iterations = int(getattr(param, "online_sqp_iters", 1))
        if iterations < 1:
            raise ValueError("Online SQP iterations must be positive")
        param.mpc_acados_profile_ = dict(nlp_solver_type="SQP_RTI" if iterations == 1 else "SQP",
                                        qp_solver="FULL_CONDENSING_HPIPM", nlp_solver_max_iter=iterations,
                                        levenberg_marquardt=.01)
        param.mpc_ipopt_fallback_ = bool(getattr(param, "online_ipopt_fallback", False))
        param.mpc_prune_codegen_ = bool(getattr(param, 'mpc_prune_codegen_', True))
        param.ipopt_max_iter_ = int(getattr(param, "online_ipopt_iters", getattr(param, "ipopt_max_iter_", 80)))
        if param.ipopt_max_iter_ < 1:
            raise ValueError("Online IPOPT iterations must be positive")
        param.robot_stiff_ = float(param.online_task_stiffness) * np.eye(6)
        limit = min(float(param.planner_cmd_limit), float(param.contact_command_force_limit_) /
                    (float(param.online_task_stiffness) * np.sqrt(3)))
        param.mpc_u_lb_, param.mpc_u_ub_ = -np.full(6, limit), np.full(6, limit)
        param.mpc_q_lb_, param.mpc_q_ub_ = -np.full(25, 1e7), np.full(25, 1e7)
        param.online_horizontal_recovery_weight = float(getattr(param, "online_horizontal_recovery_weight", 2e6))
        if not np.isfinite(param.online_horizontal_recovery_weight) or param.online_horizontal_recovery_weight <= 0:
            raise ValueError("Horizontal recovery weight must be finite and positive")
        param.support_surface_tangents_ = np.array([[1., 0., 0.], [0., 1., 0.]])
        param.support_surface_normal_ = np.array([0., 0., 1.])
        model = BimanualRigidContactModel(param)
        self.horizon, self.knot_substeps = int(param.mpc_horizon_), 1
        self._physics = model
        param.init_cost_fns = lambda: self._build_online_costs(param, model)
        self._context = None
        self._previous_action = np.zeros(6)
        self._warm_controls = None
        super().__init__(param, cost_kind="param", model=model)
        # The custom predictor transports both pads internally. Table support
        # patches come from the current finite pedestal footprint, not the
        # single-finger support-refresh adapter in MPCExplicit.
        self.support_vertices = None

    def _build_online_costs(self, param, model):
        self._head_size = 42
        self._phi_size, self._jac_size = 24, 288
        cp = cs.SX.sym("dual_objective", self._head_size + 24 + 288 + model.model_parameter_dim)
        x, u = cs.SX.sym("state", 25), cs.SX.sym("action", 6)
        target, points, normals, forces = cp[:7], cs.reshape(cp[7:13], 3, 2), cs.reshape(cp[13:19], 3, 2), cp[19:25]
        established, previous = cp[25], cp[29:35]
        selected_established = cp[37]
        reference_moment = cp[39:42]
        # Contact mode per pad: -1 fixed world waypoint, 0 prospective, 1 loaded.
        phi = cp[42:66]
        jac = cs.reshape(cp[66:354], 24, 12)
        mp = cp[354:]
        tolerance = float(getattr(param, "online_max_horizontal_error", .005))
        radius = cs.fmax(cs.fabs(mp[:2] - cp[26:28]), tolerance)
        corridor_lower = cs.if_else(established > .5, cs.fmin(mp[:2], target[:2]) - tolerance,
                                    cp[26:28] - radius)
        corridor_upper = cs.if_else(established > .5, cs.fmax(mp[:2], target[:2]) + tolerance,
                                    cp[26:28] + radius)

        def state_residual(state, predicted_force=None, predicted_moment=None):
            rotation = rotation_cs(state[3:7])
            errors = []
            for arm in range(2):
                center = state[:3] + rotation @ (points[:, arm] - param.online_contact_offset * normals[:, arm])
                world_waypoint = mp[:3] + rotation_cs(mp[3:7]) @ (
                    points[:, arm] - param.online_contact_offset * normals[:, arm])
                center = cs.if_else(cp[35+arm] < -.5, world_waypoint, center)
                error = state[7 + 3 * arm:10 + 3 * arm] - center
                # A made contact follows its real tangent patch. Its normal
                # displacement is free to build the allocated pressure.
                normal = rotation @ normals[:, arm]
                # Retain a soft compression attraction. Dropping the entire
                # normal residual removed the preload that made the previous
                # grasp hold while the force model settled.
                error -= .75 * cs.fmax(cp[35+arm], 0) * cp[38] * normal * cs.dot(normal, error)
                errors.append(error)
            # Keep actual load while correcting anchors. An off-anchor grasp
            # must not use the full pre-contact attraction against the grip.
            anchor_scale = 1 - established * (1 - selected_established) * .75
            terms = [cs.sqrt(param.planner_ee_position_weight_ * anchor_scale) * cs.vertcat(*errors)]
            terms.append(np.sqrt(param.planner_object_lateral_weight_) * (state[:2] - target[:2]))
            # Recovery bounds must permit braking an already-observed drift.
            # A hard face at x_current makes outward momentum infeasible at
            # the very first step. Strong soft penalties retain the corridor.
            terms.append(np.sqrt(param.online_horizontal_recovery_weight) * cs.fmax(corridor_lower - state[:2], 0))
            terms.append(np.sqrt(param.online_horizontal_recovery_weight) * cs.fmax(state[:2] - corridor_upper, 0))
            # Price the final goal at every predicted stage. A transient loss
            # of one contact must not change the objective into lowering the
            # object back to its initial table pose.
            terms.append(np.sqrt(param.planner_object_target_weight_) * (state[:3] - target[:3]))
            terms.append(np.sqrt(param.planner_object_orientation_weight_) * (
                rotation_cs(target[3:7]).T @ rotation - cs.DM.eye(3)).reshape((9, 1)))
            distances = [cs.sqrt(cs.sumsqr(error) + 1e-10) for error in errors]
            terms.append(cs.sqrt(param.planner_synchronization_weight_ * anchor_scale) * (distances[0] - distances[1]))
            terms.append(np.sqrt(param.online_object_velocity_weight) * state[13:19])
            if predicted_force is not None:
                # In approach, unilateral pressure is costly even if the
                # contacted pad itself already has a small position error.
                normal_force = [cs.dot(rotation @ normals[:, arm], predicted_force[3 * arm:3 * arm + 3])
                                for arm in range(2)]
                # Convert force error to its equivalent OSC increment. A
                # contact-gate length divided by a force cap makes this term
                # numerically disappear near the unloaded friction branch.
                scale = 1.0 / param.online_task_stiffness
                terms.append(np.sqrt(param.planner_contact_gate_sync_weight_) * scale *
                             (1 - established) * (normal_force[0] - normal_force[1]))
                reference_normal = [cs.dot(normals[:, arm], forces[3*arm:3*arm+3]) for arm in range(2)]
                terms.append(np.sqrt(param.planner_force_tracking_weight_) * scale *
                    cs.vertcat(*[cs.fmax(cp[35+arm], 0)*(normal_force[arm]-reference_normal[arm]) for arm in range(2)]))
                # Penalize the resulting wrench, allowing useful tangential
                # force redistribution rather than tracking each tangent force.
                predicted_local = [rotation.T @ predicted_force[3*arm:3*arm+3] for arm in range(2)]
                force_error = sum(predicted_local)-forces[:3]-forces[3:6]
                torque_error = sum(cs.cross(points[:, arm], predicted_local[arm]-forces[3*arm:3*arm+3])
                                   for arm in range(2))
                if predicted_moment is not None:
                    torque_error += sum(rotation.T@predicted_moment[3*arm:3*arm+3] for arm in range(2))
                torque_error -= reference_moment
                radius = cs.fmax(cs.norm_2(points[:, 1]-points[:, 0])*.5, .01)
                terms.append(np.sqrt(float(getattr(param, 'online_wrench_cost_weight', 1.))) *
                    scale*cp[38]*cs.fmax(cp[35], 0)*cs.fmax(cp[36], 0)*cs.vertcat(force_error, torque_error/radius))
            lower, upper = cs.DM(param.planner_workspace_lower_), cs.DM(param.planner_workspace_upper_)
            for tip in (state[7:10], state[10:13]):
                terms.append(np.sqrt(param.planner_workspace_weight_) * cs.fmax(lower - tip, 0))
                terms.append(np.sqrt(param.planner_workspace_weight_) * cs.fmax(tip - upper, 0))
            separation = cs.sqrt(cs.sumsqr(state[7:10] - state[10:13]) + 1e-20)
            terms.append(np.sqrt(param.planner_workspace_weight_) * cs.fmax(2 * param.tip_radius - separation, 0))
            return terms

        predicted = model.prediction_fn(x, u, phi, jac, mp)
        predicted_force = model.contact_force_fn(x, u, phi, jac, mp)
        predicted_moment = model.contact_moment_fn(x, u, phi, jac, mp)
        terms = state_residual(predicted, predicted_force, predicted_moment)
        terms += [np.sqrt(param.planner_action_weight_) * u,
                  np.sqrt(param.planner_smooth_action_weight_) * (u - previous)]
        residual = cs.vertcat(*terms)
        terminal = np.sqrt(param.online_terminal_weight) * cs.vertcat(*state_residual(x))
        self.residual_fn = cs.Function("dual_cost_residual", [x, u, cp], [residual], {"cse": True})
        self.terminal_residual_fn = cs.Function("dual_terminal_residual", [x, cp], [terminal], {"cse": True})
        return (cs.Function("dual_online_stage_cost", [x, u, cp], [cs.sumsqr(residual)], {"cse": True}),
                cs.Function("dual_online_terminal_cost", [x, cp], [cs.sumsqr(terminal)], {"cse": True}))

    def _cost_params(self, **unused):
        return self._context

    def reset(self):
        super().reset()
        self._previous_action = np.zeros(6)
        self._warm_controls = None

    def plan_once(self, target_p, target_q, curr_x, phi_vec, jac_mat, *,
                  contact_points_local, normals_local, desired_force_local,
                  object_inertia, external_wrench, contact_established=False,
                  selected_contact_mask=(False, False), approach_origin=None,
                  grasp_established=None, measured_normal_force=None, live_grasp=None,
                  approach_centers_world=None, approach_mask=None, **unused):
        started = time.perf_counter()
        state = np.asarray(curr_x, float).reshape(25)
        grasp = bool(contact_established if grasp_established is None else grasp_established)
        self.model.set_observation(state, object_inertia, external_wrench, grasp_established=grasp)
        phi = np.asarray(phi_vec, float).reshape(24)
        jac = np.asarray(jac_mat, float).reshape(24, 12)
        if measured_normal_force is not None:
            self.model.set_contact_preload(state, phi, jac, measured_normal_force)
        mp = self.model.runtime_parameters()
        normals = np.asarray(normals_local, float).reshape(2, 3)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        if not np.isfinite(normals).all() or np.any(lengths < 1e-10):
            raise ValueError("Selected contact normals must be finite and nonzero")
        normals = normals / lengths
        physical_normals = jac[:8].reshape(2, 4, 12).mean(axis=1)[:, :3]
        force_reference = np.asarray(desired_force_local, float).reshape(2, 3)
        if grasp and not contact_established and live_grasp is None:
            hold_force = self._grasp_hold_force(jac, external_wrench, measured_normal_force)
            rotation = np.asarray(rotation_cs(cs.DM(state[3:7])), float)
            force_reference = hold_force @ rotation
        origin = state[:3] if approach_origin is None else np.asarray(approach_origin, float).reshape(3)
        anchor_points = np.asarray(contact_points_local, float).reshape(2, 3).copy()
        contact_modes = np.asarray(selected_contact_mask, float).reshape(2).copy()
        if approach_centers_world is not None:
            mask = np.asarray(approach_mask, bool).reshape(2)
            rotation = np.asarray(rotation_cs(cs.DM(state[3:7])), float)
            centers = np.asarray(approach_centers_world, float).reshape(2, 3)
            anchor_points[mask] = ((centers[mask]-state[:3]) @ rotation +
                                   self.param_.online_contact_offset*normals[mask])
            contact_modes[mask] = -1.
        head = np.r_[target_p, target_q, anchor_points.reshape(6),
                     normals.reshape(6), force_reference.reshape(6),
                     float(grasp), origin, self._previous_action, contact_modes,
                     float(contact_established), float(live_grasp is not None and live_grasp['force_allocation_feasible']),
                     np.zeros(3) if live_grasp is None else np.asarray(
                         live_grasp.get('desired_moments_local', np.zeros((2, 3)))).reshape(2, 3).sum(0)]
        self._context = np.r_[head, phi, jac.reshape(-1, order="F"), mp]
        if not np.isfinite(self._context).all():
            raise ValueError("Online MPC objective must be finite")
        tolerance = float(getattr(self.param_, "online_max_horizontal_error", .005))
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Maximum horizontal error must be finite and positive")
        if grasp:
            lower = np.minimum(state[:2], np.asarray(target_p)[:2]) - tolerance
            upper = np.maximum(state[:2], np.asarray(target_p)[:2]) + tolerance
        else:
            # A contact-position cost alone can lower its error by dragging
            # the object toward the farther pad. Bound that displacement at
            # the original approach pose. Already-observed drift may recover
            # without making the next-state NLP immediately infeasible.
            radius = np.maximum(np.abs(state[:2] - origin[:2]), tolerance)
            lower, upper = origin[:2] - radius, origin[:2] + radius
        self._horizontal_bounds = np.stack((lower, upper))
        # Regenerate a feasible warm trajectory from this measured state and
        # this cycle's patches, including on contact-pair or solver changes.
        controls = np.zeros((self.horizon, 6)) if self._warm_controls is None else self._warm_controls.copy()
        if self._warm_controls is None:
            centers = state[:3] + (anchor_points - self.param_.online_contact_offset * normals) @ np.asarray(
                rotation_cs(cs.DM(state[3:7])), float).T
            controls[:] = np.clip((centers.reshape(6) - state[7:13]) * .2,
                                   self.param_.mpc_u_lb_, self.param_.mpc_u_ub_)
        if np.all(phi[:8].reshape(2, 4).mean(axis=1) <= .003) and np.all(
                np.linalg.norm(jac[:8].reshape(2, 4, 12).mean(axis=1)[:, :3], axis=1) > .5):
            rotation = np.asarray(rotation_cs(cs.DM(state[3:7])), float)
            inward = normals @ rotation.T
            pressure = np.sum(controls.reshape(self.horizon, 2, 3) * inward, axis=2)
            # At exactly zero normal load Coulomb tangential mobility has a
            # zero derivative. Seed the loaded branch of the same objective;
            # the optimizer can still release either pad if that costs less.
            controls = (controls.reshape(self.horizon, 2, 3) +
                        (np.maximum(.001 - pressure, 0)*(contact_modes >= 0))[..., None] * inward).reshape(self.horizon, 6)
            controls = np.clip(controls, self.param_.mpc_u_lb_, self.param_.mpc_u_ub_)
        warm_states = self._rollout(state, controls, phi, jac, mp)
        guess = dict(x0=np.column_stack((controls, warm_states)).reshape(-1),
                     u_traj=controls, x_traj=np.vstack((state, warm_states)))
        result = super().plan_once(target_p, target_q, state, phi, jac, sol_guess=guess)
        controls = np.asarray(result["u_traj"], float).reshape(self.horizon, 6)
        solver_success = (result.get("solver_backend") == "acados" or
                          result.get("solve_status") in {"Solve_Succeeded", "Solved_To_Acceptable_Level"})
        fallback_control = None
        if not solver_success:
            # Never execute a failed NLP iterate. Retain measured-state grip
            # control without moving toward the goal while a solver recovers.
            controls, fallback_control = self._fallback_controls(
                state, phi, jac, mp, guess["u_traj"], external_wrench,
                measured_normal_force, grasp,
                force_reference_world=None if live_grasp is None or not live_grasp['force_allocation_feasible']
                    else force_reference @ np.asarray(rotation_cs(cs.DM(state[3:7])), float).T)
        controls = self._bounded_controls(controls)
        nonlinear_step_scale = 1.
        if solver_success:
            controls, nonlinear_step_scale = self._select_nonlinear_controls(
                state, controls, guess['u_traj'], phi, jac, mp)
        rollout = self._rollout(state, controls, phi, jac, mp)
        if not np.isfinite(rollout).all() or not np.isfinite(controls).all():
            raise FloatingPointError("Online MPC returned a non-finite action or rollout")
        self._previous_action = controls[0].copy()
        self._warm_controls = np.vstack((controls[1:], controls[-1:])) if (solver_success or grasp) else None
        first_force = np.asarray(self.model.contact_force_fn(state, controls[0], phi, jac, mp)).reshape(2, 3)
        first_moment = np.asarray(self.model.contact_moment_fn(state, controls[0], phi, jac, mp)).reshape(2, 3)
        result.update(action=controls[0].copy(), u_traj=controls, rollout_q=rollout,
                      x_plus_opt=rollout[0, :7].copy(), execution_object_pose_world=rollout[0, :7].copy(),
                      normal_force=np.maximum(np.sum(first_force * physical_normals, axis=1), 0)[None],
                      solve_time=time.perf_counter() - started)
        result["contact_mask"] = result["normal_force"] > 1e-6
        objective_now = self._object_cost(state, target_p, target_q)
        objective_next = self._object_cost(rollout[0], target_p, target_q)
        cost = sum(float(self.path_cost_fn(x, u, self._context)) for x, u in zip(np.vstack((state, rollout[:-1])), controls))
        cost += float(self.final_cost_fn(rollout[-1], self._context))
        result.update(cost=cost, cost_opt=np.array([cost]))
        result["diagnostics"] = dict(contact_model=self.model.contact_model,
            live_grasp=live_grasp,
            fallback_reason=result.get("fallback_reason"),
            acados_error=result.get("acados_error"),
            acados_attempt_errors=result.get("acados_attempt_errors", []),
            acados_qp_diagnostics=result.get("acados_qp_diagnostics"),
            acados_failure_snapshot=result.get("acados_failure_snapshot"),
            solve_status=result.get("solve_status"),
            ipopt_iterations=result.get("ipopt_iterations"),
            ipopt_initialization_time=result.get("ipopt_initialization_time", 0.),
            ipopt_solve_time=result.get("ipopt_solve_time", 0.),
            solver_success=solver_success,
            nonlinear_step_scale=nonlinear_step_scale,
            fallback_control=fallback_control,
            grasp_established=grasp,
            selected_contact_established=bool(contact_established),
            phase=("manipulate" if contact_established and grasp else "grasp_recovery" if grasp else "approach"),
            recovery_stage=None if live_grasp is None else live_grasp.get('recovery_stage'),
            approach_centers_world=None if approach_centers_world is None else np.asarray(approach_centers_world).copy(),
            approach_mask=None if approach_mask is None else np.asarray(approach_mask, bool).copy(),
            object_cost_now=objective_now, object_cost_next=objective_next,
            predicted_cost_reduction=objective_now - objective_next,
            predicted_horizontal_displacement=rollout[0, :2] - state[:2],
            horizontal_bounds=np.stack((lower, upper)),
            predicted_horizontal_bound_violation=float(np.maximum(
                np.maximum(lower - rollout[:, :2], rollout[:, :2] - upper), 0).max()),
            selected_contact_mask=np.asarray(selected_contact_mask, bool),
            approach_origin=origin.copy(),
            predicted_force_world=first_force, predicted_moment_world=first_moment,
            observed_preload_bias=self.model.normal_preload_bias.copy(), state_dim=25, action_dim=6)
        return result

    def _select_nonlinear_controls(self, state, controls, warm, phi, jac, mp):
        """Validate RTI steps against the actual transported contact model.

        The QP linearization can cross a contact/cone branch. Executing that
        iterate solely because acados returned success can lose a real grip.
        """
        warm = self._bounded_controls(warm)
        best, best_scale, best_cost = warm, 0., np.inf
        for scale in (0., 1., .5, .25, .1):
            candidate = warm+scale*(controls-warm)
            rollout = self._rollout(state, candidate, phi, jac, mp)
            if not np.isfinite(rollout).all():
                continue
            states = np.vstack((state, rollout[:-1]))
            cost = sum(float(self.path_cost_fn(x, u, self._context)) for x, u in zip(states, candidate))
            cost += float(self.final_cost_fn(rollout[-1], self._context))
            if np.isfinite(cost) and cost < best_cost:
                best, best_scale, best_cost = candidate, scale, cost
        return best, best_scale

    def _grasp_hold_force(self, jac, external_wrench, measured_normal_force):
        blocks = np.asarray(jac).reshape(6, 4, 12)[:2]
        normals = blocks.mean(axis=1)[:, :3]
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
        mu = np.maximum(np.linalg.norm((blocks[:, 0, :3] - blocks[:, 2, :3]) / 2, axis=1), 1e-4)
        external = np.asarray(external_wrench, float)[:3]
        observed = np.zeros(2) if measured_normal_force is None else np.asarray(measured_normal_force, float).reshape(2)
        if not np.isfinite(observed).all() or np.any(observed < 0):
            raise ValueError("Measured normal forces must be finite and nonnegative")
        pressure = np.minimum(np.maximum(observed, np.maximum(.01, 1.5 * np.linalg.norm(external) / (2 * mu))),
                              self.param_.contact_command_force_limit_ / 2)
        gravity_share = -external / 2
        tangent = gravity_share - normals * np.sum(normals * gravity_share, axis=1)[:, None]
        return pressure[:, None] * normals + tangent

    def _bounded_controls(self, controls):
        controls = np.clip(np.asarray(controls, float), self.param_.mpc_u_lb_, self.param_.mpc_u_ub_)
        norms = np.linalg.norm(controls @ self.param_.robot_stiff_.T, axis=1)
        return controls * np.minimum(1., self.param_.contact_command_force_limit_ / np.maximum(norms, 1e-12))[:, None]

    def _fallback_controls(self, state, phi, jac, mp, warm, external, measured, grasp,
                           force_reference_world=None):
        if not grasp:
            return np.zeros((self.horizon, 6)), "zero_approach"
        force = (self._grasp_hold_force(jac, external, measured) if force_reference_world is None
                 else np.asarray(force_reference_world).reshape(2, 3))
        hold = ((force + self.param_.mpc_tip_damping_ * state[19:25].reshape(2, 3)) /
                self.param_.online_task_stiffness).reshape(6)
        candidates = [("force_hold", np.tile(hold, (self.horizon, 1))),
                      ("previous_action", np.tile(self._previous_action, (self.horizon, 1))),
                      ("shifted_plan", warm)]
        best, score = None, np.inf
        emergency_margin = float(getattr(self.param_, "online_max_horizontal_error", .005))
        for label, candidate in candidates:
            candidate = self._bounded_controls(candidate)
            prediction = self._rollout(state, candidate, phi, jac, mp)
            if not np.isfinite(prediction).all():
                continue
            force_next = np.asarray(self.model.contact_force_fn(state, candidate[0], phi, jac, mp)).reshape(2, 3)
            normal = jac[:8].reshape(2, 4, 12).mean(axis=1)[:, :3]
            requested = (self.param_.online_task_stiffness * candidate[0].reshape(2, 3) -
                         self.param_.mpc_tip_damping_ * state[19:25].reshape(2, 3))
            if np.any(np.sum(requested * normal, axis=1) <= 1e-6):
                continue
            retained = np.all(np.sum(force_next * normal, axis=1) > 1e-6)
            horizontal_ok = np.all(prediction[:, :2] >= self._horizontal_bounds[0] - emergency_margin) and np.all(
                prediction[:, :2] <= self._horizontal_bounds[1] + emergency_margin)
            if not retained or not horizontal_ok:
                continue
            # Hold the observed object pose, rather than pursuing an airborne
            # goal using an unsuccessful nonlinear solve.
            error = np.max(np.sum((prediction[:, :3] - state[:3]) ** 2, axis=1))
            if error < score:
                best, score = (candidate, label), error
        if best is not None:
            return best
        # The reduced model can reject real margin contacts. A bounded normal
        # preload at measured contact directions keeps active OSC pressure;
        # report this branch separately from a revalidated plan.
        return self._bounded_controls(candidates[0][1]), "force_hold_unverified"

    def _rollout(self, state, controls, phi, jac, mp):
        rows = []
        for control in controls:
            state = np.asarray(self.model.prediction_fn(state, control, phi, jac, mp)).reshape(25)
            rows.append(state)
        return np.asarray(rows)

    def _object_cost(self, state, target_p, target_q):
        p = self.param_
        rotation = np.asarray(rotation_cs(cs.DM(state[3:7])), float)
        target_rotation = np.asarray(rotation_cs(cs.DM(target_q)), float)
        return (p.planner_object_target_weight_ * np.sum((state[:3] - target_p) ** 2)
                + p.planner_object_lateral_weight_ * np.sum((state[:2] - np.asarray(target_p)[:2]) ** 2)
                + p.planner_object_orientation_weight_ * np.sum((target_rotation.T @ rotation - np.eye(3)) ** 2))
