"""In-memory, singleton ForceAware retargeting for the two Panda arms.

The trajectory targets and optimizer are DexForge's original interfaces.  The
adapter supplies Panda collision geometry and angular coordinate scales where
the bundled dexterous-hand implementation assumes a six-coordinate wrist.
"""

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np


_CLASSES = None


def mppi_weights(costs, temperature, valid=None):
    """Stable MPPI softmax weights, excluding failed or nonfinite rollouts."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("MPPI temperature must be positive and finite")
    costs = np.asarray(costs, np.float64)
    if costs.ndim != 1 or not costs.size:
        raise ValueError("MPPI costs must be a nonempty vector")
    usable = np.isfinite(costs)
    if valid is not None:
        valid = np.asarray(valid, bool)
        if valid.shape != costs.shape:
            raise ValueError("MPPI validity must match costs")
        usable &= valid
    if not np.any(usable):
        raise FloatingPointError("all MPPI candidates are invalid")
    result = np.zeros(costs.shape, np.float64)
    with np.errstate(over="ignore", under="ignore"):
        result[usable] = np.exp(-(costs[usable] - costs[usable].min()) / temperature)
    result /= result.sum()
    return result


def mppi_weighted_mean(samples, costs, *, temperature, valid=None):
    """Update the MPPI trajectory mean using only finite candidate controls."""
    samples = np.asarray(samples, np.float64)
    costs = np.asarray(costs, np.float64)
    if samples.ndim < 2 or samples.shape[0] != costs.size:
        raise ValueError("MPPI samples must have one trajectory per cost")
    usable = np.isfinite(samples.reshape(samples.shape[0], -1)).all(axis=1)
    if valid is not None:
        valid = np.asarray(valid, bool)
        if valid.shape != usable.shape:
            raise ValueError("MPPI validity must match samples")
        usable &= valid
    weights = mppi_weights(costs, temperature, usable)
    positive = weights > 0.0
    return np.tensordot(weights[positive], samples[positive], axes=(0, 0))


def _classes():
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    from planning.dexforge_imports import ensure_dexforge

    ensure_dexforge()
    import warp as wp
    from forceaware import config, mpc_loss, optimizer
    from forceaware.control import action_knot_interpolation_map
    from forceaware.numerics import LOSS_GRADIENT_SCALE
    from comfree_warp.native_adjoint.dynamics import DynamicsModel, DynamicsInput

    @wp.kernel(enable_backward=False)
    def servo_implicit_damping(model: DynamicsModel, inputs: DynamicsInput,
                               damping: wp.array2d(dtype=float)):
        world, dof = wp.tid()
        value = float(model.damper_enabled) * model.dof_damping[world % model.dof_damping.shape[0], dof]
        if model.actuation_enabled != 0:
            for actuator in range(model.actuator_count):
                if model.actuator_dofid[actuator] == dof:
                    row = world % model.actuator_gainprm.shape[0]
                    gear = model.actuator_gear[row, actuator][0]
                    ctrl = inputs.ctrl[world, actuator]
                    if model.clamp_control != 0 and model.actuator_ctrllimited[actuator]:
                        limits = model.actuator_ctrlrange[row, actuator]
                        ctrl = wp.clamp(ctrl, limits[0], limits[1])
                    bias = model.actuator_biasprm[row, actuator]
                    force = (model.actuator_gainprm[row, actuator][0] * ctrl + bias[0]
                             + bias[1] * inputs.qpos[world, model.actuator_qposadr[actuator]] * gear
                             + bias[2] * inputs.qvel[world, dof] * gear)
                    active = bool(True)
                    if model.actuator_forcelimited[actuator]:
                        limits = model.actuator_forcerange[row, actuator]
                        active = force > limits[0] and force < limits[1]
                    if active:
                        value -= bias[2] * gear * gear
        damping[world, dof] = value

    @wp.kernel
    def panda_pair_loss(job: mpc_loss.LossJob, loss: wp.array(dtype=float)):
        world, pair = wp.tid()
        left = job.model.finger_joint_pair_left[pair]
        right = job.model.finger_joint_pair_right[pair]
        # Panda query primitives are spheres (zero-length capsules). This is
        # exactly the upstream segment-distance expression for those primitives.
        delta = (mpc_loss._finger_joint_world(job, world, left, job.model.finger_joint_start[left])
                 - mpc_loss._finger_joint_world(job, world, right, job.model.finger_joint_start[right]))
        distance = wp.sqrt(wp.dot(delta, delta) + mpc_loss.LOSS_EPSILON) * mpc_loss.MILLIMETERS
        radius = (job.model.finger_joint_radius[left] + job.model.finger_joint_radius[right]) * mpc_loss.MILLIMETERS
        deficit = wp.max(job.weights.finger_joint_surface_margin_mm - distance + radius, 0.0)
        # Zero penalties have zero VJP. Avoid hundreds of contended atomic
        # additions per world when most sphere pairs have ample clearance.
        if deficit > 0.0:
            wp.atomic_add(loss, world, job.weights.finger_joint_collision * deficit * deficit)

    @wp.kernel(enable_backward=False)
    def project_offsets(
        raw: wp.array2d(dtype=float),
        target: wp.array(dtype=float),
        lower: wp.array2d(dtype=float),
        upper: wp.array2d(dtype=float),
        knot: int,
        action_dim: int,
    ):
        world, action = wp.tid()
        offset = action_dim * (knot + 1) + action
        value = target[action] + raw[world, offset]
        raw[world, offset] = wp.clamp(value, lower[knot, action], upper[knot, action]) - target[action]

    @wp.kernel(enable_backward=False)
    def mppi_sample(mean: wp.array(dtype=float), raw: wp.array2d(dtype=float),
                    iteration: wp.array(dtype=int), random_seed: wp.array(dtype=int),
                    action_dim: int, noise_scale: float, noise_decay: float):
        world, parameter = wp.tid()
        value = float(0.0)
        if parameter >= action_dim:
            value = mean[parameter]
            if world > 0:
                # Changing the offset per iteration gives fresh, reproducible
                # noise while keeping all sampling inside the captured graph.
                random = wp.rand_init(random_seed[0] + iteration[0] * 104729,
                                      world * raw.shape[1] + parameter)
                value += noise_scale * wp.pow(noise_decay, float(iteration[0])) * wp.randn(random)
        raw[world, parameter] = value

    @wp.kernel(enable_backward=False)
    def mppi_project_mean(mean: wp.array(dtype=float), target: wp.array(dtype=float),
                          lower: wp.array2d(dtype=float), upper: wp.array2d(dtype=float),
                          knot: int, action_dim: int):
        action = wp.tid()
        offset = action_dim * (knot + 1) + action
        value = target[action] + mean[offset]
        mean[offset] = wp.clamp(value, lower[knot, action], upper[knot, action]) - target[action]

    @wp.kernel(enable_backward=False)
    def mppi_baseline(loss: wp.array(dtype=float), valid: wp.array(dtype=int),
                      baseline: wp.array(dtype=float), statistics: wp.array(dtype=float)):
        minimum = float(0.0)
        count = int(0)
        for world in range(loss.shape[0]):
            if valid[world] != 0 and wp.isfinite(loss[world]):
                if count == 0:
                    minimum = loss[world]
                else:
                    minimum = wp.min(minimum, loss[world])
                count += 1
        baseline[0] = minimum
        statistics[0] = float(count)

    @wp.kernel(enable_backward=False)
    def mppi_softmax(loss: wp.array(dtype=float), valid: wp.array(dtype=int),
                     baseline: wp.array(dtype=float), weights: wp.array(dtype=float),
                     temperature: float):
        world = wp.tid()
        weight = float(0.0)
        if valid[world] != 0 and wp.isfinite(loss[world]):
            weight = wp.exp(-(loss[world] - baseline[0]) / temperature)
        weights[world] = weight

    @wp.kernel(enable_backward=False)
    def mppi_normalize(weights: wp.array(dtype=float), statistics: wp.array(dtype=float)):
        total = wp.float32(0.0)
        for world in range(weights.shape[0]):
            total += weights[world]
        square_sum = float(0.0)
        for world in range(weights.shape[0]):
            value = float(0.0)
            if total > 0.0:
                value = weights[world] / total
            weights[world] = value
            square_sum += value * value
        statistics[1] = total
        statistics[2] = float(0.0)
        if square_sum > 0.0:
            statistics[2] = 1.0 / square_sum

    @wp.kernel(enable_backward=False)
    def mppi_update(raw: wp.array2d(dtype=float), weights: wp.array(dtype=float),
                    statistics: wp.array(dtype=float), mean: wp.array(dtype=float), action_dim: int):
        parameter = wp.tid()
        if parameter < action_dim:
            mean[parameter] = 0.0
        elif statistics[1] > 0.0:
            value = wp.float32(0.0)
            for world in range(raw.shape[0]):
                if weights[world] > 0.0:
                    value += weights[world] * raw[world, parameter]
            mean[parameter] = value

    @wp.kernel(enable_backward=False)
    def mppi_final_candidates(mean: wp.array(dtype=float), best: wp.array2d(dtype=float),
                              best_loss: wp.array(dtype=float), available: wp.array(dtype=int),
                              raw: wp.array2d(dtype=float)):
        world, parameter = wp.tid()
        value = wp.float32(mean[parameter])
        if world == 1:
            # Preserve the best sampled trajectory even if it came from the
            # nominal world that is now occupied by the updated mean.
            selected = int(-1)
            for candidate in range(best.shape[0]):
                if available[candidate] != 0 and wp.isfinite(best_loss[candidate]):
                    if selected < 0:
                        selected = candidate
                    elif best_loss[candidate] < best_loss[selected]:
                        selected = candidate
            if selected >= 0:
                value = best[selected, parameter]
        elif world > 1 and available[world] != 0:
            value = best[world, parameter]
        raw[world, parameter] = value

    @wp.kernel(enable_backward=False)
    def fa_cost_components(job: mpc_loss.LossJob, loss: wp.array(dtype=float),
                           selected: wp.array(dtype=int), column: int,
                           values: wp.array2d(dtype=float)):
        world = selected[0]
        hand = job.reference_scale * 1000000.0 * mpc_loss._hand_reference_squared(job, world)
        obj = (job.reference_scale * 1000000.0 * job.weights.reference_pose_scale
               * mpc_loss._object_reference_squared(job, world))
        query = float(0.0)
        depth = float(0.0)
        for contact in range(job.model.guidance_count):
            if job.query_enabled != 0:
                query += mpc_loss._marked_query_loss(job, world, contact)
            if job.guidance_enabled != 0:
                depth += mpc_loss._physical_contact_loss(job, world, contact)
        penetration = float(0.0)
        if job.penetration_enabled != 0:
            penetration = mpc_loss._penetration_loss(job, world)
        values[column, 0] = hand
        values[column, 1] = obj
        values[column, 2] = query
        values[column, 3] = depth
        values[column, 4] = penetration
        # The parallel sphere-pair loss is already included in workspace.loss.
        # This residual avoids repeating its 625 pair calculations for logging.
        # A zero residual can be slightly negative through float32 summation.
        values[column, 5] = wp.max(loss[world] - (hand + obj + query + depth + penetration), 0.0)

    class PandaRegularizer(mpc_loss.MpcRegularizer):
        def __init__(self, cpu_model, index, cfg, *, worlds, device):
            super().__init__(cpu_model, index, cfg, worlds=worlds, device=device)
            # All fourteen Panda coordinates are angles.  Keep the original
            # regularization kernels, replacing the virtual-wrist unit scales.
            model = self.model
            model.init_bound.fill_(cfg.bounds.init_finger)
            model.control_velocity_scale.fill_(cfg.smoothing.control_velocity_finger_scale)
            model.control_velocity_weight.fill_(cfg.smoothing.control_velocity_weight)
            model.control_acceleration_scale.fill_(cfg.smoothing.control_acceleration_finger_scale)
            model.state_velocity_scale.fill_(mpc_loss.HAND_STATE_FINGER_VELOCITY_SCALE_RAD_S)
            model.state_acceleration_scale.fill_(mpc_loss.HAND_STATE_FINGER_ACCELERATION_SCALE_RAD_S2)
            model.state_component_ratio.fill_(1.0)

    class PandaObjective(mpc_loss.MpcObjective):
        def cost_breakdown(self, selected):
            workspaces = (*self.training.immediate, self.training.terminal)
            if not hasattr(self, "_cost_component_values"):
                self._cost_component_selected = wp.zeros(1, dtype=int, device=self.device)
                self._cost_component_values = wp.empty((len(workspaces), 6), dtype=float, device=self.device)
                def launch():
                    for column, workspace in enumerate(workspaces):
                        wp.launch(fa_cost_components, dim=1,
                                  inputs=[workspace.job, workspace.loss, self._cost_component_selected, column],
                                  outputs=[self._cost_component_values], device=self.device)
                # Warm the diagnostic kernel before capturing; buffers remain
                # tied to the original loss workspaces across MPC windows.
                launch()
                if wp.get_device(self.device).is_cuda:
                    with wp.ScopedCapture(device=self.device) as capture:
                        launch()
                    self._cost_component_graph = capture.graph
                else:
                    self._cost_component_launch = launch
            self._cost_component_selected.assign(np.asarray([selected], np.int32))
            if hasattr(self, "_cost_component_graph"):
                wp.capture_launch(self._cost_component_graph)
            else:
                self._cost_component_launch()
            names = ("hand_reference", "object_reference", "marked_query", "contact_depth",
                     "penetration", "interarm_collision")
            return dict(zip(names, map(float, self._cost_component_values.numpy().sum(axis=0, dtype=np.float64))))

        def __init__(self, *args, panda_geometry, **kwargs):
            self.panda_geometry = panda_geometry
            super().__init__(*args, **kwargs)

        def _make_regularizer(self, cpu_model, index, cfg, *, worlds, device):
            return PandaRegularizer(cpu_model, index, cfg, worlds=worlds, device=device)

        def _workspace(self, runtime, *, guidance_enabled, query_enabled, qpos=None,
                       reference_scale=1.0, contact_active=None):
            # This is the original FA loss workspace wiring with the registered
            # hand geometry replaced by explicit Panda body-local capsules.
            result = runtime.recorded.result
            state = mpc_loss.LossState()
            state.qpos = result.qpos if qpos is None else qpos
            for name in ("body_position", "body_matrix", "contact_distance",
                         "contact_position", "contact_frame"):
                setattr(state, name, getattr(result, name))
            if contact_active is None:
                contact_active = runtime.workspace.forward.base.collision.contacts.active
            state.contact_active = contact_active
            collision = self.compiled.base.collision
            force_sign, margin, friction = _object_contact_properties(
                self.cpu_model, self.index, collision)
            com, radius = mpc_loss._object_grasp_geometry(self.cpu_model, self.index, collision)
            bodies, starts, ends, radii, pairs = self.panda_geometry
            directions = mpc_loss.grasp_wrench_directions()
            model = mpc_loss.LossModel()
            for name, value, dtype in (
                ("hand_qpos", self.index.hand_qpos, int),
                ("finger_joint_body", bodies, int),
                ("finger_joint_start", starts, wp.vec3),
                ("finger_joint_end", ends, wp.vec3),
                ("finger_joint_radius", radii, float),
                ("finger_joint_pair_left", pairs[:, 0], int),
                ("finger_joint_pair_right", pairs[:, 1], int),
                ("output_includemargin", margin, float),
                ("output_force_sign", force_sign, float),
                ("output_friction", friction, wp.vec2),
                ("wrench_force_direction", directions[:, :3], wp.vec3),
                ("wrench_torque_direction", directions[:, 3:], wp.vec3),
            ):
                setattr(model, name, wp.array(value, dtype=dtype, device=self.device))
            model.output_contact_rows = collision.output_contact_rows
            model.object_com_local = wp.vec3(*com)
            model.object_radius = radius
            model.object_qpos = self.index.object_qpos
            model.object_body = self.index.object_body
            model.action_dim = self.index.action_dim
            model.guidance_count = 2
            model.output_count = collision.output_count
            # Keep the exact upstream cost, but distribute the 625 Panda pairs
            # across GPU threads instead of serializing them inside each world.
            model.finger_joint_pair_count = 0
            model.wrench_direction_count = len(directions)
            model.contact_thresholds = collision.output_thresholds
            job = mpc_loss.LossJob()
            job.model = model
            job.weights = mpc_loss.make_weights(self.cfg)
            job.target = mpc_loss.allocate_target(self.index.action_dim, 2, self.device)
            job.state = state
            job.guidance_enabled = int(guidance_enabled)
            job.penetration_enabled = 1
            job.query_enabled = int(query_enabled)
            job.reference_scale = reference_scale
            workspace = mpc_loss.build_workspace(job, self.worlds, self.device)
            if len(pairs):
                workspace.tape.record_launch(panda_pair_loss, (self.worlds, len(pairs)), 0,
                                             [job], [workspace.loss], wp.get_device(self.device), block_dim=256)
            return workspace

        def _evaluate_trajectory(self, runtime):
            for column, workspace in enumerate((*runtime.immediate, runtime.terminal)):
                mpc_loss.evaluate(workspace)
                if len(self.panda_geometry[-1]):
                    wp.launch(panda_pair_loss, dim=(self.worlds, len(self.panda_geometry[-1])),
                              inputs=[workspace.job], outputs=[workspace.loss], device=self.device)
                wp.launch(mpc_loss._store_trajectory_loss, dim=self.worlds,
                          inputs=[workspace.loss, column], outputs=[runtime.trajectory_terms], device=self.device)

        def _allocate_training(self, guidance_count):
            if guidance_count != 2:
                raise ValueError("Panda ForceAware requires exactly two lambda contacts")
            immediate = tuple(self._workspace(
                runtime, guidance_enabled=step > 0, query_enabled=step > 0,
                reference_scale=0.0 if step == self.dense_steps - 1 else 1.0,
            ) for step, runtime in enumerate(self.steps))
            return mpc_loss.TrainingLossRuntime(
                immediate=immediate,
                terminal=self._terminal_workspace(guidance_count),
                trajectory_terms=wp.empty((self.worlds, self.dense_steps + 1), dtype=float, device=self.device),
                regularization=self.regularizer.runtime.total,
                total=wp.empty(self.worlds, dtype=float, device=self.device),
            )

        def _terminal_workspace(self, guidance_count):
            del guidance_count
            observation = self.terminal_observation
            return self._workspace(
                observation, guidance_enabled=True, query_enabled=True,
                qpos=self.steps[-1].recorded.result.qpos,
                reference_scale=self.cfg.loss.terminal,
                contact_active=observation.workspace.collision.contacts.active,
            )

    class PandaWindowOptimizer(optimizer.WindowOptimizer):
        def __init__(self, fast_step, index, cfg, geometry):
            # Upstream has no compile hook.  Reuse its existing compiled model
            # and all of its allocation/solve methods, without compiling twice.
            self.cpu_model = fast_step.cpu_model
            self.index = index
            self.cfg = cfg
            self.settings = optimizer.OptimizerSettings.from_config(cfg)
            self.guidance_count = 2
            self.dense_knot_map = action_knot_interpolation_map(
                self.settings.horizon, self.settings.knot_substeps)
            self.compiled = fast_step.compiled
            self.device = self.compiled.base.device
            # Euler's native integration already solves (M + h D) dv = h f.
            # Include the affine servo's velocity derivative, as implicitfast
            # does. Keep the original force law and torque clipping intact;
            # saturated actuators contribute zero velocity derivative.
            self.servo_damping = wp.zeros((self.settings.worlds, self.cpu_model.nv),
                                          dtype=float, device=self.device)
            self.compiled.base.integration.model.dof_damping = self.servo_damping
            self.control = self._allocate_control()
            self.control.model.init_bound.fill_(cfg.bounds.init_finger)
            self.steps = self._allocate_steps()
            self.terminal_observation = self._allocate_terminal_observation()
            self.objective = PandaObjective(
                self.cpu_model, self.compiled, index, self.steps,
                self.terminal_observation, cfg=cfg, guidance_count=2,
                worlds=self.settings.worlds, panda_geometry=geometry)
            self.loss = self.objective.training
            self.iteration = wp.zeros(1, dtype=int, device=self.device)
            self.graphs = {}
            self._regularization_previous_action = None
            shape = (self.settings.horizon, index.action_dim)
            self.action_lower = wp.empty(shape, dtype=float, device=self.device)
            self.action_upper = wp.empty(shape, dtype=float, device=self.device)

        def _multistart(self, warm, seed):
            candidates = super()._multistart(warm, seed)
            # A shifted trajectory can trap Adam after CA changes its contact
            # pair. Also optimize the fresh CA reference (zero correction),
            # using the same executed boundary, bounds and original FA cost.
            # Deterministic interpolants preserve the warm candidate at row 0.
            if self.settings.worlds > 1:
                fraction = np.linspace(1.0, 0.0, self.settings.worlds, dtype=np.float32)
                candidates = fraction[:, None] * np.asarray(warm, np.float32)[None]
            return candidates

        def _forward(self, *, allow_init):
            optimizer._build_initial(self.control, allow_init=allow_init)
            for step in range(self.settings.horizon):
                optimizer._build_control(self.control, step)
            for dense_step, knot_map in enumerate(self.dense_knot_map):
                optimizer._build_dense_control(self.control, dense_step, knot_map)
            for runtime in self.steps:
                wp.launch(servo_implicit_damping, dim=self.servo_damping.shape,
                          inputs=[self.compiled.base.dynamics.model, runtime.inputs.dynamics],
                          outputs=[self.servo_damping], device=self.device)
                optimizer.step_forward(self.compiled, runtime.inputs, runtime.workspace)
            optimizer.observation_forward(self.compiled.base, self.terminal_observation.qpos,
                                           self.terminal_observation.workspace)

        def _project_controls(self):
            for knot, target in enumerate(self.control.target_actions):
                wp.launch(project_offsets, dim=(self.settings.worlds, self.index.action_dim),
                          inputs=[self.control.raw, target, self.action_lower,
                                  self.action_upper, knot, self.index.action_dim], device=self.device)

        def _iteration(self, *, allow_init):
            self._project_controls()
            super()._iteration(allow_init=allow_init)

        def solve(self, *args, **kwargs):
            # Some upstream helper kernels use Warp's current device.
            with wp.ScopedDevice(self.device):
                plan = super().solve(*args, **kwargs)
                # Upstream restores and forwards the best candidate but leaves
                # the last training losses in its buffers.  Re-evaluate the
                # restored plan so cost components describe actual execution.
                self._evaluate_losses(allow_init=kwargs.get("allow_init", False))
                wp.synchronize()
                scores = self.loss.total.numpy().copy()
                plan.diag["scores"] = scores
                return replace(plan, loss=float(scores[plan.diag["selected"]]))

    class PandaMppiWindowOptimizer(PandaWindowOptimizer):
        def _multistart(self, warm, seed):
            # MPPI generates its own perturbations around the shifted mean.
            return optimizer.WindowOptimizer._multistart(self, warm, seed)

        """Sampling update over the original FA objective and physical rollout."""

        def __init__(self, fast_step, index, cfg, geometry, *, iterations, init_iterations,
                     temperature, noise_scale, noise_decay, seed, cost_scale=1.0):
            self.mppi_iterations = iterations
            self.mppi_init_iterations = init_iterations
            self.temperature = temperature
            self.cost_scale = cost_scale
            self.noise_scale = noise_scale
            self.noise_decay = noise_decay
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.solve_count = 0
            super().__init__(fast_step, index, cfg, geometry)
            width = index.action_dim * (self.settings.horizon + 1)
            self.mean = wp.empty(width, dtype=float, device=self.device)
            self.random_seed = wp.zeros(1, dtype=int, device=self.device)
            self.baseline = wp.zeros(1, dtype=float, device=self.device)
            self.weights = wp.zeros(self.settings.worlds, dtype=float, device=self.device)
            self.statistics = wp.zeros(3, dtype=float, device=self.device)

        def _allocate_steps(self):
            # No rollout tapes, native adjoint buffers or backward kernels are
            # needed by sampling.  Reuse the production native forward step.
            from comfree_warp.native_adjoint.fast_step import allocate_workspace, step
            from comfree_warp.native_adjoint.dynamics import DynamicsInput
            from comfree_warp.native_adjoint.runtime import StepInput

            worlds = self.settings.worlds
            qpos, qvel = self.control.initial_qpos, self.control.initial_qvel
            qpos.assign(np.broadcast_to(np.asarray(self.cpu_model.qpos0, np.float32), qpos.shape))
            qvel.zero_()
            time = wp.zeros(worlds, dtype=float, device=self.device)
            steps = []
            for control in self.control.controls:
                control.zero_()
                dynamics = DynamicsInput()
                dynamics.qpos, dynamics.qvel, dynamics.ctrl = qpos, qvel, control
                dynamics.qfrc_applied = wp.zeros((worlds, self.cpu_model.nv), dtype=float, device=self.device)
                inputs = StepInput(dynamics, time, wp.zeros(worlds, dtype=float, device=self.device))
                workspace = allocate_workspace(self.compiled, worlds)
                result = step(self.compiled, inputs, workspace)
                cotangent = SimpleNamespace(
                    qvel=wp.empty((worlds, self.cpu_model.nv), dtype=float, device=self.device),
                    qacc=wp.empty((worlds, self.cpu_model.nv), dtype=float, device=self.device))
                runtime = SimpleNamespace(inputs=inputs, workspace=SimpleNamespace(forward=workspace),
                                          recorded=SimpleNamespace(result=result), cotangent=cotangent)
                steps.append(runtime)
                qpos, qvel, time = result.qpos, result.qvel, result.time
            return tuple(steps)

        def _allocate_terminal_observation(self):
            from comfree_warp.native_adjoint.observation_adjoint import allocate_workspace, forward

            qpos = self.steps[-1].recorded.result.qpos
            workspace = allocate_workspace(self.compiled.base, self.settings.worlds, requires_grad=False)
            result = forward(self.compiled.base, qpos, workspace)
            return SimpleNamespace(qpos=qpos, workspace=workspace, recorded=SimpleNamespace(result=result))

        def _forward(self, *, allow_init):
            from comfree_warp.native_adjoint.fast_step import step
            from comfree_warp.native_adjoint.observation_adjoint import forward

            optimizer._build_initial(self.control, allow_init=False)
            for knot in range(self.settings.horizon):
                optimizer._build_control(self.control, knot)
            for dense, knot_map in enumerate(self.dense_knot_map):
                optimizer._build_dense_control(self.control, dense, knot_map)
            for runtime in self.steps:
                wp.launch(servo_implicit_damping, dim=self.servo_damping.shape,
                          inputs=[self.compiled.base.dynamics.model, runtime.inputs.dynamics],
                          outputs=[self.servo_damping], device=self.device)
                step(self.compiled, runtime.inputs, runtime.workspace.forward)
            forward(self.compiled.base, self.terminal_observation.qpos, self.terminal_observation.workspace)

        def _reset_candidate_status(self):
            self.control.valid.fill_(1)
            self.control.failure_iteration.fill_(optimizer.NO_FAILURE_ITERATION)
            self.control.failure_code.fill_(optimizer.CANDIDATE_FAILURE_NONE)
            self.control.gradient.zero_()
            self.control.gradient_norm.zero_()
            self._clear_contact_history()

        def _evaluate_batch(self):
            self._set_hard_contact()
            self._forward(allow_init=False)
            self._validate_simulator_candidates()
            self._evaluate_losses(allow_init=False)
            optimizer.validate_evaluated_candidates(self.control, self.loss.total, iteration=self.iteration)

        def _iteration(self, *, allow_init):
            self._reset_candidate_status()
            # New CA references and shifted warm starts can be outside the
            # current reach.  Sample around a feasible nominal knot so small
            # inward perturbations survive rather than all clipping identically.
            for knot, target in enumerate(self.control.target_actions):
                wp.launch(mppi_project_mean, dim=self.index.action_dim,
                          inputs=[self.mean, target, self.action_lower, self.action_upper,
                                  knot, self.index.action_dim], device=self.device)
            wp.launch(mppi_sample, dim=self.control.raw.shape,
                      inputs=[self.mean, self.control.raw, self.iteration, self.random_seed,
                              self.index.action_dim, self.noise_scale, self.noise_decay], device=self.device)
            self._project_controls()
            self._evaluate_batch()
            optimizer._track_best(self.control, self.loss.total, config=self.settings, iteration=self.iteration)
            wp.launch(mppi_baseline, dim=1, inputs=[self.loss.total, self.control.valid,
                      self.baseline, self.statistics], device=self.device)
            wp.launch(mppi_softmax, dim=self.settings.worlds, inputs=[self.loss.total,
                      self.control.valid, self.baseline, self.weights,
                      self.temperature / getattr(self, "cost_scale", 1.0)], device=self.device)
            wp.launch(mppi_normalize, dim=1, inputs=[self.weights, self.statistics], device=self.device)
            wp.launch(mppi_update, dim=self.mean.shape, inputs=[self.control.raw, self.weights,
                      self.statistics, self.mean, self.index.action_dim], device=self.device)
            wp.launch(optimizer._advance_iteration, dim=1, inputs=[self.iteration], device=self.device)

        def reset(self):
            self.rng = np.random.default_rng(self.seed)
            self.solve_count = 0
            self.mean.zero_()
            self.iteration.zero_()

        def _final_evaluate(self):
            wp.launch(mppi_final_candidates, dim=self.control.raw.shape,
                      inputs=[self.mean, self.control.best_raw, self.control.best_loss,
                              self.control.best_available, self.control.raw], device=self.device)
            self._project_controls()
            self._reset_candidate_status()
            self._evaluate_batch()
            self.control.best_raw.assign(self.control.raw)
            self.control.best_loss.assign(self.loss.total)
            self.control.best_available.assign(self.control.valid)
            self.control.best_grad_norm.zero_()

        def _evaluate_final_graph(self):
            if not wp.get_device(self.device).is_cuda:
                self._final_evaluate()
                return
            graph = getattr(self, "final_graph", None)
            if graph is None:
                # Warm all final-only kernels before capture. The warm pass
                # also supplies the first result; capture does not execute it.
                self._final_evaluate()
                wp.synchronize_device(self.device)
                with wp.ScopedCapture(device=self.device) as capture:
                    self._final_evaluate()
                self.final_graph = capture.graph
            else:
                wp.capture_launch(graph)

        def solve(self, *args, **kwargs):
            with wp.ScopedDevice(self.device):
                return self._solve(*args, **kwargs)

        def _solve(self, qpos, qvel, targets, *, warm, previous_action, allow_init, seed,
                   execution_steps, start_time):
            from time import perf_counter
            from forceaware.time_grid import validate_execution_steps

            if allow_init:
                raise ValueError("Panda FA MPPI keeps the measured initial state fixed")
            execution_steps = validate_execution_steps(execution_steps, self.settings.execution_substeps)
            warm = np.asarray(warm, np.float32).copy()
            warm[:self.index.action_dim] = 0.0
            arguments = dict(warm=warm, previous_action=previous_action, seed=seed, start_time=start_time)
            self._prepare_window(qpos, qvel, targets, **arguments)
            self.mean.assign(warm)
            self.random_seed.assign(np.asarray([self.rng.integers(0, 2 ** 30)], np.int32))
            if self._ensure_graph(allow_init=False):
                self._prepare_window(qpos, qvel, targets, **arguments)
                self.mean.assign(warm)
            self.iteration.zero_()
            iterations = self.mppi_init_iterations if self.solve_count == 0 else self.mppi_iterations
            started = perf_counter()
            for _ in range(iterations):
                wp.capture_launch(self.graphs[False])
            wp.synchronize()
            sampled_costs = self.loss.total.numpy().copy()
            weights = self.weights.numpy().copy()
            statistics = self.statistics.numpy().copy()
            # Evaluate the updated mean itself.  Other worlds retain finite
            # best candidates, providing a physical fallback if the mean fails.
            self._evaluate_final_graph()
            wp.synchronize()
            evaluation = self._evaluate_candidates(targets, execution_steps)
            # Contact rollouts are nonlinear: averaging successful trajectories
            # can lose contact, even when every control and state stays finite.
            # Evaluate and compare the mean and retained samples under the same
            # FA objective, including all control/state smoothing penalties.
            selected = self._select_candidate(evaluation)
            plan = self._plan(selected, started, evaluation=evaluation, execution_steps=execution_steps)
            plan.diag.update(mppi_weights=weights, sampled_costs=sampled_costs,
                             effective_samples=float(statistics[2]),
                             valid_samples=int(statistics[0]), mppi_iterations=iterations,
                             selected_mean=selected == 0, adjoint_used=False,
                             mean_cost=float(evaluation.scores[0]),
                             best_candidate_cost=float(evaluation.scores[selected]),
                             final_evaluation_graph_cached=getattr(self, "final_graph", None) is not None,
                             mppi_temperature=float(self.temperature),
                             mppi_cost_scale=float(getattr(self, "cost_scale", 1.0)),
                             mppi_effective_temperature=float(
                                 self.temperature / getattr(self, "cost_scale", 1.0)))
            self._commit_regularization_history()
            self.solve_count += 1
            return plan

    _CLASSES = SimpleNamespace(wp=wp, config=config, optimizer=optimizer,
                              mpc_loss=mpc_loss, PandaWindowOptimizer=PandaWindowOptimizer,
                              PandaMppiWindowOptimizer=PandaMppiWindowOptimizer,
                              PandaRegularizer=PandaRegularizer,
                              loss_gradient_scale=LOSS_GRADIENT_SCALE,
                              mppi_project_mean=mppi_project_mean, mppi_sample=mppi_sample,
                              project_offsets=project_offsets)
    _CLASSES.panda_pair_loss = panda_pair_loss
    _CLASSES.servo_implicit_damping = servo_implicit_damping
    return _CLASSES


def _panda_geometry(fast_step):
    """Preserve collision-source spheres, including analytic fingertips."""
    api = _classes()
    model = fast_step.cpu_model
    clouds = api.mpc_loss._collision_source_clouds(fast_step.compiled.base.collision)
    bodies, starts, ends, radii, sides = [], [], [], [], []
    import mujoco

    for body, spheres in sorted(clouds.items()):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
        side = 0 if name.startswith("left_") else 1 if name.startswith("right_") else -1
        if side < 0:
            continue
        # Preserve the source union of spheres. Collapsing an offset/thin
        # Panda wrist to one enclosing capsule creates fictitious overlap
        # even at the measured collision-free home configuration.
        for sphere in spheres:
            bodies.append(body)
            starts.append(sphere[:3])
            ends.append(sphere[:3])
            radii.append(sphere[3])
            sides.append(side)
    pairs = np.asarray([(a, b) for a, b in combinations(range(len(bodies)), 2)
                        if sides[a] != sides[b]], dtype=np.int32).reshape(-1, 2)
    if not len(pairs):
        raise ValueError("Panda ForceAware requires collision source geometry for both arms")
    return (np.asarray(bodies, np.int32), np.asarray(starts, np.float32).reshape(-1, 3),
            np.asarray(ends, np.float32).reshape(-1, 3), np.asarray(radii, np.float32), pairs)


def _object_contact_properties(model, index, collision):
    """FA grasp rows are object contacts; Panda also has inter-arm outputs."""
    rows = collision.output_contact_rows.numpy().astype(np.int32)
    layout = collision.contact_layout
    geometry = layout.contact_geom.numpy().astype(np.int32)
    pairs = model.geom_bodyid[geometry]
    source = collision.output_body_ids.numpy().astype(np.int32)
    sign = layout.contact_normal_sign.numpy().astype(np.float32)
    margin = layout.contact_includemargin.numpy().astype(np.float32)
    friction = layout.contact_friction.numpy().astype(np.float32)
    signs = np.zeros(rows.size, np.float32)
    margins = np.zeros(rows.size, np.float32)
    frictions = np.zeros((rows.size, 2), np.float32)
    for output in np.flatnonzero(rows >= 0):
        row = rows[output]
        if set(map(int, pairs[row])) == {int(source[output]), index.object_body}:
            signs[output], margins[output] = sign[row], margin[row]
            frictions[output] = friction[row, :2]
    return signs, margins, frictions


def _object_output_bodies(model, index, collision):
    # The same source body can have object and cross-arm outputs.  Preserve
    # original FA semantic mapping, excluding outputs for another target body.
    source = collision.output_body_ids.numpy().astype(np.int32)
    rows = collision.output_contact_rows.numpy().astype(np.int32)
    geometry = collision.contact_layout.contact_geom.numpy().astype(np.int32)
    pairs = model.geom_bodyid[geometry]
    result = source.copy()
    for output, row in enumerate(rows):
        if row < 0 or set(map(int, pairs[row])) != {int(source[output]), index.object_body}:
            result[output] = model.nbody + output
    return result


def _singleton(value, shape, name):
    result = np.asarray(value, np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {result.shape}")
    return result


class PandaForceAware:
    """Call DexForge's original FA optimizer with a length-one CA reference.

    Native contact-depth robustness remains active for both fingertips.  The
    current upstream loss does not use its legacy grasp-wrench quality field.
    """

    solver_backend = "dexforge_forceaware"

    def __init__(self, fast_step, *, horizon=5, iters=40, lr=0.0015,
                 u_limit=0.15, knot_substeps=6, weights=None, solver="adam",
                 num_samples=64, mppi_iterations=4, mppi_init_iterations=None,
                 temperature=0.01, noise_scale=0.02, noise_decay=0.85, seed=None,
                 cost_scale=None, adam_multistart_count=2):
        api = _classes()
        from forceaware.targets import ModelIndex
        from forceaware.time_grid import TimeGrid

        self.model = fast_step
        self.horizon = int(horizon)
        self.knot_substeps = int(knot_substeps)
        self.n_action = 14
        self.dense_steps = self.horizon * self.knot_substeps
        self.u_limit = float(u_limit)
        self.solver = str(solver).lower()
        if self.solver not in ("adam", "mppi"):
            raise ValueError("FA solver must be 'adam' or 'mppi'")
        if self.solver == "mppi":
            self.solver_backend = "dexforge_forceaware_mppi"
            num_samples, mppi_iterations = int(num_samples), int(mppi_iterations)
            mppi_init_iterations = mppi_iterations if mppi_init_iterations is None else int(mppi_init_iterations)
            if num_samples < 2 or min(mppi_iterations, mppi_init_iterations) < 1:
                raise ValueError("FA MPPI requires at least two samples and positive iteration counts")
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("FA MPPI temperature must be positive and finite")
            if cost_scale is None:
                from forceaware.numerics import LOSS_GRADIENT_SCALE

                # FA reports squared-millimeter losses.  Adam applies this
                # same global conversion to gradients.  Apply it only inside
                # MPPI's softmax; keep the original objective and reported cost.
                cost_scale = LOSS_GRADIENT_SCALE
            if not np.isfinite(cost_scale) or cost_scale <= 0:
                raise ValueError("FA MPPI cost_scale must be positive and finite")
            if not np.isfinite(noise_scale) or noise_scale < 0:
                raise ValueError("FA MPPI noise_scale must be finite and nonnegative radians")
            if not np.isfinite(noise_decay) or not 0 < noise_decay <= 1:
                raise ValueError("FA MPPI noise_decay must be in (0, 1]")
        adam_multistart_count = int(adam_multistart_count)
        if adam_multistart_count < 1:
            raise ValueError("FA Adam multistart count must be positive")
        if self.horizon < 1 or self.knot_substeps < 1 or int(iters) < 1:
            raise ValueError("FA horizon, knot_substeps and iterations must be positive")
        if not np.isfinite(self.u_limit) or self.u_limit <= 0 or not np.isfinite(lr) or lr <= 0:
            raise ValueError("FA u_limit and learning rate must be positive and finite")
        ids = fast_step._ids
        self.index = ModelIndex(
            np.concatenate((ids["left_qpos"], ids["right_qpos"])).astype(np.int32),
            np.concatenate((ids["left_act"], ids["right_act"])).astype(np.int32),
            ids["obj_qpos"], ids["obj_body"])
        self.query_body = np.asarray([ids["left_tip"], ids["right_tip"]], np.int32)
        self.tip_offsets = np.asarray([ids["left_tip_offset"], ids["right_tip_offset"]], np.float32)
        self.tip_radius = float(getattr(weights, "tip_sphere_radius", 0.01))
        base = api.config.load_config()
        dt = float(fast_step.cpu_model.opt.timestep)
        grid = TimeGrid.create(mpc_dt=dt, action_dt=dt * self.knot_substeps,
                               knot_dt=dt * self.knot_substeps, exec_dt=dt, ref_dt=dt)
        loss = api.config.LossWeights(**base["loss"])
        # The reference is now fourteen radians, not a virtual-wrist translation
        # followed by rotation and finger angles.
        loss = replace(loss, action_ref_base=loss.action_ref_finger,
                       action_ref_base_rot=loss.action_ref_finger)
        smoothing = api.config.SmoothingConfig(**base["smoothing"])
        opt = api.config.OptimizerConfig(**base["optimizer"])
        opt = replace(opt, horizon=self.horizon, n_iter=int(iters), lr_max=float(lr),
                      lr_min=min(float(lr), opt.lr_min),
                      multistart_count=num_samples if self.solver == "mppi" else adam_multistart_count,
                      multistart_init_std=0.0, multistart_ctrl_std=0.0)
        contact = api.config.ContactConfig(**base["contact"])
        if weights is not None:
            # Keep FA's object objective enabled: the old BigRasp loss defaults
            # to zero object weights and would discard x_plus_opt entirely.
            contact = replace(contact, target_phi=float(getattr(weights, "target_phi", contact.target_phi)),
                              penetration_limit=float(getattr(weights, "penetration_limit", contact.penetration_limit)))
        root = Path(api.config.ROOT)
        self.cfg = api.config.RetargetConfig(
            loss, smoothing, api.config.BoundsConfig(self.u_limit, self.u_limit, self.u_limit),
            opt, contact,
            api.config.SimConfig(grid, "comfree", root / "third_party/comfree_warp", 0.0, 1),
            api.config.SeqConfig(Path("."), "bimanual_panda", Path("."), Path("."), 0, 0, 0,
                                 grid.action_dt, ""), api.config.VideoConfig(1.0 / dt))
        geometry = _panda_geometry(fast_step)
        self.geometry = geometry
        if self.solver == "mppi":
            self.optimizer = api.PandaMppiWindowOptimizer(
                fast_step, self.index, self.cfg, geometry, iterations=mppi_iterations,
                init_iterations=mppi_init_iterations, temperature=float(temperature),
                noise_scale=float(noise_scale), noise_decay=float(noise_decay), seed=seed,
                cost_scale=float(cost_scale))
        else:
            self.optimizer = api.PandaWindowOptimizer(fast_step, self.index, self.cfg, geometry)
        joint_ids = np.asarray([np.flatnonzero(fast_step.cpu_model.jnt_qposadr == q)[0]
                                for q in self.index.hand_qpos])
        joint_limits = np.asarray(fast_step.cpu_model.jnt_range[joint_ids], np.float32)
        ctrl_limits = np.asarray(fast_step.cpu_model.actuator_ctrlrange[self.index.hand_ctrl], np.float32)
        joint_limits = np.where(fast_step.cpu_model.jnt_limited[joint_ids, None], joint_limits,
                                np.asarray([-np.inf, np.inf], np.float32))
        ctrl_limits = np.where(fast_step.cpu_model.actuator_ctrllimited[self.index.hand_ctrl, None],
                               ctrl_limits, np.asarray([-np.inf, np.inf], np.float32))
        self.lower = np.maximum(joint_limits[:, 0], ctrl_limits[:, 0])
        self.upper = np.minimum(joint_limits[:, 1], ctrl_limits[:, 1])
        self._warm = np.zeros(14 * (self.horizon + 1), np.float32)
        self._warm_absolute = None
        self._previous_action = None

    def reset(self):
        self._warm.fill(0.0)
        self._warm_absolute = None
        self._previous_action = None
        self.optimizer._regularization_previous_action = None
        self.optimizer.objective.regularizer._previous_previous_cpu = None
        if self.solver == "mppi":
            self.optimizer.reset()

    def solve(self, qpos0, qvel0, *, hand_reference, object_pose_wxyz,
              contact_points_local, normals_local, query_local=None):
        """Return a joint increment; pose input is ``[xyz, qw, qx, qy, qz]``."""
        api = _classes()
        import mujoco
        from forceaware.contact_mapping import map_guidance_to_gs_outputs
        from forceaware.control import resample_warm_corrections
        from forceaware.targets import ContactGuidance, SequenceData, make_window

        qpos = np.asarray(qpos0, np.float32).reshape(-1)
        qvel = np.asarray(qvel0, np.float32).reshape(-1)
        if qpos.shape != (self.model.nq,) or qvel.shape != (self.model.nv,) or not (
                np.isfinite(qpos).all() and np.isfinite(qvel).all()):
            raise ValueError("FA requires finite full-model qpos and qvel")
        hand = _singleton(hand_reference, (1, 14), "hand_reference")
        pose = _singleton(object_pose_wxyz, (1, 7), "object_pose_wxyz").copy()
        points = _singleton(contact_points_local, (1, 2, 3), "contact_points_local")
        normals = _singleton(normals_local, (1, 2, 3), "normals_local")
        lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
        if np.any(lengths < 1.e-8):
            raise ValueError("FA contact normals must be nonzero")
        normals = normals / lengths
        quat_length = np.linalg.norm(pose[:, 3:7], axis=1, keepdims=True)
        if np.any(quat_length < 1.e-8):
            raise ValueError("FA object quaternion must be nonzero")
        pose[:, 3:7] /= quat_length
        if query_local is None:
            data = mujoco.MjData(self.model.cpu_model)
            data.qpos[:] = qpos
            data.qpos[self.index.hand_qpos] = hand[0]
            mujoco.mj_forward(self.model.cpu_model, data)
            rotation = np.empty(9, np.float64)
            mujoco.mju_quat2Mat(rotation, pose[0, 3:7].astype(np.float64))
            inward_world = normals[0] @ rotation.reshape(3, 3).T
            local = self.tip_offsets.copy()
            for i, body in enumerate(self.query_body):
                local[i] += data.xmat[body].reshape(3, 3).T @ inward_world[i] * self.tip_radius
            query_local = local[None]
        query_local = _singleton(query_local, (1, 2, 3), "query_local")
        mask = np.ones((1, 2), np.float32)
        bodies = self.query_body[None]
        collision = self.model.compiled.base.collision
        output = map_guidance_to_gs_outputs(
            _object_output_bodies(self.model.cpu_model, self.index, collision), bodies, mask)
        if np.any(collision.output_contact_rows.numpy()[output] < 0):
            raise ValueError("FA fingertip guidance must map to enabled object collision outputs")
        guidance = ContactGuidance(mask, mask.copy(), points, output, bodies, query_local)
        # Original FA uses [qx,qy,qz,qw,tx,ty,tz], unlike the planner's pose.
        xyzw = pose[:, [4, 5, 6, 3, 0, 1, 2]]
        sequence = SequenceData(hand, xyzw, guidance, hand_source="lambda_contactaware")
        targets = make_window(sequence, executed_steps=0, horizon=self.horizon,
                              time_grid=self.cfg.simulator.time_grid, start_frame=0.0,
                              end_frame=0, hold_frame=0)
        current = qpos[self.index.hand_qpos]
        reach = self.u_limit * np.arange(1, self.horizon + 1, dtype=np.float32)[:, None]
        self.optimizer.action_lower.assign(np.maximum(self.lower, current[None] - reach))
        self.optimizer.action_upper.assign(np.minimum(self.upper, current[None] + reach))
        previous = current if self._previous_action is None else self._previous_action
        # Keep the executed command boundary even when the measured joint lags
        # behind it.  Replacing this endpoint creates a setpoint discontinuity.
        warm = self._warm.copy()
        if self._warm_absolute is not None:
            # CA solves a new singleton joint reference each cycle.  Warm
            # corrections must be expressed relative to this new reference.
            warm[14:] = (self._warm_absolute - targets.actions).reshape(-1)
        solve_started = perf_counter()
        plan = self.optimizer.solve(qpos, qvel, targets, warm=warm,
                                    previous_action=previous, allow_init=False, seed=0,
                                    execution_steps=self.knot_substeps, start_time=0.0)
        fa_solve_seconds = perf_counter() - solve_started
        self._warm = resample_warm_corrections(
            plan.raw, action_dim=14, horizon=self.horizon,
            executed_dense_steps=self.knot_substeps,
            physics_steps_per_knot=self.knot_substeps, warm_scale=1.0)
        absolute = np.zeros_like(plan.raw)
        absolute[14:] = (targets.actions + plan.raw[14:].reshape(self.horizon, 14)).reshape(-1)
        shifted_absolute = resample_warm_corrections(
            absolute, action_dim=14, horizon=self.horizon,
            executed_dense_steps=self.knot_substeps,
            physics_steps_per_knot=self.knot_substeps, warm_scale=1.0)
        self._warm_absolute = shifted_absolute[14:].reshape(self.horizon, 14).copy()
        self._previous_action = plan.action.copy()
        action = plan.action - current
        selected = int(plan.diag["selected"])
        contact_rows = collision.output_contact_rows.numpy()[output[0]]
        row_contacts = self.model.compiled.base.contact_rows.parameters.row_contact.numpy()
        contact_trace, force_trace = [], []
        for runtime in self.optimizer.steps[:self.knot_substeps]:
            contact_state = runtime.workspace.forward.base.collision.contacts
            active = contact_state.active.numpy()[selected, contact_rows] > 0
            forces = runtime.workspace.forward.contact.output.force.numpy()[selected]
            normal = np.asarray([forces[np.flatnonzero(row_contacts == row)].sum()
                                 for row in contact_rows], np.float32)
            contact_trace.append(active)
            force_trace.append(np.maximum(normal, 0.0))
        cost_components = {}
        loss_runtime = getattr(self.optimizer, "loss", None)
        regularization = getattr(self.optimizer.objective.regularizer, "runtime", None)
        if loss_runtime is not None and regularization is not None:
            terms = loss_runtime.trajectory_terms.numpy()[selected]
            components = regularization.components.numpy()[selected]
            names = api.mpc_loss.REGULARIZATION_TERM_NAMES
            cost_components = dict(zip(names, map(float, components)))
            cost_components["trajectory"] = float(np.sum(terms, dtype=np.float64))
            cost_components["total"] = float(plan.loss)
            breakdown = getattr(self.optimizer.objective, "cost_breakdown", None)
            if callable(breakdown):
                cost_components.update(breakdown(selected))
        return dict(action=action, ctrl=plan.action.copy(), cost=plan.loss, cost_opt=plan.loss,
                    solve_time=fa_solve_seconds, solver_backend=self.solver_backend,
                    sol_guess={"raw": plan.raw.copy()}, u_traj=plan.control_sequence - current,
                    control_sequence=plan.control_sequence.copy(), rollout_q=plan.qpos_sequence.copy(),
                    contact_mask=np.asarray(contact_trace), normal_force=np.asarray(force_trace),
                    diagnostics={**plan.diag, "grasp_quality_enabled": False,
                                 "contact_model": "asu_comfree",
                                 "comfree_stiffness": getattr(self.model, "comfree_stiffness", .2),
                                 "comfree_damping": getattr(self.model, "comfree_damping", .001),
                                 "contact_iterations": 0,
                                 "fa_solve_time": fa_solve_seconds,
                                 "iteration_time": plan.seconds,
                                 "dense_steps": self.dense_steps,
                                 "worlds": self.cfg.optimizer.multistart_count,
                                 "contact_batches": len(getattr(collision, "batches", ())),
                                 "contact_count": getattr(collision, "contact_count", None),
                                 "implicit_servo_damping": True,
                                 "parallel_interarm_pair_count": len(self.geometry[-1]) if self.geometry is not None else 0,
                                 "gws_loss_enabled": False,
                                 "contact_depth_robustness_enabled": self.cfg.loss.contact_robustness > 0,
                                 "reference_frames": 1, "object_drift": plan.drift,
                                 "gradient_norm": plan.grad_norm,
                                 "cost_components": cost_components},
                    hand_reference=hand.copy(), object_reference=pose.copy(),
                    query_local=query_local.copy())
