"""Sampling C3 with actual rigid-object geometry and a pose objective."""

import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.transform import Rotation
from collections import deque

from . import scip_miqp
from .rigid import EE, OBJ, QUAT, orientation_error, rotation, stage_costs, unit_quat
from .sampling import _spherical_reposition


class RigidSamplingController:
    def __init__(self, scene, *, horizon=5, n_samples=2, dt=.05,
                 admm_iterations=3, seed=0, command_dt=.02, projection_time_limit=.1):
        self.scene = scene
        self.horizon, self.n_samples = int(horizon), int(n_samples)
        self.dt, self.command_dt = float(dt), float(command_dt)
        self.admm_iterations = int(admm_iterations)
        self.projection_time_limit = float(projection_time_limit)
        self.rng = np.random.default_rng(seed)
        self.reposition_target = None
        self.is_c3 = True
        self._last_rotation_axis = np.zeros(3)
        self._progress = deque(maxlen=13)
        # SCIP and Isaac/PhysX load conflicting native solver libraries in
        # this environment. A persistent spawn worker keeps SCIP independent
        # of the simulator, and also avoids importing Torch in the planner.
        self._solver = ProcessPoolExecutor(max_workers=min(4, max(1, self.n_samples+2)),
                                           mp_context=multiprocessing.get_context('spawn'))

    def close(self):
        self._solver.shutdown(wait=True)

    def _samples(self, x):
        samples = [x[EE].copy()]
        if not self.is_c3 and self.reposition_target is not None:
            samples.append(self.reposition_target.copy())
        R = rotation(x[QUAT])
        phase = self.rng.uniform(-np.pi, np.pi)
        for i in range(self.n_samples):
            # Native jack samples the full capped spherical shell, including
            # low side contacts; fixed latitude bands can miss a flip pivot.
            height = self.rng.uniform(np.cos(1.95), np.cos(.2))
            theta = phase + i * np.pi * (3 - np.sqrt(5))
            direction = np.array([np.sqrt(1-height**2)*np.cos(theta),
                                  np.sqrt(1-height**2)*np.sin(theta), height])
            local = R.T @ direction * (self.scene.radius + .1)
            candidates = []
            for part in range(len(self.scene.parts)):
                _, point, normal = self.scene.surface(local, part)
                candidate = x[OBJ] + R @ (point + normal * (self.scene.tip_radius + .008))
                if candidate[2] < self.scene.table_height + self.scene.tip_radius + .002:
                    continue
                probe = x.copy()
                probe[EE] = candidate
                if min(c.gap for c in self.scene.contacts(probe) if c.kind == 'finger') < .004:
                    continue
                candidates.append(candidate)
            if candidates:
                samples.append(min(candidates, key=lambda p: np.linalg.norm(p-(x[OBJ]+direction*self.scene.radius))))
        return samples

    def plan(self, measured, target_p, target_q):
        x = np.asarray(measured, dtype=float).copy()
        target_q = unit_quat(target_q)
        if x[QUAT] @ target_q < 0:
            target_q = -target_q
        final_target_q = target_q.copy()
        # Same short lookahead as jack's goal generator: at most 2 rad and
        # 15 cm. The EE reference is above the CURRENT object, not the final
        # translated object location.
        current_rotation = Rotation.from_quat(x[QUAT][[1, 2, 3, 0]])
        error_rotation = Rotation.from_quat(target_q[[1, 2, 3, 0]]) * current_rotation.inv()
        vector = error_rotation.as_rotvec()
        angle = float(np.linalg.norm(vector))
        if angle > 1e-10:
            axis = vector/angle
            if np.pi-angle < .4 and axis @ self._last_rotation_axis < 0:
                axis, angle = -axis, 2*np.pi-angle
            self._last_rotation_axis = axis
            target_q = (Rotation.from_rotvec(axis*min(angle, 2.))*current_rotation).as_quat()[[3, 0, 1, 2]]
        if target_q @ x[QUAT] < 0:
            target_q = -target_q
        desired = np.zeros_like(x)
        delta = np.asarray(target_p)-x[OBJ]
        desired[QUAT] = target_q
        desired[OBJ] = x[OBJ] + delta * min(1., .15/max(np.linalg.norm(delta), 1e-9))
        desired[EE] = x[OBJ] + [0., 0., .06]
        samples = self._samples(x)
        plans, costs, failures, jobs = [], [], [], []
        for index, location in enumerate(samples):
            state = x.copy()
            state[EE] = location
            # Current candidate keeps its measured velocity. Relocated
            # candidates start at rest, as in dairlib's contact-free stage.
            if index > 0:
                state[10:13] = 0.
            model = self.scene.linearize(state, self.dt, for_planning=True)
            weights = stage_costs(self.horizon, model.m, target_q, current_quaternion=x[QUAT])
            future = self._solver.submit(scip_miqp.solve, model, weights, state, desired,
                                         admm_iterations=self.admm_iterations,
                                         projection_time_limit=self.projection_time_limit)
            jobs.append((state, model, weights, future))
        for state, model, weights, future in jobs:
            try:
                local = future.result()
                reference = local['x'].copy()
                rollout, total, forces = [state], 0., []
                # Score complementary forward dynamics, not the relaxed
                # ADMM state/force trajectory. Re-query contacts after rotation.
                for t, control in enumerate(local['u']):
                    # Jack cost_type=3 emulates its position/velocity PD
                    # tracker as well as actuator force feedforward.
                    applied = (control + 100.*(reference[t, EE]-state[EE])
                               + .5*(reference[t, 10:13]-state[10:13]))
                    error = state - desired
                    total += float(error @ weights[0][t] @ error + applied @ weights[1][t] @ applied)
                    physical = self.scene.linearize(state, self.dt)
                    state, force = physical.simulate(state, applied)
                    rollout.append(state)
                    forces.append(force)
                error = state - desired
                total += float(error @ weights[0][-1] @ error)
                local['x'] = np.asarray(rollout)
                local['nominal_x'] = reference
                local['physical_lambda'] = forces
                local['contact_count'] = len(model.contacts)
                plans.append(local)
                costs.append(total)
            except RuntimeError as error:
                plans.append(None)
                costs.append(float('inf'))
                failures.append(str(error))
        valid = [i for i, p in enumerate(plans) if p is not None]
        if not valid:
            raise RuntimeError("all rigid C3 candidates failed: " + '; '.join(failures))
        best = min(valid, key=lambda i: costs[i])
        others = [i for i in valid if i != 0]
        best_other = min(others, key=lambda i: costs[i]) if others else None
        configuration_cost = 9000.*np.sum((x[OBJ]-target_p)**2) + 2500.*orientation_error(x[QUAT], final_target_q)**2
        # A candidate must improve the complete pose rollout to pay for a
        # contact-free reposition. Keep the previous target while travelling.
        if self.is_c3:
            self._progress.append(configuration_cost)
            # Native jack's pose-mode switch requires an 85% cost gain, or
            # insufficient configuration-cost progress over 12 iterations.
            stagnant = (len(self._progress) == 13 and
                        configuration_cost > .982*self._progress[0])
            if best_other is not None and (plans[0] is None or stagnant or costs[best_other] < .15*costs[0]):
                self.is_c3 = False
                self.reposition_target = samples[best_other].copy()
                self._progress.clear()
        else:
            distance = np.linalg.norm(x[EE] - self.reposition_target)
            # A near tie must not cancel an unfinished reposition on the
            # next cycle; native jack uses a separate, strong return margin.
            tracked_cost = costs[1] if len(costs) > 1 else float('inf')
            if plans[0] is not None and (distance < .006 or costs[0] < .7 * tracked_cost):
                self.is_c3, self.reposition_target = True, None
                self._progress.clear()
            # A separate destination latch prevents approximate MIQP cost
            # noise from moving the target before the arm arrives.
            elif best > 1 and costs[best] < .4 * costs[1]:
                self.reposition_target = samples[best].copy()
        if self.is_c3:
            plan = plans[0]
            next_tip = plan['nominal_x'][1, EE]
            alpha = min(1., self.command_dt / self.dt)
            command = x[EE] + alpha * (next_tip - x[EE])
            # UpdateC3ExecutionTrajectory uses GetInputSolution (the final
            # dynamics QP), not the projected debug/visualization vector.
            actuator = plan['u'][0]
            mode = 'c3'
        else:
            plan = None
            route = _spherical_reposition(x[EE], x[OBJ], self.reposition_target, 2,
                                          .20, max(.12, self.scene.radius + .02), self.command_dt)
            command, actuator, mode = route[1], np.zeros(3), 'reposition'
        return dict(ee_command=command, force_command=actuator, mode=mode,
                    is_c3=self.is_c3, plan=plan, sample_locations=samples,
                    sample_costs=costs, candidate_failures=failures,
                    candidate_projection_statistics=[[] if p is None else p['projection_statistics'] for p in plans],
                    orientation_error=orientation_error(x[QUAT], final_target_q),
                    position_error=float(np.linalg.norm(x[OBJ]-target_p)))
