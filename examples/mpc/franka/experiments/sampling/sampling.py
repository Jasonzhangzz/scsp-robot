"""Sampling loop around a local contact plan.

At every call the fingertip is imagined at several locations on a sphere
around the object. Each location gets its own linear complementarity model
and a local plan (compiled C3, CasADi MIQP, or acados MPC). The current
location is executed while its predicted cost remains competitive. A better
sample switches the output to a contact-free approach toward that sample,
which is the reposition stage in dairlib's sampling C3 controller.
"""

from __future__ import annotations

import numpy as np

from examples.mpc.franka.experiments.sampling import c3, miqp
from examples.mpc.franka.experiments.sampling.lcs import (
    EE,
    OBJ,
    desired_trajectory,
    pack_state,
    point_push_lcs,
    stage_costs,
    unit_normal,
)
from examples.mpc.franka.experiments.sampling.mpc import ContactMPC


class SamplingController:
    def __init__(self, method="c3", horizon=5, n_samples=6, dt=0.1,
                 radius=0.05, ee_mass=0.1, object_mass=0.2, travel_cost=0.0,
                 reposition_speed=0.20, sphere_radius=0.12, planning_dt=0.1,
                 reach_tolerance=0.01, use_relative_hysteresis=True,
                 c3_to_reposition_frac=0.6, reposition_to_c3_frac=0.5,
                 c3_to_reposition=0.0, reposition_to_c3=0.0,
                 u_limit=50.0, admm_iterations=3):
        if method not in ("c3", "miqp", "mpc"):
            raise ValueError("method must be c3, miqp, or mpc")
        self.method = method
        self.horizon = int(horizon)
        self.n_samples = int(n_samples)
        self.dt = float(dt)
        self.radius = float(radius)
        self.ee_mass = float(ee_mass)
        self.object_mass = float(object_mass)
        self.travel_cost = float(travel_cost)
        self.reposition_speed = float(reposition_speed)
        self.sphere_radius = float(sphere_radius)
        self.planning_dt = float(planning_dt)
        self.reach_tolerance = float(reach_tolerance)
        self.use_relative_hysteresis = bool(use_relative_hysteresis)
        self.c3_to_reposition_frac = float(c3_to_reposition_frac)
        self.reposition_to_c3_frac = float(reposition_to_c3_frac)
        self.c3_to_reposition = float(c3_to_reposition)
        self.reposition_to_c3 = float(reposition_to_c3)
        self.u_limit = float(u_limit)
        self.admm_iterations = int(admm_iterations)
        self.is_c3 = True
        self.reposition_target = None
        self._mpc = None

    def plan(self, ee_position, object_position, object_goal,
             ee_velocity=None, object_velocity=None):
        current = np.asarray(ee_position, dtype=float).reshape(3)
        obj = np.asarray(object_position, dtype=float).reshape(3)
        goal = np.asarray(object_goal, dtype=float).reshape(-1)
        if goal.size == 2:
            goal = np.array([goal[0], goal[1], obj[2]])
        goal = goal.reshape(3)
        measured = pack_state(current, obj, ee_velocity, object_velocity)
        samples = self._samples(current, obj)
        costs = []
        plans = []
        models = []
        for location in samples:
            state = measured.copy()
            state[EE] = location
            state[6:9] = 0.0
            model = point_push_lcs(
                location - obj, self.dt, self.ee_mass, self.object_mass, self.radius
            )
            # dairlib's x_lcs_des puts the end effector above the object goal.
            # The sample is only the linearization point, not the setpoint.
            desired = desired_trajectory(goal + np.array([0.0, 0.0, 0.06]), goal, self.horizon)
            plan = self._local_plan(model, state, desired)
            predicted = _predicted_cost(model, state, plan["u"], desired, self.horizon)
            predicted += self.travel_cost * float(np.linalg.norm(location - current))
            costs.append(predicted)
            plans.append(plan)
            models.append(model)
        self._select_mode(samples, costs, current)
        if self.is_c3:
            ee = plans[0]["x"][:, :3]
            force = plans[0]["u"]
            mode = self.method
            executed = plans[0]
        else:
            ee = _spherical_reposition(
                current, obj, self.reposition_target, self.horizon + 1,
                self.reposition_speed, self.sphere_radius, self.planning_dt,
            )
            force = np.zeros((self.horizon, 3))
            mode = "reposition"
            executed = None
        return {
            "mode": mode,
            "is_c3": self.is_c3,
            "ee": ee,
            "force": force,
            "ee_command": ee[min(1, len(ee) - 1)],
            "force_command": force[0],
            "sample_locations": samples,
            "sample_costs": costs,
            "reposition_target": None if self.reposition_target is None else self.reposition_target.copy(),
            "plan": executed,
            "normal": unit_normal(current - obj),
        }

    def _samples(self, current, obj):
        samples = [current.copy()]
        if not self.is_c3 and self.reposition_target is not None:
            samples.append(self.reposition_target.copy())
        z_min = float(obj[2]) - 0.25 * self.radius
        golden = np.pi * (3.0 - np.sqrt(5.0))
        generated = 0
        while len(samples) < 1 + self.n_samples and generated < self.n_samples * 8:
            fraction = generated / max(self.n_samples * 4, 1)
            height = 1.0 - 2.0 * fraction
            radial = np.sqrt(max(0.0, 1.0 - height * height))
            theta = golden * generated
            offset = np.array([np.cos(theta) * radial, np.sin(theta) * radial, height])
            generated += 1
            point = obj + self.radius * offset
            if point[2] < z_min:
                continue
            if min(np.linalg.norm(point - previous) for previous in samples) < 1e-4:
                continue
            samples.append(point)
        return samples

    def _local_plan(self, model, state, desired):
        costs = stage_costs(self.horizon)
        if self.method == "c3":
            return c3.solve(
                model, costs, state, desired,
                admm_iterations=self.admm_iterations,
                u_lower=-self.u_limit, u_upper=self.u_limit,
            )
        if self.method == "miqp":
            return miqp.solve(
                model, costs, state, desired,
                admm_iterations=self.admm_iterations,
                u_lower=-self.u_limit, u_upper=self.u_limit,
            )
        if self._mpc is None:
            self._mpc = ContactMPC(
                self.horizon, self.dt, self.ee_mass, self.object_mass, self.radius,
                self.u_limit,
            )
        return self._mpc.solve(state, desired)

    def _select_mode(self, samples, costs, current):
        if len(costs) == 1:
            self.is_c3 = True
            return
        others = list(enumerate(costs[1:], start=1))
        best_index, best_cost = min(others, key=lambda item: item[1])
        current_cost = costs[0]
        if self.is_c3:
            if _worth_switching(current_cost, best_cost, self.use_relative_hysteresis,
                                self.c3_to_reposition_frac, self.c3_to_reposition):
                self.is_c3 = False
                self.reposition_target = samples[best_index].copy()
            return
        if np.linalg.norm(current - self.reposition_target) <= self.reach_tolerance:
            self.is_c3 = True
            self.reposition_target = None
            return
        tracked = costs[1]
        if _worth_switching(best_cost, current_cost, self.use_relative_hysteresis,
                            self.reposition_to_c3_frac, self.reposition_to_c3):
            self.is_c3 = True
            self.reposition_target = None
        elif best_index != 1 and _worth_switching(
                tracked, best_cost, self.use_relative_hysteresis,
                self.c3_to_reposition_frac, self.c3_to_reposition):
            self.reposition_target = samples[best_index].copy()


def _predicted_cost(model, state, controls, desired, horizon):
    costs = stage_costs(horizon)
    Q, R = costs[0], costs[1]
    total = 0.0
    for t in range(horizon):
        error = state - desired[t]
        total += float(error @ Q[t] @ error)
        total += float(controls[t] @ R[t] @ controls[t])
        state, _force = model.simulate(state, controls[t])
    error = state - desired[-1]
    total += float(error @ Q[-1] @ error)
    return total


def _worth_switching(incumbent, challenger, relative, fraction, absolute):
    """dairlib switches only when the challenger beats the incumbent by the hysteresis."""
    incumbent = float(incumbent)
    challenger = float(challenger)
    if relative:
        return incumbent > challenger + fraction * incumbent
    return challenger + absolute < incumbent


def _spherical_reposition(start, center, target, count, speed, radius, dt):
    """Jacktoy reposition: out to a sphere, along the arc, then in. Knot spacing is speed*dt."""
    start = np.asarray(start, dtype=float).reshape(3)
    center = np.asarray(center, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)
    step = float(speed) * float(dt)
    points = [start.copy()]
    v1 = start - center
    v2 = target - center
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9 or step <= 0.0:
        return _straight_reposition(start, target, count, step)
    v1 = v1 / n1
    v2 = v2 / n2
    travel_angle = float(np.arccos(np.clip(np.dot(v1, v2), -0.9999, 0.9999)))
    if travel_angle < 0.3:
        return _straight_reposition(start, target, count, step)
    waypoint1 = center + radius * v1
    waypoint2 = center + radius * v2
    axis = np.cross(v1, v2)
    axis_norm = float(np.linalg.norm(axis))
    tangent = np.cross(axis / max(axis_norm, 1e-9), v1)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1e-9:
        return _straight_reposition(start, target, count, step)
    tangent = tangent / tangent_norm
    if tangent[2] < -0.5 and travel_angle > np.pi / 2:
        tangent = -tangent
        travel_angle = 2 * np.pi - travel_angle
    dtheta = step / max(radius, 1e-9)
    cursor = start.copy()
    remaining = [(_segment(cursor, waypoint1), "line", waypoint1),
                 (travel_angle * radius, "arc", waypoint2),
                 (_segment(waypoint2, target), "line", target)]
    arc_progress = 0.0
    for _ in range(count - 1):
        advance = step
        while advance > 1e-9 and remaining:
            length, kind, endpoint = remaining[0]
            if length <= 1e-9:
                remaining.pop(0)
                continue
            take = min(advance, length)
            if kind == "line":
                direction = endpoint - cursor
                distance = float(np.linalg.norm(direction))
                if distance > 1e-9:
                    cursor = cursor + direction * (take / distance)
            else:
                arc_progress += take / max(radius, 1e-9)
                cursor = center + radius * (
                    np.cos(arc_progress) * v1 + np.sin(arc_progress) * tangent
                )
            remaining[0] = (length - take, kind, endpoint)
            advance -= take
        points.append(cursor.copy())
    return np.vstack(points)


def _segment(start, end):
    return float(np.linalg.norm(np.asarray(end) - np.asarray(start)))


def _straight_reposition(start, target, count, step):
    points = [np.asarray(start, dtype=float).reshape(3)]
    target = np.asarray(target, dtype=float).reshape(3)
    for _ in range(count - 1):
        delta = target - points[-1]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9 or step <= 0.0:
            points.append(points[-1].copy())
        else:
            points.append(points[-1] + delta * min(1.0, step / distance))
    return np.vstack(points)
