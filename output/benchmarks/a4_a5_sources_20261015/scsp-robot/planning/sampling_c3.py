"""Isaac adapter for the compiled dairlib Sampling C3 controller.

The plan is ``SamplingC3Controller::ComputePlan``. Contact linearization is
``LCSFactory::LinearizePlantToLCS`` on the jacktoy plant that ships with
``sampling_based_c3_public``. This module only maps the Isaac object pose and
fingertip into that plant's frame and publishes the controller's end-effector
trajectory on the existing via/action bus.
"""

from __future__ import annotations

import time

import numpy as np

# dairlib welds the table so its top is at z = -0.029.
DAIRLIB_GROUND_Z = -0.029
# examples/sampling_c3/jacktoy/parameters/goal_params.yaml
EE_TARGET_Z_OFFSET = 0.06
# examples/sampling_c3/jacktoy/parameters/sampling_c3_options.yaml
WORKSPACE_LIMITS = (
    (np.array([1.0, 0.0, 0.0]), 0.15, 0.75),
    (np.array([0.0, 1.0, 0.0]), -0.6, 0.6),
    (np.array([0.0, 0.0, 1.0]), -0.01, 0.3),
)
RADIUS_LIMITS = (0.25, 0.75)
WORKSPACE_MARGIN = 0.02


def config_from_args(args, trial_count=0):
    return {
        "table_height": float(getattr(args, "table_height", 0.35) or 0.35),
        "trial": int(trial_count),
    }


class SamplingC3Controller:
    """Owns one compiled dairlib controller for a trial."""

    def __init__(self, config=None):
        import dairlib_sampling_c3

        self._impl = dairlib_sampling_c3.CompiledSamplingC3()
        self.n_x = int(self._impl.n_x)
        self.table_height = float((config or {}).get("table_height", 0.35))
        self.z_offset = DAIRLIB_GROUND_Z - self.table_height
        self.xy_offset = None
        self._last_tip = None
        self._last_time = None

    def step(self, payload, target_p, target_q, reach_distance):
        from planning.franka_contact_motion import bounded_step

        started = time.perf_counter()
        state = np.asarray(payload["policy_q"], dtype=np.float64).reshape(-1)
        tip = state[7:10].copy()
        obj_p = state[0:3].copy()
        obj_q = _unit_quat(state[3:7])
        target_p = np.asarray(target_p, dtype=np.float64).reshape(3)
        target_q = _unit_quat(target_q)
        self._lock_xy_offset(tip)

        x = np.zeros(self.n_x, dtype=np.float64)
        x[0:3] = self._to_dairlib(tip)
        x[3:7] = obj_q
        x[7:10] = self._to_dairlib(obj_p)
        x[10:13] = self._tip_velocity(tip, payload.get("sim_time"))
        linear = np.asarray(
            payload.get("object_linear_velocity", np.zeros(3)), dtype=np.float64
        ).reshape(3)
        angular = np.asarray(
            payload.get("object_angular_velocity", np.zeros(3)), dtype=np.float64
        ).reshape(3)
        x[13:16] = angular
        x[16:19] = linear
        _require_workspace(x[0:3])

        desired = np.zeros(self.n_x, dtype=np.float64)
        desired[3:7] = target_q
        desired[7:10] = self._to_dairlib(target_p)
        desired[0:3] = desired[7:10].copy()
        desired[2] += EE_TARGET_Z_OFFSET
        _require_workspace(desired[0:3])

        stamp = float(payload.get("sim_time", 0.0) or 0.0)
        plan = self._impl.step(x, desired, desired.copy(), stamp)
        ee = np.asarray(plan["ee"], dtype=np.float64)
        times = np.asarray(plan["time"], dtype=np.float64).reshape(-1)
        target = self._to_isaac(_sample_trajectory(ee, times, 0.02))
        action = bounded_step(tip, target, reach_distance)
        is_c3 = bool(plan["is_c3"])
        elapsed = time.perf_counter() - started
        print(
            "sampling_c3:",
            "c3" if is_c3 else "reposition",
            "via:",
            np.round(target, 4).tolist(),
            "release:",
            int(not is_c3),
            flush=True,
        )
        return _result(action, target, is_c3, elapsed)


    def _lock_xy_offset(self, tip):
        if self.xy_offset is not None:
            return
        radius = float(np.linalg.norm(tip[:2]))
        if radius < RADIUS_LIMITS[0] + 0.05 or radius > RADIUS_LIMITS[1] - 0.05:
            direction = tip[:2] / max(radius, 1e-9)
            self.xy_offset = 0.50 * direction - tip[:2]
        else:
            self.xy_offset = np.zeros(2, dtype=np.float64)

    def _to_dairlib(self, point):
        mapped = np.asarray(point, dtype=np.float64).reshape(3).copy()
        mapped[:2] += self.xy_offset
        mapped[2] += self.z_offset
        return mapped

    def _to_isaac(self, point):
        mapped = np.asarray(point, dtype=np.float64).reshape(3).copy()
        mapped[:2] -= self.xy_offset
        mapped[2] -= self.z_offset
        return mapped

    def _tip_velocity(self, tip, sim_time):
        velocity = np.zeros(3, dtype=np.float64)
        if self._last_tip is not None and sim_time is not None and self._last_time is not None:
            dt = float(sim_time) - float(self._last_time)
            if dt > 1e-6:
                velocity = (np.asarray(tip, dtype=np.float64) - self._last_tip) / dt
        self._last_tip = np.asarray(tip, dtype=np.float64).copy()
        self._last_time = None if sim_time is None else float(sim_time)
        return velocity


def sampling_c3_step(controller, args, param, mpc, trackers, payload, reach_distance, pos_err, quat_err):
    del args, mpc, trackers, pos_err, quat_err
    return controller.step(payload, param.target_p_, param.target_q_, reach_distance)


def _unit_quat(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def _require_workspace(point):
    point = np.asarray(point, dtype=np.float64).reshape(3)
    for normal, lower, upper in WORKSPACE_LIMITS:
        projection = float(normal @ point)
        if projection <= lower - WORKSPACE_MARGIN or projection >= upper + WORKSPACE_MARGIN:
            raise RuntimeError(f"fingertip {point.tolist()} is outside the dairlib workspace")
    radius = float(np.linalg.norm(point[:2]))
    if radius <= RADIUS_LIMITS[0] + WORKSPACE_MARGIN or radius >= RADIUS_LIMITS[1] - WORKSPACE_MARGIN:
        raise RuntimeError(f"fingertip radius {radius:.3f} is outside the dairlib workspace")


def _sample_trajectory(ee, times, horizon):
    if ee.ndim != 2 or ee.shape[0] != 3 or ee.shape[1] == 0:
        raise RuntimeError("compiled controller returned an empty end-effector trajectory")
    if ee.shape[1] == 1 or times.size < 2:
        return ee[:, 0]
    dt = float(times[1] - times[0])
    if dt <= 1e-9:
        return ee[:, 1]
    alpha = min(1.0, max(0.0, float(horizon) / dt))
    return (1.0 - alpha) * ee[:, 0] + alpha * ee[:, 1]


def _result(action, target, is_c3, elapsed):
    target = np.asarray(target, dtype=np.float64).reshape(3)
    policy = {
        "min_error": 0.0,
        "max_error": 0.0,
        "error": 0.0,
        "cached_cost": 0.0,
        "choose_dt": float(elapsed),
        "mpc_virtual_point": target,
        "mpc_contact_point": target,
        "p_arm_world": target,
        "best_contact_world": target,
        "best_contact_idx": None,
        "value_info": {
            "accept_p_arm": False,
            "path_blocked": False,
            "via_phase": "c3" if is_c3 else "reposition",
            "occupied_idx": None,
        },
        "escape_on": False,
        "release_pressure": not is_c3,
        "preserve_upright": False,
        "execution_override": True,
        "x_plus_opt": None,
        "info": None,
    }
    return {
        "action": np.asarray(action, dtype=np.float64).reshape(3),
        "policy": policy,
        "verify_cost": 0.0,
        "verify_chatter": False,
        "if_contact": bool(is_c3),
        "opt_snapshot": None,
        "model_tightness": 0.0,
        "model_accum": 0.0,
        "rank_dt": float(elapsed),
        "plan_dt": float(elapsed),
        "mppi_dt": 0.0,
        "cost_opt": 0.0,
        "solver_backend": "dairlib-sampling-c3",
        "lambda_failures": 0,
        "mpc_failures": 0,
        "lambda_solves": 0,
        "mpc_solves": 0,
        "mpc_init_error": "",
    }
