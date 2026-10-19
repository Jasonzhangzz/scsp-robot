"""Isaac observations and actor properties for the rigid C3MIQP planner.

Configure geometry/inertia after Isaac creates the actor. Each command tracks
20 ms of a 50 ms pose-planning knot, with a 4 mm execution limit.
"""

from __future__ import annotations

import time

import numpy as np

from examples.mpc.franka.experiments.sampling.rigid import RigidScene, pack_state
from examples.mpc.franka.experiments.sampling.rigid_sampling import RigidSamplingController

# Execute only 20 ms of a 50 ms pose-planning knot, at 0.20 m/s.
EXECUTION_STEP_M = 0.004


def config_from_args(args, trial_count=0):
    method = str(getattr(args, "sampling_method", "c3") or "c3")
    return {
        "method": method,
        "n_samples": int(getattr(args, "sampling_c3_samples", 2)),
        "trial": int(trial_count),
        "admm_iterations": int(getattr(args, 'sampling_admm_iterations', 3)),
        "projection_time_limit": float(getattr(args, 'sampling_projection_time_limit', .1)),
    }


class SamplingC3Controller:
    def __init__(self, config=None):
        self.config = config or {}
        if self.config.get('method', 'c3') not in ('c3', 'miqp'):
            raise ValueError('The rigid Isaac pose controller supports C3MIQP (--method c3 or miqp); '
                             'the standalone MIQP/acados examples use the 12-state point model.')
        self.controller = None
        self._scene_env = None

    def configure_scene(self, env):
        if env is self._scene_env:
            return
        if self.controller is not None:
            self.controller.close()
        from examples.mpc.franka.runtime.contact import fingertip_radius
        param = env.param_
        prop = env.gym.get_actor_rigid_body_properties(env.env, env.obj_actor)[0]
        inertia = prop.inertia
        matrix = np.array([[getattr(getattr(inertia, row), col) for col in ('x', 'y', 'z')]
                           for row in ('x', 'y', 'z')])
        scene = RigidScene(
            getattr(param, 'collision_mesh_paths_', None) or [param.mesh_path_],
            mass=float(prop.mass), inertia=matrix, table_height=param.table_height,
            com=np.array([prop.com.x, prop.com.y, prop.com.z]),
            tip_radius=fingertip_radius(env),
            mu_tip=float(param.mu_fingertip_object_), mu_table=float(param.mu_table_object_),
        )
        self.controller = RigidSamplingController(scene,
            n_samples=int(self.config.get('n_samples', 2)),
            admm_iterations=int(self.config.get('admm_iterations', 3)),
            projection_time_limit=float(self.config.get('projection_time_limit', .1)),
            seed=int(self.config.get('trial', 0)))
        self._scene_env = env
        print('[sampling-scene] projection=MIQP/SCIP rigid-body n_x=19 mass=%.6g inertia=%s parts=%d mu=(%.3g,%.3g) dt=.05 command_dt=.02' %
              (scene.mass, np.diag(matrix).tolist(), len(scene.parts), scene.mu_tip, scene.mu_table), flush=True)

    def close(self):
        if self.controller is not None:
            self.controller.close()

    def step(self, payload, target_p, target_q, reach_distance=EXECUTION_STEP_M):
        from planning.franka_contact_motion import bounded_step

        if self.controller is None:
            raise RuntimeError('configure_scene(env) must run before the first rigid C3 step')

        started = time.perf_counter()
        state = np.asarray(payload["policy_q"], dtype=np.float64).reshape(-1)
        tip = state[7:10]
        obj = state[0:3]
        measured = pack_state(tip, obj, state[3:7],
            payload.get('fingertip_linear_velocity'), payload.get('object_angular_velocity'),
            payload.get('object_linear_velocity'))
        result = self.controller.plan(measured, np.asarray(target_p, dtype=float), target_q)
        command = np.asarray(result["ee_command"], dtype=np.float64).reshape(3)
        action = bounded_step(tip, command, reach_distance if reach_distance is not None else EXECUTION_STEP_M)
        elapsed = time.perf_counter() - started
        print(
            "sampling:",
            result["mode"],
            "via:",
            np.round(command, 4).tolist(),
            'pose_error:', round(result['position_error'], 4), round(np.rad2deg(result['orientation_error']), 2),
            flush=True,
        )
        output = _result(action, command, result["is_c3"], result["mode"], elapsed)
        output['rigid_plan'] = result
        output['actuator_force_world'] = np.asarray(result.get('force_command', np.zeros(3)), dtype=float)
        output['cost_opt'] = min(result['sample_costs'])
        # A via 4 mm ahead is not a contact point: passing it as ``press``
        # falsely enables the near-contact force cap throughout free motion.
        # Use an actual geometry witness for contact C3 and no press in air.
        output['execution_press'] = None
        if result['is_c3']:
            contact = min((c for c in self.controller.scene.contacts(measured) if c.kind == 'finger'),
                          key=lambda c: c.gap)
            output['execution_press'] = contact.point + self.controller.scene.tip_radius * contact.normal
        # Reposition is an explicitly planned free trajectory. The shared
        # emergency pressure-release servo discards velocity feedforward and
        # must not replace tracking this trajectory on every reposition step.
        output['policy']['release_pressure'] = False
        return output


def sampling_c3_step(controller, args, param, mpc, trackers, payload, reach_distance, pos_err, quat_err):
    del args, mpc, trackers, pos_err, quat_err
    return controller.step(payload, param.target_p_, param.target_q_, reach_distance)


def _result(action, target, is_c3, mode, elapsed):
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
            "via_phase": mode,
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
        "solver_backend": "sampling-" + mode,
        "lambda_failures": 0,
        "mpc_failures": 0,
        "lambda_solves": 0,
        "mpc_solves": 0,
        "mpc_init_error": "",
    }
