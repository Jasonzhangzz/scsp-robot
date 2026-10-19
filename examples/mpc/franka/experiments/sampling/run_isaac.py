"""Run rigid-object sampling C3 in Isaac without the shared MPC entry.

This does not go through ``ik2/test_mpc_isaac.py`` or ``ik2/test_mpc_ablation.py``.
Those runners keep their own costs and the 5 mm Cartesian step. Here the
executed increment interpolates the next pose-planning knot, capped by
``EXECUTION_STEP_M``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(root))
    # Isaac Gym must be imported before Torch. runtime.isaac does that.
    from examples.mpc.franka.runtime.isaac import SynchronousMPCBus
    from examples.mpc.franka.runtime.mpc import (
        POLICY_INTERVAL,
        _add_rollout_policy_args,
        _apply_execution_mode,
    )
    from examples.mpc.franka.experiments.sampling.isaac import (
        EXECUTION_STEP_M,
        SamplingC3Controller,
        config_from_args,
    )

    parser = _add_rollout_policy_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--method", choices=("c3", "miqp"), default="c3",
                        help='Both names select rigid C3MIQP with SCIP.')
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument('--sampling-admm-iterations', type=int, default=3)
    parser.add_argument('--sampling-projection-time-limit', type=float, default=.1)
    args = parser.parse_args()
    if args.samples < 0 or args.sampling_admm_iterations < 1 or args.sampling_projection_time_limit <= 0:
        parser.error('samples must be nonnegative; ADMM iterations and MIQP time limit must be positive')
    args.sampling_method = args.method
    args.sampling_c3_samples = int(args.samples)
    args.sampling_c3 = False
    _apply_execution_mode(args)
    args.async_planner = False
    args.rollout = True

    controller = SamplingC3Controller(config_from_args(args, int(args.trial_start)))
    bus = SynchronousMPCBus(args)
    trace = None
    if args.rollout_trace_jsonl:
        path = Path(args.rollout_trace_jsonl).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        trace = path.open('w')
    steps = max(1, int(args.max_rollout_length))
    try:
        for _ in range(steps):
            obs = bus.wait_obs()
            if obs.get("cmd") == "stop" or obs.get("break_out") or obs.get("viewer_closed"):
                break
            target = bus.env.param_.target_p_
            controller.configure_scene(bus.env)
            result = controller.step(obs, target, bus.env.param_.target_q_, EXECUTION_STEP_M)
            _admit_sampling_step(bus.env)
            action = result["action"]
            via = result["policy"]["mpc_virtual_point"]
            bus.env._planner_force_world = result['actuator_force_world']
            if trace is not None:
                plan = result['rigid_plan']
                trace.write(json.dumps(dict(seq=int(obs['seq']), sim_time=float(obs['sim_time']),
                    policy_q=np.asarray(obs['policy_q']).tolist(), target_p=np.asarray(target).tolist(),
                    target_q=np.asarray(bus.env.param_.target_q_).tolist(), mode=plan['mode'],
                    action=np.asarray(action).tolist(), via=np.asarray(via).tolist(),
                    actuator_force_world=result['actuator_force_world'].tolist(),
                    orientation_error=plan['orientation_error'], position_error=plan['position_error'],
                    sample_costs=[float(c) if np.isfinite(c) else None for c in plan['sample_costs']],
                    sample_locations=[np.asarray(p).tolist() for p in plan['sample_locations']],
                    reposition_target=(None if controller.controller.reposition_target is None else
                        controller.controller.reposition_target.tolist()),
                    projection_statistics=plan['candidate_projection_statistics'],
                    candidate_failures=plan['candidate_failures'], plan_dt=float(result['plan_dt'])))+'\n')
                trace.flush()
            print(
                "sampling_exec:",
                result["policy"]["value_info"]["via_phase"],
                "action_m:",
                round(float(np.linalg.norm(action)), 4),
                "via:",
                [round(float(v), 4) for v in via],
                flush=True,
            )
            bus.publish_cmd({
                "kind": "via",
                "seq": int(obs.get("seq", 0)),
                "via": via,
                "action": action,
                "policy_dt": POLICY_INTERVAL,
                "planner_dt": float(result["plan_dt"]),
                "press": result['execution_press'],
                "release_pressure": bool(result["policy"]["release_pressure"]),
                "approach_phase": result["policy"]["value_info"]["via_phase"],
                "markers": {"target": via, "best_contact": via},
            })
    finally:
        controller.close()
        bus.close()
        if trace is not None:
            trace.close()


def _admit_sampling_step(env):
    """Let this process execute a sampling knot. Do not change the shared default."""
    from examples.mpc.franka.experiments.sampling.isaac import EXECUTION_STEP_M

    env.param_.mpc_u_ub_ = EXECUTION_STEP_M
    env.param_.mpc_u_lb_ = -EXECUTION_STEP_M
    env.param_.isaac_via_max_step_ = EXECUTION_STEP_M
    schedule = getattr(env, "_policy_schedule", None)
    if schedule is not None:
        schedule.max_step = EXECUTION_STEP_M


if __name__ == "__main__":
    os.environ.pop("SCSP_PLANNER_ONLY", None)
    main()
