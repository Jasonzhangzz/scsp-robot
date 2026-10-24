"""Render a cached Isaac trajectory CSV from the fixed main camera.

The CSVs under ``output/trials/trajectories/isaac`` contain the successful
flipping rollouts, including the object pose and seven arm joint positions at
every 20 ms policy sample.  This utility restores those states directly and
captures them without running a second planner, so the recorded video matches
the cached successful trajectory exactly.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rows(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty trajectory CSV: {path}")
    if not all(str(row.get("success", "0")).lower() in ("1", "true") for row in rows):
        raise ValueError(f"trajectory is not marked successful: {path}")
    return rows


def _float(row, key):
    return float(row[key])


def replay(args):
    from examples.mpc.franka.runtime.isaac import build_osc_param
    from examples.mpc.franka.runtime.mpc import IsaacFrankaOSCSimulator

    rows = _rows(args.csv)
    parser_args = argparse.Namespace(**vars(args))
    parser_args.headless = False
    parser_args.target_type = "ground-rotation"
    parser_args.trial_num = 1
    parser_args.trial_start = int(args.trial_id)
    parser_args.video_output_path = str(args.output)
    parser_args.video_output_dir = ""
    parser_args.video_wall_clock_timing = False
    param = build_osc_param(parser_args, int(args.trial_id))
    env = IsaacFrankaOSCSimulator(
        param,
        headless=False,
        sim_device=args.sim_device,
        graphics_device_id=int(args.graphics_device_id),
    )
    try:
        for row in rows:
            position = np.array([_float(row, key) for key in ("x", "y", "z")], dtype=np.float32)
            quat_wxyz = np.array(
                [_float(row, key) for key in ("qw", "qx", "qy", "qz")],
                dtype=np.float32,
            )
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
            joint = np.array(
                [_float(row, f"robot_q{i}") for i in range(7)], dtype=np.float32
            )
            env._set_visual_actor_pose_xyzw(env.obj_actor, position, quat_xyzw)
            env._set_arm_qpos(joint)
            env._sync_pose_axes()
            env.gym.step_graphics(env.sim)
            env.video_recorder_.capture(
                sim_time=float(row.get("sim_time_s", 0.0)),
                step_graphics=False,
            )
    finally:
        env.close()
    print(f"replayed {len(rows)} frames from {args.csv} -> {args.output}")


def main():
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    parser = _add_rollout_policy_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trial-id", type=int, required=True)
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
