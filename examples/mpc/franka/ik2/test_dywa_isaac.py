"""Run the DyWA student on the same trials as test_mpc_isaac.py.

Use the acc environment. The script switches to it when another interpreter
launches this file. Pose sampling, object mesh, mass, friction, 20 ms control
period, and the 2 cm / quaternion success test match the MPC Isaac entry.
DyWA keeps its own camera and arm controller; object positions are placed on
its table with the same relative translation and the same quaternions.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planning.dywa_interface import ACC_PYTHON, build_experiment, execute_trial


def main():
    if ACC_PYTHON.is_file() and Path(sys.executable).resolve() != ACC_PYTHON.resolve():
        os.execv(str(ACC_PYTHON), [str(ACC_PYTHON), *sys.argv])

    os.environ.setdefault("SCSP_PLANNER_ONLY", "1")
    from examples.mpc.franka.ik2.test_mpc_isaac import _add_rollout_policy_args
    from planning.dywa_interface import DEFAULT_CKPT_DIR

    parser = argparse.ArgumentParser(
        description="DyWA student on the test_mpc_isaac experiment.",
    )
    _add_rollout_policy_args(parser)
    parser.add_argument("--dywa-ckpt-dir", default=str(DEFAULT_CKPT_DIR))
    parser.set_defaults(headless=False)
    args = parser.parse_args()
    args.dywa_ckpt_dir = args.dywa_ckpt_dir

    if args.trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {args.trial_start}")
    trial_num = int(args.trial_num)
    if trial_num <= 0:
        raise ValueError(f"trial_num must be positive, got {trial_num}")

    success_count = 0
    for trial in range(int(args.trial_start), int(args.trial_start) + trial_num):
        spec = build_experiment(args, trial)
        print(
            "dywa_experiment:",
            {
                "trial": trial,
                "object": spec["object"],
                "mesh": spec["mesh_path"],
                "mass": spec["mass"],
                "table_friction": spec["table_friction"],
                "object_friction": spec["object_friction"],
                "dt": spec["dt"],
                "action_period": spec["action_period"],
                "init_xy": spec["init_obj_qpos"][:2],
                "target_xy": spec["target_p"][:2],
            },
            flush=True,
        )
        summary = execute_trial(spec)
        success_count += int(summary["success"])
        print("trial_summary: " + repr(summary), flush=True)

    print(
        f"Success rate over {trial_num} trials "
        f"(trial ids {args.trial_start} to {args.trial_start + trial_num - 1}): "
        f"{success_count}/{trial_num} = {success_count / trial_num:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
