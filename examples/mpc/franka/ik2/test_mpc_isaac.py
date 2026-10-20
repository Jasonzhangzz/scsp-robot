"""Run Franka MPC with Isaac Gym or MJWarp.

Examples, from the repository root:
    python examples/mpc/franka/ik2/test_mpc_isaac.py --headless --sync-planner
    python examples/mpc/franka/ik2/test_mpc_isaac.py --viewer --async-planner

Implementation: examples/mpc/franka/runtime/. Experiments: ../experiments/ik2/.
"""
import argparse
import os
from pathlib import Path
import sys


def main():
    # Dispatch before importing Gym, Torch or the planner. Spawned planner
    # processes must be able to import this entry without initializing physics.
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))
    from models.contactsdf import dispatch_standalone
    from examples.mpc.franka.stocs3d.dispatch import dispatch_execution
    dispatch_execution(root, sys.argv[1:])
    dispatch_standalone(root, sys.argv[1:])
    from planning.cpu_runtime import configure_compute_threads
    configure_compute_threads()
    os.environ.pop("SCSP_PLANNER_ONLY", None)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--physics-backend", choices=("isaac", "mjwarp"), default="isaac")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--viewer", dest="headless", action="store_false")
    parser.set_defaults(headless=False)
    args, _ = parser.parse_known_args()
    flags = sys.argv[1:]
    if args.physics_backend == "mjwarp" or (args.headless and "--floating-fingertip" not in flags):
        script = root / "examples/mpc/franka/runtime/rollout.py"
    elif "--floating-fingertip" in flags:
        script = root / "examples/mpc/fingertips/test/test_0902_isaac.py"
        flags = [arg for arg in flags if arg != "--floating-fingertip"]
    else:
        script = root / "examples/mpc/franka/runtime/isaac.py"
        flags = ["--planner", "mpc", *flags]
    if args.physics_backend == "mjwarp":
        os.environ["SCSP_PLANNER_ONLY"] = "1"
    os.execv(sys.executable, [sys.executable, "-u", str(script), *flags])


if __name__ == "__main__":
    main()
