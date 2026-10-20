"""Route --stocs3d before any other planner is imported."""
import os
import sys
from pathlib import Path


def offline_requested(argv):
    return '--stocs3d' in argv


def trajectory_argument(argv):
    for i, flag in enumerate(argv):
        if flag == '--stocs-trajectory' and i + 1 < len(argv):
            return argv[i + 1]
        if flag.startswith('--stocs-trajectory='):
            return flag.split('=', 1)[1]
    return None


def dispatch_offline(root, argv):
    """Solve STOCS-3D offline, then hand the trajectory to test_mpc_isaac.py."""
    if not offline_requested(argv):
        return
    script = Path(root) / 'examples/mpc/franka/stocs3d/run.py'
    os.execv(sys.executable, [sys.executable, '-u', str(script), *argv])


def dispatch_execution(root, argv):
    """Track a solved STOCS-3D trajectory with the Isaac OSC executor."""
    if trajectory_argument(argv) is None:
        return
    script = Path(root) / 'examples/mpc/franka/stocs3d/execute.py'
    os.execv(sys.executable, [sys.executable, '-u', str(script), *argv])
