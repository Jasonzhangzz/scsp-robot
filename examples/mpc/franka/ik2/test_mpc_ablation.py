"""Franka ablations, including --ema / --threshold sensitivity grids."""
from pathlib import Path
import sys


def main():
    flags = sys.argv[1:]
    defaults = []
    if not any(flag in flags for flag in ('--viewer', '--headless')):
        defaults.append('--viewer')
    if not any(flag in flags for flag in ('--sync-planner', '--async-planner', '--trial')):
        defaults.append('--sync-planner')
    sys.argv[1:1] = defaults
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from examples.mpc.franka.experiments.test_mpc_ablation import main as run
    run()


if __name__ == '__main__':
    main()
