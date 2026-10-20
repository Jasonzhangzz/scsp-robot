"""Unified Franka MPC ablation entry; all baseline runtime flags are forwarded."""
import argparse
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root))
    from models.contactsdf import dispatch_standalone
    from examples.mpc.franka.stocs3d.dispatch import dispatch_offline
    dispatch_offline(root, sys.argv[1:])
    dispatch_standalone(root, sys.argv[1:])
    from planning.cpu_runtime import configure_compute_threads
    configure_compute_threads()
    from examples.mpc.franka.experiments.mpc_ablation.config import selection_parser
    # Validate selections before loading Gym, Torch, MuJoCo or solver libraries.
    select_parser = selection_parser()
    selection, rest = select_parser.parse_known_args()
    if not (selection.ema or selection.threshold or selection.acceptance) and any(
            flag.split('=')[0] in ('--ema-grid', '--ema-offsets', '--threshold-grid',
                                 '--acceptance-grid', '--acceptance-offsets',
                                 '--sensitivity-output', '--sensitivity-cohort', '--sensitivity-workers', '--baseline-results',
                                 '--plot', '--dry-run') for flag in sys.argv[1:]):
        select_parser.error('sensitivity options require --ema or --threshold')
    if (selection.ema or selection.threshold or selection.acceptance) and not any(
            flag in sys.argv[1:] for flag in ('--help', '-h')):
        from examples.mpc.franka.experiments.mpc_ablation.sensitivity import run_sensitivity
        try:
            run_sensitivity(selection, rest)
        except ValueError as exc:
            select_parser.error(str(exc))
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--physics-backend', choices=('isaac', 'mjwarp'), default='isaac')
    parser.add_argument('--headless', dest='headless', action='store_true')
    parser.add_argument('--viewer', dest='headless', action='store_false')
    parser.add_argument('--floating-fingertip', action='store_true')
    parser.set_defaults(headless=False)
    args, _ = parser.parse_known_args()
    if selection.ablation_mode == 'a6' and args.physics_backend != 'isaac':
        parser.error('A6 executes in Isaac Gym; use --physics-backend isaac')
    if args.floating_fingertip and selection.ablation_mode != 'full':
        parser.error('MPC ablations use the Franka OSC environment; remove --floating-fingertip')
    os.environ.pop('SCSP_PLANNER_ONLY', None)
    if '--help' in sys.argv[1:] or '-h' in sys.argv[1:]:
        # Help must work on CPU-only machines without loading Isaac Gym.
        os.environ['SCSP_PLANNER_ONLY'] = '1'
        from examples.mpc.franka.experiments.mpc_ablation.mpc import _add_rollout_policy_args
        _add_rollout_policy_args(argparse.ArgumentParser(description=__doc__)).parse_args()
        return
    folder = Path(__file__).resolve().with_name('mpc_ablation')
    if args.physics_backend == 'mjwarp' or (args.headless and not args.floating_fingertip):
        script, flags = folder/'rollout.py', sys.argv[1:]
    elif args.floating_fingertip:
        script = root/'examples/mpc/fingertips/test/test_0902_isaac.py'
        # Full mode also preserves the baseline's floating-fingertip dispatch.
        _, flags = selection_parser().parse_known_args()
        flags = [arg for arg in flags if arg != '--floating-fingertip']
    else:
        script, flags = folder/'isaac.py', ['--planner', 'mpc', *sys.argv[1:]]
    if args.physics_backend == 'mjwarp':
        os.environ['SCSP_PLANNER_ONLY'] = '1'
    os.execv(sys.executable, [sys.executable, '-u', str(script), *flags])


if __name__ == '__main__':
    main()
