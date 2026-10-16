"""Run the retained native FA, compiled-gradient FA or Direct ComFree MPC."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    selection = argparse.ArgumentParser(add_help=False, description=__doc__)
    selection.add_argument('--solver', choices=['fa', 'compiled', 'direct-mpc'], default='fa')
    selection.add_argument('--forge', type=Path, default=root / 'thirdparty/retarget_assets/DexForge')
    selection.add_argument('--experiment', type=Path, default=root / 'outputs/fa_admm_solver_20261016')
    args, remaining = selection.parse_known_args()
    if any(flag in remaining for flag in ('-h', '--help')):
        selection.print_help()
        print('\nSelected solver options:')
    forge = args.forge.resolve()
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('COMFREE_WARP_ROOT', str(forge / 'forceaware/third_party/comfree_warp'))
    os.environ.setdefault('HAND_ASSETS_ROOT', str(forge / 'assets'))
    sys.argv = [sys.argv[0], *remaining, '--forge', str(forge),
                '--experiment', str(args.experiment.resolve())]
    if args.solver == 'direct-mpc':
        if '--contact-admm' not in sys.argv:
            sys.argv.append('--contact-admm')
        from examples.retarget.direct.run import main as direct
        direct()
        return
    defaults = {
        '--executor': 'comfree',
        '--case-name': args.solver + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'),
        '--multistarts': '8',
        '--iterations': '40',
        '--evaluation-backend': 'compiled' if args.solver == 'compiled' else 'native',
    }
    for key, value in defaults.items():
        if not any(a == key or a.startswith(key + '=') for a in sys.argv[1:]):
            sys.argv.extend([key, value])
    from examples.retarget.fa.worker import main as worker
    worker()


if __name__ == '__main__':
    main()
