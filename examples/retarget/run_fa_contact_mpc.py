"""Contact ADMM with analytic ComFree batch LM and the FA hand solve.

See FA_MATCHED_ABLATION.md for matched solver/trajectory comparisons and
the optional Direct tracking, warm-start and template-commit corrections.
FA_COMPILED_EVALUATOR.md documents the equation/analytic-gradient backend
and exact per-candidate value reuse during line search.
"""
import os
from pathlib import Path
import sys
from datetime import datetime


def main():
    import argparse
    selection=argparse.ArgumentParser(add_help=False)
    selection.add_argument('--solver',choices=['contact-admm','direct-mpc'],default='contact-admm')
    args,remaining=selection.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    if args.solver=='direct-mpc':
        if '--contact-admm' not in sys.argv:sys.argv.append('--contact-admm')
        from examples.retarget.run_fa_direct_mpc import main as direct
        direct()
        return
    root=Path(__file__).resolve().parents[2]
    forge=root/'thirdparty/retarget_assets/DexForge'
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('COMFREE_WARP_ROOT',str(forge/'forceaware/third_party/comfree_warp'))
    os.environ.setdefault('HAND_ASSETS_ROOT',str(forge/'assets'))
    defaults={'--experiment':str(root/'outputs/fa_admm_solver_20261016'),
              '--forge':str(forge),'--executor':'mujoco','--variant':'admm-force',
              '--case-name':'force_contact_'+datetime.now().strftime('%Y%m%d_%H%M%S'),
              '--multistarts':'2','--iterations':'20','--admm-outer':'2',
              '--contact-solver':'batch-lm'}
    for key,value in defaults.items():
        if not any(a==key or a.startswith(key+'=') for a in sys.argv[1:]):sys.argv.extend([key,value])
    from examples.retarget.matrix_fa_worker import main as worker
    worker()


if __name__=='__main__':main()
