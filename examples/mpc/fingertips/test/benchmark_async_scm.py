"""Compatibility entry for the corrected-SCM contact-model benchmark.

Legacy soft-SCM solver sweeps are historical experiments. Use the shared
controller so model preflight, visualizer parity and timing rules stay aligned.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from examples.mpc.fingertips.test.benchmark_contact_models import main

if __name__ == '__main__':
    if '--phase' not in sys.argv:
        sys.argv.extend(['--phase', 'async'])
    main()
