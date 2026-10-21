"""Select DyWA before importing any local planner or physics backend."""
import os
from pathlib import Path
import sys


def dispatch(root, argv):
    if '--dywa' not in argv:
        return
    conflicts = {'--contactsdf', '--stocs3d', '--ema', '--threshold', '--acceptance',
                 '--a1', '--a2', '--a3', '--a4', '--a5', '--a6', '--a8',
                 '--sampling-c3', '--floating-fingertip'}
    if any(flag.split('=')[0] in conflicts for flag in argv):
        raise ValueError('--dywa selects the independent official policy; incompatible planner flags')
    script = Path(root) / 'planning/dywa/run.py'
    os.execv(sys.executable, [sys.executable, '-u', str(script), *argv])
