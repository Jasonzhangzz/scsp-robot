"""Run the Franka MPC entry point with synchronous planning enabled.

The implementation is shared with the normal launcher; this file only fixes
its execution mode so the mode is visible in the directory layout and command.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[5]
ENTRY = ROOT / "examples/mpc/franka/ik2/test_mpc_isaac.py"
_MODE_FLAGS = {"--async-planner", "--sync-planner", "--trial"}


def main() -> None:
    args = [arg for arg in sys.argv[1:] if arg not in _MODE_FLAGS]
    os.execv(sys.executable, [sys.executable, "-u", str(ENTRY), "--sync-planner", *args])


if __name__ == "__main__":
    main()
