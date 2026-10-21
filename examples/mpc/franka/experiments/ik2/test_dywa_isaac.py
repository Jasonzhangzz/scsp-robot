"""Compatibility entry for the consolidated official DyWA runner."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from planning.dywa.run import main

if __name__ == "__main__":
    sys.exit(main())
