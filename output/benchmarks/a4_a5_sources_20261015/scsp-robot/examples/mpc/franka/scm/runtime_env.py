"""Use the pi0 interpreter with this benchmark's pinned local dependencies."""

from pathlib import Path
import sys


DEPENDENCIES = Path(__file__).resolve().parent / ".deps"


def activate_dependencies():
    if Path(sys.prefix).name != "pi0":
        raise RuntimeError("Run with the pi0 environment: conda run -n pi0 python <script>.")
    if not (DEPENDENCIES / "comfree_warp").is_dir():
        raise RuntimeError("Install benchmark dependencies: python -m pip install --target .deps --no-deps -r requirements.txt")
    for name in ("mujoco", "warp", "comfree_warp"):
        module = sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(DEPENDENCIES):
            raise RuntimeError(f"{name} was already imported from another location. Start a fresh pi0 Python process.")
    location = str(DEPENDENCIES)
    if location not in sys.path:
        sys.path.insert(0, location)
