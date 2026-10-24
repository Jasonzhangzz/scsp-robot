"""Use the pi0 interpreter with this benchmark's pinned local dependencies."""

from pathlib import Path
import sys


DEPENDENCIES = Path(__file__).resolve().parent / ".deps"


def preload_isaacgym(argv=None):
    """Import Isaac Gym before MuJoCo when the benchmark requests it.

    MuJoCo wheels (>=3.2) bundle their own USD libraries. If they are loaded
    first, Isaac Gym's ``libcarb.gym.plugin.so`` fails with an undefined
    ``pxrInternal_v0_19`` symbol and ``acquire_gym`` returns an unusable
    interface, forcing the MuJoCo fallback. Importing the legacy SDK first
    avoids the symbol clash. Only done when ``isaacgym`` is actually requested.
    """
    argv = sys.argv[1:] if argv is None else argv
    if "isaacgym" not in argv or "mujoco" in sys.modules:
        return False
    try:
        import isaacgym  # noqa: F401
    except ImportError:
        return False
    return True


def activate_dependencies():
    # The original benchmark was tied to the author's ``pi0`` environment.
    # The CPU MuJoCo/Isaac comparison is intentionally usable in a regular
    # environment as well; GPU backends still fail with their own actionable
    # dependency message when selected.
    if not (DEPENDENCIES / "comfree_warp").is_dir():
        return
    for name in ("mujoco", "warp", "comfree_warp"):
        module = sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(DEPENDENCIES):
            raise RuntimeError(f"{name} was already imported from another location. Start a fresh pi0 Python process.")
    location = str(DEPENDENCIES)
    if location not in sys.path:
        sys.path.insert(0, location)
