"""Locate the scsp-robot repo and a local acados tree in-process.

Entry points should put this repository on ``sys.path`` by walking up to the
``scsp-robot`` directory. A shell ``PYTHONPATH`` / ``ACADOS_SOURCE_DIR``
export is not required.
"""
import ctypes
import os
import sys
from pathlib import Path

DEFAULT_ACADOS_SOURCE_DIR = "/home/lab423/scsp/thirdparty/acados"
REPO_DIR_NAME = "scsp-robot"


def find_repo_root(start=None):
    """Return the scsp-robot root, walking up from ``start`` or this file."""
    if start is None:
        start = Path(__file__).resolve()
    else:
        start = Path(start).resolve()
    if start.is_file():
        start = start.parent

    named = None
    marked = None
    for candidate in [start, *start.parents]:
        has_planning = (candidate / "planning").is_dir()
        if candidate.name == REPO_DIR_NAME and has_planning:
            named = candidate
            break
        if marked is None and (candidate / "planning" / "acados_env.py").is_file():
            marked = candidate
    root = named or marked
    if root is None:
        raise RuntimeError(
            f"Could not find the {REPO_DIR_NAME} repository root starting from {start}"
        )
    return root


def ensure_repo_root(start=None):
    """Insert the scsp-robot root at the front of ``sys.path``."""
    root = find_repo_root(start)
    root_s = str(root)
    sys.path[:] = [
        entry for entry in sys.path
        if os.path.abspath(entry or os.curdir) != root_s
    ]
    sys.path.insert(0, root_s)
    return root


def _repo_root_dir():
    return str(find_repo_root())


def acados_root_candidates(repo_root=None):
    if repo_root is None:
        repo_root = _repo_root_dir()
    home = str(Path.home())
    candidates = []
    acados_source_dir = os.environ.get("ACADOS_SOURCE_DIR")
    if acados_source_dir:
        candidates.append(os.path.abspath(acados_source_dir))
    candidates.extend((
        os.path.abspath(os.path.join(repo_root, "..", "thirdparty", "acados")),
        os.path.abspath(os.path.join(repo_root, "..", "acados")),
        os.path.abspath(os.path.join(repo_root, "thirdparty", "acados")),
        os.path.abspath(os.path.join(repo_root, "..", "zz_ws", "acados")),
        os.path.abspath(os.path.join(home, "zz_ws", "acados")),
        os.path.abspath(os.path.join(home, "acados")),
        os.path.abspath(DEFAULT_ACADOS_SOURCE_DIR),
    ))

    deduped = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _is_acados_tree(acados_root):
    interface = os.path.join(
        acados_root, "interfaces", "acados_template", "acados_template", "__init__.py"
    )
    if not os.path.isfile(interface):
        return False
    lib_dir = os.path.join(acados_root, "lib")
    has_lib = any(
        os.path.isfile(os.path.join(lib_dir, name))
        for name in ("libacados.so", "libacados.so.1", "libacados.dylib", "acados.dll")
    )
    has_link_libs = os.path.isfile(os.path.join(lib_dir, "link_libs.json"))
    return has_lib and has_link_libs


def _preload_acados_shared_libraries(acados_root):
    if os.name == "nt":
        return
    lib_dir = os.path.join(acados_root, "lib")
    if not os.path.isdir(lib_dir):
        return
    load_mode = getattr(ctypes, "RTLD_GLOBAL", None)
    for lib_name in (
        "libblasfeo.so.0",
        "libblasfeo.so",
        "libhpipm.so",
        "libqpOASES_e.so",
        "libdaqp.so",
        "libosqp.so",
        "libacados.so",
    ):
        lib_path = os.path.join(lib_dir, lib_name)
        if not os.path.isfile(lib_path):
            continue
        try:
            if load_mode is None:
                ctypes.CDLL(lib_path)
            else:
                ctypes.CDLL(lib_path, mode=load_mode)
        except OSError:
            continue


def ensure_acados_env():
    """Set ``ACADOS_SOURCE_DIR`` and Python/library paths in this process."""
    ensure_repo_root()
    chosen = None
    for acados_root in acados_root_candidates():
        if _is_acados_tree(acados_root):
            chosen = acados_root
            break
    if chosen is None:
        chosen = os.path.abspath(os.environ.get("ACADOS_SOURCE_DIR") or DEFAULT_ACADOS_SOURCE_DIR)

    os.environ["ACADOS_SOURCE_DIR"] = chosen
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    acados_lib_dir = os.path.join(chosen, "lib")
    if os.path.isdir(acados_lib_dir):
        ld_entries = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
        if acados_lib_dir not in ld_entries:
            os.environ["LD_LIBRARY_PATH"] = ":".join([acados_lib_dir] + ld_entries)
        _preload_acados_shared_libraries(chosen)

    interface_root = os.path.join(chosen, "interfaces", "acados_template")
    if os.path.isdir(interface_root) and interface_root not in sys.path:
        sys.path.insert(0, interface_root)
    return chosen
