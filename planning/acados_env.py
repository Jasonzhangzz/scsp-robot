"""Locate the scsp-robot repo and a local acados tree in-process.

Entry points should put this repository on ``sys.path`` by walking up to the
``scsp-robot`` directory. A shell ``PYTHONPATH`` / ``ACADOS_SOURCE_DIR``
export is not required.
"""
import ctypes
import os
import sys
from pathlib import Path

# The locally installed/build-tested acados tree.  Keep ACADOS_SOURCE_DIR as
# an explicit override, but prefer this installation over repository copies.
DEFAULT_ACADOS_SOURCE_DIR = "/home/lab423/push_ws/acados"
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
        os.path.abspath(DEFAULT_ACADOS_SOURCE_DIR),
        os.path.abspath(os.path.join(repo_root, "..", "thirdparty", "acados")),
        os.path.abspath(os.path.join(repo_root, "..", "acados")),
        os.path.abspath(os.path.join(repo_root, "thirdparty", "acados")),
        os.path.abspath(os.path.join(repo_root, "..", "zz_ws", "acados")),
        os.path.abspath(os.path.join(home, "zz_ws", "acados")),
        os.path.abspath(os.path.join(home, "acados")),
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


class quiet_acados_stderr:
    """Hide HPIPM/SQP_RTI MINSTEP spam; the Python caller still sees the status."""

    def __enter__(self):
        self._fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        return self

    def __exit__(self, exc_type, exc, tb):
        os.dup2(self._fd, 2)
        os.close(self._fd)
        return False


def build_cached_solver(ocp, json_file, shared_file):
    """Serialize code generation shared by the simulator and planner workers.

    Current acados-template assigns a hash to an unnamed OCP. Use the model's
    existing version/configuration identity so both workers load the same
    library, then publish/load it while holding an inter-process cache lock.
    """
    import fcntl
    import json
    import subprocess
    import time
    from pathlib import Path
    from acados_template import AcadosOcpSolver
    if hasattr(ocp, 'name'):
        ocp.name = ocp.model.name
    directory = Path(json_file).parent
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.codegen.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('[acados] waiting for solver cache: %s' % ocp.model.name, flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
        current = False
        try:
            data = json.loads(Path(json_file).read_text())
            options = data.get('code_gen_options', data)
            current = (Path(shared_file).is_file()
                       and all(k in options for k in ('acados_lib_path', 'code_export_directory'))
                       and data.get('name', ocp.model.name) == ocp.model.name)
        except (OSError, ValueError, TypeError):
            pass
        if not current:
            started = time.perf_counter()
            print('[acados] generating solver: %s' % ocp.model.name, flush=True)
            AcadosOcpSolver.generate(ocp, json_file=str(json_file))
            makefile = directory / 'Makefile'
            if os.name != 'nt' and makefile.is_file():
                # Unrolled SCM Hessians can contain megabytes of scalar C.
                # GCC's optimizer spends minutes on these files at -O2.
                # Keep the normal flags for small functions and the solver;
                # only skip optimization on the oversized generated maps.
                large_sources = sorted(p for p in directory.rglob('*.c')
                                       if p.stat().st_size >= 1024*1024)
                if large_sources:
                    with makefile.open('a') as handle:
                        handle.write('\n# Bound cold-start time for large CasADi maps.\n')
                        for source in large_sources:
                            target = source.relative_to(directory).with_suffix('.o')
                            handle.write('%s: CFLAGS += -O0\n' % target.as_posix())
                print('[acados] compiling solver (%d large maps); log: %s' %
                      (len(large_sources), directory / 'build.log'), flush=True)
                with (directory / 'build.log').open('w') as output:
                    result = subprocess.run(
                        ['make', '-j2', 'ocp_shared_lib'], cwd=str(directory),
                        stdout=output, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError('acados compilation failed; see %s' %
                                       (directory / 'build.log'))
            else:
                AcadosOcpSolver.build(str(directory), verbose=False)
            print('[acados] solver ready in %.2fs' %
                  (time.perf_counter() - started), flush=True)
        return AcadosOcpSolver(ocp, json_file=str(json_file), generate=False,
                               build=False, verbose=False)
