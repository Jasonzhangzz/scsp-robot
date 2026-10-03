"""Small, dependency-light runtime compatibility helpers.

The Isaac entry point is run in two processes and optional packages are loaded
at different times in those processes.  Keeping the profile here avoids each
process silently selecting a different solver or time base.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib
import importlib.metadata
import platform
import sys


def _module_version(name):
    try:
        version = importlib.metadata.version(name)
        # Isaac Gym's top-level import can fail when Torch was imported first;
        # package metadata is still a valid capability diagnostic.
        if name == "isaacgym":
            return {"version": version, "available": True, "error": None}
    except Exception:
        pass
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # optional packages are expected to be absent
        return {"version": None, "available": False, "error": type(exc).__name__}
    return {
        "version": getattr(module, "__version__", None),
        "available": True,
        "error": None,
    }


def acados_available():
    # The repository ships an external acados tree rather than a wheel.  Add
    # its interface/library paths before probing so standalone parameter
    # construction sees the same backend as the planner worker.
    try:
        from planning.acados_env import ensure_acados_env
        ensure_acados_env()
    except Exception:
        pass
    try:
        importlib.import_module("acados_template")
        return True
    except Exception:
        return False


def resolve_solver_backend(requested=None, portable_default=False):
    """Return the concrete backend shared by planner and simulation workers.

    ``portable`` deliberately maps to IPOPT.  acados remains available as an
    opt-in backend, but an unavailable acados install never causes an implicit
    mid-run backend switch.
    """
    requested = "portable" if requested is None else str(requested).strip().lower()
    if requested in {"portable", "stable", "deterministic"}:
        return "ipopt"
    if requested in {"auto", "acados", "snopt"}:
        if requested == "auto" and portable_default:
            return "ipopt"
        return "acados" if acados_available() else "ipopt"
    if requested in {"ipopt", "torch-lbfgs", "torch-gn"}:
        return requested
    raise ValueError(
        "Unsupported solver backend %r; expected portable, auto, acados or ipopt"
        % requested
    )


@dataclass(frozen=True)
class RuntimeProfile:
    solver_requested: str = "auto"
    solver_backend: str = "ipopt"
    policy_dt: float = 0.02
    sim_dt: float = 0.002
    control_substeps: int = 10
    table_clearance: float = 0.012

    def as_dict(self):
        return asdict(self)


def make_runtime_profile(requested="auto", policy_dt=0.02, sim_dt=0.002,
                         table_clearance=0.012):
    policy_dt = max(float(policy_dt), 1e-6)
    sim_dt = max(float(sim_dt), 1e-6)
    return RuntimeProfile(
        solver_requested=str(requested),
        solver_backend=resolve_solver_backend(requested),
        policy_dt=policy_dt,
        sim_dt=sim_dt,
        control_substeps=max(1, int(round(policy_dt / sim_dt))),
        table_clearance=max(float(table_clearance), 0.0),
    )


def runtime_fingerprint(profile=None):
    """Return JSON-safe diagnostics without requiring Isaac Gym."""
    profile = profile or make_runtime_profile()
    packages = {
        name: _module_version(name)
        for name in ("numpy", "scipy", "mujoco", "trimesh", "casadi", "torch", "isaacgym")
    }
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "profile": profile.as_dict(),
        "packages": packages,
        "acados_available": acados_available(),
    }


def print_runtime_profile(profile=None, prefix="[runtime]"):
    info = runtime_fingerprint(profile)
    p = info["profile"]
    print(
        prefix,
        "solver=%s requested=%s policy_dt=%.6f sim_dt=%.6f substeps=%d clearance=%.4f"
        % (
            p["solver_backend"], p["solver_requested"], p["policy_dt"],
            p["sim_dt"], p["control_substeps"], p["table_clearance"],
        ),
        flush=True,
    )
    print(
        prefix,
        "python=%s numpy=%s scipy=%s mujoco=%s trimesh=%s casadi=%s torch=%s isaacgym=%s acados=%s"
        % tuple(
            [info["python"]]
            + [info["packages"][name]["version"] for name in
               ("numpy", "scipy", "mujoco", "trimesh", "casadi", "torch", "isaacgym")]
            + [info["acados_available"]]
        ),
        flush=True,
    )
    return info
