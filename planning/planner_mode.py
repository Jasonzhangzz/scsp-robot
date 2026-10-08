"""Planner-mode aliases shared by BigRasp CLI and tests."""

from __future__ import annotations


PLANNER_MODE_ALIASES = {
    "mppi": "mppi_ee",
    "mppi_ee": "mppi_ee",
    "mppi-ee": "mppi_ee",
    "mppi_joint": "mppi_joint",
    "mppi-joint": "mppi_joint",
    "mpc": "mpc_ee",
    "mpc_ee": "mpc_ee",
    "mpc-ee": "mpc_ee",
    "mpc_joint": "mpc_joint",
    "mpc-joint": "mpc_joint",
    "explicit_mjwp": "mppi_joint",
    "spider_mjwp": "mppi_joint",
    "surrogate_mppi": "mppi_ee",
    "physical_mppi": "physical_mppi",
    "acados": "mpc_ee",
    "ipopt": "mpc_ee",
}


def resolve_planner_mode(args) -> str:
    """Resolve the active planner. New --mppi/--mpc flags win over legacy names."""
    explicit = getattr(args, "planner_mode", None)
    if explicit:
        return PLANNER_MODE_ALIASES.get(str(explicit).strip().lower(), "mppi_ee")
    solver = str(getattr(args, "planner_solver", "") or "").strip().lower()
    backend = str(getattr(args, "planner_backend", "") or "").strip().lower()
    if solver in PLANNER_MODE_ALIASES:
        return PLANNER_MODE_ALIASES[solver]
    if backend in PLANNER_MODE_ALIASES:
        return PLANNER_MODE_ALIASES[backend]
    return "mppi_ee"


def resolve_planner_backend(args) -> str:
    """Compatibility wrapper used by older call sites."""
    return resolve_planner_mode(args)
