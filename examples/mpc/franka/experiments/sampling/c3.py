"""C3 through the compiled ADMM extension."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


def _extension():
    name = "c3_core"
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        try:
            from .build_c3 import build
        except ImportError:
            from build_c3 import build

        build()
        return importlib.import_module(name)


def solve(lcs, costs, x0, x_desired, admm_iterations=3, rho_scale=3.0,
          projection_alpha=0.01, u_lower=None, u_upper=None):
    """Solve one local C3 problem.

    ``lcs`` is an ``LCS`` or the dict from ``LCS.sequence``. ``costs`` is
    ``(Q, R, G, U)`` with N+1 state weights and N input, consensus, and
    projection weights. The returned force is in the caller's units; the
    extension's internal scaling stays inside the ADMM loop.
    """
    core = _extension()
    sequence = lcs if isinstance(lcs, dict) else lcs.sequence(len(costs[1]))
    horizon = len(sequence["A"])
    desired = _as_rows(x_desired, sequence["A"][0].shape[0], horizon + 1)
    lower, upper = _input_bounds(u_lower, u_upper, sequence["B"][0].shape[1])
    Q, R, G, U = costs
    plan = core.solve_c3(
        sequence["A"], sequence["B"], sequence["D"], sequence["d"],
        sequence["E"], sequence["F"], sequence["H"], sequence["c"],
        Q, R, G, U, np.asarray(x0, dtype=float).reshape(-1), desired,
        int(admm_iterations), float(rho_scale), float(projection_alpha),
        lower, upper,
    )
    x = np.asarray(plan["x"], dtype=float)
    u = np.asarray(plan["u"], dtype=float)
    force = np.asarray(plan["lambda"], dtype=float)
    if x.shape[0] != sequence["A"][0].shape[0]:
        x = x.T
        u = u.T
        force = force.T
    if not bool(plan["success"]):
        raise RuntimeError("compiled C3 returned a non-finite plan")
    return {"x": x.T, "u": u.T, "lambda": force.T, "solver": "c3"}


def _as_rows(values, width, count):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if array.size != width:
            raise ValueError("desired state has the wrong dimension")
        return [array.copy() for _ in range(count)]
    if array.shape != (count, width):
        raise ValueError("desired trajectory must have shape (N+1, n)")
    return [array[i].copy() for i in range(count)]


def _input_bounds(lower, upper, width):
    if lower is None and upper is None:
        return np.zeros(0), np.zeros(0)
    lower = np.full(width, -np.inf if lower is None else float(lower))
    upper = np.full(width, np.inf if upper is None else float(upper))
    if np.isinf(lower).any() or np.isinf(upper).any():
        return np.zeros(0), np.zeros(0)
    return lower, upper
