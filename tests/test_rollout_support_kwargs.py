"""Planner payload kwargs must match compute_rollout_contact_via."""

import ast
import inspect
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _supported_call_kwargs(fn, kwargs):
    params = inspect.signature(fn).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in params}


def _function_arg_names(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [arg.arg for arg in node.args.args]
    raise AssertionError("function %s not found in %s" % (name, path))


def test_compute_rollout_contact_via_accepts_support_plane():
    # handle_mpc_request always forwards support_point/support_normal,
    # including None on a flat Isaac table.  Missing kwargs raise TypeError
    # in the planner worker before ranking starts.
    names = _function_arg_names(
        ROOT / "examples/mpc/fingertips/test/test_0902.py",
        "compute_rollout_contact_via",
    )
    assert "support_point" in names
    assert "support_normal" in names


def test_handle_mpc_request_filters_support_plane_through_helper():
    src = (ROOT / "planning/mpc_explicit.py").read_text(encoding="utf-8")
    assert "def _supported_call_kwargs" in src
    assert "_supported_call_kwargs(" in src
    assert "support_point=support_point" in src
    assert "support_normal=support_normal" in src


def test_supported_call_kwargs_drops_unknown_support_point():
    def old_via(param, floor_ground=0.012, floor_z=0.0):
        return floor_z

    filtered = _supported_call_kwargs(
        old_via,
        dict(floor_ground=0.4, floor_z=0.4, support_point=None, support_normal=None),
    )
    assert filtered == {"floor_ground": 0.4, "floor_z": 0.4}
    assert old_via(None, **filtered) == 0.4


def test_supported_call_kwargs_keeps_support_point_when_declared():
    def new_via(param, floor_z=0.0, support_point=None, support_normal=None):
        return support_point

    point = [0.0, 0.0, 0.4]
    filtered = _supported_call_kwargs(
        new_via,
        dict(floor_z=0.4, support_point=point, support_normal=[0.0, 0.0, 1.0]),
    )
    assert filtered["support_point"] is point
    assert new_via(None, **filtered) is point
