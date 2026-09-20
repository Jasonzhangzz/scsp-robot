"""Planner payload kwargs must match compute_rollout_contact_via."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def test_handle_mpc_request_forwards_support_plane():
    src = (ROOT / "planning/mpc_explicit.py").read_text(encoding="utf-8")
    assert "support_point=support_point" in src
    assert "support_normal=support_normal" in src
