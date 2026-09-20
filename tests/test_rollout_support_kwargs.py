"""Planner payload kwargs must match compute_rollout_contact_via."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function_node(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("function %s not found in %s" % (name, path))


def _function_arg_names(path, name):
    return [arg.arg for arg in _function_node(path, name).args.args]


def _call_keyword_names(path, caller_name, callee_name):
    caller = _function_node(path, caller_name)
    names = []
    for node in ast.walk(caller):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = None
        if isinstance(func, ast.Name):
            called = func.id
        elif isinstance(func, ast.Attribute):
            called = func.attr
        if called == callee_name:
            names.extend(keyword.arg for keyword in node.keywords if keyword.arg)
    if not names:
        raise AssertionError("%s does not call %s in %s" % (
            caller_name, callee_name, path))
    return names


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


def test_handle_mpc_request_kwargs_are_accepted():
    via_args = set(_function_arg_names(
        ROOT / "examples/mpc/fingertips/test/test_0902.py",
        "compute_rollout_contact_via",
    ))
    call_kwargs = set(_call_keyword_names(
        ROOT / "planning/mpc_explicit.py",
        "handle_mpc_request",
        "compute_rollout_contact_via",
    ))
    unknown = call_kwargs - via_args
    assert not unknown, "handle_mpc_request passes unsupported kwargs: %s" % (
        sorted(unknown),)
    assert "support_point" in call_kwargs
    assert "support_normal" in call_kwargs


def test_mppi_planner_kwargs_are_accepted():
    via_args = set(_function_arg_names(
        ROOT / "examples/mpc/fingertips/test/test_0902.py",
        "compute_rollout_contact_via",
    ))
    call_kwargs = set(_call_keyword_names(
        ROOT / "planning/MPPIWarp.py",
        "handle_planner_request",
        "compute_rollout_contact_via",
    ))
    unknown = call_kwargs - via_args
    assert not unknown, "handle_planner_request passes unsupported kwargs: %s" % (
        sorted(unknown),)
    assert "support_point" in call_kwargs
    assert "support_normal" in call_kwargs


def test_rollout_via_forwards_support_plane_to_ranking():
    names = set(_call_keyword_names(
        ROOT / "examples/mpc/fingertips/test/test_0902.py",
        "compute_rollout_contact_via",
        "get_availble_point_idx",
    ))
    assert "support_point" in names
    assert "support_normal" in names
    assert "floor_z" in names
