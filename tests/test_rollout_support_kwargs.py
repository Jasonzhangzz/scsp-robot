"""Planner payload kwargs must not TypeError on compute_rollout_contact_via."""

import ast
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from planning.mpc_explicit import _call_rollout_contact_via, _supported_call_kwargs


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


def _old_via(
    param,
    args,
    curr_q,
    r_obj_to_world,
    gravity,
    jac_mat_env,
    fingertip_radius,
    value_tracker,
    model_cost_conf,
    approach_via,
    arrived_hold,
    arrived_dest_idx,
    floor_ground=0.012,
    floor_z=0.0,
):
    return {
        "arrived_hold": arrived_hold,
        "arrived_dest_idx": arrived_dest_idx,
        "verify_cost": 0.0,
        "mpc_virtual_point": np.zeros(3),
        "mpc_contact_point": np.zeros(3),
        "value_info": {},
        "floor_z": floor_z,
        "floor_ground": floor_ground,
    }


def _new_via(*args, floor_ground=0.012, floor_z=0.0, support_point=None, support_normal=None, **kwargs):
    return {
        "support_point": support_point,
        "support_normal": support_normal,
        "floor_z": floor_z,
    }


def _via_args():
    return (
        None, None, None, None, None, None, 0.01,
        None, None, None, False, None,
    )


def test_unfiltered_old_via_raises_the_isaac_typeerror():
    # This is the planner-worker crash on 20260920 /home/zz/scsp-robot.
    with pytest.raises(TypeError, match="unexpected keyword argument 'support_point'"):
        _old_via(
            *_via_args(),
            floor_ground=0.4,
            floor_z=0.4,
            support_point=None,
            support_normal=None,
        )


def test_filtered_call_survives_old_via_with_support_point():
    policy = _call_rollout_contact_via(
        _old_via,
        *_via_args(),
        floor_ground=0.4,
        floor_z=0.4,
        support_point=None,
        support_normal=None,
    )
    assert policy["floor_z"] == 0.4
    assert policy["floor_ground"] == 0.4


def test_filtered_call_forwards_support_point_when_declared():
    point = np.array([0.0, 0.0, 0.4])
    normal = np.array([0.0, 0.0, 1.0])
    policy = _call_rollout_contact_via(
        _new_via,
        *_via_args(),
        floor_ground=0.4,
        floor_z=0.4,
        support_point=point,
        support_normal=normal,
    )
    assert policy["support_point"] is point
    assert policy["support_normal"] is normal


def test_supported_call_kwargs_drops_unknown_support_point():
    filtered = _supported_call_kwargs(
        _old_via,
        dict(floor_ground=0.4, floor_z=0.4, support_point=None, support_normal=None),
    )
    assert filtered == {"floor_ground": 0.4, "floor_z": 0.4}


def test_compute_rollout_contact_via_accepts_support_plane():
    names = _function_arg_names(
        ROOT / "examples/mpc/fingertips/test/test_0902.py",
        "compute_rollout_contact_via",
    )
    assert "support_point" in names
    assert "support_normal" in names


def test_handle_mpc_request_filters_support_plane_through_helper():
    src = (ROOT / "planning/mpc_explicit.py").read_text(encoding="utf-8")
    assert "def _supported_call_kwargs" in src
    assert "def _call_rollout_contact_via" in src
    assert "_call_rollout_contact_via(" in src
    names = set(_call_keyword_names(
        ROOT / "planning/mpc_explicit.py",
        "handle_mpc_request",
        "_call_rollout_contact_via",
    ))
    assert "support_point" in names
    assert "support_normal" in names


def test_mppi_planner_filters_support_plane_through_helper():
    names = set(_call_keyword_names(
        ROOT / "planning/MPPIWarp.py",
        "handle_planner_request",
        "_call_rollout_contact_via",
    ))
    assert "support_point" in names
    assert "support_normal" in names


def test_rollout_via_forwards_support_plane_to_ranking():
    names = set(_call_keyword_names(
        ROOT / "examples/mpc/fingertips/test/test_0902.py",
        "compute_rollout_contact_via",
        "get_availble_point_idx",
    ))
    assert "support_point" in names
    assert "support_normal" in names
    assert "floor_z" in names


def test_handle_mpc_request_survives_old_via(monkeypatch):
    """Execute the planner entry that Isaac spawns, with the old helper."""
    import planning.mpc_explicit as mpc_explicit
    from examples.mpc.franka.ik2 import contact_frames

    captured = {}

    def old_via(*args, floor_ground=0.012, floor_z=0.0):
        captured["floor_z"] = floor_z
        captured["floor_ground"] = floor_ground
        captured["nargs"] = len(args)
        return {
            "arrived_hold": False,
            "arrived_dest_idx": None,
            "verify_cost": 1.25,
            "mpc_virtual_point": np.array([0.45, 0.0, 0.36]),
            "mpc_contact_point": np.array([0.46, 0.0, 0.36]),
            "value_info": {},
        }

    fake_0902 = types.ModuleType("examples.mpc.fingertips.test.test_0902")
    fake_0902.compute_rollout_contact_via = old_via
    fake_0902._verify_is_chatter = lambda prev, cur: False
    monkeypatch.setitem(
        sys.modules,
        "examples.mpc.fingertips.test.test_0902",
        fake_0902,
    )

    fake_mppi = types.ModuleType("planning.MPPIWarp")
    fake_mppi._apply_dwell_payload = lambda *a, **k: None
    fake_mppi._opt_snapshot = lambda opt: {}
    fake_mppi._pickle_safe = lambda value: value
    fake_mppi.clamp_via_to_tip = lambda tip, pt, lead: pt
    monkeypatch.setitem(sys.modules, "planning.MPPIWarp", fake_mppi)

    monkeypatch.setattr(
        contact_frames,
        "gravity_wrench_object_frame",
        lambda *a, **k: np.zeros(6),
    )

    class _Opt:
        m = 0.2
        acados_failure_count = 0
        acados_solve_count = 0

    class _Param:
        table_height = 0.4
        gravity_ = np.array([0.0, 0.0, -9.8])
        lambda_optimizer = _Opt()
        target_p_ = np.zeros(3)
        target_q_ = np.array([1.0, 0.0, 0.0, 0.0])
        max_ncon_ = 10
        n_qvel_ = 9
        mu_object_ = 0.5
        lambda_obj_mass_ = 0.2

    class _Mpc:
        acados_failure_count = 0
        acados_solve_count = 0
        _acados_init_error = ""

        def plan_once(self, *args, **kwargs):
            return {
                "action": np.array([0.01, 0.0, 0.0]),
                "sol_guess": None,
                "cost_opt": 0.0,
            }

    class _Conf:
        accum = 0.0

        def tightness(self):
            return 0.0

    args = types.SimpleNamespace(via_max_lead=0.005, mpc_step_limit=0.005)
    trackers = {
        "value_tracker": object(),
        "model_cost_conf": _Conf(),
        "approach_via": object(),
        "arrived_hold": False,
        "arrived_dest_idx": None,
        "sol_guess": None,
        "last_verify_cost": None,
    }
    msg = {
        "policy_q": np.array(
            [0.45, 0.0, 0.42, 1.0, 0.0, 0.0, 0.0, 0.40, 0.0, 0.41],
            dtype=np.float32,
        ),
        "table_ground": 0.412,
        "floor_z": 0.4,
        "support_point": None,
        "support_normal": None,
        "jac_mat_env": np.zeros((40, 9)),
        "phi_vec": np.ones(40),
        "jac_mat": np.zeros((40, 9)),
        "if_contact": False,
    }
    result = mpc_explicit.handle_mpc_request(args, _Param(), _Mpc(), trackers, msg)
    assert captured["floor_z"] == 0.4
    assert captured["floor_ground"] == 0.412
    assert np.allclose(result["action"], [0.01, 0.0, 0.0])
    assert result["verify_cost"] == pytest.approx(1.25)
