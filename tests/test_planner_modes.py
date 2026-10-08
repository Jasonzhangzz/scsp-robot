"""Lightweight checks for planner-mode aliases and explicit-MPC layouts."""

from types import SimpleNamespace

import numpy as np

from planning.mpc_costs import infer_cost_kind
from planning.mpc_explicit import (
    MPCExplicit,
    MPCExplicitEE,
    MPCExplicitJoint,
    _configure_explicit_layout,
)
from planning.planner_mode import resolve_planner_mode


def test_resolve_planner_mode_defaults_to_mppi_ee():
    assert resolve_planner_mode(SimpleNamespace()) == "mppi_ee"
    assert resolve_planner_mode(SimpleNamespace(planner_mode="mppi")) == "mppi_ee"
    assert resolve_planner_mode(SimpleNamespace(planner_mode="mppi-joint")) == "mppi_joint"
    assert resolve_planner_mode(SimpleNamespace(planner_mode="mpc")) == "mpc_ee"
    assert resolve_planner_mode(SimpleNamespace(planner_mode="mpc-joint")) == "mpc_joint"


def test_resolve_planner_mode_maps_legacy_flags():
    assert resolve_planner_mode(SimpleNamespace(planner_backend="explicit_mjwp")) == "mppi_joint"
    assert resolve_planner_mode(SimpleNamespace(planner_solver="acados")) == "mpc_ee"
    assert resolve_planner_mode(SimpleNamespace(planner_mode="mppi_ee", planner_solver="acados")) == "mppi_ee"


def test_infer_cost_kind_from_layout():
    assert infer_cost_kind(SimpleNamespace(n_qpos_=13, n_cmd_=6, planner_solver_="acados")) == "bigrasp"
    assert infer_cost_kind(SimpleNamespace(n_qpos_=13, n_cmd_=6, mpc_cost_kind="bigrasp_gs")) == "bigrasp_gs"
    assert infer_cost_kind(SimpleNamespace(n_qpos_=21, n_cmd_=12, planner_solver_="acados")) == "bigrasp_ee"
    assert infer_cost_kind(SimpleNamespace(n_qpos_=21, n_cmd_=14, planner_solver_="acados")) == "bigrasp_joint"


def test_configure_explicit_layout_sets_cmd_dims():
    ee = SimpleNamespace(
        obj_inertia_=np.eye(6, dtype=np.float32),
        robot_stiff_=np.eye(12, dtype=np.float32),
        planner_cmd_limit=0.04,
        planner_joint_delta_limit=0.2,
        Q=np.zeros((0, 0)),
    )
    _configure_explicit_layout(ee, n_qpos=13, n_qvel=12, n_cmd=6, n_robot_qpos=6, cost_kind="bigrasp")
    assert ee.n_cmd_ == 6
    assert ee.n_qpos_ == 13
    assert ee.Q.shape == (12, 12)
    assert ee.mpc_u_ub_.shape == (6,)

    joint = SimpleNamespace(
        obj_inertia_=np.eye(6, dtype=np.float32),
        robot_stiff_=np.eye(12, dtype=np.float32),
        planner_cmd_limit=0.04,
        planner_joint_delta_limit=0.2,
        Q=np.zeros((0, 0)),
    )
    _configure_explicit_layout(joint, n_qpos=21, n_qvel=20, n_cmd=14, n_robot_qpos=14, cost_kind="bigrasp_joint")
    assert joint.n_cmd_ == 14
    assert joint.n_qvel_ == 20
    assert joint.Q.shape == (20, 20)
    assert joint.mpc_cost_kind == "bigrasp_joint"


def test_explicit_mpc_exposes_reset():
    assert callable(getattr(MPCExplicit, "reset", None))
    assert callable(getattr(MPCExplicitEE, "reset", None))
    assert callable(getattr(MPCExplicitJoint, "reset", None))
    planner = MPCExplicit.__new__(MPCExplicit)
    planner.reset()
    assert planner.acados_solve_count == 0
    assert planner.acados_failure_count == 0
    assert planner.acados_fallback_count == 0
