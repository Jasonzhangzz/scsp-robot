"""CPU contract checks for explicit-contact MPPI, plus an optional GPU step test."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from planning.mppi_explicit import (
    PLAN_ONCE_KEYS,
    ExplicitBimanualMPPI,
    ExplicitEEMPPI,
    ExplicitJointMPPI,
)


def test_plan_once_contract_keys():
    assert PLAN_ONCE_KEYS == (
        "action",
        "ctrl",
        "cost",
        "cost_opt",
        "solve_time",
        "solver_backend",
        "contact_mask",
        "normal_force",
    )
    assert ExplicitBimanualMPPI is ExplicitJointMPPI
    assert ExplicitJointMPPI.plan_once.__code__.co_varnames[:5] == (
        "self",
        "contact_points_local",
        "target_object_pos",
        "target_object_quat",
        "approach_q",
    )
    assert ExplicitEEMPPI.action_dim == 12
    assert ExplicitJointMPPI.action_dim == 14
    assert ExplicitJointMPPI.solver_backend == "mppi_joint"
    assert ExplicitEEMPPI.solver_backend == "mppi_ee"
    assert callable(getattr(ExplicitJointMPPI, "reset", None))
    assert callable(getattr(ExplicitEEMPPI, "reset", None))


def test_plan_once_result_shapes_match_delta_q_contract():
    horizon = 4
    result = {
        "action": np.zeros(14),
        "ctrl": np.zeros((horizon, 14)),
        "cost": 0.0,
        "cost_opt": 0.0,
        "solve_time": 0.0,
        "solver_backend": "mppi_joint",
        "contact_mask": np.zeros((horizon, 2), dtype=bool),
        "normal_force": np.zeros((horizon, 2)),
    }
    assert set(result) == set(PLAN_ONCE_KEYS)
    assert result["action"].shape == (14,)
    assert result["ctrl"].shape == (horizon, 14)


def test_ee_plan_once_result_shapes():
    horizon = 4
    result = {
        "action": np.zeros(12),
        "ctrl": np.zeros((horizon, 12)),
        "cost": 0.0,
        "cost_opt": 0.0,
        "solve_time": 0.0,
        "solver_backend": "mppi_ee",
        "contact_mask": np.zeros((horizon, 2), dtype=bool),
        "normal_force": np.zeros((horizon, 2)),
    }
    assert set(result) == set(PLAN_ONCE_KEYS)
    assert result["action"].shape == (12,)
    assert result["solver_backend"] == "mppi_ee"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Warp rollouts")
def test_explicit_model_warp_step_advances_qpos_without_mjwarp_step():
    mujoco = pytest.importorskip("mujoco")
    mjwarp = pytest.importorskip("mujoco_warp")
    from models.explicit_model_warp import ExplicitModelWarp

    xml = """
    <mujoco>
      <option timestep="0.002" integrator="Euler"/>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <body name="box" pos="0 0 0.2">
          <freejoint/>
          <geom type="box" size="0.05 0.05 0.05" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    stepper = ExplicitModelWarp(
        model,
        data,
        nworld=2,
        nconmax=16,
        njmax=32,
        device="cuda:0",
    )
    before = stepper.data_wp.qpos.numpy().copy()

    called = {"step": 0}

    def _blocked(*_args, **_kwargs):
        called["step"] += 1
        raise AssertionError("rollouts must not call mujoco_warp.step")

    original = mjwarp.step
    mjwarp.step = _blocked
    try:
        stepper.step()
    finally:
        mjwarp.step = original

    after = stepper.data_wp.qpos.numpy()
    assert called["step"] == 0
    assert after.shape == before.shape
    assert np.isfinite(after).all()
    assert not np.allclose(after, before)
