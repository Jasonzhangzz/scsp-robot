"""Warp MPPI contact selection and batched DexForge cost."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from models.comfree_gs_torch import GaussianCloud
from planning.mpc_explicit_adam import BigraspGSCostWeights, evaluate_bigrasp_gs_cost
from planning.mppi_dexforge import command_from_raw


def _box_cloud(half=0.04, n=4, radius=0.006):
    axis = np.linspace(-half, half, n, dtype=np.float32)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    pts = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    on_skin = np.max(np.abs(pts), axis=1) > (half * 0.75)
    return GaussianCloud(pts[on_skin], radius=radius)


def test_command_from_raw_is_scaled_tanh():
    raw = np.array([[0.0, 0.5], [-1.0, 2.0]], dtype=np.float64)
    cmd = command_from_raw(raw, 0.15)
    assert cmd.shape == (2, 2)
    assert np.allclose(cmd, 0.15 * np.tanh(raw))


def test_lambda_mppi_separates_surface_points():
    pytest.importorskip("warp")
    from planning.lambda_contact_warp import LambdaContactMPPI

    cloud = _box_cloud()
    opt = LambdaContactMPPI(
        cloud=cloud,
        device="cpu",
        num_samples=8,
        iterations=2,
        noise=0.004,
        temperature=0.2,
        tip_radius=0.01,
        min_pair_distance=0.03,
    )
    points, normals, cost, score, margin = opt.choose_contact_set()
    assert points.shape == (2, 3)
    assert normals.shape == (2, 3)
    assert np.isfinite(points).all()
    assert float(np.linalg.norm(points[0] - points[1])) > 0.03
    assert opt.last_grasp_result["witness_contact_forces_local"].shape == (2, 3)
    assert np.isfinite(cost)
    _ = score, margin


def test_batched_cost_matches_torch():
    pytest.importorskip("warp")
    from planning.bigrasp_warp_loss import evaluate_batched_cost_wp

    states = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, -0.05, 0.0, 0.0]],
        dtype=torch.float32,
    )
    extras = {
        "phi": torch.zeros(1, 2),
        "normal": torch.zeros(1, 2, 3),
        "contact_force": torch.zeros(1, 2),
    }
    cmd = torch.zeros(1, 14)
    ctx = {
        "contact_points_local": torch.tensor([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]]),
        "normals_local": torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    }
    weights = BigraspGSCostWeights(
        contact_attract=10.0,
        contact_depth=0.0,
        penetration=0.0,
        object_position=0.0,
        action=1.0,
        smooth=0.0,
        sync=0.0,
        force=0.0,
        swap=4.0,
        inter_arm=0.0,
        tip_sep=0.0,
        query_mm=1.0,
        terminal=0.0,
        approach_offset=0.0,
    )
    torch_cost = float(evaluate_bigrasp_gs_cost(states, extras, cmd, ctx, weights))
    warp_cost = evaluate_batched_cost_wp(states, extras, cmd, ctx, weights, device="cpu")
    assert warp_cost.shape == (1,)
    assert abs(float(warp_cost[0]) - torch_cost) < 1.0e-3


def test_plan_once_contract():
    if not os.environ.get("SCSP_TEST_DEXFORGE"):
        pytest.skip("Set SCSP_TEST_DEXFORGE=1 to compile the DexForge MPPI step")
    pytest.importorskip("warp")
    from planning.mppi_dexforge import BigraspWarpMPPI
    from planning.warp_adam import ensure_warp

    wp = ensure_warp()
    if not any(str(device).startswith("cuda") for device in wp.get_devices()):
        pytest.skip("DexForge MPPI plan_once needs CUDA")
    from models.dexforge_fast_step import DexForgeFastStep
    from models.dexforge_planner_scene import write_dexforge_planner_scene

    scene = os.environ.get("SCSP_TEST_DEXFORGE_XML")
    if not scene:
        pytest.skip("Set SCSP_TEST_DEXFORGE_XML to a planner scene")
    step = DexForgeFastStep(scene, device="cuda:0", nworld=2)
    planner = BigraspWarpMPPI(
        step,
        horizon=1,
        knot_substeps=1,
        num_samples=2,
        iterations=1,
        temperature=0.2,
        noise_scale=0.1,
        u_limit=0.15,
        weights=BigraspGSCostWeights(contact_attract=1.0, query_mm=1.0, terminal=0.0, force=0.0),
    )
    qpos = np.zeros(step.nq, dtype=np.float32)
    qpos[step._ids["obj_qpos"] + 3] = 1.0
    result = planner.plan_once(
        np.zeros(3, dtype=np.float32),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        contact_points_local=np.zeros((2, 3), dtype=np.float32),
        normals_local=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
        qpos0=qpos,
        qvel0=np.zeros(step.nv, dtype=np.float32),
        approach_offset=0.007,
    )
    assert result["action"].shape == (14,)
    assert np.isfinite(result["action"]).all()
    assert np.isfinite(result["solve_time"])
    assert result["solver_backend"] == "mppi_dexforge"
