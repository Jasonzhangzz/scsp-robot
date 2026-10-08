"""Warp kernel tests for lambda seeds and BigRasp loss parity."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from models.comfree_gs_torch import GaussianCloud
from planning.lambda_contact_warp import pca_opposite_seeds
from planning.mpc_explicit_adam import BigraspGSCostWeights, evaluate_bigrasp_gs_cost


def test_pca_opposite_seeds_are_far():
    points = np.array(
        [
            [0.05, 0.0, 0.0],
            [0.04, 0.01, 0.0],
            [-0.05, 0.0, 0.0],
            [-0.04, -0.01, 0.0],
            [0.0, 0.01, 0.0],
        ],
        dtype=np.float64,
    )
    a, b = pca_opposite_seeds(points)
    assert float(np.linalg.norm(a - b)) > 0.08
    assert a[0] * b[0] < 0.0


def _maybe_import_warp():
    try:
        from planning.bigrasp_warp_loss import evaluate_bigrasp_gs_cost_wp
        from planning.warp_adam import ensure_warp

        wp = ensure_warp()
        return evaluate_bigrasp_gs_cost_wp, wp
    except Exception as exc:
        pytest.skip(f"warp is not available: {exc}")


def test_warp_loss_matches_torch_assigned_and_swap():
    evaluate_wp, _ = _maybe_import_warp()
    contact = torch.tensor([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
    normals = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    extras = {
        "phi": torch.zeros(2, 2),
        "normal": torch.zeros(2, 2, 3),
        "contact_force": torch.zeros(2, 2),
        "capsules": torch.tensor(
            [
                [[0.1, 0.0, 0.0], [0.07, 0.0, 0.0], [0.05, 0.0, 0.0], [-0.1, 0.0, 0.0], [-0.07, 0.0, 0.0], [-0.05, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [0.07, 0.0, 0.0], [0.05, 0.0, 0.0], [-0.1, 0.0, 0.0], [-0.07, 0.0, 0.0], [-0.05, 0.0, 0.0]],
            ]
        ),
    }
    cmd = torch.zeros(2, 6)
    ctx = {
        "contact_points_local": contact,
        "normals_local": normals,
        "target_object_pos": torch.tensor([0.0, 0.0, 0.1]),
        "target_object_quat": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    }
    weights = BigraspGSCostWeights(
        contact_attract=10.0,
        contact_depth=0.0,
        penetration=0.0,
        object_position=0.0,
        object_lateral=0.0,
        object_orientation=0.0,
        action=1.0,
        smooth=0.0,
        sync=0.0,
        force=0.0,
            swap=20.0,
            inter_arm=0.0,
            tip_sep=0.0,
            query_mm=1.0,
            terminal=10.0,
        )
    assigned = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, -0.05, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, -0.05, 0.0, 0.0],
        ]
    )
    swapped = assigned.clone()
    swapped[:, 7:13] = torch.tensor([ -0.05, 0.0, 0.0, 0.05, 0.0, 0.0])
    torch_ok = float(evaluate_bigrasp_gs_cost(assigned, extras, cmd, ctx, weights))
    torch_swap = float(evaluate_bigrasp_gs_cost(swapped, extras, cmd, ctx, weights))
    warp_ok = evaluate_wp(assigned, extras, cmd, ctx, weights, device="cpu")
    warp_swap = evaluate_wp(swapped, extras, cmd, ctx, weights, device="cpu")
    assert abs(warp_ok - torch_ok) < 1.0e-3
    assert abs(warp_swap - torch_swap) < 1.0e-2
    assert warp_ok < warp_swap


def test_extract_reads_contact_distance_and_seeds_phi():
    _, wp = _maybe_import_warp()
    from planning.bigrasp_warp_loss import ensure_kernels

    k = ensure_kernels()
    qpos = wp.array(np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32), dtype=float)
    body_pos = wp.zeros((1, 1), dtype=wp.vec3)
    body_mat = wp.zeros((1, 1), dtype=wp.mat33)
    eye = np.zeros((1, 1, 3, 3), dtype=np.float32)
    eye[0, 0] = np.eye(3, dtype=np.float32)
    body_mat.assign(eye)
    distance = wp.array(np.array([[0.25, -0.5]], dtype=np.float32), dtype=float)
    frame = np.zeros((1, 2, 3, 3), dtype=np.float32)
    frame[0, 0, 0, :] = (0.0, 0.0, 1.0)
    frame[0, 1, 0, :] = (1.0, 0.0, 0.0)
    contact_frame = wp.array(frame, dtype=wp.mat33)
    phi = wp.zeros((1, 2), dtype=float)
    inward = wp.zeros((1, 2), dtype=wp.vec3)
    obj_pos = wp.zeros(1, dtype=wp.vec3)
    obj_quat = wp.zeros(1, dtype=wp.vec4)
    left = wp.zeros(1, dtype=wp.vec3)
    right = wp.zeros(1, dtype=wp.vec3)
    capsules = wp.zeros((1, 6), dtype=wp.vec3)
    job = k.ExtractJob()
    job.qpos = qpos
    job.body_pos = body_pos
    job.body_mat = body_mat
    job.obj_qpos = 0
    job.left_tip = 0
    job.right_tip = 0
    job.left_forearm = 0
    job.right_forearm = 0
    job.left_wrist = 0
    job.right_wrist = 0
    job.left_off = wp.vec3(0.0, 0.0, 0.06)
    job.right_off = wp.vec3(0.0, 0.0, 0.06)
    job.contact_distance = distance
    job.contact_frame = contact_frame
    job.left_row = 0
    job.right_row = 1
    job.step = 0
    job.obj_pos = obj_pos
    job.obj_quat = obj_quat
    job.left = left
    job.right = right
    job.capsules = capsules
    job.phi = phi
    job.inward = inward
    wp.launch(k.extract_step, dim=1, inputs=[job])
    g_phi = wp.array(np.array([[2.0, -3.0]], dtype=np.float32), dtype=float)
    g_left = wp.zeros(1, dtype=wp.vec3)
    g_right = wp.zeros(1, dtype=wp.vec3)
    g_obj_pos = wp.zeros(1, dtype=wp.vec3)
    g_obj_quat = wp.zeros(1, dtype=wp.vec4)
    g_capsules = wp.zeros((1, 6), dtype=wp.vec3)
    contact_grad = wp.zeros((1, 2), dtype=float)
    qpos_grad = wp.zeros((1, 7), dtype=float)
    body_pos_grad = wp.zeros((1, 1), dtype=wp.vec3)
    body_mat_grad = wp.zeros((1, 1), dtype=wp.mat33)
    seed = k.SeedJob()
    seed.step = 0
    seed.obj_qpos = 0
    seed.left_tip = 0
    seed.right_tip = 0
    seed.left_forearm = 0
    seed.right_forearm = 0
    seed.left_wrist = 0
    seed.right_wrist = 0
    seed.left_off = wp.vec3(0.0, 0.0, 0.0)
    seed.right_off = wp.vec3(0.0, 0.0, 0.0)
    seed.g_obj_pos = g_obj_pos
    seed.g_obj_quat = g_obj_quat
    seed.g_left = g_left
    seed.g_right = g_right
    seed.g_phi = g_phi
    seed.g_capsules = g_capsules
    seed.inward = inward
    seed.contact_grad = contact_grad
    seed.left_row = 0
    seed.right_row = 1
    seed.qpos_grad = qpos_grad
    seed.body_pos_grad = body_pos_grad
    seed.body_mat_grad = body_mat_grad
    wp.launch(k.seed_cotangent, dim=1, inputs=[seed])
    wp.synchronize()
    phi_h = np.asarray(phi.numpy(), dtype=np.float32).reshape(2)
    inward_h = np.asarray(inward.numpy(), dtype=np.float32).reshape(2, 3)
    grad_h = np.asarray(contact_grad.numpy(), dtype=np.float32).reshape(2)
    assert np.allclose(phi_h, [0.25, -0.5])
    assert np.allclose(inward_h[0], [0.0, 0.0, 1.0])
    assert np.allclose(inward_h[1], [1.0, 0.0, 0.0])
    assert np.allclose(grad_h, [2.0, -3.0])


def test_lambda_warp_opposite_sides():
    if not os.environ.get("SCSP_TEST_DEXFORGE") and not os.environ.get("SCSP_TEST_WARP"):
        pytest.skip("Set SCSP_TEST_WARP=1 to run Warp lambda on CPU/GPU")
    try:
        from planning.lambda_contact_warp import LambdaContactWarp
    except Exception as exc:
        pytest.skip(f"warp lambda unavailable: {exc}")
    axis = np.linspace(-0.04, 0.04, 5, dtype=np.float32)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    pts = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    on_skin = np.max(np.abs(pts), axis=1) > 0.03
    cloud = GaussianCloud(pts[on_skin], radius=0.006)
    opt = LambdaContactWarp(cloud=cloud, adam_steps=8, adam_lr=0.03, device="cpu")
    points, normals, cost, score, margin = opt.choose_contact_set()
    assert points.shape == (2, 3)
    assert normals.shape == (2, 3)
    assert np.isfinite(points).all()
    assert float(np.linalg.norm(points[0] - points[1])) > 0.03
    _ = cost, score, margin
