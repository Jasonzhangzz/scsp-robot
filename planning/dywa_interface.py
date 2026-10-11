"""DyWA student policy as an algorithm for the Isaac Franka experiment.

The policy owns its camera and low-level controller, so it runs inside DyWA's
simulator. The object mesh, init/goal orientation, mass, friction, control
period, and success metric come from the same parameter builder as
``examples/mpc/franka/ik2/test_mpc_isaac.py``. World positions are shifted onto
DyWA's table while the relative translation and both quaternions are kept.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DYWA_ROOT = REPO_ROOT / "thirdparty" / "DyWA"
DYWA_SRC = DYWA_ROOT / "dywa" / "src"
DYWA_TRAIN = DYWA_ROOT / "dywa" / "exp" / "train"
DYWA_CFG = DYWA_SRC / "data" / "cfg"
DEFAULT_CKPT_DIR = DYWA_ROOT / "ckpts"
ACC_PYTHON = Path("/home/lab423/miniconda3/envs/acc/bin/python")

# DyWA table is 0.4 m by 1.0 m with margin_scale 0.95. Keep both poses inside
# the inner region so the object radius still fits after prevent_fall.
_TABLE_HALF_EXTENTS = np.array([0.12, 0.35], dtype=np.float64)
POLICY_INTERVAL = 0.02
SIM_DT = 0.002
ACTION_PERIOD = 10
# test_mpc_isaac.SVG_SCREENSHOT_CAMERA_POSITION / TARGET, relative to a Franka
# base at the origin. DyWA puts that base on the -X table edge.
_MPC_CAMERA_FROM_BASE = np.array([0.7, 0.0, 0.63], dtype=np.float64)
_MPC_CAMERA_TARGET_FROM_BASE = np.array([0.1, 0.0, 0.32], dtype=np.float64)


def place_pair_on_table(init_xy, target_xy, half_extents=_TABLE_HALF_EXTENTS):
    """Move a world-frame pair onto the table without changing its direction."""
    init_xy = np.asarray(init_xy, dtype=np.float64).reshape(2)
    target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
    half_extents = np.asarray(half_extents, dtype=np.float64).reshape(2)
    delta = target_xy - init_xy
    init_local = -0.5 * delta
    target_local = 0.5 * delta
    scale = 1.0
    for axis in range(2):
        need = max(abs(float(init_local[axis])), abs(float(target_local[axis])))
        limit = float(half_extents[axis])
        if need > limit and need > 1e-9:
            scale = min(scale, limit / need)
    return init_local * scale, target_local * scale


def viewer_camera_for_scene(table_pos, table_dims, keepout_radius, base_height):
    """Place the MPC viewer on DyWA's Franka base."""
    table_pos = np.asarray(table_pos, dtype=np.float64).reshape(3)
    table_dims = np.asarray(table_dims, dtype=np.float64).reshape(3)
    if base_height == "table":
        base_z = float(table_dims[2])
    elif base_height == "ground":
        base_z = 0.0
    else:
        raise KeyError(f"Unknown base_height = {base_height}")
    base = np.array([
        float(table_pos[0] - 0.5 * table_dims[0] - float(keepout_radius)),
        0.0,
        base_z,
    ], dtype=np.float64)
    return base + _MPC_CAMERA_FROM_BASE, base + _MPC_CAMERA_TARGET_FROM_BASE


def look_at_rotation_and_translation(eye, at, up):
    """Match pytorch3d.renderer.cameras.look_at_view_transform for an explicit eye.

    DyWA unprojects depth with this rotation. The cross-product order and the
    final transpose are part of that convention.
    """
    def as_rows(value):
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, 3)
        return array.reshape(-1, 3)

    eye = as_rows(eye)
    at = as_rows(at)
    up = as_rows(up)
    count = max(eye.shape[0], at.shape[0], up.shape[0])
    eye = np.broadcast_to(eye, (count, 3)).copy()
    at = np.broadcast_to(at, (count, 3)).copy()
    up = np.broadcast_to(up, (count, 3)).copy()

    def normalize(vector):
        length = np.linalg.norm(vector, axis=-1, keepdims=True)
        return vector / np.maximum(length, 1e-5)

    z_axis = normalize(at - eye)
    x_axis = normalize(np.cross(up, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    degenerate = np.all(np.abs(x_axis) <= 5e-3, axis=-1, keepdims=True)
    if np.any(degenerate):
        x_axis = np.where(degenerate, normalize(np.cross(y_axis, z_axis)), x_axis)
    rotation = np.swapaxes(np.stack([x_axis, y_axis, z_axis], axis=1), 1, 2)
    translation = -np.einsum("nji,nj->ni", rotation, eye)
    return rotation, translation


def _as_list(values):
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


def build_experiment(args, trial_count):
    """Sample the same trial as the MPC Isaac entry and pack a DyWA spec."""
    os.environ.setdefault("SCSP_PLANNER_ONLY", "1")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from examples.mpc.franka.ik2.params import ExplicitMPCParams
    from examples.mpc.franka.ik2.test_mpc_isaac import (
        DYWA_PHYSX_POSITION_ITERATIONS,
        DYWA_PHYSX_VELOCITY_ITERATIONS,
        DYWA_SIM_DT,
        DYWA_SIM_SUBSTEPS,
        _apply_dywa_physics_to_param,
    )

    param = ExplicitMPCParams(
        args,
        rand_seed=int(trial_count),
        target_type=getattr(args, "target_type", "ground-rotation"),
        mpc_model="explicit",
    )
    param = _apply_dywa_physics_to_param(param)
    init_qpos = np.asarray(param.init_obj_qpos_, dtype=np.float64).reshape(7)
    target_p = np.asarray(param.target_p_, dtype=np.float64).reshape(3)
    target_q = np.asarray(param.target_q_, dtype=np.float64).reshape(4)
    init_xy, target_xy = place_pair_on_table(init_qpos[:2], target_p[:2])
    mapped_init = np.concatenate([init_xy, [float(init_qpos[2])], init_qpos[3:7]])
    mapped_target = np.concatenate([target_xy, [float(target_p[2])]])
    ckpt_dir = Path(getattr(args, "dywa_ckpt_dir", DEFAULT_CKPT_DIR)).expanduser()
    mesh_path = (REPO_ROOT / param.mesh_path_).resolve()
    if not mesh_path.is_file():
        mesh_path = Path(param.mesh_path_).expanduser().resolve()
    # The student outputs a Cartesian delta of at most 6 cm. Training holds
    # that delta for dt * action_period = 0.0125 * 8 = 0.1 s so the joint PD
    # can reach it. A 20 ms hold leaves the tool short of every command.
    action_window_s = 0.1
    period = max(1, int(round(action_window_s / max(float(DYWA_SIM_DT), 1e-6))))
    return {
        "trial": int(trial_count),
        "object": str(getattr(args, "obj", "foam_brick")),
        "mesh_path": str(mesh_path),
        "init_obj_qpos": _as_list(mapped_init),
        "target_p": _as_list(mapped_target),
        "target_q": _as_list(target_q),
        "source_init_obj_qpos": _as_list(init_qpos),
        "source_target_p": _as_list(target_p),
        "source_target_q": _as_list(target_q),
        "mass": float(param.sim_obj_mass_),
        "table_friction": float(param.table_friction_),
        "object_friction": float(param.object_friction_),
        "dt": float(DYWA_SIM_DT),
        "substeps": int(DYWA_SIM_SUBSTEPS),
        "action_period": int(period),
        "pos_iter": int(DYWA_PHYSX_POSITION_ITERATIONS),
        "vel_iter": int(DYWA_PHYSX_VELOCITY_ITERATIONS),
        "policy_interval_s": float(action_window_s),
        "max_steps": max(1, int(args.max_rollout_length)),
        "success_pos": 0.02,
        "success_quat": 0.015,
        "metric_gamma": float(getattr(args, "ranking_metric_gamma", 0.9)),
        "metric_eta": float(getattr(args, "ranking_metric_eta", 1.0)),
        "headless": bool(getattr(args, "headless", False)),
        "device": str(getattr(args, "sim_device", "cuda:0")),
        "graphics_device_id": int(getattr(args, "graphics_device_id", 0)),
        "seed": int(100 + trial_count),
        "ckpt_dir": str(ckpt_dir),
        "student_ckpt": str(ckpt_dir / "last.ckpt"),
        "teacher_ckpt": str(ckpt_dir / "teacher-last.ckpt"),
        "env_ckpt": str(ckpt_dir / "env-last.ckpt"),
        "icp_ckpt": str(ckpt_dir / "512-32-balanced-SAM-wd-5e-05-920.ckpt"),
        "output_dir": str(REPO_ROOT / "outputs" / "dywa_isaac" / f"trial_{int(trial_count):03d}"),
        "use_custom_fingertip": True,
    }


def _hydra_overrides(spec):
    init_q = ",".join(str(v) for v in spec["init_obj_qpos"])
    target_p = ",".join(str(v) for v in spec["target_p"])
    target_q = ",".join(str(v) for v in spec["target_q"])
    device = spec["device"]
    viewer = "false" if spec["headless"] else "true"
    return [
        "+platform=debug",
        "+env=abs_goal_1view",
        "+run=teacher_base",
        "+student=dywa/base",
        "++name=dywa",
        f"++path.root={spec['output_dir']}",
        "++env.num_env=1",
        f"++global_device={device}",
        "++student.norm=ln",
        f"++load_ckpt={spec['teacher_ckpt']}",
        f"++load_environment={spec['env_ckpt']}",
        f"+load_student={spec['student_ckpt']}",
        "++plot_pc=false",
        "++use_nvdr_record_episode=false",
        "++use_nvdr_record_viewer=false",
        "++add_teacher_state=true",
        "++student.decoder.film_mlp=true",
        "++dagger_train_env.agent_type=student",
        "++dagger_train_env.deterministic_action=true",
        "++dagger_train_env.anneal_step=1",
        "++log_categorical_results=false",
        f"++test_step={int(spec['max_steps'])}",
        f"++env.use_viewer={viewer}",
        f"++env.dt={spec['dt']}",
        f"++env.substeps={int(spec['substeps'])}",
        f"++env.action_period={int(spec['action_period'])}",
        f"++env.pos_iter={int(spec['pos_iter'])}",
        f"++env.vel_iter={int(spec['vel_iter'])}",
        f"++env.seed={int(spec['seed'])}",
        "++env.single_object_scene.base_set=[mesh]",
        f"++env.single_object_scene.mesh.filename={spec['mesh_path']}",
        f"++env.single_object_scene.mesh.mass={spec['mass']}",
        "++env.single_object_scene.mesh.num_poses=8",
        "++env.single_object_scene.mesh.num_points=256",
        f"++env.single_object_scene.mesh.cache_dir={spec['output_dir']}/pose_cache",
        "++env.single_object_scene.filter_file=null",
        "++env.single_object_scene.filter_complex=false",
        "++env.single_object_scene.num_object_types=1",
        "++env.single_object_scene.use_wall=false",
        "++env.single_object_scene.use_dr=false",
        "++env.single_object_scene.use_dr_on_setup=false",
        "++env.single_object_scene.use_scale_dr=false",
        "++env.single_object_scene.mode=valid",
        f"++env.single_object_scene.table_friction={spec['table_friction']}",
        f"++env.single_object_scene.object_friction={spec['object_friction']}",
        f"++env.single_object_scene.init_obj_qpos_=[{init_q}]",
        f"++env.task.target_p_=[{target_p}]",
        f"++env.task.target_q_=[{target_q}]",
        "++env.task.goal_radius=0.0001",
        "++env.task.goal_angle=0.0001",
        f"++env.task.timeout={int(spec['max_steps']) + 10}",
        "++env.franka.use_custom_end_effector=true",
        "++env.franka.add_control_noise=false",
        f"++icp_obs.icp.ckpt={spec['icp_ckpt']}",
        "++use_wandb=false",
    ]


def _install_flash_attention_api():
    """Provide the FlashAttention 1.x classes DyWA imports.

    The acc environment's flash-attn 2.7 extension was built for a different
    libtorch and does not expose ``flash_attn.flash_attention``.
    """
    import types

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from einops import rearrange

    class FlashAttention(nn.Module):
        def __init__(self, softmax_scale=None, attention_dropout=0.0):
            super().__init__()
            self.softmax_scale = softmax_scale
            self.dropout_p = float(attention_dropout)

        def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                    max_s=None, need_weights=False):
            if cu_seqlens is not None:
                raise NotImplementedError("packed FlashAttention sequences are not used here")
            query, key, value = qkv.unbind(dim=2)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            attn_mask = None
            if key_padding_mask is not None:
                attn_mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            dropout = self.dropout_p if self.training else 0.0
            context = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout,
                is_causal=bool(causal) and attn_mask is None,
                scale=self.softmax_scale,
            )
            return context.transpose(1, 2).contiguous(), None

    class FlashMHA(nn.Module):
        def __init__(self, embed_dim, num_heads, bias=True, batch_first=True,
                     attention_dropout=0.0, causal=False, device=None, dtype=None):
            if not batch_first:
                raise ValueError("FlashMHA shim expects batch_first=True")
            super().__init__()
            self.causal = causal
            self.num_heads = num_heads
            self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, device=device, dtype=dtype)
            self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, device=device, dtype=dtype)

        def forward(self, x, key_padding_mask=None, need_weights=False):
            qkv = rearrange(self.Wqkv(x), "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
            context, weights = self.inner_attn(
                qkv, key_padding_mask=key_padding_mask,
                need_weights=need_weights, causal=self.causal,
            )
            return self.out_proj(rearrange(context, "b s h d -> b s (h d)")), weights

    package = types.ModuleType("flash_attn")
    package.__path__ = []
    module = types.ModuleType("flash_attn.flash_attention")
    module.FlashAttention = FlashAttention
    module.FlashMHA = FlashMHA
    package.flash_attention = module
    sys.modules["flash_attn"] = package
    sys.modules["flash_attn.flash_attention"] = module


def _install_pytorch3d_ops():
    """Replace PyTorch3D CUDA ops that were built for a different libtorch.

    DyWA only needs KNN, farthest-point sampling, Chamfer, and ICP on the
    student path. The installed ``pytorch3d._C`` does not load in acc.
    """
    import types
    from collections import namedtuple

    import pytorch3d
    import torch

    KNN = namedtuple("KNN", "dists idx knn")
    SimilarityTransform = namedtuple("SimilarityTransform", "R T s")
    ICPSolution = namedtuple("ICPSolution", "converged rmse Xt RTs t_history")

    def knn_gather(x, idx, lengths=None):
        del lengths
        batch, _, _ = idx.shape
        flat = idx.reshape(batch, -1)
        gathered = torch.gather(
            x, 1, flat[..., None].expand(-1, -1, x.shape[-1])
        )
        return gathered.reshape(batch, idx.shape[1], idx.shape[2], x.shape[-1])

    def knn_points(p1, p2, lengths1=None, lengths2=None, K=1, version=-1,
                   norm=2, return_nn=False, return_sorted=True):
        del lengths1, version
        if norm not in (1, 2):
            raise ValueError("knn norm must be 1 or 2")
        distance = torch.cdist(p1.float(), p2.float(), p=float(norm))
        if norm == 2:
            distance = distance.square()
        if lengths2 is not None:
            valid = torch.arange(p2.shape[1], device=p2.device)[None] < lengths2[:, None]
            distance = distance.masked_fill(~valid[:, None, :], 1e10)
        dists, idx = torch.topk(
            distance, k=int(K), dim=-1, largest=False, sorted=bool(return_sorted)
        )
        dists = dists.to(dtype=p1.dtype)
        neighbors = knn_gather(p2, idx) if return_nn else None
        return KNN(dists, idx, neighbors)

    def ball_query(p1, p2, lengths1=None, lengths2=None, K=500, radius=0.2,
                   return_nn=True):
        distance = torch.cdist(p1.float(), p2.float(), p=2).square()
        if lengths2 is not None:
            valid = torch.arange(p2.shape[1], device=p2.device)[None] < lengths2[:, None]
            distance = distance.masked_fill(~valid[:, None, :], float("inf"))
        distance = distance.masked_fill(distance > float(radius) ** 2, float("inf"))
        k_count = min(int(K), p2.shape[1])
        dists, idx = torch.topk(distance, k=k_count, dim=-1, largest=False, sorted=True)
        if k_count < int(K):
            pad = int(K) - k_count
            dists = torch.nn.functional.pad(dists, (0, pad), value=0.0)
            idx = torch.nn.functional.pad(idx, (0, pad), value=-1)
        missing = ~torch.isfinite(dists)
        dists = dists.masked_fill(missing, 0.0).to(dtype=p1.dtype)
        idx = idx.masked_fill(missing, -1)
        if lengths1 is not None:
            valid1 = torch.arange(p1.shape[1], device=p1.device)[None] < lengths1[:, None]
            dists = dists.masked_fill(~valid1[:, :, None], 0.0)
            idx = idx.masked_fill(~valid1[:, :, None], -1)
        neighbors = None
        if return_nn:
            neighbors = knn_gather(p2, idx.clamp(min=0))
            neighbors = neighbors.masked_fill((idx < 0)[..., None], 0.0)
        return dists, idx, neighbors

    def sample_farthest_points(points, lengths=None, K=50, random_start_point=False):
        batch, count, _ = points.shape
        if torch.is_tensor(K):
            k_count = int(K.max().item()) if K.ndim else int(K.item())
        elif isinstance(K, (list, tuple)):
            k_count = max(int(v) for v in K)
        else:
            k_count = int(K)
        k_count = max(1, min(k_count, count))
        chosen = torch.zeros(batch, k_count, dtype=torch.long, device=points.device)
        if random_start_point:
            limit = count if lengths is None else lengths
            chosen[:, 0] = (torch.rand(batch, device=points.device) * limit).long().clamp(max=count - 1)
        nearest = torch.full((batch, count), 1e10, device=points.device, dtype=points.dtype)
        for index in range(1, k_count):
            center = points[torch.arange(batch, device=points.device), chosen[:, index - 1]]
            nearest = torch.minimum(nearest, torch.sum((points - center[:, None, :]) ** 2, dim=-1))
            if lengths is not None:
                valid = torch.arange(count, device=points.device)[None] < lengths[:, None]
                nearest = nearest.masked_fill(~valid, -1.0)
            chosen[:, index] = nearest.argmax(dim=-1)
        sampled = points[torch.arange(batch, device=points.device)[:, None], chosen]
        return sampled, chosen

    def chamfer_distance(x, y, **_kwargs):
        forward = knn_points(x, y, K=1).dists[..., 0].mean()
        backward = knn_points(y, x, K=1).dists[..., 0].mean()
        return forward + backward, None

    def iterative_closest_point(X, Y, init_transform=None, max_iterations=100,
                                relative_rmse_thr=1e-6, estimate_scale=False,
                                allow_reflection=False, verbose=False):
        del init_transform, estimate_scale, allow_reflection, verbose
        batch = X.shape[0]
        rotation = torch.eye(3, device=X.device, dtype=X.dtype).expand(batch, 3, 3).clone()
        translation = torch.zeros(batch, 3, device=X.device, dtype=X.dtype)
        scale = torch.ones(batch, device=X.device, dtype=X.dtype)
        current = X
        previous = None
        rmse = current.new_zeros(())
        for _ in range(int(max_iterations)):
            neighbor = torch.cdist(current, Y).argmin(dim=-1)
            matched = Y.gather(1, neighbor[..., None].expand(-1, -1, Y.shape[-1]))
            source_mean = current.mean(dim=1)
            target_mean = matched.mean(dim=1)
            covariance = (current - source_mean[:, None]).transpose(1, 2) @ (matched - target_mean[:, None])
            left, _, right = torch.linalg.svd(covariance)
            right = right.clone()
            reflection = torch.det(right.transpose(-1, -2) @ left.transpose(-1, -2)) < 0
            right[reflection, -1] *= -1
            delta = right.transpose(-1, -2) @ left.transpose(-1, -2)
            shift = target_mean - (source_mean[:, None, :] @ delta).squeeze(1)
            current = current @ delta + shift[:, None, :]
            rotation = rotation @ delta
            translation = (translation[:, None, :] @ delta).squeeze(1) + shift
            rmse = (current - matched).square().mean()
            if previous is not None and abs(float(previous - rmse)) <= float(relative_rmse_thr) * (float(previous) + 1e-8):
                break
            previous = rmse.detach()
        return ICPSolution(
            True, rmse, current, SimilarityTransform(rotation, translation, scale), []
        )

    def look_at_view_transform(dist=1.0, elev=0.0, azim=0.0, degrees=True,
                               eye=None, at=((0, 0, 0),), up=((0, 1, 0),), device="cpu"):
        at_t = torch.as_tensor(at, dtype=torch.float32, device=device).reshape(-1, 3)
        up_t = torch.as_tensor(up, dtype=torch.float32, device=device).reshape(-1, 3)
        if eye is None:
            elev_t = torch.as_tensor(elev, dtype=torch.float32, device=device).reshape(-1)
            azim_t = torch.as_tensor(azim, dtype=torch.float32, device=device).reshape(-1)
            dist_t = torch.as_tensor(dist, dtype=torch.float32, device=device).reshape(-1)
            if degrees:
                elev_t = torch.deg2rad(elev_t)
                azim_t = torch.deg2rad(azim_t)
            eye_t = torch.stack([
                dist_t * torch.cos(elev_t) * torch.sin(azim_t),
                dist_t * torch.sin(elev_t),
                dist_t * torch.cos(elev_t) * torch.cos(azim_t),
            ], dim=-1) + at_t
        else:
            eye_t = torch.as_tensor(eye, dtype=torch.float32, device=device).reshape(-1, 3)
        count = max(eye_t.shape[0], at_t.shape[0], up_t.shape[0])
        eye_t = eye_t.expand(count, 3)
        at_t = at_t.expand(count, 3)
        up_t = up_t.expand(count, 3)
        rotation, translation = look_at_rotation_and_translation(
            eye_t.detach().cpu().contiguous().numpy(),
            at_t.detach().cpu().contiguous().numpy(),
            up_t.detach().cpu().contiguous().numpy(),
        )
        return (
            torch.as_tensor(rotation, device=eye_t.device, dtype=eye_t.dtype),
            torch.as_tensor(translation, device=eye_t.device, dtype=eye_t.dtype),
        )

    renderer = types.ModuleType("pytorch3d.renderer")
    renderer.look_at_view_transform = look_at_view_transform
    sys.modules["pytorch3d.renderer"] = renderer

    ops = types.ModuleType("pytorch3d.ops")
    knn_mod = types.ModuleType("pytorch3d.ops.knn")
    knn_mod.knn_points = knn_points
    knn_mod.knn_gather = knn_gather
    fps_mod = types.ModuleType("pytorch3d.ops.sample_farthest_points")
    fps_mod.sample_farthest_points = sample_farthest_points
    align_mod = types.ModuleType("pytorch3d.ops.points_alignment")
    align_mod.iterative_closest_point = iterative_closest_point
    align_mod.SimilarityTransform = SimilarityTransform
    align_mod.ICPSolution = ICPSolution
    ops.knn_points = knn_points
    ops.knn_gather = knn_gather
    ops.ball_query = ball_query
    ops.sample_farthest_points = sample_farthest_points
    ops.iterative_closest_point = iterative_closest_point
    loss = types.ModuleType("pytorch3d.loss")
    loss.chamfer_distance = chamfer_distance
    sys.modules["pytorch3d.ops"] = ops
    sys.modules["pytorch3d.ops.knn"] = knn_mod
    sys.modules["pytorch3d.ops.sample_farthest_points"] = fps_mod
    sys.modules["pytorch3d.ops.points_alignment"] = align_mod
    sys.modules["pytorch3d.loss"] = loss
    sys.modules["pytorch3d.loss.chamfer"] = types.ModuleType("pytorch3d.loss.chamfer")
    pytorch3d.ops = ops


def _activate_dywa_import_path():
    for path in (str(DYWA_TRAIN), str(DYWA_SRC)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    roots = (DYWA_SRC.resolve(), DYWA_TRAIN.resolve())
    conflicting = {"models", "envs", "env", "data", "util", "train", "cxx"}
    for name, module in list(sys.modules.items()):
        if name.split(".", 1)[0] not in conflicting:
            continue
        file_name = getattr(module, "__file__", None)
        if not file_name:
            del sys.modules[name]
            continue
        resolved = Path(file_name).resolve()
        if any(_is_relative_to(resolved, root) for root in roots):
            continue
        del sys.modules[name]


def _is_relative_to(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _xyzw_to_wxyz(quat_xyzw):
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def _object_and_goal(env):
    root = env.tensors["root"]
    actor = int(env.scene.cur_ids.reshape(-1)[0].item())
    pose = root[actor, :7].detach().cpu().numpy()
    goal = env.task.goal.reshape(-1, env.task.goal.shape[-1])[0, :7].detach().cpu().numpy()
    return pose, goal


def _compose_config(spec):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.types import ConvertMode
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(DYWA_CFG), version_base=None):
        cli_cfg = compose(config_name="show", overrides=_hydra_overrides(spec))
    from test_rma import Config

    default_cfg = OmegaConf.structured(Config)
    cli_cfg = instantiate(cli_cfg, _convert_=ConvertMode.PARTIAL)
    cfg = OmegaConf.to_object(OmegaConf.structured(OmegaConf.merge(default_cfg, cli_cfg)))
    gpu = int(str(spec["device"]).split(":")[-1])
    cfg.env.use_viewer = not bool(spec["headless"])
    scene = cfg.env.single_object_scene
    camera_origin, camera_target = viewer_camera_for_scene(
        scene.table_pos,
        scene.table_dims,
        cfg.env.franka.keepout_radius,
        cfg.env.franka.base_height,
    )
    cfg.env.viewer_camera_origin = tuple(float(v) for v in camera_origin)
    cfg.env.viewer_camera_target = tuple(float(v) for v in camera_target)
    cfg.env.graphics_device_id = int(spec.get("graphics_device_id", gpu))
    cfg.env.compute_device_id = gpu
    cfg.sync_frame_time = False
    return cfg


def _disable_student_update(student):
    def _no_update(*_args, **_kwargs):
        return None

    student.update_policy = _no_update
    student.eval()


def run_spec(spec):
    """Load the student and roll the matched trial. Isaac Gym must already be imported."""
    import torch as th
    from torch.utils.tensorboard import SummaryWriter

    os.environ.setdefault("PYTORCH_JIT", "0")
    _install_flash_attention_api()
    _install_pytorch3d_ops()
    _activate_dywa_import_path()
    from env.env.base import EnvBase
    from test_rma import update_net_cfg
    from train.ckpt import last_ckpt
    from train_ppo_arm import AddTensorboardWriter, load_agent, load_env, setup as setup_logging
    from train_rma import DAggerTrainerEnv, setup_rma_env_v2
    from distill import StudentAgentRMA
    from dataclasses import replace

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from utils import metrics
    from utils.ranking_metrics import discounted_terminal_reward

    for label, key in (
        ("mesh", "mesh_path"),
        ("student", "student_ckpt"),
        ("teacher", "teacher_ckpt"),
        ("environment stats", "env_ckpt"),
        ("ICP", "icp_ckpt"),
    ):
        if not Path(spec[key]).is_file():
            raise FileNotFoundError(f"DyWA {label} file is missing: {spec[key]}")

    cfg = _compose_config(spec)
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    writer = SummaryWriter(path.tb_train)
    cfg, env = load_env(cfg, path, freeze_env=True, check_viewer=False)
    env.unwrap(target=AddTensorboardWriter).set_writer(writer)
    cfg = replace(cfg, net=update_net_cfg(cfg.net, env, cfg.state_net_blocklist))
    teacher_agent = load_agent(cfg, env, None, None)
    teacher_agent.eval()
    student = StudentAgentRMA(
        replace(cfg.student, batch_size=env.num_env),
        writer=writer,
        device=env.device,
    ).to(env.device)
    student.load(last_ckpt(spec["student_ckpt"]), strict=False)
    _disable_student_update(student)
    env = setup_rma_env_v2(
        cfg,
        env,
        teacher_agent,
        state_size=128,
        is_student=False,
        dagger=True,
    )
    env.load(Path(spec["env_ckpt"]), strict=True)
    env = DAggerTrainerEnv(cfg.dagger_train_env, env, student)
    env.reset()

    success_pos = float(spec["success_pos"])
    success_quat = float(spec["success_quat"])
    max_steps = int(spec["max_steps"])
    min_pos = float("inf")
    min_quat = float("inf")
    pos_err = quat_err = None
    trial_success = False
    steps = 0
    choose_times = []
    base = env.unwrap(target=EnvBase)
    try:
        for _step in range(max_steps):
            started = time.perf_counter()
            _obs, _rew, done, info = env.step()
            choose_times.append(time.perf_counter() - started)
            pose, goal = _object_and_goal(base)
            pos_err = float(metrics.comp_pos_error(pose[:3], goal[:3]))
            quat_err = float(metrics.comp_quat_error(
                _xyzw_to_wxyz(pose[3:7]), _xyzw_to_wxyz(goal[3:7])))
            min_pos = min(min_pos, pos_err)
            min_quat = min(min_quat, quat_err)
            steps = _step + 1
            internal_success = bool(info.get("success", th.zeros(1))[0].item()) if info else False
            print(
                "dywa_step:", steps,
                "pos_err:", round(pos_err, 5),
                "quat_err:", round(quat_err, 5),
                "internal_success:", int(internal_success),
                flush=True,
            )
            if pos_err <= success_pos and quat_err <= success_quat:
                trial_success = True
                break
            if done is not None and bool(done.reshape(-1)[0].item()) and not trial_success:
                break
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
        writer.close()

    choose = np.asarray(choose_times, dtype=np.float64) if choose_times else np.array([0.0])
    summary = {
        "trial": int(spec["trial"]),
        "mode": "dywa",
        "object": spec["object"],
        "success": int(trial_success),
        "steps": int(steps),
        "completion_steps": int(steps) if trial_success else None,
        "terminal_reward": discounted_terminal_reward(
            trial_success, steps if trial_success else None,
            gamma=float(spec["metric_gamma"]), eta=float(spec["metric_eta"]),
        ),
        "metric_gamma": float(spec["metric_gamma"]),
        "metric_eta": float(spec["metric_eta"]),
        "policy_interval_s": float(spec["policy_interval_s"]),
        "final_pos_err": None if pos_err is None else round(float(pos_err), 5),
        "final_quat_err": None if quat_err is None else round(float(quat_err), 5),
        "min_pos_err": None if not np.isfinite(min_pos) else round(float(min_pos), 5),
        "min_quat_err": None if not np.isfinite(min_quat) else round(float(min_quat), 5),
        "choose_dt_mean": round(float(np.mean(choose)), 5),
        "choose_dt_max": round(float(np.max(choose)), 5),
        "mass": float(spec["mass"]),
        "table_friction": float(spec["table_friction"]),
        "object_friction": float(spec["object_friction"]),
        "dt": float(spec["dt"]),
        "action_period": int(spec["action_period"]),
    }
    result_path = spec.get("result_json")
    if result_path:
        Path(result_path).write_text(json.dumps(summary), encoding="utf-8")
    print("trial_summary: " + repr(summary), flush=True)
    return summary


def execute_trial(spec):
    """Run one trial in a fresh acc interpreter so Isaac Gym loads before torch."""
    python = ACC_PYTHON if ACC_PYTHON.is_file() else Path(sys.executable)
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "spec.json"
    result_path = output_dir / "summary.json"
    payload = dict(spec)
    payload["result_json"] = str(result_path)
    spec_path.write_text(json.dumps(payload), encoding="utf-8")
    if result_path.exists():
        result_path.unlink()
    child_env = os.environ.copy()
    child_env["PATH"] = str(python.parent) + os.pathsep + child_env.get("PATH", "")
    child_env.setdefault("PYTORCH_JIT", "0")
    completed = subprocess.run(
        [str(python), str(Path(__file__).resolve()), "--spec", str(spec_path)],
        cwd=str(REPO_ROOT),
        env=child_env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"DyWA trial {spec['trial']} exited with status {completed.returncode}. "
            f"See {output_dir}."
        )
    if not result_path.is_file():
        raise RuntimeError(f"DyWA trial {spec['trial']} did not write {result_path}.")
    return json.loads(result_path.read_text(encoding="utf-8"))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run one DyWA spec inside Isaac Gym.")
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    from isaacgym import gymapi, gymtorch  # noqa: F401
    import torch  # noqa: F401

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    run_spec(spec)


if __name__ == "__main__":
    main()
