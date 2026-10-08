"""Torch comfree-GS collision and reduced-object dynamics.

Adam MPC imports this module so it never loads Warp / CUDA kernels, which
segfault when the MuJoCo OpenGL viewer already owns the GPU context.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# DexForge comfree-GS reduced model (object Gaussians + fingertip queries)
# ---------------------------------------------------------------------------
#
# Extracted from GraspSONIC/thirdparty/DexForge/forceaware/third_party/comfree_warp:
#   * gaussian sphere pair: d = |c - s| - r - r_s, inward n = unit(c - s),
#     contact = c - n r  (collision_targets / gaussian_collision)
#   * soft support: temperature-weighted minimum over the cloud
#     (smooth_contact.py uses a 1 mm complete-support field; softmax is the
#     Adam-friendly equivalent that stays differentiable at sphere switches)
#   * complementarity-free force: f = softplus(D (-b v - k pen))
#     (native_adjoint/explicit_damp.py + annealed_contact.py / this file's
#     closed-form spring-damper)
#

GS_SOFTMIN_TAU = 0.002
GS_SOFTPLUS_BETA = 80.0
TIP_QUERY_RADIUS = 0.01


def _dbg(location, message, data=None, hypothesis_id="A"):
    try:
        import json
        import time

        with open("/home/zz/scsp-robot/.cursor/debug-a4aad5.log", "a") as fh:
            fh.write(
                json.dumps(
                    {
                        "sessionId": "a4aad5",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data or {},
                        "timestamp": int(time.time() * 1000),
                    }
                )
                + "\n"
            )
    except OSError:
        return


def _as_tensor(value, device, dtype=None):
    import torch

    if dtype is None:
        dtype = torch.float32
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def quat_normalize_wxyz(quat):
    import torch

    return quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _broadcast_last3(a, b):
    import torch

    while a.ndim < b.ndim:
        a = a.unsqueeze(0)
    while b.ndim < a.ndim:
        b = b.unsqueeze(0)
    shape = torch.broadcast_shapes(a.shape, b.shape)
    return a.expand(shape), b.expand(shape)


def quat_rotate_wxyz(quat, vec):
    """Rotate ``vec`` by a wxyz quaternion.  Broadcasts over a leading batch."""
    import torch

    quat = quat_normalize_wxyz(quat)
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    xyz, vec = _broadcast_last3(xyz, vec)
    w, _ = _broadcast_last3(w, vec[..., :1])
    t = 2.0 * torch.linalg.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_integrate_wxyz(quat, omega, dt):
    """First-order body-frame quaternion integration used by ExplicitModel."""
    import torch

    quat = quat_normalize_wxyz(quat)
    w, x, y, z = quat.unbind(-1)
    # H_q_body.T @ omega, matching models/explicit_model.py
    dq0 = -x * omega[..., 0] - y * omega[..., 1] - z * omega[..., 2]
    dq1 = w * omega[..., 0] + z * omega[..., 1] - y * omega[..., 2]
    dq2 = -z * omega[..., 0] + w * omega[..., 1] + x * omega[..., 2]
    dq3 = y * omega[..., 0] - x * omega[..., 1] + w * omega[..., 2]
    next_q = quat + 0.5 * float(dt) * torch.stack((dq0, dq1, dq2, dq3), dim=-1)
    return quat_normalize_wxyz(next_q)


def orthonormal_tangent_frame(normal):
    import torch

    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1.0e-8)
    helper = torch.zeros_like(normal)
    helper[..., 0] = 1.0
    swap = normal[..., 0].abs() > 0.9
    helper = helper.masked_fill(swap.unsqueeze(-1), 0.0)
    helper[..., 1] = torch.where(swap, torch.ones_like(normal[..., 1]), helper[..., 1])
    tangent1 = torch.linalg.cross(normal, helper, dim=-1)
    tangent1 = tangent1 / torch.linalg.vector_norm(tangent1, dim=-1, keepdim=True).clamp_min(1.0e-8)
    tangent2 = torch.linalg.cross(normal, tangent1, dim=-1)
    return tangent1, tangent2


class GaussianCloud:
    """Object-frame Gaussian sphere cloud, built from mesh surface samples."""

    def __init__(self, centers, radii=None, radius=0.004):
        centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
        if centers.size == 0:
            raise ValueError("GaussianCloud requires at least one sphere")
        if radii is None:
            radii = np.full((centers.shape[0],), float(radius), dtype=np.float32)
        else:
            radii = np.asarray(radii, dtype=np.float32).reshape(-1)
            if radii.shape[0] == 1:
                radii = np.full((centers.shape[0],), float(radii[0]), dtype=np.float32)
            if radii.shape[0] != centers.shape[0]:
                raise ValueError("radii must match centers")
        self.centers = centers
        self.radii = np.maximum(radii, 1.0e-5)

    @classmethod
    def from_mesh(cls, mesh_path, scale=(1.0, 1.0, 1.0), sample_num=70, radius=0.004):
        from planning.project_point import ProjectionPoint

        projector = ProjectionPoint(str(mesh_path), scale_factors=list(scale))
        frames = projector.sample_vertices_with_normals(int(sample_num))
        return cls(frames["points"], radius=radius), projector, frames

    def as_spheres(self):
        return np.concatenate((self.centers, self.radii[:, None]), axis=1).astype(np.float32)


def gs_sphere_features(query, spheres, query_radius=TIP_QUERY_RADIUS, tau=GS_SOFTMIN_TAU):
    """DexForge Gaussian pair + softmax fusion.

    ``query``: (..., K, 3) world query centers (fingertips).
    ``spheres``: (N, 4) world (x, y, z, r).
    Returns distance, contact position, inward normal, each (..., K, 3) / (..., K).
    """
    import torch

    query = query.unsqueeze(-2)
    centers = spheres[..., :3]
    radii = spheres[..., 3]
    delta = centers - query
    pair = torch.linalg.vector_norm(delta, dim=-1)
    distance = pair - radii - float(query_radius)
    weights = torch.softmax(-distance / max(float(tau), 1.0e-6), dim=-1)
    phi = torch.sum(weights * distance, dim=-1)
    inward = delta / pair.unsqueeze(-1).clamp_min(1.0e-8)
    normal = torch.sum(weights.unsqueeze(-1) * inward, dim=-2)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1.0e-8)
    surface = centers - inward * radii.unsqueeze(-1)
    position = torch.sum(weights.unsqueeze(-1) * surface, dim=-2)
    return phi, position, normal


def transform_spheres(spheres_local, obj_pos, obj_quat):
    import torch

    local = spheres_local[..., :3]
    radii = spheres_local[..., 3]
    world = quat_rotate_wxyz(obj_quat.unsqueeze(-2), local) + obj_pos.unsqueeze(-2)
    return torch.cat((world, radii.unsqueeze(-1)), dim=-1)


def gs_contact_rows(phi, position, normal, obj_pos, tip, tip_jac_index):
    """Build one normal-row Jacobian in the 12-D (object twist + two tips) space."""
    import torch

    batch = phi.shape[:-1]
    jac = phi.new_zeros(*batch, 12)
    lever = position - obj_pos
    jac[..., 0:3] = normal
    jac[..., 3:6] = torch.linalg.cross(lever, normal, dim=-1)
    start = 6 + int(tip_jac_index) * 3
    jac[..., start:start + 3] = -normal
    return phi, jac


def table_contact_row(obj_pos, table_z, support_radius):
    import torch

    phi = obj_pos[..., 2] - float(table_z) - float(support_radius)
    jac = obj_pos.new_zeros(*obj_pos.shape[:-1], 12)
    jac[..., 2] = 1.0
    return phi, jac


def comfree_normal_force(phi, jac, v_pred, stiffness, damping, timestep, mass_scale=1.0):
    """Closed-form unilateral spring-damper from DexForge / ExplicitModelWarp."""
    import torch
    import torch.nn.functional as F

    efc_vel = torch.sum(jac * v_pred, dim=-1)
    penetration = efc_vel * float(timestep) + phi
    raw = float(mass_scale) * (
        -float(damping) * efc_vel - float(stiffness) * penetration
    )
    return F.softplus(raw * GS_SOFTPLUS_BETA) / GS_SOFTPLUS_BETA


class ComfreeGSModel:
    """Differentiable reduced-object + dual-fingertip stepper for Adam MPC.

    State ``x`` is 13-D ``[obj_pos, obj_quat_wxyz, left_tip, right_tip]``.
    Control ``u`` is 6-D fingertip increments.  Collision lives inside the
    step: two GS queries plus an optional table plane.
    """

    n_qpos = 13
    n_qvel = 12
    n_cmd = 6

    def __init__(
        self,
        cloud,
        *,
        obj_mass=0.2,
        gravity=(0.0, 0.0, -9.8),
        obj_inertia_pos=50.0,
        obj_inertia_rot=0.05,
        robot_stiffness=8.0,
        contact_stiffness=12.5,
        contact_damping=0.05,
        timestep=0.01,
        tip_radius=TIP_QUERY_RADIUS,
        table_z=None,
        support_radius=0.02,
        gs_tau=GS_SOFTMIN_TAU,
        device="cpu",
    ):
        import torch

        self.device = torch.device(device)
        self.cloud = cloud if isinstance(cloud, GaussianCloud) else GaussianCloud(cloud)
        self.spheres_local = _as_tensor(self.cloud.as_spheres(), self.device)
        self.obj_mass = float(obj_mass)
        self.gravity = _as_tensor(gravity, self.device).reshape(3)
        self.timestep = float(timestep)
        self.tip_radius = float(tip_radius)
        self.table_z = None if table_z is None else float(table_z)
        self.support_radius = float(support_radius)
        self.gs_tau = float(gs_tau)
        self.contact_stiffness = float(contact_stiffness)
        self.contact_damping = float(contact_damping)
        q = torch.zeros(12, 12, dtype=torch.float32, device=self.device)
        q[:3, :3] = float(obj_inertia_pos) * torch.eye(3, device=self.device)
        q[3:6, 3:6] = float(obj_inertia_rot) * torch.eye(3, device=self.device)
        q[6:, 6:] = float(robot_stiffness) * torch.eye(6, device=self.device)
        self.Q = q
        self.Q_inv = torch.linalg.inv(q + 1.0e-8 * torch.eye(12, device=self.device))
        self.robot_stiff = float(robot_stiffness)

    def to(self, device):
        import torch

        self.device = torch.device(device)
        self.spheres_local = self.spheres_local.to(self.device)
        self.gravity = self.gravity.to(self.device)
        self.Q = self.Q.to(self.device)
        self.Q_inv = self.Q_inv.to(self.device)
        return self

    def split_state(self, state):
        state = _as_tensor(state, self.device)
        return state[..., 0:3], state[..., 3:7], state[..., 7:10], state[..., 10:13]

    def collide(self, state):
        import torch

        obj_pos, obj_quat, left_tip, right_tip = self.split_state(state)
        # #region agent log
        _dbg("comfree_gs_torch.py:collide", "before transform_spheres", {"spheres": list(self.spheres_local.shape), "state": list(state.shape)})
        # #endregion
        spheres = transform_spheres(self.spheres_local, obj_pos, obj_quat)
        tips = torch.stack((left_tip, right_tip), dim=-2)
        # #region agent log
        _dbg("comfree_gs_torch.py:collide", "before gs_sphere_features", {"spheres_w": list(spheres.shape), "tips": list(tips.shape)})
        # #endregion
        phi, position, normal = gs_sphere_features(
            tips, spheres, query_radius=self.tip_radius, tau=self.gs_tau
        )
        # #region agent log
        _dbg("comfree_gs_torch.py:collide", "after gs_sphere_features", {"phi": list(phi.shape)})
        # #endregion
        return {
            "phi": phi,
            "position": position,
            "normal": normal,
            "spheres_world": spheres,
        }

    def step(self, state, cmd):
        import torch

        state = _as_tensor(state, self.device)
        cmd = _as_tensor(cmd, self.device)
        leading = state.shape[:-1]
        obj_pos, obj_quat, left_tip, right_tip = self.split_state(state)
        contacts = self.collide(state)
        # #region agent log
        _dbg("comfree_gs_torch.py:step", "after collide", {"table_z": self.table_z})
        # #endregion
        bias = cmd.new_zeros(*leading, 12)
        bias = bias.clone()
        bias[..., 0:3] = self.obj_mass * self.gravity
        bias[..., 6:12] = self.robot_stiff * cmd
        v_nc = torch.linalg.solve(self.Q, bias) / self.timestep
        # #region agent log
        _dbg("comfree_gs_torch.py:step", "after v_nc", {"v_nc": [float(x) for x in v_nc.detach().reshape(-1)[:6]], "cmd": [float(x) for x in cmd.detach().reshape(-1)]})
        # #endregion

        force_vel = torch.zeros_like(v_nc)
        contact_force = []
        for idx in range(2):
            phi = contacts["phi"][..., idx]
            jac = gs_contact_rows(
                phi,
                contacts["position"][..., idx, :],
                contacts["normal"][..., idx, :],
                obj_pos,
                (left_tip, right_tip)[idx],
                idx,
            )[1]
            force = comfree_normal_force(
                phi, jac, v_nc, self.contact_stiffness, self.contact_damping, self.timestep
            )
            contact_force.append(force)
            force_vel = force_vel + jac * force.unsqueeze(-1)
        # #region agent log
        _dbg("comfree_gs_torch.py:step", "after contact forces", {"has_table": self.table_z is not None})
        # #endregion
        table_force = None
        if self.table_z is not None:
            phi_t, jac_t = table_contact_row(obj_pos, self.table_z, self.support_radius)
            table_force = comfree_normal_force(
                phi_t, jac_t, v_nc, self.contact_stiffness, self.contact_damping, self.timestep
            )
            force_vel = force_vel + jac_t * table_force.unsqueeze(-1)
        # #region agent log
        _dbg("comfree_gs_torch.py:step", "after table force", {"force_vel": list(force_vel.shape), "has_table_force": table_force is not None}, hypothesis_id="E")
        # #endregion
        v = v_nc + torch.linalg.solve(self.Q, force_vel) / self.timestep
        # #region agent log
        _dbg("comfree_gs_torch.py:step", "after v_matmul", {"v": list(v.shape)}, hypothesis_id="E")
        # #endregion
        next_pos = obj_pos + self.timestep * v[..., 0:3]
        # #region agent log
        _dbg("comfree_gs_torch.py:step", "before quat_integrate", {"v": list(v.shape)})
        # #endregion
        next_quat = quat_integrate_wxyz(obj_quat, v[..., 3:6], self.timestep)
        next_left = left_tip + self.timestep * v[..., 6:9]
        next_right = right_tip + self.timestep * v[..., 9:12]
        next_state = torch.cat((next_pos, next_quat, next_left, next_right), dim=-1)
        extras = {
            "velocity": v,
            "contact_force": torch.stack(contact_force, dim=-1),
            "table_force": table_force,
            **contacts,
        }
        return next_state, extras

    def rollout(self, state, cmd_traj):
        import torch

        state = _as_tensor(state, self.device)
        cmd_traj = _as_tensor(cmd_traj, self.device)
        horizon = int(cmd_traj.shape[-2])
        states = []
        extras = []
        current = state
        for step in range(horizon):
            # #region agent log
            _dbg("comfree_gs_torch.py:rollout", "step enter", {"step": int(step)})
            # #endregion
            current, extra = self.step(current, cmd_traj[..., step, :])
            states.append(current)
            extras.append(extra)
        return torch.stack(states, dim=-2), extras
