"""Planner-side ramp geometry and objective, without an Isaac dependency."""
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from planning.object_pose_cost import cost_config


REPO_ROOT = Path(__file__).resolve().parents[4]


def repo_asset_path(path):
    """Resolve repository assets independently of the launch directory."""
    path = Path(path).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def prepare_object_geometry(param):
    """Bake URDF mesh scale once so Lambda, MPC and PhysX share coordinates."""
    scale = np.asarray(param.obj_mesh_scale_, dtype=float).reshape(3)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('mesh scale must be finite and positive')
    source = repo_asset_path(param.source_mesh_path_)
    param.source_mesh_path_ = str(source)
    mesh = trimesh.load_mesh(source, process=True)
    mesh.apply_scale(scale)
    hull = mesh.convex_hull
    collision = hull if param.collision_hull else mesh
    key = hashlib.sha256(source.read_bytes() + scale.tobytes()
                         + bytes([bool(param.collision_hull)])).hexdigest()[:20]
    directory = Path(tempfile.gettempdir()) / 'scsp_tilted_meshes' / key
    directory.mkdir(parents=True, exist_ok=True)
    for name, geometry in [('collision.stl', collision), ('visual.stl', mesh)]:
        destination = directory / name
        if not destination.exists():
            geometry.export(destination)
    param.mesh_path_ = str(directory / 'collision.stl')
    param.visual_mesh_path_ = str(directory / 'visual.stl')
    param._collision_mesh_extracted = True
    param.requested_mesh_scale_ = scale.copy()
    param.obj_mesh_scale_ = np.ones(3)
    param.object_vertices_ = np.asarray(mesh.vertices).copy()
    param.support_vertices_ = np.asarray(hull.vertices).copy()
    param.support_edges_ = np.asarray(hull.edges_unique).copy()
    param.object_aabb_lo, param.object_aabb_hi = mesh.bounds.copy()
    # Conservative eight-corner envelope keeps the symbolic edge objective
    # small even when a scanned convex hull has thousands of vertices.
    param.support_footprint_vertices_ = np.array(np.meshgrid(*zip(
        param.object_aabb_lo, param.object_aabb_hi))).T.reshape(-1, 3)
    param.object_circumradius = float(np.linalg.norm(.5 * mesh.extents))


def configure_support_objective(param, args):
    """Preserve placement stability while optionally ignoring target yaw."""
    normal = np.asarray(param.support_surface_normal_, dtype=float)
    initial = Rotation.from_quat(np.asarray(param.init_obj_qpos_)[[4, 5, 6, 3]])
    param.object_up_axis_ = initial.inv().apply(normal)
    param.ori_coef = 0. if param.position_only_goal_ else float(args.ori_coef)
    tilt = float(getattr(args, 'pose_tilt_limit_deg', 10.))
    barrier = float(getattr(args, 'pose_tilt_barrier_coef', 2000.))
    upright = float(getattr(args, 'pose_upright_coef', args.ori_coef))
    if not np.isfinite([tilt, barrier, upright]).all() or not 0 < tilt < 90 or min(barrier, upright) < 0:
        raise ValueError('invalid ramp tilt objective')
    param.object_cost_config_ = np.r_[cost_config(
        param.pos_coef, param.ori_coef, upright,
        body_axis=param.object_up_axis_, reference_up=normal),
        0., 1-np.cos(np.radians(tilt)), barrier]
    param.support_edge_margin_ = float(getattr(args, 'ramp_edge_margin', .04))
    param.support_edge_cost_ = float(getattr(args, 'ramp_edge_cost', 2000.))
    if (not np.isfinite([param.support_edge_margin_, param.support_edge_cost_]).all()
            or min(param.support_edge_margin_, param.support_edge_cost_) < 0
            or param.support_edge_margin_ >= min(param.support_surface_half_extents_)):
        raise ValueError('invalid ramp edge objective')


def fit_mesh_center_on_ramp(param, local_xy, rotation):
    """Clip COM placement using the rotated footprint, including rim margin."""
    footprint = getattr(param, 'support_footprint_vertices_', param.support_vertices_)
    offset = (footprint @ np.asarray(rotation).T
              @ np.asarray(param.support_surface_tangents_).T)
    half = np.asarray(param.support_surface_half_extents_)
    margin = float(getattr(param, 'support_edge_margin_', .04))
    lower = -half + margin - offset.min(axis=0)
    upper = half - margin - offset.max(axis=0)
    if np.any(lower > upper):
        raise ValueError('object footprint does not fit on the ramp with the requested edge margin')
    return np.clip(local_xy, lower, upper)
