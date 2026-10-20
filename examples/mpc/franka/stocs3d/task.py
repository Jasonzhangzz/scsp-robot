"""Build the original STOCS parameter dictionary from one Franka scene.

The mesh, mass, inertia, initial pose and goal are the scene that
test_mpc_isaac.py executes. Contact selection and the CITO program stay in
the vendored STOCS-3D sources.
"""
from pathlib import Path

import numpy as np
import trimesh
from klampt import Geometry3D
from klampt.math import so3, se3

from examples.mpc.franka.runtime.contact import DEFAULT_FINGERTIP_RADIUS

# cfg/plane_cube.py. These are the solver settings, not the Franka task.
OPTIMIZATION = dict(
    comp_threshold=1e-4,
    friction_dim=6,
    mu_mnp=1.0,
    mu_env=1.0,
    N=11,
    stocs_max_iter=20,
    velocity_complementarity=True,
    initial_pose_relaxation=1e-3,
    goal_pose_relaxation=1e-3,
    max_mpcc_iter=20,
    major_feasibility_tolerance=1e-4,
    major_optimality_tolerance=1e-4,
    time_step=0.1,
    initialization='linear',
    elastic_weight=1000,
    disturbances=[1e-2],
    add_threshold=1e-3,
    active_threshold=1e-2,
    time_smoothing_step=1,
    assumption='quasi_dynamic',
    force_on_off=False,
    use_alpha=False,
)
OBJECT_GRID = 0.004
TABLE_GRID = 0.01
PATCH_RADIUS = 0.008
MAX_PATCH = 4


def _center_of_mass(mesh_path):
    mesh = trimesh.load_mesh(mesh_path, process=False)
    try:
        com = np.asarray(mesh.center_mass, dtype=float).reshape(3)
        if np.isfinite(com).all():
            return com
    except Exception:
        pass
    return np.asarray(mesh.vertices, dtype=float).mean(axis=0)


def _inward_normal(point, com):
    normal = np.asarray(com, dtype=float) - np.asarray(point, dtype=float)
    length = float(np.linalg.norm(normal))
    if length < 1e-8:
        raise ValueError('STOCS contact point coincides with the center of mass')
    return normal / length


def manipulation_patch(vertices, com, pose_wxyz_xyz, fingertip_xyz):
    """Points on the object nearest the initial fingertip, normals into the body."""
    rotation = so3.from_quaternion(np.asarray(pose_wxyz_xyz[3:7], dtype=float).tolist())
    origin = np.asarray(pose_wxyz_xyz[:3], dtype=float).tolist()
    local_tip = np.asarray(se3.apply(se3.inv((rotation, origin)), np.asarray(fingertip_xyz, dtype=float).tolist()), dtype=float)
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    distance = np.linalg.norm(vertices - local_tip.reshape(1, 3), axis=1)
    primary = int(np.argmin(distance))
    chosen = [primary]
    order = np.argsort(distance)
    for index in order:
        if int(index) == primary:
            continue
        if distance[index] > distance[primary] + PATCH_RADIUS:
            break
        if np.linalg.norm(vertices[index] - vertices[primary]) <= PATCH_RADIUS:
            chosen.append(int(index))
        if len(chosen) == MAX_PATCH:
            break
    normal = _inward_normal(vertices[primary], com)
    points = [vertices[index].tolist() for index in chosen]
    normals = [normal.tolist() for _ in chosen]
    return points, normals


def _write_table(path, low, high, top, thickness):
    extents = np.array([high[0] - low[0], high[1] - low[1], thickness], dtype=float)
    center = np.array([(low[0] + high[0]) * 0.5, (low[1] + high[1]) * 0.5, top - 0.5 * thickness], dtype=float)
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    path.parent.mkdir(parents=True, exist_ok=True)
    box.export(path)
    return path


def _geometry(path, resolution):
    from semiinfinite.geometryopt import PenetrationDepthGeometry
    shape = Geometry3D()
    shape.loadFile(str(path))
    return PenetrationDepthGeometry(shape, resolution, resolution)


def build_stocs_params(scene, cache_dir):
    """Original STOCS dictionaries for this scene. Geometry objects are live."""
    mesh_path = Path(scene.mesh_path_).resolve()
    com = _center_of_mass(mesh_path)
    pose = np.asarray(scene.init_obj_qpos_, dtype=float).reshape(7)
    goal_p = np.asarray(scene.target_p_, dtype=float).reshape(3)
    goal_q = np.asarray(scene.target_q_, dtype=float).reshape(4)
    points, normals = manipulation_patch(
        scene.object_vertices_, com, pose, scene.init_fingertip_pos_)
    margin = 0.2
    low = np.minimum(pose[:3], goal_p) - margin
    high = np.maximum(pose[:3], goal_p) + margin
    table = float(scene.table_height)
    low[2] = min(float(low[2]), table - 0.02)
    high[2] = max(float(high[2]), table + 0.3)
    table_path = _write_table(
        Path(cache_dir) / ('table_%s.stl' % scene.trial_seed_),
        low[:2] - 0.05, high[:2] + 0.05, table, 0.04)
    inertia = np.diag(np.asarray(scene.sim_obj_inertia_diag_, dtype=float).reshape(3))
    task = dict(
        q_init=pose[3:7].tolist(),
        x_init=pose[:3].tolist(),
        q_goal=goal_q.tolist(),
        x_goal=goal_p.tolist(),
        task_name='franka',
        manipulation_contact_points=points,
        manipulation_contact_normals=normals,
        x_bound=[low.tolist(), high.tolist()],
        v_bound=[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
    )
    return dict(
        manipuland_params=dict(
            manipuland_name=str(getattr(scene, 'model_path_', 'object')),
            manipuland_fn=str(mesh_path),
            manipuland=_geometry(mesh_path, OBJECT_GRID),
            scale_x=1.0, scale_y=1.0, scale_z=1.0,
            com=com.tolist(),
            m=float(scene.sim_obj_mass_),
            I=inertia,
        ),
        environment_params=dict(
            environment_name='table',
            environments=[_geometry(table_path, TABLE_GRID)],
        ),
        optimization_params=dict(OPTIMIZATION),
        task_params=task,
        fingertip_radius=float(DEFAULT_FINGERTIP_RADIUS),
    )
