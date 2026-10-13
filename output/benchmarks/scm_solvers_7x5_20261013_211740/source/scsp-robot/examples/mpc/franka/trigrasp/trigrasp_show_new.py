from __future__ import annotations

import argparse
import copy
import itertools
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import casadi as cs
import mujoco
import mujoco.viewer
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from models.explicit_model import ExplicitModel
from planning.mpc_explicit import MPCExplicit
from planning.mlqp_point_v2 import LambdaContactControlOptimizer
from planning.screenshot import (
    PeriodicSVGScreenshotRecorder,
    build_free_camera_config_from_position,
    create_mujoco_mp4_recorder,
)


OBJECT_ASSET_DIR = REPO_ROOT / "envs" / "assets" / "objects"
GENERATED_COLLISION_ASSET_DIR = OBJECT_ASSET_DIR / "_generated_collision"
DEFAULT_SCENE_OUTPUT = REPO_ROOT / "envs" / "xmls" / "trigrasp.xml"
DEFAULT_SCREENSHOT_DIR = CURRENT_DIR / "figs"
DEFAULT_VIDEO_OUTPUT_DIR = REPO_ROOT / "outputs" / "videos_grasping"
_SPIDER_ALLEGRO_REL = Path("thirdparty/spider/spider/assets/robots/allegro/right.xml")
_local_spider_xml = REPO_ROOT / _SPIDER_ALLEGRO_REL
SPIDER_ALLEGRO_XML = _local_spider_xml if _local_spider_xml.is_file() else (REPO_ROOT.parent / _SPIDER_ALLEGRO_REL)
SPIDER_ALLEGRO_ASSET_DIR = SPIDER_ALLEGRO_XML.parent / "assets"
DEFAULT_MESH_SCALE_MULTIPLIER = 1.0
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

FINGER_NAMES = ("thumb", "index", "middle", "ring")
FINGERTIP_SITE_NAMES = tuple(f"right_{finger}_tip" for finger in FINGER_NAMES)
TIP_COLLISION_GEOM_NAMES = {
    "thumb": "collision_hand_right_thumb_0",
    "index": "collision_hand_right_index_0",
    "middle": "collision_hand_right_middle_0",
    "ring": "collision_hand_right_ring_0",
}
BASE_ACTUATOR_COUNT = 6
BASE_TRANSLATION_DOF_COUNT = 3
BASE_TRANSLATION_JOINT_NAMES = ("right_pos_x", "right_pos_y", "right_pos_z")
BASE_ROTATION_JOINT_NAMES = ("right_rot_x", "right_rot_y", "right_rot_z")
REFERENCE_PALM_FIXED_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
WRAPPER_BODY_INERTIAL_MASS = 1e-3
WRAPPER_BODY_INERTIAL_DIAG = np.array([1e-6, 1e-6, 1e-6], dtype=np.float64)
DEFAULT_SCALE_MAP = {
    "stanford_bunny2": np.array([1.5, 1.5, 1.5], dtype=np.float64),
    "rubber_duck": np.array([1.3, 1.4, 1.4], dtype=np.float64),
    "Wolf_Duck": np.array([0.002, 0.002, 0.002], dtype=np.float64),
}
@dataclass
class IkSolveResult:
    success: bool
    qpos: np.ndarray
    score: float
    mean_tip_error: float
    max_tip_error: float
    palm_pos_error: float
    palm_rot_error: float
    permutation: tuple[int, ...]
    target_palm_pos_world: np.ndarray
    target_palm_rot_world: np.ndarray
    assigned_targets_world: np.ndarray
    solved_tip_positions_world: np.ndarray


@dataclass
class ReferenceTrajectory:
    initial_base_q: np.ndarray
    initial_finger_q: np.ndarray
    base_traj: np.ndarray
    finger_traj: np.ndarray
    object_pos_traj: np.ndarray
    object_quat_traj: np.ndarray
    stage_names: tuple[str, ...]


def normalize(vec, eps=1e-9):
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def project_to_rotation_matrix(rotation_matrix):
    u, _, vh = np.linalg.svd(np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3))
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected


def rotation_error(current_rot, target_rot):
    current_rot = np.asarray(current_rot, dtype=np.float64).reshape(3, 3)
    target_rot = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    return 0.5 * (
        np.cross(current_rot[:, 0], target_rot[:, 0])
        + np.cross(current_rot[:, 1], target_rot[:, 1])
        + np.cross(current_rot[:, 2], target_rot[:, 2])
    )


def project_to_plane(vec, normal):
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    normal = normalize(normal)
    return vec - np.dot(vec, normal) * normal


def quat_xyzw_to_wxyz(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def quat_wxyz_to_mat(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat_wxyz = quat_wxyz / max(np.linalg.norm(quat_wxyz), 1e-9)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_matrix()


def mat_to_quat_wxyz(rotation_matrix):
    rotation_matrix = project_to_rotation_matrix(rotation_matrix)
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rotation_matrix.reshape(-1))
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def quat_from_yaw(yaw):
    quat_xyzw = Rotation.from_euler("z", float(yaw)).as_quat()
    return quat_xyzw_to_wxyz(quat_xyzw)


def format_vec(vec):
    return " ".join(f"{float(v):.8f}" for v in np.asarray(vec, dtype=np.float64).reshape(-1))


def indent_xml(element, level=0):
    indent = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indent + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not element.tail or not element.tail.strip():
            element.tail = indent
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def resolve_mesh_path(obj_name=None, mesh_path=None):
    if mesh_path:
        path = Path(mesh_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Mesh file does not exist: {path}")
        return path

    if not obj_name:
        raise ValueError("Either --obj or --mesh must be provided.")

    candidates = [
        OBJECT_ASSET_DIR / f"{obj_name}{suffix}"
        for suffix in ("", ".stl", ".obj", ".ply", ".off")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find mesh for object '{obj_name}' in {OBJECT_ASSET_DIR}")


def load_mesh_bounds(mesh_path, scale):
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    mesh.apply_scale(np.asarray(scale, dtype=np.float64))
    return np.asarray(mesh.bounds, dtype=np.float64)


def resolve_mesh_scale(mesh_path, scale_override=None, scale_multiplier=DEFAULT_MESH_SCALE_MULTIPLIER):
    if scale_override is not None:
        return np.asarray(scale_override, dtype=np.float64).reshape(3)
    return float(scale_multiplier) * DEFAULT_SCALE_MAP.get(mesh_path.stem, np.ones(3, dtype=np.float64)).copy()


def prepare_collision_mesh_path(mesh_path):
    mesh_path = Path(mesh_path).resolve()
    collision_path = GENERATED_COLLISION_ASSET_DIR / f"{mesh_path.stem}_convex_hull.stl"
    collision_path.parent.mkdir(parents=True, exist_ok=True)

    if collision_path.exists() and collision_path.stat().st_mtime >= mesh_path.stat().st_mtime:
        return collision_path

    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    collision_mesh = mesh.convex_hull if hasattr(mesh, "convex_hull") else mesh
    collision_mesh.export(collision_path)
    return collision_path


def patch_robot_asset_paths(asset_element, asset_dir):
    asset_element = copy.deepcopy(asset_element)
    for node in asset_element.iter():
        if node.tag == "mesh" and "file" in node.attrib:
            mesh_file = Path(node.attrib["file"])
            if not mesh_file.is_absolute():
                node.attrib["file"] = str((asset_dir / mesh_file).resolve())
    return asset_element


def build_world_aligned_allegro_robot_body(robot_body, palm_quat=REFERENCE_PALM_FIXED_QUAT_WXYZ):
    robot_body = copy.deepcopy(robot_body)

    base_joint_names = set(BASE_TRANSLATION_JOINT_NAMES + BASE_ROTATION_JOINT_NAMES)
    base_joints = {}
    for child in list(robot_body):
        if child.tag != "joint":
            continue
        joint_name = child.attrib.get("name")
        if joint_name in base_joint_names:
            base_joints[joint_name] = copy.deepcopy(child)
            robot_body.remove(child)

    expected_joint_names = set(BASE_TRANSLATION_JOINT_NAMES + BASE_ROTATION_JOINT_NAMES)
    if set(base_joints) != expected_joint_names:
        missing_joint_names = sorted(expected_joint_names.difference(base_joints))
        raise RuntimeError(
            "Failed to extract Allegro floating-base joints from the robot XML. "
            f"Missing joints: {missing_joint_names}"
        )

    palm_quat = np.asarray(palm_quat, dtype=np.float64).reshape(4)
    palm_quat = palm_quat / max(np.linalg.norm(palm_quat), 1e-9)
    robot_body.attrib["quat"] = format_vec(palm_quat)

    hand_base = ET.Element("body", {"name": "right_hand_base", "pos": "0 0 0"})
    ET.SubElement(
        hand_base,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{float(WRAPPER_BODY_INERTIAL_MASS):.8f}",
            "diaginertia": format_vec(WRAPPER_BODY_INERTIAL_DIAG),
        },
    )
    for joint_name in BASE_TRANSLATION_JOINT_NAMES:
        hand_base.append(base_joints[joint_name])

    hand_rot_base = ET.SubElement(hand_base, "body", {"name": "right_hand_rot_base", "pos": "0 0 0"})
    ET.SubElement(
        hand_rot_base,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": f"{float(WRAPPER_BODY_INERTIAL_MASS):.8f}",
            "diaginertia": format_vec(WRAPPER_BODY_INERTIAL_DIAG),
        },
    )
    for joint_name in BASE_ROTATION_JOINT_NAMES:
        hand_rot_base.append(base_joints[joint_name])

    hand_rot_base.append(robot_body)
    return hand_base


def build_trigrasp_scene_xml(
    visual_mesh_path,
    collision_mesh_path,
    mesh_scale,
    object_pos,
    object_quat,
    pedestal_pos,
    pedestal_size,
    object_mass,
    object_friction,
    pedestal_friction,
    scene_output_path=DEFAULT_SCENE_OUTPUT,
    floor_z=0.0,
    mujoco_timestep=0.01,
):
    robot_root = ET.parse(SPIDER_ALLEGRO_XML).getroot()
    robot_body = robot_root.find("./worldbody/body[@name='right_palm']")
    robot_asset = robot_root.find("asset")
    robot_default = robot_root.find("default")
    robot_actuator = robot_root.find("actuator")
    if robot_body is None or robot_asset is None or robot_default is None or robot_actuator is None:
        raise RuntimeError(f"Unexpected Allegro XML layout in {SPIDER_ALLEGRO_XML}")

    root = ET.Element("mujoco", {"model": "allegro_tabletop_trigrasp"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": f"{float(mujoco_timestep):.8f}",
            "iterations": "40",
            "ls_iterations": "80",
        },
    )
    ET.SubElement(root, "statistic", {"center": format_vec(object_pos), "extent": "0.55"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.72 0.72 0.72", "ambient": "0.25 0.25 0.25", "specular": "0.08 0.08 0.08"},
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "145", "elevation": "-28"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.32 0.46 0.60",
            "rgb2": "0.05 0.06 0.08",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.26 0.28 0.31",
            "rgb2": "0.16 0.18 0.20",
            "markrgb": "0.88 0.88 0.88",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.25",
        },
    )
    ET.SubElement(asset, "material", {"name": "pedestal_mat", "rgba": "0.28 0.30 0.34 1"})
    ET.SubElement(asset, "material", {"name": "object_visual_mat", "rgba": "0.92 0.66 0.30 1"})
    ET.SubElement(asset, "material", {"name": "contact_marker_red", "rgba": "0.95 0.28 0.22 1"})
    ET.SubElement(asset, "material", {"name": "contact_marker_green", "rgba": "0.18 0.78 0.36 1"})
    ET.SubElement(asset, "material", {"name": "contact_marker_blue", "rgba": "0.22 0.54 0.96 1"})
    ET.SubElement(asset, "material", {"name": "contact_marker_orange", "rgba": "0.96 0.62 0.18 1"})
    ET.SubElement(asset, "material", {"name": "thumb_target_mat", "rgba": "0.95 0.80 0.15 0.95"})
    ET.SubElement(asset, "material", {"name": "index_target_mat", "rgba": "0.18 0.78 0.36 0.95"})
    ET.SubElement(asset, "material", {"name": "middle_target_mat", "rgba": "0.22 0.54 0.96 0.95"})
    ET.SubElement(asset, "material", {"name": "ring_target_mat", "rgba": "0.82 0.30 0.92 0.95"})
    ET.SubElement(asset, "material", {"name": "palm_target_mat", "rgba": "0.92 0.92 0.96 0.70"})
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "object_visual_mesh",
            "file": str(Path(visual_mesh_path).resolve()),
            "scale": format_vec(mesh_scale),
        },
    )
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "object_collision_mesh",
            "file": str(Path(collision_mesh_path).resolve()),
            "scale": format_vec(mesh_scale),
        },
    )
    patched_robot_asset = patch_robot_asset_paths(robot_asset, SPIDER_ALLEGRO_ASSET_DIR)
    for child in list(patched_robot_asset):
        asset.append(copy.deepcopy(child))

    default = ET.SubElement(root, "default")
    for child in list(robot_default):
        default.append(copy.deepcopy(child))

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0.6 -0.3 1.4", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": "0.95 -0.62 0.56",
            "xyaxes": "0.57 0.82 0.00 -0.31 0.21 0.93",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": f"0 0 {float(floor_z):.8f}",
            "size": "0 0 0.05",
            "material": "groundplane",
            "friction": f"{float(pedestal_friction):.8f} 0.08 0.01",
        },
    )

    # Keep the Allegro hand before the freejoint object so hand qpos/dof indexing stays simple.
    worldbody.append(build_world_aligned_allegro_robot_body(robot_body))

    pedestal_body = ET.SubElement(worldbody, "body", {"name": "pedestal", "pos": format_vec(pedestal_pos)})
    ET.SubElement(
        pedestal_body,
        "geom",
        {
            "name": "pedestal_geom",
            "type": "box",
            "size": format_vec(pedestal_size),
            "material": "pedestal_mat",
            "contype": "1",
            "conaffinity": "1",
            "friction": f"{float(pedestal_friction):.8f} 0.08 0.01",
            "condim": "3",
        },
    )

    obj_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "obj", "pos": format_vec(object_pos), "quat": format_vec(object_quat)},
    )
    ET.SubElement(obj_body, "freejoint", {"name": "obj_freejoint"})
    ET.SubElement(
        obj_body,
        "geom",
        {
            "name": "obj",
            "type": "mesh",
            "mesh": "object_collision_mesh",
            "mass": f"{float(object_mass):.8f}",
            "contype": "1",
            "conaffinity": "1",
            "condim": "3",
            "friction": f"{float(object_friction):.8f} 0.08 0.01",
            "group": "3",
            "rgba": "1 1 1 0",
        },
    )
    ET.SubElement(
        obj_body,
        "geom",
        {
            "name": "obj_visual",
            "type": "mesh",
            "mesh": "object_visual_mesh",
            "material": "object_visual_mat",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
        },
    )
    ET.SubElement(obj_body, "site", {"name": "obj_center", "size": "0.004", "rgba": "1 1 1 0"})

    contact_materials = (
        "contact_marker_red",
        "contact_marker_green",
        "contact_marker_blue",
        "contact_marker_orange",
    )
    for idx, material in enumerate(contact_materials, start=1):
        marker_body = ET.SubElement(worldbody, "body", {"name": f"contact_point{idx}", "pos": format_vec(object_pos)})
        ET.SubElement(
            marker_body,
            "geom",
            {
                "name": f"contact_point{idx}_geom",
                "type": "sphere",
                "size": "0.0055",
                "material": material,
                "contype": "0",
                "conaffinity": "0",
            },
        )

    for finger_name in FINGER_NAMES:
        marker_body = ET.SubElement(worldbody, "body", {"name": f"{finger_name}_target", "pos": format_vec(object_pos)})
        ET.SubElement(
            marker_body,
            "geom",
            {
                "name": f"{finger_name}_target_geom",
                "type": "sphere",
                "size": "0.0045",
                "material": f"{finger_name}_target_mat",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    palm_target = ET.SubElement(worldbody, "body", {"name": "palm_target", "pos": format_vec(object_pos), "quat": "1 0 0 0"})
    ET.SubElement(
        palm_target,
        "geom",
        {
            "name": "palm_target_geom",
            "type": "box",
            "size": "0.015 0.03 0.012",
            "material": "palm_target_mat",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    actuator = ET.SubElement(root, "actuator")
    for child in list(robot_actuator):
        actuator.append(copy.deepcopy(child))

    indent_xml(root)
    tree = ET.ElementTree(root)
    scene_output_path = Path(scene_output_path)
    scene_output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(scene_output_path, encoding="utf-8", xml_declaration=False)
    return scene_output_path


class AllegroExplicitMPCParams:
    def __init__(self, demo):
        self.object_names_ = ["obj"]
        self.h_ = float(demo.command_dt)
        self.frame_skip_ = int(demo.args.mj_steps_per_command)

        self.n_robot_qpos_ = int(BASE_TRANSLATION_DOF_COUNT + demo.finger_lower.size)
        self.n_qpos_ = int(7 + self.n_robot_qpos_)
        self.n_qvel_ = int(6 + self.n_robot_qpos_)
        self.n_cmd_ = int(self.n_robot_qpos_)

        self.n_mj_q_ = int(demo.model.nq)
        self.n_mj_v_ = int(demo.model.nv)
        self.max_ncon_ = int(demo.args.mpc_max_contacts)

        self.mu_object_ = float(demo.args.object_friction)
        self.obj_inertia_ = np.identity(6, dtype=np.float64)
        self.obj_inertia_[:3, :3] = 50.0 * np.eye(3, dtype=np.float64)
        self.obj_inertia_[3:, 3:] = 0.1 * np.eye(3, dtype=np.float64)
        self.robot_stiff_ = np.diag([20.0, 20.0, 20.0] + [1.0] * int(demo.finger_lower.size)).astype(np.float64)

        self.Q = np.zeros((self.n_qvel_, self.n_qvel_), dtype=np.float64)
        self.Q[:6, :6] = self.obj_inertia_
        self.Q[6:, 6:] = self.robot_stiff_

        self.obj_mass_ = float(demo.args.obj_mass)
        self.gravity_ = np.array([0.0, 0.0, -9.8, 0.0, 0.0, 0.0], dtype=np.float64)
        self.model_params = float(demo.args.contact_stiffness)

        self.mpc_horizon_ = int(demo.args.mpc_horizon)
        self.ipopt_max_iter_ = int(demo.args.mpc_ipopt_max_iter)
        self.mpc_model = "explicit"

        base_step_limit = float(demo.args.mpc_base_step_limit)
        finger_step_limit = float(demo.args.mpc_velocity_limit) * float(self.h_)
        self.mpc_u_lb_ = np.hstack(
            (
                -base_step_limit * np.ones(BASE_TRANSLATION_DOF_COUNT, dtype=np.float64),
                -finger_step_limit * np.ones(int(demo.finger_lower.size), dtype=np.float64),
            )
        )
        self.mpc_u_ub_ = -self.mpc_u_lb_

        obj_pos_lb = np.array(
            [
                demo.initial_object_pos[0] - 1.0,
                demo.initial_object_pos[1] - 1.0,
                float(demo.args.floor_z) - 0.05,
            ],
            dtype=np.float64,
        )
        obj_pos_ub = np.array(
            [
                demo.initial_object_pos[0] + 1.0,
                demo.initial_object_pos[1] + 1.0,
                demo.initial_object_pos[2] + float(demo.args.lift_height) + 0.4,
            ],
            dtype=np.float64,
        )
        base_q_lb = np.asarray(demo.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT, 0], dtype=np.float64).copy()
        base_q_ub = np.asarray(demo.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT, 1], dtype=np.float64).copy()
        self.mpc_q_lb_ = np.hstack((obj_pos_lb, -1e7 * np.ones(4), base_q_lb, demo.finger_lower))
        self.mpc_q_ub_ = np.hstack((obj_pos_ub, 1e7 * np.ones(4), base_q_ub, demo.finger_upper))

        self.base_track_weight_ = float(demo.args.mpc_base_track_weight)
        self.finger_track_weight_ = float(demo.args.mpc_q_weight)
        self.object_pos_weight_ = float(demo.args.mpc_object_pos_weight)
        self.object_quat_weight_ = float(demo.args.mpc_object_quat_weight)
        self.terminal_weight_ = float(demo.args.mpc_terminal_weight)
        self.u_weight_ = float(demo.args.mpc_u_weight)

        self.sol_guess_ = None

    def build_cost_param_vector(self, target_object_pos, target_object_quat, target_base_pos, target_finger_q):
        return np.concatenate(
            [
                np.asarray(target_object_pos, dtype=np.float64).reshape(3),
                np.asarray(target_object_quat, dtype=np.float64).reshape(4),
                np.asarray(target_base_pos, dtype=np.float64).reshape(BASE_TRANSLATION_DOF_COUNT),
                np.asarray(target_finger_q, dtype=np.float64).reshape(self.n_robot_qpos_ - BASE_TRANSLATION_DOF_COUNT),
            ],
            axis=0,
        )

    def init_cost_fns(self):
        x = cs.SX.sym("x", self.n_qpos_)
        u = cs.SX.sym("u", self.n_cmd_)

        obj_pose = x[:7]
        base_pos = x[7 : 7 + BASE_TRANSLATION_DOF_COUNT]
        finger_q = x[7 + BASE_TRANSLATION_DOF_COUNT :]

        target_object_pos = cs.SX.sym("target_object_pos", 3)
        target_object_quat = cs.SX.sym("target_object_quat", 4)
        target_base_pos = cs.SX.sym("target_base_pos", BASE_TRANSLATION_DOF_COUNT)
        target_finger_q = cs.SX.sym("target_finger_q", self.n_robot_qpos_ - BASE_TRANSLATION_DOF_COUNT)
        cost_param = cs.vvcat([target_object_pos, target_object_quat, target_base_pos, target_finger_q])

        position_cost = cs.sumsqr(obj_pose[:3] - target_object_pos)
        quaternion_cost = 1.0 - cs.dot(obj_pose[3:7], target_object_quat) ** 2
        base_track_cost = cs.sumsqr(base_pos - target_base_pos)
        finger_track_cost = cs.sumsqr(finger_q - target_finger_q)
        control_cost = cs.sumsqr(u)

        path_cost = (
            self.base_track_weight_ * base_track_cost
            + self.finger_track_weight_ * finger_track_cost
            + self.object_pos_weight_ * position_cost
            + self.object_quat_weight_ * quaternion_cost
            + self.u_weight_ * control_cost
        )
        final_cost = self.terminal_weight_ * (
            self.base_track_weight_ * base_track_cost
            + self.finger_track_weight_ * finger_track_cost
            + self.object_pos_weight_ * position_cost
            + self.object_quat_weight_ * quaternion_cost
        )

        path_cost_fn = cs.Function("allegro_trigrasp_path_cost_fn", [x, u, cost_param], [path_cost])
        final_cost_fn = cs.Function("allegro_trigrasp_final_cost_fn", [x, cost_param], [final_cost])
        return path_cost_fn, final_cost_fn


class AllegroExplicitMPC(MPCExplicit):
    def __init__(self, param):
        super().__init__(param, cost_kind="param")

    def plan_once(
        self,
        target_object_pos,
        target_object_quat,
        target_base_pos,
        target_finger_q,
        curr_x,
        phi_vec,
        jac_mat,
        sol_guess=None,
    ):
        cost_params = self.param_.build_cost_param_vector(
            target_object_pos=target_object_pos,
            target_object_quat=target_object_quat,
            target_base_pos=target_base_pos,
            target_finger_q=target_finger_q,
        )
        return super().plan_once(
            curr_x=curr_x,
            phi_vec=phi_vec,
            jac_mat=jac_mat,
            cost_params=cost_params,
            sol_guess=sol_guess,
        )


class AllegroTabletopTrigraspDemo:
    def __init__(self, args):
        if int(args.num_grasp_contacts) != 4:
            raise ValueError("This demo expects --num-grasp-contacts=4 so the four Allegro fingertips can be assigned.")
        if int(args.lift_contact_fingers) < 1 or int(args.lift_contact_fingers) > len(FINGER_NAMES):
            raise ValueError(f"--lift-contact-fingers must be within [1, {len(FINGER_NAMES)}].")
        if int(args.max_lift_wait_steps) < 0:
            raise ValueError("--max-lift-wait-steps must be non-negative.")

        self.args = args
        self.mesh_path = resolve_mesh_path(args.obj, args.mesh)
        self.collision_mesh_path = prepare_collision_mesh_path(self.mesh_path)
        self.mesh_scale = resolve_mesh_scale(self.mesh_path, args.scale, args.scale_multiplier)
        self.mesh_bounds = load_mesh_bounds(self.mesh_path, self.mesh_scale)

        self.pedestal_size = np.asarray(args.pedestal_size, dtype=np.float64).copy()
        self.pedestal_pos = np.asarray(args.pedestal_pos, dtype=np.float64).copy()
        self.pedestal_pos[2] += float(args.initial_object_lift)
        self.support_top = float(self.pedestal_pos[2] + self.pedestal_size[2])
        self.support_surface_point = np.array(
            [self.pedestal_pos[0], self.pedestal_pos[1], self.support_top],
            dtype=np.float64,
        )
        self.support_surface_normal = WORLD_UP.copy()

        if args.object_pos is None:
            object_pos = np.array(
                [
                    self.pedestal_pos[0],
                    self.pedestal_pos[1],
                    self.support_top - self.mesh_bounds[0, 2] + float(args.object_z_offset),
                ],
                dtype=np.float64,
            )
        else:
            object_pos = np.asarray(args.object_pos, dtype=np.float64).reshape(3)
        object_quat = quat_from_yaw(args.object_yaw)

        self.initial_object_pos = object_pos.copy()
        self.initial_object_quat = object_quat.copy()
        self.initial_object_rot = quat_wxyz_to_mat(self.initial_object_quat)
        self.object_top_z = float(self.initial_object_pos[2] + self.mesh_bounds[1, 2])
        self.hover_center_xy = self.initial_object_pos[:2].copy()

        z_min = float(self.mesh_bounds[0, 2])
        z_max = float(self.mesh_bounds[1, 2])
        self.local_top_half_z_threshold = z_min + (2.0 / 3.0) * (z_max - z_min)

        self.scene_path = build_trigrasp_scene_xml(
            visual_mesh_path=self.mesh_path,
            collision_mesh_path=self.collision_mesh_path,
            mesh_scale=self.mesh_scale,
            object_pos=self.initial_object_pos,
            object_quat=self.initial_object_quat,
            pedestal_pos=self.pedestal_pos,
            pedestal_size=self.pedestal_size,
            object_mass=float(args.obj_mass),
            object_friction=float(args.object_friction),
            pedestal_friction=float(args.pedestal_friction),
            scene_output_path=args.scene_output,
            floor_z=float(args.floor_z),
            mujoco_timestep=float(args.mujoco_dt),
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = float(args.mujoco_dt)
        self.data = mujoco.MjData(self.model)
        self.solve_data = mujoco.MjData(self.model)
        self.viewer = None
        self.scene_camera_config = None
        self.screenshot_recorder = None
        self.video_recorder = None
        self.video_output_path = self._resolve_video_output_path()

        self.palm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        self.object_qpos_adr = int(self.model.jnt_qposadr[self.object_joint_id])
        self.object_dof_adr = int(self.model.jnt_dofadr[self.object_joint_id])
        self.object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj")
        self.tip_site_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name) for site_name in FINGERTIP_SITE_NAMES],
            dtype=np.int32,
        )
        self.marker_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (
                "contact_point1",
                "contact_point2",
                "contact_point3",
                "contact_point4",
                "thumb_target",
                "index_target",
                "middle_target",
                "ring_target",
                "palm_target",
            )
        }
        self.fingertip_geom_id_to_name = {}
        for finger_name, geom_name in TIP_COLLISION_GEOM_NAMES.items():
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.fingertip_geom_id_to_name[int(geom_id)] = finger_name

        self.hand_joint_ids = np.array(
            [self.model.actuator_trnid[actuator_id, 0] for actuator_id in range(self.model.nu)],
            dtype=np.int32,
        )
        self.hand_qpos_adr = np.array([self.model.jnt_qposadr[joint_id] for joint_id in self.hand_joint_ids], dtype=np.int32)
        self.hand_dof_adr = np.array([self.model.jnt_dofadr[joint_id] for joint_id in self.hand_joint_ids], dtype=np.int32)
        self.hand_joint_ranges = np.asarray(self.model.jnt_range[self.hand_joint_ids], dtype=np.float64).copy()
        self.hand_joint_limited = np.asarray(self.model.jnt_limited[self.hand_joint_ids], dtype=bool).copy()
        self.base_qpos_adr = self.hand_qpos_adr[:BASE_ACTUATOR_COUNT].copy()
        self.base_dof_adr = self.hand_dof_adr[:BASE_ACTUATOR_COUNT].copy()
        self.finger_qpos_adr = self.hand_qpos_adr[BASE_ACTUATOR_COUNT:].copy()
        self.finger_dof_adr = self.hand_dof_adr[BASE_ACTUATOR_COUNT:].copy()
        self.base_pos_qpos_adr = self.base_qpos_adr[:BASE_TRANSLATION_DOF_COUNT].copy()
        self.base_pos_dof_adr = self.base_dof_adr[:BASE_TRANSLATION_DOF_COUNT].copy()
        self.base_joint_ranges = self.hand_joint_ranges[:BASE_ACTUATOR_COUNT].copy()
        self.finger_joint_ranges = self.hand_joint_ranges[BASE_ACTUATOR_COUNT:].copy()
        self.finger_lower = self.finger_joint_ranges[:, 0].copy()
        self.finger_upper = self.finger_joint_ranges[:, 1].copy()
        self.mpc_qvel_full_indices = np.concatenate(
            [
                np.arange(self.object_dof_adr, self.object_dof_adr + 6, dtype=np.int32),
                self.base_pos_dof_adr,
                self.finger_dof_adr,
            ],
            axis=0,
        )

        self.ik_reference_qpos = self.model.qpos0.copy()
        self._refresh_initial_object_pose_cache(self.initial_object_pos, self.initial_object_quat)

        self.cached_contact_points_local = None
        self.cached_target_points_local = None
        self.cached_palm_target_local_pos = None
        self.cached_palm_target_local_rot = None
        self.fixed_palm_rot_world = quat_wxyz_to_mat(REFERENCE_PALM_FIXED_QUAT_WXYZ)

        self.default_hand_q = self.build_default_hand_q()
        self.settle_initial_object_pose(self.default_hand_q)

        self.optimizer = LambdaContactControlOptimizer(
            mesh_path=str(self.mesh_path),
            obj_mass=float(args.obj_mass),
            contact_stiffness=float(args.contact_stiffness),
            sample_num=int(args.sample_num),
            scale_factors=tuple(self.mesh_scale.tolist()),
            num_grasp_contacts=int(args.num_grasp_contacts),
            region_contact_samples=int(args.region_contact_samples),
            top_region_pairs=int(args.top_region_pairs),
            preselect_region_pairs=int(args.preselect_region_pairs),
            max_point_combination_eval=int(args.max_point_combination_eval),
            nlp_solver=args.solver,
            static_nlp_solver=args.solver,
            support_surface_point=self.support_surface_point,
            support_surface_normal=self.support_surface_normal,
            support_surface_clearance=float(args.ground_height_margin),
            support_surface_normal_alignment_threshold=float(args.support_normal_alignment_threshold),
        )

        self.command_dt = max(int(args.mj_steps_per_command), 1) * float(self.model.opt.timestep)
        self.mpc_param = AllegroExplicitMPCParams(self)
        self.mpc = AllegroExplicitMPC(self.mpc_param)

        self.reset(self.initial_object_pos, self.initial_object_quat, self.default_hand_q)

    def close(self):
        if self.video_recorder is not None:
            try:
                saved_path = self.finalize_video_recording()
            except Exception as exc:
                print(f"Warning: failed to finalize MP4 recording during close: {exc}")
            else:
                if saved_path is not None:
                    print(f"Saved mp4: {saved_path}")
        if self.screenshot_recorder is not None:
            self.screenshot_recorder.close()
            self.screenshot_recorder = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def maybe_launch_viewer(self):
        if not bool(self.args.visualize):
            return
        if not os.environ.get("DISPLAY"):
            print("DISPLAY is not available, falling back to headless execution.")
            return
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.scene_camera_config is None:
            self.scene_camera_config = self._build_focus_camera_config()
        self._apply_camera_config_to_viewer(self.scene_camera_config)
        self._ensure_screenshot_recorder()
        self.sync_viewer()

    def _build_focus_camera_config(self, lookat=None):
        if lookat is None:
            try:
                lookat = self.get_object_pose(self.data)[0]
            except Exception:
                lookat = self.initial_object_pos.copy()
        lookat = np.asarray(lookat, dtype=np.float64).reshape(3) + np.array([0.0, 0.0, 0.02], dtype=np.float64)
        camera_position = lookat + np.array([-0.45, -0.42, 0.24], dtype=np.float64)
        return build_free_camera_config_from_position(camera_position=camera_position, lookat=lookat)

    def _apply_camera_config_to_viewer(self, camera_config):
        if self.viewer is None:
            return
        self.viewer.cam.lookat[:] = np.asarray(camera_config.lookat, dtype=np.float64)
        self.viewer.cam.distance = float(camera_config.distance)
        self.viewer.cam.azimuth = float(camera_config.azimuth_deg)
        self.viewer.cam.elevation = float(camera_config.elevation_deg)

    def _build_screenshot_recorder(self):
        self.scene_camera_config = self._build_focus_camera_config()
        if float(self.args.screenshot_interval) <= 0.0:
            return None
        return PeriodicSVGScreenshotRecorder(
            self.model,
            output_dir=self.args.screenshot_dir,
            camera_config=self.scene_camera_config,
            interval_seconds=float(self.args.screenshot_interval),
            width=int(self.args.screenshot_width),
            height=int(self.args.screenshot_height),
            filename_prefix="trigrasp",
            capture_on_start=False,
        )

    def _resolve_video_output_path(self):
        if bool(getattr(self.args, "no_video", False)):
            return None
        if getattr(self.args, "video_output_path", None) is not None:
            return Path(self.args.video_output_path).expanduser()
        video_output_dir = getattr(self.args, "video_output_dir", None)
        if video_output_dir is None:
            return None
        output_dir = Path(video_output_dir).expanduser()
        object_stem = self.mesh_path.stem or "trigrasp"
        return output_dir / f"{object_stem}_trigrasp.mp4"

    def _build_video_recorder(self):
        self.scene_camera_config = self._build_focus_camera_config()
        if self.video_output_path is None:
            return None
        return create_mujoco_mp4_recorder(
            self.model,
            output_path=self.video_output_path,
            camera_config=self.scene_camera_config,
            fps=float(self.args.video_fps),
            width=int(self.args.video_width),
            height=int(self.args.video_height),
            capture_on_start=True,
        )

    def _ensure_screenshot_recorder(self):
        if self.screenshot_recorder is not None or float(self.args.screenshot_interval) <= 0.0:
            return
        try:
            self.screenshot_recorder = self._build_screenshot_recorder()
        except Exception as exc:
            print(f"Warning: failed to initialize SVG screenshot recorder: {exc}")
            self.screenshot_recorder = None

    def _ensure_video_recorder(self):
        if self.video_recorder is not None or self.video_output_path is None:
            return
        try:
            self.video_recorder = self._build_video_recorder()
            if self.video_recorder is not None:
                print(
                    "[AllegroTabletopTrigraspDemo] MP4 capture enabled: "
                    f"path={self.video_output_path}, fps={float(self.args.video_fps):.2f}, "
                    f"size={int(self.args.video_width)}x{int(self.args.video_height)}"
                )
        except Exception as exc:
            print(f"Warning: failed to initialize MP4 recorder: {exc}")
            self.video_recorder = None

    def sync_viewer(self):
        self.scene_camera_config = self._build_focus_camera_config()
        if self.screenshot_recorder is not None:
            self.screenshot_recorder.camera.lookat[:] = np.asarray(self.scene_camera_config.lookat, dtype=np.float64)
            self.screenshot_recorder.camera.distance = float(self.scene_camera_config.distance)
            self.screenshot_recorder.camera.azimuth = float(self.scene_camera_config.azimuth_deg)
            self.screenshot_recorder.camera.elevation = float(self.scene_camera_config.elevation_deg)
        if self.video_recorder is not None:
            self.video_recorder.camera.lookat[:] = np.asarray(self.scene_camera_config.lookat, dtype=np.float64)
            self.video_recorder.camera.distance = float(self.scene_camera_config.distance)
            self.video_recorder.camera.azimuth = float(self.scene_camera_config.azimuth_deg)
            self.video_recorder.camera.elevation = float(self.scene_camera_config.elevation_deg)
        if self.viewer is not None:
            self.viewer.sync()
        if self.screenshot_recorder is not None:
            self.screenshot_recorder.capture_if_due(self.data)
        if self.video_recorder is not None:
            self.video_recorder.capture_if_due(self.data)

    def _refresh_initial_object_pose_cache(self, object_pos, object_quat):
        self.initial_object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3).copy()
        self.initial_object_quat = np.asarray(object_quat, dtype=np.float64).reshape(4).copy()
        self.initial_object_quat /= max(np.linalg.norm(self.initial_object_quat), 1e-9)
        self.initial_object_rot = quat_wxyz_to_mat(self.initial_object_quat)
        self.object_top_z = float(self.initial_object_pos[2] + self.mesh_bounds[1, 2])
        self.hover_center_xy = self.initial_object_pos[:2].copy()
        if hasattr(self, "object_qpos_adr"):
            self.object_qpos_init = np.concatenate([self.initial_object_pos, self.initial_object_quat], axis=0)
            self.ik_reference_qpos[self.object_qpos_adr : self.object_qpos_adr + 7] = self.object_qpos_init

    def settle_initial_object_pose(self, hand_qpos):
        settle_steps = max(int(self.args.initial_settle_steps), 0)
        if settle_steps <= 0:
            return

        hand_qpos = np.asarray(hand_qpos, dtype=np.float64).reshape(self.model.nu)
        self.reset(self.initial_object_pos, self.initial_object_quat, hand_qpos)
        self.data.ctrl[:] = hand_qpos

        stable_steps = 0
        velocity_tol = float(self.args.initial_settle_velocity_tol)
        required_stable_steps = min(25, settle_steps)
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
            object_twist = np.asarray(self.data.qvel[self.object_dof_adr : self.object_dof_adr + 6], dtype=np.float64)
            if float(np.linalg.norm(object_twist)) <= velocity_tol:
                stable_steps += 1
                if stable_steps >= required_stable_steps:
                    break
            else:
                stable_steps = 0

        settled_pos, settled_quat, _ = self.get_object_pose(self.data)
        self._refresh_initial_object_pose_cache(settled_pos, settled_quat)
        self.reset(self.initial_object_pos, self.initial_object_quat, hand_qpos)

    def reset(self, object_pos, object_quat, hand_qpos):
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.set_object_pose(object_pos, object_quat, data=self.data, forward=False)
        self.set_hand_qpos(hand_qpos, data=self.data, forward=False)
        mujoco.mj_forward(self.model, self.data)

        self.solve_data.qpos[:] = self.model.qpos0
        self.solve_data.qvel[:] = 0.0
        self.set_object_pose(object_pos, object_quat, data=self.solve_data, forward=False)
        mujoco.mj_forward(self.model, self.solve_data)
        self.refresh_markers()
        self.sync_viewer()

    def set_object_pose(self, object_pos, object_quat, data=None, forward=True):
        data = self.data if data is None else data
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_quat = np.asarray(object_quat, dtype=np.float64).reshape(4)
        object_quat = object_quat / max(np.linalg.norm(object_quat), 1e-9)
        data.qpos[self.object_qpos_adr : self.object_qpos_adr + 3] = object_pos
        data.qpos[self.object_qpos_adr + 3 : self.object_qpos_adr + 7] = object_quat
        data.qvel[self.object_dof_adr : self.object_dof_adr + 6] = 0.0
        if forward:
            mujoco.mj_forward(self.model, data)

    def get_object_pose(self, data=None):
        data = self.data if data is None else data
        pos = np.asarray(data.qpos[self.object_qpos_adr : self.object_qpos_adr + 3], dtype=np.float64).copy()
        quat = np.asarray(data.qpos[self.object_qpos_adr + 3 : self.object_qpos_adr + 7], dtype=np.float64).copy()
        quat /= max(np.linalg.norm(quat), 1e-9)
        rot = quat_wxyz_to_mat(quat)
        return pos, quat, rot

    def set_marker_pose(self, marker_name, pos, quat=None):
        body_id = self.marker_body_ids[marker_name]
        self.model.body_pos[body_id] = np.asarray(pos, dtype=np.float64).reshape(3)
        if quat is not None:
            quat = np.asarray(quat, dtype=np.float64).reshape(4)
            quat /= max(np.linalg.norm(quat), 1e-9)
            self.model.body_quat[body_id] = quat

    def set_hand_qpos(self, hand_qpos, data=None, forward=True):
        data = self.data if data is None else data
        hand_qpos = np.asarray(hand_qpos, dtype=np.float64).reshape(self.model.nu)
        data.qpos[self.hand_qpos_adr] = hand_qpos
        data.qvel[self.hand_dof_adr] = 0.0
        if data.ctrl.shape[0] == self.model.nu:
            data.ctrl[:] = hand_qpos
        if forward:
            mujoco.mj_forward(self.model, data)

    def set_base_qpos(self, base_qpos, data=None, forward=True):
        data = self.data if data is None else data
        base_qpos = np.asarray(base_qpos, dtype=np.float64).reshape(BASE_ACTUATOR_COUNT)
        data.qpos[self.base_qpos_adr] = base_qpos
        data.qvel[self.base_dof_adr] = 0.0
        if data.ctrl.shape[0] >= BASE_ACTUATOR_COUNT:
            data.ctrl[:BASE_ACTUATOR_COUNT] = base_qpos
        if forward:
            mujoco.mj_forward(self.model, data)

    def get_hand_qpos(self, data=None):
        data = self.data if data is None else data
        return np.asarray(data.qpos[self.hand_qpos_adr], dtype=np.float64).copy()

    def get_finger_qpos(self, data=None):
        data = self.data if data is None else data
        return np.asarray(data.qpos[self.finger_qpos_adr], dtype=np.float64).copy()

    def get_base_qpos(self, data=None):
        data = self.data if data is None else data
        return np.asarray(data.qpos[self.base_qpos_adr], dtype=np.float64).copy()

    def get_mpc_state(self, data=None):
        data = self.data if data is None else data
        object_pos, object_quat, _ = self.get_object_pose(data)
        base_pos = np.asarray(data.qpos[self.base_pos_qpos_adr], dtype=np.float64).copy()
        finger_q = self.get_finger_qpos(data)
        return np.concatenate((object_pos, object_quat, base_pos, finger_q), axis=0)

    def detect_mpc_contacts(self):
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

        packed_contacts = []
        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            if contact.geom1 == self.object_geom_id:
                other_body_id = int(self.model.geom_bodyid[contact.geom2])
            elif contact.geom2 == self.object_geom_id:
                other_body_id = int(self.model.geom_bodyid[contact.geom1])
            else:
                continue

            con_pos = np.asarray(contact.pos, dtype=np.float64).copy()
            con_dist = float(contact.dist) * 0.5
            con_frame = np.asarray(contact.frame, dtype=np.float64).reshape((-1, 3)).T
            con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

            jacp_obj = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jac(self.model, self.data, jacp=jacp_obj, jacr=None, point=con_pos, body=self.object_body_id)
            jacp_obj = con_frame_pmd.T @ jacp_obj

            jacp_other = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jac(self.model, self.data, jacp=jacp_other, jacr=None, point=con_pos, body=other_body_id)
            jacp_other = con_frame_pmd.T @ jacp_other

            con_jacp = jacp_obj - jacp_other
            con_jacp_n = con_jacp[0]
            con_jacp_f = con_jacp[1:]
            con_jac_full = con_jacp_n + float(self.mpc_param.mu_object_) * con_jacp_f
            packed_contacts.append((con_dist, con_jac_full[:, self.mpc_qvel_full_indices]))

        packed_contacts.sort(key=lambda item: item[0])
        phi_vec = np.ones((self.mpc_param.max_ncon_ * 4,), dtype=np.float64)
        jac_mat = np.zeros((self.mpc_param.max_ncon_ * 4, self.mpc_param.n_qvel_), dtype=np.float64)
        for contact_idx, (con_dist, con_jac) in enumerate(packed_contacts[: self.mpc_param.max_ncon_]):
            phi_vec[4 * contact_idx : 4 * contact_idx + 4] = con_dist
            jac_mat[4 * contact_idx : 4 * contact_idx + 4] = con_jac
        return phi_vec, jac_mat

    def apply_mpc_action(self, action, base_orientation, target_base_q=None, freeze_fingers=False):
        action = np.asarray(action, dtype=np.float64).reshape(self.mpc_param.n_cmd_)
        base_orientation = np.asarray(base_orientation, dtype=np.float64).reshape(BASE_ACTUATOR_COUNT - BASE_TRANSLATION_DOF_COUNT)
        if target_base_q is not None:
            target_base_q = np.asarray(target_base_q, dtype=np.float64).reshape(BASE_ACTUATOR_COUNT)

        curr_base_q = self.get_base_qpos(self.data)
        next_base_q = curr_base_q.copy()
        next_base_q[:BASE_TRANSLATION_DOF_COUNT] = np.clip(
            curr_base_q[:BASE_TRANSLATION_DOF_COUNT] + action[:BASE_TRANSLATION_DOF_COUNT],
            self.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT, 0],
            self.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT, 1],
        )
        next_base_q[BASE_TRANSLATION_DOF_COUNT:] = base_orientation

        curr_finger_q = self.get_finger_qpos(self.data)
        should_freeze_fingers = bool(freeze_fingers)
        if target_base_q is not None:
            base_pos_error = float(
                np.linalg.norm(
                    curr_base_q[:BASE_TRANSLATION_DOF_COUNT] - target_base_q[:BASE_TRANSLATION_DOF_COUNT]
                )
            )
            if base_pos_error > float(self.args.finger_ctrl_release_pos_tol):
                should_freeze_fingers = True

        if should_freeze_fingers:
            target_finger_q = curr_finger_q.copy()
        else:
            target_finger_q = np.clip(
                curr_finger_q + action[BASE_TRANSLATION_DOF_COUNT:],
                self.finger_lower,
                self.finger_upper,
            )

        self.data.ctrl[:BASE_ACTUATOR_COUNT] = next_base_q
        self.data.ctrl[BASE_ACTUATOR_COUNT:] = target_finger_q

        for _ in range(max(int(self.args.mj_steps_per_command), 1)):
            mujoco.mj_step(self.model, self.data)

    def assemble_ik_qpos(self, hand_qpos):
        qpos = self.ik_reference_qpos.copy()
        qpos[self.hand_qpos_adr] = np.asarray(hand_qpos, dtype=np.float64).reshape(self.model.nu)
        return qpos

    def forward_kinematics(self, hand_qpos, data=None):
        data = self.solve_data if data is None else data
        data.qpos[:] = self.assemble_ik_qpos(hand_qpos)
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        palm_pos = np.asarray(data.body(self.palm_body_id).xpos, dtype=np.float64).copy()
        palm_rot = np.asarray(data.body(self.palm_body_id).xmat, dtype=np.float64).reshape(3, 3).copy()
        tip_positions = np.stack([np.asarray(data.site(site_id).xpos, dtype=np.float64).copy() for site_id in self.tip_site_ids], axis=0)
        return palm_pos, palm_rot, tip_positions

    def local_tip_positions(self, hand_qpos):
        palm_pos, palm_rot, tip_positions = self.forward_kinematics(hand_qpos)
        return (palm_rot.T @ (tip_positions - palm_pos[None, :]).T).T

    def build_world_contact_targets(self):
        visible_idx = np.where(
            np.asarray(self.optimizer.sample_point[:, 2], dtype=np.float64) >= self.local_top_half_z_threshold
        )[0]
        if visible_idx.size == 0:
            visible_idx = np.arange(self.optimizer.sample_num, dtype=int)

        grasp_result = self.optimizer.get_best_grasp(
            visible_face_idx=visible_idx,
            object_pos=self.initial_object_pos,
            object_rot=self.initial_object_rot,
            support_surface_point=self.support_surface_point,
            support_surface_normal=self.support_surface_normal,
            support_surface_clearance=float(self.args.ground_height_margin),
            support_surface_normal_alignment_threshold=float(self.args.support_normal_alignment_threshold),
        )
        if grasp_result is None:
            raise RuntimeError("mlqp_point_v2 failed to produce a 4-contact grasp candidate.")

        contact_points_local = np.asarray(grasp_result["contact_points"], dtype=np.float64).reshape(-1, 3)
        contact_normals_local = np.asarray(grasp_result["contact_normals"], dtype=np.float64).reshape(-1, 3)
        contact_points_world = (self.initial_object_rot @ contact_points_local.T).T + self.initial_object_pos[None, :]
        contact_normals_world = (self.initial_object_rot @ contact_normals_local.T).T
        fingertip_targets_world = contact_points_world + float(self.args.contact_tip_offset) * contact_normals_world
        return grasp_result, contact_points_local, contact_normals_local, contact_points_world, contact_normals_world, fingertip_targets_world

    def build_top_down_palm_rotation(self, nominal_local_tip_positions, assigned_targets_world):
        nominal_local_tip_positions = np.asarray(nominal_local_tip_positions, dtype=np.float64).reshape(-1, 3)
        assigned_targets_world = np.asarray(assigned_targets_world, dtype=np.float64).reshape(-1, 3)

        local_centroid = np.mean(nominal_local_tip_positions, axis=0)
        world_centroid = np.mean(assigned_targets_world, axis=0)
        centered_local = nominal_local_tip_positions - local_centroid[None, :]
        _, _, vh = np.linalg.svd(centered_local, full_matrices=False)

        palm_normal_local = normalize(vh[-1])
        if float(np.dot(palm_normal_local, local_centroid)) < 0.0:
            palm_normal_local = -palm_normal_local

        thumb_local = project_to_plane(nominal_local_tip_positions[0] - local_centroid, palm_normal_local)
        if np.linalg.norm(thumb_local) < 1e-9:
            thumb_local = project_to_plane(vh[0], palm_normal_local)
        thumb_local = normalize(thumb_local)
        if np.linalg.norm(thumb_local) < 1e-9:
            thumb_local = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        span_local = normalize(np.cross(palm_normal_local, thumb_local))

        down_axis_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        thumb_world = project_to_plane(assigned_targets_world[0] - world_centroid, down_axis_world)
        if np.linalg.norm(thumb_world) < 1e-9:
            thumb_world = project_to_plane(np.mean(assigned_targets_world[1:], axis=0) - world_centroid, down_axis_world)
        thumb_world = normalize(thumb_world)
        if np.linalg.norm(thumb_world) < 1e-9:
            thumb_world = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        span_world = normalize(np.cross(down_axis_world, thumb_world))

        other_local = project_to_plane(np.mean(nominal_local_tip_positions[1:], axis=0) - local_centroid, palm_normal_local)
        other_world = project_to_plane(np.mean(assigned_targets_world[1:], axis=0) - world_centroid, down_axis_world)
        if np.linalg.norm(other_local) >= 1e-9 and np.linalg.norm(other_world) >= 1e-9:
            if float(np.dot(normalize(other_local), span_local)) * float(np.dot(normalize(other_world), span_world)) < 0.0:
                thumb_world = -thumb_world
                span_world = -span_world

        local_frame = np.column_stack([palm_normal_local, thumb_local, span_local])
        world_frame = np.column_stack([down_axis_world, thumb_world, span_world])
        return project_to_rotation_matrix(world_frame @ local_frame.T)

    def solve_single_ik_candidate(self, q_init, target_palm_pos, target_palm_rot, tip_targets_world):
        q = np.asarray(q_init, dtype=np.float64).reshape(self.model.nu).copy()
        q_nominal = q.copy()
        best_payload = None
        best_score = np.inf
        identity = np.eye(self.hand_dof_adr.size, dtype=np.float64)
        regularization_weights = np.concatenate(
            [
                float(self.args.ik_base_regularization) * np.ones(BASE_ACTUATOR_COUNT, dtype=np.float64),
                float(self.args.ik_finger_regularization) * np.ones(self.hand_dof_adr.size - BASE_ACTUATOR_COUNT, dtype=np.float64),
            ]
        )

        for _ in range(int(self.args.ik_max_iters)):
            palm_pos, palm_rot, tip_positions = self.forward_kinematics(q)
            error_terms = []
            jacobian_terms = []

            palm_jacp_full = np.zeros((3, self.model.nv), dtype=np.float64)
            palm_jacr_full = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self.solve_data, palm_jacp_full, palm_jacr_full, self.palm_body_id)
            palm_jacp = palm_jacp_full[:, self.hand_dof_adr]
            palm_jacr = palm_jacr_full[:, self.hand_dof_adr]
            error_terms.append(float(self.args.ik_palm_pos_weight) * (target_palm_pos - palm_pos))
            jacobian_terms.append(float(self.args.ik_palm_pos_weight) * palm_jacp)
            error_terms.append(float(self.args.ik_palm_rot_weight) * rotation_error(palm_rot, target_palm_rot))
            jacobian_terms.append(float(self.args.ik_palm_rot_weight) * palm_jacr)

            for site_id, target_tip in zip(self.tip_site_ids, tip_targets_world):
                jacp_full = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr_full = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, self.solve_data, jacp_full, jacr_full, site_id)
                jacp = jacp_full[:, self.hand_dof_adr]
                error_terms.append(float(self.args.ik_tip_weight) * (target_tip - self.solve_data.site(site_id).xpos))
                jacobian_terms.append(float(self.args.ik_tip_weight) * jacp)

            error_terms.append(regularization_weights * (q_nominal - q))
            jacobian_terms.append(np.diag(regularization_weights))

            error_vector = np.concatenate(error_terms)
            jacobian = np.vstack(jacobian_terms)
            lhs = jacobian.T @ jacobian + float(self.args.ik_damping) * identity
            rhs = jacobian.T @ error_vector
            dq = np.linalg.solve(lhs, rhs)
            q += float(self.args.ik_step_size) * dq

            for joint_idx in range(q.shape[0]):
                if self.hand_joint_limited[joint_idx]:
                    q[joint_idx] = np.clip(q[joint_idx], self.hand_joint_ranges[joint_idx, 0], self.hand_joint_ranges[joint_idx, 1])

            palm_pos, palm_rot, tip_positions = self.forward_kinematics(q)
            tip_error_norms = np.linalg.norm(tip_positions - tip_targets_world, axis=1)
            mean_tip_error = float(np.mean(tip_error_norms))
            max_tip_error = float(np.max(tip_error_norms))
            palm_pos_error = float(np.linalg.norm(target_palm_pos - palm_pos))
            palm_rot_error = float(np.linalg.norm(rotation_error(palm_rot, target_palm_rot)))
            score = mean_tip_error + 0.35 * max_tip_error + 0.20 * palm_pos_error + 0.05 * palm_rot_error
            if score < best_score:
                best_score = score
                best_payload = (
                    q.copy(),
                    mean_tip_error,
                    max_tip_error,
                    palm_pos_error,
                    palm_rot_error,
                    tip_positions.copy(),
                )
            if mean_tip_error <= float(self.args.ik_tip_tol) and max_tip_error <= float(self.args.ik_tip_tol) * 1.5:
                break

        if best_payload is None:
            raise RuntimeError("IK solver did not produce any candidate pose.")
        return best_score, *best_payload

    def solve_grasp_ik(self, fingertip_targets_world):
        best_result = None
        best_score = np.inf
        open_finger_q = self.build_open_finger_pose(None)
        q_seed = np.zeros(self.model.nu, dtype=np.float64)
        q_seed[BASE_ACTUATOR_COUNT:] = open_finger_q.copy()
        nominal_local_tip_positions = self.local_tip_positions(q_seed)
        nominal_local_tip_centroid = np.mean(nominal_local_tip_positions, axis=0)

        for permutation in itertools.permutations(range(len(FINGER_NAMES))):
            assigned_targets_world = fingertip_targets_world[list(permutation)]
            target_palm_rot_world = self.build_top_down_palm_rotation(nominal_local_tip_positions, assigned_targets_world)
            target_palm_pos_world = np.mean(assigned_targets_world, axis=0) - target_palm_rot_world @ nominal_local_tip_centroid

            hover_palm_pos_world = target_palm_pos_world.copy()
            hover_palm_pos_world[0] = self.initial_object_pos[0]
            hover_palm_pos_world[1] = self.initial_object_pos[1]
            hover_palm_pos_world[2] = max(
                float(target_palm_pos_world[2] + self.args.approach_height),
                float(self.object_top_z + self.args.approach_height),
            )

            q_init = np.zeros(self.model.nu, dtype=np.float64)
            q_init[:3] = hover_palm_pos_world
            q_init[3:6] = self.base_orientation_euler_from_target_palm_rot(target_palm_rot_world)
            q_init[BASE_ACTUATOR_COUNT:] = open_finger_q.copy()

            score, qpos, mean_tip_error, max_tip_error, palm_pos_error, palm_rot_error, solved_tip_positions = self.solve_single_ik_candidate(
                q_init=q_init,
                target_palm_pos=target_palm_pos_world,
                target_palm_rot=target_palm_rot_world,
                tip_targets_world=assigned_targets_world,
            )

            if score < best_score:
                best_score = score
                best_result = IkSolveResult(
                    success=mean_tip_error <= float(self.args.ik_tip_tol) and max_tip_error <= float(self.args.ik_tip_tol) * 1.5,
                    qpos=qpos,
                    score=score,
                    mean_tip_error=mean_tip_error,
                    max_tip_error=max_tip_error,
                    palm_pos_error=palm_pos_error,
                    palm_rot_error=palm_rot_error,
                    permutation=tuple(int(idx) for idx in permutation),
                    target_palm_pos_world=np.asarray(target_palm_pos_world, dtype=np.float64).copy(),
                    target_palm_rot_world=np.asarray(target_palm_rot_world, dtype=np.float64).copy(),
                    assigned_targets_world=np.asarray(assigned_targets_world, dtype=np.float64).copy(),
                    solved_tip_positions_world=np.asarray(solved_tip_positions, dtype=np.float64).copy(),
                )

        if best_result is None:
            raise RuntimeError("Failed to find a valid Allegro IK solution for the four contact targets.")
        return best_result

    def build_open_finger_pose(self, final_finger_q):
        open_q = self.finger_lower + float(self.args.open_pose_fraction) * (self.finger_upper - self.finger_lower)
        if final_finger_q is not None:
            final_finger_q = np.asarray(final_finger_q, dtype=np.float64).reshape(open_q.shape)
            open_q = np.minimum(open_q, final_finger_q - float(self.args.min_closing_margin))
        return np.clip(open_q, self.finger_lower, self.finger_upper)

    def build_default_hand_q(self):
        hover_clearance = max(float(self.args.approach_height), 0.08)
        default_palm_pos_world = np.array(
            [
                self.initial_object_pos[0],
                self.initial_object_pos[1],
                max(
                    float(self.object_top_z + hover_clearance),
                    float(self.support_top + 0.5 * hover_clearance),
                ),
            ],
            dtype=np.float64,
        )
        return self.build_hover_hand_q(
            palm_pos_world=default_palm_pos_world,
            finger_q=self.build_open_finger_pose(None),
        )

    def base_orientation_euler_from_target_palm_rot(self, target_palm_rot_world):
        target_palm_rot_world = project_to_rotation_matrix(target_palm_rot_world)
        target_base_rot_world = project_to_rotation_matrix(target_palm_rot_world @ self.fixed_palm_rot_world.T)
        return Rotation.from_matrix(target_base_rot_world).as_euler("xyz")

    def build_hover_hand_q(self, palm_pos_world, base_orientation=None, finger_q=None, max_iters=3):
        palm_pos_world = np.asarray(palm_pos_world, dtype=np.float64).reshape(3)
        if base_orientation is None:
            base_orientation = np.zeros(BASE_ACTUATOR_COUNT - BASE_TRANSLATION_DOF_COUNT, dtype=np.float64)
        else:
            base_orientation = np.asarray(base_orientation, dtype=np.float64).reshape(BASE_ACTUATOR_COUNT - BASE_TRANSLATION_DOF_COUNT)
        if finger_q is None:
            finger_q = self.build_open_finger_pose(None)
        else:
            finger_q = np.asarray(finger_q, dtype=np.float64).reshape(self.finger_lower.shape)

        hand_q = np.concatenate([palm_pos_world.copy(), base_orientation.copy(), finger_q.copy()], axis=0)
        min_safe_base_pos = np.asarray(self.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT, 0], dtype=np.float64).copy()
        max_safe_base_pos = np.asarray(self.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT, 1], dtype=np.float64).copy()
        min_safe_base_pos[2] = max(
            min_safe_base_pos[2],
            float(self.support_top + self.args.approach_height * 0.5),
            float(palm_pos_world[2]),
        )
        for _ in range(max(int(max_iters), 1)):
            measured_palm_pos, _, _ = self.forward_kinematics(hand_q)
            hand_q[:BASE_TRANSLATION_DOF_COUNT] += palm_pos_world - measured_palm_pos
            hand_q[:BASE_TRANSLATION_DOF_COUNT] = np.clip(
                hand_q[:BASE_TRANSLATION_DOF_COUNT],
                min_safe_base_pos,
                max_safe_base_pos,
            )

        measured_palm_pos, _, _ = self.forward_kinematics(hand_q)
        if measured_palm_pos[2] < palm_pos_world[2]:
            hand_q[2] = min(
                max_safe_base_pos[2],
                max(hand_q[2] + (palm_pos_world[2] - measured_palm_pos[2]), min_safe_base_pos[2]),
            )
        hand_q[self.base_qpos_adr.size :] = finger_q
        return hand_q

    def build_reference_trajectory(self, ik_result):
        final_base_q = np.asarray(ik_result.qpos[:BASE_ACTUATOR_COUNT], dtype=np.float64).copy()
        final_finger_q = np.asarray(ik_result.qpos[BASE_ACTUATOR_COUNT:], dtype=np.float64).copy()
        open_finger_q = self.build_open_finger_pose(final_finger_q)
        default_base_q = np.asarray(self.default_hand_q[:BASE_ACTUATOR_COUNT], dtype=np.float64).copy()
        default_finger_q = np.asarray(self.default_hand_q[BASE_ACTUATOR_COUNT:], dtype=np.float64).copy()

        hover_palm_pos_world = np.array(
            [
                self.initial_object_pos[0],
                self.initial_object_pos[1],
                max(
                    float(final_base_q[2] + self.args.approach_height),
                    float(self.object_top_z + self.args.approach_height),
                ),
            ],
            dtype=np.float64,
        )
        hover_hand_q = self.build_hover_hand_q(
            palm_pos_world=hover_palm_pos_world,
            base_orientation=final_base_q[BASE_TRANSLATION_DOF_COUNT:],
            finger_q=open_finger_q,
        )
        hover_base_q = np.asarray(hover_hand_q[:BASE_ACTUATOR_COUNT], dtype=np.float64).copy()

        lift_base_q = final_base_q.copy()
        lift_base_q[2] += float(self.args.lift_height)
        lift_object_pos = self.initial_object_pos.copy()
        lift_object_pos[2] += float(self.args.lift_height)

        base_traj = []
        finger_traj = []
        object_pos_traj = []
        object_quat_traj = []
        stage_names = []

        def add_stage(name, steps, start_base, end_base, start_finger, end_finger, start_object_pos, end_object_pos):
            if int(steps) <= 0:
                return (
                    np.asarray(end_base, dtype=np.float64).copy(),
                    np.asarray(end_finger, dtype=np.float64).copy(),
                    np.asarray(end_object_pos, dtype=np.float64).copy(),
                )
            start_base = np.asarray(start_base, dtype=np.float64).reshape(BASE_ACTUATOR_COUNT)
            end_base = np.asarray(end_base, dtype=np.float64).reshape(BASE_ACTUATOR_COUNT)
            start_finger = np.asarray(start_finger, dtype=np.float64).reshape(final_finger_q.shape)
            end_finger = np.asarray(end_finger, dtype=np.float64).reshape(final_finger_q.shape)
            start_object_pos = np.asarray(start_object_pos, dtype=np.float64).reshape(3)
            end_object_pos = np.asarray(end_object_pos, dtype=np.float64).reshape(3)
            for step_idx in range(int(steps)):
                alpha = float(step_idx + 1) / float(steps)
                base_traj.append((1.0 - alpha) * start_base + alpha * end_base)
                finger_traj.append((1.0 - alpha) * start_finger + alpha * end_finger)
                object_pos_traj.append((1.0 - alpha) * start_object_pos + alpha * end_object_pos)
                object_quat_traj.append(self.initial_object_quat.copy())
                stage_names.append(str(name))
            return end_base.copy(), end_finger.copy(), end_object_pos.copy()

        curr_base = default_base_q.copy()
        curr_finger = default_finger_q.copy()
        curr_object_pos = self.initial_object_pos.copy()
        curr_base, curr_finger, curr_object_pos = add_stage(
            "hover_open",
            self.args.hover_steps,
            curr_base,
            hover_base_q,
            curr_finger,
            open_finger_q,
            curr_object_pos,
            curr_object_pos,
        )
        curr_base, curr_finger, curr_object_pos = add_stage(
            "translate_to_contact",
            self.args.approach_steps,
            curr_base,
            final_base_q,
            curr_finger,
            curr_finger,
            curr_object_pos,
            self.initial_object_pos,
        )
        curr_base, curr_finger, curr_object_pos = add_stage(
            "squeeze",
            self.args.squeeze_steps,
            curr_base,
            curr_base,
            curr_finger,
            final_finger_q,
            curr_object_pos,
            self.initial_object_pos,
        )
        curr_base, curr_finger, curr_object_pos = add_stage(
            "lift",
            self.args.lift_steps,
            curr_base,
            lift_base_q,
            curr_finger,
            final_finger_q,
            curr_object_pos,
            lift_object_pos,
        )
        curr_base, curr_finger, curr_object_pos = add_stage(
            "hold",
            self.args.hold_steps,
            curr_base,
            lift_base_q,
            curr_finger,
            final_finger_q,
            curr_object_pos,
            lift_object_pos,
        )

        if not base_traj:
            raise RuntimeError("The staged trajectory is empty. Increase hover/approach/squeeze/lift/hold steps.")

        return ReferenceTrajectory(
            initial_base_q=default_base_q,
            initial_finger_q=default_finger_q,
            base_traj=np.asarray(base_traj, dtype=np.float64),
            finger_traj=np.asarray(finger_traj, dtype=np.float64),
            object_pos_traj=np.asarray(object_pos_traj, dtype=np.float64),
            object_quat_traj=np.asarray(object_quat_traj, dtype=np.float64),
            stage_names=tuple(stage_names),
        )

    def cache_visual_targets(self, contact_points_local, contact_normals_local, ik_result):
        self.cached_contact_points_local = np.asarray(contact_points_local, dtype=np.float64).reshape(-1, 3).copy()
        perm = np.asarray(ik_result.permutation, dtype=np.int32)
        assigned_local_points = np.asarray(contact_points_local, dtype=np.float64).reshape(-1, 3)[perm]
        assigned_local_normals = np.asarray(contact_normals_local, dtype=np.float64).reshape(-1, 3)[perm]
        self.cached_target_points_local = assigned_local_points + float(self.args.contact_tip_offset) * assigned_local_normals
        self.cached_palm_target_local_pos = self.initial_object_rot.T @ (ik_result.target_palm_pos_world - self.initial_object_pos)
        self.cached_palm_target_local_rot = self.initial_object_rot.T @ ik_result.target_palm_rot_world
        self.refresh_markers()

    def refresh_markers(self):
        if self.cached_contact_points_local is None:
            mujoco.mj_forward(self.model, self.data)
            return

        object_pos, object_quat, object_rot = self.get_object_pose(self.data)
        contact_points_world = (object_rot @ self.cached_contact_points_local.T).T + object_pos[None, :]
        target_points_world = (object_rot @ self.cached_target_points_local.T).T + object_pos[None, :]
        palm_target_pos_world = object_rot @ self.cached_palm_target_local_pos + object_pos
        palm_target_rot_world = object_rot @ self.cached_palm_target_local_rot
        palm_target_quat_world = mat_to_quat_wxyz(palm_target_rot_world)

        for idx, point in enumerate(contact_points_world, start=1):
            self.set_marker_pose(f"contact_point{idx}", point)
        for finger_name, point in zip(FINGER_NAMES, target_points_world):
            self.set_marker_pose(f"{finger_name}_target", point)
        self.set_marker_pose("palm_target", palm_target_pos_world, palm_target_quat_world)
        mujoco.mj_forward(self.model, self.data)

    def get_fingertip_contact_summary(self):
        summary = {finger_name: 0.0 for finger_name in FINGER_NAMES}
        for contact_idx in range(self.data.ncon):
            contact = self.data.contact[contact_idx]
            if contact.geom1 == self.object_geom_id:
                other_geom = int(contact.geom2)
            elif contact.geom2 == self.object_geom_id:
                other_geom = int(contact.geom1)
            else:
                continue
            finger_name = self.fingertip_geom_id_to_name.get(other_geom)
            if finger_name is None:
                continue
            contact_force = np.zeros(6, dtype=np.float64)
            if hasattr(mujoco, "mj_contactForce"):
                mujoco.mj_contactForce(self.model, self.data, contact_idx, contact_force)
            summary[finger_name] = max(summary[finger_name], float(abs(contact_force[0])))
        return summary

    def count_active_fingertip_contacts(self, contact_summary=None, min_force=None):
        if contact_summary is None:
            contact_summary = self.get_fingertip_contact_summary()
        threshold = float(self.args.contact_force_threshold if min_force is None else min_force)
        return sum(float(normal_force) > threshold for normal_force in contact_summary.values())

    def run(self):
        grasp_result, contact_points_local, contact_normals_local, contact_points_world, contact_normals_world, fingertip_targets_world = self.build_world_contact_targets()
        ik_result = self.solve_grasp_ik(fingertip_targets_world)
        trajectory = self.build_reference_trajectory(ik_result)

        initial_hand_q = np.concatenate([trajectory.initial_base_q, trajectory.initial_finger_q], axis=0)
        self.reset(self.initial_object_pos, self.initial_object_quat, initial_hand_q)
        self.cache_visual_targets(contact_points_local, contact_normals_local, ik_result)
        self._ensure_screenshot_recorder()
        self._ensure_video_recorder()
        self.maybe_launch_viewer()
        if self.screenshot_recorder is not None:
            self.screenshot_recorder.capture(self.data, label="traj_start")
        self.sync_viewer()

        print("\nScene")
        print(f"scene_xml: {self.scene_path}")
        print(f"mesh_path: {self.mesh_path}")
        print(f"mesh_scale: {np.array2string(self.mesh_scale, precision=4)}")
        print(f"pedestal_pos_world: {np.array2string(self.pedestal_pos, precision=4)}")
        print(f"pedestal_size: {np.array2string(self.pedestal_size, precision=4)}")
        print(f"object_pos_world: {np.array2string(self.initial_object_pos, precision=4)}")
        print(f"object_quat_wxyz: {np.array2string(self.initial_object_quat, precision=4)}")

        print("\nGrasp Result")
        print(f"num_grasp_contacts: {grasp_result['contact_indices'].shape[0]}")
        print(f"contact_indices: {grasp_result['contact_indices']}")
        print(f"contact_points_local:\n{np.array2string(contact_points_local, precision=4)}")
        print(f"contact_normals_local:\n{np.array2string(contact_normals_local, precision=4)}")
        print(f"contact_points_world:\n{np.array2string(contact_points_world, precision=4)}")
        print(f"contact_normals_world:\n{np.array2string(contact_normals_world, precision=4)}")
        print(f"fingertip_targets_world:\n{np.array2string(fingertip_targets_world, precision=4)}")
        print(f"total_cost: {float(grasp_result['total_cost']):.6f}")
        print(f"region_score: {float(grasp_result.get('region_score', 0.0)):.6f}")
        print(f"antipodal_margin: {float(grasp_result['antipodal_margin']):.6f}")

        print("\nIK Result")
        print(f"success: {ik_result.success}")
        print(f"finger_assignment(permutation of contact order for thumb/index/middle/ring): {ik_result.permutation}")
        print(f"score: {ik_result.score:.6f}")
        print(f"mean_tip_error: {ik_result.mean_tip_error:.6f}")
        print(f"max_tip_error: {ik_result.max_tip_error:.6f}")
        print(f"palm_pos_error: {ik_result.palm_pos_error:.6f}")
        print(f"palm_rot_error: {ik_result.palm_rot_error:.6f}")
        print(f"target_palm_pos_world: {np.array2string(ik_result.target_palm_pos_world, precision=4)}")
        print(
            "target_palm_euler_xyz: "
            + np.array2string(Rotation.from_matrix(ik_result.target_palm_rot_world).as_euler("xyz"), precision=4)
        )
        print(f"solved_hand_qpos:\n{np.array2string(ik_result.qpos, precision=5)}")
        print(f"solved_tip_positions_world:\n{np.array2string(ik_result.solved_tip_positions_world, precision=4)}")

        print("\nTrajectory")
        print(f"command_dt: {self.command_dt:.4f}s")
        print(f"total_command_steps: {trajectory.base_traj.shape[0]}")
        print(f"stages: {', '.join(dict.fromkeys(trajectory.stage_names))}")
        print(f"initial_base_q: {np.array2string(trajectory.initial_base_q, precision=4)}")
        print(f"final_base_q: {np.array2string(ik_result.qpos[:BASE_ACTUATOR_COUNT], precision=4)}")
        print(f"initial_finger_q: {np.array2string(trajectory.initial_finger_q, precision=4)}")
        print(f"final_finger_q: {np.array2string(ik_result.qpos[BASE_ACTUATOR_COUNT:], precision=4)}")
        print(f"mpc_state_dim: {self.mpc_param.n_qpos_}")
        print(f"mpc_action_dim: {self.mpc_param.n_cmd_}")

        best_contact_count = 0
        best_height_gain = 0.0
        last_status_text = "N/A"
        current_stage = None

        nominal_step_count = int(trajectory.base_traj.shape[0])
        pre_lift_base_q = np.asarray(ik_result.qpos[:BASE_ACTUATOR_COUNT], dtype=np.float64).copy()
        pre_lift_finger_q = np.asarray(ik_result.qpos[BASE_ACTUATOR_COUNT:], dtype=np.float64).copy()
        lift_contact_force_threshold = float(
            self.args.contact_force_threshold
            if self.args.lift_contact_force_threshold is None
            else self.args.lift_contact_force_threshold
        )
        lift_contact_fingers = int(self.args.lift_contact_fingers)
        lift_contact_stable_steps = max(int(self.args.lift_contact_stable_steps), 1)
        max_lift_wait_steps = int(self.args.max_lift_wait_steps)
        lift_gate_released = False
        lift_stable_steps = 0
        lift_wait_steps = 0
        step_idx = 0
        traj_idx = 0

        while traj_idx < nominal_step_count:
            nominal_stage_name = trajectory.stage_names[traj_idx]
            target_base_q = trajectory.base_traj[traj_idx].copy()
            target_finger_q = trajectory.finger_traj[traj_idx].copy()
            target_object_pos = trajectory.object_pos_traj[traj_idx].copy()
            target_object_quat = trajectory.object_quat_traj[traj_idx].copy()

            stage_name = nominal_stage_name
            wait_for_lift_contact = False
            pre_step_contact_summary = self.get_fingertip_contact_summary()
            pre_step_contact_count = self.count_active_fingertip_contacts(
                contact_summary=pre_step_contact_summary,
                min_force=lift_contact_force_threshold,
            )

            # Do not chase the lifted object reference until real fingertip contacts are present.
            if nominal_stage_name in {"lift", "hold"} and not lift_gate_released:
                if pre_step_contact_count >= lift_contact_fingers:
                    lift_stable_steps += 1
                else:
                    lift_stable_steps = 0

                if lift_stable_steps >= lift_contact_stable_steps:
                    lift_gate_released = True
                    print(
                        f"Lift gate released: {pre_step_contact_count} fingertip contacts exceeded "
                        f"{lift_contact_force_threshold:.4g} N for {lift_stable_steps} consecutive command steps."
                    )
                else:
                    wait_for_lift_contact = True
                    stage_name = "await_lift_contact"
                    target_base_q = pre_lift_base_q.copy()
                    target_finger_q = pre_lift_finger_q.copy()
                    target_object_pos = self.initial_object_pos.copy()
                    target_object_quat = self.initial_object_quat.copy()
                    lift_wait_steps += 1

            if stage_name != current_stage:
                current_stage = stage_name
                print(f"\nStage -> {current_stage}")

            curr_x = self.get_mpc_state(self.data)
            phi_vec, jac_mat = self.detect_mpc_contacts()
            mpc_result = self.mpc.plan_once(
                target_object_pos=target_object_pos,
                target_object_quat=target_object_quat,
                target_base_pos=target_base_q[:BASE_TRANSLATION_DOF_COUNT],
                target_finger_q=target_finger_q,
                curr_x=curr_x,
                phi_vec=phi_vec,
                jac_mat=jac_mat,
                sol_guess=self.mpc_param.sol_guess_,
            )
            self.mpc_param.sol_guess_ = mpc_result["sol_guess"]
            last_status_text = str(mpc_result["solve_status"])

            self.apply_mpc_action(
                action=mpc_result["action"],
                base_orientation=target_base_q[BASE_TRANSLATION_DOF_COUNT:],
                target_base_q=target_base_q,
                freeze_fingers=stage_name in {"hover_open", "translate_to_contact", "await_lift_contact"},
            )

            self.refresh_markers()
            self.sync_viewer()

            object_pos, _, _ = self.get_object_pose(self.data)
            height_gain = float(object_pos[2] - self.initial_object_pos[2])
            best_height_gain = max(best_height_gain, height_gain)

            contact_summary = self.get_fingertip_contact_summary()
            contact_count = self.count_active_fingertip_contacts(
                contact_summary=contact_summary,
                min_force=self.args.contact_force_threshold,
            )
            best_contact_count = max(best_contact_count, contact_count)

            if step_idx % max(int(self.args.log_every), 1) == 0 or traj_idx == nominal_step_count - 1:
                finger_err = float(np.linalg.norm(self.get_finger_qpos(self.data) - target_finger_q))
                base_err = float(
                    np.linalg.norm(
                        self.get_base_qpos(self.data)[:BASE_TRANSLATION_DOF_COUNT]
                        - target_base_q[:BASE_TRANSLATION_DOF_COUNT]
                    )
                )
                obj_err = float(np.linalg.norm(object_pos - target_object_pos))
                extra_gate_text = ""
                if wait_for_lift_contact:
                    extra_gate_text = (
                        f" gate={pre_step_contact_count}/{lift_contact_fingers}"
                        f" stable={lift_stable_steps}/{lift_contact_stable_steps}"
                    )
                print(
                    f"  step={step_idx + 1:04d}/{nominal_step_count:04d} "
                    f"stage={stage_name:<13} "
                    f"obj_z={object_pos[2]:.4f} "
                    f"height_gain={height_gain:.4f} "
                    f"contact_fingers={contact_count} "
                    f"base_err={base_err:.4f} "
                    f"finger_err={finger_err:.4f} "
                    f"obj_err={obj_err:.4f} "
                    f"mpc={last_status_text}"
                    f"{extra_gate_text}"
                )

            step_idx += 1
            if wait_for_lift_contact and (max_lift_wait_steps <= 0 or lift_wait_steps < max_lift_wait_steps):
                continue
            if wait_for_lift_contact and max_lift_wait_steps > 0 and lift_wait_steps == max_lift_wait_steps:
                print(
                    "Warning: lift gate wait budget was exhausted before enough fingertip contact was detected; "
                    "continuing with the nominal lift trajectory."
                )
                lift_gate_released = True
            traj_idx += 1

        final_object_pos, final_object_quat, _ = self.get_object_pose(self.data)
        final_contact_summary = self.get_fingertip_contact_summary()

        print("\nExecution Summary")
        print(f"final_object_pos: {np.array2string(final_object_pos, precision=4)}")
        print(f"final_object_quat_wxyz: {np.array2string(final_object_quat, precision=4)}")
        print(f"final_height_gain: {float(final_object_pos[2] - self.initial_object_pos[2]):.6f}")
        print(f"best_height_gain: {best_height_gain:.6f}")
        print(f"best_contact_finger_count: {best_contact_count}")
        print(f"final_contact_summary: {final_contact_summary}")
        print(f"last_mpc_status: {last_status_text}")

        if self.screenshot_recorder is not None:
            self.screenshot_recorder.capture(self.data, label="traj_end")

        if self.viewer is not None and not bool(self.args.no_wait):
            while self.viewer.is_running():
                self.sync_viewer()
                time.sleep(0.02)

        saved_video_path = self.finalize_video_recording()
        if saved_video_path is not None:
            print(f"saved mp4: {saved_video_path}")

        return {
            "grasp_result": grasp_result,
            "ik_result": ik_result,
            "final_object_pos": final_object_pos,
            "final_object_quat": final_object_quat,
            "best_height_gain": best_height_gain,
            "best_contact_finger_count": best_contact_count,
        }

    def finalize_video_recording(self):
        if self.video_recorder is None:
            return None
        saved_path = self.video_recorder.close()
        self.video_recorder = None
        return saved_path


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Floating-base Allegro tabletop trigrasp demo: solve 4-contact IK and use an explicit contact-aware MPC to control base xyz and finger motion."
    )
    parser.add_argument("--obj", type=str, default="teapot", help="Object asset name in envs/assets/objects.")
    parser.add_argument("--mesh", type=str, default=None, help="Optional absolute mesh path. Overrides --obj.")
    parser.add_argument("--scene-output", type=Path, default=DEFAULT_SCENE_OUTPUT, help="Path of the generated MuJoCo XML scene.")
    parser.add_argument("--scale", type=float, nargs=3, default=None, help="Optional mesh scale, e.g. --scale 1 1 1.")
    parser.add_argument("--scale-multiplier", type=float, default=DEFAULT_MESH_SCALE_MULTIPLIER, help="Default object-scale boost when --scale is omitted.")
    parser.add_argument("--object-pos", type=float, nargs=3, default=None, help="Optional world position override. By default the object is placed on the pedestal top.")
    parser.add_argument("--object-yaw", type=float, default=0.0, help="Object yaw in radians.")
    parser.add_argument("--obj-mass", type=float, default=0.03, help="Dynamic object mass in MuJoCo and the grasp scorer.")
    parser.add_argument("--object-friction", type=float, default=2.5, help="Tangential friction used by the dynamic object geom.")
    parser.add_argument("--pedestal-friction", type=float, default=2.5, help="Tangential friction used by the pedestal top and floor geoms.")
    parser.add_argument("--pedestal-pos", type=float, nargs=3, default=(0.58, 0.0, 0.06), help="Central pedestal position.")
    parser.add_argument("--pedestal-size", type=float, nargs=3, default=(0.08, 0.10, 0.06), help="Central pedestal half sizes.")
    parser.add_argument("--object-z-offset", type=float, default=0.0, help="Extra object height above the pedestal.")
    parser.add_argument("--initial-object-lift", type=float, default=0.2, help="Extra height added to the pedestal/support under the object.")
    parser.add_argument("--initial-settle-steps", type=int, default=200, help="Maximum MuJoCo steps used to let the object settle on the pedestal before grasp planning.")
    parser.add_argument("--initial-settle-velocity-tol", type=float, default=5e-3, help="Object twist-norm threshold used to stop the initial settle rollout early.")
    parser.add_argument("--floor-z", type=float, default=0.0, help="Scene floor height.")
    parser.add_argument("--mujoco-dt", type=float, default=0.0025, help="MuJoCo timestep.")
    parser.add_argument("--mj-steps-per-command", type=int, default=10, help="Number of MuJoCo steps executed after each MPC solve.")
    parser.add_argument("--contact-stiffness", type=float, default=12.5, help="Contact stiffness passed to mlqp_point_v2.")
    parser.add_argument("--sample-num", type=int, default=70, help="Surface sample count used by mlqp_point_v2.")
    parser.add_argument(
        "--solver",
        type=str,
        choices=("acados", "ipopt"),
        default=None,
        help="Optional solver used by mlqp_point_v2 for grasp scoring and static-equilibrium checks.",
    )
    parser.add_argument("--num-grasp-contacts", type=int, default=4, help="Must stay at 4 for Allegro thumb/index/middle/ring.")
    parser.add_argument("--region-contact-samples", type=int, default=5, help="Region sample count passed to mlqp_point_v2.")
    parser.add_argument("--top-region-pairs", type=int, default=3, help="Number of top region groups evaluated by mlqp_point_v2.")
    parser.add_argument("--preselect-region-pairs", type=int, default=200, help="Preselected region group budget passed to mlqp_point_v2.")
    parser.add_argument("--max-point-combination-eval", type=int, default=256, help="Max contact combinations evaluated inside mlqp_point_v2.")
    parser.add_argument("--ground-height-margin", type=float, default=0.003, help="Support-surface clearance given to mlqp_point_v2.")
    parser.add_argument("--support-normal-alignment-threshold", type=float, default=0.25, help="Filter out support-facing contacts below this alignment threshold.")
    parser.add_argument("--contact-tip-offset", type=float, default=0.0, help="Offset fingertip IK targets outward along the contact normals.")
    parser.add_argument("--ik-max-iters", type=int, default=220, help="Maximum iterations per Allegro IK candidate.")
    parser.add_argument("--ik-step-size", type=float, default=0.35, help="Damped least-squares IK step size.")
    parser.add_argument("--ik-damping", type=float, default=2e-4, help="Damped least-squares regularization.")
    parser.add_argument("--ik-tip-weight", type=float, default=4.0, help="Weight on fingertip position residuals.")
    parser.add_argument("--ik-palm-pos-weight", type=float, default=1.2, help="Weight on palm position residual.")
    parser.add_argument("--ik-palm-rot-weight", type=float, default=0.6, help="Weight on palm rotation residual.")
    parser.add_argument("--ik-base-regularization", type=float, default=0.02, help="IK regularization weight for the floating base joints.")
    parser.add_argument("--ik-finger-regularization", type=float, default=0.04, help="IK regularization weight for the finger joints.")
    parser.add_argument("--ik-tip-tol", type=float, default=0.006, help="Tip position error threshold used to mark IK success.")
    parser.add_argument("--approach-height", type=float, default=0.20, help="Hover height above the object used during contact planning and as the stage-1 start pose.")
    parser.add_argument("--hover-steps", type=int, default=20, help="Command steps spent hovering with the hand open.")
    parser.add_argument("--approach-steps", type=int, default=160, help="Command steps used to move the open hand from directly above the object to the final IK target pose.")
    parser.add_argument("--squeeze-steps", type=int, default=70, help="Command steps used to close the fingers only after the hand base has reached the target pose.")
    parser.add_argument("--finger-ctrl-release-pos-tol", type=float, default=0.008, help="Finger ctrl stays frozen until the hand-base xyz error to the stage target drops below this threshold.")
    parser.add_argument("--lift-height", type=float, default=0.2, help="Lift distance after the squeeze stage.")
    parser.add_argument("--lift-steps", type=int, default=120, help="Command steps spent lifting the grasped object.")
    parser.add_argument("--hold-steps", type=int, default=20, help="Extra command steps after lifting.")
    parser.add_argument("--lift-contact-fingers", type=int, default=3, help="Minimum number of fingertips that must report object contact before the lift stage is allowed to start.")
    parser.add_argument("--lift-contact-stable-steps", type=int, default=6, help="Number of consecutive command steps that must satisfy the lift-contact requirement.")
    parser.add_argument("--lift-contact-force-threshold", type=float, default=None, help="Optional normal-force threshold used by the lift gate. Defaults to --contact-force-threshold.")
    parser.add_argument("--max-lift-wait-steps", type=int, default=160, help="Extra command steps spent waiting for fingertip contact before giving up on the lift gate.")
    parser.add_argument("--open-pose-fraction", type=float, default=0.05, help="How far from the lower joint limits the initial open finger pose starts.")
    parser.add_argument("--min-closing-margin", type=float, default=0.02, help="Minimum opening margin maintained between the open pose and the final IK finger pose.")
    parser.add_argument("--mpc-horizon", type=int, default=4, help="Explicit MPC horizon length in command steps.")
    parser.add_argument("--mpc-ipopt-max-iter", type=int, default=50, help="Maximum IPOPT iterations used by the explicit MPC.")
    parser.add_argument("--mpc-velocity-limit", type=float, default=3.5, help="Finger velocity bound used to derive each command-step finger action limit.")
    parser.add_argument("--mpc-base-step-limit", type=float, default=0.01, help="Maximum xyz increment per command applied to the Allegro base.")
    parser.add_argument("--mpc-max-contacts", type=int, default=24, help="Maximum number of object contacts included in the explicit MPC linearization.")
    parser.add_argument("--mpc-base-track-weight", type=float, default=4000.0, help="Running-state tracking weight on the base xyz reference.")
    parser.add_argument("--mpc-q-weight", type=float, default=30.0, help="Running-state tracking weight on the 16 finger joints.")
    parser.add_argument("--mpc-object-pos-weight", type=float, default=120.0, help="Running-state tracking weight on the object position target.")
    parser.add_argument("--mpc-object-quat-weight", type=float, default=5.0, help="Running-state tracking weight on the object quaternion target.")
    parser.add_argument("--mpc-terminal-weight", type=float, default=4.0, help="Multiplier applied to the terminal tracking costs.")
    parser.add_argument("--mpc-u-weight", type=float, default=0.03, help="Control-effort penalty used by the explicit MPC.")
    parser.add_argument("--contact-force-threshold", type=float, default=1e-4, help="Normal-force threshold used when reporting fingertip-object contacts.")
    parser.add_argument("--log-every", type=int, default=10, help="Print one execution log every N command steps.")
    parser.add_argument("--screenshot-dir", type=Path, default=DEFAULT_SCREENSHOT_DIR, help="Directory where SVG screenshots will be written.")
    parser.add_argument("--screenshot-interval", type=float, default=1.0, help="Seconds between automatic SVG screenshots. Use 0 to disable.")
    parser.add_argument("--screenshot-width", type=int, default=1280, help="Screenshot render width.")
    parser.add_argument("--screenshot-height", type=int, default=960, help="Screenshot render height.")
    parser.add_argument("--video-output-dir", type=Path, default=DEFAULT_VIDEO_OUTPUT_DIR, help="Directory where MuJoCo MP4 recordings are written.")
    parser.add_argument("--video-output-path", type=Path, default=None, help="Optional explicit MP4 output path. Overrides --video-output-dir.")
    parser.add_argument("--video-fps", type=float, default=20.0, help="MP4 capture frame rate.")
    parser.add_argument("--video-width", type=int, default=1280, help="MP4 render width.")
    parser.add_argument("--video-height", type=int, default=960, help="MP4 render height.")
    parser.add_argument("--no-video", action="store_true", help="Disable MP4 recording.")
    parser.add_argument("--visualize", dest="visualize", action="store_true", help="Open a passive MuJoCo viewer while executing the trajectory.")
    parser.add_argument("--headless", dest="visualize", action="store_false", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--no-wait", action="store_true", help="With --visualize, exit immediately after the scripted trajectory finishes.")
    parser.set_defaults(visualize=True)
    return parser


def main():
    args = build_argparser().parse_args()
    demo = AllegroTabletopTrigraspDemo(args)
    try:
        demo.run()
    except KeyboardInterrupt:
        print("\nCtrl+C received. Finalizing MP4 recording before exit...")
        try:
            saved_video_path = demo.finalize_video_recording()
        except Exception as exc:
            print(f"Warning: failed to finalize MP4 on Ctrl+C: {exc}")
        else:
            if saved_video_path is not None:
                print(f"saved mp4: {saved_video_path}")
    finally:
        demo.close()


if __name__ == "__main__":
    main()
