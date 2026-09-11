from __future__ import annotations

import argparse
import ast
import copy
import itertools
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

try:
    import torch
except ImportError:
    torch = None

try:
    import warp as wp
    import mujoco_warp as mjwarp
except ImportError:
    wp = None
    mjwarp = None

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from models.explicit_model import ExplicitModel
from planning.screenshot import (
    PeriodicSVGScreenshotRecorder,
    build_free_camera_config_from_position,
    create_mujoco_mp4_recorder,
)
from trigrasp_casadi_param import allegro_tip_positions, grasp_closure_cost

OBJECT_ASSET_DIR = REPO_ROOT / "envs" / "assets" / "objects"
GENERATED_COLLISION_ASSET_DIR = OBJECT_ASSET_DIR / "_generated_collision"
DEFAULT_SCENE_OUTPUT = REPO_ROOT / "envs" / "xmls" / "trigrasp.xml"
DEFAULT_SCREENSHOT_DIR = CURRENT_DIR / "figs"
DEFAULT_VIDEO_OUTPUT_DIR = REPO_ROOT / "outputs" / "videos_grasping"
_SPIDER_ALLEGRO_REL = Path("thirdparty/spider/spider/assets/robots/allegro/right.xml")
_local_spider_xml = REPO_ROOT / _SPIDER_ALLEGRO_REL
SPIDER_ALLEGRO_XML = _local_spider_xml if _local_spider_xml.is_file() else (REPO_ROOT.parent / _SPIDER_ALLEGRO_REL)
SPIDER_ALLEGRO_ASSET_DIR = SPIDER_ALLEGRO_XML.parent / "assets"
DEFAULT_SPIDER_REFERENCE_SCENE = REPO_ROOT / "spider" / "example_datasets" / "processed" / "fair_mon" / "allegro" / "right" / "cat" / "scene.xml"
DEFAULT_SPIDER_REFERENCE_TRAJ = REPO_ROOT / "spider" / "example_datasets" / "processed" / "fair_mon" / "allegro" / "right" / "cat" / "0" / "trajectory_kinematic.npz"
DEFAULT_MESH_SCALE_MULTIPLIER = 1.2
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)

FINGER_NAMES = ("thumb", "index", "middle", "ring")
FK_TO_FINGER_ORDER = np.array([3, 0, 1, 2], dtype=np.int32)
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


@dataclass
class MjwpRolloutPlanResult:
    action: np.ndarray
    best_cost: float
    mean_cost: float
    force_cost: float
    contact_fingers: float
    best_index: int
    elapsed_ms: float
    status: str


@dataclass
class SpiderReferencePose:
    object_xy: np.ndarray
    object_quat: np.ndarray
    object_minus_palm: np.ndarray
    base_orientation: np.ndarray
    finger_q: np.ndarray


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


def torch_quat_wxyz_to_mat(quat_wxyz):
    quat_wxyz = quat_wxyz / torch.clamp(torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True), min=1e-9)
    w, x, y, z = torch.unbind(quat_wxyz, dim=-1)
    rot = torch.empty((*quat_wxyz.shape[:-1], 3, 3), dtype=quat_wxyz.dtype, device=quat_wxyz.device)
    rot[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[..., 0, 1] = 2.0 * (x * y - z * w)
    rot[..., 0, 2] = 2.0 * (x * z + y * w)
    rot[..., 1, 0] = 2.0 * (x * y + z * w)
    rot[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[..., 1, 2] = 2.0 * (y * z - x * w)
    rot[..., 2, 0] = 2.0 * (x * z - y * w)
    rot[..., 2, 1] = 2.0 * (y * z + x * w)
    rot[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


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


def euler_with_z_rotation(base_orientation, delta_yaw):
    base_orientation = np.asarray(base_orientation, dtype=np.float64).reshape(3).copy()
    base_orientation[2] += float(delta_yaw)
    return base_orientation


def base_orientation_to_matrix(base_orientation):
    return Rotation.from_euler("xyz", np.asarray(base_orientation, dtype=np.float64).reshape(3)).as_matrix()


def allegro_tip_positions_np(base_pos, finger_q, base_rot=None):
    base_pos_dm = cs.DM(np.asarray(base_pos, dtype=np.float64).reshape(3))
    finger_q_dm = cs.DM(np.asarray(finger_q, dtype=np.float64).reshape(16))
    base_rot_dm = None
    if base_rot is not None:
        base_rot_dm = cs.DM(np.asarray(base_rot, dtype=np.float64).reshape(3, 3))
    tip_positions = allegro_tip_positions(base_pos_dm, finger_q_dm, base_rot_dm)
    return np.stack([np.asarray(tip_position, dtype=np.float64).reshape(3) for tip_position in tip_positions], axis=0)


def format_vec(vec):
    return " ".join(f"{float(v):.8f}" for v in np.asarray(vec, dtype=np.float64).reshape(-1))


def parse_bool_arg(value):
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_camera_free_arg(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "f", "no", "n", "off", "none"}:
        return None

    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--camera_free must be None/False or a Python-style list like "
            "'[[cam_x, cam_y, cam_z], [look_x, look_y, look_z]]'."
        ) from exc

    parsed_array = np.asarray(parsed, dtype=np.float64)
    if parsed_array.shape == (2, 3):
        camera_position = parsed_array[0].copy()
        lookat = parsed_array[1].copy()
        return camera_position, lookat
    if parsed_array.shape == (6,):
        camera_position = parsed_array[:3].copy()
        lookat = parsed_array[3:].copy()
        return camera_position, lookat

    raise argparse.ArgumentTypeError(
        "--camera_free must contain either 2x3 values or 6 flat values."
    )


def load_spider_reference_pose(scene_path, trajectory_path):
    scene_path = Path(scene_path).expanduser().resolve()
    trajectory_path = Path(trajectory_path).expanduser().resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Spider reference scene does not exist: {scene_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Spider reference trajectory does not exist: {trajectory_path}")

    ref_data = np.load(trajectory_path)
    if "qpos" not in ref_data:
        raise KeyError(f"Spider reference trajectory has no 'qpos' array: {trajectory_path}")
    qpos0 = np.asarray(ref_data["qpos"][0], dtype=np.float64).reshape(-1)
    if qpos0.size < BASE_ACTUATOR_COUNT + 16 + 7:
        raise ValueError(f"Spider reference qpos is too short: expected at least 29 values, got {qpos0.size}.")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    if qpos0.size != model.nq:
        raise ValueError(
            f"Spider reference qpos/model size mismatch: qpos has {qpos0.size}, model.nq is {model.nq}."
        )
    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)

    palm_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    object_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_object")
    if palm_site_id < 0 or object_site_id < 0:
        raise RuntimeError("Spider reference scene must contain right_palm and right_object sites.")

    palm_pos = np.asarray(data.site_xpos[palm_site_id], dtype=np.float64).copy()
    object_pos = np.asarray(data.site_xpos[object_site_id], dtype=np.float64).copy()
    return SpiderReferencePose(
        object_xy=object_pos[:2].copy(),
        object_quat=np.asarray(qpos0[-4:], dtype=np.float64).copy(),
        object_minus_palm=(object_pos - palm_pos).copy(),
        base_orientation=np.asarray(qpos0[BASE_TRANSLATION_DOF_COUNT:BASE_ACTUATOR_COUNT], dtype=np.float64).copy(),
        finger_q=np.asarray(qpos0[BASE_ACTUATOR_COUNT:BASE_ACTUATOR_COUNT + 16], dtype=np.float64).copy(),
    )


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


def patch_robot_default_for_contacts(default_element):
    default_element = copy.deepcopy(default_element)
    collision_default = None
    for node in default_element.iter("default"):
        if node.attrib.get("class") in {"right_collision", "collision"}:
            collision_default = node
            break

    if collision_default is None:
        raise RuntimeError("Failed to find Allegro collision default block in the robot XML.")

    collision_geom_default = None
    for child in list(collision_default):
        if child.tag == "geom":
            collision_geom_default = child
            break

    if collision_geom_default is None:
        raise RuntimeError("Failed to find Allegro collision geom default in the robot XML.")

    collision_geom_default.attrib["contype"] = "1"
    collision_geom_default.attrib["conaffinity"] = "1"
    collision_geom_default.attrib["condim"] = "3"
    return default_element


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
    show=False,
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
    patched_robot_default = patch_robot_default_for_contacts(robot_default)
    for child in list(patched_robot_default):
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
        {
            "name": "obj",
            "pos": format_vec(object_pos),
            "quat": format_vec(object_quat),
            "gravcomp": "1" if bool(show) else "0",
        },
    )
    ET.SubElement(obj_body, "freejoint", {"name": "obj_freejoint"})
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
            "group": "1",
            "rgba": "0.92 0.66 0.30 1",
        },
    )
    ET.SubElement(
        obj_body,
        "geom",
        {
            "name": "obj",
            "type": "mesh",
            "mesh": "object_collision_mesh",
            "mass": f"{float(object_mass):.8f}",
            "contype": "0" if bool(show) else "1",
            "conaffinity": "0" if bool(show) else "1",
            "condim": "3",
            "friction": f"{float(object_friction):.8f} 0.08 0.01",
            "rgba": "1 1 1 0",
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

    ET.SubElement(worldbody, "body", {"name": "palm_target", "pos": format_vec(object_pos), "quat": "1 0 0 0"})

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
        self.contact_weight_ = float(demo.args.mpc_contact_weight)
        self.grasp_closure_weight_ = float(demo.args.mpc_grasp_closure_weight)

        self.sol_guess_ = None

    def build_cost_param_vector(
        self,
        target_object_pos,
        target_object_quat,
        target_base_pos,
        target_base_rot,
        target_finger_q,
        grasp_activation,
    ):
        target_base_rot = np.asarray(target_base_rot, dtype=np.float64).reshape(3, 3)
        return np.concatenate(
            [
                np.asarray(target_object_pos, dtype=np.float64).reshape(3),
                np.asarray(target_object_quat, dtype=np.float64).reshape(4),
                np.asarray(target_base_pos, dtype=np.float64).reshape(BASE_TRANSLATION_DOF_COUNT),
                target_base_rot.reshape(9, order="F"),
                np.asarray(target_finger_q, dtype=np.float64).reshape(self.n_robot_qpos_ - BASE_TRANSLATION_DOF_COUNT),
                np.array([float(grasp_activation)], dtype=np.float64),
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
        target_base_rot = cs.SX.sym("target_base_rot", 9)
        target_finger_q = cs.SX.sym("target_finger_q", self.n_robot_qpos_ - BASE_TRANSLATION_DOF_COUNT)
        grasp_activation = cs.SX.sym("grasp_activation", 1)
        cost_param = cs.vvcat(
            [
                target_object_pos,
                target_object_quat,
                target_base_pos,
                target_base_rot,
                target_finger_q,
                grasp_activation,
            ]
        )

        position_cost = cs.sumsqr(obj_pose[:3] - target_object_pos)
        quaternion_cost = 1.0 - cs.dot(obj_pose[3:7], target_object_quat) ** 2
        base_track_cost = cs.sumsqr(base_pos - target_base_pos)
        finger_track_cost = cs.sumsqr(finger_q - target_finger_q)
        tip_positions = allegro_tip_positions(base_pos, finger_q, target_base_rot)
        contact_cost = sum(cs.sumsqr(obj_pose[:3] - tip_position) for tip_position in tip_positions)
        grasp_closure = grasp_closure_cost(obj_pose[:3], tip_positions)
        control_cost = cs.sumsqr(u)
        grasp_cost = grasp_activation[0] * (
            self.contact_weight_ * contact_cost
            + self.grasp_closure_weight_ * grasp_closure
        )

        path_cost = (
            self.base_track_weight_ * base_track_cost
            + self.finger_track_weight_ * finger_track_cost
            + self.object_pos_weight_ * position_cost
            + self.object_quat_weight_ * quaternion_cost
            + grasp_cost
            + self.u_weight_ * control_cost
        )
        final_cost = self.terminal_weight_ * (
            self.base_track_weight_ * base_track_cost
            + self.finger_track_weight_ * finger_track_cost
            + self.object_pos_weight_ * position_cost
            + self.object_quat_weight_ * quaternion_cost
            + grasp_cost
        )

        path_cost_fn = cs.Function("allegro_trigrasp_path_cost_fn", [x, u, cost_param], [path_cost])
        final_cost_fn = cs.Function("allegro_trigrasp_final_cost_fn", [x, cost_param], [final_cost])
        return path_cost_fn, final_cost_fn


class AllegroExplicitMPC:
    def __init__(self, param):
        self.param_ = param
        self.path_cost_fn, self.final_cost_fn = self.param_.init_cost_fns()
        self.model = ExplicitModel(param)
        self.init_MPC()

    def plan_once(
        self,
        target_object_pos,
        target_object_quat,
        target_base_pos,
        target_base_rot,
        target_finger_q,
        curr_x,
        phi_vec,
        jac_mat,
        grasp_activation=1.0,
        sol_guess=None,
    ):
        if sol_guess is None:
            sol_guess = dict(x0=self.nlp_w0_, lam_x0=self.nlp_lam_x0_, lam_g0=self.nlp_lam_g0_)

        cost_params = self.param_.build_cost_param_vector(
            target_object_pos=target_object_pos,
            target_object_quat=target_object_quat,
            target_base_pos=target_base_pos,
            target_base_rot=target_base_rot,
            target_finger_q=target_finger_q,
            grasp_activation=grasp_activation,
        )
        nlp_param = self.nlp_params_fn_(curr_x, phi_vec, jac_mat, cost_params, self.param_.model_params)
        nlp_lbw, nlp_ubw = self.nlp_bounds_fn_(
            self.param_.mpc_u_lb_,
            self.param_.mpc_u_ub_,
            self.param_.mpc_q_lb_,
            self.param_.mpc_q_ub_,
        )

        raw_sol = self.ipopt_solver(
            x0=sol_guess["x0"],
            lam_x0=sol_guess["lam_x0"],
            lam_g0=sol_guess["lam_g0"],
            lbx=nlp_lbw,
            ubx=nlp_ubw,
            lbg=0.0,
            ubg=0.0,
            p=nlp_param,
        )

        w_opt = raw_sol["x"].full().flatten()
        cost_opt = raw_sol["f"].full().flatten()
        sol_traj = np.reshape(w_opt, (self.param_.mpc_horizon_, self.param_.n_cmd_ + self.param_.n_qpos_))
        opt_u_traj = sol_traj[:, : self.param_.n_cmd_]

        return dict(
            action=opt_u_traj[0, :],
            u_traj=opt_u_traj,
            rollout_q=sol_traj[:, self.param_.n_cmd_ :],
            sol_guess=dict(
                x0=w_opt,
                lam_x0=raw_sol["lam_x"],
                lam_g0=raw_sol["lam_g"],
                opt_cost=raw_sol["f"].full().item(),
            ),
            cost_opt=cost_opt,
            solve_status=self.ipopt_solver.stats()["return_status"],
        )

    def init_MPC(self):
        model_params = cs.SX.sym("model_param", 1)
        phi_vec = cs.SX.sym("phi_vec", self.param_.max_ncon_ * 4)
        jac_mat = cs.SX.sym("jac_mat", self.param_.max_ncon_ * 4, self.param_.n_qvel_)
        cost_params = cs.SX.sym("cost_params", self.path_cost_fn.size_in(2))

        lbu = cs.SX.sym("lbu", self.param_.n_cmd_)
        ubu = cs.SX.sym("ubu", self.param_.n_cmd_)
        lbq = cs.SX.sym("lbq", self.param_.n_qpos_)
        ubq = cs.SX.sym("ubq", self.param_.n_qpos_)

        w, w0, lbw, ubw, g = [], [], [], [], []
        j = 0.0
        q0 = cs.SX.sym("q0", self.param_.n_qpos_)
        qk = q0
        for k in range(self.param_.mpc_horizon_):
            uk = cs.SX.sym(f"u{k}", self.param_.n_cmd_)
            w += [uk]
            lbw += [lbu]
            ubw += [ubu]
            w0 += [cs.DM.zeros(self.param_.n_cmd_)]

            pred_q = self.model.step_once_fn(qk, uk, phi_vec, jac_mat, model_params)
            j += self.path_cost_fn(qk, uk, cost_params)

            qk = cs.SX.sym(f"q{k + 1}", self.param_.n_qpos_)
            w += [qk]
            w0 += [cs.DM.zeros(self.param_.n_qpos_)]
            lbw += [lbq]
            ubw += [ubq]
            g += [pred_q - qk]

        j += self.final_cost_fn(qk, cost_params)

        nlp_params = cs.vvcat([q0, phi_vec, jac_mat, cost_params, model_params])
        nlp_prog = {"f": j, "x": cs.vcat(w), "g": cs.vcat(g), "p": nlp_params}
        nlp_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
            "ipopt.max_iter": self.param_.ipopt_max_iter_,
        }
        self.ipopt_solver = cs.nlpsol("solver", "ipopt", nlp_prog, nlp_opts)

        self.nlp_w0_ = cs.vcat(w0)
        self.nlp_lam_x0_ = cs.DM.zeros(self.nlp_w0_.shape)
        self.nlp_lam_g0_ = cs.DM.zeros(cs.vcat(g).shape)
        self.nlp_bounds_fn_ = cs.Function("nlp_bounds_fn", [lbu, ubu, lbq, ubq], [cs.vcat(lbw), cs.vvcat(ubw)])
        self.nlp_params_fn_ = cs.Function(
            "nlp_params_fn",
            [q0, phi_vec, jac_mat, cost_params, model_params],
            [nlp_params],
        )


class MujocoWarpGraspRolloutController:
    def __init__(self, demo):
        if torch is None or wp is None or mjwarp is None:
            missing = []
            if torch is None:
                missing.append("torch")
            if wp is None:
                missing.append("warp")
            if mjwarp is None:
                missing.append("mujoco_warp")
            raise ImportError(f"Missing required MJWP rollout packages: {', '.join(missing)}")

        self.demo = demo
        self.model = demo.model
        self.args = demo.args
        self.num_samples = int(self.args.mjwp_num_samples)
        self.horizon = int(self.args.mjwp_horizon)
        self.num_iterations = int(self.args.mjwp_iterations)
        self.command_substeps = max(int(self.args.mj_steps_per_command), 1)
        self.temperature = float(self.args.mjwp_temperature)
        self.elite_fraction = float(self.args.mjwp_elite_fraction)
        self.device = str(self.args.mjwp_device)
        self.action_traj = None
        self.graph = None

        if self.num_samples < 2:
            raise ValueError("--mjwp-num-samples must be at least 2.")
        if self.horizon < 1:
            raise ValueError("--mjwp-horizon must be at least 1.")

        kernel_cache_dir = getattr(self.args, "mjwp_kernel_cache_dir", None)
        if kernel_cache_dir:
            wp.config.kernel_cache_dir = str(Path(kernel_cache_dir).expanduser())

        try:
            wp.init()
        except RuntimeError:
            pass

        self._setup_indices()
        self._setup_warp_objects()
        self._setup_cost_constants()

    def _setup_indices(self):
        device = self.device
        self.hand_qpos_idx_t = torch.as_tensor(self.demo.hand_qpos_adr, dtype=torch.long, device=device)
        self.base_qpos_idx_t = torch.as_tensor(self.demo.base_qpos_adr, dtype=torch.long, device=device)
        self.base_pos_qpos_idx_t = torch.as_tensor(self.demo.base_pos_qpos_adr, dtype=torch.long, device=device)
        self.finger_qpos_idx_t = torch.as_tensor(self.demo.finger_qpos_adr, dtype=torch.long, device=device)
        self.tip_site_ids_t = torch.as_tensor(self.demo.tip_site_ids, dtype=torch.long, device=device)
        self.fingertip_geom_ids_t = torch.as_tensor(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, TIP_COLLISION_GEOM_NAMES[finger_name])
                for finger_name in FINGER_NAMES
            ],
            dtype=torch.long,
            device=device,
        )

        base_pos_ranges = self.demo.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT]
        self.base_pos_lower_t = torch.as_tensor(base_pos_ranges[:, 0], dtype=torch.float32, device=device)
        self.base_pos_upper_t = torch.as_tensor(base_pos_ranges[:, 1], dtype=torch.float32, device=device)
        self.finger_lower_t = torch.as_tensor(self.demo.finger_lower, dtype=torch.float32, device=device)
        self.finger_upper_t = torch.as_tensor(self.demo.finger_upper, dtype=torch.float32, device=device)

        action_lb = np.asarray(self.demo.mpc_param.mpc_u_lb_, dtype=np.float32)
        action_ub = np.asarray(self.demo.mpc_param.mpc_u_ub_, dtype=np.float32)
        self.action_lb_t = torch.as_tensor(action_lb, dtype=torch.float32, device=device)
        self.action_ub_t = torch.as_tensor(action_ub, dtype=torch.float32, device=device)

        noise = np.hstack(
            [
                float(self.args.mjwp_base_noise) * np.ones(BASE_TRANSLATION_DOF_COUNT, dtype=np.float32),
                float(self.args.mjwp_finger_noise) * np.ones(self.demo.finger_lower.size, dtype=np.float32),
            ]
        )
        self.noise_scale_t = torch.as_tensor(noise, dtype=torch.float32, device=device)

    def _setup_warp_objects(self):
        wp.set_device(self.device)
        with wp.ScopedDevice(self.device):
            self.model_wp = mjwarp.put_model(self.model)
            nconmax_per_world = max(int(self.args.mjwp_nconmax_per_world), int(self.demo.data.ncon) + 8)
            njmax_per_world = max(int(self.args.mjwp_njmax_per_world), int(self.demo.data.nefc) + 32)
            self.data_wp = mjwarp.put_data(
                self.model,
                self.demo.data,
                nworld=self.num_samples,
                nconmax=nconmax_per_world,
                njmax=njmax_per_world,
            )
            self.naconmax = int(self.data_wp.naconmax)
            self.contact_ids_wp = wp.from_numpy(
                np.arange(self.naconmax, dtype=np.int32),
                dtype=wp.int32,
                device=self.device,
            )
            self.contact_force_local_wp = wp.empty(
                (self.naconmax,),
                dtype=wp.spatial_vector,
                device=self.device,
            )

            if wp.get_device().is_cuda and bool(self.args.mjwp_capture_graph):
                try:
                    mjwarp.step(self.model_wp, self.data_wp)
                    wp.synchronize()
                    with wp.ScopedCapture() as capture:
                        mjwarp.step(self.model_wp, self.data_wp)
                    wp.synchronize()
                    self.graph = capture.graph
                except Exception as exc:
                    print(f"Warning: MJWP CUDA graph capture failed; using eager mjwarp.step. Error: {exc}")
                    self.graph = None

    def _setup_cost_constants(self):
        object_radius = np.linalg.norm(self.demo.collision_mesh_bounds[1] - self.demo.collision_mesh_bounds[0])
        self.object_radius = float(max(object_radius, 1e-3))
        configured_force = float(self.args.mjwp_target_normal_force)
        if configured_force > 0.0:
            self.target_normal_force = configured_force
        else:
            divisor = max(int(self.args.lift_contact_fingers), 1)
            self.target_normal_force = float(self.args.obj_mass) * 9.81 * float(self.args.mjwp_force_safety) / divisor
        self.minimum_total_force = float(self.args.obj_mass) * 9.81 * float(self.args.mjwp_force_safety)

    def _zero_field(self, field_name):
        if not hasattr(self.data_wp, field_name):
            return
        field = getattr(self.data_wp, field_name)
        if hasattr(field, "zero_"):
            field.zero_()

    def sync_from_mujoco(self, data):
        qpos = torch.as_tensor(np.asarray(data.qpos, dtype=np.float32), device=self.device).unsqueeze(0)
        qvel = torch.as_tensor(np.asarray(data.qvel, dtype=np.float32), device=self.device).unsqueeze(0)
        ctrl = torch.as_tensor(np.asarray(data.ctrl, dtype=np.float32), device=self.device).unsqueeze(0)
        time_value = torch.full((self.num_samples,), float(data.time), dtype=torch.float32, device=self.device)

        qpos = qpos.repeat(self.num_samples, 1).contiguous()
        qvel = qvel.repeat(self.num_samples, 1).contiguous()
        ctrl = ctrl.repeat(self.num_samples, 1).contiguous()

        with wp.ScopedDevice(self.device):
            wp.copy(self.data_wp.qpos, wp.from_torch(qpos))
            wp.copy(self.data_wp.qvel, wp.from_torch(qvel))
            wp.copy(self.data_wp.ctrl, wp.from_torch(ctrl))
            wp.copy(self.data_wp.time, wp.from_torch(time_value))
            for field_name in ("qacc", "qacc_warmstart", "qfrc_applied", "xfrc_applied"):
                self._zero_field(field_name)
            mjwarp.forward(self.model_wp, self.data_wp)

    def _launch_step(self):
        if self.graph is None:
            mjwarp.step(self.model_wp, self.data_wp)
        else:
            wp.capture_launch(self.graph)

    def _sample_action_sequences(self, nominal_traj, iteration_idx):
        noise_decay = float(self.args.mjwp_noise_decay) ** float(iteration_idx)
        noise = torch.randn(
            (self.num_samples, self.horizon, self.demo.mpc_param.n_cmd_),
            dtype=torch.float32,
            device=self.device,
        )
        samples = nominal_traj.unsqueeze(0) + noise * self.noise_scale_t.view(1, 1, -1) * noise_decay
        samples[0] = nominal_traj
        return torch.maximum(torch.minimum(samples, self.action_ub_t.view(1, 1, -1)), self.action_lb_t.view(1, 1, -1))

    def _softmax_elite_weights(self, rewards):
        invalid = torch.isnan(rewards) | torch.isinf(rewards)
        if invalid.any():
            finite = rewards[~invalid]
            replacement = finite.min() if finite.numel() else torch.tensor(-1e9, dtype=rewards.dtype, device=rewards.device)
            rewards = torch.where(invalid, replacement, rewards)

        top_k = max(1, int(round(self.elite_fraction * self.num_samples)))
        top_k = min(top_k, self.num_samples)
        top_idx = torch.topk(rewards, k=top_k, largest=True).indices
        weights = torch.zeros_like(rewards)
        top_rewards = rewards[top_idx]
        normalized = (top_rewards - top_rewards.mean()) / (top_rewards.std(unbiased=False) + 1e-3)
        weights[top_idx] = torch.softmax(normalized / max(self.temperature, 1e-6), dim=0)
        return weights

    def _step_samples(self, actions, base_orientation_t, freeze_fingers):
        qpos = wp.to_torch(self.data_wp.qpos)
        curr_base_xyz = qpos.index_select(1, self.base_pos_qpos_idx_t)
        curr_finger_q = qpos.index_select(1, self.finger_qpos_idx_t)

        base_delta = actions[:, :BASE_TRANSLATION_DOF_COUNT]
        finger_delta = actions[:, BASE_TRANSLATION_DOF_COUNT:]
        target_base_xyz = torch.clamp(curr_base_xyz + base_delta, self.base_pos_lower_t, self.base_pos_upper_t)
        target_finger_q = curr_finger_q if bool(freeze_fingers) else torch.clamp(curr_finger_q + finger_delta, self.finger_lower_t, self.finger_upper_t)

        ctrl = torch.empty((self.num_samples, self.model.nu), dtype=torch.float32, device=self.device)
        ctrl[:, :BASE_TRANSLATION_DOF_COUNT] = target_base_xyz
        ctrl[:, BASE_TRANSLATION_DOF_COUNT:BASE_ACTUATOR_COUNT] = base_orientation_t.view(1, -1)
        ctrl[:, BASE_ACTUATOR_COUNT:] = target_finger_q

        with wp.ScopedDevice(self.device):
            wp.copy(self.data_wp.ctrl, wp.from_torch(ctrl.contiguous()))
            for _ in range(self.command_substeps):
                self._launch_step()

    def _target_tip_positions(self, obj_pos, obj_quat):
        if self.demo.cached_target_points_local is None:
            return None
        target_local = torch.as_tensor(
            self.demo.cached_target_points_local,
            dtype=torch.float32,
            device=self.device,
        )
        obj_rot = torch_quat_wxyz_to_mat(obj_quat)
        return torch.einsum("nij,kj->nki", obj_rot, target_local) + obj_pos[:, None, :]

    def _finger_contact_metrics(self, obj_pos):
        with wp.ScopedDevice(self.device):
            mjwarp.contact_force(
                self.model_wp,
                self.data_wp,
                self.contact_ids_wp,
                False,
                self.contact_force_local_wp,
            )

        contact_geom = wp.to_torch(self.data_wp.contact.geom).long()
        contact_worldid = wp.to_torch(self.data_wp.contact.worldid).long()
        contact_pos = wp.to_torch(self.data_wp.contact.pos).float()
        local_force = wp.to_torch(self.contact_force_local_wp).float()
        nacon_t = wp.to_torch(self.data_wp.nacon)
        nacon = int(nacon_t[0].detach().cpu().item()) if nacon_t.numel() else 0

        if nacon <= 0:
            zeros = torch.zeros((self.num_samples, len(FINGER_NAMES)), dtype=torch.float32, device=self.device)
            return zeros, torch.zeros(self.num_samples, dtype=torch.float32, device=self.device), zeros[:, 0], zeros[:, 0]

        valid = torch.arange(self.naconmax, device=self.device) < min(nacon, self.naconmax)
        geom0 = contact_geom[:, 0]
        geom1 = contact_geom[:, 1]
        object_geom = int(self.demo.object_geom_id)
        normal_force = torch.abs(local_force[:, 0])

        force_by_finger = torch.zeros((self.num_samples, len(FINGER_NAMES)), dtype=torch.float32, device=self.device)
        object_fingertip_mask = torch.zeros(self.naconmax, dtype=torch.bool, device=self.device)
        for finger_idx, finger_geom_t in enumerate(self.fingertip_geom_ids_t):
            finger_geom = int(finger_geom_t.detach().cpu().item())
            mask = valid & (
                ((geom0 == object_geom) & (geom1 == finger_geom))
                | ((geom1 == object_geom) & (geom0 == finger_geom))
            )
            object_fingertip_mask |= mask
            if mask.any():
                world_idx = torch.clamp(contact_worldid[mask], 0, self.num_samples - 1)
                force_by_finger[:, finger_idx].scatter_reduce_(
                    0,
                    world_idx,
                    normal_force[mask],
                    reduce="amax",
                    include_self=True,
                )

        total_force_vec = torch.zeros((self.num_samples, 3), dtype=torch.float32, device=self.device)
        total_torque_vec = torch.zeros((self.num_samples, 3), dtype=torch.float32, device=self.device)
        if object_fingertip_mask.any():
            world_idx = torch.clamp(contact_worldid[object_fingertip_mask], 0, self.num_samples - 1)
            pos = contact_pos[object_fingertip_mask]
            center = obj_pos[world_idx]
            force_dir = torch.nn.functional.normalize(center - pos, dim=1, eps=1e-6)
            force_vec = normal_force[object_fingertip_mask, None] * force_dir
            torque_vec = torch.cross(pos - center, force_vec, dim=1)
            total_force_vec.index_add_(0, world_idx, force_vec)
            total_torque_vec.index_add_(0, world_idx, torque_vec)

        net_force_norm = torch.linalg.norm(total_force_vec, dim=1)
        torque_norm = torch.linalg.norm(total_torque_vec, dim=1)
        return force_by_finger, force_by_finger.sum(dim=1), net_force_norm, torque_norm

    def _force_closure_surrogate_cost(self, obj_pos):
        force_by_finger, total_normal_force, net_force_norm, torque_norm = self._finger_contact_metrics(obj_pos)
        target_force = max(float(self.target_normal_force), 1e-6)
        active = force_by_finger > float(self.args.contact_force_threshold)
        active_count = active.sum(dim=1).float()
        force_deficit = torch.relu(target_force - force_by_finger) / target_force
        count_deficit = torch.relu(float(self.args.lift_contact_fingers) - active_count)
        force_mean = force_by_finger.mean(dim=1)
        force_balance = force_by_finger.std(dim=1, unbiased=False) / (force_mean + 1e-5)
        total_deficit = torch.relu(float(self.minimum_total_force) - total_normal_force) / max(float(self.minimum_total_force), 1e-6)
        net_force_cost = net_force_norm / (total_normal_force + 1e-5)
        torque_cost = torque_norm / (total_normal_force * self.object_radius + 1e-5)
        closure_cost = (
            force_deficit.mean(dim=1)
            + 0.35 * count_deficit.square()
            + 0.20 * force_balance
            + 0.25 * total_deficit.square()
            + 0.15 * net_force_cost
            + 0.20 * torque_cost
        )
        return closure_cost, active_count, force_by_finger

    def _state_cost(
        self,
        target_object_pos_t,
        target_object_quat_t,
        target_base_pos_t,
        target_finger_q_t,
        force_cost_weight,
    ):
        qpos = wp.to_torch(self.data_wp.qpos)
        obj_pos = qpos[:, self.demo.object_qpos_adr : self.demo.object_qpos_adr + 3]
        obj_quat = qpos[:, self.demo.object_qpos_adr + 3 : self.demo.object_qpos_adr + 7]
        base_pos = qpos.index_select(1, self.base_pos_qpos_idx_t)
        finger_q = qpos.index_select(1, self.finger_qpos_idx_t)

        obj_pos_cost = torch.sum((obj_pos - target_object_pos_t.view(1, 3)) ** 2, dim=1)
        quat_dot = torch.sum(obj_quat * target_object_quat_t.view(1, 4), dim=1).clamp(-1.0, 1.0)
        obj_quat_cost = 1.0 - quat_dot.square()
        base_cost = torch.sum((base_pos - target_base_pos_t.view(1, 3)) ** 2, dim=1)
        finger_cost = torch.mean((finger_q - target_finger_q_t.view(1, -1)) ** 2, dim=1)

        tip_cost = torch.zeros(self.num_samples, dtype=torch.float32, device=self.device)
        target_tip_world = self._target_tip_positions(obj_pos, obj_quat)
        if target_tip_world is not None:
            tip_pos = wp.to_torch(self.data_wp.site_xpos).index_select(1, self.tip_site_ids_t)
            tip_cost = torch.mean(torch.linalg.norm(tip_pos - target_tip_world, dim=2), dim=1)

        closure_cost, active_count, force_by_finger = self._force_closure_surrogate_cost(obj_pos)
        total_cost = (
            float(self.args.mjwp_object_cost_weight) * obj_pos_cost
            + float(self.args.mjwp_object_quat_cost_weight) * obj_quat_cost
            + float(self.args.mjwp_base_cost_weight) * base_cost
            + float(self.args.mjwp_finger_cost_weight) * finger_cost
            + float(self.args.mjwp_tip_cost_weight) * tip_cost
            + float(force_cost_weight) * closure_cost
        )
        return total_cost, closure_cost, active_count, force_by_finger

    def _rollout_cost(
        self,
        action_samples,
        target_object_pos_t,
        target_object_quat_t,
        target_base_q_t,
        target_finger_q_t,
        freeze_fingers,
        stage_name,
    ):
        force_stages = {"squeeze", "await_lift_contact", "lift", "hold"}
        force_cost_weight = float(self.args.mjwp_force_cost_weight) if str(stage_name) in force_stages else 0.0
        base_orientation_t = target_base_q_t[BASE_TRANSLATION_DOF_COUNT:]
        target_base_pos_t = target_base_q_t[:BASE_TRANSLATION_DOF_COUNT]

        cumulative_cost = torch.zeros(self.num_samples, dtype=torch.float32, device=self.device)
        last_force_cost = torch.zeros(self.num_samples, dtype=torch.float32, device=self.device)
        last_active_count = torch.zeros(self.num_samples, dtype=torch.float32, device=self.device)

        for horizon_idx in range(self.horizon):
            actions = action_samples[:, horizon_idx]
            self._step_samples(actions, base_orientation_t, freeze_fingers)
            step_cost, force_cost, active_count, _ = self._state_cost(
                target_object_pos_t=target_object_pos_t,
                target_object_quat_t=target_object_quat_t,
                target_base_pos_t=target_base_pos_t,
                target_finger_q_t=target_finger_q_t,
                force_cost_weight=force_cost_weight,
            )
            action_cost = torch.mean(actions.square(), dim=1)
            cumulative_cost += step_cost + float(self.args.mjwp_action_cost_weight) * action_cost
            last_force_cost = force_cost
            last_active_count = active_count

        return cumulative_cost / float(self.horizon), last_force_cost, last_active_count

    def _initial_nominal_traj(self, nominal_action):
        nominal_action_t = torch.as_tensor(nominal_action, dtype=torch.float32, device=self.device)
        if self.action_traj is None or self.action_traj.shape != (self.horizon, self.demo.mpc_param.n_cmd_):
            traj = torch.zeros((self.horizon, self.demo.mpc_param.n_cmd_), dtype=torch.float32, device=self.device)
        else:
            traj = self.action_traj.clone()
        traj[0] = nominal_action_t
        return torch.maximum(torch.minimum(traj, self.action_ub_t.view(1, -1)), self.action_lb_t.view(1, -1))

    def plan_once(
        self,
        data,
        nominal_action,
        target_object_pos,
        target_object_quat,
        target_base_q,
        target_finger_q,
        freeze_fingers,
        stage_name,
    ):
        start_time = time.perf_counter()
        nominal_traj = self._initial_nominal_traj(nominal_action)
        target_object_pos_t = torch.as_tensor(target_object_pos, dtype=torch.float32, device=self.device)
        target_object_quat_t = torch.as_tensor(target_object_quat, dtype=torch.float32, device=self.device)
        target_base_q_t = torch.as_tensor(target_base_q, dtype=torch.float32, device=self.device)
        target_finger_q_t = torch.as_tensor(target_finger_q, dtype=torch.float32, device=self.device)

        best_cost = None
        best_force_cost = None
        best_active_count = None
        best_index = 0
        mean_cost = float("nan")
        for iteration_idx in range(self.num_iterations):
            action_samples = self._sample_action_sequences(nominal_traj, iteration_idx)
            self.sync_from_mujoco(data)
            costs, force_cost, active_count = self._rollout_cost(
                action_samples=action_samples,
                target_object_pos_t=target_object_pos_t,
                target_object_quat_t=target_object_quat_t,
                target_base_q_t=target_base_q_t,
                target_finger_q_t=target_finger_q_t,
                freeze_fingers=freeze_fingers,
                stage_name=stage_name,
            )
            mean_cost = float(costs.mean().detach().cpu().item())
            rewards = -costs
            weights = self._softmax_elite_weights(rewards)
            nominal_traj = torch.sum(weights.view(-1, 1, 1) * action_samples, dim=0)

            iteration_best_index = int(torch.argmin(costs).detach().cpu().item())
            iteration_best_cost = float(costs[iteration_best_index].detach().cpu().item())
            if best_cost is None or iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_force_cost = float(force_cost[iteration_best_index].detach().cpu().item())
                best_active_count = float(active_count[iteration_best_index].detach().cpu().item())
                best_index = iteration_best_index

        self.action_traj = torch.zeros_like(nominal_traj)
        if self.horizon > 1:
            self.action_traj[:-1] = nominal_traj[1:].detach()
            self.action_traj[-1] = nominal_traj[-1].detach()
        else:
            self.action_traj[0] = nominal_traj[0].detach()

        sampled_action = nominal_traj[0].detach()
        blend = float(self.args.mjwp_action_blend)
        nominal_action_t = torch.as_tensor(nominal_action, dtype=torch.float32, device=self.device)
        action = (1.0 - blend) * nominal_action_t + blend * sampled_action
        action = torch.maximum(torch.minimum(action, self.action_ub_t), self.action_lb_t)
        elapsed_ms = 1000.0 * (time.perf_counter() - start_time)

        return MjwpRolloutPlanResult(
            action=action.detach().cpu().numpy().astype(np.float64),
            best_cost=float(best_cost if best_cost is not None else np.inf),
            mean_cost=float(mean_cost),
            force_cost=float(best_force_cost if best_force_cost is not None else np.inf),
            contact_fingers=float(best_active_count if best_active_count is not None else 0.0),
            best_index=int(best_index),
            elapsed_ms=float(elapsed_ms),
            status="mjwp_sampling",
        )


class AllegroTabletopTrigraspDemo:
    def __init__(self, args):
        if int(args.lift_contact_fingers) < 1 or int(args.lift_contact_fingers) > len(FINGER_NAMES):
            raise ValueError(f"--lift-contact-fingers must be within [1, {len(FINGER_NAMES)}].")
        if int(args.max_lift_wait_steps) < 0:
            raise ValueError("--max-lift-wait-steps must be non-negative.")

        self.args = args
        self.spider_reference_pose = None
        if bool(args.spider_reference_pose):
            self.spider_reference_pose = load_spider_reference_pose(
                scene_path=args.spider_reference_scene,
                trajectory_path=args.spider_reference_traj,
            )

        self.mesh_path = resolve_mesh_path(args.obj, args.mesh)
        self.collision_mesh_path = prepare_collision_mesh_path(self.mesh_path)
        self.mesh_scale = resolve_mesh_scale(self.mesh_path, args.scale, args.scale_multiplier)
        self.visual_mesh_bounds = load_mesh_bounds(self.mesh_path, self.mesh_scale)
        self.collision_mesh_bounds = load_mesh_bounds(self.collision_mesh_path, self.mesh_scale)

        self.pedestal_size = np.asarray(args.pedestal_size, dtype=np.float64).copy()
        self.pedestal_pos = np.asarray(args.pedestal_pos, dtype=np.float64).copy()
        if args.object_pos is None and self.spider_reference_pose is not None:
            self.pedestal_pos[:2] = self.spider_reference_pose.object_xy
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
                    self.support_top - self.collision_mesh_bounds[0, 2] + float(args.object_z_offset) + float(args.obj_init_height),
                ],
                dtype=np.float64,
            )
        else:
            object_pos = np.asarray(args.object_pos, dtype=np.float64).reshape(3)
        if self.spider_reference_pose is not None:
            object_quat = self.spider_reference_pose.object_quat.copy()
        else:
            object_quat = quat_from_yaw(args.object_yaw)

        self.initial_object_pos = object_pos.copy()
        self.initial_object_quat = object_quat.copy()
        self.initial_object_rot = quat_wxyz_to_mat(self.initial_object_quat)
        self.object_top_z = float(self.initial_object_pos[2] + self.collision_mesh_bounds[1, 2])
        self.hover_center_xy = self.initial_object_pos[:2].copy()

        z_min = float(self.collision_mesh_bounds[0, 2])
        z_max = float(self.collision_mesh_bounds[1, 2])
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
            show=args.show,
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = float(args.mujoco_dt)
        if bool(args.show):
            self.model.opt.gravity[:] = 0.0
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
        self.reset(self.initial_object_pos, self.initial_object_quat, self.default_hand_q)
        self.maybe_launch_viewer()

        self.command_dt = max(int(args.mj_steps_per_command), 1) * float(self.model.opt.timestep)
        self.mpc_param = AllegroExplicitMPCParams(self)
        self.mpc = AllegroExplicitMPC(self.mpc_param)
        self.mjwp_rollout = None
        if bool(args.mjwp_rollout):
            try:
                self.mjwp_rollout = MujocoWarpGraspRolloutController(self)
            except Exception as exc:
                print(f"Warning: MJWP sampling rollout is disabled: {exc}")

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
            return False
        if self.viewer is not None:
            return True
        try:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        except Exception as exc:
            print(f"Warning: failed to launch MuJoCo viewer; falling back to headless execution: {exc}")
            self.args.visualize = False
            return False
        if self.scene_camera_config is None:
            self.scene_camera_config = self._build_focus_camera_config()
        self._apply_camera_config_to_viewer(self.scene_camera_config)
        self._ensure_screenshot_recorder()
        self.sync_viewer()
        return True

    def _build_focus_camera_config(self, lookat=None):
        if self.args.camera_free is not None:
            camera_position, fixed_lookat = self.args.camera_free
            return build_free_camera_config_from_position(
                camera_position=np.asarray(camera_position, dtype=np.float64).reshape(3),
                lookat=np.asarray(fixed_lookat, dtype=np.float64).reshape(3),
            )
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
        self.object_top_z = float(self.initial_object_pos[2] + self.collision_mesh_bounds[1, 2])
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

    def apply_object_wrench_world(self, force_world=None, torque_world=None):
        self.data.xfrc_applied[:] = 0.0
        if force_world is not None:
            self.data.xfrc_applied[self.object_body_id, :3] = np.asarray(
                force_world,
                dtype=np.float64,
            ).reshape(3)
        if torque_world is not None:
            self.data.xfrc_applied[self.object_body_id, 3:] = np.asarray(
                torque_world,
                dtype=np.float64,
            ).reshape(3)

    def _gravity_wrench_local(self, object_rot=None):
        if object_rot is None:
            _, _, object_rot = self.get_object_pose(self.data)
        object_rot = project_to_rotation_matrix(object_rot)
        gravity_force_world = np.array([0.0, 0.0, -float(self.args.obj_mass) * 9.81], dtype=np.float64)
        gravity_force_local = object_rot.T @ gravity_force_world
        return np.hstack([gravity_force_local, np.zeros(3, dtype=np.float64)])

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

    def apply_mpc_action(
        self,
        action,
        base_orientation,
        target_base_q=None,
        freeze_fingers=False,
        ignore_base_release_gate=False,
        object_force_world=None,
        object_torque_world=None,
    ):
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
        if target_base_q is not None and not bool(ignore_base_release_gate):
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
            self.apply_object_wrench_world(
                force_world=object_force_world,
                torque_world=object_torque_world,
            )
            mujoco.mj_step(self.model, self.data)

    def get_target_palm_pose_world(self):
        if self.cached_palm_target_local_pos is None or self.cached_palm_target_local_rot is None:
            return None, None
        object_pos, _, object_rot = self.get_object_pose(self.data)
        palm_target_pos_world = object_rot @ self.cached_palm_target_local_pos + object_pos
        palm_target_rot_world = object_rot @ self.cached_palm_target_local_rot
        return palm_target_pos_world, palm_target_rot_world

    def should_freeze_fingers(self, stage_name):
        stage_name = str(stage_name)
        if stage_name in {"hover_open", "await_lift_contact"}:
            return True, False
        if stage_name != "translate_to_contact":
            return False, False

        target_palm_pos_world, _ = self.get_target_palm_pose_world()
        if target_palm_pos_world is None:
            return True, False

        curr_palm_pos_world = np.asarray(self.data.body(self.palm_body_id).xpos, dtype=np.float64).copy()
        palm_target_distance = float(np.linalg.norm(curr_palm_pos_world - target_palm_pos_world))
        near_contact = palm_target_distance <= float(self.args.finger_ctrl_release_palm_tol)
        return (not near_contact), near_contact

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

    def build_direct_grasp_finger_pose(self):
        fraction = float(self.args.grasp_pose_fraction)
        return np.clip(
            self.finger_lower + fraction * (self.finger_upper - self.finger_lower),
            self.finger_lower,
            self.finger_upper,
        )

    def build_direct_grasp_base_q(self, grasp_finger_q):
        base_orientation = euler_with_z_rotation(np.zeros(3, dtype=np.float64), float(self.args.grasp_base_yaw))
        base_rot = base_orientation_to_matrix(base_orientation)
        local_tips = allegro_tip_positions_np(np.zeros(3, dtype=np.float64), grasp_finger_q)
        local_tip_centroid = np.mean(local_tips, axis=0)

        base_pos = self.initial_object_pos - base_rot @ local_tip_centroid
        base_pos[2] += float(self.args.direct_grasp_z_bias)
        base_pos = np.clip(
            base_pos,
            self.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT, 0],
            self.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT, 1],
        )
        return np.concatenate([base_pos, base_orientation], axis=0)

    def build_world_contact_targets(self, base_q, finger_q):
        base_q = np.asarray(base_q, dtype=np.float64).reshape(BASE_ACTUATOR_COUNT)
        finger_q = np.asarray(finger_q, dtype=np.float64).reshape(self.finger_lower.shape)
        base_rot = base_orientation_to_matrix(base_q[BASE_TRANSLATION_DOF_COUNT:])
        fingertip_targets_world = allegro_tip_positions_np(base_q[:BASE_TRANSLATION_DOF_COUNT], finger_q, base_rot)[
            FK_TO_FINGER_ORDER
        ]

        contact_points_world = fingertip_targets_world.copy()
        raw_normals_world = contact_points_world - self.initial_object_pos[None, :]
        contact_normals_world = np.stack([normalize(normal) for normal in raw_normals_world], axis=0)
        contact_points_local = (self.initial_object_rot.T @ (contact_points_world - self.initial_object_pos[None, :]).T).T
        contact_normals_local = (self.initial_object_rot.T @ contact_normals_world.T).T
        return contact_points_local, contact_normals_local, contact_points_world, contact_normals_world, fingertip_targets_world

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

    def build_spider_reference_hand_q(self):
        ref = self.spider_reference_pose
        if ref is None:
            raise RuntimeError("Spider reference pose is not loaded.")
        target_palm_pos_world = self.initial_object_pos - ref.object_minus_palm
        target_palm_pos_world = np.clip(
            target_palm_pos_world,
            self.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT, 0],
            self.base_joint_ranges[:BASE_TRANSLATION_DOF_COUNT, 1],
        )
        finger_q = np.clip(ref.finger_q, self.finger_lower, self.finger_upper)
        return np.concatenate(
            [
                target_palm_pos_world,
                ref.base_orientation.copy(),
                finger_q,
            ],
            axis=0,
        )

    def build_default_hand_q(self):
        if self.spider_reference_pose is not None:
            return self.build_spider_reference_hand_q()

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
            base_orientation=euler_with_z_rotation(np.zeros(3, dtype=np.float64), np.pi),
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

    def build_direct_reference_trajectory(self):
        final_finger_q = self.build_direct_grasp_finger_pose()
        final_base_q = self.build_direct_grasp_base_q(final_finger_q)
        open_finger_q = self.build_open_finger_pose(None)
        default_base_q = np.asarray(self.default_hand_q[:BASE_ACTUATOR_COUNT], dtype=np.float64).copy()
        default_finger_q = np.asarray(self.default_hand_q[BASE_ACTUATOR_COUNT:], dtype=np.float64).copy()

        hover_base_q = final_base_q.copy()
        hover_base_q[:BASE_TRANSLATION_DOF_COUNT] = final_base_q[:BASE_TRANSLATION_DOF_COUNT].copy()
        hover_base_q[2] = max(
            float(final_base_q[2] + self.args.approach_height),
            float(self.object_top_z + self.args.approach_height),
        )
        hover_hand_q = self.build_hover_hand_q(
            palm_pos_world=hover_base_q[:BASE_TRANSLATION_DOF_COUNT],
            base_orientation=hover_base_q[BASE_TRANSLATION_DOF_COUNT:],
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
            open_finger_q,
            curr_object_pos,
            self.initial_object_pos,
        )
        curr_base, curr_finger, curr_object_pos = add_stage(
            "squeeze",
            self.args.squeeze_steps,
            curr_base,
            final_base_q,
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
        self.cached_target_points_local = assigned_local_points + float(getattr(self.args, "contact_tip_offset", 0.0)) * assigned_local_normals
        self.cached_palm_target_local_pos = self.initial_object_rot.T @ (ik_result.target_palm_pos_world - self.initial_object_pos)
        self.cached_palm_target_local_rot = self.initial_object_rot.T @ ik_result.target_palm_rot_world
        self.refresh_markers()

    def cache_direct_visual_targets(self, contact_points_local, contact_normals_local, target_base_q):
        self.cached_contact_points_local = np.asarray(contact_points_local, dtype=np.float64).reshape(-1, 3).copy()
        self.cached_target_points_local = self.cached_contact_points_local.copy()
        target_base_q = np.asarray(target_base_q, dtype=np.float64).reshape(BASE_ACTUATOR_COUNT)
        target_base_rot = base_orientation_to_matrix(target_base_q[BASE_TRANSLATION_DOF_COUNT:])
        target_palm_pos_world = np.asarray(target_base_q[:BASE_TRANSLATION_DOF_COUNT], dtype=np.float64).copy()
        self.cached_palm_target_local_pos = self.initial_object_rot.T @ (target_palm_pos_world - self.initial_object_pos)
        self.cached_palm_target_local_rot = self.initial_object_rot.T @ target_base_rot @ self.fixed_palm_rot_world
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
        trajectory = self.build_direct_reference_trajectory()
        stage_array = np.asarray(trajectory.stage_names)
        grasp_indices = np.flatnonzero(stage_array == "squeeze")
        if grasp_indices.size == 0:
            grasp_indices = np.flatnonzero(stage_array == "translate_to_contact")
        if grasp_indices.size == 0:
            grasp_indices = np.array([0], dtype=np.int64)
        grasp_idx = int(grasp_indices[-1])
        grasp_base_q = trajectory.base_traj[grasp_idx].copy()
        grasp_finger_q = trajectory.finger_traj[grasp_idx].copy()
        final_base_q = trajectory.base_traj[-1].copy()
        final_finger_q = trajectory.finger_traj[-1].copy()
        contact_points_local, contact_normals_local, contact_points_world, contact_normals_world, fingertip_targets_world = (
            self.build_world_contact_targets(grasp_base_q, grasp_finger_q)
        )

        initial_hand_q = np.concatenate([trajectory.initial_base_q, trajectory.initial_finger_q], axis=0)
        self.reset(self.initial_object_pos, self.initial_object_quat, initial_hand_q)
        self.cache_direct_visual_targets(contact_points_local, contact_normals_local, grasp_base_q)
        self._ensure_screenshot_recorder()
        self._ensure_video_recorder()
        self.maybe_launch_viewer()
        if self.screenshot_recorder is not None:
            self.screenshot_recorder.capture(self.data, label="traj_start")
        self.sync_viewer()

        print("\nScene")
        print(f"scene_xml: {self.scene_path}")
        print(f"visual_mesh_path: {self.mesh_path}")
        print(f"collision_mesh_path: {self.collision_mesh_path}")
        print(f"mesh_scale: {np.array2string(self.mesh_scale, precision=4)}")
        print(f"pedestal_pos_world: {np.array2string(self.pedestal_pos, precision=4)}")
        print(f"pedestal_size: {np.array2string(self.pedestal_size, precision=4)}")
        print(f"object_pos_world: {np.array2string(self.initial_object_pos, precision=4)}")
        print(f"object_quat_wxyz: {np.array2string(self.initial_object_quat, precision=4)}")
        if self.spider_reference_pose is not None:
            print(f"spider_object_minus_palm: {np.array2string(self.spider_reference_pose.object_minus_palm, precision=4)}")

        print("\nDirect MPC Grasp Reference")
        print(f"virtual_contact_points_local:\n{np.array2string(contact_points_local, precision=4)}")
        print(f"virtual_contact_normals_local:\n{np.array2string(contact_normals_local, precision=4)}")
        print(f"virtual_contact_points_world:\n{np.array2string(contact_points_world, precision=4)}")
        print(f"virtual_contact_normals_world:\n{np.array2string(contact_normals_world, precision=4)}")
        print(f"nominal_fingertip_targets_world:\n{np.array2string(fingertip_targets_world, precision=4)}")
        print(f"mpc_contact_weight: {self.mpc_param.contact_weight_:.6g}")
        print(f"mpc_grasp_closure_weight: {self.mpc_param.grasp_closure_weight_:.6g}")

        print("\nTrajectory")
        print(f"command_dt: {self.command_dt:.4f}s")
        print(f"total_command_steps: {trajectory.base_traj.shape[0]}")
        print(f"stages: {', '.join(dict.fromkeys(trajectory.stage_names))}")
        print(f"initial_base_q: {np.array2string(trajectory.initial_base_q, precision=4)}")
        print(f"grasp_base_q: {np.array2string(grasp_base_q, precision=4)}")
        print(f"final_lift_base_q: {np.array2string(final_base_q, precision=4)}")
        print(f"initial_finger_q: {np.array2string(trajectory.initial_finger_q, precision=4)}")
        print(f"grasp_finger_q: {np.array2string(grasp_finger_q, precision=4)}")
        print(f"mpc_state_dim: {self.mpc_param.n_qpos_}")
        print(f"mpc_action_dim: {self.mpc_param.n_cmd_}")

        best_contact_count = 0
        best_height_gain = 0.0
        last_status_text = "N/A"
        last_mjwp_rollout_info = None
        current_stage = None

        nominal_step_count = int(trajectory.base_traj.shape[0])
        pre_lift_base_q = np.asarray(grasp_base_q, dtype=np.float64).copy()
        pre_lift_finger_q = np.asarray(grasp_finger_q, dtype=np.float64).copy()
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
            if stage_name == "hover_open":
                grasp_activation = 0.0
            elif stage_name == "translate_to_contact":
                grasp_activation = float(self.args.approach_grasp_activation)
            else:
                grasp_activation = 1.0
            mpc_result = self.mpc.plan_once(
                target_object_pos=target_object_pos,
                target_object_quat=target_object_quat,
                target_base_pos=target_base_q[:BASE_TRANSLATION_DOF_COUNT],
                target_base_rot=base_orientation_to_matrix(target_base_q[BASE_TRANSLATION_DOF_COUNT:]),
                target_finger_q=target_finger_q,
                curr_x=curr_x,
                phi_vec=phi_vec,
                jac_mat=jac_mat,
                grasp_activation=grasp_activation,
                sol_guess=self.mpc_param.sol_guess_,
            )
            self.mpc_param.sol_guess_ = mpc_result["sol_guess"]
            last_status_text = str(mpc_result["solve_status"])

            freeze_fingers, ignore_base_release_gate = self.should_freeze_fingers(stage_name)
            last_mjwp_rollout_info = None
            if self.mjwp_rollout is not None:
                force_adjust_stages = {"squeeze", "await_lift_contact", "lift", "hold"}
                rollout_freeze_fingers = bool(freeze_fingers) and stage_name not in force_adjust_stages
                try:
                    mjwp_result = self.mjwp_rollout.plan_once(
                        data=self.data,
                        nominal_action=mpc_result["action"],
                        target_object_pos=target_object_pos,
                        target_object_quat=target_object_quat,
                        target_base_q=target_base_q,
                        target_finger_q=target_finger_q,
                        freeze_fingers=rollout_freeze_fingers,
                        stage_name=stage_name,
                    )
                except Exception as exc:
                    print(f"Warning: MJWP sampling rollout failed at step {step_idx + 1}; disabling it. Error: {exc}")
                    self.mjwp_rollout = None
                else:
                    mpc_result["action"] = mjwp_result.action
                    freeze_fingers = rollout_freeze_fingers
                    last_mjwp_rollout_info = mjwp_result
                    last_status_text = (
                        f"{last_status_text}+mjwp"
                        f"[c={mjwp_result.best_cost:.3g},fc={mjwp_result.force_cost:.3g}]"
                    )

            self.apply_mpc_action(
                action=mpc_result["action"],
                base_orientation=target_base_q[BASE_TRANSLATION_DOF_COUNT:],
                target_base_q=target_base_q,
                freeze_fingers=freeze_fingers,
                ignore_base_release_gate=ignore_base_release_gate,
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
                target_palm_pos_world, _ = self.get_target_palm_pose_world()
                if target_palm_pos_world is None:
                    palm_target_distance = np.nan
                else:
                    curr_palm_pos_world = np.asarray(self.data.body(self.palm_body_id).xpos, dtype=np.float64).copy()
                    palm_target_distance = float(np.linalg.norm(curr_palm_pos_world - target_palm_pos_world))
                obj_err = float(np.linalg.norm(object_pos - target_object_pos))
                extra_gate_text = ""
                if wait_for_lift_contact:
                    extra_gate_text = (
                        f" gate={pre_step_contact_count}/{lift_contact_fingers}"
                        f" stable={lift_stable_steps}/{lift_contact_stable_steps}"
                    )
                extra_mjwp_text = ""
                if last_mjwp_rollout_info is not None:
                    extra_mjwp_text = (
                        f" mjwp_cost={last_mjwp_rollout_info.best_cost:.3g}"
                        f" mjwp_fc={last_mjwp_rollout_info.force_cost:.3g}"
                        f" mjwp_cf={last_mjwp_rollout_info.contact_fingers:.1f}"
                        f" mjwp_ms={last_mjwp_rollout_info.elapsed_ms:.1f}"
                    )
                print(
                    f"  step={step_idx + 1:04d}/{nominal_step_count:04d} "
                    f"stage={stage_name:<13} "
                    f"obj_z={object_pos[2]:.4f} "
                    f"height_gain={height_gain:.4f} "
                    f"contact_fingers={contact_count} "
                    f"base_err={base_err:.4f} "
                    f"finger_err={finger_err:.4f} "
                    f"palm_target_dist={palm_target_distance:.4f} "
                    f"obj_err={obj_err:.4f} "
                    f"mpc={last_status_text}"
                    f"{extra_gate_text}"
                    f"{extra_mjwp_text}"
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
            "trajectory": trajectory,
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
        description="Floating-base Allegro tabletop trigrasp demo using CasADi contact dynamics, fingertip contact cost, and grasp-closure MPC."
    )
    parser.add_argument("--obj", type=str, default="rubber_duck", help="Object asset name in envs/assets/objects.")
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
    parser.add_argument("--obj_init_height", type=float, default=0.0, help="Extra initial object height relative to the pedestal top when the viewer starts.")
    parser.add_argument("--initial-object-lift", type=float, default=0.2, help="Extra height added to the pedestal/support under the object.")
    parser.add_argument(
        "--spider-reference-pose",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool_arg,
        help="Initialize the Allegro hand/object relative pose from Spider's processed Allegro grasping reference.",
    )
    parser.add_argument("--spider-reference-scene", type=Path, default=DEFAULT_SPIDER_REFERENCE_SCENE, help="Spider reference scene used to compute object-minus-palm offset.")
    parser.add_argument("--spider-reference-traj", type=Path, default=DEFAULT_SPIDER_REFERENCE_TRAJ, help="Spider reference trajectory whose first qpos supplies hand/object relative pose.")
    parser.add_argument(
        "--camera_free",
        type=parse_camera_free_arg,
        default=None,
        help="Fix the viewer camera using '[[cam_x, cam_y, cam_z], [look_x, look_y, look_z]]' or 6 flat values.",
    )
    parser.add_argument(
        "--show",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_arg,
        help="If true, disable object collision and gravity so the object is shown only visually.",
    )
    parser.add_argument("--initial-settle-steps", type=int, default=200, help="Maximum MuJoCo steps used to let the object settle on the pedestal before grasp planning.")
    parser.add_argument("--initial-settle-velocity-tol", type=float, default=5e-3, help="Object twist-norm threshold used to stop the initial settle rollout early.")
    parser.add_argument("--floor-z", type=float, default=0.0, help="Scene floor height.")
    parser.add_argument("--mujoco-dt", type=float, default=0.0025, help="MuJoCo timestep.")
    parser.add_argument("--mj-steps-per-command", type=int, default=10, help="Number of MuJoCo steps executed after each MPC solve.")
    parser.add_argument("--contact-stiffness", type=float, default=12.5, help="Soft contact stiffness used by the explicit CasADi contact dynamics.")
    parser.add_argument("--grasp-base-yaw", type=float, default=float(np.pi), help="Yaw used for the direct top-down Allegro grasp reference.")
    parser.add_argument("--grasp-pose-fraction", type=float, default=0.82, help="Fraction between lower and upper finger limits used as the nominal closed grasp pose.")
    parser.add_argument("--direct-grasp-z-bias", type=float, default=0.0, help="Vertical bias added to the direct grasp base reference.")
    parser.add_argument("--approach-grasp-activation", type=float, default=0.25, help="Contact/grasp-cost activation during translate_to_contact.")
    parser.add_argument("--approach-height", type=float, default=0.20, help="Hover height above the object used before the direct MPC grasp.")
    parser.add_argument("--hover-steps", type=int, default=20, help="Command steps spent hovering with the hand open.")
    parser.add_argument("--approach-steps", type=int, default=160, help="Command steps used to move the open hand to the direct grasp reference.")
    parser.add_argument("--squeeze-steps", type=int, default=70, help="Command steps used to close the fingers only after the hand base has reached the target pose.")
    parser.add_argument("--finger-ctrl-release-pos-tol", type=float, default=0.008, help="Finger ctrl stays frozen until the hand-base xyz error to the stage target drops below this threshold.")
    parser.add_argument(
        "--finger-ctrl-release-palm-tol",
        type=float,
        default=0.15,
        help="During translate_to_contact, release finger control once the palm is within this world-distance of the direct palm target pose.",
    )
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
    parser.add_argument("--mpc-contact-weight", type=float, default=650.0, help="MPC weight on fingertip-to-object contact cost.")
    parser.add_argument("--mpc-grasp-closure-weight", type=float, default=35.0, help="MPC weight on the fingertip grasp-closure surrogate.")
    parser.add_argument(
        "--mjwp-rollout",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_arg,
        help="Enable MuJoCo Warp parallel sampling rollout as an optional correction layer after the explicit MPC.",
    )
    parser.add_argument("--mjwp-device", type=str, default="cuda:0", help="Warp device used for batched rollouts, e.g. cuda:0 or cpu.")
    parser.add_argument("--mjwp-kernel-cache-dir", type=Path, default=Path("/tmp/warp-cache"), help="Writable Warp kernel cache directory.")
    parser.add_argument("--mjwp-num-samples", type=int, default=256, help="Number of parallel MuJoCo Warp rollout samples.")
    parser.add_argument("--mjwp-horizon", type=int, default=6, help="MuJoCo Warp sampling horizon in command steps.")
    parser.add_argument("--mjwp-iterations", type=int, default=3, help="Sampling optimization iterations per command step.")
    parser.add_argument("--mjwp-temperature", type=float, default=0.7, help="Softmax temperature for elite rollout weighting.")
    parser.add_argument("--mjwp-elite-fraction", type=float, default=0.10, help="Fraction of rollout samples used for weighted elite averaging.")
    parser.add_argument("--mjwp-noise-decay", type=float, default=0.65, help="Per-iteration decay applied to rollout action noise.")
    parser.add_argument("--mjwp-base-noise", type=float, default=0.003, help="Stddev of sampled xyz base increments.")
    parser.add_argument("--mjwp-finger-noise", type=float, default=0.035, help="Stddev of sampled finger joint increments.")
    parser.add_argument("--mjwp-action-blend", type=float, default=1.0, help="Blend from explicit-MPC action to MJWP sampled action.")
    parser.add_argument("--mjwp-nconmax-per-world", type=int, default=128, help="MuJoCo Warp contact allocation per sampled world.")
    parser.add_argument("--mjwp-njmax-per-world", type=int, default=512, help="MuJoCo Warp constraint allocation per sampled world.")
    parser.add_argument("--mjwp-capture-graph", nargs="?", const=True, default=True, type=parse_bool_arg, help="Capture a CUDA graph for the single-step MuJoCo Warp rollout.")
    parser.add_argument("--mjwp-target-normal-force", type=float, default=0.0, help="Desired normal force per fingertip. If <=0, derive it from object mass.")
    parser.add_argument("--mjwp-force-safety", type=float, default=2.0, help="Mass-based safety multiplier for derived fingertip normal-force targets.")
    parser.add_argument("--mjwp-force-cost-weight", type=float, default=12.0, help="Weight on the online force-closure surrogate during squeeze/lift stages.")
    parser.add_argument("--mjwp-tip-cost-weight", type=float, default=500.0, help="Weight on fingertip-to-moving-contact-target distance in MJWP rollout.")
    parser.add_argument("--mjwp-base-cost-weight", type=float, default=200.0, help="Weight on base xyz tracking inside MJWP rollout.")
    parser.add_argument("--mjwp-finger-cost-weight", type=float, default=2.0, help="Weight on finger q tracking inside MJWP rollout.")
    parser.add_argument("--mjwp-object-cost-weight", type=float, default=80.0, help="Weight on object position tracking inside MJWP rollout.")
    parser.add_argument("--mjwp-object-quat-cost-weight", type=float, default=5.0, help="Weight on object quaternion tracking inside MJWP rollout.")
    parser.add_argument("--mjwp-action-cost-weight", type=float, default=0.1, help="Weight on sampled action magnitude inside MJWP rollout.")
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
