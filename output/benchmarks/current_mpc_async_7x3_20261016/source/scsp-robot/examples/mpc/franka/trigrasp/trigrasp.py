from __future__ import annotations

import argparse
import ast
import copy
import itertools
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
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
from planning.mpc_explicit import MPCExplicit
from planning.lambda_contact_lm import LambdaContactBatchedLM
from planning.grasp_verification import verify_grasp
from models.contact_geometry import rotation_cs
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
DEFAULT_MESH_SCALE_MULTIPLIER = 1.2
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
# SPIDER Allegro pads face +X in the palm frame. This is a model axis,
# independent of the current finger posture or a fitted fingertip plane.
PALM_NORMAL_LOCAL = np.array([1.0, 0.0, 0.0], dtype=np.float64)

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
    max_self_penetration: float = 0.


@dataclass
class OnlineGraspRecovery:
    """Debounced loss/drift detection, using simulated rather than wall time."""
    origin: np.ndarray
    rotation: np.ndarray
    dt: float
    position_threshold: float = .025
    rotation_threshold: float = .3
    cooldown: float = 1.
    stall_timeout: float = 1.
    elapsed: float = 0.
    last_replan: float = -np.inf
    unconfirmed_time: float = 0.
    touched: bool = False
    pending_loss: bool = False
    recovering: bool = False
    opening: bool = False
    replan_count: int = 0
    opening_time: float = 0.
    opening_timeout: float = 3.
    repair_time: float = 0.

    def __post_init__(self):
        self.origin = np.asarray(self.origin, float).reshape(3).copy()
        self.rotation = np.asarray(self.rotation, float).reshape(3, 3).copy()
        for name in ('dt', 'position_threshold', 'rotation_threshold', 'stall_timeout', 'opening_timeout'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('Recovery '+name+' must be finite and positive')
        if not np.isfinite(self.cooldown) or self.cooldown < 0:
            raise ValueError('Recovery cooldown must be finite and nonnegative')

    def observe(self, pos, rot, *, contacts, was_loaded, loaded, closing, repair_available=False):
        self.elapsed += self.dt
        self.opening_time = self.opening_time+self.dt if self.opening else 0.
        self.touched |= contacts > 0
        self.pending_loss |= was_loaded and not loaded
        self.unconfirmed_time = self.unconfirmed_time+self.dt if (self.touched or closing) and not self.opening else 0.
        if loaded:
            self.pending_loss = self.recovering = self.opening = False
            self.unconfirmed_time = 0.
            self.opening_time = 0.
            self.repair_time = 0.
            return None
        drift = np.linalg.norm(pos-self.origin) >= self.position_threshold
        rotation_drift = np.linalg.norm(Rotation.from_matrix(rot@self.rotation.T).as_rotvec()) >= self.rotation_threshold
        reason = None
        if self.pending_loss:
            # Give a certified nearby replacement a short chance to make
            # physical contact before opening and discarding the held pads.
            self.repair_time = self.repair_time+self.dt if repair_available and contacts >= 2 else 0.
            if self.repair_time == 0. or self.repair_time >= .5:
                reason = "lost_grasp"
        elif drift or rotation_drift:
            reason = "object_moved"
        elif self.opening_time >= self.opening_timeout:
            reason = "blocked_pregrasp"
        elif self.unconfirmed_time >= self.stall_timeout:
            reason = "missed_contact" if contacts == 0 else "unconfirmed_grasp"
        if reason is None or self.elapsed-self.last_replan < self.cooldown:
            return None
        self.origin, self.rotation = np.asarray(pos).copy(), np.asarray(rot).copy()
        self.last_replan, self.unconfirmed_time = self.elapsed, 0.
        self.pending_loss, self.touched, self.recovering = False, False, True
        self.opening = True
        self.opening_time = 0.
        self.repair_time = 0.
        self.replan_count += 1
        return reason


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
    tip_center_sites=False,
    point_pads=False,
    tip_radius=.012,
    contact_cone="pyramidal",
    contact_impratio=1.,
    contact_noslip_iterations=0,
):
    robot_root = ET.parse(SPIDER_ALLEGRO_XML).getroot()
    robot_body = robot_root.find("./worldbody/body[@name='right_palm']")
    robot_asset = robot_root.find("asset")
    robot_default = robot_root.find("default")
    robot_actuator = robot_root.find("actuator")
    if robot_body is None or robot_asset is None or robot_default is None or robot_actuator is None:
        raise RuntimeError(f"Unexpected Allegro XML layout in {SPIDER_ALLEGRO_XML}")

    if tip_center_sites:
        # Track the spherical end of each collision capsule, rather than
        # the original SPIDER site on the pad surface. Lambda uses centres.
        for finger in FINGER_NAMES:
            site = robot_body.find(f".//site[@name='right_{finger}_tip']")
            height = .043 if finger == "thumb" else .029
            site.set("pos", format_vec([0., 0., height]))
            if point_pads:
                geom = robot_body.find(f".//geom[@name='{TIP_COLLISION_GEOM_NAMES[finger]}']")
                geom.set("type", "sphere")
                geom.set("pos", site.get("pos"))
                geom.set("size", format_vec([tip_radius]))

    root = ET.Element("mujoco", {"model": "allegro_tabletop_trigrasp"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": f"{float(mujoco_timestep):.8f}",
            "iterations": "40",
            "ls_iterations": "80",
            "cone": contact_cone,
            "impratio": str(float(contact_impratio)),
            "solver": "Newton",
            "noslip_iterations": str(int(contact_noslip_iterations)),
            "tolerance": "1e-8",
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
            "mass": "0",
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
        self.demo = demo
        self.tip_weight_ = float(demo.args.online_tip_weight)
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
        demo = self.demo
        tips, full_jacobian = demo.online_tip_kinematics()
        # Legacy staged dynamics keep the palm orientation outside MPC.
        joint_columns = np.r_[np.arange(3), np.arange(6, demo.model.nu)]
        tip_jacobian = full_jacobian[:, joint_columns]
        state = demo.get_mpc_state()
        anchors = demo.cached_target_points_local.copy()
        # Follow the pregrasp approach translation; at squeeze/lift these
        # offsets vanish and the objective is pad-to-object alignment.
        rotation = quat_wxyz_to_mat(target_object_quat)
        palm_target = np.asarray(target_object_pos) + rotation @ demo.cached_palm_target_local_pos
        anchors += (np.asarray(target_base_pos)-palm_target) @ rotation
        pad_gap = min(float(mujoco.mj_geomDistance(demo.model, demo.data,
            mujoco.mj_name2id(demo.model, mujoco.mjtObj.mjOBJ_GEOM, TIP_COLLISION_GEOM_NAMES[name]),
            demo.object_geom_id, 1., None)) for name in demo.active_finger_names)
        attraction = float(np.clip((pad_gap-.002)/.018, 0., 1.)**2)
        return np.concatenate(
            [
                np.asarray(target_object_pos, dtype=np.float64).reshape(3),
                np.asarray(target_object_quat, dtype=np.float64).reshape(4),
                np.asarray(target_base_pos, dtype=np.float64).reshape(BASE_TRANSLATION_DOF_COUNT),
                np.asarray(target_finger_q, dtype=np.float64).reshape(self.n_robot_qpos_ - BASE_TRANSLATION_DOF_COUNT),
                state[7:], tips.reshape(-1), tip_jacobian.reshape(-1, order="F"),
                anchors.reshape(-1), np.array([attraction]),
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
        hand_reference = cs.SX.sym("hand_reference", self.n_robot_qpos_)
        tips = cs.SX.sym("measured_tips", 12)
        tip_jacobian = cs.SX.sym("tip_jacobian", 12, self.n_robot_qpos_)
        anchors = cs.SX.sym("pad_targets_local", 3, 4)
        attraction = cs.SX.sym("ik_attraction")
        cost_param = cs.vvcat([target_object_pos, target_object_quat, target_base_pos, target_finger_q,
            hand_reference, tips, cs.vec(tip_jacobian), cs.vec(anchors), attraction])
        predicted_tips = tips + tip_jacobian @ (x[7:]-hand_reference)
        target_tips = cs.repmat(obj_pose[:3], 1, 4) + rotation_cs(obj_pose[3:7]) @ anchors
        tip_cost = cs.sumsqr(predicted_tips-cs.vec(target_tips))

        position_cost = cs.sumsqr(obj_pose[:3] - target_object_pos)
        quaternion_cost = 1.0 - cs.dot(obj_pose[3:7], target_object_quat) ** 2
        base_track_cost = cs.sumsqr(base_pos - target_base_pos)
        finger_track_cost = cs.sumsqr(finger_q - target_finger_q)
        control_cost = cs.sumsqr(u)

        path_cost = (
            self.tip_weight_ * tip_cost
            + attraction * (self.base_track_weight_ * base_track_cost
                + self.finger_track_weight_ * finger_track_cost)
            + self.object_pos_weight_ * position_cost
            + self.object_quat_weight_ * quaternion_cost
            + self.u_weight_ * control_cost
        )
        final_cost = self.terminal_weight_ * (
            self.tip_weight_ * tip_cost
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


class AllegroOnlineMPCParams:
    """Live pad alignment, contact force and object-pose MPC."""

    def __init__(self, demo):
        args = demo.args
        self.h_ = demo.command_dt
        self.n_robot_qpos_ = self.n_cmd_ = demo.model.nu
        self.pose_size = 7 + self.n_cmd_
        self.n_qvel_ = 6 + self.n_cmd_
        # Carry the position-servo target as well as measured velocity.
        # Resetting it to q_measured every tick discarded grasp preload.
        self.n_qpos_ = self.pose_size + self.n_qvel_ + self.n_cmd_
        self.num_contacts = len(demo.active_finger_names)
        self.max_ncon_ = int(args.mpc_max_contacts)
        self.mu_object_ = float(args.object_friction)
        self.obj_mass_ = float(args.obj_mass)
        self.gravity_ = np.array([0., 0., -9.81, 0., 0., 0.])
        physical_mass = np.empty((demo.model.nv, demo.model.nv))
        try:
            mujoco.mj_fullM(demo.model, demo.data, physical_mass)
        except TypeError:
            mujoco.mj_fullM(demo.model, physical_mass, demo.data.qM)
        object_dofs = np.arange(demo.object_dof_adr, demo.object_dof_adr+6)
        self.obj_inertia_ = physical_mass[np.ix_(object_dofs, object_dofs)] / self.h_**2
        self.gravity_ = -np.asarray(demo.data.qfrc_bias[object_dofs], float) / self.obj_mass_
        self.robot_stiff_ = np.diag(np.asarray(demo.model.actuator_gainprm[:, 0], float))
        self.robot_damping_ = -np.asarray(demo.model.actuator_biasprm[:, 2], float)
        self.Q = np.zeros((self.n_qvel_, self.n_qvel_))
        self.Q[:6, :6], self.Q[6:, 6:] = self.obj_inertia_, self.robot_stiff_
        # Commands are servo displacements, as in the shared joint MPC.
        self.contact_dynamics = "quasi-dynamic"
        self.grouped_contact_response_ = True
        self.runtime_object_cost_ = True
        self.mpc_carry_velocity_ = False
        self.mpc_compact_dynamics_ = True
        self.mpc_compact_cost_ = True
        self.mpc_prune_codegen_ = True
        self.model_params = float(args.contact_stiffness)
        self.mpc_model = "explicit"
        self.planner_solver_ = args.mpc_solver
        # Online control already has a bounded measured-tip fallback. Do
        # not freeze execution to construct/solve an IPOPT NLP on RTI failure.
        self.mpc_ipopt_fallback_ = False
        self.mpc_horizon_ = int(args.mpc_horizon)
        self.ipopt_max_iter_ = int(args.mpc_ipopt_max_iter)
        self.mpc_acados_profile_ = dict(nlp_solver_type="SQP_RTI", qp_solver="FULL_CONDENSING_HPIPM",
            nlp_solver_max_iter=1, levenberg_marquardt=.01)
        self.mpc_u_ub_ = np.r_[np.full(3, args.mpc_base_step_limit),
            np.full(3, args.mpc_rotation_step_limit),
            np.full(self.n_cmd_ - 6, args.mpc_velocity_limit * self.h_)]
        self.mpc_u_lb_ = -self.mpc_u_ub_
        self.contact_u_limit = np.minimum(self.mpc_u_ub_, self.h_*np.r_[
            np.full(3, args.online_contact_base_speed),
            np.full(3, args.online_contact_rotation_speed),
            np.full(self.n_cmd_-6, args.online_contact_joint_speed)])
        self.mpc_q_lb_ = np.r_[demo.initial_object_pos[:2] - 1., args.floor_z - .05,
            np.full(4, -1e7), demo.hand_joint_ranges[:, 0]]
        self.mpc_q_ub_ = np.r_[demo.initial_object_pos[:2] + 1.,
            demo.initial_object_pos[2] + args.lift_height + .4, np.full(4, 1e7),
            demo.hand_joint_ranges[:, 1]]
        self.mpc_q_lb_ = np.r_[self.mpc_q_lb_, np.full(self.n_qvel_, -100.)]
        self.mpc_q_ub_ = np.r_[self.mpc_q_ub_, np.full(self.n_qvel_, 100.)]
        self.mpc_q_lb_ = np.r_[self.mpc_q_lb_, demo.hand_joint_ranges[:, 0]]
        self.mpc_q_ub_ = np.r_[self.mpc_q_ub_, demo.hand_joint_ranges[:, 1]]
        import hashlib
        self.cost_tag = hashlib.sha1(np.r_[self.mpc_u_lb_, self.mpc_u_ub_, self.mpc_q_lb_, self.mpc_q_ub_].tobytes()).hexdigest()[:8]
        self.tip_weight = float(args.online_tip_weight)
        self.contact_transport_weight = float(args.online_contact_transport_weight)
        self.contact_transport_distance = float(args.online_contact_transport_distance
            if args.online_contact_transport_distance is not None
            else args.online_lift_speed*self.h_*self.mpc_horizon_)
        self.lift_speed = float(args.online_lift_speed)
        self.self_collision_weight = float(args.online_self_collision_weight)
        self.self_collision_clearance = float(args.online_self_collision_clearance)
        self.base_position_weight = float(args.online_base_position_weight)
        self.posture_weight = float(args.online_posture_weight)
        self.posture_weights = np.full(self.n_cmd_, self.posture_weight)
        self.posture_weights[:3] = self.base_position_weight
        self.posture_weights[3:6] = float(args.online_palm_orientation_weight)
        self.object_weight = float(args.mpc_object_pos_weight)
        self.quat_weight = 10 * float(args.mpc_object_quat_weight)
        self.lateral_weight = float(args.online_object_lateral_weight)
        self.horizontal_recovery_weight = float(args.online_horizontal_recovery_weight)
        self.horizontal_tolerance = float(args.online_max_horizontal_error)
        self.force_tracking_weight = float(args.online_force_tracking_weight)
        self.force_cost_as_wrench = True
        self.linearize_force_cost = True
        self.force_feedback = bool(args.online_force_feedback)
        self.pressure_control = bool(args.online_pressure_control and args.online_execution == 'position')
        self.normal_admittance = float(args.online_normal_admittance)
        self.minimum_normal_force = float(args.online_min_normal_force)
        self.maximum_normal_force = float(args.lambda_force_limit)/np.sqrt(
            1+(args.object_friction*args.online_friction_margin)**2)
        # Compare forces in load units. Dividing by pad stiffness converted
        # Newtons to metres and effectively removed force tracking from MPC.
        self.force_residual_scale = 1. / max(self.obj_mass_*9.81, 1e-6)
        # Shared SCM damping/regularization; the response time is derived
        # from this scene's MuJoCo solref unless explicitly overridden.
        self.soft_scm_response_time = float(args.soft_scm_response_time)
        self.soft_scm_damping_ratio = float(args.soft_scm_damping_ratio)
        self.soft_scm_regularization = float(args.soft_scm_regularization)
        self.pressure_sync_weight = float(args.online_pressure_sync_weight)
        self.terminal_weight = float(args.mpc_terminal_weight)
        self.action_weight = float(args.mpc_u_weight)
        self.action_weights = np.full(self.n_cmd_, self.action_weight)
        self.action_weights[3:6] = args.online_rotation_action_weight
        self.action_weights[6:] = args.online_joint_action_weight
        self.sol_guess_ = None

    def init_cost_fns(self):
        from planning.allegro_mpc_costs import build_allegro_contact_costs
        return build_allegro_contact_costs(self)


class AllegroOnlineContactModel(ExplicitModel):
    """Transport measured contact gaps through the joint-space horizon."""

    def __init__(self, param):
        from models.soft_scm_contact import grouped_contact_velocity_cs, soft_scm_coefficients
        self.param_ = param
        self.contact_model = "soft-scm-allegro"
        self.transports_contacts = True
        self.pose_size = 7 + param.n_cmd_
        self.physical_parameter_dim = 1 + self.pose_size + 36 + param.n_cmd_ + 6
        self.command_parameter_dim = self.physical_parameter_dim + param.n_cmd_**2 + 2*param.n_cmd_
        n = param.num_contacts
        self.model_parameter_dim = self.command_parameter_dim + 115+40*n+3*n*param.n_cmd_
        self.command_projection = np.eye(param.n_cmd_)
        self.pressure_bias = np.zeros(param.n_cmd_)
        self.command_limits = np.asarray(getattr(param, 'mpc_u_ub_', np.full(param.n_cmd_, 100.))).copy()
        self.reference = np.zeros(self.pose_size)
        self.reference[3] = 1.
        self.object_inverse = np.linalg.inv(param.obj_inertia_ * param.h_**2)
        self.object_mass = param.obj_inertia_ * param.h_**2
        self.grasp_velocity_map = np.zeros((6, 3*n))
        self.grasp_robot_jac = np.zeros((3*n, param.n_cmd_))
        self.grasp_inverse = np.eye(6)
        self.grasp_mobility = np.zeros((6, 6))
        self.grasp_internal = np.zeros((6, 3*n))
        self.grasp_reference_force = np.zeros(3*n)
        self.grasp_normal_limits = np.zeros(n)
        self.grasp_other_wrench = np.zeros(6)
        self.grasp_active = False
        self.external_wrench = param.obj_mass_ * param.gravity_
        h = param.h_
        # Lambda supplies ideal point forces; SCM supplies contact impulses.
        # Expose the former through the same object operator so its pose
        # convention, time step, mass and wrench mapping can be checked.
        pose = cs.SX.sym("object_pose", 7)
        object_velocity = cs.SX.sym("object_velocity", 6)
        local_points = cs.SX.sym("force_points", 3, param.num_contacts)
        local_forces = cs.SX.sym("point_forces", 3, param.num_contacts)
        object_inv = cs.SX.sym("object_inverse", 6, 6)
        external_mixed = cs.SX.sym("external_mixed", 6)
        torque = cs.SX.zeros(3)
        for i in range(param.num_contacts):
            torque += cs.cross(local_points[:, i], local_forces[:, i])
        wrench = cs.vertcat(rotation_cs(pose[3:7]) @ cs.sum2(local_forces), torque)
        predicted_velocity = object_velocity + h*object_inv@(external_mixed+wrench)
        q_object = pose[3:7]
        omega_object = predicted_velocity[3:6]
        predicted_quat = q_object + .5*h*cs.vertcat(-cs.dot(q_object[1:], omega_object),
            q_object[0]*omega_object+cs.cross(q_object[1:], omega_object))
        predicted_quat /= cs.sqrt(cs.sumsqr(predicted_quat)+1e-20)
        self.applied_force_prediction_fn = cs.Function("allegro_ideal_point_force_step",
            [pose, object_velocity, local_points, local_forces, object_inv, external_mixed],
            [cs.vertcat(pose[:3]+h*predicted_velocity[:3], predicted_quat), predicted_velocity])
        self.robot_response = h * param.robot_damping_ + h*h*np.diag(param.robot_stiff_)
        self.robot_inverse = 1. / (np.ones(param.n_cmd_) + self.robot_response)
        x, u = cs.SX.sym("hand_state", param.n_qpos_), cs.SX.sym("hand_increment", param.n_cmd_)
        phi = cs.SX.sym("hand_gaps", 4 * param.max_ncon_)
        jac = cs.SX.sym("hand_contacts", 4 * param.max_ncon_, param.n_qvel_)
        mp = cs.SX.sym("hand_reference", self.model_parameter_dim)
        offset = 1 + self.pose_size
        reference = mp[1:offset]
        object_inverse = cs.reshape(mp[offset:offset+36], 6, 6)
        hand_inverse = mp[offset+36:offset+36+param.n_cmd_]
        external = mp[offset+36+param.n_cmd_:offset+36+param.n_cmd_+6]
        inverse = cs.diagcat(object_inverse, cs.diag(hand_inverse))
        q, qr = x[3:7], reference[3:7]
        angular = 2 * (qr[0]*q[1:] - q[0]*qr[1:] - cs.cross(qr[1:], q[1:]))
        displacement = cs.vertcat(x[:3]-reference[:3], angular,
            x[7:self.pose_size]-reference[7:])
        v = x[self.pose_size:self.pose_size+param.n_qvel_]
        cursor = self.physical_parameter_dim
        projection = cs.reshape(mp[cursor:cursor+param.n_cmd_**2], param.n_cmd_, param.n_cmd_)
        cursor += param.n_cmd_**2
        bias, limits = mp[cursor:cursor+param.n_cmd_], mp[cursor+param.n_cmd_:cursor+2*param.n_cmd_]
        priority = cs.fmin(limits, cs.fmax(-limits, projection[:, :6]@u[:6]+bias))
        tangent_command = projection[:, 6:]@u[6:]
        available = cs.if_else(tangent_command >= 0, limits-priority, limits+priority)
        ratios = cs.if_else(cs.fabs(tangent_command) > 1e-10,
            available/cs.sqrt(tangent_command*tangent_command+1e-20), 1.)
        scale = cs.fmin(1., cs.mmin(ratios))
        command = priority+scale*tangent_command
        next_servo_target = x[-param.n_cmd_:] + command
        if hasattr(param, 'mpc_q_lb_'):
            next_servo_target = cs.fmin(cs.DM(param.mpc_q_ub_[-param.n_cmd_:]),
                cs.fmax(cs.DM(param.mpc_q_lb_[-param.n_cmd_:]), next_servo_target))
        # Backward Euler for the same position spring/damper executed in
        # MuJoCo. Carry hand momentum; instantaneous u/h was unsafe on the
        # SPIDER joints, whose 1 kg*m^2 armature slows fingertip closing.
        servo_error = next_servo_target-x[7:self.pose_size]
        hand_free = v[6:] + hand_inverse * (h * cs.DM(param.robot_stiff_) @ servo_error - cs.DM(self.robot_response) * v[6:])
        free = cs.vertcat(v[:6] + h * object_inverse @ external, hand_free)
        from models.soft_scm_contact import (
            SOFT_SCM_RESPONSE_TIME, SOFT_SCM_DAMPING_RATIO, SOFT_SCM_REGULARIZATION)
        self.soft_scm_response_time = float(getattr(param, 'soft_scm_response_time', SOFT_SCM_RESPONSE_TIME))
        self.soft_scm_damping_ratio = float(getattr(param, 'soft_scm_damping_ratio', SOFT_SCM_DAMPING_RATIO))
        self.soft_scm_regularization = float(getattr(param, 'soft_scm_regularization', SOFT_SCM_REGULARIZATION))
        a, b = soft_scm_coefficients(h, self.soft_scm_response_time,
            self.soft_scm_damping_ratio, self.soft_scm_regularization)
        # A friction-pyramid row includes tangential velocity. Only its
        # shared normal row changes the separation distance; sliding along
        # a pad must not open two of the four friction constraints.
        transported_gaps = phi + cs.vertcat(*[
            cs.repmat((cs.sum1(jac[start:start+4, :])/4) @ displacement, 4, 1)
            for start in range(0, 4*param.max_ncon_, 4)])
        self.contact_gap_fn = cs.Function("allegro_contact_gaps", [x, phi, jac, mp], [transported_gaps])
        velocity, impulses = grouped_contact_velocity_cs(free, jac, transported_gaps,
            inverse, h, a, b, self.soft_scm_regularization, return_impulses=True)
        forces = []
        for i in range(param.num_contacts):
            block = jac[4*i:4*i+4,:3]
            normal = cs.sum1(block).T/4
            mu = cs.sqrt(cs.sumsqr((block[0,:]-block[2,:])/2)+1e-20)
            tangents = cs.horzcat((block[0,:]-block[2,:]).T, (block[1,:]-block[3,:]).T)/(2*cs.fmax(mu,1e-10))
            fn, ft = impulses[i]
            forces.append((normal*fn + tangents@ft)/h)
        raw_forces = cs.vertcat(*forces)
        cursor = self.command_parameter_dim
        def matrix(rows, columns):
            nonlocal cursor
            result = cs.reshape(mp[cursor:cursor+rows*columns], rows, columns)
            cursor += rows*columns
            return result
        velocity_map, robot_jac = matrix(6, 3*n), matrix(3*n, param.n_cmd_)
        object_mass, grasp_inverse, grasp_mobility = matrix(6, 6), matrix(6, 6), matrix(6, 6)
        internal_map = matrix(6, 3*n)
        reference_force, normal_limits, other_wrench = matrix(3*n, 1), matrix(n, 1), matrix(6, 1)
        grasp_active = mp[cursor]
        from models.allegro_grasp_response import force_limited_grasp_cs
        current_ref_velocity = cs.vertcat(v[:3], rotation_cs(qr).T@rotation_cs(q)@v[3:6])
        supported_velocity, supported_forces = force_limited_grasp_cs(velocity, raw_forces,
            current_ref_velocity, hand_inverse, object_inverse, external, jac, transported_gaps,
            h=h, friction=float(getattr(param, 'mu_object_', 0.)), velocity_map=velocity_map,
            robot_jac=robot_jac, object_mass=object_mass, grasp_inverse=grasp_inverse,
            grasp_mobility=grasp_mobility, internal_map=internal_map,
            reference_force=reference_force, normal_limits=normal_limits, other_wrench=other_wrench)
        supported_velocity = cs.vertcat(supported_velocity[:3],
            rotation_cs(q).T@rotation_cs(qr)@supported_velocity[3:6], supported_velocity[6:])
        velocity = cs.if_else(grasp_active > .5, supported_velocity, velocity)
        pad_force = cs.if_else(grasp_active > .5, supported_forces, raw_forces)
        self.contact_force_fn = cs.Function("allegro_pad_forces", [x,u,phi,jac,mp], [pad_force], {"cse":True})
        if getattr(param, 'linearize_force_cost', False):
            seed = cs.SX.sym('force_seed', pad_force.numel())
            vjp = cs.Function('allegro_force_vjp', [x,u,phi,jac,mp,seed],
                [cs.jtimes(pad_force, cs.vertcat(x,u), seed, True)], {'cse':True})
            mx = [cs.MX.sym(name, *value.shape) for name, value in
                  zip(('fx','fu','fphi','fj','fmp'), (x,u,phi,jac,mp))]
            basis = cs.DM.eye(pad_force.numel())
            derivative = cs.vertcat(*[vjp(*mx, basis[:, i]).T for i in range(pad_force.numel())])
            self.force_linearization_fn = cs.Function('allegro_force_linearization', mx,
                [self.contact_force_fn(*mx), derivative[:, :param.n_qpos_], derivative[:, param.n_qpos_:]])
        omega = velocity[3:6]
        next_q = q + .5*h*cs.vertcat(-cs.dot(q[1:], omega), q[0]*omega + cs.cross(q[1:], omega))
        next_q /= cs.sqrt(cs.sumsqr(next_q) + 1e-20)
        next_pose = cs.vertcat(x[:3]+h*velocity[:3], next_q,
            x[7:self.pose_size]+h*velocity[6:])
        next_velocity = cs.vertcat(velocity[:3], rotation_cs(next_q).T @ rotation_cs(q) @ omega, velocity[6:])
        self.prediction_fn = cs.Function("allegro_contact_step", [x, u, phi, jac, mp],
            [cs.vertcat(next_pose, next_velocity, next_servo_target)], {"cse": True})

    def runtime_parameters(self, sigma):
        return np.r_[float(sigma), self.reference, self.object_inverse.reshape(-1, order="F"),
            self.robot_inverse, self.external_wrench, self.command_projection.reshape(-1, order='F'),
            self.pressure_bias, self.command_limits,
            self.grasp_velocity_map.reshape(-1, order='F'), self.grasp_robot_jac.reshape(-1, order='F'),
            self.object_mass.reshape(-1, order='F'), self.grasp_inverse.reshape(-1, order='F'),
            self.grasp_mobility.reshape(-1, order='F'), self.grasp_internal.reshape(-1, order='F'),
            self.grasp_reference_force, self.grasp_normal_limits, self.grasp_other_wrench,
            float(self.grasp_active)]

    def effective_command(self, action):
        action = np.asarray(action)
        priority = np.clip(self.command_projection[:, :6]@action[:6]+self.pressure_bias,
            -self.command_limits, self.command_limits)
        tangent = self.command_projection[:, 6:]@action[6:]
        available = np.where(tangent >= 0, self.command_limits-priority, self.command_limits+priority)
        ratios = np.where(np.abs(tangent) > 1e-10, available/np.sqrt(tangent*tangent+1e-20), 1.)
        scale = min(1., float(np.min(ratios)))
        return priority+scale*tangent


class AllegroOnlineMPC(MPCExplicit):
    def __init__(self, param):
        physics = AllegroOnlineContactModel(param)
        param.online_model = physics
        param.init_cost_fns()
        self.residual_fn = param.residual_fn
        self.terminal_residual_fn = param.terminal_residual_fn
        super().__init__(param, cost_kind="param", model=physics)
        self.stage_cost_on_next = False
        self.pressure_floor = None
        self.pressure_reference = None

    def plan_contacts(self, state, phi, jac, *, tips, tip_jacobian, anchors,
                      goal, hand_posture, loaded, approach_origin=None,
                      object_inertia=None, external_wrench=None, hand_mass_diagonal=None,
                      desired_force_local=None, measured_force_world=None, pressure_control_allowed=True,
                      other_hand_wrench_world=None, grasp_established=None, sol_guess=None):
        state = np.asarray(state, float).reshape(self.param_.n_qpos_)
        self.model.reference = state[:self.model.pose_size].copy()
        if hand_mass_diagonal is not None:
            self.model.robot_inverse = 1. / (np.asarray(hand_mass_diagonal, float).reshape(self.param_.n_cmd_) + self.model.robot_response)
        if object_inertia is not None:
            self.model.object_mass = np.asarray(object_inertia, float).reshape(6, 6).copy()
            self.model.object_inverse = np.linalg.inv(self.model.object_mass)
        if external_wrench is not None:
            self.model.external_wrench = np.asarray(external_wrench, float).reshape(6).copy()
            self.model.external_wrench[3:] = quat_wxyz_to_mat(state[3:7]).T @ self.model.external_wrench[3:]
        origin = state[:3] if approach_origin is None else np.asarray(approach_origin, float).reshape(3)
        tolerance = self.param_.horizontal_tolerance
        if loaded:
            lower = np.minimum(state[:2], np.asarray(goal)[:2]) - tolerance
            upper = np.maximum(state[:2], np.asarray(goal)[:2]) + tolerance
        else:
            # Keep the ungrasped object anchored to its initial corridor.
            # Expanding it with observed drift reset the penalty each cycle
            # and let an early thumb contact push the index patch away.
            lower, upper = origin[:2] - tolerance, origin[:2] + tolerance
        self._horizontal_bounds = np.stack((lower, upper))
        tips = np.asarray(tips, float).reshape(self.param_.num_contacts, 3)
        tip_jacobian = np.asarray(tip_jacobian, float).reshape(3*self.param_.num_contacts, self.param_.n_cmd_)
        anchors = np.asarray(anchors, float).reshape(self.param_.num_contacts, 3)
        gaps = np.asarray(phi).reshape(-1, 4)[:self.param_.num_contacts].mean(axis=1)
        near_contact = np.min(gaps) <= .02
        self.model.command_limits = (self.param_.contact_u_limit if near_contact else self.param_.mpc_u_ub_).copy()
        self.model.command_projection = np.eye(self.param_.n_cmd_)
        self.model.pressure_bias = np.zeros(self.param_.n_cmd_)
        self.pressure_reference = None
        self.model.grasp_active = bool((loaded if grasp_established is None else grasp_established)
            and pressure_control_allowed and measured_force_world is not None)
        if (self.param_.pressure_control and pressure_control_allowed and measured_force_world is not None
                and (loaded or (np.count_nonzero(gaps <= 0) >= 2
                    and np.max(gaps) <= .02))):
            from planning.grasp_force_allocation import normal_pressure_command
            normals = np.asarray(jac).reshape(-1, 4, self.param_.n_qvel_)[:self.param_.num_contacts, :, :3].mean(axis=1)
            measured = np.einsum('ij,ij->i', normals, np.asarray(measured_force_world))
            if self.pressure_floor is None:
                self.pressure_floor = np.full(self.param_.num_contacts, self.param_.minimum_normal_force)
            desired = self.pressure_floor.copy()
            if desired_force_local is not None:
                desired = np.maximum(desired, np.einsum('ij,ij->i', normals,
                    np.asarray(desired_force_local)@quat_wxyz_to_mat(state[3:7]).T))
            desired = np.clip(desired, self.param_.minimum_normal_force, self.param_.maximum_normal_force)
            self.pressure_reference = desired.copy()
            self.model.command_projection, self.model.pressure_bias = normal_pressure_command(
                tip_jacobian, normals, measured, desired, self.param_.normal_admittance,
                float(self.param_.contact_u_limit[:3].min()), grasp_established=self.model.grasp_active)
        elif not pressure_control_allowed:
            self.pressure_floor = None
            self.pressure_reference = None
        if self.model.grasp_active:
            from models.allegro_grasp_response import grasp_operators
            grasp_jac = np.asarray(jac).copy()
            normal_rows = grasp_jac.reshape(-1, 4, self.param_.n_qvel_)[:self.param_.num_contacts, :, :3].mean(axis=1)
            active = (gaps <= 0) & (np.einsum('ij,ij->i', normal_rows, np.asarray(measured_force_world)) > .01)
            for i in np.flatnonzero(~active):
                grasp_jac[4*i:4*i+4] = 0.
            (self.model.grasp_velocity_map, self.model.grasp_robot_jac,
             self.model.grasp_inverse, self.model.grasp_mobility,
             self.model.grasp_internal) = grasp_operators(grasp_jac, self.model.robot_inverse,
                 self.model.object_mass, self.param_.num_contacts)
            self.model.grasp_reference_force = np.asarray(measured_force_world).reshape(-1).copy()
            normals = np.asarray(jac).reshape(-1, 4, self.param_.n_qvel_)[:self.param_.num_contacts, :, :3].mean(axis=1)
            self.model.grasp_normal_limits = np.maximum(np.einsum('ij,ij->i', normals,
                np.asarray(measured_force_world)), 0.)
            self.model.grasp_reference_force.reshape(-1, 3)[~active] = 0.
            other = np.zeros(6) if other_hand_wrench_world is None else np.asarray(other_hand_wrench_world)
            self.model.grasp_other_wrench = np.r_[other[:3], quat_wxyz_to_mat(state[3:7]).T@other[3:]]
        # Fade the IK attractor over the last 2 cm and remove it before
        # any pad touches, freeing all coupled DOFs to establish contact.
        gap = float(np.min(np.asarray(phi)[:4*self.param_.num_contacts]))
        attraction = float(np.clip((gap-.002)/.018, 0., 1.)**2)
        cp = np.r_[goal, hand_posture, float(loaded), lower, upper,
            state[7:7+self.param_.n_cmd_], tips.reshape(-1),
            tip_jacobian.reshape(-1, order="F"), anchors.reshape(-1), attraction]
        if self.param_.cost_uses_contact_geometry:
            cp = np.r_[cp, np.zeros(3*self.param_.num_contacts) if desired_force_local is None
                else np.asarray(desired_force_local).reshape(-1), phi,
                np.asarray(jac).reshape(-1, order="F"), self.model.runtime_parameters(self.param_.model_params)]
        if self.param_.cost_linearizes_force:
            force_action = np.zeros(self.param_.n_cmd_)
            if sol_guess is not None:
                controls = np.asarray(sol_guess['x0']).reshape(self.param_.mpc_horizon_, -1)[:, :self.param_.n_cmd_]
                force_action = controls[min(1, len(controls)-1)].copy()
                force_action = np.clip(force_action, -self.param_.contact_u_limit, self.param_.contact_u_limit)
            force_value, force_jx, force_ju = self.model.force_linearization_fn(state, force_action, phi, jac,
                self.model.runtime_parameters(self.param_.model_params))
            feedback_data = None
            if self.param_.cost_force_feedback_offset is not None:
                raw_baseline = np.asarray(self.model.contact_force_fn(state, np.zeros(self.param_.n_cmd_),
                    phi, jac, self.model.runtime_parameters(self.param_.model_params))).reshape(-1, 3)
                measured_force = raw_baseline if measured_force_world is None else np.asarray(measured_force_world).reshape(-1, 3)
                normals = np.asarray(jac).reshape(-1, 4, self.param_.n_qvel_)[:self.param_.num_contacts, :, :3].mean(axis=1)
                raw_normal = np.sum(raw_baseline*normals, axis=1)
                actual_normal = np.sum(measured_force*normals, axis=1)
                gain = np.ones(self.param_.num_contacts) if measured_force_world is None else np.clip(
                    actual_normal/np.maximum(raw_normal, .01), .2, 50.)
                row_gain = np.repeat(gain, 3)
                force_value = measured_force.reshape(-1)+row_gain*(np.asarray(force_value).reshape(-1)-raw_baseline.reshape(-1))
                force_jx = row_gain[:, None]*np.asarray(force_jx)
                force_ju = row_gain[:, None]*np.asarray(force_ju)
                feedback_data = np.r_[gain, raw_baseline.reshape(-1), measured_force.reshape(-1)]
            cp = np.r_[cp, state, force_action, np.asarray(force_value).reshape(-1),
                np.asarray(force_jx).reshape(-1, order='F'), np.asarray(force_ju).reshape(-1, order='F')]
            if feedback_data is not None:
                cp = np.r_[cp, feedback_data]
        n, nu = self.param_.mpc_horizon_, self.param_.n_cmd_
        near_contact = np.min(np.asarray(phi)[:4*self.param_.num_contacts]) <= .02
        u_ub = self.param_.contact_u_limit if near_contact else self.param_.mpc_u_ub_
        u_lb = -u_ub
        mp = self.model.runtime_parameters(self.param_.model_params)

        def rollout_score(commands):
            from planning.allegro_mpc_costs import contact_cost_parameters_at_state
            current, trajectory, cost = state.copy(), [], 0.
            for command in commands:
                exact_cp = contact_cost_parameters_at_state(cp, current, command,
                    self.model, phi, jac, mp, self.param_.cost_force_linearization_offset,
                    self.param_.cost_force_feedback_offset)
                cost += float(self.path_cost_fn(current, command, exact_cp))
                current = np.asarray(self.model.prediction_fn(current, command, phi, jac, mp)).reshape(-1)
                trajectory.append(current.copy())
                if not np.isfinite(current).all():
                    return np.inf, trajectory
            cost += float(self.final_cost_fn(current, cp))
            return cost, trajectory

        # Use IK for distant approach initialization only. Near contact,
        # seed and rescue the solve from the measured pad residual, so a
        # changed contact target can move joints away from the startup IK.
        alignment_controls, current = [], state.copy()
        for _ in range(n):
            predicted_tips = tips + (tip_jacobian @ (current[7:7+nu]-state[7:7+nu])).reshape(-1, 3)
            target_tips = current[:3] + anchors @ quat_wxyz_to_mat(current[3:7]).T
            error = (target_tips-predicted_tips).reshape(-1)
            delta_q = tip_jacobian.T @ np.linalg.solve(
                tip_jacobian @ tip_jacobian.T + 1e-5*np.eye(error.size), error)
            gaps = np.asarray(self.model.contact_gap_fn(current, phi, jac, mp)).reshape(-1)
            gate = 0. if loaded else min(attraction,
                float(np.clip((np.min(gaps[:4*self.param_.num_contacts])-.002)/.018, 0., 1.)**2))
            target_q = current[7:7+nu] + delta_q
            target_q = gate*np.asarray(hand_posture) + (1-gate)*target_q
            command = np.clip(.25*(target_q-current[-nu:]), u_lb, u_ub)
            command = np.clip(command, self.param_.mpc_q_lb_[-nu:]-current[-nu:],
                self.param_.mpc_q_ub_[-nu:]-current[-nu:])
            alignment_controls.append(command)
            current = np.asarray(self.model.prediction_fn(current, command, phi, jac, mp)).reshape(-1)
        alignment_controls = np.asarray(alignment_controls)
        if sol_guess is None:
            controls = alignment_controls
        else:
            controls = np.asarray(sol_guess["x0"]).reshape(n, nu+self.param_.n_qpos_)[:, :nu]
            controls = np.vstack((controls[1:], controls[-1:]))
        controls = np.clip(controls, u_lb, u_ub)
        predicted, current = [], state.copy()
        for action in controls:
            current = np.asarray(self.model.prediction_fn(current, action, phi, jac,
                self.model.runtime_parameters(self.param_.model_params))).reshape(-1)
            predicted.append(current.copy())
        guess = dict(x0=np.column_stack((controls, predicted)).reshape(-1),
            u_traj=controls, x_traj=np.vstack((state, predicted)))
        result = super().plan_once(curr_x=state, phi_vec=phi, jac_mat=jac,
            cost_params=cp, sol_guess=guess, u_lb=u_lb, u_ub=u_ub)
        success = result.get("solver_backend") == "acados" or result["solve_status"] in {
            "Solve_Succeeded", "Solved_To_Acceptable_Level"}
        if not success or not np.isfinite(result["action"]).all():
            # Compare complete horizons, including force/object costs, rather
            # than executing an unverified correction on solver failure.
            value, trajectory = rollout_score(alignment_controls)
            zero = np.zeros_like(alignment_controls)
            baseline, zero_trajectory = rollout_score(zero)
            selected = alignment_controls if np.isfinite(value) and value < baseline else zero
            if selected is zero:
                trajectory = zero_trajectory
            result.update(action=selected[0].copy(), u_traj=selected,
                rollout_q=np.asarray(trajectory))
            result["fallback_control"] = "contact_alignment_correction"
            result["sol_guess"] = None
        elif result.get("solver_backend") == "acados":
            # One RTI QP can overshoot a contact-mode boundary. Evaluate
            # bounded backtracking candidates using the actual nonlinear
            # horizon, retaining the measured state and contact geometry.
            proposed = np.asarray(result["u_traj"]).reshape(n, nu)
            best_controls = np.zeros_like(proposed)
            best_cost, best_trajectory = rollout_score(best_controls)
            best_alpha = 0.
            for alpha in (1., .5, .25, .125):
                candidate = alpha*proposed
                value, trajectory = rollout_score(candidate)
                if np.isfinite(value) and value < best_cost:
                    best_cost, best_trajectory, best_controls, best_alpha = value, trajectory, candidate, alpha
            value, trajectory = rollout_score(alignment_controls)
            if np.isfinite(value) and value < best_cost:
                best_cost, best_trajectory, best_controls = value, trajectory, alignment_controls
                best_alpha = None
                result["selected_control"] = "contact_alignment_by_mpc_cost"
            if self.model.grasp_active:
                # Seed a motion of the existing grasp as a rigid body. The
                # normal-pressure map retains preload; prospective
                # pads are not assumed to transmit forces in the rollout.
                lift_command = np.zeros(nu)
                delta = np.asarray(goal)[:3]-state[:3]
                lift_command[:3] = delta*min(1., self.param_.h_*self.param_.lift_speed/
                    max(np.linalg.norm(delta), 1e-10))
                lift_command = np.clip(lift_command, u_lb, u_ub)
                for scale in (1., .5, .25):
                    candidate = np.tile(scale*lift_command, (n, 1))
                    value, trajectory = rollout_score(candidate)
                    if np.isfinite(value) and value < best_cost:
                        best_cost, best_trajectory, best_controls = value, trajectory, candidate
                        best_alpha = None
                        result['selected_control'] = 'grasp_transport_by_mpc_cost'
            result.update(action=best_controls[0].copy(), u_traj=best_controls,
                rollout_q=np.asarray(best_trajectory), nonlinear_step_scale=best_alpha)
            result["sol_guess"] = dict(x0=np.column_stack((best_controls, best_trajectory)).reshape(-1),
                u_traj=best_controls, x_traj=np.vstack((state, best_trajectory)))
        result['applied_action'] = self.model.effective_command(result['action'])
        return result



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
        # Joint attraction helps distant approach samples only. Once any
        # pad carries force the rollout is ranked by contact/object costs.
        attraction = (active_count == 0).float()
        if target_tip_world is not None:
            nearest_tip_error = torch.linalg.norm(tip_pos-target_tip_world, dim=2).amin(dim=1)
            attraction *= torch.clamp((nearest_tip_error-.002)/.018, 0., 1.).square()
        total_cost = (
            float(self.args.mjwp_object_cost_weight) * obj_pos_cost
            + float(self.args.mjwp_object_quat_cost_weight) * obj_quat_cost
            + attraction * (float(self.args.mjwp_base_cost_weight) * base_cost
                + float(self.args.mjwp_finger_cost_weight) * finger_cost)
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
        if int(args.num_grasp_contacts) not in (3, 4):
            raise ValueError("Allegro supports --num-grasp-contacts=3 or 4.")
        if args.controller == "staged" and int(args.num_grasp_contacts) != 4:
            raise ValueError("The legacy staged controller requires four fingertips.")
        self.active_finger_names = FINGER_NAMES[:int(args.num_grasp_contacts)]
        if args.lift_contact_fingers is None:
            args.lift_contact_fingers = 3
        if int(args.lift_contact_fingers) > len(self.active_finger_names):
            raise ValueError("--lift-contact-fingers exceeds the number of active fingers.")
        if int(args.lift_contact_fingers) < 1 or int(args.lift_contact_fingers) > len(FINGER_NAMES):
            raise ValueError(f"--lift-contact-fingers must be within [1, {len(FINGER_NAMES)}].")
        if int(args.max_lift_wait_steps) < 0:
            raise ValueError("--max-lift-wait-steps must be non-negative.")

        if args.mpc_max_contacts is None:
            args.mpc_max_contacts = 12 if args.controller == "online" else 24
        if args.controller == "online":
            if args.contact_noslip_iterations < 0:
                raise ValueError("--contact-noslip-iterations must be nonnegative")
            positive = ("mpc_horizon", "mpc_max_contacts", "mpc_ipopt_max_iter", "lambda_lm_iters",
                "max_point_combination_eval", "ik_contact_candidates", "ik_startup_time_budget", "online_tip_radius", "online_pad_stiffness",
                "online_lift_speed", "online_horizontal_recovery_weight", "online_max_horizontal_error",
                "online_contact_base_speed", "online_contact_rotation_speed", "online_contact_joint_speed", "online_pregrasp_clearance",
                "mpc_rotation_step_limit", "mpc_base_step_limit", "mpc_velocity_limit", "goal_tolerance",
                "contact_impratio", "online_normal_admittance", "online_local_contact_distance", "lambda_force_limit")
            for name in positive:
                if not np.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                    raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
            for name in ("online_tip_weight", "online_contact_transport_weight", "online_self_collision_weight",
                         "online_self_collision_clearance", "online_force_tracking_weight", "online_pressure_sync_weight",
                         "online_posture_weight", "online_base_position_weight", "online_palm_orientation_weight",
                         "online_min_normal_force"):
                if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
                    raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
            if not np.isfinite(args.online_friction_margin) or not 0 < args.online_friction_margin <= 1:
                raise ValueError("--online-friction-margin must be in (0, 1]")
            normal_budget = args.lambda_force_limit/np.sqrt(1+(args.object_friction*args.online_friction_margin)**2)
            if args.online_min_normal_force > normal_budget:
                raise ValueError("--online-min-normal-force exceeds the Lambda normal-force budget")
            if args.online_contact_transport_distance is not None and (
                    not np.isfinite(args.online_contact_transport_distance) or args.online_contact_transport_distance <= 0):
                raise ValueError("--online-contact-transport-distance must be finite and positive")
            if args.max_steps is not None and args.max_steps < 1:
                raise ValueError("--max-steps must be positive")
            if not 0 <= args.squeeze_depth < args.online_tip_radius:
                raise ValueError("--squeeze-depth must be in [0, online-tip-radius)")
            if args.mpc_max_contacts < args.num_grasp_contacts + 4:
                raise ValueError("Online MPC needs at least n fingertips plus four support contact slots")
            if args.test_force:
                raise ValueError("--test-force is supported only by the legacy staged controller")

        self.args = args
        self.startup_timings = {}
        scene_started = time.perf_counter()
        if args.controller == "online":
            print("[startup] building MuJoCo scene with palm facing the object", flush=True)
        self.mesh_path = resolve_mesh_path(args.obj, args.mesh)
        self.collision_mesh_path = prepare_collision_mesh_path(self.mesh_path)
        self.mesh_scale = resolve_mesh_scale(self.mesh_path, args.scale, args.scale_multiplier)
        self.visual_mesh_bounds = load_mesh_bounds(self.mesh_path, self.mesh_scale)
        self.collision_mesh_bounds = load_mesh_bounds(self.collision_mesh_path, self.mesh_scale)

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
                    self.support_top - self.collision_mesh_bounds[0, 2] + float(args.object_z_offset) + float(args.obj_init_height),
                ],
                dtype=np.float64,
            )
        else:
            object_pos = np.asarray(args.object_pos, dtype=np.float64).reshape(3)
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
            tip_center_sites=(args.controller == "online"),
            point_pads=(args.controller == "online" and args.online_point_pads),
            tip_radius=args.online_tip_radius,
            contact_cone="elliptic" if args.controller == "online" else "pyramidal",
            contact_impratio=args.contact_impratio if args.controller == "online" else 1.,
            contact_noslip_iterations=args.contact_noslip_iterations if args.controller == "online" else 0,
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = float(args.mujoco_dt)
        execution_noslip_iterations = int(self.model.opt.noslip_iterations)
        # Keep the startup resting pose/IK inputs independent of the online
        # friction refinement. NoSlip is needed for the subsequent grasp,
        # not for selecting a different initial pedestal equilibrium.
        self.model.opt.noslip_iterations = 0
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
        # Physical contact proxies belonging to the same hand must also
        # constrain IK. Otherwise neighbouring phalanges block its targets.
        self.hand_collision_geom_ids = set()
        for geom in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[geom])
            while body and body != self.palm_body_id:
                body = int(self.model.body_parentid[body])
            if body == self.palm_body_id and (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]):
                self.hand_collision_geom_ids.add(geom)
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
        if args.soft_scm_response_time is None:
            # Match the simulator's contact time scale; the shared .05 s
            # quasi-dynamic default is not the MuJoCo .02 s material setting.
            args.soft_scm_response_time = max(2*float(self.model.opt.timestep),
                float(self.model.geom_solref[self.object_geom_id, 0]))
        if args.lambda_prediction_dt is None:
            args.lambda_prediction_dt = self.command_dt
        if not np.isfinite(args.lambda_prediction_dt) or args.lambda_prediction_dt <= 0:
            raise ValueError("--lambda-prediction-dt must be finite and positive")
        self.active_tip_site_ids = self.tip_site_ids[:len(self.active_finger_names)].copy()
        if args.controller == "online":
            self.startup_timings["scene_seconds"] = time.perf_counter() - scene_started
            started = time.perf_counter()
            print("[startup] preprocessing contact surface", flush=True)
            collision_mesh = trimesh.load_mesh(self.collision_mesh_path, process=False).convex_hull
            self.online_vertices = np.asarray(collision_mesh.vertices) * self.mesh_scale
            self.optimizer = LambdaContactBatchedLM(
                mesh_path=collision_mesh, num_contacts=len(self.active_finger_names),
                scale_factors=tuple(self.mesh_scale), obj_mass=args.obj_mass,
                sample_num=args.sample_num, arm_friction=args.object_friction,
                contact_stiffness=args.contact_stiffness, time_step=args.lambda_prediction_dt, ori_coef=.05,
                tip_radius=args.online_tip_radius, device=args.lambda_device,
                lm_iterations=args.lambda_lm_iters, seed=args.seed,
                candidate_capacity=args.max_point_combination_eval,
                max_contact_force=args.lambda_force_limit,
                friction_cone="coulomb", require_force_closure=True,
                max_point_curvature=1., max_mean_curvature=1.,
                body_sides=(args.lambda_contact_region == "body-sides"), body_core_fraction=.65,
                support_surface_point=self.support_surface_point,
                support_surface_normal=self.support_surface_normal,
                support_surface_clearance=args.ground_height_margin)
            self.startup_timings["surface_preprocessing_seconds"] = time.perf_counter() - started
            print(f"[startup] surface preprocessing: {self.startup_timings['surface_preprocessing_seconds']:.3f}s", flush=True)
            self.mpc_qvel_full_indices = np.r_[
                np.arange(self.object_dof_adr, self.object_dof_adr + 6), self.hand_dof_adr]
            self.mpc_param = AllegroOnlineMPCParams(self)
            started = time.perf_counter()
            print("[startup] constructing MPC (acados compiles on cache miss)", flush=True)
            self.mpc = AllegroOnlineMPC(self.mpc_param)
            self.startup_timings["mpc_initialization_seconds"] = time.perf_counter() - started
            print(f"[startup] MPC initialization: {self.startup_timings['mpc_initialization_seconds']:.3f}s", flush=True)
            self.mjwp_rollout = None
        else:
            from planning.mlqp_point_v2 import LambdaContactControlOptimizer

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
                static_equilibrium_cost_weight=float(args.grasp_lift_cost_weight),
                static_equilibrium_residual_weight=float(args.grasp_lift_residual_weight),
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
            self.mjwp_rollout = None
            if bool(args.mjwp_rollout):
                try:
                    self.mjwp_rollout = MujocoWarpGraspRolloutController(self)
                except Exception as exc:
                    print(f"Warning: MJWP sampling rollout is disabled: {exc}")
        self.reset(self.initial_object_pos, self.initial_object_quat, self.default_hand_q)
        self.model.opt.noslip_iterations = execution_noslip_iterations

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
        if not bool(getattr(self.args, "save", False)) or bool(getattr(self.args, "no_video", False)):
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
        self.online_local_points = np.full((len(self.active_finger_names), 3), np.nan)
        self.online_local_normals = np.full_like(self.online_local_points, np.nan)
        self.online_sticking_anchors = np.full_like(self.online_local_points, np.nan)
        if hasattr(self, 'mpc'):
            self.mpc.pressure_floor = None
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.set_object_pose(object_pos, object_quat, data=self.data, forward=False)
        self.set_hand_qpos(hand_qpos, data=self.data, forward=False)
        self.online_servo_target = np.asarray(hand_qpos, float).copy()
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

    def _best_object_wrench_world(self, optimizer_result, object_rot):
        if optimizer_result is None:
            return None, None

        object_rot = project_to_rotation_matrix(object_rot)
        wrench_source = optimizer_result
        static_result = None
        if isinstance(optimizer_result, dict):
            static_result = optimizer_result.get("static_equilibrium", None)
        if isinstance(static_result, dict) and bool(static_result.get("valid", False)):
            wrench_source = static_result

        wrench_info = self.optimizer._result_contact_wrench(
            wrench_source,
            r_obj_to_world=object_rot,
        )
        force_world = np.asarray(
            wrench_info.get("contact_force_world", np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3)
        torque_world = np.asarray(
            wrench_info.get("contact_torque_world", np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3)
        return force_world, torque_world

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
        if data is self.data:
            self.online_servo_target = hand_qpos.copy()
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
        if self.args.controller == "online":
            return np.r_[object_pos, object_quat, data.qpos[self.hand_qpos_adr],
                data.qvel[self.mpc_qvel_full_indices], self.online_servo_target]
        base_pos = np.asarray(data.qpos[self.base_pos_qpos_adr], dtype=np.float64).copy()
        finger_q = self.get_finger_qpos(data)
        return np.concatenate((object_pos, object_quat, base_pos, finger_q), axis=0)

    def detect_mpc_contacts(self):
        if self.args.controller == "online":
            return self.detect_online_contacts()
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

    def detect_online_contacts(self):
        from models.support_geometry import mesh_support_points, support_jacobians
        pos, _, rotation = self.get_object_pose()
        rows = 4*self.mpc_param.max_ncon_
        phi, jac = np.ones(rows), np.zeros((rows, self.mpc_param.n_qvel_))
        self.online_contact_points_local = np.full((len(self.active_finger_names), 3), np.nan)
        self.online_contact_normals_local = np.full_like(self.online_contact_points_local, np.nan)
        for index, name in enumerate(self.active_finger_names):
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, TIP_COLLISION_GEOM_NAMES[name])
            fromto = np.zeros(6)
            gap = mujoco.mj_geomDistance(self.model, self.data, geom, self.object_geom_id, 1., fromto)
            normal = normalize(fromto[3:]-fromto[:3])
            point = fromto[3:].copy()
            # At penetration the measured normal/point is more reliable than
            # a closest-point difference, exactly as in bigrasp's pad query.
            matching = [self.data.contact[k] for k in range(self.data.ncon)
                if {int(self.data.contact[k].geom1), int(self.data.contact[k].geom2)} == {geom, self.object_geom_id}]
            if matching:
                contact = min(matching, key=lambda c: c.dist)
                point, gap = np.asarray(contact.pos).copy(), float(contact.dist)
                normal = np.asarray(contact.frame[:3]).copy()
                if int(contact.geom1) == self.object_geom_id:
                    normal *= -1
            if np.linalg.norm(normal) < .5:
                continue
            self.online_contact_points_local[index] = (point-pos) @ rotation
            self.online_contact_normals_local[index] = normal @ rotation
            block, _ = support_jacobians(pos, rotation, point, self.mpc_param.n_qvel_,
                self.mpc_param.mu_object_, normal)
            jp = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, None, point, int(self.model.geom_bodyid[geom]))
            directions = block[:, :3].copy()
            block[:, 6:] = -directions @ jp[:, self.hand_dof_adr]
            phi[4*index:4*index+4] = gap
            jac[4*index:4*index+4] = block
        start = len(self.active_finger_names)
        support = mesh_support_points(self.online_vertices, pos, rotation, self.support_surface_point,
            self.support_surface_normal, tangents=np.array([[1., 0., 0.], [0., 1., 0.]]),
            half_extents=self.pedestal_size[:2])
        support = support[(support-self.support_surface_point) @ self.support_surface_normal >= -.02]
        for index, point in enumerate(support[:4]):
            block, _ = support_jacobians(pos, rotation, point, self.mpc_param.n_qvel_,
                self.mpc_param.mu_object_, self.support_surface_normal)
            gap = float((point-self.support_surface_point) @ self.support_surface_normal)
            phi[4*(start+index):4*(start+index)+4] = gap
            jac[4*(start+index):4*(start+index)+4] = block
        # Allegro's distal links can touch before their pads. Their actual
        # contacts must enter the same horizontal-displacement objective.
        pad_geoms = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
            TIP_COLLISION_GEOM_NAMES[name]) for name in self.active_finger_names}
        extra = {}
        self_contacts = {}
        pedestal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pedestal")
        for contact in self.data.contact:
            if int(contact.geom1) in self.hand_collision_geom_ids and int(contact.geom2) in self.hand_collision_geom_ids:
                key = tuple(sorted((int(contact.geom1), int(contact.geom2))))
                if key not in self_contacts or contact.dist < self_contacts[key].dist:
                    self_contacts[key] = contact
            if self.object_geom_id not in (contact.geom1, contact.geom2):
                continue
            other = int(contact.geom2) if contact.geom1 == self.object_geom_id else int(contact.geom1)
            if other in pad_geoms:
                continue
            body = int(self.model.geom_bodyid[other])
            normal = np.asarray(contact.frame[:3]).copy()
            if contact.geom1 == self.object_geom_id:
                normal *= -1
            if body == pedestal and normal @ self.support_surface_normal > .9:
                continue
            if other not in extra or contact.dist < extra[other][0]:
                extra[other] = (float(contact.dist), np.asarray(contact.pos).copy(), normal, body)
        slot = start+4
        candidates = {key: (float(c.dist), np.asarray(c.pos).copy(), np.asarray(c.frame[:3]).copy(),
            int(c.geom1), int(c.geom2)) for key, c in self_contacts.items()}
        # Add separated inter-finger patches before impact, so the optimizer
        # does not use joint nullspace freedom to collide neighbouring links.
        finger_geoms = []
        for geom in self.hand_collision_geom_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''
            finger = next((f for f in FINGER_NAMES if f'_{f}_' in name), None)
            if finger is not None:
                finger_geoms.append((geom, finger))
        margin = max(self.args.online_self_collision_clearance*2, .003)
        for (first_geom, first_finger), (second_geom, second_finger) in itertools.combinations(finger_geoms, 2):
            if first_finger == second_finger or tuple(sorted((first_geom, second_geom))) in candidates:
                continue
            segment = np.zeros(6)
            gap = float(mujoco.mj_geomDistance(self.model, self.data, first_geom, second_geom, margin, segment))
            if gap >= margin:
                continue
            normal = normalize(segment[3:]-segment[:3])
            if np.linalg.norm(normal) > .5:
                candidates[tuple(sorted((first_geom, second_geom)))] = (
                    gap, .5*(segment[:3]+segment[3:]), normal, first_geom, second_geom)
        # Robot-only normal constraints transmit no force to the object.
        # Omitting these let MPC predict that blocked fingers could move.
        for gap, point, normal, first_geom, second_geom in sorted(candidates.values(), key=lambda c:c[0]):
            if slot >= self.mpc_param.max_ncon_:
                break
            first, second = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, first, None, point, int(self.model.geom_bodyid[first_geom]))
            mujoco.mj_jac(self.model, self.data, second, None, point, int(self.model.geom_bodyid[second_geom]))
            normal_row = normal@(second-first)[:, self.hand_dof_adr]
            phi[4*slot:4*slot+4] = gap
            jac[4*slot:4*slot+4, 6:] = normal_row
            slot += 1
        for gap, point, normal, body in sorted(extra.values(), key=lambda c:c[0]):
            if slot >= self.mpc_param.max_ncon_:
                break
            block, _ = support_jacobians(pos, rotation, point, self.mpc_param.n_qvel_,
                self.mpc_param.mu_object_, normal)
            jp = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, None, point, body)
            block[:, 6:] = -block[:, :3] @ jp[:, self.hand_dof_adr]
            phi[4*slot:4*slot+4] = gap
            jac[4*slot:4*slot+4] = block
            slot += 1
        return phi, jac

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

    def build_world_contact_targets(self):
        visible_idx = np.where(
            np.asarray(self.optimizer.sample_point[:, 2], dtype=np.float64) >= self.local_top_half_z_threshold
        )[0]
        if visible_idx.size == 0:
            visible_idx = np.arange(self.optimizer.sample_num, dtype=int)

        gravity_local = self._gravity_wrench_local(self.initial_object_rot)
        grasp_result = self.optimizer.get_best_grasp(
            visible_face_idx=visible_idx,
            object_pos=self.initial_object_pos,
            object_rot=self.initial_object_rot,
            external_wrench=gravity_local,
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

        palm_normal_local = PALM_NORMAL_LOCAL

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

    def solve_single_ik_candidate(self, q_init, target_palm_pos, target_palm_rot, tip_targets_world, tip_normals_world=None):
        q = np.asarray(q_init, dtype=np.float64).reshape(self.model.nu).copy()
        align_tip_axes = tip_normals_world is not None and not (
            self.args.controller == "online" and self.args.online_point_pads)
        q_nominal = q.copy()
        best_payload = None
        best_score = np.inf
        best_feasible = False
        identity = np.eye(self.hand_dof_adr.size, dtype=np.float64)
        regularization_weights = np.concatenate(
            [
                float(self.args.ik_base_regularization) * np.ones(BASE_ACTUATOR_COUNT, dtype=np.float64),
                float(self.args.ik_finger_regularization) * np.ones(self.hand_dof_adr.size - BASE_ACTUATOR_COUNT, dtype=np.float64),
            ]
        )
        if self.args.controller == "online":
            regularization_weights[:] = self.args.online_ik_posture_regularization

        last_improvement = 0
        for iteration in range(int(self.args.ik_max_iters)):
            if best_payload is not None and time.perf_counter() >= getattr(self, "_ik_startup_deadline", np.inf):
                break
            palm_pos, palm_rot, tip_positions = self.forward_kinematics(q)
            error_terms = []
            jacobian_terms = []

            palm_jacp_full = np.zeros((3, self.model.nv), dtype=np.float64)
            palm_jacr_full = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self.solve_data, palm_jacp_full, palm_jacr_full, self.palm_body_id)
            palm_jacp = palm_jacp_full[:, self.hand_dof_adr]
            palm_jacr = palm_jacr_full[:, self.hand_dof_adr]
            # The palm height inferred from OPEN fingertips is not a grasp
            # constraint. Let coupled IK move it as the fingers curl.
            palm_position_weight = self.args.online_ik_palm_position_weight if self.args.controller == "online" else float(self.args.ik_palm_pos_weight)
            error_terms.append(palm_position_weight * (target_palm_pos - palm_pos))
            jacobian_terms.append(palm_position_weight * palm_jacp)
            error_terms.append(float(self.args.ik_palm_rot_weight) * rotation_error(palm_rot, target_palm_rot))
            jacobian_terms.append(float(self.args.ik_palm_rot_weight) * palm_jacr)

            for tip_index, (site_id, target_tip) in enumerate(zip(self.tip_site_ids, tip_targets_world)):
                jacp_full = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr_full = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, self.solve_data, jacp_full, jacr_full, site_id)
                jacp = jacp_full[:, self.hand_dof_adr]
                error_terms.append(float(self.args.ik_tip_weight) * (target_tip - self.solve_data.site(site_id).xpos))
                jacobian_terms.append(float(self.args.ik_tip_weight) * jacp)
                if align_tip_axes:
                    axis = self.solve_data.site(site_id).xmat.reshape(3,3)[:,2]
                    error_terms.append(self.args.ik_tip_orientation_weight * np.cross(axis, tip_normals_world[tip_index]))
                    jacobian_terms.append(self.args.ik_tip_orientation_weight * jacr_full[:, self.hand_dof_adr])

            if tip_normals_world is not None:
                pads = set(self.fingertip_geom_id_to_name)
                for contact in self.solve_data.contact:
                    if self.object_geom_id not in (contact.geom1, contact.geom2):
                        continue
                    other = int(contact.geom2) if contact.geom1 == self.object_geom_id else int(contact.geom1)
                    if other in pads or contact.dist >= .002:
                        continue
                    body = int(self.model.geom_bodyid[other])
                    if body in (0, mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pedestal")):
                        continue
                    normal = np.asarray(contact.frame[:3]).copy()
                    if contact.geom1 == self.object_geom_id:
                        normal *= -1
                    jp = np.zeros((3, self.model.nv))
                    mujoco.mj_jac(self.model, self.solve_data, jp, None, np.asarray(contact.pos), body)
                    error_terms.append(np.array([self.args.ik_collision_weight * (.002-contact.dist)]))
                    jacobian_terms.append(-self.args.ik_collision_weight * normal[None] @ jp[:, self.hand_dof_adr])

            if self.args.controller == "online":
                for contact in self.solve_data.contact:
                    if int(contact.geom1) not in self.hand_collision_geom_ids or int(contact.geom2) not in self.hand_collision_geom_ids:
                        continue
                    normal = np.asarray(contact.frame[:3])
                    first, second = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
                    point = np.asarray(contact.pos)
                    mujoco.mj_jac(self.model, self.solve_data, first, None, point, int(self.model.geom_bodyid[contact.geom1]))
                    mujoco.mj_jac(self.model, self.solve_data, second, None, point, int(self.model.geom_bodyid[contact.geom2]))
                    error_terms.append(np.array([self.args.ik_collision_weight*(.002-float(contact.dist))]))
                    jacobian_terms.append(self.args.ik_collision_weight*normal[None]@(second-first)[:, self.hand_dof_adr])

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
            tip_error_norms = np.linalg.norm(tip_positions[:len(tip_targets_world)] - tip_targets_world, axis=1)
            mean_tip_error = float(np.mean(tip_error_norms))
            max_tip_error = float(np.max(tip_error_norms))
            palm_pos_error = float(np.linalg.norm(target_palm_pos - palm_pos))
            palm_rot_error = float(np.linalg.norm(rotation_error(palm_rot, target_palm_rot)))
            self_depths = [max(-float(c.dist), 0.) for c in self.solve_data.contact
                if int(c.geom1) in self.hand_collision_geom_ids and int(c.geom2) in self.hand_collision_geom_ids]
            max_self_penetration = max(self_depths, default=0.)
            palm_score_weight = .005 if self.args.controller == "online" else .20
            score = mean_tip_error + 0.35 * max_tip_error + palm_score_weight * palm_pos_error + 0.05 * palm_rot_error
            if self.args.controller == "online":
                score += 5*sum(self_depths)
            alignment = 1.
            if tip_normals_world is not None:
                if align_tip_axes:
                    axes = np.array([self.solve_data.site(site).xmat.reshape(3,3)[:,2] for site in self.tip_site_ids[:len(tip_targets_world)]])
                    alignment = float(np.mean(np.einsum("ij,ij->i", axes, tip_normals_world)))
                    score += .02 * (1-alignment)
                pads = set(self.fingertip_geom_id_to_name)
                depth = sum(max(-float(c.dist), 0.) for c in self.solve_data.contact
                    if self.object_geom_id in (c.geom1,c.geom2) and
                    (int(c.geom2) if c.geom1==self.object_geom_id else int(c.geom1)) not in pads and
                    int(self.model.geom_bodyid[int(c.geom2) if c.geom1==self.object_geom_id else int(c.geom1)]) not in (0, mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"pedestal")))
                score += 2*depth
            tip_tolerance = self.args.online_ik_tip_tolerance if self.args.controller == "online" else float(self.args.ik_tip_tol)
            max_tip_tolerance = tip_tolerance*1.5
            feasible = (mean_tip_error <= tip_tolerance and max_tip_error <= max_tip_tolerance
                and alignment >= .75 and max_self_penetration <= 1e-4)
            # A lower palm/posture score must not displace a seed that
            # already passes the reachability and collision requirements.
            if (best_payload is None or (feasible and not best_feasible)
                    or (feasible == best_feasible and score < best_score)):
                if feasible != best_feasible or best_score - score > 1e-5:
                    last_improvement = iteration
                best_score = score
                best_feasible = feasible
                best_payload = (
                    q.copy(),
                    mean_tip_error,
                    max_tip_error,
                    palm_pos_error,
                    palm_rot_error,
                    tip_positions.copy(),
                    max_self_penetration,
                )
            if feasible:
                break
            if self.args.controller == "online" and iteration - last_improvement >= 15:
                break

        if best_payload is None:
            raise RuntimeError("IK solver did not produce any candidate pose.")
        return best_score, *best_payload

    def solve_grasp_ik(self, fingertip_targets_world, contact_side_signs=None, contact_normals_world=None):
        best_result = None
        best_score = np.inf
        open_finger_q = self.build_open_finger_pose(None)
        q_seed = np.zeros(self.model.nu, dtype=np.float64)
        q_seed[BASE_ACTUATOR_COUNT:] = open_finger_q.copy()
        nominal_local_tip_positions = self.local_tip_positions(q_seed)[:len(fingertip_targets_world)]
        nominal_local_tip_centroid = np.mean(nominal_local_tip_positions, axis=0)

        for permutation in itertools.permutations(range(len(fingertip_targets_world))):
            if best_result is not None and time.perf_counter() >= getattr(self, "_ik_startup_deadline", np.inf):
                break
            if contact_side_signs is not None:
                signs = np.asarray(contact_side_signs)[list(permutation)]
                if not np.all(signs[1:] == -signs[0]):
                    continue
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

            score, qpos, mean_tip_error, max_tip_error, palm_pos_error, palm_rot_error, solved_tip_positions, self_penetration = self.solve_single_ik_candidate(
                q_init=q_init,
                target_palm_pos=target_palm_pos_world,
                target_palm_rot=target_palm_rot_world,
                tip_targets_world=assigned_targets_world,
                tip_normals_world=None if contact_normals_world is None else contact_normals_world[list(permutation)],
            )
            feasible = mean_tip_error <= (self.args.online_ik_tip_tolerance if self.args.controller == "online" else float(self.args.ik_tip_tol)) and max_tip_error <= 1.5*(self.args.online_ik_tip_tolerance if self.args.controller == "online" else float(self.args.ik_tip_tol)) and (self.args.controller != "online" or self_penetration <= 1e-4)
            if best_result is None or (feasible and not best_result.success) or (feasible == best_result.success and score < best_score):
                best_score = score
                best_result = IkSolveResult(
                    success=feasible,
                    qpos=qpos,
                    score=score,
                    mean_tip_error=mean_tip_error,
                    max_tip_error=max_tip_error,
                    palm_pos_error=palm_pos_error,
                    palm_rot_error=palm_rot_error,
                    permutation=tuple(int(idx) for idx in permutation),
                    target_palm_pos_world=self.forward_kinematics(qpos)[0].copy(),
                    target_palm_rot_world=(self.forward_kinematics(qpos)[1].copy() if self.args.controller == "online"
                                           else np.asarray(target_palm_rot_world, dtype=np.float64).copy()),
                    assigned_targets_world=np.asarray(assigned_targets_world, dtype=np.float64).copy(),
                    solved_tip_positions_world=np.asarray(solved_tip_positions, dtype=np.float64).copy(),
                    max_self_penetration=self_penetration,
                )

        if best_result is None:
            raise RuntimeError("Failed to find an Allegro IK solution for the selected contact targets.")
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
        # The initial palm is above the object: face it down before any
        # optimizer or viewer starts, rather than applying a yaw to an
        # upward-facing reference pose.
        target_palm_rot = np.column_stack(([0., 0., -1.], [0., 1., 0.], [1., 0., 0.]))
        return self.build_hover_hand_q(
            palm_pos_world=default_palm_pos_world,
            base_orientation=self.base_orientation_euler_from_target_palm_rot(target_palm_rot),
            finger_q=self.build_open_finger_pose(None),
        )

    def base_orientation_euler_from_target_palm_rot(self, target_palm_rot_world, reference=None):
        target_palm_rot_world = project_to_rotation_matrix(target_palm_rot_world)
        target_base_rot_world = project_to_rotation_matrix(target_palm_rot_world @ self.fixed_palm_rot_world.T)
        # MuJoCo applies these co-located hinges as Rx @ Ry @ Rz.
        angles = Rotation.from_matrix(target_base_rot_world).as_euler("XYZ")
        if reference is None:
            return angles
        # Equivalent Euler solutions must remain near the measured hinges.
        # Crossing pi previously requested a nearly 2*pi servo rotation.
        reference = np.asarray(reference, float).reshape(3)
        alternatives = (angles, np.array([angles[0]+np.pi, np.pi-angles[1], angles[2]+np.pi]))
        candidates = []
        for solution in alternatives:
            choices = [solution[i]+2*np.pi*np.arange(-2, 3) for i in range(3)]
            choices = [values[(values >= self.base_joint_ranges[i+3, 0]) &
                              (values <= self.base_joint_ranges[i+3, 1])] for i, values in enumerate(choices)]
            candidates.extend(np.array(value) for value in itertools.product(*choices))
        if not candidates:
            raise ValueError("Palm orientation is outside the base hinge ranges")
        return min(candidates, key=lambda value: np.sum((value-reference)**2))

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
        for finger_name, point in zip(self.active_finger_names, target_points_world):
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
        if self.args.controller == "staged":
            return self.run_staged()
        return self.run_online()

    def lambda_object_dynamics(self, rotation):
        # Identical MuJoCo mass/bias convention to bigrasp.py.
        mass = np.empty((self.model.nv, self.model.nv))
        try:
            mujoco.mj_fullM(self.model, self.data, mass)
        except TypeError:
            mujoco.mj_fullM(self.model, mass, self.data.qM)
        dofs = np.arange(self.object_dof_adr, self.object_dof_adr + 6)
        external = (self.data.qfrc_passive[dofs] + self.data.qfrc_applied[dofs]
                    - self.data.qfrc_bias[dofs]).copy()
        external[3:] = rotation @ external[3:]
        self.hand_mass_diagonal = np.diag(mass)[self.hand_dof_adr].copy()
        return mass[np.ix_(dofs, dofs)], external

    def online_tip_kinematics(self):
        tips, jacobians = [], []
        for site in self.active_tip_site_ids:
            jp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jp, None, int(site))
            tips.append(self.data.site(site).xpos.copy())
            jacobians.append(jp[:, self.hand_dof_adr])
        return np.asarray(tips), np.vstack(jacobians)

    def refine_online_contact_patches(self, points, normals, phi, tips, tip_jac, external):
        """Adopt reachable nearby patches while preserving existing contacts.

        Certificates on prospective patches are predictive only. The lift
        audit and contact model continue to use measured physical contacts.
        """
        n = len(points)
        if not hasattr(self, 'online_local_points') or self.online_local_points.shape != (n, 3):
            self.online_local_points = np.full((n, 3), np.nan)
            self.online_local_normals = np.full((n, 3), np.nan)
        selected, directions = np.asarray(points).copy(), np.asarray(normals).copy()
        saved = np.isfinite(self.online_local_points).all(axis=1)
        selected[saved], directions[saved] = self.online_local_points[saved], self.online_local_normals[saved]
        gaps = np.asarray(phi).reshape(-1, 4)[:n, 0]
        summary = self.get_fingertip_contact_summary()
        touching = (gaps <= 0) & np.array([summary[name] > self.args.contact_force_threshold
                                         for name in self.active_finger_names])
        valid = np.isfinite(self.online_contact_points_local).all(axis=1)
        selected[touching & valid] = self.online_contact_points_local[touching & valid]
        directions[touching & valid] = self.online_contact_normals_local[touching & valid]
        candidates = (~touching) & valid & (gaps <= self.args.online_local_contact_distance)
        reachable = np.zeros(n, bool)
        pos, quat, rotation = self.get_object_pose()
        hand = self.get_hand_qpos()
        depth = min(self.args.squeeze_depth, .0005)
        baseline_self = max((max(-float(c.dist), 0.) for c in self.data.contact
            if c.geom1 in self.hand_collision_geom_ids and c.geom2 in self.hand_collision_geom_ids), default=0.)
        for i in np.flatnonzero(candidates):
            target = pos+rotation@(self.online_contact_points_local[i]
                -(self.args.online_tip_radius-depth)*self.online_contact_normals_local[i])
            trial = hand.copy()
            for attempt in range(3):
                _, _, trial_tips = self.forward_kinematics(trial)
                if np.linalg.norm(trial_tips[i]-target) < .0005:
                    break
                if attempt == 0:
                    finger_jac = tip_jac[3*i:3*i+3, 6:]
                else:
                    jp = np.zeros((3, self.model.nv))
                    mujoco.mj_jacSite(self.model, self.solve_data, jp, None, int(self.active_tip_site_ids[i]))
                    finger_jac = jp[:, self.hand_dof_adr[6:]]
                delta = finger_jac.T@np.linalg.solve(finger_jac@finger_jac.T+1e-6*np.eye(3), target-trial_tips[i])
                trial[6:] = np.clip(trial[6:]+np.clip(delta, -.15, .15),
                    self.hand_joint_ranges[6:, 0], self.hand_joint_ranges[6:, 1])
            _, _, trial_tips = self.forward_kinematics(trial)
            if np.linalg.norm(trial_tips[i]-target) > .002:
                continue
            self.set_object_pose(pos, quat, data=self.solve_data)
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                TIP_COLLISION_GEOM_NAMES[self.active_finger_names[i]])
            # The local IK is only a reachability witness/initial guess.
            # It is never written into the real hand or used as a cost.
            if mujoco.mj_geomDistance(self.model, self.solve_data, geom, self.object_geom_id, .03, None) > 1e-5:
                continue
            self_depth = max((max(-float(c.dist), 0.) for c in self.solve_data.contact
                if c.geom1 in self.hand_collision_geom_ids and c.geom2 in self.hand_collision_geom_ids), default=0.)
            if self_depth > max(baseline_self, 1e-4)+1e-5:
                continue
            reachable[i] = True
        events = []
        self.online_local_repair_feasible = False
        if reachable.any():
            from planning.grasp_diagnostics import object_contact_wrenches
            from planning.local_contact_replanning import choose_local_contact_updates
            wrenches, _ = object_contact_wrenches(self.model, self.data, self.object_body_id,
                self.fingertip_geom_id_to_name, self.hand_collision_geom_ids)
            total_external = external+wrenches['other_hand']
            local_external = np.r_[rotation.T@total_external[:3], rotation.T@total_external[3:]]
            caps = np.array([summary[name] if touching[i] else max(self.args.online_min_normal_force, .6)
                             for i, name in enumerate(self.active_finger_names)])
            selected, directions, indices, certificate = choose_local_contact_updates(
                selected, directions, self.online_contact_points_local, self.online_contact_normals_local,
                touching & valid, reachable, caps, self.args.object_friction, local_external)
            self.online_local_repair_feasible = len(indices) > 0
            for i in indices:
                changed = (not saved[i] or np.linalg.norm(selected[i]-self.online_local_points[i]) > .001)
                self.online_local_points[i], self.online_local_normals[i] = selected[i], directions[i]
                if changed:
                    events.append(dict(finger=self.active_finger_names[i], gap=float(gaps[i]),
                        point_local=selected[i].tolist(), predictive_force_closure=certificate))
        return selected, directions, events

    def update_online_contact_forces(self, points, inward, info, goal, pose, mass, external, phi):
        """Re-solve forces at measured/nearest pad contacts, in finger order.

        Far pads keep their planned patches as a contact initial guess. The
        same closest points and normals drive the MPC contact Jacobians.
        A certificate on this mixed set is predictive; only the measured
        touching-contact audit can authorize lifting.
        """
        force_points, force_normals = points.copy(), inward.copy()
        gaps = np.asarray(phi).reshape(-1, 4)[:len(points), 0]
        measured = (gaps <= .02) & np.isfinite(self.online_contact_points_local).all(axis=1)
        force_points[measured] = self.online_contact_points_local[measured]
        force_normals[measured] = self.online_contact_normals_local[measured]
        if measured.any():
            result = self.optimizer.lm.solve(force_points[None], force_normals[None],
                current_x=pose, x_d=goal, object_inertia=mass, tau_o=external,
                v_obj=self.data.qvel[self.object_dof_adr:self.object_dof_adr+6],
                prediction_dt=self.args.lambda_prediction_dt)
            # Keep contact assignment metadata from the patch policy, while
            # reporting forces and predictions from the live geometry solve.
            info = dict(info, **{key: result[key] for key in (
                "force_vectors_local", "control_input", "x_plus_opt", "velocity_next",
                "contact_force_world", "contact_torque_world", "cost", "pose_cost",
                "pose_reduction", "baseline_reduction", "solver_failed", "solver_converged")},
                live_force_solve_time=result["solve_time"])
        certificate = verify_grasp(force_points, force_normals, self.args.object_friction,
            external_wrench=np.r_[quat_wxyz_to_mat(pose[3:7]).T@external[:3],
                                 quat_wxyz_to_mat(pose[3:7]).T@external[3:]],
            force_limit=self.args.lambda_force_limit)
        allocation = None
        other_hand_wrench = np.zeros(6)
        summary = self.get_fingertip_contact_summary()
        active = (gaps <= 0) & measured & np.array([summary[name] > self.args.contact_force_threshold
                                                  for name in self.active_finger_names])
        if np.count_nonzero(active) >= 3:
            from planning.grasp_force_allocation import allocate_contact_wrench
            from planning.grasp_diagnostics import object_contact_wrenches
            wrenches, _ = object_contact_wrenches(self.model, self.data, self.object_body_id,
                self.fingertip_geom_id_to_name, self.hand_collision_geom_ids)
            other_hand_wrench = wrenches['other_hand']
            rotation = quat_wxyz_to_mat(pose[3:7])
            velocity = self.data.qvel[self.object_dof_adr:self.object_dof_adr+6].copy()
            target_linear = (np.asarray(goal)[:3]-pose[:3])/self.args.lambda_prediction_dt
            target_linear *= min(1., self.args.online_lift_speed/max(np.linalg.norm(target_linear), 1e-10))
            target_angular = Rotation.from_matrix(quat_wxyz_to_mat(goal[3:7])@rotation.T).as_rotvec()/.25
            target_angular *= min(1., self.args.online_contact_rotation_speed/max(np.linalg.norm(target_angular), 1e-10))
            frame = np.eye(6); frame[3:, 3:] = rotation
            acceleration = (np.r_[target_linear, target_angular]-frame@velocity)/.1
            # Distal/proximal contacts also load the object. Allocate the
            # remaining wrench to the controlled pads. Table support is not
            # subtracted: the pads must replace it to lift the object.
            wrench_world = frame@mass@frame.T@acceleration-external-other_hand_wrench
            wrench_local = np.r_[rotation.T@wrench_world[:3], rotation.T@wrench_world[3:]]
            caps = np.array([summary[name] for name in self.active_finger_names])[active]
            # Current pressure limits certify the forces available NOW; they
            # must not also freeze the requested future preload. Optimize a
            # balanced load-supporting allocation and let the servo build it.
            allocation = allocate_contact_wrench(force_points[active], force_normals[active], wrench_local,
                self.args.object_friction, minimum_normal=self.args.online_min_normal_force,
                force_limit=self.args.lambda_force_limit, friction_margin=self.args.online_friction_margin)
            available = verify_grasp(force_points[active], force_normals[active], self.args.object_friction,
                external_wrench=-wrench_local, normal_force_limits=caps)
            if allocation['feasible']:
                info['force_vectors_local'] = np.zeros_like(info['force_vectors_local'])
                info['force_vectors_local'][active] = allocation['forces']
                predicted, next_velocity = self.mpc.model.applied_force_prediction_fn(pose, velocity,
                    force_points.T, np.asarray(info['force_vectors_local']).T, np.linalg.inv(mass),
                    np.r_[external[:3]+other_hand_wrench[:3],
                        rotation.T@(external[3:]+other_hand_wrench[3:])])
                info.update(x_plus_opt=np.asarray(predicted).reshape(-1),
                    velocity_next=np.asarray(next_velocity).reshape(-1),
                    contact_force_world=rotation@allocation['forces'].sum(0),
                    contact_torque_world=rotation@np.cross(force_points[active], allocation['forces']).sum(0))
        return dict(info, force_points_local=force_points, force_normals_local=force_normals,
            measured_force_contacts=measured, live_force_closure_certificate=certificate,
            force_allocation_feasible=None if allocation is None else allocation['feasible'],
            allocated_wrench_available=None if allocation is None else available['load_feasible'],
            other_hand_wrench_world=other_hand_wrench,
            allocated_wrench_local=None if allocation is None or not allocation['feasible']
                else allocation['requested_wrench'])

    def execute_online_joint_target(self, desired):
        """Execute an absolute IK/servo target with the same bias convention."""
        self.online_servo_target = np.clip(np.asarray(desired, float),
            self.hand_joint_ranges[:, 0], self.hand_joint_ranges[:, 1])
        kp = self.model.actuator_gainprm[:, 0]
        for _ in range(max(self.args.mj_steps_per_command, 1)):
            mujoco.mj_forward(self.model, self.data)
            bias = (self.data.qfrc_bias[self.hand_dof_adr] - self.data.qfrc_passive[self.hand_dof_adr]
                    - self.data.qfrc_applied[self.hand_dof_adr])
            self.data.ctrl[:] = np.clip(self.online_servo_target + bias/kp,
                self.hand_joint_ranges[:, 0], self.hand_joint_ranges[:, 1])
            mujoco.mj_step(self.model, self.data)

    def online_ik_posture_reference(self, ik, *, closing, loaded, reach_pos, open_posture, lift_offset=None):
        """Keep the planned IK pose fixed until a verified grasp can move."""
        reference = (1-float(closing))*np.asarray(open_posture) + float(closing)*ik.qpos
        if lift_offset is not None:
            reference[:3] += np.asarray(lift_offset)
        elif loaded:
            reference[:3] += np.asarray(reach_pos)-self.initial_object_pos
        return reference

    def transport_online_posture(self, posture, source_pos, source_rot, pos, rot):
        """Move the approach attractor with the observed object, without IK."""
        palm, palm_rot, _ = self.forward_kinematics(posture)
        target_palm = pos + rot @ source_rot.T @ (palm-source_pos)
        target_rot = rot @ source_rot.T @ palm_rot
        result = np.asarray(posture).copy()
        result[3:6] = self.base_orientation_euler_from_target_palm_rot(target_rot, result[3:6])
        moved_palm, _, _ = self.forward_kinematics(result)
        result[:3] += target_palm-moved_palm
        return np.clip(result, self.hand_joint_ranges[:, 0], self.hand_joint_ranges[:, 1])

    def replan_online_grasp(self, ik, open_posture, points, inward, source_pos, source_rot,
                            goal, pos, quat, rot, tips, mass, external):
        """Search live patches once on loss; retain only a reachable assignment.

        IK supplies a bounded initial guess, never a replacement for the
        contact MPC. A rejected patch search keeps the previous assignment
        transported to the current object pose.
        """
        old_ids = self.optimizer.active_ids.copy()
        self.ik_reference_qpos[self.object_qpos_adr:self.object_qpos_adr+7] = np.r_[pos, quat]
        grasp_q = self.transport_online_posture(ik.qpos, source_pos, source_rot, pos, rot)
        opened_q = self.transport_online_posture(open_posture, source_pos, source_rot, pos, rot)
        palm, palm_rot, tip_positions = self.forward_kinematics(grasp_q)
        moved_ik = replace(ik, qpos=grasp_q, target_palm_pos_world=palm,
            target_palm_rot_world=palm_rot, solved_tip_positions_world=tip_positions,
            assigned_targets_world=pos+(points-(self.args.online_tip_radius-self.args.squeeze_depth)*inward)@rot.T)
        dynamics = dict(tips_world=tips, object_inertia=mass, tau_o=external,
            v_obj=self.data.qvel[self.object_dof_adr:self.object_dof_adr+6],
            prediction_dt=self.args.lambda_prediction_dt)
        accepted = False
        try:
            candidate_points, candidate_normals, _, _, _ = self.optimizer.update_contacts(
                goal, np.r_[pos, quat], freeze_contact_pair=False, lock_contact_pair=False, **dynamics)
            targets = pos+(candidate_points-max(self.args.online_tip_radius-self.args.squeeze_depth, .001)*candidate_normals)@rot.T
            # Preserve fingertip order: an Allegro thumb cannot be exchanged
            # with an arbitrary finger as in an independent-pad controller.
            self._ik_startup_deadline = time.perf_counter()+self.args.regrasp_ik_time_budget
            payload = self.solve_single_ik_candidate(grasp_q, palm, palm_rot, targets, candidate_normals@rot.T)
            _, q, mean_error, max_error, _, _, solved_tips, self_depth = payload
            if (mean_error <= self.args.online_ik_tip_tolerance
                    and max_error <= 1.5*self.args.online_ik_tip_tolerance and self_depth <= 1e-4):
                candidate_ik = replace(moved_ik, qpos=q, mean_tip_error=mean_error,
                    max_tip_error=max_error, assigned_targets_world=targets,
                    solved_tip_positions_world=solved_tips, max_self_penetration=self_depth)
                opened_q, _ = self.solve_online_pregrasp(candidate_ik, candidate_points, candidate_normals, pos, rot)
                moved_ik, points, inward = candidate_ik, candidate_points, candidate_normals
                accepted = True
        except (RuntimeError, FloatingPointError) as exc:
            print(f"[regrasp] keep transported patches: {exc}", flush=True)
        finally:
            self._ik_startup_deadline = np.inf
        selected_ids = self.optimizer.active_ids.copy() if accepted else old_ids
        _, _, _, _, info = self.optimizer.update_fixed_contacts(selected_ids, goal, np.r_[pos, quat], **dynamics)
        # Remove the stale squeeze/lift target and old trajectory. Actual
        # object/hand positions and velocities remain the measured state.
        self.online_servo_target = self.get_hand_qpos().copy()
        self.mpc_param.sol_guess_ = None
        self.mpc.pressure_floor = None
        self.online_local_points = np.full((len(points), 3), np.nan)
        self.online_local_normals = np.full((len(points), 3), np.nan)
        self.online_sticking_anchors = np.full((len(points), 3), np.nan)
        self.cached_palm_target_local_pos = rot.T @ (moved_ik.target_palm_pos_world-pos)
        self.cached_palm_target_local_rot = rot.T @ moved_ik.target_palm_rot_world
        return moved_ik, opened_q, points, inward, info, accepted

    def solve_online_pregrasp(self, ik, points, inward, pos, rotation):
        """Open all pads outside the frozen patches before descending."""
        clearance = self.args.online_pregrasp_clearance
        targets = pos + (points-(self.args.online_tip_radius+clearance)*inward)@rotation.T
        self._ik_startup_deadline = min(getattr(self, "_ik_startup_deadline", np.inf), time.perf_counter()+1.)
        try:
            result = self.solve_single_ik_candidate(ik.qpos, ik.target_palm_pos_world,
                ik.target_palm_rot_world, targets, inward@rotation.T)
        finally:
            self._ik_startup_deadline = np.inf
        _, qpos, mean_error, max_error, _, _, _, self_depth = result
        self.forward_kinematics(qpos)
        gaps = [mujoco.mj_geomDistance(self.model, self.solve_data,
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, TIP_COLLISION_GEOM_NAMES[name]),
            self.object_geom_id, 1., None) for name in self.active_finger_names]
        feasible = mean_error <= .004 and max_error <= .006 and self_depth <= 1e-4 and min(gaps) >= .003
        if not feasible:
            raise RuntimeError(f"No collision-free pregrasp IK: mean/max tip error "
                f"{mean_error:.4f}/{max_error:.4f} m; minimum pad clearance {min(gaps):.4f} m. "
                "Adjust --online-pregrasp-clearance; a colliding approach posture will not be executed.")
        return qpos, float(min(gaps))

    def audit_online_grasp(self, points, inward, *, offset, external_wrench=None, grasp_active=False):
        """Verify measured pad contacts, their planned patches and force closure."""
        pos, _, rotation = self.get_object_pose()
        tips, _ = self.online_tip_kinematics()
        target = pos + (points-offset*inward) @ rotation.T
        planned = pos + points @ rotation.T
        summary = self.get_fingertip_contact_summary()
        entries, actual_points, actual_normals, normal_loads = [], [], [], []
        threshold = self.args.contact_force_threshold if self.args.lift_contact_force_threshold is None else self.args.lift_contact_force_threshold
        for i, name in enumerate(self.active_finger_names):
            geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, TIP_COLLISION_GEOM_NAMES[name])
            contacts = [c for c in self.data.contact if {int(c.geom1), int(c.geom2)} == {geom, self.object_geom_id}]
            entry = dict(finger=name, normal_force=summary[name],
                touching=bool(contacts and summary[name] > threshold),
                tip_error=float(np.linalg.norm(tips[i]-target[i])),
                planned_point=planned[i].tolist(), tip_target=target[i].tolist(),
                tip_position=tips[i].tolist(), height_error=float(tips[i, 2]-target[i, 2]),
                geometric_gap=float(mujoco.mj_geomDistance(self.model, self.data, geom, self.object_geom_id, 1., None)),
                actual_point=None, actual_normal=None, patch_error=None)
            if contacts:
                contact = min(contacts, key=lambda c: c.dist)
                normal = np.asarray(contact.frame[:3]).copy()
                if int(contact.geom1) == self.object_geom_id:
                    normal *= -1
                point = np.asarray(contact.pos).copy()
                entry.update(actual_point=point.tolist(), actual_normal=normal.tolist(),
                    patch_error=float(np.linalg.norm(point-planned[i])), gap=float(contact.dist))
                if entry["touching"]:
                    actual_points.append((point-pos) @ rotation)
                    actual_normals.append(normal @ rotation)
                    normal_loads.append(summary[name])
            entries.append(entry)
        if external_wrench is not None:
            from planning.grasp_diagnostics import object_contact_wrenches
            measured_wrenches, _ = object_contact_wrenches(self.model, self.data, self.object_body_id,
                self.fingertip_geom_id_to_name, self.hand_collision_geom_ids)
            external_wrench = np.asarray(external_wrench)+measured_wrenches['other_hand']
        actual_points = np.asarray(actual_points).reshape(-1, 3)
        actual_normals = np.asarray(actual_normals).reshape(-1, 3)
        local_external = None if external_wrench is None else np.r_[
            rotation.T@external_wrench[:3], rotation.T@external_wrench[3:]]
        certificate = verify_grasp(actual_points,
            actual_normals, self.args.object_friction, external_wrench=local_external,
            normal_force_limits=normal_loads)
        minimum = max(self.args.contact_force_threshold,
            (.375 if grasp_active else .75)*self.args.online_min_normal_force)
        supporting = np.asarray(normal_loads) >= minimum
        support_certificate = verify_grasp(actual_points[supporting], actual_normals[supporting],
            self.args.object_friction, external_wrench=local_external,
            normal_force_limits=np.asarray(normal_loads)[supporting])
        planned_reached = all(e["touching"] and e["tip_error"] <= self.args.online_handoff_tip_tolerance
            and e["patch_error"] <= self.args.online_handoff_patch_tolerance for e in entries)
        self_collision = []
        for index, contact in enumerate(self.data.contact):
            if int(contact.geom1) in self.hand_collision_geom_ids and int(contact.geom2) in self.hand_collision_geom_ids:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, index, force)
                self_collision.append(dict(geom1=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                    geom2=mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
                    gap=float(contact.dist), normal_force=float(abs(force[0]))))
        return dict(fingers=entries, self_collisions=self_collision, actual_force_closure=certificate,
            supporting_force_closure=support_certificate, supporting_contact_count=int(supporting.sum()),
            all_planned_contacts_reached=bool(planned_reached),
            contact_count=sum(e["touching"] for e in entries))

    def online_lift_readiness(self, audit, contact_count, *, grasp_active=False):
        """Authorize lift from physical load, independently of startup IK patches."""
        certificate = audit.get('supporting_force_closure', audit['actual_force_closure'])
        checks = dict(contacts=contact_count >= self.args.lift_contact_fingers,
            force_closure=bool(certificate['force_closure']),
            load_support=bool(certificate['load_feasible']))
        if audit.get('fingers'):
            minimum = max(getattr(self.args, 'contact_force_threshold', .01),
                (.375 if grasp_active else .75)*getattr(self.args, 'online_min_normal_force', .2))
            checks['normal_preload'] = sum(e['touching'] and e['normal_force'] >= minimum
                for e in audit['fingers']) >= self.args.lift_contact_fingers
        return all(checks.values()), [name for name, passed in checks.items() if not passed]

    def execute_online_action(self, action, hand_posture, points, inward, loaded, desired_force_local=None):
        if self.args.online_execution == "position":
            self.execute_online_joint_target(self.online_servo_target + self.mpc.model.effective_command(action))
            return
        # Same capped Cartesian spring/damper as bigrasp, applied through
        # one stacked Jacobian because the Allegro tips share a palm.
        from models.bimanual_rigid_contact import online_tip_velocity_reference
        _, initial_jacobian = self.online_tip_kinematics()
        increments = (initial_jacobian @ action).reshape(-1, 3)
        args = self.args
        for _ in range(max(args.mj_steps_per_command, 1)):
            mujoco.mj_forward(self.model, self.data)
            tips, j = self.online_tip_kinematics()
            pos, _, rotation = self.get_object_pose()
            normals = inward @ rotation.T
            world_points = pos + points @ rotation.T
            gaps = -np.einsum("ij,ij->i", tips-world_points, normals) - args.online_tip_radius
            summary = self.get_fingertip_contact_summary()
            touching = np.array([summary[name] > args.contact_force_threshold for name in self.active_finger_names])
            velocity = (j @ self.data.qvel[self.hand_dof_adr]).reshape(-1, 3)
            forces = []
            for index, increment in enumerate(increments):
                gap = min(gaps[index], 0.) if touching[index] else max(gaps[index], 0.)
                reference = online_tip_velocity_reference(increment, gap, loaded,
                    self.command_dt, args.online_approach_speed)
                force = args.online_task_stiffness * increment - args.online_task_damping * (velocity[index]-reference)
                if loaded and desired_force_local is not None:
                    force += rotation @ desired_force_local[index]
                limit = args.online_force_limit
                if not loaded and touching[index] and not touching.all() and force @ normals[index] > 0:
                    limit = min(limit, args.online_unilateral_force_limit)
                force *= min(1., limit/max(np.linalg.norm(force), 1e-10))
                forces.append(force)
            mass = np.empty((self.model.nv, self.model.nv))
            try:
                mujoco.mj_fullM(self.model, self.data, mass)
            except TypeError:
                mujoco.mj_fullM(self.model, mass, self.data.qM)
            hand_mass = mass[np.ix_(self.hand_dof_adr, self.hand_dof_adr)]
            inverse_jt = np.linalg.solve(hand_mass, j.T)
            projector = np.eye(self.model.nu) - j.T @ np.linalg.pinv(j @ inverse_jt, rcond=1e-8) @ inverse_jt.T
            q, dq = self.data.qpos[self.hand_qpos_adr], self.data.qvel[self.hand_dof_adr]
            attraction = 0. if loaded or touching.any() else float(
                np.clip((np.min(gaps)-.002)/.018, 0., 1.)**2)
            posture = attraction * args.online_nullspace_stiffness * (hand_posture-q)
            posture -= 2*np.sqrt(args.online_nullspace_stiffness) * dq
            torque = j.T @ np.asarray(forces).reshape(-1) + projector @ posture + (self.data.qfrc_bias[self.hand_dof_adr] - self.data.qfrc_passive[self.hand_dof_adr]
                - self.data.qfrc_applied[self.hand_dof_adr])
            kp = self.model.actuator_gainprm[:, 0]
            damping = -self.model.actuator_biasprm[:, 2]
            self.data.ctrl[:] = np.clip(q + (torque + damping*dq)/kp,
                self.hand_joint_ranges[:, 0], self.hand_joint_ranges[:, 1])
            self.online_servo_target = self.data.ctrl.copy() - (
                self.data.qfrc_bias[self.hand_dof_adr] - self.data.qfrc_passive[self.hand_dof_adr]
                - self.data.qfrc_applied[self.hand_dof_adr])/kp
            mujoco.mj_step(self.model, self.data)

    def run_online(self):
        args = self.args
        startup_times = self.startup_timings.copy()
        goal = np.r_[self.initial_object_pos + [0., 0., args.lift_height], self.initial_object_quat]
        pos, quat, rot = self.get_object_pose()
        mass, external = self.lambda_object_dynamics(rot)
        tips, _ = self.online_tip_kinematics()
        started = time.perf_counter()
        print("[startup] selecting contact anchors with shared Lambda LM", flush=True)
        points, inward, _, _, info = self.optimizer.initialize_offline(goal, np.r_[pos, quat],
            tips_world=tips, object_inertia=mass, tau_o=external,
            v_obj=self.data.qvel[self.object_dof_adr:self.object_dof_adr+6],
            prediction_dt=args.lambda_prediction_dt)
        startup_times["contact_selection_seconds"] = time.perf_counter() - started
        print(f"[startup] contact selection: {startup_times['contact_selection_seconds']:.3f}s", flush=True)
        offset = max(args.online_tip_radius - args.squeeze_depth, .001)
        # Lambda ranks object dynamics first. Verify a bounded shortlist
        # against the hand's coupled IK before committing an assignment.
        candidate_ids = self.optimizer.last_batch_ids
        quality = self.optimizer.last_candidate_quality
        geometry = self.optimizer.last_candidate_geometry
        preferred = np.flatnonzero(quality)
        preferred = preferred[np.argsort(geometry[preferred])]
        admissible = self.optimizer.last_candidate_valid & (self.optimizer.last_candidate_reductions > 0)
        alternatives = np.flatnonzero(admissible & ~quality)
        alternatives = alternatives[np.argsort(geometry[alternatives])]
        rows = np.r_[preferred, alternatives]
        side_axis = self.optimizer.surface.body_side_axis
        if side_axis is not None:
            signs = np.sign(self.optimizer.normal[candidate_ids] @ side_axis)
            counts = np.sum(signs > 0, axis=1)
            rows = np.array([row for row in rows if counts[row] in (1, len(self.active_finger_names)-1)], int)
        rows = rows[:args.ik_contact_candidates]
        started = time.perf_counter()
        self._ik_startup_deadline = started + args.ik_startup_time_budget
        print(f"[startup] hand IK: at most {len(rows)} sets, {args.ik_startup_time_budget:.1f}s budget", flush=True)
        best = None
        checked_sets = 0
        for row in rows:
            if best is not None and time.perf_counter() >= self._ik_startup_deadline:
                break
            candidate_points = self.optimizer.sample_point[candidate_ids[row]]
            candidate_inward = self.optimizer.normal[candidate_ids[row]]
            targets = pos + (candidate_points - offset * candidate_inward) @ rot.T
            side_signs = None if side_axis is None else np.sign(candidate_inward @ side_axis)
            candidate_ik = self.solve_grasp_ik(targets, contact_side_signs=side_signs, contact_normals_world=candidate_inward @ rot.T)
            checked_sets += 1
            if best is None or (candidate_ik.success and not best[1].success) or (candidate_ik.success == best[1].success and candidate_ik.score < best[0]):
                best = (candidate_ik.score, candidate_ik, candidate_ids[row].copy())
            if candidate_ik.success:
                break
        budget_exhausted = time.perf_counter() >= self._ik_startup_deadline
        self._ik_startup_deadline = np.inf
        startup_times["hand_ik_seconds"] = time.perf_counter() - started
        print(f"[startup] hand IK: {startup_times['hand_ik_seconds']:.3f}s", flush=True)
        if best is None:
            raise RuntimeError("Lambda did not provide a contact set with one thumb opposing the finger bank")
        _, ik, selected_ids = best
        if not ik.success:
            tolerance = args.online_ik_tip_tolerance
            blockers = []
            if ik.mean_tip_error > tolerance:
                blockers.append('mean_tip_error')
            if ik.max_tip_error > 1.5*tolerance:
                blockers.append('max_tip_error')
            if ik.max_self_penetration > 1e-4:
                blockers.append('self_collision')
            raise RuntimeError(f"No accepted approach IK seed: "
                f"mean/max tip error={ik.mean_tip_error*1000:.3f}/{ik.max_tip_error*1000:.3f} mm "
                f"(limits {tolerance*1000:.3f}/{1.5*tolerance*1000:.3f} mm); "
                f"self penetration={ik.max_self_penetration*1000:.3f} mm (limit 0.100 mm); "
                f"blocked_by={','.join(blockers) or 'invalid_result'}; "
                f"checked_sets={checked_sets}/{len(rows)}; budget_exhausted={budget_exhausted}. "
                "Try more IK time/candidates if the budget was exhausted; colliding seeds are rejected.")
        # IK seeds the pregrasp and initial contact assignment. The MPC
        # aligns measured pads to object-frame targets every cycle.
        ids = selected_ids[list(ik.permutation)]
        points = self.optimizer.sample_point[ids]
        inward = self.optimizer.normal[ids]
        planned_certificate = verify_grasp(points, inward, args.object_friction,
            external_wrench=np.r_[rot.T@external[:3], rot.T@external[3:]], force_limit=args.lambda_force_limit)
        if not planned_certificate["force_closure"] or not planned_certificate["load_feasible"]:
            raise RuntimeError("Selected IK contact set fails force closure or bounded gravity support")
        print(f"[startup] planned force closure: {planned_certificate['force_closure']}; "
              f"rank={planned_certificate['wrench_rank']}; bounded gravity support={planned_certificate['load_feasible']}", flush=True)
        started = time.perf_counter()
        open_posture, clearance = self.solve_online_pregrasp(ik, points, inward, pos, rot)
        initial_ik, initial_open_posture = ik, open_posture.copy()
        startup_times["pregrasp_ik_seconds"] = time.perf_counter()-started
        print(f"[startup] pregrasp IK: {startup_times['pregrasp_ik_seconds']:.3f}s; "
            f"minimum pad clearance={clearance*1000:.1f} mm", flush=True)
        initial = open_posture.copy()
        initial[2] += args.approach_height
        self.reset(self.initial_object_pos, self.initial_object_quat, initial)
        self.cached_contact_points_local = points.copy()
        self.cached_target_points_local = points - offset * inward
        self.cached_palm_target_local_pos = rot.T @ (ik.target_palm_pos_world - pos)
        self.cached_palm_target_local_rot = rot.T @ ik.target_palm_rot_world
        self._ensure_screenshot_recorder()
        self._ensure_video_recorder()
        print(f"Online Allegro MPC: n_EE={len(ids)} state={self.mpc_param.n_qpos_} "
              f"action={self.mpc_param.n_cmd_} objective=pad_contact_and_object_pose "
              f"solver={self.mpc.planner_solver_} dt={self.command_dt:.3f}s")
        print(f"Initial IK mean/max tip error: {ik.mean_tip_error:.4f}/{ik.max_tip_error:.4f} m")
        print(f"Initial IK max hand self-penetration: {ik.max_self_penetration*1000:.3f} mm")
        print(f"Object goal: {goal[:3].tolist()}; lift={args.lift_height:.3f} m; "
              "confirmed physical force closure and load support enable contact transport", flush=True)
        stable, hold, released, closing_started, release_streak = 0, 0, False, False, 0
        best_height, best_contacts, status = 0., 0, "N/A"
        reach_pos = self.initial_object_pos.copy()
        lift_offset = np.zeros(3)
        records = []
        ik_phase = "ik_approach" if args.online_guidance == "ik-mpc" else None
        ik_reference = initial.copy()
        handoff_step = None
        ik_near_streak = 0
        approach_origin, approach_rot = pos.copy(), rot.copy()
        recovery = OnlineGraspRecovery(approach_origin.copy(), approach_rot.copy(), self.command_dt,
            args.regrasp_position_threshold, args.regrasp_rotation_threshold,
            args.regrasp_cooldown, args.regrasp_stall_timeout,
            opening_timeout=args.regrasp_clearance_timeout)
        mass, external = self.lambda_object_dynamics(rot)
        initial_tips, _ = self.online_tip_kinematics()
        self.optimizer.update_fixed_contacts(ids, goal, np.r_[pos, quat], tips_world=initial_tips,
            object_inertia=mass, tau_o=external, prediction_dt=args.lambda_prediction_dt)
        max_steps = args.max_steps or sum((args.hover_steps, args.approach_steps, args.squeeze_steps,
            args.lift_steps, args.hold_steps, args.max_lift_wait_steps))
        for step in range(max_steps):
            if self.viewer is not None and not self.viewer.is_running():
                break
            mujoco.mj_forward(self.model, self.data)
            pos, quat, rot = self.get_object_pose()
            tips, tip_jac = self.online_tip_kinematics()
            mass, external = self.lambda_object_dynamics(rot)
            summary = self.get_fingertip_contact_summary()
            threshold = args.contact_force_threshold if args.lift_contact_force_threshold is None else args.lift_contact_force_threshold
            count = sum(summary[name] > threshold for name in self.active_finger_names)
            best_contacts = max(best_contacts, count)
            local_updates = []
            self.online_local_repair_feasible = False
            if not recovery.opening and ik_phase is None:
                phi, jac = self.detect_mpc_contacts()
                points, inward, local_updates = self.refine_online_contact_patches(
                    points, inward, phi, tips, tip_jac, external)
            audit = self.audit_online_grasp(points, inward, offset=offset, external_wrench=external,
                grasp_active=released)
            safe_grasp, lift_blockers = self.online_lift_readiness(audit, count, grasp_active=released)
            if recovery.opening:
                safe_grasp = False
                lift_blockers.append('regrasp_clearance')
            was_loaded = released
            stable = stable + 1 if safe_grasp else 0
            if stable >= args.lift_contact_stable_steps:
                released = True
            release_streak = release_streak + 1 if not safe_grasp else 0
            if release_streak >= args.grasp_release_stable_steps:
                released = False
            replan_reason = recovery.observe(pos, rot, contacts=count,
                was_loaded=was_loaded, loaded=released, closing=closing_started,
                repair_available=self.online_local_repair_feasible)
            patches_replanned = False
            if replan_reason is not None:
                local_updates = []
                ik, open_posture, points, inward, info, patches_replanned = self.replan_online_grasp(
                    ik, open_posture, points, inward, approach_origin, approach_rot,
                    goal, pos, quat, rot, tips, mass, external)
                ids = info["contact_indices"].copy()
                approach_origin, approach_rot = pos.copy(), rot.copy()
                stable = hold = release_streak = 0
                released = closing_started = False
                lift_offset[:] = 0.
                ik_phase = None  # All subsequent commands come from contact MPC.
                audit = self.audit_online_grasp(points, inward, offset=offset, external_wrench=external)
                print(f"[regrasp] step={step} reason={replan_reason} attempt={recovery.replan_count} "
                      f"object={pos.tolist()} new_patches={patches_replanned}", flush=True)
            # Execute the one-time coupled IK before asking local MPC to
            # stabilize contact. Absolute servo references advance at bounded
            # speed, avoiding the persistent lag of q_measured + tiny u.
            if ik_phase is not None:
                q_target = open_posture.copy()
                limits = np.r_[np.full(3, args.online_ik_base_speed), np.full(3, args.online_ik_joint_speed),
                               np.full(self.model.nu-6, args.online_ik_joint_speed)]*self.command_dt
                ik_reference += np.clip(q_target-ik_reference, -limits, limits)
                measured_q = self.data.qpos[self.hand_qpos_adr]
                near_ik = (np.linalg.norm(measured_q[:3]-q_target[:3]) <= .015
                           and np.max(np.abs(measured_q[6:]-q_target[6:])) <= args.online_ik_handoff_joint_tolerance)
                ik_near_streak = ik_near_streak+1 if near_ik else 0
                min_pad_gap = min(e["geometric_gap"] for e in audit["fingers"])
                if ik_near_streak >= args.lift_contact_stable_steps or min_pad_gap <= .02:
                    print(f"[handoff] pregrasp IK -> contact MPC at step {step}", flush=True)
                    handoff_step, ik_phase = step, None
                    self.mpc_param.sol_guess_ = None
                else:
                    before = self.data.qpos[self.hand_qpos_adr].copy()
                    self.execute_online_joint_target(ik_reference)
                    self.refresh_markers()
                    self.sync_viewer()
                    new_pos, _, _ = self.get_object_pose()
                    best_height = max(best_height, float(new_pos[2]-self.initial_object_pos[2]))
                    records.append(dict(step=step, phase=ik_phase, contacts=count,
                        tip_error=float(np.mean([e["tip_error"] for e in audit["fingers"]])),
                        object_pos=new_pos.tolist(), status="IK_SERVO", loaded=False,
                        action=(ik_reference-before).tolist(), contact_indices=ids.tolist(),
                        hand_q=self.get_hand_qpos().tolist(), ik_target=q_target.tolist(),
                        contact_audit=audit, contact_forces=summary))
                    if step % max(args.log_every, 1) == 0:
                        print(f"step={step:04d} phase={ik_phase} contacts={count}/{len(ids)} "
                              f"max_tip_error={max(e['tip_error'] for e in audit['fingers']):.4f} "
                              f"actual_force_closure={audit['actual_force_closure']['force_closure']}", flush=True)
                    continue
            loaded = released
            if loaded:
                # Advance an independent lift trajectory. Following
                # object_measured + one tiny step stalled when pad pressure
                # compressed the object below its initial height, and reset
                # the hand downwards on a single failed contact check.
                delta = goal[:3] - (self.initial_object_pos+lift_offset)
                lift_offset += delta * min(1., args.online_lift_speed*self.command_dt/max(np.linalg.norm(delta), 1e-10))
                reach_pos = self.initial_object_pos+lift_offset
            else:
                reach_pos = pos.copy()
            lambda_goal = goal.copy()
            delta_goal = goal[:3]-pos
            lambda_goal[:3] = pos + delta_goal * min(1., args.online_lift_speed*args.lambda_prediction_dt/max(np.linalg.norm(delta_goal),1e-10))
            points, inward, _, _, info = self.optimizer.update_contacts(lambda_goal, np.r_[pos, quat],
                tips_world=tips, freeze_contact_pair=args.freeze_contact_set,
                lock_contact_pair=loaded, object_inertia=mass, tau_o=external,
                v_obj=self.data.qvel[self.object_dof_adr:self.object_dof_adr+6],
                prediction_dt=args.lambda_prediction_dt)
            ids = info["contact_indices"].copy()
            # Lambda's compressive load is converted to pad indentation;
            # tangential object motion remains the contact-aware MPC's job.
            # Freeze the IK pad-centre offset as well as the surface IDs.
            # Changing indentation after IK changed the planned geometry.
            if recovery.opening:
                open_anchors = points-(args.online_tip_radius+args.online_pregrasp_clearance)*inward
                open_targets = pos+open_anchors@rot.T
                if (min(e['geometric_gap'] for e in audit['fingers']) >= .003
                        and np.max(np.linalg.norm(tips-open_targets, axis=1)) <= .01):
                    recovery.opening = False
                    recovery.touched = False
                    print(f"[regrasp] step={step} pads clear; resume contact MPC", flush=True)
            anchors = points - (args.online_tip_radius+args.online_pregrasp_clearance if recovery.opening else offset) * inward
            self.cached_contact_points_local = points.copy()
            self.cached_target_points_local = anchors.copy()
            measured_hand = self.get_hand_qpos()
            live_open_posture = self.transport_online_posture(open_posture, approach_origin, approach_rot, pos, rot)
            live_grasp_q = self.transport_online_posture(ik.qpos, approach_origin, approach_rot, pos, rot)
            near_pregrasp = (np.linalg.norm(measured_hand[:3]-live_open_posture[:3]) <= .005
                and np.max(np.abs(measured_hand[3:]-live_open_posture[3:])) <= args.online_ik_handoff_joint_tolerance)
            closing_started = not recovery.opening and (closing_started or near_pregrasp)
            closing = float(closing_started or released)
            hand_posture = self.online_ik_posture_reference(replace(ik, qpos=live_grasp_q), closing=closing, loaded=loaded,
                reach_pos=reach_pos, open_posture=live_open_posture,
                lift_offset=lift_offset if loaded else np.zeros(3))
            state = self.get_mpc_state()
            phi, jac = self.detect_mpc_contacts()
            if not recovery.opening:
                anchors = points-offset*inward
                local = np.isfinite(self.online_local_points).all(axis=1)
                anchors[local] = points[local]-(args.online_tip_radius-min(args.squeeze_depth, .0005))*inward[local]
                # Existing physical pads retain their current contact region;
                # only the normal pressure controller changes indentation.
                actual = np.array([summary[name] > args.contact_force_threshold
                                   for name in self.active_finger_names])
                self.online_sticking_anchors[~actual] = np.nan
                new_contact = actual & ~np.isfinite(self.online_sticking_anchors).all(axis=1)
                self.online_sticking_anchors[new_contact] = (tips[new_contact]-pos)@rot
                anchors[actual] = self.online_sticking_anchors[actual]
                self.cached_contact_points_local, self.cached_target_points_local = points.copy(), anchors.copy()
                for update in local_updates:
                    print(f"[local-contact] step={step} finger={update['finger']} "
                          f"gap={update['gap']*1000:.2f}mm; nearby force-closed patch adopted", flush=True)
            info = self.update_online_contact_forces(points, inward, info, lambda_goal,
                np.r_[pos, quat], mass, external, phi)
            if recovery.opening:
                # Clear partial contacts before chasing the object. Otherwise
                # one touching pad suppresses IK attraction while the other
                # pads remain far away and the hand just pushes/slides.
                info['force_vectors_local'] = np.zeros_like(info['force_vectors_local'])
            first_mpc = "first_mpc_seconds" not in startup_times
            if first_mpc:
                started = time.perf_counter()
                print("[startup] solving the first MPC command", flush=True)
            mpc_goal = goal.copy()
            measured_force = None
            if args.online_force_feedback or args.online_pressure_control:
                from planning.grasp_diagnostics import object_contact_wrenches
                _, contacts = object_contact_wrenches(self.model, self.data, self.object_body_id,
                    self.fingertip_geom_id_to_name, self.hand_collision_geom_ids)
                measured_force = np.zeros((len(self.active_finger_names), 3))
                for contact in contacts:
                    if contact['group'] != 'pads':
                        continue
                    name = self.fingertip_geom_id_to_name[contact['geom']]
                    if name in self.active_finger_names:
                        measured_force[self.active_finger_names.index(name)] += contact['force']
            result = self.mpc.plan_contacts(state, phi, jac, tips=tips, tip_jacobian=tip_jac,
                anchors=anchors, goal=mpc_goal, hand_posture=hand_posture, loaded=loaded,
                approach_origin=approach_origin,
                object_inertia=mass, external_wrench=external, hand_mass_diagonal=self.hand_mass_diagonal,
                desired_force_local=info["force_vectors_local"], measured_force_world=measured_force,
                pressure_control_allowed=not recovery.opening,
                other_hand_wrench_world=info['other_hand_wrench_world'],
                # A preload target is not a contact-mode switch. As in
                # bigrasp, keep made-contact physics while the measured grasp
                # remains load-bearing; every pad still has its actual force
                # cap and separates unilaterally in the prediction.
                grasp_established=(audit['actual_force_closure']['force_closure']
                    and bool(audit['actual_force_closure']['load_feasible']) and count >= 3),
                sol_guess=self.mpc_param.sol_guess_)
            if first_mpc:
                startup_times["first_mpc_seconds"] = time.perf_counter() - started
                print(f"[startup] first MPC solve: {startup_times['first_mpc_seconds']:.3f}s", flush=True)
            self.mpc_param.sol_guess_ = result["sol_guess"]
            status = str(result["solve_status"])
            predicted_pad_forces = np.asarray(self.mpc.model.contact_force_fn(state, result["action"],
                phi, jac, self.mpc.model.runtime_parameters(self.mpc_param.model_params))).reshape(-1, 3)
            ideal_pose_error = None
            if abs(args.lambda_prediction_dt-self.command_dt) < 1e-12:
                ideal_other = info['other_hand_wrench_world'] if info['force_allocation_feasible'] else np.zeros(6)
                ideal_pose, _ = self.mpc.model.applied_force_prediction_fn(state[:7],
                    state[self.mpc.model.pose_size:self.mpc.model.pose_size+6], info["force_points_local"].T,
                    np.asarray(info["force_vectors_local"]).T, self.mpc.model.object_inverse,
                    self.mpc.model.external_wrench+np.r_[ideal_other[:3], rot.T@ideal_other[3:]])
                ideal_pose_error = float(np.linalg.norm(np.asarray(ideal_pose).reshape(-1)[:3]-np.asarray(info["x_plus_opt"])[:3]))
            self.execute_online_action(result["action"], hand_posture, info['force_points_local'],
                info['force_normals_local'], loaded, info["force_vectors_local"])
            self.refresh_markers()
            self.sync_viewer()
            new_pos, _, _ = self.get_object_pose()
            best_height = max(best_height, float(new_pos[2] - self.initial_object_pos[2]))
            tip_error = float(np.mean(np.linalg.norm(tips - (pos + anchors @ rot.T), axis=1)))
            phase = "hold" if loaded and np.linalg.norm(new_pos-goal[:3]) <= args.goal_tolerance else ("lift" if loaded else "regrasp_open" if recovery.opening else "regrasp" if recovery.recovering else "approach")
            hold = hold + 1 if phase == "hold" else 0
            records.append(dict(step=step, phase=phase, contacts=count, tip_error=tip_error,
                object_pos=new_pos.tolist(), status=status, action=result["applied_action"].tolist(),
                base_q=self.get_base_qpos().tolist(), predicted_object_pos=np.asarray(result["rollout_q"])[0,:3].tolist(),
                contact_forces=summary, closing=closing, loaded=loaded))
            records[-1].update(contact_indices=ids.tolist(), contact_audit=audit,
                object_target_pose=goal.tolist(), mpc_object_target_pose=mpc_goal.tolist(),
                lift_ready=safe_grasp, lift_blockers=lift_blockers, lift_confirm_streak=stable,
                hand_ik_target=hand_posture.tolist(), hand_q=self.get_hand_qpos().tolist(),
                finger_ik_error=float(np.linalg.norm(self.get_hand_qpos()[6:]-hand_posture[6:])),
                base_ik_position_error=float(np.linalg.norm(self.get_hand_qpos()[:3]-hand_posture[:3])),
                lambda_forces_local=np.asarray(info["force_vectors_local"]).tolist(),
                force_points_local=info["force_points_local"].tolist(),
                force_normals_local=info["force_normals_local"].tolist(),
                measured_force_contacts=info["measured_force_contacts"].tolist(),
                local_contact_updates=local_updates,
                pad_targets_local=anchors.tolist(),
                object_quat=self.get_object_pose()[1].tolist(),
                mpc_raw_action=result['action'].tolist(),
                live_force_closure_certificate=info["live_force_closure_certificate"],
                force_allocation_feasible=info['force_allocation_feasible'],
                allocated_wrench_available=info['allocated_wrench_available'],
                pressure_reference=None if self.mpc.pressure_reference is None
                    else self.mpc.pressure_reference.tolist(),
                other_hand_wrench_world=info['other_hand_wrench_world'].tolist(),
                allocated_wrench_local=None if info['allocated_wrench_local'] is None
                    else info['allocated_wrench_local'].tolist(),
                lambda_predicted_pose=np.asarray(info["x_plus_opt"]).tolist(),
                ideal_force_object_prediction_error=ideal_pose_error,
                mpc_contact_forces_world=predicted_pad_forces.tolist(),
                mpc_object_prediction_error=float(np.linalg.norm(new_pos-np.asarray(result["rollout_q"])[0,:3])),
                servo_target=self.online_servo_target.tolist(),
                selected_control=result.get("selected_control", result.get("fallback_control", "mpc")),
                nonlinear_step_scale=result.get("nonlinear_step_scale"))
            records[-1].update(replan_reason=replan_reason, replan_count=recovery.replan_count,
                patches_replanned=patches_replanned, approach_origin=approach_origin.tolist())
            if step % max(args.log_every, 1) == 0:
                pad_status = " ".join(f"{e['finger']}={e['geometric_gap']*1000:+.1f}mm/{e['normal_force']:.2f}N"
                    for e in audit["fingers"])
                print(f"step={step:04d} phase={phase} contacts={count}/{len(ids)} "
                      f"tip_error={tip_error:.4f} height={new_pos[2]-self.initial_object_pos[2]:.4f} "
                      f"target_z={mpc_goal[2]:.4f} lift_gate={','.join(lift_blockers) or 'ready'} "
                      f"{pad_status} status={status}")
            if hold >= max(args.hold_steps, 1):
                break
        final_pos, final_quat, _ = self.get_object_pose()
        success = hold >= max(args.hold_steps, 1)
        report = dict(success=success, controller="online", num_contacts=len(ids),
            local_contact_replanning=True,
            nearby_contact_update_count=sum(len(row.get('local_contact_updates', [])) for row in records),
            mpc_objective="pad_contact_and_object_pose",
            startup_timings=startup_times,
            planned_grasp_certificate=planned_certificate, ik_handoff_step=handoff_step,
            contact_set_frozen=bool(args.freeze_contact_set),
            replan_count=recovery.replan_count,
            prediction_intervals=dict(lambda_dt=args.lambda_prediction_dt, mpc_dt=self.command_dt),
            final_object_pos=final_pos.tolist(), final_object_quat=final_quat.tolist(),
            object_target_pose=goal.tolist(),
            best_height_gain=best_height, best_contact_finger_count=best_contacts,
            last_mpc_status=status, initial_ik_mean_tip_error=initial_ik.mean_tip_error,
            initial_ik_max_tip_error=initial_ik.max_tip_error, contact_indices=ids.tolist(),
            initial_ik_max_self_penetration=initial_ik.max_self_penetration,
            initial_ik_hand_q=initial_ik.qpos.tolist(),
            pregrasp_hand_q=initial_open_posture.tolist(), pregrasp_minimum_pad_clearance=clearance,
            contact_points_local=points.tolist(), steps=records)
        if args.diagnostics_output is not None:
            import json
            args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostics_output.write_text(json.dumps(report, indent=2))
        print(f"grasp_success: {success}; best_contacts: {best_contacts}; best_height_gain: {best_height:.4f} m")
        saved = self.finalize_video_recording()
        if saved is not None:
            print(f"saved mp4: {saved}")
        if self.viewer is not None and not args.no_wait:
            while self.viewer.is_running():
                self.sync_viewer()
                time.sleep(.02)
        return report

    def run_staged(self):
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
        print(f"visual_mesh_path: {self.mesh_path}")
        print(f"collision_mesh_path: {self.collision_mesh_path}")
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
        static_result = grasp_result.get("static_equilibrium", None)
        if static_result is not None:
            print(
                "static_equilibrium: "
                f"valid={bool(static_result.get('valid', False))} "
                f"cost={float(static_result.get('cost', float('inf'))):.6f} "
                f"scaled_residual={float(static_result.get('scaled_residual_norm', float('inf'))):.6f}"
            )

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
        last_mjwp_rollout_info = None
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

            applied_object_torque_world = None
            if bool(self.args.test_force):
                _, _, current_object_rot = self.get_object_pose(self.data)
                _, applied_object_torque_world = self._best_object_wrench_world(
                    grasp_result,
                    current_object_rot,
                )

            self.apply_mpc_action(
                action=mpc_result["action"],
                base_orientation=target_base_q[BASE_TRANSLATION_DOF_COUNT:],
                target_base_q=target_base_q,
                freeze_fingers=freeze_fingers,
                ignore_base_release_gate=ignore_base_release_gate,
                object_torque_world=applied_object_torque_world,
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
        description="Allegro n-contact grasping with shared Warp Lambda LM and live joint-space MPC."
    )
    parser.add_argument("--controller", choices=("online", "staged"), default="online", help="Live Lambda/MPC or the legacy reference trajectory.")
    parser.add_argument("--online-point-pads", nargs="?", const=True, default=True, type=parse_bool_arg, help="Use spherical pads at the tracked centres, matching bigrasp point-contact geometry; false keeps Allegro capsules.")
    parser.add_argument("--online-execution", choices=("position", "osc"), default="position", help="Position servos or bigrasp-style capped stacked-tip OSC.")
    parser.add_argument("--online-task-stiffness", type=float, default=600.)
    parser.add_argument("--online-task-damping", type=float, default=40.)
    parser.add_argument("--online-force-limit", type=float, default=12.)
    parser.add_argument("--online-unilateral-force-limit", type=float, default=.05)
    parser.add_argument("--online-approach-speed", type=float, default=.25)
    parser.add_argument("--online-nullspace-stiffness", type=float, default=5., help="OSC IK nullspace attraction during distant approach only.")
    parser.add_argument("--mpc-solver", choices=("acados", "ipopt"), default="acados")
    parser.add_argument("--lambda-device", default="cpu", help="Warp LM device: cpu or cuda:0.")
    parser.add_argument("--lambda-contact-region", choices=("body-sides", "all"), default="body-sides", help="Use the shared Lambda body-flank filter for an opposing Allegro grasp, or all surface patches.")
    parser.add_argument("--freeze-contact-set", nargs="?", const=True, default=True, type=parse_bool_arg, help="Keep Lambda sample assignments within an attempt; certified nearby replacements and measured contacts can update MPC regions.")
    parser.add_argument("--regrasp-position-threshold", type=float, default=.025, help="Replan an ungrasped object after this displacement (m).")
    parser.add_argument("--regrasp-rotation-threshold", type=float, default=.3, help="Replan an ungrasped object after this rotation (rad).")
    parser.add_argument("--regrasp-cooldown", type=float, default=1., help="Minimum simulated seconds between recovery patch searches.")
    parser.add_argument("--regrasp-stall-timeout", type=float, default=2., help="Retry a contact attempt after this many simulated seconds without a confirmed grasp, including stalled partial contacts.")
    parser.add_argument("--regrasp-clearance-timeout", type=float, default=3., help="Search again if the MPC cannot reach clear pregrasp pads within this many simulated seconds.")
    parser.add_argument("--regrasp-ik-time-budget", type=float, default=.25, help="Time budget for the single coupled IK recovery seed (s).")
    parser.add_argument("--lambda-prediction-dt", type=float, default=None, help="Lambda prediction interval; defaults to the same command_dt used by MPC.")
    parser.add_argument("--lambda-lm-iters", type=int, default=20)
    parser.add_argument("--lambda-force-limit", type=float, default=10.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--online-tip-radius", type=float, default=.012, help="Allegro collision pad radius.")
    parser.add_argument("--squeeze-depth", type=float, default=.005)
    parser.add_argument("--online-pad-stiffness", type=float, default=600.)
    parser.add_argument("--contact-impratio", type=float, default=10.,
        help="Tangential-to-normal contact impedance ratio for the online elliptic scene; changes softness, not the Coulomb coefficient.")
    parser.add_argument("--contact-noslip-iterations", type=int, default=3,
        help="MuJoCo friction-only NoSlip sweeps for the online scene (default 3); 0 retains regularized tangential creep. Does not change friction coefficients.")
    from models.soft_scm_contact import (
        SOFT_SCM_RESPONSE_TIME, SOFT_SCM_DAMPING_RATIO, SOFT_SCM_REGULARIZATION)
    parser.add_argument("--soft-scm-response-time", "--force-response-time", type=float, default=None,
        help="Contact response time (s); defaults to the MuJoCo object's solref time scale.")
    parser.add_argument("--soft-scm-damping-ratio", type=float, default=SOFT_SCM_DAMPING_RATIO)
    parser.add_argument("--soft-scm-regularization", type=float, default=SOFT_SCM_REGULARIZATION)
    parser.add_argument("--online-tip-weight", type=float, default=5000., help="MPC weight on pad centres aligning with moving object contact targets.")
    parser.add_argument("--online-contact-transport-weight", type=float, default=1000., help="After confirmed force closure, carry pad contact regions to the object's goal pose; no joint IK tracking.")
    parser.add_argument("--online-contact-transport-distance", type=float, default=None,
        help="Maximum pad transport reference advance (m); defaults to lift speed times prediction horizon. The object pose goal stays unchanged.")
    parser.add_argument("--online-self-collision-weight", type=float, default=1e7, help="Penalty on predicted inter-finger overlap and clearance violations.")
    parser.add_argument("--online-self-collision-clearance", type=float, default=.001, help="Desired separation between neighbouring finger links (m).")
    parser.add_argument("--online-force-tracking-weight", type=float, default=10., help="Weight on the resulting pad wrench error normalized by object weight.")
    parser.add_argument("--online-force-feedback", action=argparse.BooleanOptionalAction, default=False,
        help="Refresh the force-tracking output bias and preload gain from measured pad forces each cycle; the softSCM motion model is retained.")
    parser.add_argument("--online-pressure-control", action=argparse.BooleanOptionalAction, default=True,
        help="Track pad normal preload while MPC controls tangential motion; prediction and execution share the same joint command map.")
    parser.add_argument("--online-normal-admittance", type=float, default=.0003, help="Measured normal-force feedback gain (m/N).")
    parser.add_argument("--online-local-contact-distance", type=float, default=.02,
        help="Each MPC cycle may adopt a reachable nearest pad patch inside this distance if it forms force closure with the current contacts (m).")
    parser.add_argument("--online-min-normal-force", type=float, default=.6, help="Compressive preload per contacting pad in the live wrench allocator (N).")
    parser.add_argument("--online-friction-margin", type=float, default=.8, help="Fraction of the physical Coulomb cone used for live wrench allocation.")
    parser.add_argument("--online-pressure-sync-weight", type=float, default=0.)
    parser.add_argument("--online-base-position-weight", type=float, default=1.)
    parser.add_argument("--online-posture-weight", type=float, default=5., help="Weak IK approach attraction; fades to zero before pad contact.")
    parser.add_argument("--online-joint-action-weight", type=float, default=50.)
    parser.add_argument("--online-rotation-action-weight", type=float, default=50.)
    parser.add_argument("--online-contact-base-speed", type=float, default=.04, help="Palm translation speed bound within 2 cm of fingertip contact (m/s).")
    parser.add_argument("--online-contact-rotation-speed", type=float, default=.2, help="Palm hinge speed bound near fingertip contact (rad/s).")
    parser.add_argument("--online-contact-joint-speed", type=float, default=.5, help="Finger joint speed bound near fingertip contact (rad/s).")
    parser.add_argument("--online-palm-orientation-weight", type=float, default=.5, help="Weak palm orientation attraction during approach only.")
    parser.add_argument("--online-object-lateral-weight", type=float, default=300.)
    parser.add_argument("--online-horizontal-recovery-weight", type=float, default=2e6, help="bigrasp soft corridor penalty on additional horizontal object displacement.")
    parser.add_argument("--online-max-horizontal-error", type=float, default=.005, help="Allowed approach displacement along x/y in metres.")
    parser.add_argument("--online-synchronization-weight", type=float, default=1500., help="Legacy option; use --online-pressure-sync-weight for contact force synchronization.")
    parser.add_argument("--online-lift-speed", type=float, default=.05)
    parser.add_argument("--mpc-rotation-step-limit", type=float, default=.03)
    parser.add_argument("--max-steps", type=int, default=None, help="Online command-step limit; defaults to the legacy stage budgets plus contact wait.")
    parser.add_argument("--goal-tolerance", type=float, default=.015)
    parser.add_argument("--diagnostics-output", type=Path, default=None)
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
    parser.add_argument("--contact-stiffness", type=float, default=12.5, help="Contact stiffness passed to mlqp_point_v2.")
    parser.add_argument("--sample-num", type=int, default=70, help="Surface sample count used by mlqp_point_v2.")
    parser.add_argument(
        "--solver",
        type=str,
        choices=("acados", "ipopt"),
        default="acados",
        help="Legacy staged solver used by mlqp_point_v2 for grasp scoring and static-equilibrium checks.",
    )
    parser.add_argument("--num-grasp-contacts", type=int, default=4, choices=(3, 4), help="Active fingers: thumb/index/middle, optionally ring.")
    parser.add_argument("--region-contact-samples", type=int, default=5, help="Region sample count passed to mlqp_point_v2.")
    parser.add_argument("--top-region-pairs", type=int, default=3, help="Number of top region groups evaluated by mlqp_point_v2.")
    parser.add_argument("--preselect-region-pairs", type=int, default=200, help="Preselected region group budget passed to mlqp_point_v2.")
    parser.add_argument("--max-point-combination-eval", type=int, default=256, help="Max contact combinations evaluated inside mlqp_point_v2.")
    parser.add_argument("--ground-height-margin", type=float, default=0.003, help="Support-surface clearance given to mlqp_point_v2.")
    parser.add_argument("--support-normal-alignment-threshold", type=float, default=0.25, help="Filter out support-facing contacts below this alignment threshold.")
    parser.add_argument("--contact-tip-offset", type=float, default=0.0, help="Offset fingertip IK targets outward along the contact normals.")
    parser.add_argument("--ik-contact-candidates", type=int, default=16, help="Lambda quality candidates checked for coupled-hand reachability within the startup time budget.")
    parser.add_argument("--ik-startup-time-budget", type=float, default=3., help="Total seconds allowed for online startup IK; retain the best posture and let live MPC approach the anchors.")
    parser.add_argument("--ik-max-iters", type=int, default=220, help="Maximum iterations per Allegro IK candidate.")
    parser.add_argument("--ik-step-size", type=float, default=0.35, help="Damped least-squares IK step size.")
    parser.add_argument("--ik-damping", type=float, default=2e-4, help="Damped least-squares regularization.")
    parser.add_argument("--ik-tip-orientation-weight", type=float, default=.2)
    parser.add_argument("--ik-collision-weight", type=float, default=10.)
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
    parser.add_argument(
        "--finger-ctrl-release-palm-tol",
        type=float,
        default=0.15,
        help="During translate_to_contact, release finger control once the palm is within this world-distance of the IK palm target pose.",
    )
    parser.add_argument("--lift-height", type=float, default=0.2, help="Lift distance after the squeeze stage.")
    parser.add_argument("--lift-steps", type=int, default=120, help="Command steps spent lifting the grasped object.")
    parser.add_argument("--hold-steps", type=int, default=20, help="Extra command steps after lifting.")
    parser.add_argument(
        "--grasp-lift-cost-weight",
        type=float,
        default=1.0,
        help="Weight on the static-equilibrium objective added to mlqp_point_v2 grasp ranking when evaluating liftable contact sets.",
    )
    parser.add_argument(
        "--grasp-lift-residual-weight",
        type=float,
        default=10.0,
        help="Weight on the static-equilibrium residual norm added to mlqp_point_v2 grasp ranking.",
    )
    parser.add_argument("--lift-contact-fingers", type=int, default=None, help="Required physical fingertip contacts (default: three); force closure and bounded load support are also required.")
    parser.add_argument("--online-guidance", choices=("mpc", "ik-mpc"), default="mpc", help="Contact MPC throughout, or an initial collision-free pregrasp IK servo approach.")
    parser.add_argument("--online-ik-base-speed", type=float, default=.15)
    parser.add_argument("--online-ik-joint-speed", type=float, default=.8)
    parser.add_argument("--online-ik-palm-position-weight", type=float, default=.05)
    parser.add_argument("--online-ik-posture-regularization", type=float, default=.005)
    parser.add_argument("--online-ik-tip-tolerance", type=float, default=.002,
        help="Mean position tolerance for the approach IK seed (m); maximum is 1.5 times this. Live MPC refines contact.")
    parser.add_argument("--online-ik-handoff-joint-tolerance", type=float, default=.12)
    parser.add_argument("--online-preshape-margin", type=float, default=.1)
    parser.add_argument("--online-pregrasp-clearance", type=float, default=.015, help="Outward pad clearance used by the one-time pregrasp IK (m).")
    parser.add_argument("--online-handoff-tip-tolerance", type=float, default=.008)
    parser.add_argument("--online-handoff-patch-tolerance", type=float, default=.012)
    parser.add_argument("--grasp-release-stable-steps", type=int, default=3, help="Release the grasp latch after contact/closure/patch verification fails for this many commands.")
    parser.add_argument("--lift-contact-stable-steps", type=int, default=6, help="Number of consecutive command steps that must satisfy the lift-contact requirement.")
    parser.add_argument("--lift-contact-force-threshold", type=float, default=None, help="Optional normal-force threshold used by the lift gate. Defaults to --contact-force-threshold.")
    parser.add_argument("--max-lift-wait-steps", type=int, default=160, help="Extra command steps spent waiting for fingertip contact before giving up on the lift gate.")
    parser.add_argument("--open-pose-fraction", type=float, default=0.05, help="How far from the lower joint limits the initial open finger pose starts.")
    parser.add_argument("--min-closing-margin", type=float, default=0.02, help="Minimum opening margin maintained between the open pose and the final IK finger pose.")
    parser.add_argument("--mpc-horizon", type=int, default=4, help="Explicit MPC horizon length in command steps.")
    parser.add_argument("--mpc-ipopt-max-iter", type=int, default=50, help="Maximum IPOPT iterations used by the explicit MPC.")
    parser.add_argument("--mpc-velocity-limit", type=float, default=3.5, help="Finger velocity bound used to derive each command-step finger action limit.")
    parser.add_argument("--mpc-base-step-limit", type=float, default=0.01, help="Maximum xyz increment per command applied to the Allegro base.")
    parser.add_argument("--mpc-max-contacts", type=int, default=None, help="Maximum number of object contacts included in the explicit MPC linearization.")
    parser.add_argument("--mpc-base-track-weight", type=float, default=4000.0, help="Legacy staged base approach attraction; disabled near pad contact.")
    parser.add_argument("--mpc-q-weight", type=float, default=30.0, help="Legacy staged finger approach attraction; disabled near pad contact.")
    parser.add_argument("--mpc-object-pos-weight", type=float, default=120.0, help="Running-state tracking weight on the object position target.")
    parser.add_argument("--mpc-object-quat-weight", type=float, default=5.0, help="Running-state tracking weight on the object quaternion target.")
    parser.add_argument("--mpc-terminal-weight", type=float, default=4.0, help="Multiplier applied to the terminal tracking costs.")
    parser.add_argument("--mpc-u-weight", type=float, default=0.03, help="Control-effort penalty used by the explicit MPC.")
    parser.add_argument(
        "--mjwp-rollout",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool_arg,
        help="Enable MuJoCo Warp sampling correction for --controller staged.",
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
    parser.add_argument("--mjwp-finger-cost-weight", type=float, default=2.0, help="Finger approach attraction inside MJWP; disabled at contact.")
    parser.add_argument("--mjwp-object-cost-weight", type=float, default=80.0, help="Weight on object position tracking inside MJWP rollout.")
    parser.add_argument("--mjwp-object-quat-cost-weight", type=float, default=5.0, help="Weight on object quaternion tracking inside MJWP rollout.")
    parser.add_argument("--mjwp-action-cost-weight", type=float, default=0.1, help="Weight on sampled action magnitude inside MJWP rollout.")
    parser.add_argument("--contact-force-threshold", type=float, default=1e-4, help="Normal-force threshold used when reporting fingertip-object contacts.")
    parser.add_argument("--log-every", type=int, default=10, help="Print one execution log every N command steps.")
    parser.add_argument("--screenshot-dir", type=Path, default=DEFAULT_SCREENSHOT_DIR, help="Directory where SVG screenshots will be written.")
    parser.add_argument("--screenshot-interval", type=float, default=1.0, help="Seconds between automatic SVG screenshots. Use 0 to disable.")
    parser.add_argument("--screenshot-width", type=int, default=1280, help="Screenshot render width.")
    parser.add_argument("--screenshot-height", type=int, default=960, help="Screenshot render height.")
    parser.add_argument("--save", action="store_true", help="Record an MP4 video; recording is off by default.")
    parser.add_argument("--video-output-dir", type=Path, default=DEFAULT_VIDEO_OUTPUT_DIR, help="MP4 output directory when --save is supplied.")
    parser.add_argument("--video-output-path", type=Path, default=None, help="Optional explicit MP4 output path. Overrides --video-output-dir.")
    parser.add_argument("--video-fps", type=float, default=20.0, help="MP4 capture frame rate.")
    parser.add_argument("--video-width", type=int, default=1280, help="MP4 render width.")
    parser.add_argument("--video-height", type=int, default=960, help="MP4 render height.")
    parser.add_argument("--no-video", action="store_true", help="Disable MP4 recording.")
    parser.add_argument(
        "--test_force",
        "--test-force",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_arg,
        help="If true, project the best optimizer wrench to world coordinates and apply it directly to the object during execution.",
    )
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
