import argparse
import copy
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import trimesh


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "scsp-robot"), None) or next(p for p in Path(__file__).resolve().parents if (p / "planning" / "acados_env.py").is_file())
CUROBO_SRC_ROOT = REPO_ROOT.parent / "thirdparty" / "curobo" / "src"
for path in (REPO_ROOT, CUROBO_SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))
from planning.acados_env import ensure_acados_env
ensure_acados_env()

try:
    import torch
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.rollout.rollout_base import Goal
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

    _HAS_CUROBO = True
    _CUROBO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime dependency
    torch = None
    Cuboid = None
    WorldConfig = None
    Goal = None
    Pose = None
    JointState = None
    IKSolver = None
    IKSolverConfig = None
    MpcSolver = None
    MpcSolverConfig = None
    _HAS_CUROBO = False
    _CUROBO_IMPORT_ERROR = exc

from planning.mlqp_point_v2 import LambdaContactControlOptimizer


PANDA_XML_PATH = REPO_ROOT / "envs" / "xmls" / "panda_nohand.xml"
GENERATED_SCENE_PATH = REPO_ROOT / "envs" / "xmls" / "_generated_bigrasp_scene.xml"
OBJECT_ASSET_DIR = REPO_ROOT / "envs" / "assets" / "objects"

PANDA_HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
TIP_RADIUS = 0.01
TIP_CENTER_OFFSET = 0.06
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
DEFAULT_SCALE_MAP = {
    "stanford_bunny2": np.array([1.5, 1.5, 1.5], dtype=np.float64),
    "rubber_duck": np.array([1.3, 1.4, 1.4], dtype=np.float64),
    "Wolf_Duck": np.array([0.002, 0.002, 0.002], dtype=np.float64),
}
ARM_OBSTACLE_SEGMENTS = (
    ("link6", "link5", "link6", 0.12),
    ("link7", "link6", "link7", 0.10),
    ("fingertip", "attachment", "tip_center", 0.07),
)
ARM_OBSTACLE_LENGTH_PADDING = 0.06


def _normalize(vec, eps=1e-9):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def _project_to_plane(vec, normal):
    vec = np.asarray(vec, dtype=np.float64)
    normal = _normalize(normal)
    return vec - np.dot(vec, normal) * normal


def _project_to_rotation_matrix(rotation_matrix):
    u, _, vh = np.linalg.svd(np.asarray(rotation_matrix, dtype=np.float64))
    projected = u @ vh
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vh
    return projected


def _rotation_error(current_rot, target_rot):
    current_rot = np.asarray(current_rot, dtype=np.float64).reshape(3, 3)
    target_rot = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
    return 0.5 * (
        np.cross(current_rot[:, 0], target_rot[:, 0])
        + np.cross(current_rot[:, 1], target_rot[:, 1])
        + np.cross(current_rot[:, 2], target_rot[:, 2])
    )


def _rotate_vector_toward(source_vec, target_vec, max_angle):
    source_vec = _normalize(source_vec)
    target_vec = _normalize(target_vec)
    max_angle = max(float(max_angle), 0.0)
    if np.linalg.norm(source_vec) < 1e-9:
        return target_vec if np.linalg.norm(target_vec) >= 1e-9 else np.array([0.0, 0.0, -1.0], dtype=np.float64)
    if np.linalg.norm(target_vec) < 1e-9 or max_angle <= 1e-9:
        return source_vec

    dot = float(np.clip(np.dot(source_vec, target_vec), -1.0, 1.0))
    angle = float(np.arccos(dot))
    if angle <= 1e-9:
        return source_vec

    step_angle = min(angle, max_angle)
    rot_axis = np.cross(source_vec, target_vec)
    axis_norm = float(np.linalg.norm(rot_axis))
    if axis_norm < 1e-9:
        rot_axis = _project_to_plane(np.array([1.0, 0.0, 0.0], dtype=np.float64), source_vec)
        if np.linalg.norm(rot_axis) < 1e-9:
            rot_axis = _project_to_plane(np.array([0.0, 1.0, 0.0], dtype=np.float64), source_vec)
        axis_norm = float(np.linalg.norm(rot_axis))
        if axis_norm < 1e-9:
            return source_vec
    rot_axis = rot_axis / axis_norm

    rotated = (
        source_vec * np.cos(step_angle)
        + np.cross(rot_axis, source_vec) * np.sin(step_angle)
        + rot_axis * np.dot(rot_axis, source_vec) * (1.0 - np.cos(step_angle))
    )
    return _normalize(rotated)


def _joint_index(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    if match is None:
        raise ValueError(f"Unable to extract joint index from name: {name}")
    return int(match.group(1))


def _tensor_to_numpy(value) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _scalar(value) -> float:
    return float(_tensor_to_numpy(value).reshape(-1)[0])


def quat_wxyz_to_mat(quat):
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def mat_to_quat_wxyz(rotation_matrix):
    rotation_matrix = _project_to_rotation_matrix(rotation_matrix)
    quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, rotation_matrix.reshape(-1))
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def quat_from_axis_angle(axis, angle):
    axis = _normalize(axis)
    if np.linalg.norm(axis) < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half = 0.5 * float(angle)
    return np.array(
        [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
        dtype=np.float64,
    )


def quat_from_yaw(yaw):
    return quat_from_axis_angle([0.0, 0.0, 1.0], yaw)


def _extract_pose_from_kinematics(kinematics_state):
    ee_pose = getattr(kinematics_state, "ee_pose", None)
    if ee_pose is not None:
        pos = _tensor_to_numpy(ee_pose.position).reshape(-1, 3)[0]
        quat = _tensor_to_numpy(ee_pose.quaternion).reshape(-1, 4)[0]
        return pos.astype(np.float64), quat.astype(np.float64)

    pos = _tensor_to_numpy(kinematics_state.ee_pos_seq).reshape(-1, 3)[0]
    quat = _tensor_to_numpy(kinematics_state.ee_quat_seq).reshape(-1, 4)[0]
    return pos.astype(np.float64), quat.astype(np.float64)


def build_curobo_state(arm_q_mj, curobo_joint_names, default_joint_vector):
    arm_q_mj = np.asarray(arm_q_mj, dtype=np.float64).reshape(-1)
    joint_vec = np.asarray(default_joint_vector, dtype=np.float64).reshape(len(curobo_joint_names)).copy()
    arm_map = {
        _joint_index(f"joint{joint_idx + 1}"): float(value)
        for joint_idx, value in enumerate(arm_q_mj)
    }
    for idx, joint_name in enumerate(curobo_joint_names):
        if "finger" in joint_name:
            continue
        joint_vec[idx] = arm_map[_joint_index(joint_name)]
    return joint_vec


def extract_mujoco_arm_configuration(curobo_joint_vector, curobo_joint_names):
    curobo_joint_vector = np.asarray(curobo_joint_vector, dtype=np.float64).reshape(len(curobo_joint_names))
    arm_map = {
        _joint_index(name): value
        for name, value in zip(curobo_joint_names, curobo_joint_vector)
        if "finger" not in name
    }
    return np.array([arm_map[idx] for idx in range(1, 8)], dtype=np.float64)


def make_joint_state(controller, joint_vector, joint_names=None):
    joint_names = controller.joint_names if joint_names is None else joint_names
    tensor = torch.tensor(
        np.asarray(joint_vector, dtype=np.float32).reshape(1, -1),
        device=controller.tensor_args.device,
        dtype=controller.tensor_args.dtype,
    )
    return JointState.from_position(tensor, joint_names=joint_names)


def make_pose(controller, position, quat_wxyz):
    pos_t = torch.tensor(
        np.asarray(position, dtype=np.float32).reshape(1, 3),
        device=controller.tensor_args.device,
        dtype=controller.tensor_args.dtype,
    )
    quat_t = torch.tensor(
        np.asarray(quat_wxyz, dtype=np.float32).reshape(1, 4),
        device=controller.tensor_args.device,
        dtype=controller.tensor_args.dtype,
    )
    return Pose(position=pos_t, quaternion=quat_t)


def make_joint_tensor(controller, joint_vector, extra_dim=False):
    shape = (1, 1, -1) if extra_dim else (1, -1)
    return torch.tensor(
        np.asarray(joint_vector, dtype=np.float32).reshape(shape),
        device=controller.tensor_args.device,
        dtype=controller.tensor_args.dtype,
    )


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


def format_vec(vec):
    return " ".join(f"{float(v):.8f}" for v in np.asarray(vec, dtype=np.float64).reshape(-1))


def prefix_robot_tree(element, prefix):
    element = copy.deepcopy(element)
    queue = [element]
    while queue:
        node = queue.pop()
        if "name" in node.attrib:
            node.attrib["name"] = f"{prefix}{node.attrib['name']}"
        queue.extend(list(node))
    return element


def prefix_reference_attributes(element, prefix):
    element = copy.deepcopy(element)
    rename_keys = (
        "joint",
        "joint1",
        "joint2",
        "body",
        "body1",
        "body2",
        "site",
        "geom",
        "geom1",
        "geom2",
        "camera",
        "light",
    )
    queue = [element]
    while queue:
        node = queue.pop()
        for key in rename_keys:
            if key in node.attrib:
                node.attrib[key] = f"{prefix}{node.attrib[key]}"
        if "name" in node.attrib:
            node.attrib["name"] = f"{prefix}{node.attrib['name']}"
        queue.extend(list(node))
    return element


def build_bimanual_scene_xml(
    mesh_path,
    mesh_scale,
    object_mass,
    object_pos,
    object_quat,
    scene_center_x,
    robot_span,
    pedestal_size=(0.05, 0.07, 0.06),
    pedestal_pos=(0.58, 0.0, 0.06),
    scene_output_path=GENERATED_SCENE_PATH,
):
    panda_root = ET.parse(PANDA_XML_PATH).getroot()

    root = ET.Element("mujoco", {"model": "dual panda bigrasp"})
    root.append(ET.Element("compiler", {"angle": "radian", "meshdir": "assets", "autolimits": "true"}))
    root.append(ET.Element("option", {"integrator": "implicitfast", "impratio": "10"}))
    root.append(ET.Element("statistic", {"center": f"{scene_center_x:.4f} 0 0.35", "extent": "1.2"}))

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.6 0.6 0.6", "ambient": "0.25 0.25 0.25", "specular": "0.1 0.1 0.1"},
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "135", "elevation": "-25"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.30 0.45 0.60",
            "rgb2": "0.02 0.03 0.04",
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
            "rgb1": "0.20 0.28 0.34",
            "rgb2": "0.12 0.16 0.20",
            "markrgb": "0.85 0.85 0.85",
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
            "reflectance": "0.2",
        },
    )
    ET.SubElement(asset, "material", {"name": "pedestal_mat", "rgba": "0.28 0.30 0.34 1"})
    ET.SubElement(asset, "material", {"name": "obj_mat", "rgba": "0.88 0.52 0.22 1"})
    ET.SubElement(asset, "material", {"name": "ghost_obj_mat", "rgba": "0.88 0.52 0.22 0.20"})
    ET.SubElement(asset, "material", {"name": "left_marker_mat", "rgba": "0.15 0.75 0.95 1"})
    ET.SubElement(asset, "material", {"name": "right_marker_mat", "rgba": "0.95 0.25 0.35 1"})
    ET.SubElement(
        asset,
        "mesh",
        {
            "name": "object_mesh",
            "file": str(Path(mesh_path).resolve()),
            "scale": format_vec(mesh_scale),
        },
    )
    for child in list(panda_root.find("asset")):
        asset.append(copy.deepcopy(child))

    default = ET.SubElement(root, "default")
    for child in list(panda_root.find("default")):
        default.append(copy.deepcopy(child))

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0.4 -0.2 1.6", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": "1.15 -0.70 0.65",
            "xyaxes": "0.51 0.86 0.00 -0.30 0.18 0.94",
        },
    )
    ET.SubElement(worldbody, "geom", {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane"})

    ET.SubElement(
        ET.SubElement(
            worldbody,
            "body",
            {"name": "pedestal", "pos": format_vec(pedestal_pos)},
        ),
        "geom",
        {
            "name": "pedestal_geom",
            "type": "box",
            "size": format_vec(pedestal_size),
            "material": "pedestal_mat",
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.0 0.08 0.01",
            "condim": "3",
        },
    )

    goal_body = ET.SubElement(worldbody, "body", {"name": "goal", "pos": format_vec(object_pos), "quat": format_vec(object_quat)})
    ET.SubElement(
        goal_body,
        "geom",
        {
            "name": "goal_geom",
            "type": "mesh",
            "mesh": "object_mesh",
            "material": "ghost_obj_mat",
            "contype": "0",
            "conaffinity": "0",
        },
    )

    obj_body = ET.SubElement(worldbody, "body", {"name": "obj"})
    ET.SubElement(obj_body, "freejoint", {"name": "obj_freejoint"})
    ET.SubElement(
        obj_body,
        "geom",
        {
            "name": "obj",
            "type": "mesh",
            "mesh": "object_mesh",
            "material": "obj_mat",
            "mass": f"{float(object_mass):.8f}",
            "condim": "6",
            "friction": "0.9 0.08 0.01",
            "solimp": "0.9 0.95 0.01",
            "solref": "0.02 1.0",
            "margin": "0.0",
            "gap": "0.0",
        },
    )

    marker_specs = [
        ("obj_point", "1 1 0 1"),
        ("contact_point1", "0 1 0 1"),
        ("contact_point2", "0 0 1 1"),
        ("left_goal", "0.15 0.75 0.95 1"),
        ("right_goal", "0.95 0.25 0.35 1"),
    ]
    for name, rgba in marker_specs:
        marker_body = ET.SubElement(worldbody, "body", {"name": name, "pos": format_vec(object_pos)})
        ET.SubElement(
            marker_body,
            "geom",
            {
                "name": f"{name}_geom",
                "type": "sphere",
                "size": "0.008",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
            },
        )

    ghost_specs = [
        ("left_ghost_tip", "0.15 0.75 0.95 0.28"),
        ("right_ghost_tip", "0.95 0.25 0.35 0.28"),
    ]
    for name, rgba in ghost_specs:
        ghost_body = ET.SubElement(worldbody, "body", {"name": name, "pos": format_vec(object_pos)})
        ET.SubElement(
            ghost_body,
            "geom",
            {
                "name": f"{name}_rod",
                "type": "cylinder",
                "size": "0.005 0.03",
                "pos": "0 0 0.03",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
            },
        )
        ET.SubElement(
            ghost_body,
            "geom",
            {
                "name": f"{name}_sphere",
                "type": "sphere",
                "size": "0.010",
                "pos": "0 0 0.06",
                "rgba": rgba,
                "contype": "0",
                "conaffinity": "0",
            },
        )

    robot_body = panda_root.find("./worldbody/body[@name='link0']")
    if robot_body is None:
        raise RuntimeError(f"Failed to find Panda root body in {PANDA_XML_PATH}")
    robot_contact = panda_root.find("contact")
    panda_actuator = panda_root.find("actuator")

    left_root = prefix_robot_tree(robot_body, "left_")
    left_root.attrib["pos"] = format_vec([scene_center_x - 0.5 * robot_span, 0.0, 0.0])
    left_root.attrib["quat"] = "1 0 0 0"
    left_attachment = left_root.find(".//body[@name='left_attachment']")
    if left_attachment is not None:
        ET.SubElement(left_attachment, "site", {"name": "left_tip_center", "pos": "0 0 0.06", "size": "0.002"})

    right_root = prefix_robot_tree(robot_body, "right_")
    right_root.attrib["pos"] = format_vec([scene_center_x + 0.5 * robot_span, 0.0, 0.0])
    right_root.attrib["quat"] = "0 0 0 1"
    right_attachment = right_root.find(".//body[@name='right_attachment']")
    if right_attachment is not None:
        ET.SubElement(right_attachment, "site", {"name": "right_tip_center", "pos": "0 0 0.06", "size": "0.002"})

    worldbody.append(left_root)
    worldbody.append(right_root)

    actuator = ET.SubElement(root, "actuator")
    for child in list(panda_actuator):
        actuator.append(prefix_reference_attributes(child, "left_"))
    for child in list(panda_actuator):
        actuator.append(prefix_reference_attributes(child, "right_"))

    contact = ET.SubElement(root, "contact")
    for child in list(robot_contact):
        contact.append(prefix_reference_attributes(child, "left_"))
    for child in list(robot_contact):
        contact.append(prefix_reference_attributes(child, "right_"))

    tree = ET.ElementTree(root)
    scene_output_path = Path(scene_output_path)
    scene_output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(scene_output_path, encoding="utf-8", xml_declaration=False)
    return scene_output_path


@dataclass
class ArmHandles:
    prefix: str
    joint_ids: np.ndarray
    qpos_adr: np.ndarray
    dof_adr: np.ndarray
    actuator_ids: np.ndarray
    body_id: int
    tip_geom_id: int
    tip_site_id: int
    ghost_body_id: int
    base_pos: np.ndarray
    base_rot: np.ndarray
    body_ids_by_name: dict
    mpc: object = None
    ik_solver: object = None
    curobo_joint_names: tuple = ()
    retract_cfg: np.ndarray = None
    goal_buffer: object = None
    static_world_with_pedestal: object = None
    static_world_floor_only: object = None
    static_world: object = None
    current_world: object = None
    current_world_mode: str = "with_pedestal"


@dataclass
class ArmIkResult:
    q_mj: np.ndarray
    success: bool
    position_error: float
    rotation_error: float
    target_hand_pos_world: np.ndarray
    target_hand_rot_world: np.ndarray
    solved_hand_pos_world: np.ndarray
    solved_hand_rot_world: np.ndarray
    solved_tip_pos_world: np.ndarray
    solved_tip_rot_world: np.ndarray
    constraint_total: float = 0.0
    bound_constraint: float = 0.0
    world_constraint: float = 0.0
    static_world_constraint: float = 0.0
    self_constraint: float = 0.0
    failure_reason: str = ""


class BimanualPandaGrasper:
    def __init__(self, args):
        if not _HAS_CUROBO:
            raise ImportError(
                "Failed to import cuRobo. Make sure cuRobo is installed or "
                f"{CUROBO_SRC_ROOT} is available on PYTHONPATH. "
                f"Original error: {_CUROBO_IMPORT_ERROR!r}"
            )

        self.args = args
        logging.getLogger("curobo").setLevel(logging.WARNING)
        self.mesh_path = resolve_mesh_path(args.obj, args.mesh)
        self.mesh_scale = self._resolve_mesh_scale(args)
        self.mesh_bounds = load_mesh_bounds(self.mesh_path, self.mesh_scale)
        self.pedestal_size = np.asarray(args.pedestal_size, dtype=np.float64).copy()
        self.pedestal_pos = np.asarray(args.pedestal_pos, dtype=np.float64).copy()
        self.pedestal_pos[2] += float(args.initial_object_lift)

        support_top = self.pedestal_pos[2] + self.pedestal_size[2]
        self.support_top = float(support_top)
        self.support_surface_point = np.array(
            [self.pedestal_pos[0], self.pedestal_pos[1], self.support_top],
            dtype=np.float64,
        )
        self.support_surface_normal = WORLD_UP.copy()
        object_pos = np.array(
            [
                args.scene_center_x,
                0.0,
                support_top - self.mesh_bounds[0, 2] + args.object_z_offset,
            ],
            dtype=np.float64,
        )
        object_quat = quat_from_yaw(args.object_yaw)

        self.left_base_pos = np.array(
            [args.scene_center_x - 0.5 * args.robot_span, 0.0, 0.0],
            dtype=np.float64,
        )
        self.right_base_pos = np.array(
            [args.scene_center_x + 0.5 * args.robot_span, 0.0, 0.0],
            dtype=np.float64,
        )
        self.left_base_rot = np.eye(3, dtype=np.float64)
        self.right_base_rot = quat_wxyz_to_mat(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))

        self.scene_path = build_bimanual_scene_xml(
            mesh_path=self.mesh_path,
            mesh_scale=self.mesh_scale,
            object_mass=args.obj_mass,
            object_pos=object_pos,
            object_quat=object_quat,
            scene_center_x=args.scene_center_x,
            robot_span=args.robot_span,
            pedestal_size=self.pedestal_size,
            pedestal_pos=self.pedestal_pos,
            scene_output_path=args.scene_output,
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.cam.distance = 1.8
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -25

        self.left_arm = self._build_arm_handles("left_", self.left_base_pos, self.left_base_rot)
        self.right_arm = self._build_arm_handles("right_", self.right_base_pos, self.right_base_rot)
        self.obj_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        self.obj_qpos_adr = int(self.model.jnt_qposadr[self.obj_joint_id])
        self.obj_dof_adr = int(self.model.jnt_dofadr[self.obj_joint_id])
        self.obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj")
        self.marker_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ("obj_point", "contact_point1", "contact_point2", "left_goal", "right_goal", "goal")
        }
        self.support_height_threshold = self.pedestal_pos[2] + self.pedestal_size[2] + args.ground_height_margin

        self.reset(object_pos, object_quat)
        self.optimizer = LambdaContactControlOptimizer(
            mesh_path=str(self.mesh_path),
            obj_mass=args.obj_mass,
            arm_friction=args.arm_friction,
            contact_stiffness=args.contact_stiffness,
            time_step=self.model.opt.timestep,
            sample_num=args.sample_num,
            pos_coef=args.pos_coef,
            ori_coef=args.ori_coef,
            scale_factors=tuple(self.mesh_scale.tolist()),
            support_surface_point=self.support_surface_point,
            support_surface_normal=self.support_surface_normal,
            support_surface_clearance=args.ground_height_margin,
            support_surface_normal_alignment_threshold=args.support_normal_alignment_threshold,
        )
        self.command_dt = (
            float(self.model.opt.timestep)
            * max(int(self.args.mj_steps_per_command), 1)
            * max(int(self.args.command_substeps), 1)
        )
        self._setup_curobo()

    def _resolve_mesh_scale(self, args):
        if args.scale is not None:
            return np.asarray(args.scale, dtype=np.float64)
        key = self.mesh_path.stem
        return DEFAULT_SCALE_MAP.get(key, np.ones(3, dtype=np.float64)).copy()

    def _build_arm_handles(self, prefix, base_pos, base_rot):
        joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}joint{i}")
                for i in range(1, 8)
            ],
            dtype=np.int32,
        )
        qpos_adr = np.array([self.model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32)
        dof_adr = np.array([self.model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32)
        actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}actuator{i}")
                for i in range(1, 8)
            ],
            dtype=np.int32,
        )
        body_ids_by_name = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{name}")
            for name in ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "attachment")
        }
        return ArmHandles(
            prefix=prefix,
            joint_ids=joint_ids,
            qpos_adr=qpos_adr,
            dof_adr=dof_adr,
            actuator_ids=actuator_ids,
            body_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}attachment"),
            tip_geom_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}fingertip"),
            tip_site_id=mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}tip_center"),
            ghost_body_id=mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"{prefix}ghost_tip",
            ),
            base_pos=np.asarray(base_pos, dtype=np.float64).copy(),
            base_rot=np.asarray(base_rot, dtype=np.float64).reshape(3, 3).copy(),
            body_ids_by_name=body_ids_by_name,
        )

    def _build_curobo_world_config_dict(self, arm, include_pedestal=True):
        floor_world_pos = np.array([self.args.scene_center_x, 0.0, -0.05], dtype=np.float64)
        floor_world_rot = np.eye(3, dtype=np.float64)

        floor_local_pos, floor_local_rot = self.world_pose_to_arm_frame(arm, floor_world_pos, floor_world_rot)
        cuboids = {
            "floor": {
                "dims": [4.0, 4.0, 0.1],
                "pose": [*floor_local_pos.tolist(), *mat_to_quat_wxyz(floor_local_rot).tolist()],
            }
        }
        if include_pedestal:
            pedestal_world_pos = self.pedestal_pos.copy()
            pedestal_world_rot = np.eye(3, dtype=np.float64)
            pedestal_local_pos, pedestal_local_rot = self.world_pose_to_arm_frame(
                arm,
                pedestal_world_pos,
                pedestal_world_rot,
            )
            cuboids["pedestal"] = {
                "dims": (2.0 * self.pedestal_size).tolist(),
                "pose": [*pedestal_local_pos.tolist(), *mat_to_quat_wxyz(pedestal_local_rot).tolist()],
            }

        return {"cuboid": cuboids}

    def _build_curobo_world_config(self, arm, include_pedestal=True):
        return WorldConfig.from_dict(self._build_curobo_world_config_dict(arm, include_pedestal=include_pedestal))

    def _set_curobo_world_mode(self, world_mode):
        if world_mode not in ("with_pedestal", "floor_only"):
            raise ValueError(f"Unsupported cuRobo world mode: {world_mode}")

        for arm in (self.left_arm, self.right_arm):
            if world_mode == "floor_only":
                static_world = arm.static_world_floor_only
            else:
                static_world = arm.static_world_with_pedestal
            if static_world is None:
                raise RuntimeError(f"cuRobo static world '{world_mode}' is not initialized for {arm.prefix}.")
            arm.static_world = static_world
            arm.current_world_mode = world_mode

    def _build_segment_world_rotation(self, segment_dir, tangent_hint=None):
        return self._build_contact_rotation(segment_dir, tangent_hint=tangent_hint)

    def _get_body_pose(self, body_id, data=None):
        data = self.data if data is None else data
        pos = data.body(body_id).xpos.copy()
        rot = data.body(body_id).xmat.reshape(3, 3).copy()
        return pos, rot

    def _get_obstacle_anchor_pose(self, arm, anchor_name, data=None):
        if anchor_name == "tip_center":
            return self.get_tip_pose(arm, data=data)
        body_id = arm.body_ids_by_name[anchor_name]
        return self._get_body_pose(body_id, data=data)

    def _build_other_arm_obstacle_cuboids(self, target_arm, obstacle_arm):
        cuboids = []
        for name, start_name, end_name, thickness in ARM_OBSTACLE_SEGMENTS:
            start_pos_world, _ = self._get_obstacle_anchor_pose(obstacle_arm, start_name)
            end_pos_world, _ = self._get_obstacle_anchor_pose(obstacle_arm, end_name)
            segment_world = end_pos_world - start_pos_world
            segment_length = float(np.linalg.norm(segment_world))
            if segment_length < 1e-5:
                continue
            center_world = 0.5 * (start_pos_world + end_pos_world)
            seg_rot_world = self._build_segment_world_rotation(segment_world, tangent_hint=WORLD_UP)
            center_local, seg_rot_local = self.world_pose_to_arm_frame(
                target_arm,
                center_world,
                seg_rot_world,
            )
            cuboids.append(
                Cuboid(
                    name=f"{obstacle_arm.prefix}{name}_obs",
                    pose=[*center_local.tolist(), *mat_to_quat_wxyz(seg_rot_local).tolist()],
                    dims=[
                        float(thickness),
                        float(thickness),
                        float(segment_length + ARM_OBSTACLE_LENGTH_PADDING),
                    ],
                )
            )
        return cuboids

    def _update_inter_arm_worlds(self):
        for target_arm, obstacle_arm in (
            (self.left_arm, self.right_arm),
            (self.right_arm, self.left_arm),
        ):
            world = target_arm.static_world.clone()
            for obstacle in self._build_other_arm_obstacle_cuboids(target_arm, obstacle_arm):
                world.add_obstacle(obstacle)
            target_arm.current_world = world
            target_arm.mpc.update_world(world)
            target_arm.ik_solver.update_world(world)

    def _setup_curobo_arm(self, arm):
        world_config = self._build_curobo_world_config_dict(arm, include_pedestal=True)
        mpc_config = MpcSolverConfig.load_from_robot_config(
            self.args.curobo_robot_cfg,
            world_config,
            store_rollouts=True,
            step_dt=self.command_dt,
            self_collision_check=not self.args.disable_curobo_self_collision,
            collision_cache={"obb": 16},
            collision_activation_distance=self.args.curobo_collision_activation_distance,
            use_cuda_graph=not self.args.disable_curobo_cuda_graph,
            use_cuda_graph_metrics=not self.args.disable_curobo_cuda_graph,
        )
        arm.mpc = MpcSolver(mpc_config)
        arm.mpc.enable_pose_cost(enable=True)
        arm.mpc.enable_cspace_cost(enable=not self.args.pose_only_mpc)
        arm.curobo_joint_names = tuple(arm.mpc.joint_names)
        arm.retract_cfg = _tensor_to_numpy(arm.mpc.rollout_fn.dynamics_model.retract_config).reshape(-1).astype(np.float64)

        current_q_curobo = build_curobo_state(PANDA_HOME_Q, arm.curobo_joint_names, arm.retract_cfg)
        start_state = make_joint_state(arm.mpc, current_q_curobo)
        current_hand_pos_world, current_hand_rot_world = self.get_hand_pose(arm)
        current_hand_pos_local, current_hand_rot_local = self.world_pose_to_arm_frame(
            arm,
            current_hand_pos_world,
            current_hand_rot_world,
        )
        current_goal_pose = make_pose(
            arm.mpc,
            current_hand_pos_local,
            mat_to_quat_wxyz(current_hand_rot_local),
        )
        goal = Goal(
            current_state=start_state.clone(),
            goal_state=start_state.clone(),
            goal_pose=current_goal_pose,
        )
        arm.goal_buffer = arm.mpc.setup_solve_single(goal, 1)
        arm.mpc.update_goal(arm.goal_buffer)
        arm.static_world_with_pedestal = self._build_curobo_world_config(arm, include_pedestal=True)
        arm.static_world_floor_only = self._build_curobo_world_config(arm, include_pedestal=False)
        arm.static_world = arm.static_world_with_pedestal
        arm.current_world = arm.static_world.clone()
        arm.current_world_mode = "with_pedestal"

        ik_config = IKSolverConfig.load_from_robot_config(
            self.args.curobo_robot_cfg,
            world_config,
            position_threshold=self.args.ik_pos_tol,
            rotation_threshold=self.args.ik_rot_tol,
            num_seeds=self.args.ik_num_seeds,
            self_collision_check=not self.args.disable_curobo_self_collision,
            self_collision_opt=not self.args.disable_curobo_self_collision,
            collision_cache={"obb": 16},
            collision_activation_distance=self.args.curobo_collision_activation_distance,
            use_cuda_graph=False,
            regularization=True,
        )
        arm.ik_solver = IKSolver(ik_config)
        self.set_ghost_pose(arm, current_hand_pos_world, current_hand_rot_world)

    def _setup_curobo(self):
        self._setup_curobo_arm(self.left_arm)
        self._setup_curobo_arm(self.right_arm)
        self._update_inter_arm_worlds()

    def _set_mpc_goal_mode(self, use_cspace_goal):
        enable_cspace = bool(use_cspace_goal) and (not self.args.pose_only_mpc)
        for arm in (self.left_arm, self.right_arm):
            arm.mpc.enable_pose_cost(enable=True)
            arm.mpc.enable_cspace_cost(enable=enable_cspace)

    def reset(self, object_pos, object_quat):
        self.data.qpos[self.left_arm.qpos_adr] = PANDA_HOME_Q
        self.data.qpos[self.right_arm.qpos_adr] = PANDA_HOME_Q
        self.data.ctrl[self.left_arm.actuator_ids] = PANDA_HOME_Q
        self.data.ctrl[self.right_arm.actuator_ids] = PANDA_HOME_Q
        self.data.qpos[self.obj_qpos_adr : self.obj_qpos_adr + 7] = np.hstack([object_pos, object_quat])
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.set_ghost_pose(self.left_arm, *self.get_hand_pose(self.left_arm))
        self.set_ghost_pose(self.right_arm, *self.get_hand_pose(self.right_arm))
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def is_running(self):
        if self.viewer is None:
            return True
        is_running = getattr(self.viewer, "is_running", None)
        if callable(is_running):
            return bool(is_running())
        return True

    def get_object_pose(self):
        qpos = self.data.qpos[self.obj_qpos_adr : self.obj_qpos_adr + 7].copy()
        pos = qpos[:3]
        quat = qpos[3:]
        return pos, quat, quat_wxyz_to_mat(quat)

    def get_tip_pos(self, arm):
        return self.data.site_xpos[arm.tip_site_id].copy()

    def get_tip_pose(self, arm, data=None):
        data = self.data if data is None else data
        pos = data.site_xpos[arm.tip_site_id].copy()
        rot = data.site_xmat[arm.tip_site_id].reshape(3, 3).copy()
        return pos, rot

    def get_hand_pose(self, arm, data=None):
        data = self.data if data is None else data
        pos = data.body(arm.body_id).xpos.copy()
        rot = data.body(arm.body_id).xmat.reshape(3, 3).copy()
        return pos, rot

    def world_pose_to_arm_frame(self, arm, pos_world, rot_world):
        pos_world = np.asarray(pos_world, dtype=np.float64).reshape(3)
        rot_world = _project_to_rotation_matrix(rot_world)
        pos_local = arm.base_rot.T @ (pos_world - arm.base_pos)
        rot_local = arm.base_rot.T @ rot_world
        return pos_local, _project_to_rotation_matrix(rot_local)

    def arm_pose_to_world_frame(self, arm, pos_local, rot_local):
        pos_local = np.asarray(pos_local, dtype=np.float64).reshape(3)
        rot_local = _project_to_rotation_matrix(rot_local)
        pos_world = arm.base_pos + arm.base_rot @ pos_local
        rot_world = arm.base_rot @ rot_local
        return pos_world, _project_to_rotation_matrix(rot_world)

    def tip_target_to_hand_pose(self, tip_pos_world, tip_rot_world):
        tip_pos_world = np.asarray(tip_pos_world, dtype=np.float64).reshape(3)
        tip_rot_world = _project_to_rotation_matrix(tip_rot_world)
        hand_pos_world = tip_pos_world - tip_rot_world[:, 2] * TIP_CENTER_OFFSET
        return hand_pos_world, tip_rot_world

    def world_to_object(self, world_point):
        obj_pos, _, obj_rot = self.get_object_pose()
        return obj_rot.T @ (np.asarray(world_point, dtype=np.float64) - obj_pos)

    @staticmethod
    def _build_contact_rotation(approach_dir, tangent_hint=None):
        z_axis = _normalize(approach_dir)
        if np.linalg.norm(z_axis) < 1e-8:
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        candidate_hints = []
        if tangent_hint is not None:
            candidate_hints.append(np.asarray(tangent_hint, dtype=np.float64))
        candidate_hints.extend(
            [
                WORLD_UP,
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            ]
        )

        x_axis = None
        for hint in candidate_hints:
            tangent = _project_to_plane(hint, z_axis)
            if np.linalg.norm(tangent) > 1e-6:
                x_axis = _normalize(tangent)
                break
        if x_axis is None:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        y_axis = _normalize(np.cross(z_axis, x_axis))
        if np.linalg.norm(y_axis) < 1e-8:
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = _normalize(np.cross(y_axis, z_axis))
        return _project_to_rotation_matrix(np.column_stack([x_axis, y_axis, z_axis]))

    @staticmethod
    def _max_normal_force(contact_items):
        if not contact_items:
            return 0.0
        return float(max(float(item.get("normal_force", 0.0)) for item in contact_items))

    def _ordered_contact_data(self, contact_points_local, normals_local, object_pos, object_rot):
        contact_points_local = np.asarray(contact_points_local, dtype=np.float64).reshape(-1, 3)
        normals_local = np.asarray(normals_local, dtype=np.float64).reshape(-1, 3)
        contact_points_world = (object_rot @ contact_points_local.T).T + object_pos[None, :]
        left_tip = self.get_tip_pos(self.left_arm)
        right_tip = self.get_tip_pos(self.right_arm)
        keep_cost = np.linalg.norm(left_tip - contact_points_world[0]) + np.linalg.norm(
            right_tip - contact_points_world[1]
        )
        swap_cost = np.linalg.norm(left_tip - contact_points_world[1]) + np.linalg.norm(
            right_tip - contact_points_world[0]
        )
        order = np.array([0, 1], dtype=int) if keep_cost <= swap_cost else np.array([1, 0], dtype=int)
        return (
            contact_points_local[order],
            normals_local[order],
            contact_points_world[order],
        )

    def _compute_preferred_tip_approach(self, inward_normal_world):
        downward_dir = -WORLD_UP
        inward_normal_world = _normalize(inward_normal_world)
        max_pitch = min(float(self.args.fingertip_max_pitch), np.pi * 0.49)
        return _rotate_vector_toward(downward_dir, inward_normal_world, max_pitch)

    def _compute_fingertip_targets(self, contact_points_world, inward_normals_world, center_offset):
        contact_points_world = np.asarray(contact_points_world, dtype=np.float64).reshape(-1, 3)
        inward_normals_world = np.asarray(inward_normals_world, dtype=np.float64).reshape(-1, 3)
        if contact_points_world.shape[0] != 2:
            raise ValueError(f"Expected exactly 2 contact points, got {contact_points_world.shape[0]}.")

        pair_axis = contact_points_world[1] - contact_points_world[0]
        if np.linalg.norm(pair_axis) < 1e-8:
            pair_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        pair_axis = _normalize(pair_axis)

        outward_normals_world = -inward_normals_world
        left_pos = contact_points_world[0] + outward_normals_world[0] * float(center_offset)
        right_pos = contact_points_world[1] + outward_normals_world[1] * float(center_offset)
        # Bias the fingertip to point mostly downward, while allowing a limited
        # inward pitch toward the contact normal to make the IK target easier to reach.
        left_approach = self._compute_preferred_tip_approach(inward_normals_world[0])
        right_approach = self._compute_preferred_tip_approach(inward_normals_world[1])
        left_rot = self._build_contact_rotation(left_approach, tangent_hint=pair_axis)
        right_rot = self._build_contact_rotation(right_approach, tangent_hint=-pair_axis)
        return (left_pos, left_rot), (right_pos, right_rot)

    def _targets_from_object_pose(
        self,
        contact_points_local,
        normals_local,
        object_pos,
        object_rot,
        center_offset,
    ):
        contact_points_world = (object_rot @ np.asarray(contact_points_local, dtype=np.float64).T).T + object_pos[None, :]
        inward_normals_world = (object_rot @ np.asarray(normals_local, dtype=np.float64).T).T
        return self._compute_fingertip_targets(contact_points_world, inward_normals_world, center_offset)

    def set_marker(self, name, pos, quat=None):
        body_id = self.marker_body_ids[name]
        self.model.body_pos[body_id] = np.asarray(pos, dtype=np.float64)
        if quat is not None:
            self.model.body_quat[body_id] = np.asarray(quat, dtype=np.float64)

    def set_ghost_pose(self, arm, hand_pos_world, hand_rot_world):
        if arm.ghost_body_id < 0:
            return
        self.model.body_pos[arm.ghost_body_id] = np.asarray(hand_pos_world, dtype=np.float64)
        self.model.body_quat[arm.ghost_body_id] = mat_to_quat_wxyz(hand_rot_world)

    @staticmethod
    def _constraint_scalar(value):
        if value is None:
            return 0.0
        array = np.asarray(_tensor_to_numpy(value), dtype=np.float64)
        if array.size == 0:
            return 0.0
        return float(np.max(array))

    def _diagnose_ik_solution(self, arm, q_curobo):
        joint_state = make_joint_state(arm.ik_solver, q_curobo)
        rollout_fn = arm.ik_solver.rollout_fn
        aug_state = rollout_fn._get_augmented_state(joint_state)

        bound_constraint = self._constraint_scalar(rollout_fn.bound_constraint.forward(aug_state.state_seq))

        world_constraint = 0.0
        static_world_constraint = 0.0
        primitive_constraint = getattr(rollout_fn, "primitive_collision_constraint", None)
        if primitive_constraint is not None and getattr(primitive_constraint, "enabled", False):
            world_constraint = self._constraint_scalar(
                primitive_constraint.forward(aug_state.robot_spheres, env_query_idx=None)
            )
            if arm.static_world is not None:
                restore_world = arm.current_world if arm.current_world is not None else arm.static_world
                arm.ik_solver.update_world(arm.static_world)
                try:
                    static_world_constraint = self._constraint_scalar(
                        primitive_constraint.forward(aug_state.robot_spheres, env_query_idx=None)
                    )
                finally:
                    arm.ik_solver.update_world(restore_world)

        self_constraint = 0.0
        self_collision_constraint = getattr(rollout_fn, "robot_self_collision_constraint", None)
        if self_collision_constraint is not None and getattr(self_collision_constraint, "enabled", False):
            self_constraint = self._constraint_scalar(self_collision_constraint.forward(aug_state.robot_spheres))

        total_constraint = bound_constraint + world_constraint + self_constraint
        eps = 1e-6
        failure_reasons = []
        static_world_reason = "floor clearance" if arm.current_world_mode == "floor_only" else "pedestal/floor clearance"
        if bound_constraint > eps:
            failure_reasons.append("joint bound")
        if world_constraint > eps:
            if static_world_constraint > eps and world_constraint > static_world_constraint + eps:
                failure_reasons.append(f"{static_world_reason} + other-arm clearance")
            elif static_world_constraint > eps:
                failure_reasons.append(static_world_reason)
            else:
                failure_reasons.append("other-arm clearance")
        if self_constraint > eps:
            failure_reasons.append("self-collision clearance")
        if not failure_reasons and total_constraint > eps:
            failure_reasons.append("feasibility constraint")

        return {
            "constraint_total": total_constraint,
            "bound_constraint": bound_constraint,
            "world_constraint": world_constraint,
            "static_world_constraint": static_world_constraint,
            "self_constraint": self_constraint,
            "failure_reason": ", ".join(failure_reasons),
        }

    def _current_arm_q_curobo(self, arm):
        return build_curobo_state(
            self.data.qpos[arm.qpos_adr].copy(),
            arm.curobo_joint_names,
            arm.retract_cfg,
        )

    def solve_arm_ik(self, arm, target_tip_pos_world, target_tip_rot_world):
        target_hand_pos_world, target_hand_rot_world = self.tip_target_to_hand_pose(
            target_tip_pos_world,
            target_tip_rot_world,
        )
        target_hand_pos_local, target_hand_rot_local = self.world_pose_to_arm_frame(
            arm,
            target_hand_pos_world,
            target_hand_rot_world,
        )
        goal_pose = make_pose(
            arm.ik_solver,
            target_hand_pos_local,
            mat_to_quat_wxyz(target_hand_rot_local),
        )
        current_q_curobo = self._current_arm_q_curobo(arm)
        retract_cfg = make_joint_tensor(arm.ik_solver, current_q_curobo, extra_dim=False)
        seed_cfg = make_joint_tensor(arm.ik_solver, current_q_curobo, extra_dim=True)
        result = arm.ik_solver.solve_single(
            goal_pose,
            retract_config=retract_cfg,
            seed_config=seed_cfg,
            return_seeds=1,
            num_seeds=self.args.ik_num_seeds,
            use_nn_seed=False,
            newton_iters=self.args.ik_max_iters,
        )

        ik_success = bool(_tensor_to_numpy(result.success).reshape(-1)[0])
        ik_pos_err = float(_tensor_to_numpy(result.position_error).reshape(-1)[0])
        ik_rot_err = float(_tensor_to_numpy(result.rotation_error).reshape(-1)[0])
        solution = _tensor_to_numpy(result.solution).reshape(-1, len(arm.curobo_joint_names))[0]
        q_mj = extract_mujoco_arm_configuration(solution, arm.curobo_joint_names)
        ik_diag = {
            "constraint_total": 0.0,
            "bound_constraint": 0.0,
            "world_constraint": 0.0,
            "static_world_constraint": 0.0,
            "self_constraint": 0.0,
            "failure_reason": "",
        }
        if not ik_success:
            ik_diag = self._diagnose_ik_solution(arm, solution)

        fk_state = arm.ik_solver.fk(
            torch.tensor(
                np.asarray(solution, dtype=np.float32).reshape(1, -1),
                device=arm.ik_solver.tensor_args.device,
                dtype=arm.ik_solver.tensor_args.dtype,
            )
        )
        solved_hand_pos_local, solved_hand_quat_local = _extract_pose_from_kinematics(fk_state)
        solved_hand_rot_local = quat_wxyz_to_mat(solved_hand_quat_local)
        solved_hand_pos_world, solved_hand_rot_world = self.arm_pose_to_world_frame(
            arm,
            solved_hand_pos_local,
            solved_hand_rot_local,
        )
        solved_tip_pos_world = solved_hand_pos_world + solved_hand_rot_world[:, 2] * TIP_CENTER_OFFSET
        solved_tip_rot_world = solved_hand_rot_world.copy()
        self.set_ghost_pose(arm, solved_hand_pos_world, solved_hand_rot_world)

        return ArmIkResult(
            q_mj=q_mj,
            success=ik_success,
            position_error=ik_pos_err,
            rotation_error=ik_rot_err,
            target_hand_pos_world=target_hand_pos_world,
            target_hand_rot_world=target_hand_rot_world,
            solved_hand_pos_world=solved_hand_pos_world,
            solved_hand_rot_world=solved_hand_rot_world,
            solved_tip_pos_world=solved_tip_pos_world,
            solved_tip_rot_world=solved_tip_rot_world,
            constraint_total=ik_diag["constraint_total"],
            bound_constraint=ik_diag["bound_constraint"],
            world_constraint=ik_diag["world_constraint"],
            static_world_constraint=ik_diag["static_world_constraint"],
            self_constraint=ik_diag["self_constraint"],
            failure_reason=ik_diag["failure_reason"],
        )

    def build_pose_goal_result(self, arm, target_tip_pos_world, target_tip_rot_world, reference_q_mj=None):
        target_tip_pos_world = np.asarray(target_tip_pos_world, dtype=np.float64).reshape(3)
        target_tip_rot_world = _project_to_rotation_matrix(target_tip_rot_world)
        target_hand_pos_world, target_hand_rot_world = self.tip_target_to_hand_pose(
            target_tip_pos_world,
            target_tip_rot_world,
        )
        self.set_ghost_pose(arm, target_hand_pos_world, target_hand_rot_world)
        if reference_q_mj is None:
            q_mj = self.data.qpos[arm.qpos_adr].copy()
        else:
            q_mj = np.asarray(reference_q_mj, dtype=np.float64).reshape(-1).copy()
        return ArmIkResult(
            q_mj=q_mj,
            success=True,
            position_error=0.0,
            rotation_error=0.0,
            target_hand_pos_world=target_hand_pos_world,
            target_hand_rot_world=target_hand_rot_world,
            solved_hand_pos_world=target_hand_pos_world.copy(),
            solved_hand_rot_world=target_hand_rot_world.copy(),
            solved_tip_pos_world=target_tip_pos_world.copy(),
            solved_tip_rot_world=target_tip_rot_world.copy(),
            constraint_total=0.0,
            bound_constraint=0.0,
            world_constraint=0.0,
            static_world_constraint=0.0,
            self_constraint=0.0,
            failure_reason="",
        )

    def _update_arm_mpc_goal(self, arm, ik_result):
        goal_hand_pos_local, goal_hand_rot_local = self.world_pose_to_arm_frame(
            arm,
            ik_result.solved_hand_pos_world,
            ik_result.solved_hand_rot_world,
        )
        goal_pose = make_pose(
            arm.mpc,
            goal_hand_pos_local,
            mat_to_quat_wxyz(goal_hand_rot_local),
        )
        goal_joint_curobo = build_curobo_state(
            ik_result.q_mj,
            arm.curobo_joint_names,
            arm.retract_cfg,
        )
        goal_state = make_joint_state(arm.mpc, goal_joint_curobo)
        arm.goal_buffer.goal_pose.copy_(goal_pose)
        arm.goal_buffer.goal_state.copy_(goal_state)
        arm.mpc.update_goal(arm.goal_buffer)

    def _step_arm_mpc(self, arm):
        current_q_curobo = self._current_arm_q_curobo(arm)
        current_state = make_joint_state(arm.mpc, current_q_curobo)
        result = arm.mpc.step(current_state, max_attempts=self.args.mpc_max_attempts)
        command_q_curobo = _tensor_to_numpy(result.action.position).reshape(-1, len(arm.curobo_joint_names))[0]
        command_q_mj = extract_mujoco_arm_configuration(command_q_curobo, arm.curobo_joint_names)
        pose_error = float("nan")
        if result.metrics is not None and hasattr(result.metrics, "pose_error"):
            pose_error = _scalar(result.metrics.pose_error)
        return command_q_mj, pose_error

    def set_arm_targets(self, left_q, right_q):
        self.data.ctrl[self.left_arm.actuator_ids] = left_q
        self.data.ctrl[self.right_arm.actuator_ids] = right_q

    def step_sim(self, num_steps=1):
        for _ in range(max(int(num_steps), 1)):
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
            if self.args.real_time and self.viewer is not None:
                time.sleep(self.model.opt.timestep)

    def extract_object_contacts(self):
        obj_pos, _, obj_rot = self.get_object_pose()
        contacts = {"left": [], "right": []}
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom_ids = {int(contact.geom1), int(contact.geom2)}
            if self.obj_geom_id not in geom_ids:
                continue
            other_geom = int(contact.geom2) if int(contact.geom1) == self.obj_geom_id else int(contact.geom1)
            if other_geom == self.left_arm.tip_geom_id:
                key = "left"
            elif other_geom == self.right_arm.tip_geom_id:
                key = "right"
            else:
                continue

            world_pos = np.asarray(contact.pos, dtype=np.float64).copy()
            local_pos = obj_rot.T @ (world_pos - obj_pos)
            contact_force = np.zeros(6, dtype=np.float64)
            if hasattr(mujoco, "mj_contactForce"):
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)
            normal_force = float(abs(contact_force[0]))
            tangential_force = float(np.linalg.norm(contact_force[1:3]))
            contacts[key].append(
                {
                    "world_pos": world_pos,
                    "local_pos": local_pos,
                    "dist": float(contact.dist),
                    "force_local": contact_force.copy(),
                    "normal_force": normal_force,
                    "tangential_force": tangential_force,
                }
            )
        return contacts

    def _tip_target_error(self, arm, target_pos, target_rot=None):
        current_pos, current_rot = self.get_tip_pose(arm)
        pos_err = float(np.linalg.norm(np.asarray(target_pos, dtype=np.float64) - current_pos))
        rot_err = 0.0
        if target_rot is not None:
            rot_err = float(np.linalg.norm(_rotation_error(current_rot, target_rot)))
        return pos_err, rot_err

    def _run_dual_arm_stage(
        self,
        label,
        max_steps,
        target_fn,
        pos_tol,
        rot_tol=None,
        success_fn=None,
        solve_ik_once=False,
        use_ik=True,
        fixed_joint_goals=None,
        world_mode="with_pedestal",
    ):
        rot_tol = self.args.ik_rot_tol if rot_tol is None else float(rot_tol)
        if int(max_steps) <= 0:
            return False, {}

        self._set_curobo_world_mode(world_mode)
        self._set_mpc_goal_mode(use_cspace_goal=use_ik)
        last_report_step = -1
        info = {}
        cached_targets = None
        cached_ik = None
        cspace_goal_enabled = bool(use_ik) and (not self.args.pose_only_mpc)
        if fixed_joint_goals is None:
            left_joint_goal = None
            right_joint_goal = None
        else:
            left_joint_goal, right_joint_goal = fixed_joint_goals
        for step in range(max_steps):
            if not self.is_running():
                break

            self._update_inter_arm_worlds()
            if solve_ik_once:
                if cached_targets is None:
                    cached_targets = target_fn(step)
                    (left_target, left_rot), (right_target, right_rot) = cached_targets
                    if use_ik:
                        left_ik = self.solve_arm_ik(self.left_arm, left_target, left_rot)
                        right_ik = self.solve_arm_ik(self.right_arm, right_target, right_rot)
                    else:
                        left_ik = self.build_pose_goal_result(
                            self.left_arm,
                            left_target,
                            left_rot,
                            reference_q_mj=left_joint_goal,
                        )
                        right_ik = self.build_pose_goal_result(
                            self.right_arm,
                            right_target,
                            right_rot,
                            reference_q_mj=right_joint_goal,
                        )
                    cached_ik = (left_ik, right_ik)
                    self._update_arm_mpc_goal(self.left_arm, left_ik)
                    self._update_arm_mpc_goal(self.right_arm, right_ik)
                else:
                    (left_target, left_rot), (right_target, right_rot) = cached_targets
                    left_ik, right_ik = cached_ik
            else:
                (left_target, left_rot), (right_target, right_rot) = target_fn(step)
                if use_ik:
                    left_ik = self.solve_arm_ik(self.left_arm, left_target, left_rot)
                    right_ik = self.solve_arm_ik(self.right_arm, right_target, right_rot)
                else:
                    left_ik = self.build_pose_goal_result(
                        self.left_arm,
                        left_target,
                        left_rot,
                        reference_q_mj=left_joint_goal,
                    )
                    right_ik = self.build_pose_goal_result(
                        self.right_arm,
                        right_target,
                        right_rot,
                        reference_q_mj=right_joint_goal,
                    )
                self._update_arm_mpc_goal(self.left_arm, left_ik)
                self._update_arm_mpc_goal(self.right_arm, right_ik)

            left_q_cmd, left_mpc_pose_err = self._step_arm_mpc(self.left_arm)
            right_q_cmd, right_mpc_pose_err = self._step_arm_mpc(self.right_arm)
            self.set_arm_targets(left_q_cmd, right_q_cmd)
            self.step_sim(self.args.mj_steps_per_command * self.args.command_substeps)

            self.set_marker("left_goal", left_ik.solved_tip_pos_world)
            self.set_marker("right_goal", right_ik.solved_tip_pos_world)
            self.set_marker("obj_point", self.get_object_pose()[0])
            mujoco.mj_forward(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()

            left_err, left_rot_err = self._tip_target_error(
                self.left_arm,
                left_ik.solved_tip_pos_world,
                left_ik.solved_tip_rot_world,
            )
            right_err, right_rot_err = self._tip_target_error(
                self.right_arm,
                right_ik.solved_tip_pos_world,
                right_ik.solved_tip_rot_world,
            )
            contacts = self.extract_object_contacts()
            info = {
                "step": step,
                "contacts": contacts,
                "left_force": self._max_normal_force(contacts["left"]),
                "right_force": self._max_normal_force(contacts["right"]),
                "left_pos_err": left_err,
                "right_pos_err": right_err,
                "left_rot_err": left_rot_err,
                "right_rot_err": right_rot_err,
                "left_ik_ok": bool(left_ik.success),
                "right_ik_ok": bool(right_ik.success),
                "left_ik_pos_err": float(left_ik.position_error),
                "right_ik_pos_err": float(right_ik.position_error),
                "left_ik_rot_err": float(left_ik.rotation_error),
                "right_ik_rot_err": float(right_ik.rotation_error),
                "left_ik_constraint_total": float(left_ik.constraint_total),
                "right_ik_constraint_total": float(right_ik.constraint_total),
                "left_ik_bound_constraint": float(left_ik.bound_constraint),
                "right_ik_bound_constraint": float(right_ik.bound_constraint),
                "left_ik_world_constraint": float(left_ik.world_constraint),
                "right_ik_world_constraint": float(right_ik.world_constraint),
                "left_ik_static_world_constraint": float(left_ik.static_world_constraint),
                "right_ik_static_world_constraint": float(right_ik.static_world_constraint),
                "left_ik_self_constraint": float(left_ik.self_constraint),
                "right_ik_self_constraint": float(right_ik.self_constraint),
                "left_ik_failure_reason": str(left_ik.failure_reason),
                "right_ik_failure_reason": str(right_ik.failure_reason),
                "left_mpc_pose_err": float(left_mpc_pose_err),
                "right_mpc_pose_err": float(right_mpc_pose_err),
                "left_goal_q_mj": left_ik.q_mj.copy(),
                "right_goal_q_mj": right_ik.q_mj.copy(),
                "object_pos": self.get_object_pose()[0].copy(),
            }
            if step == 0 or step == max_steps - 1 or step - last_report_step >= 40:
                print(
                    f"[{label}] step={step:04d} "
                    f"left_err={left_err:.4f}/{left_rot_err:.4f} "
                    f"right_err={right_err:.4f}/{right_rot_err:.4f} "
                    f"left_force={info['left_force']:.3f} right_force={info['right_force']:.3f} "
                    f"left_ik={left_ik.success} right_ik={right_ik.success} "
                    f"left_ik_res={left_ik.position_error:.4f}/{left_ik.rotation_error:.4f} "
                    f"right_ik_res={right_ik.position_error:.4f}/{right_ik.rotation_error:.4f} "
                    f"left_mpc={left_mpc_pose_err:.4f} right_mpc={right_mpc_pose_err:.4f}"
                )
                if step == 0:
                    print(
                        f"  stage_cfg: world_mode={world_mode} "
                        f"use_ik={use_ik} cspace_goal={cspace_goal_enabled}"
                    )
                if not left_ik.success and left_ik.failure_reason:
                    print(
                        "  left_ik_diag: "
                        f"reason={left_ik.failure_reason} "
                        f"constraint={left_ik.constraint_total:.4f} "
                        f"bound={left_ik.bound_constraint:.4f} "
                        f"world={left_ik.world_constraint:.4f} "
                        f"static_world={left_ik.static_world_constraint:.4f} "
                        f"self={left_ik.self_constraint:.4f}"
                    )
                if not right_ik.success and right_ik.failure_reason:
                    print(
                        "  right_ik_diag: "
                        f"reason={right_ik.failure_reason} "
                        f"constraint={right_ik.constraint_total:.4f} "
                        f"bound={right_ik.bound_constraint:.4f} "
                        f"world={right_ik.world_constraint:.4f} "
                        f"static_world={right_ik.static_world_constraint:.4f} "
                        f"self={right_ik.self_constraint:.4f}"
                    )
                last_report_step = step

            pose_ok = (
                left_err < pos_tol
                and right_err < pos_tol
                and left_rot_err < rot_tol
                and right_rot_err < rot_tol
            )
            if success_fn is None:
                if pose_ok:
                    return True, info
            elif success_fn(info):
                return True, info

        return False, info

    def run(self):
        obj_pos, obj_quat, obj_rot = self.get_object_pose()
        gravity_local = np.hstack([obj_rot.T @ np.array([0.0, 0.0, -self.args.obj_mass * 9.81]), np.zeros(3)])

        visible_idx = self.optimizer.get_availble_point_idx(
            pos=obj_pos,
            R=obj_rot,
            target_pos=obj_pos,
            threshold=self.support_height_threshold,
        )
        contact_points_local, normals_local, total_cost, region_score, antipodal_margin = self.optimizer.choose_contact_set(
            visible_face_idx=visible_idx,
            object_pos=obj_pos,
            object_rot=obj_rot,
        )
        grasp_result = self.optimizer.last_grasp_result
        static_result = None
        if grasp_result is not None:
            static_result = self.optimizer.solve_static_equilibrium(grasp_result, gravity_local)

        contact_points_local = np.asarray(contact_points_local, dtype=np.float64).reshape(-1, 3)
        normals_local = np.asarray(normals_local, dtype=np.float64).reshape(-1, 3)
        if contact_points_local.shape[0] != 2:
            raise RuntimeError(f"Expected 2 grasp contacts, got {contact_points_local.shape[0]}")

        contact_points_local, normals_local, contact_points_world = self._ordered_contact_data(
            contact_points_local,
            normals_local,
            obj_pos,
            obj_rot,
        )
        inward_normals_world = (obj_rot @ normals_local.T).T
        lift_delta = np.array([0.0, 0.0, self.args.lift_height], dtype=np.float64)
        pregrasp_offset = TIP_RADIUS + self.args.pregrasp_offset
        touch_offset = TIP_RADIUS + self.args.touch_offset
        squeeze_offset = max(TIP_RADIUS - self.args.squeeze_depth, 0.001)

        self.set_marker("contact_point1", contact_points_world[0])
        self.set_marker("contact_point2", contact_points_world[1])
        self.set_marker("goal", obj_pos, obj_quat)
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

        required_normal_force = (
            0.35 * self.args.obj_mass * 9.81
            if self.args.min_normal_force is None
            else float(self.args.min_normal_force)
        )
        if static_result is not None and static_result["valid"]:
            modeled_normal = float(np.max(np.asarray(static_result["contact_forces_local"], dtype=np.float64)[:, 0]))
            required_normal_force = max(required_normal_force, 0.5 * modeled_normal)

        print("Generated scene:", self.scene_path)
        print("Mesh:", self.mesh_path)
        print("Scale:", self.mesh_scale)
        print("Object pose:", obj_pos, obj_quat)
        print("Contact points local:\n", contact_points_local)
        print("Contact points world:\n", contact_points_world)
        print("Inward normals world:\n", inward_normals_world)
        print(
            f"Grasp score: total_cost={float(total_cost):.6f}, "
            f"region_score={float(region_score):.6f}, antipodal_margin={float(antipodal_margin):.6f}"
        )
        print(f"Required normal force per fingertip: {required_normal_force:.3f} N")
        if static_result is not None:
            print(
                f"Static equilibrium: valid={static_result['valid']} "
                f"residual_norm={static_result['residual_norm']:.6f} "
                f"solve_time={static_result['solve_time']:.4f}s"
            )

        def current_object_target(center_offset):
            def _target(_step):
                curr_pos, _, curr_rot = self.get_object_pose()
                return self._targets_from_object_pose(
                    contact_points_local,
                    normals_local,
                    curr_pos,
                    curr_rot,
                    center_offset=center_offset,
                )

            return _target

        print("Stage 1: cuRobo MPC approach to pre-grasp pose")
        pregrasp_ok, pregrasp_info = self._run_dual_arm_stage(
            "pregrasp",
            self.args.approach_steps,
            current_object_target(pregrasp_offset),
            self.args.target_tol,
            solve_ik_once=True,
        )
        if not pregrasp_ok:
            print("Pre-grasp stage reached its step limit; continuing with the best available state.")

        print("Stage 2: cuRobo MPC approach to contact pose")
        pregrasp_ik_ready = bool(pregrasp_info.get("left_ik_ok")) and bool(pregrasp_info.get("right_ik_ok"))
        fixed_joint_goals = None
        if pregrasp_ik_ready:
            fixed_joint_goals = (
                np.asarray(pregrasp_info["left_goal_q_mj"], dtype=np.float64).copy(),
                np.asarray(pregrasp_info["right_goal_q_mj"], dtype=np.float64).copy(),
            )
        contact_ok, _ = self._run_dual_arm_stage(
            "contact",
            self.args.touch_steps,
            current_object_target(touch_offset),
            self.args.target_tol,
            solve_ik_once=True,
            use_ik=not pregrasp_ik_ready,
            fixed_joint_goals=fixed_joint_goals,
            world_mode="floor_only",
        )
        if not contact_ok:
            print("Contact stage reached its step limit; proceeding to squeeze with the current fingertip pose.")

        print("Stage 3: squeeze until both fingertips build force")
        stable_contact_steps = 0

        def squeeze_success(info):
            nonlocal stable_contact_steps
            force_ready = (
                info["left_force"] >= required_normal_force
                and info["right_force"] >= required_normal_force
            )
            stable_contact_steps = stable_contact_steps + 1 if force_ready else 0
            return stable_contact_steps >= self.args.contact_stable_steps

        squeeze_ok, squeeze_info = self._run_dual_arm_stage(
            "squeeze",
            self.args.squeeze_steps,
            current_object_target(squeeze_offset),
            self.args.target_tol * 1.5,
            rot_tol=self.args.ik_rot_tol * 1.5,
            success_fn=squeeze_success,
            use_ik=not pregrasp_ik_ready,
            fixed_joint_goals=fixed_joint_goals,
            world_mode="floor_only",
        )
        if not squeeze_ok and self.args.squeeze_extra_steps > 0:
            print(
                f"Squeeze stage did not reach the required force within {self.args.squeeze_steps} steps; "
                f"extending by {self.args.squeeze_extra_steps} more steps."
            )
            squeeze_ok, squeeze_info = self._run_dual_arm_stage(
                "squeeze-extend",
                self.args.squeeze_extra_steps,
                current_object_target(squeeze_offset),
                self.args.target_tol * 1.5,
                rot_tol=self.args.ik_rot_tol * 1.5,
                success_fn=squeeze_success,
                use_ik=not pregrasp_ik_ready,
                fixed_joint_goals=fixed_joint_goals,
                world_mode="floor_only",
            )
        if not squeeze_ok:
            print("Squeeze stage did not reach the required bilateral contact force.")
            return

        contacts = squeeze_info["contacts"]
        for key, marker_name in (("left", "contact_point1"), ("right", "contact_point2")):
            if contacts[key]:
                best_contact = max(contacts[key], key=lambda item: item.get("normal_force", 0.0))
                self.set_marker(marker_name, best_contact["world_pos"])
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

        print(
            "Measured contacts after squeeze:",
            {key: len(value) for key, value in contacts.items()},
        )
        for side in ("left", "right"):
            if contacts[side]:
                best_contact = max(contacts[side], key=lambda item: item.get("normal_force", 0.0))
                print(
                    f"  {side}: world={np.array2string(best_contact['world_pos'], precision=4)} "
                    f"local={np.array2string(best_contact['local_pos'], precision=4)} "
                    f"dist={best_contact['dist']:.6f} "
                    f"normal_force={best_contact['normal_force']:.4f}"
                )

        print("Stage 4: lift while maintaining the contact pose")
        lift_start_pos, _, lift_start_rot = self.get_object_pose()
        target_lift_height = obj_pos[2] + self.args.lift_height - self.args.lift_success_margin

        def lift_success(info):
            contact_ok = (
                info["left_force"] > 0.1 * required_normal_force
                and info["right_force"] > 0.1 * required_normal_force
            )
            return info["object_pos"][2] >= target_lift_height and contact_ok

        def lift_target(step):
            alpha = min(1.0, float(step + 1) / max(1, self.args.lift_steps))
            desired_pos = lift_start_pos + alpha * lift_delta
            return self._targets_from_object_pose(
                contact_points_local,
                normals_local,
                desired_pos,
                lift_start_rot,
                center_offset=squeeze_offset,
            )

        lift_ok, _ = self._run_dual_arm_stage(
            "lift",
            self.args.lift_steps,
            lift_target,
            self.args.target_tol * 2.0,
            rot_tol=self.args.ik_rot_tol * 1.5,
            success_fn=lift_success,
            use_ik=not pregrasp_ik_ready,
            fixed_joint_goals=fixed_joint_goals,
        )

        final_obj_pos, final_obj_quat, _ = self.get_object_pose()
        lifted_height = final_obj_pos[2] - obj_pos[2]
        print(
            f"Lift result: success={lift_ok} final_height_gain={lifted_height:.4f} "
            f"target={self.args.lift_height:.4f}"
        )
        print("Final object pose:", final_obj_pos, final_obj_quat)

        if lift_ok and self.args.hold_steps > 0:
            self.step_sim(self.args.hold_steps)


def build_argparser():
    parser = argparse.ArgumentParser(description="Dual Panda MuJoCo grasp demo driven by mlqp_point_v2 and cuRobo.")
    parser.add_argument("--obj", type=str, default="stanford_bunny2", help="Object asset name in envs/assets/objects.")
    parser.add_argument("--mesh", type=str, default=None, help="Absolute or relative mesh path. Overrides --obj.")
    parser.add_argument("--scale", type=float, nargs=3, default=None, help="Mesh scale factors sx sy sz.")
    parser.add_argument("--obj-mass", type=float, default=0.15, help="Object mass used in MuJoCo and grasp scoring.")
    parser.add_argument("--arm-friction", type=float, default=0.9, help="Friction coefficient passed to mlqp_point_v2.")
    parser.add_argument("--contact-stiffness", type=float, default=12.5, help="Contact stiffness passed to mlqp_point_v2.")
    parser.add_argument("--sample-num", type=int, default=96, help="Surface samples used by mlqp_point_v2.")
    parser.add_argument("--pos-coef", type=float, default=1.0, help="Position coefficient for mlqp_point_v2.")
    parser.add_argument("--ori-coef", type=float, default=0.0005, help="Orientation coefficient for mlqp_point_v2.")
    parser.add_argument("--scene-center-x", type=float, default=0.58, help="Midpoint between the two Panda bases.")
    parser.add_argument("--robot-span", type=float, default=0.75, help="Distance between the two Panda bases.") # 間距
    parser.add_argument("--pedestal-pos", type=float, nargs=3, default=(0.58, 0.0, 0.06), help="Central pedestal position.")
    parser.add_argument("--pedestal-size", type=float, nargs=3, default=(0.05, 0.07, 0.06), help="Central pedestal half sizes.")
    parser.add_argument("--object-yaw", type=float, default=0.0, help="Initial object yaw in radians.")
    parser.add_argument("--object-z-offset", type=float, default=0.0, help="Extra object height above the pedestal.")
    parser.add_argument("--initial-object-lift", type=float, default=0.2, help="Extra height added to the pedestal/support under the object.") # 臺子高度
    parser.add_argument("--ground-height-margin", type=float, default=0.003, help="Margin above support top for visible point filtering.")
    parser.add_argument(
        "--support-normal-alignment-threshold",
        type=float,
        default=0.25,
        help="Normal alignment threshold used to reject support-facing bottom contacts.",
    )
    parser.add_argument("--pregrasp-offset", type=float, default=0.01, help="Extra stand-off beyond the fingertip radius.")
    parser.add_argument("--touch-offset", type=float, default=0.004, help="Stand-off used for initial touch.")
    parser.add_argument("--squeeze-depth", type=float, default=0.003, help="Inward squeeze depth relative to fingertip radius.")
    parser.add_argument(
        "--fingertip-max-pitch",
        type=float,
        default=0.35,
        help="Maximum inward pitch, in radians, allowed away from the vertical-down fingertip pose.",
    )
    parser.add_argument("--lift-height", type=float, default=0.06, help="Lift distance after the squeeze stage.")
    parser.add_argument("--approach-steps", type=int, default=320, help="Simulation steps for the pregrasp stage.")
    parser.add_argument("--touch-steps", type=int, default=320, help="Simulation steps for the touch stage.")
    parser.add_argument("--squeeze-steps", type=int, default=480, help="Simulation steps for the squeeze stage.")
    parser.add_argument("--squeeze-extra-steps", type=int, default=360, help="Extra squeeze steps automatically used if the first squeeze window is not enough.")
    parser.add_argument("--lift-steps", type=int, default=320, help="Simulation steps for the lift stage.")
    parser.add_argument("--hold-steps", type=int, default=0, help="Extra simulation steps after lifting.")
    parser.add_argument("--target-tol", type=float, default=0.006, help="Tip target tolerance in meters.")
    parser.add_argument("--ik-pos-tol", type=float, default=0.0015, help="cuRobo IK position threshold in meters.")
    parser.add_argument("--ik-rot-tol", type=float, default=0.10, help="cuRobo IK rotation threshold in radians.")
    parser.add_argument("--ik-max-iters", type=int, default=120, help="Maximum cuRobo gradient iterations used by each IK solve.")
    parser.add_argument("--ik-num-seeds", type=int, default=32, help="Number of cuRobo IK seeds.")
    parser.add_argument("--ik-damping", type=float, default=0.05, help="Legacy option kept for CLI compatibility; unused with cuRobo IK.")
    parser.add_argument("--ik-step-scale", type=float, default=0.7, help="Legacy option kept for CLI compatibility; unused with cuRobo IK.")
    parser.add_argument("--ik-home-weight", type=float, default=0.01, help="Legacy option kept for CLI compatibility; unused with cuRobo IK.")
    parser.add_argument("--ik-pos-weight", type=float, default=1.0, help="Legacy option kept for CLI compatibility; unused with cuRobo IK.")
    parser.add_argument("--ik-rot-weight", type=float, default=0.35, help="Legacy option kept for CLI compatibility; unused with cuRobo IK.")
    parser.add_argument("--curobo-robot-cfg", type=str, default="franka.yml", help="Robot config passed to cuRobo.")
    parser.add_argument("--mpc-max-attempts", type=int, default=2, help="Max attempts used by each cuRobo MPC step.")
    parser.add_argument("--mj-steps-per-command", type=int, default=3, help="Number of MuJoCo steps executed after each MPC command.")
    parser.add_argument("--command-substeps", type=int, default=1, help="Multiplier used when matching cuRobo MPC dt to the MuJoCo control cadence.")
    parser.add_argument("--curobo-collision-activation-distance", type=float, default=0.06, help="Collision activation distance passed to cuRobo.")
    parser.add_argument("--disable-curobo-self-collision", action="store_true", help="Disable cuRobo self-collision checking.")
    parser.add_argument("--disable-curobo-cuda-graph", action="store_true", help="Disable cuRobo CUDA graph capture.")
    parser.add_argument("--pose-only-mpc", action="store_true", help="Disable MPC cspace goal cost and use pose tracking only.")
    parser.add_argument("--min-normal-force", type=float, default=None, help="Required normal force per fingertip before lifting. Defaults to a mass-based value.")
    parser.add_argument("--contact-stable-steps", type=int, default=15, help="Number of consecutive squeeze steps that must satisfy the normal-force threshold.")
    parser.add_argument("--lift-success-margin", type=float, default=0.005, help="Allowed height error when deciding whether the lift succeeded.")
    parser.add_argument("--visualize", action="store_true", help="Launch the MuJoCo passive viewer.")
    parser.add_argument("--real-time", action="store_true", help="Sleep to approximate real-time playback when visualizing.")
    parser.add_argument(
        "--scene-output",
        type=Path,
        default=GENERATED_SCENE_PATH,
        help="Path where the generated dual-Panda scene XML will be written.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    grasper = BimanualPandaGrasper(args)
    try:
        grasper.run()
    finally:
        grasper.close()


if __name__ == "__main__":
    main()
