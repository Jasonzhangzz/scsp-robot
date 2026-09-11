import argparse
import ast
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
from scipy.linalg import pinv
from scipy.spatial.transform import Rotation


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[3]
CUROBO_SRC_ROOT = REPO_ROOT.parent / "thirdparty" / "curobo" / "src"
for path in (REPO_ROOT, CUROBO_SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    import torch
    from curobo.geom.types import Cuboid, WorldConfig
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

    _HAS_CUROBO = True
    _CUROBO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime dependency
    torch = None
    Cuboid = None
    WorldConfig = None
    Pose = None
    JointState = None
    IKSolver = None
    IKSolverConfig = None
    _HAS_CUROBO = False
    _CUROBO_IMPORT_ERROR = exc

from planning.mlqp_point_v2 import LambdaContactControlOptimizer
# from planning.mpppi_explicit import MPPIExplicit
from planning.mpc_explicit2_bigrasp import MPCExplicit
from planning.screenshot import (
    PeriodicSVGScreenshotRecorder,
    build_free_camera_config_from_position,
)

PANDA_XML_PATH = REPO_ROOT / "envs" / "xmls" / "panda_nohand.xml"
GENERATED_SCENE_PATH = REPO_ROOT / "envs" / "xmls" / "_generated_bigrasp_scene.xml"
OBJECT_ASSET_DIR = REPO_ROOT / "envs" / "assets" / "objects"
DEFAULT_SCREENSHOT_DIR = REPO_ROOT / "outputs" / "bigrasp_show_screenshots"

PANDA_HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
TIP_RADIUS = 0.01
TIP_CENTER_OFFSET = 0.06
GHOST_OBJECT_HOVER_HEIGHT = 0.1
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
DEFAULT_SCALE_MAP = {
    "stanford_bunny2": np.array([1.5, 1.5, 1.5], dtype=np.float64),
    "rubber_duck": np.array([1.3, 1.4, 1.4], dtype=np.float64),
    "Wolf_Duck": np.array([0.002, 0.002, 0.002], dtype=np.float64),
}
DEFAULT_OBJECT_SCALE_BOOST = 1.2
ARM_OBSTACLE_SEGMENTS = (
    ("link6", "link5", "link6", 0.12),
    ("link7", "link6", "link7", 0.10),
    ("fingertip", "attachment", "tip_center", 0.07),
)
ARM_OBSTACLE_LENGTH_PADDING = 0.06
PANDA_TORQUE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)


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


def _slerp_rotation_matrix(start_rot, end_rot, alpha):
    start_rot = _project_to_rotation_matrix(start_rot)
    end_rot = _project_to_rotation_matrix(end_rot)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 1e-9:
        return start_rot
    if alpha >= 1.0 - 1e-9:
        return end_rot

    q0 = Rotation.from_matrix(start_rot).as_quat()
    q1 = Rotation.from_matrix(end_rot).as_quat()
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        q /= max(np.linalg.norm(q), 1e-9)
        return _project_to_rotation_matrix(Rotation.from_quat(q).as_matrix())

    theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta)) / max(sin_theta_0, 1e-9)
    s1 = sin_theta / max(sin_theta_0, 1e-9)
    q = s0 * q0 + s1 * q1
    q /= max(np.linalg.norm(q), 1e-9)
    return _project_to_rotation_matrix(Rotation.from_quat(q).as_matrix())


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


def _tilt_rotation_upward(rotation_matrix, tilt_angle):
    rotation_matrix = _project_to_rotation_matrix(rotation_matrix)
    tilt_angle = abs(float(tilt_angle))
    if tilt_angle <= 1e-9:
        return rotation_matrix

    local_pitch_plus = Rotation.from_rotvec(np.array([tilt_angle, 0.0, 0.0], dtype=np.float64)).as_matrix()
    local_pitch_minus = Rotation.from_rotvec(np.array([-tilt_angle, 0.0, 0.0], dtype=np.float64)).as_matrix()
    candidate_plus = _project_to_rotation_matrix(rotation_matrix @ local_pitch_plus)
    candidate_minus = _project_to_rotation_matrix(rotation_matrix @ local_pitch_minus)

    if float(np.dot(candidate_plus[:, 2], WORLD_UP)) >= float(np.dot(candidate_minus[:, 2], WORLD_UP)):
        return candidate_plus
    return candidate_minus


def _contact_force_edge_directions_world_from_frame(contact_frame, mu):
    contact_frame = np.asarray(contact_frame, dtype=np.float64).reshape(3, 3)
    normal = _normalize(contact_frame[:, 0])
    tangent_1 = _normalize(contact_frame[:, 1])
    tangent_2 = _normalize(contact_frame[:, 2])
    mu = float(mu)
    return np.column_stack(
        [
            normal + mu * tangent_1,
            normal + mu * tangent_2,
            normal - mu * tangent_1,
            normal - mu * tangent_2,
        ]
    ).astype(np.float64)


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


def _parse_bool_arg(value):
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


def _parse_camera_free_arg(value):
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
    object_friction,
    object_pos,
    object_quat,
    scene_center_x,
    robot_span,
    pedestal_size=(0.05, 0.07, 0.06),
    pedestal_pos=(0.58, 0.0, 0.06),
    mujoco_timestep=0.01,
    scene_output_path=GENERATED_SCENE_PATH,
    show=False,
):
    panda_root = ET.parse(PANDA_XML_PATH).getroot()

    root = ET.Element("mujoco", {"model": "dual panda bigrasp"})
    root.append(ET.Element("compiler", {"angle": "radian", "meshdir": "assets", "autolimits": "true"}))
    root.append(
        ET.Element(
            "option",
            {
                "integrator": "implicitfast",
                "impratio": "10",
                "timestep": f"{float(mujoco_timestep):.8f}",
            },
        )
    )
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
    ET.SubElement(asset, "material", {"name": "ghost_obj_mat", "rgba": "0.88 0.52 0.22 1"})
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

    ET.SubElement(worldbody, "body", {"name": "goal", "pos": format_vec(object_pos), "quat": format_vec(object_quat)})

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
            "name": "obj",
            "type": "mesh",
            "mesh": "object_mesh",
            "material": "obj_mat",
            "mass": f"{float(object_mass):.8f}",
            "contype": "0" if bool(show) else "1",
            "conaffinity": "0" if bool(show) else "1",
            "condim": "3",
            "friction": f"{float(object_friction):.8f} 0.08 0.01",
        },
    )

    marker_specs = [
        ("obj_point", "1 1 0 1"),
        ("contact_point1", "0 1 0 1"),
        ("contact_point2", "0 0 1 1"),
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

    robot_body = panda_root.find("./worldbody/body[@name='link0']")
    if robot_body is None:
        raise RuntimeError(f"Failed to find Panda root body in {PANDA_XML_PATH}")
    robot_contact = panda_root.find("contact")
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
    for prefix in ("left_", "right_"):
        for joint_idx, torque_limit in enumerate(PANDA_TORQUE_LIMITS, start=1):
            ET.SubElement(
                actuator,
                "motor",
                {
                    "name": f"{prefix}actuator{joint_idx}",
                    "joint": f"{prefix}joint{joint_idx}",
                    "ctrllimited": "true",
                    "ctrlrange": f"{-float(torque_limit):.8f} {float(torque_limit):.8f}",
                },
            )

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
    ik_solver: object = None
    curobo_joint_names: tuple = ()
    retract_cfg: np.ndarray = None
    static_world_with_pedestal: object = None
    static_world_floor_only: object = None
    static_world: object = None
    current_world: object = None
    current_world_mode: str = "with_pedestal"
    torque_limits: np.ndarray = None
    home_q: np.ndarray = None
    cartesian_stiffness: np.ndarray = None
    cartesian_damping: np.ndarray = None
    nullspace_stiffness: float = 10.0
    position_d: np.ndarray = None
    orientation_d: np.ndarray = None
    p_d: np.ndarray = None
    R_d: np.ndarray = None


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


class DualArmPlanOnceParams:
    def __init__(self, args, obj_mass):
        self.contact_cost_param = float(args.planner_contact_cost_param)
        self.attract_coef = float(args.planner_attract_coef)
        self.reject_coef = float(args.planner_reject_coef)
        self.contact_coef = float(args.planner_contact_coef)
        self.reject_dis = float(args.planner_reject_distance)
        self.planner_force_tracking_weight_ = float(args.planner_force_tracking_weight)
        self.planner_torque_tracking_weight_ = float(args.planner_torque_tracking_weight)

        self.h_ = float(args.planner_dt)
        self.n_robot_qpos_ = 6
        self.n_qpos_ = 13
        self.n_qvel_ = 12
        self.n_cmd_ = 6
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = int(args.planner_max_contacts)

        self.obj_inertia_ = np.identity(6, dtype=np.float32)
        self.obj_inertia_[0:3, 0:3] = float(args.planner_object_inertia_pos) * np.eye(3, dtype=np.float32)
        self.obj_inertia_[3:, 3:] = float(args.planner_object_inertia_rot) * np.eye(3, dtype=np.float32)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [float(args.planner_robot_stiffness)]).astype(np.float32)

        self.Q = np.zeros((self.n_qvel_, self.n_qvel_), dtype=np.float32)
        self.Q[:6, :6] = self.obj_inertia_
        self.Q[6:, 6:] = self.robot_stiff_

        self.obj_mass_ = float(obj_mass)
        self.gravity_ = np.array([0.0, 0.0, -9.8, 0.0, 0.0, 0.0], dtype=np.float32)
        self.model_params = float(args.contact_stiffness)

        self.mpc_horizon_ = int(args.planner_horizon)
        self.mpc_model = "explicit"
        self.planner_solver_ = str(args.planner_solver).strip().lower()
        self.mpc_u_lb_ = -float(args.planner_cmd_limit)
        self.mpc_u_ub_ = float(args.planner_cmd_limit)

        self.sol_guess_ = None
        self.mppi_samples_ = int(args.mppi_samples)
        self.mppi_iterations_ = int(args.mppi_iterations)
        self.mppi_init_iterations_ = int(args.mppi_init_iterations)
        self.mppi_lambda_ = float(args.mppi_lambda)
        self.mppi_noise_sigma_ = float(args.mppi_noise_sigma)
        self.mppi_noise_decay_ = float(args.mppi_noise_decay)
        self.mppi_elite_frac_ = float(args.mppi_elite_frac)
        self.mppi_use_torch_compile_ = bool(args.mppi_use_torch_compile)
        default_mppi_device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
        self.mppi_device_ = str(args.mppi_device or default_mppi_device)


class BimanualPandaGrasper:
    def __init__(self, args):
        if not _HAS_CUROBO:
            raise ImportError(
                "Failed to import cuRobo. Make sure cuRobo is installed or "
                f"{CUROBO_SRC_ROOT} is available on PYTHONPATH. "
                f"Original error: {_CUROBO_IMPORT_ERROR!r}"
            )

        self.args = args
        logging.getLogger("curobo").setLevel(logging.WARNING if bool(args.verbose) else logging.ERROR)
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
                self.pedestal_pos[0],
                self.pedestal_pos[1],
                support_top - self.mesh_bounds[0, 2] + args.object_z_offset + args.obj_init_height,
            ],
            dtype=np.float64,
        )
        object_quat = quat_from_yaw(args.object_yaw)
        self.object_pos = np.asarray(object_pos, dtype=np.float64).copy()
        self.object_quat = np.asarray(object_quat, dtype=np.float64).copy()
        self.object_is_ghost = False

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
            object_friction=args.object_friction,
            object_pos=object_pos,
            object_quat=object_quat,
            scene_center_x=args.scene_center_x,
            robot_span=args.robot_span,
            pedestal_size=self.pedestal_size,
            pedestal_pos=self.pedestal_pos,
            mujoco_timestep=args.mujoco_dt,
            scene_output_path=args.scene_output,
            show=args.show,
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = float(args.mujoco_dt)
        if bool(args.show):
            self.model.opt.gravity[:] = 0.0
        self.data = mujoco.MjData(self.model)

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.screenshot_recorder = self._build_screenshot_recorder()

        self.left_arm = self._build_arm_handles("left_", self.left_base_pos, self.left_base_rot)
        self.right_arm = self._build_arm_handles("right_", self.right_base_pos, self.right_base_rot)
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        self.object_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
        self.object_qpos_adr = int(self.model.jnt_qposadr[self.object_joint_id]) if self.object_joint_id >= 0 else -1
        self.object_dof_adr = int(self.model.jnt_dofadr[self.object_joint_id]) if self.object_joint_id >= 0 else -1
        self.obj_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obj")
        self.marker_body_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (
                "obj_point",
                "contact_point1",
                "contact_point2",
                "goal",
            )
        }
        self.support_height_threshold = self.pedestal_pos[2] + self.pedestal_size[2] + args.ground_height_margin

        self.reset(self.object_pos, self.object_quat)
        optimizer_support_kwargs = {}
        if bool(args.optimizer_use_support_filter):
            optimizer_support_kwargs = {
                "support_surface_point": self.support_surface_point,
                "support_surface_normal": self.support_surface_normal,
                "support_surface_clearance": args.ground_height_margin,
                "support_surface_normal_alignment_threshold": args.support_normal_alignment_threshold,
            }
        self.optimizer = LambdaContactControlOptimizer(
            mesh_path=str(self.mesh_path),
            obj_mass=args.obj_mass,
            arm_friction=args.optimizer_arm_friction,
            contact_stiffness=args.contact_stiffness,
            time_step=self.model.opt.timestep,
            sample_num=args.sample_num,
            pos_coef=args.pos_coef,
            ori_coef=args.ori_coef,
            scale_factors=tuple(self.mesh_scale.tolist()),
            curvature_neighbor_k=args.optimizer_curvature_neighbor_k,
            region_max_mean_curvature=args.optimizer_max_region_mean_curvature,
            region_max_point_curvature=args.optimizer_max_point_curvature,
            curvature_penalty_weight=args.optimizer_curvature_penalty_weight,
            nlp_solver=args.solver,
            static_nlp_solver=args.solver,
            **optimizer_support_kwargs,
        )
        self.optimizer.set_timing_print_enabled(bool(getattr(args, "print_contact_timing", True)))
        self.optimizer_contact_search_cache = None
        self.command_dt = (
            float(self.model.opt.timestep)
            * max(int(self.args.mj_steps_per_command), 1)
            * max(int(self.args.command_substeps), 1)
        )
        self.plan_params = DualArmPlanOnceParams(self.args, args.obj_mass)
        self.planner = MPCExplicit(self.plan_params)
        self._setup_curobo()

    def _resolve_mesh_scale(self, args):
        if args.scale is not None:
            return np.asarray(args.scale, dtype=np.float64)
        key = self.mesh_path.stem
        base_scale = DEFAULT_SCALE_MAP.get(key, None)
        if base_scale is None:
            return np.ones(3, dtype=np.float64)
        return DEFAULT_OBJECT_SCALE_BOOST * np.asarray(base_scale, dtype=np.float64).copy()

    def _build_focus_camera_config(self, lookat=None):
        if self.args.camera_free is not None:
            camera_position, fixed_lookat = self.args.camera_free
            return build_free_camera_config_from_position(
                camera_position=np.asarray(camera_position, dtype=np.float64).reshape(3),
                lookat=np.asarray(fixed_lookat, dtype=np.float64).reshape(3),
            )
        lookat = self.object_pos.copy() if lookat is None else np.asarray(lookat, dtype=np.float64).reshape(3)
        lookat = lookat + np.array([0.0, 0.0, 0.015], dtype=np.float64)
        camera_position = lookat + np.array(
            [
                -0.5,
                -max(0.34, 0.62 * float(self.args.robot_span)),
                max(0.18, 1.2 * float(self.pedestal_size[2]) + 0.14),
            ],
            dtype=np.float64,
        )
        return build_free_camera_config_from_position(
            camera_position=camera_position,
            lookat=lookat,
        )

    def _apply_camera_config_to_viewer(self, camera_config):
        if self.viewer is None:
            return
        self.viewer.cam.lookat[:] = np.asarray(camera_config.lookat, dtype=np.float64)
        self.viewer.cam.distance = float(camera_config.distance)
        self.viewer.cam.azimuth = float(camera_config.azimuth_deg)
        self.viewer.cam.elevation = float(camera_config.elevation_deg)

    def _build_screenshot_recorder(self):
        self.scene_camera_config = self._build_focus_camera_config()
        self._apply_camera_config_to_viewer(self.scene_camera_config)
        if float(self.args.screenshot_interval) <= 0.0:
            return None

        return PeriodicSVGScreenshotRecorder(
            self.model,
            output_dir=self.args.screenshot_dir,
            camera_config=self.scene_camera_config,
            interval_seconds=float(self.args.screenshot_interval),
            width=int(self.args.screenshot_width),
            height=int(self.args.screenshot_height),
            filename_prefix="bigrasp_show",
            capture_on_start=True,
        )

    def _sync_visualization(self):
        if self.viewer is None and self.screenshot_recorder is None:
            return
        self.scene_camera_config = self._build_focus_camera_config()
        self._apply_camera_config_to_viewer(self.scene_camera_config)
        if self.screenshot_recorder is not None:
            self.screenshot_recorder.camera.lookat[:] = np.asarray(
                self.scene_camera_config.lookat,
                dtype=np.float64,
            )
            self.screenshot_recorder.camera.distance = float(self.scene_camera_config.distance)
            self.screenshot_recorder.camera.azimuth = float(self.scene_camera_config.azimuth_deg)
            self.screenshot_recorder.camera.elevation = float(self.scene_camera_config.elevation_deg)
        if self.viewer is not None:
            self.viewer.sync()
        if self.screenshot_recorder is not None:
            self.screenshot_recorder.capture_if_due(self.data)

    def set_object_pose(self, object_pos, object_quat):
        self.object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3).copy()
        self.object_quat = np.asarray(object_quat, dtype=np.float64).reshape(4).copy()
        self.object_quat /= max(np.linalg.norm(self.object_quat), 1e-9)
        if self.object_qpos_adr >= 0:
            self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 3] = self.object_pos
            self.data.qpos[self.object_qpos_adr + 3 : self.object_qpos_adr + 7] = self.object_quat
            if self.object_dof_adr >= 0:
                self.data.qvel[self.object_dof_adr : self.object_dof_adr + 6] = 0.0
            if self.object_body_id >= 0:
                self.data.xfrc_applied[self.object_body_id, :] = 0.0
        elif self.object_body_id >= 0:
            self.model.body_pos[self.object_body_id] = self.object_pos
            self.model.body_quat[self.object_body_id] = self.object_quat

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
            ghost_body_id=-1,
            base_pos=np.asarray(base_pos, dtype=np.float64).copy(),
            base_rot=np.asarray(base_rot, dtype=np.float64).reshape(3, 3).copy(),
            body_ids_by_name=body_ids_by_name,
            torque_limits=PANDA_TORQUE_LIMITS.copy(),
            home_q=PANDA_HOME_Q.copy(),
            cartesian_stiffness=2.0
            * np.diag(
                [
                    float(self.args.cartesian_stiffness_pos),
                    float(self.args.cartesian_stiffness_pos),
                    float(self.args.cartesian_stiffness_pos),
                    float(self.args.cartesian_stiffness_rot),
                    float(self.args.cartesian_stiffness_rot),
                    float(self.args.cartesian_stiffness_rot),
                ]
            ).astype(np.float64),
            nullspace_stiffness=float(self.args.nullspace_stiffness),
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

    def _visible_optimizer_point_indices(self, object_pos, object_rot):
        if not bool(self.args.optimizer_use_support_filter):
            return self.optimizer.point_idx.copy()
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        centers_world = (object_rot @ self.optimizer.sample_point.T).T + object_pos[None, :]
        visible_idx = np.where(centers_world[:, 2] > float(self.support_height_threshold))[0]
        if visible_idx.size == 0:
            visible_idx = self.optimizer.point_idx.copy()
        visible_idx = self.optimizer.get_contact_candidate_indices(
            visible_face_idx=visible_idx,
            object_pos=object_pos,
            object_rot=object_rot,
        )
        if visible_idx.size == 0:
            visible_idx = self.optimizer.get_contact_candidate_indices(
                visible_face_idx=self.optimizer.point_idx,
                object_pos=object_pos,
                object_rot=object_rot,
            )
        return np.asarray(visible_idx, dtype=int)

    def _update_inter_arm_worlds(self):
        for target_arm, obstacle_arm in (
            (self.left_arm, self.right_arm),
            (self.right_arm, self.left_arm),
        ):
            world = target_arm.static_world.clone()
            for obstacle in self._build_other_arm_obstacle_cuboids(target_arm, obstacle_arm):
                world.add_obstacle(obstacle)
            target_arm.current_world = world
            target_arm.ik_solver.update_world(world)

    def _setup_curobo_arm(self, arm):
        world_config = self._build_curobo_world_config_dict(arm, include_pedestal=True)
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
        arm.curobo_joint_names = tuple(arm.ik_solver.joint_names)
        get_retract_config = getattr(arm.ik_solver, "get_retract_config", None)
        if callable(get_retract_config):
            retract_cfg = get_retract_config()
        else:
            retract_cfg = arm.ik_solver.rollout_fn.dynamics_model.retract_config
        arm.retract_cfg = _tensor_to_numpy(retract_cfg).reshape(-1).astype(np.float64)
        arm.cartesian_damping = 2.0 * np.sqrt(arm.cartesian_stiffness)
        arm.static_world_with_pedestal = self._build_curobo_world_config(arm, include_pedestal=True)
        arm.static_world_floor_only = self._build_curobo_world_config(arm, include_pedestal=False)
        arm.static_world = arm.static_world_with_pedestal
        arm.current_world = arm.static_world.clone()
        arm.current_world_mode = "with_pedestal"
        current_tip_pos, current_tip_rot = self.get_tip_pose(arm)
        arm.position_d = current_tip_pos.copy()
        arm.orientation_d = current_tip_rot.copy()
        arm.p_d = current_tip_pos.copy()
        arm.R_d = current_tip_rot.copy()
    def _setup_curobo(self):
        self._setup_curobo_arm(self.left_arm)
        self._setup_curobo_arm(self.right_arm)
        self._update_inter_arm_worlds()

    def reset(self, object_pos, object_quat):
        self.data.qpos[self.left_arm.qpos_adr] = PANDA_HOME_Q
        self.data.qpos[self.right_arm.qpos_adr] = PANDA_HOME_Q
        self.data.ctrl[self.left_arm.actuator_ids] = 0.0
        self.data.ctrl[self.right_arm.actuator_ids] = 0.0
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.set_object_pose(object_pos, object_quat)
        mujoco.mj_forward(self.model, self.data)
        for arm in (self.left_arm, self.right_arm):
            tip_pos, tip_rot = self.get_tip_pose(arm)
            arm.position_d = tip_pos.copy()
            arm.orientation_d = tip_rot.copy()
            arm.p_d = tip_pos.copy()
            arm.R_d = tip_rot.copy()
        mujoco.mj_forward(self.model, self.data)
        self._sync_visualization()

    def close(self):
        if self.screenshot_recorder is not None:
            self.screenshot_recorder.close()
            self.screenshot_recorder = None
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
        if self.object_qpos_adr >= 0:
            pos = np.asarray(self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 3], dtype=np.float64).copy()
            quat = np.asarray(
                self.data.qpos[self.object_qpos_adr + 3 : self.object_qpos_adr + 7],
                dtype=np.float64,
            ).copy()
            quat /= max(np.linalg.norm(quat), 1e-9)
            self.object_pos = pos.copy()
            self.object_quat = quat.copy()
        else:
            pos = self.object_pos.copy()
            quat = self.object_quat.copy()
        return pos, quat, quat_wxyz_to_mat(quat)

    def _get_tip_geom_pose(self, arm, data=None):
        data = self.data if data is None else data
        geom = data.geom(arm.tip_geom_id)
        pos = np.asarray(geom.xpos, dtype=np.float64).copy()
        rot = np.asarray(geom.xmat, dtype=np.float64).reshape(3, 3).copy()
        return pos, rot

    def get_tip_pos(self, arm):
        pos, _ = self._get_tip_geom_pose(arm)
        return pos

    def get_tip_pose(self, arm, data=None):
        return self._get_tip_geom_pose(arm, data=data)

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

    def _ordered_contact_data(
        self,
        contact_points_local,
        normals_local,
        object_pos,
        object_rot,
        reference_points_world=None,
        return_order=False,
    ):
        contact_points_local = np.asarray(contact_points_local, dtype=np.float64).reshape(-1, 3)
        normals_local = np.asarray(normals_local, dtype=np.float64).reshape(-1, 3)
        contact_points_world = (object_rot @ contact_points_local.T).T + object_pos[None, :]
        if reference_points_world is None:
            left_ref = self.get_tip_pos(self.left_arm)
            right_ref = self.get_tip_pos(self.right_arm)
        else:
            reference_points_world = np.asarray(reference_points_world, dtype=np.float64).reshape(-1, 3)
            if reference_points_world.shape[0] != 2:
                raise ValueError(
                    f"Expected 2 reference points for contact ordering, got {reference_points_world.shape[0]}."
                )
            left_ref = reference_points_world[0]
            right_ref = reference_points_world[1]

        keep_cost = np.linalg.norm(left_ref - contact_points_world[0]) + np.linalg.norm(right_ref - contact_points_world[1])
        swap_cost = np.linalg.norm(left_ref - contact_points_world[1]) + np.linalg.norm(right_ref - contact_points_world[0])
        order = np.array([0, 1], dtype=int) if keep_cost <= swap_cost else np.array([1, 0], dtype=int)
        ordered = (
            contact_points_local[order],
            normals_local[order],
            contact_points_world[order],
        )
        if return_order:
            return (*ordered, order)
        return ordered

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

    def _stage_targets_from_object_pose(
        self,
        contact_points_local,
        normals_local,
        object_pos,
        object_rot,
        center_offset,
    ):
        contact_points_world = (object_rot @ np.asarray(contact_points_local, dtype=np.float64).T).T + object_pos[None, :]
        inward_normals_world = (object_rot @ np.asarray(normals_local, dtype=np.float64).T).T
        outward_normals_world = -inward_normals_world
        (left_pos, left_rot), (right_pos, right_rot) = self._compute_fingertip_targets(
            contact_points_world,
            inward_normals_world,
            center_offset,
        )
        return {
            "left_tip_pos": np.asarray(left_pos, dtype=np.float64),
            "left_tip_rot": _project_to_rotation_matrix(left_rot),
            "right_tip_pos": np.asarray(right_pos, dtype=np.float64),
            "right_tip_rot": _project_to_rotation_matrix(right_rot),
            "left_outward_normal": _normalize(outward_normals_world[0]),
            "right_outward_normal": _normalize(outward_normals_world[1]),
            "object_target_pos": np.asarray(object_pos, dtype=np.float64).copy(),
            "object_target_quat": mat_to_quat_wxyz(object_rot),
        }

    @staticmethod
    def _offset_points_along_normals(points_world, outward_normals_world, offset):
        points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        outward_normals_world = np.asarray(outward_normals_world, dtype=np.float64).reshape(-1, 3)
        return points_world + float(offset) * np.vstack([_normalize(normal) for normal in outward_normals_world])

    @staticmethod
    def _contact_dissipation_factor(distance_rate, dissipation_velocity):
        dissipation_velocity = max(float(dissipation_velocity), 1e-9)
        s = float(distance_rate) / dissipation_velocity
        if s < 0.0:
            return 1.0 - s
        if s < 2.0:
            return 0.25 * (s - 2.0) * (s - 2.0)
        return 0.0

    @staticmethod
    def _compliant_normal_force(distance, stiffness, smoothing_factor):
        distance = float(distance)
        stiffness = max(float(stiffness), 1e-9)
        smoothing_factor = max(float(smoothing_factor), 0.0)
        if smoothing_factor <= 1e-9:
            return max(-stiffness * distance, 0.0)

        exponent = -distance / smoothing_factor
        if exponent >= 37.0:
            return max(-stiffness * distance, 0.0)
        return float(smoothing_factor * stiffness * np.log1p(np.exp(exponent)))

    @staticmethod
    def _distance_from_normal_force(target_force, stiffness, smoothing_factor):
        target_force = max(float(target_force), 0.0)
        stiffness = max(float(stiffness), 1e-9)
        smoothing_factor = max(float(smoothing_factor), 0.0)
        if target_force <= 1e-9:
            return 0.0
        if smoothing_factor <= 1e-9:
            return -target_force / stiffness

        scaled_force = target_force / (smoothing_factor * stiffness)
        if scaled_force >= 37.0:
            return -target_force / stiffness
        return float(-smoothing_factor * np.log(np.expm1(scaled_force)))

    def _build_force_control_targets(
        self,
        contact_points_world,
        outward_normals_world,
        target_normal_forces,
        previous_state=None,
        contact_stiffness=None,
        dissipation_velocity=None,
        stiction_velocity=None,
        smoothing_factor=None,
    ):
        contact_points_world = np.asarray(contact_points_world, dtype=np.float64).reshape(2, 3)
        outward_normals_world = np.asarray(outward_normals_world, dtype=np.float64).reshape(2, 3)
        target_normal_forces = np.asarray(target_normal_forces, dtype=np.float64).reshape(2)

        default_force_control_stiffness = getattr(self.args, "force_control_stiffness", None)
        contact_stiffness = (
            float(default_force_control_stiffness)
            if contact_stiffness is None and default_force_control_stiffness is not None
            else contact_stiffness
        )
        contact_stiffness = max(float(1.0 if contact_stiffness is None else contact_stiffness), 1e-9)
        dissipation_velocity = float(
            getattr(self.args, "force_control_dissipation_velocity", 0.1)
            if dissipation_velocity is None
            else dissipation_velocity
        )
        stiction_velocity = float(
            getattr(self.args, "force_control_stiction_velocity", 0.05)
            if stiction_velocity is None
            else stiction_velocity
        )
        smoothing_factor = float(
            getattr(self.args, "force_control_smoothing", 0.0)
            if smoothing_factor is None
            else smoothing_factor
        )

        current_tip_positions = np.vstack(
            [
                self.get_tip_pos(self.left_arm),
                self.get_tip_pos(self.right_arm),
            ]
        )
        if previous_state is None:
            previous_tip_positions = current_tip_positions.copy()
            previous_contact_points_world = contact_points_world.copy()
        else:
            previous_tip_positions = np.asarray(
                previous_state.get("tip_positions_world", current_tip_positions),
                dtype=np.float64,
            ).reshape(2, 3)
            previous_contact_points_world = np.asarray(
                previous_state.get("contact_points_world", contact_points_world),
                dtype=np.float64,
            ).reshape(2, 3)

        dt = max(float(self.command_dt), float(self.model.opt.timestep), 1e-6)
        goal_points_world = np.zeros((2, 3), dtype=np.float64)
        desired_force_world = np.zeros((2, 3), dtype=np.float64)
        modeled_force_world = np.zeros((2, 3), dtype=np.float64)
        modeled_normal_forces = np.zeros(2, dtype=np.float64)
        modeled_tangential_forces = np.zeros(2, dtype=np.float64)
        modeled_distance = np.zeros(2, dtype=np.float64)
        modeled_compression = np.zeros(2, dtype=np.float64)
        modeled_distance_rate = np.zeros(2, dtype=np.float64)
        desired_offsets = np.zeros(2, dtype=np.float64)

        for idx in range(2):
            outward_normal = _normalize(outward_normals_world[idx])
            if np.linalg.norm(outward_normal) < 1e-9:
                outward_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

            current_tip_pos = current_tip_positions[idx]
            previous_tip_pos = previous_tip_positions[idx]
            contact_point_world = contact_points_world[idx]
            previous_contact_point_world = previous_contact_points_world[idx]

            tip_velocity = (current_tip_pos - previous_tip_pos) / dt
            contact_point_velocity = (contact_point_world - previous_contact_point_world) / dt
            relative_velocity = tip_velocity - contact_point_velocity

            distance = float(np.dot(current_tip_pos - contact_point_world, outward_normal) - TIP_RADIUS)
            distance_rate = float(np.dot(relative_velocity, outward_normal))
            dissipation_factor = self._contact_dissipation_factor(distance_rate, dissipation_velocity)
            compliant_force = self._compliant_normal_force(distance, contact_stiffness, smoothing_factor)
            normal_force = float(compliant_force * dissipation_factor)

            tangential_velocity = relative_velocity - distance_rate * outward_normal
            tangential_speed = float(np.linalg.norm(tangential_velocity))
            regularized_speed = np.sqrt(max(stiction_velocity, 0.0) ** 2 + tangential_speed**2)
            if regularized_speed > 1e-12:
                tangential_force_world = (
                    tangential_velocity / regularized_speed
                ) * float(self.args.arm_friction) * normal_force
            else:
                tangential_force_world = np.zeros(3, dtype=np.float64)

            target_distance = self._distance_from_normal_force(
                target_normal_forces[idx],
                contact_stiffness,
                smoothing_factor,
            )
            target_distance = min(float(target_distance), 0.0)
            desired_offset = float(np.clip(TIP_RADIUS + target_distance, 0.001, TIP_RADIUS))

            desired_force_world[idx] = -outward_normal * float(target_normal_forces[idx])
            modeled_force_world[idx] = -outward_normal * normal_force + tangential_force_world
            modeled_normal_forces[idx] = normal_force
            modeled_tangential_forces[idx] = float(np.linalg.norm(tangential_force_world))
            modeled_distance[idx] = distance
            modeled_compression[idx] = max(-distance, 0.0)
            modeled_distance_rate[idx] = distance_rate
            desired_offsets[idx] = desired_offset
            goal_points_world[idx] = contact_point_world + outward_normal * desired_offset

        next_state = {
            "tip_positions_world": current_tip_positions.copy(),
            "contact_points_world": contact_points_world.copy(),
        }
        debug = {
            "force_target_points_world": goal_points_world.copy(),
            "desired_contact_force_world": desired_force_world.copy(),
            "modeled_contact_force_world": modeled_force_world.copy(),
            "desired_normal_forces": target_normal_forces.copy(),
            "modeled_normal_forces": modeled_normal_forces.copy(),
            "modeled_tangential_forces": modeled_tangential_forces.copy(),
            "modeled_contact_distance": modeled_distance.copy(),
            "modeled_contact_compression": modeled_compression.copy(),
            "modeled_contact_distance_rate": modeled_distance_rate.copy(),
            "force_target_offsets": desired_offsets.copy(),
            "force_control_stiffness": float(contact_stiffness),
            "force_control_dissipation_velocity": float(dissipation_velocity),
            "force_control_stiction_velocity": float(stiction_velocity),
            "force_control_smoothing": float(smoothing_factor),
        }
        return goal_points_world, debug, next_state

    def _copy_contact_targets(self, targets):
        copied = {}
        for key, value in dict(targets).items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied

    def _attach_contact_wrench_targets(
        self,
        targets,
        prefix,
        contact_points_local,
        force_vectors_local,
        object_rot,
    ):
        contact_points_local = np.asarray(contact_points_local, dtype=np.float64).reshape(-1, 3)
        force_vectors_local = np.asarray(force_vectors_local, dtype=np.float64).reshape(-1, 3)
        object_rot = _project_to_rotation_matrix(object_rot)
        if contact_points_local.shape[0] == 0 or force_vectors_local.shape[0] != contact_points_local.shape[0]:
            return

        contact_torques_local = np.cross(contact_points_local, force_vectors_local)
        contact_force_world = (object_rot @ force_vectors_local.T).T
        contact_torque_world = (object_rot @ contact_torques_local.T).T
        targets[f"{prefix}_contact_force_world"] = contact_force_world.copy()
        targets[f"{prefix}_contact_torque_world"] = contact_torque_world.copy()
        targets[f"{prefix}_total_force_local"] = np.sum(force_vectors_local, axis=0)
        targets[f"{prefix}_total_torque_local"] = np.sum(contact_torques_local, axis=0)
        targets[f"{prefix}_total_force_world"] = np.sum(contact_force_world, axis=0)
        targets[f"{prefix}_total_torque_world"] = np.sum(contact_torque_world, axis=0)

    def _resolve_live_wrench_targets(self, live_targets):
        for prefix in ("desired", "witness"):
            force_key = f"{prefix}_total_force_world"
            torque_key = f"{prefix}_total_torque_world"
            if force_key not in live_targets and torque_key not in live_targets:
                continue
            desired_force_world = np.asarray(
                live_targets.get(force_key, np.zeros(3, dtype=np.float64)),
                dtype=np.float64,
            ).reshape(3)
            desired_torque_world = np.asarray(
                live_targets.get(torque_key, np.zeros(3, dtype=np.float64)),
                dtype=np.float64,
            ).reshape(3)
            execute_desired_wrench = bool(
                np.linalg.norm(desired_force_world) > 1e-9 or np.linalg.norm(desired_torque_world) > 1e-9
            )
            if (not execute_desired_wrench) and prefix != "witness":
                continue
            return desired_force_world, desired_torque_world, execute_desired_wrench, prefix

        return (
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            False,
            "",
        )

    def _resolve_arm_verify_cost(self, prev_verify_active, tip_pos, contact_point, contact_items):
        tip_pos = np.asarray(tip_pos, dtype=np.float64).reshape(3)
        contact_point = np.asarray(contact_point, dtype=np.float64).reshape(3)
        enter_tol = max(float(self.args.planner_attract_tol), 1e-6)
        exit_tol = max(1.5 * enter_tol, enter_tol + 1e-6)
        tip_distance = float(np.linalg.norm(tip_pos - contact_point))
        has_contact = self._has_active_contact(contact_items)

        verify_active = bool(has_contact or tip_distance <= enter_tol)
        if not verify_active and bool(prev_verify_active) and tip_distance <= exit_tol:
            verify_active = True

        return float(verify_active), verify_active, tip_distance, has_contact

    def _get_fixed_optimizer_regions(self, object_pos, object_rot):
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        visible_idx = self._visible_optimizer_point_indices(object_pos, object_rot)
        return self.optimizer.get_best_regions(
            visible_face_idx=visible_idx,
            top_k=self.optimizer.top_region_pairs,
            object_pos=object_pos,
            object_rot=object_rot,
        )

    def _build_precomputed_optimizer_contact_cache(self, object_pos, object_rot):
        if not bool(getattr(self.args, "precompute_contact_search", True)):
            self.optimizer_contact_search_cache = None
            return None

        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        visible_idx = self._visible_optimizer_point_indices(object_pos, object_rot)
        cache = self.optimizer.precompute_contact_search_cache(
            visible_face_idx=visible_idx,
            object_pos=object_pos,
            object_rot=object_rot,
        )
        self.optimizer_contact_search_cache = cache
        return cache

    def _get_live_contact_targets(
        self,
        object_pos,
        object_rot,
        previous_targets=None,
        virtual_offset=None,
        fixed_region_groups=None,
        candidate_cache=None,
    ):
        live_t0 = time.perf_counter()
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        virtual_offset = float(self.args.planner_attract_offset if virtual_offset is None else virtual_offset)
        if candidate_cache is None:
            candidate_cache = self.optimizer_contact_search_cache

        visible_idx = None
        if candidate_cache is None:
            visible_idx = self._visible_optimizer_point_indices(object_pos, object_rot)
        contact_points_local, normals_local, total_cost, region_score, antipodal_margin = self.optimizer.choose_contact_set(
            visible_face_idx=visible_idx,
            object_pos=object_pos,
            object_rot=object_rot,
            fixed_region_groups=fixed_region_groups,
            candidate_cache=candidate_cache,
        )
        live_elapsed = time.perf_counter() - live_t0
        optimizer_search_timing = copy.deepcopy(getattr(self.optimizer, "last_contact_search_timing", {}))

        contact_points_local = np.asarray(contact_points_local, dtype=np.float64).reshape(-1, 3)
        normals_local = np.asarray(normals_local, dtype=np.float64).reshape(-1, 3)
        if contact_points_local.shape[0] != 2 or normals_local.shape[0] != 2:
            if previous_targets is None:
                raise RuntimeError(f"Expected 2 optimizer contacts, got {contact_points_local.shape[0]}")
            contact_points_local = np.asarray(previous_targets["contact_points_local"], dtype=np.float64).reshape(2, 3)
            normals_local = np.asarray(previous_targets["normals_local"], dtype=np.float64).reshape(2, 3)
        raw_contact_points_local = contact_points_local.copy()
        raw_normals_local = normals_local.copy()

        reference_points_world = None
        if previous_targets is not None:
            prev_world = np.asarray(previous_targets.get("contact_points_world", []), dtype=np.float64).reshape(-1, 3)
            if prev_world.shape[0] == 2:
                reference_points_world = prev_world

        contact_points_local, normals_local, contact_points_world, order = self._ordered_contact_data(
            contact_points_local,
            normals_local,
            object_pos,
            object_rot,
            reference_points_world=reference_points_world,
            return_order=True,
        )
        inward_normals_world = (object_rot @ normals_local.T).T
        outward_normals_world = -inward_normals_world
        virtual_points_world = self._offset_points_along_normals(
            contact_points_world,
            outward_normals_world,
            virtual_offset,
        )
        grasp_result = self.optimizer.last_grasp_result
        contact_indices = np.zeros((0,), dtype=int)
        raw_contact_indices = np.zeros((0,), dtype=int)
        witness_contact_forces_local = np.zeros((0, 3), dtype=np.float64)
        witness_force_vectors_local = np.zeros((0, 3), dtype=np.float64)
        grasp_solve_time = float("nan")
        grasp_wall_time = float("nan")
        if grasp_result is not None:
            grasp_solve_time = float(grasp_result.get("solve_time", float("nan")))
            grasp_wall_time = float(grasp_result.get("wall_time", float("nan")))
            contact_indices = np.asarray(grasp_result.get("contact_indices", []), dtype=int).reshape(-1)
            raw_contact_indices = contact_indices.copy()
            if contact_indices.shape[0] == order.shape[0]:
                contact_indices = contact_indices[order]

            witness_contact_forces_local = np.asarray(
                grasp_result.get("witness_contact_forces_local", np.zeros((0, 3), dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1, 3)
            if witness_contact_forces_local.shape[0] == order.shape[0]:
                witness_contact_forces_local = witness_contact_forces_local[order]

            witness_force_vectors_local = np.asarray(
                grasp_result.get("witness_force_vectors_local", np.zeros((0, 3), dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1, 3)
            if witness_force_vectors_local.shape[0] == order.shape[0]:
                witness_force_vectors_local = witness_force_vectors_local[order]

        live_targets = {
            "contact_points_local": contact_points_local.copy(),
            "normals_local": normals_local.copy(),
            "contact_points_world": contact_points_world.copy(),
            "inward_normals_world": inward_normals_world.copy(),
            "outward_normals_world": outward_normals_world.copy(),
            "virtual_points_world": virtual_points_world.copy(),
            "left_virtual_point": virtual_points_world[0].copy(),
            "right_virtual_point": virtual_points_world[1].copy(),
            "object_pos": object_pos.copy(),
            "object_quat": mat_to_quat_wxyz(object_rot),
            "contact_indices": contact_indices.copy(),
            "raw_contact_indices": raw_contact_indices.copy(),
            "raw_contact_points_local": raw_contact_points_local.copy(),
            "raw_normals_local": raw_normals_local.copy(),
            "contact_order": order.copy(),
            "witness_contact_forces_local": witness_contact_forces_local.copy(),
            "witness_force_vectors_local": witness_force_vectors_local.copy(),
            "total_cost": float(total_cost),
            "region_score": float(region_score),
            "antipodal_margin": float(antipodal_margin),
            "solve_time": float(grasp_solve_time),
            "wall_time": float(grasp_wall_time),
            "live_target_wall_time": float(live_elapsed),
            "optimizer_search_timing": optimizer_search_timing,
        }
        self._attach_contact_wrench_targets(
            live_targets,
            "witness",
            contact_points_local,
            witness_force_vectors_local,
            object_rot,
        )
        if "desired_force_vectors_local" in live_targets:
            self._attach_contact_wrench_targets(
                live_targets,
                "desired",
                contact_points_local,
                np.asarray(live_targets["desired_force_vectors_local"], dtype=np.float64).reshape(-1, 3),
                object_rot,
            )
        if bool(getattr(self.args, "print_contact_timing", True)):
            print(
                "[timing:bigrasp] _get_live_contact_targets "
                f"total={live_elapsed:.4f}s "
                f"mode={optimizer_search_timing.get('mode', 'unknown')}"
            )
        return live_targets

    def _project_cached_contact_targets(self, cached_targets, object_pos, object_rot, virtual_offset=None):
        cached_targets = self._copy_contact_targets(cached_targets)
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        virtual_offset = float(self.args.planner_attract_offset if virtual_offset is None else virtual_offset)

        contact_points_local = np.asarray(cached_targets["contact_points_local"], dtype=np.float64).reshape(2, 3)
        normals_local = np.asarray(cached_targets["normals_local"], dtype=np.float64).reshape(2, 3)
        contact_points_world = (object_rot @ contact_points_local.T).T + object_pos[None, :]
        inward_normals_world = (object_rot @ normals_local.T).T
        outward_normals_world = -inward_normals_world
        virtual_points_world = self._offset_points_along_normals(
            contact_points_world,
            outward_normals_world,
            virtual_offset,
        )

        projected = self._copy_contact_targets(cached_targets)
        projected.update(
            {
                "contact_points_world": contact_points_world.copy(),
                "inward_normals_world": inward_normals_world.copy(),
                "outward_normals_world": outward_normals_world.copy(),
                "virtual_points_world": virtual_points_world.copy(),
                "left_virtual_point": virtual_points_world[0].copy(),
                "right_virtual_point": virtual_points_world[1].copy(),
                "object_pos": object_pos.copy(),
                "object_quat": mat_to_quat_wxyz(object_rot),
            }
        )

        desired_force_vectors_local = np.asarray(
            projected.get("desired_force_vectors_local", np.zeros((0, 3), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1, 3)
        if desired_force_vectors_local.shape[0] == 2:
            self._attach_contact_wrench_targets(
                projected,
                "desired",
                contact_points_local,
                desired_force_vectors_local,
                object_rot,
            )

        witness_force_vectors_local = np.asarray(
            projected.get("witness_force_vectors_local", np.zeros((0, 3), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1, 3)
        if witness_force_vectors_local.shape[0] == 2:
            self._attach_contact_wrench_targets(
                projected,
                "witness",
                contact_points_local,
                witness_force_vectors_local,
                object_rot,
            )
        return projected

    def _resolve_stage_object_target(self, object_target_fn, step, current_pos, current_quat, current_rot):
        current_pos = np.asarray(current_pos, dtype=np.float64).reshape(3)
        current_quat = np.asarray(current_quat, dtype=np.float64).reshape(4)
        current_rot = _project_to_rotation_matrix(current_rot)
        if object_target_fn is None:
            return current_pos.copy(), current_quat.copy()

        target = object_target_fn(step, current_pos.copy(), current_quat.copy(), current_rot.copy())
        if isinstance(target, dict):
            target_pos = np.asarray(
                target.get("object_target_pos", target.get("pos", current_pos)),
                dtype=np.float64,
            ).reshape(3)
            if "object_target_quat" in target:
                target_quat = np.asarray(target["object_target_quat"], dtype=np.float64).reshape(4)
            elif "quat" in target:
                target_quat = np.asarray(target["quat"], dtype=np.float64).reshape(4)
            elif "object_target_rot" in target:
                target_quat = mat_to_quat_wxyz(target["object_target_rot"])
            elif "rot" in target:
                target_quat = mat_to_quat_wxyz(target["rot"])
            else:
                target_quat = current_quat.copy()
            return target_pos, target_quat

        target_pos, target_quat = target
        return (
            np.asarray(target_pos, dtype=np.float64).reshape(3),
            np.asarray(target_quat, dtype=np.float64).reshape(4),
        )

    def _run_live_contact_plan_stage(
        self,
        label,
        max_steps,
        left_step_rot,
        right_step_rot,
        verify_cost_1,
        verify_cost_2,
        virtual_offset,
        goal_offset,
        pos_tol,
        rot_tol=None,
        success_fn=None,
        world_mode="with_pedestal",
        object_target_fn=None,
        initial_contact_targets=None,
        planner_target_fn=None,
        apply_optimizer_object_torque=False,
    ):
        rot_tol = self.args.ik_rot_tol if rot_tol is None else float(rot_tol)
        if int(max_steps) <= 0:
            return False, {}
        if initial_contact_targets is None:
            raise ValueError(f"{label} requires cached contact targets from the initial optimizer solve.")

        self._set_curobo_world_mode(world_mode)
        planner_sol_guess = None
        last_report_step = -1
        info = {}
        cached_contact_targets = self._copy_contact_targets(initial_contact_targets)
        left_step_rot = _project_to_rotation_matrix(left_step_rot)
        right_step_rot = _project_to_rotation_matrix(right_step_rot)
        stage_uses_viewer = self.viewer is not None
        timing_sum = {
            "target_update": 0.0,
            "planner_contacts": 0.0,
            "planner_solve": 0.0,
            "optimizer_wrench_solve": 0.0,
            "cartesian_step": 0.0,
            "post_update": 0.0,
        }
        single_contact_hold_state = {
            "side": None,
            "target_pos": None,
            "target_quat": None,
        }

        for step in range(max_steps):
            if not self.is_running():
                break

            object_pos, object_quat, object_rot = self.get_object_pose()
            step_t0 = time.perf_counter()
            live_targets = self._project_cached_contact_targets(
                cached_contact_targets,
                object_pos,
                object_rot,
                virtual_offset=virtual_offset,
            )
            step_t1 = time.perf_counter()

            planner_contact_points_world = live_targets["contact_points_world"].copy()
            planner_virtual_points_world = live_targets["virtual_points_world"].copy()
            goal_points_world = self._offset_points_along_normals(
                live_targets["contact_points_world"],
                live_targets["outward_normals_world"],
                goal_offset,
            )
            stage_debug = {}
            post_step_debug_fn = None
            if planner_target_fn is not None:
                stage_targets = planner_target_fn(step, live_targets)
                if stage_targets is None:
                    stage_targets = {}
                if "goal_points_world" in stage_targets:
                    goal_points_world = np.asarray(stage_targets["goal_points_world"], dtype=np.float64).reshape(2, 3)
                if "planner_contact_points_world" in stage_targets:
                    planner_contact_points_world = np.asarray(
                        stage_targets["planner_contact_points_world"],
                        dtype=np.float64,
                    ).reshape(2, 3)
                if "planner_virtual_points_world" in stage_targets:
                    planner_virtual_points_world = np.asarray(
                        stage_targets["planner_virtual_points_world"],
                        dtype=np.float64,
                    ).reshape(2, 3)
                if stage_targets.get("debug") is not None:
                    stage_debug.update(dict(stage_targets["debug"]))
                post_step_debug_fn = stage_targets.get("post_step_debug_fn")
            object_target_pos, object_target_quat = self._resolve_stage_object_target(
                object_target_fn,
                step,
                object_pos,
                object_quat,
                object_rot,
            )
            pre_step_contacts = self.extract_object_contacts()

            curr_x = self.get_planner_state()
            phi_vec, jac_mat = self._detect_planner_contacts()
            step_t2 = time.perf_counter()
            planner_result = self.planner.plan_once(
                object_target_pos,
                object_target_quat,
                curr_x,
                phi_vec,
                jac_mat,
                sol_guess=planner_sol_guess,
                verify_cost_param_1=float(verify_cost_1),
                verify_cost_param_2=float(verify_cost_2),
                virtual_point_1=planner_virtual_points_world[0],
                virtual_point_2=planner_virtual_points_world[1],
                contact_point_1=planner_contact_points_world[0],
                contact_point_2=planner_contact_points_world[1],
            )
            step_t3 = time.perf_counter()
            planner_sol_guess = planner_result["sol_guess"]
            self.plan_params.sol_guess_ = planner_sol_guess
            planner_backend = str(planner_result.get("solver_backend", self.plan_params.planner_solver_))
            planner_status = str(planner_result.get("solve_status", ""))
            action = np.asarray(planner_result["action"], dtype=np.float64).reshape(-1)
            if action.shape[0] != 6:
                raise RuntimeError(f"Expected a 6D dual-arm plan_once action, got shape {action.shape}.")

            applied_object_force_world = None
            applied_object_torque_world = None
            contact_mode = self._classify_fingertip_contact_mode(pre_step_contacts)
            stage_debug["contact_mode"] = contact_mode
            step_t3a = time.perf_counter()
            if apply_optimizer_object_torque:
                if contact_mode == "bilateral":
                    single_contact_hold_state["side"] = None
                    single_contact_hold_state["target_pos"] = None
                    single_contact_hold_state["target_quat"] = None
                    applied_object_torque_world, object_wrench_debug = self._compute_optimizer_object_torque_world(
                        pre_step_contacts,
                        object_pos,
                        object_rot,
                        object_target_pos,
                        object_target_quat,
                    )
                    stage_debug.update(object_wrench_debug)
                elif contact_mode in {"left_only", "right_only"}:
                    hold_side = "left" if contact_mode == "left_only" else "right"
                    if single_contact_hold_state["side"] != hold_side:
                        single_contact_hold_state["side"] = hold_side
                        single_contact_hold_state["target_pos"] = np.asarray(object_pos, dtype=np.float64).copy()
                        single_contact_hold_state["target_quat"] = np.asarray(object_quat, dtype=np.float64).copy()
                    (
                        applied_object_force_world,
                        applied_object_torque_world,
                        object_wrench_debug,
                    ) = self._compute_single_contact_stabilizing_wrench_world(
                        pre_step_contacts,
                        object_pos,
                        object_quat,
                        object_rot,
                        single_contact_hold_state["target_pos"],
                        single_contact_hold_state["target_quat"],
                    )
                    stage_debug.update(object_wrench_debug)
                else:
                    single_contact_hold_state["side"] = None
                    single_contact_hold_state["target_pos"] = None
                    single_contact_hold_state["target_quat"] = None
            step_t3b = time.perf_counter()

            self.step_cartesian_action(
                action[:3],
                action[3:6],
                left_step_rot,
                right_step_rot,
                object_force_world=applied_object_force_world,
                object_torque_world=applied_object_torque_world,
                sync_visualization=False,
            )
            step_t4 = time.perf_counter()

            self.set_marker("contact_point1", live_targets["contact_points_world"][0])
            self.set_marker("contact_point2", live_targets["contact_points_world"][1])
            self.set_marker("obj_point", object_target_pos)
            self.set_marker("goal", object_target_pos, object_target_quat)
            mujoco.mj_forward(self.model, self.data)
            self._sync_visualization()
            step_t5 = time.perf_counter()
            if callable(post_step_debug_fn):
                post_step_debug = post_step_debug_fn()
                if post_step_debug is not None:
                    stage_debug.update(dict(post_step_debug))

            timing = {
                "target_update": step_t1 - step_t0,
                "planner_contacts": step_t2 - step_t1,
                "planner_solve": step_t3 - step_t2,
                "optimizer_wrench_solve": step_t3b - step_t3a,
                "cartesian_step": step_t4 - step_t3b,
                "post_update": step_t5 - step_t4,
            }
            for key, value in timing.items():
                timing_sum[key] += float(value)

            left_err, left_rot_err = self._tip_target_error(self.left_arm, goal_points_world[0], left_step_rot)
            right_err, right_rot_err = self._tip_target_error(self.right_arm, goal_points_world[1], right_step_rot)
            contacts = self.extract_object_contacts()
            info = {
                "step": step,
                "contacts": contacts,
                "left_force": self._max_normal_force(contacts["left"]),
                "right_force": self._max_normal_force(contacts["right"]),
                "left_pos_err": float(left_err),
                "right_pos_err": float(right_err),
                "left_rot_err": float(left_rot_err),
                "right_rot_err": float(right_rot_err),
                "left_goal_pos": goal_points_world[0].copy(),
                "right_goal_pos": goal_points_world[1].copy(),
                "left_goal_rot": left_step_rot.copy(),
                "right_goal_rot": right_step_rot.copy(),
                "left_planner_cmd": action[:3].copy(),
                "right_planner_cmd": action[3:6].copy(),
                "planner_backend": planner_backend,
                "planner_status": planner_status,
                "object_pos": self.get_object_pose()[0].copy(),
                "planner_object_target_pos": np.asarray(object_target_pos, dtype=np.float64).copy(),
                "planner_object_target_quat": np.asarray(object_target_quat, dtype=np.float64).copy(),
                "contact_points_local": live_targets["contact_points_local"].copy(),
                "contact_points_world": live_targets["contact_points_world"].copy(),
                "normals_local": live_targets["normals_local"].copy(),
                "inward_normals_world": live_targets["inward_normals_world"].copy(),
                "outward_normals_world": live_targets["outward_normals_world"].copy(),
                "planner_contact_points_world": planner_contact_points_world.copy(),
                "planner_virtual_points_world": planner_virtual_points_world.copy(),
                "left_virtual_point": planner_virtual_points_world[0].copy(),
                "right_virtual_point": planner_virtual_points_world[1].copy(),
                "grasp_cost": float(live_targets["total_cost"]),
                "region_score": float(live_targets["region_score"]),
                "antipodal_margin": float(live_targets["antipodal_margin"]),
                "contact_targets": self._copy_contact_targets(live_targets),
                "timing": dict(timing),
            }
            for key, value in stage_debug.items():
                if isinstance(value, np.ndarray):
                    info[key] = value.copy()
                else:
                    info[key] = copy.deepcopy(value)
            if "modeled_normal_forces" in info:
                modeled_normal_forces = np.asarray(info["modeled_normal_forces"], dtype=np.float64).reshape(-1)
                if modeled_normal_forces.size == 2:
                    info["left_modeled_force"] = float(modeled_normal_forces[0])
                    info["right_modeled_force"] = float(modeled_normal_forces[1])

            loop_total_time = (
                timing["target_update"]
                + timing["planner_contacts"]
                + timing["planner_solve"]
                + timing["optimizer_wrench_solve"]
                + timing["cartesian_step"]
                + timing["post_update"]
            )
            print(
                f"[timing:{label}] step={step:04d} "
                f"target_update={timing['target_update']:.4f}s "
                f"planner_contacts={timing['planner_contacts']:.4f}s "
                f"planner_solve={timing['planner_solve']:.4f}s "
                f"optimizer_wrench={timing['optimizer_wrench_solve']:.4f}s "
                f"cartesian_step={timing['cartesian_step']:.4f}s "
                f"post_update={timing['post_update']:.4f}s "
                f"total={loop_total_time:.4f}s"
            )

            if bool(self.args.verbose) and (step == 0 or step == max_steps - 1 or step - last_report_step >= 40):
                modeled_force_text = ""
                if "left_modeled_force" in info and "right_modeled_force" in info:
                    modeled_force_text = (
                        f" modeled_force=({info['left_modeled_force']:.3f},{info['right_modeled_force']:.3f})"
                    )
                contact_mode_text = ""
                if "contact_mode" in info:
                    contact_mode_text = f" contact_mode={info['contact_mode']}"
                print(
                    f"[{label}] step={step:04d} "
                    f"left_err={left_err:.4f}/{left_rot_err:.4f} "
                    f"right_err={right_err:.4f}/{right_rot_err:.4f} "
                    f"left_force={info['left_force']:.3f} right_force={info['right_force']:.3f} "
                    f"planner={planner_backend} "
                    f"{contact_mode_text}{modeled_force_text} "
                    f"grasp_cost={info['grasp_cost']:.4f} "
                    f"time(update={timing['target_update']:.4f}s "
                    f"contacts={timing['planner_contacts']:.4f}s "
                    f"plan={timing['planner_solve']:.4f}s "
                    f"ctrl={timing['cartesian_step']:.4f}s "
                    f"post={timing['post_update']:.4f}s)"
                )
                if step == 0:
                    print(
                        f"  stage_cfg: world_mode={world_mode} "
                        f"verify=({float(verify_cost_1):.1f},{float(verify_cost_2):.1f}) "
                        f"virtual_offset={float(virtual_offset):.4f} goal_offset={float(goal_offset):.4f} "
                        f"contacts=projected_local "
                        f"viewer={'on' if stage_uses_viewer else 'off'}"
                    )
                if planner_status:
                    print(f"  planner_status: {planner_status}")
                last_report_step = step

            pose_ok = (
                left_err < pos_tol
                and right_err < pos_tol
                and left_rot_err < rot_tol
                and right_rot_err < rot_tol
            )
            if success_fn is None:
                if pose_ok:
                    info["timing_avg"] = {
                        key: value / max(step + 1, 1)
                        for key, value in timing_sum.items()
                    }
                    return True, info
            elif success_fn(info):
                info["timing_avg"] = {
                    key: value / max(step + 1, 1)
                    for key, value in timing_sum.items()
                }
                return True, info

        if max_steps > 0:
            info["timing_avg"] = {
                key: value / max(int(max_steps), 1)
                for key, value in timing_sum.items()
            }
        return False, info

    def _solve_stage_ik_pose_set(self, contact_points_local, normals_local, stage_specs):
        stage_pose_set = {}
        original_world_mode = self.left_arm.current_world_mode
        for stage_name, spec in stage_specs.items():
            world_mode = str(spec.get("world_mode", "with_pedestal"))
            center_offset = float(spec["center_offset"])
            object_pos = np.asarray(spec["object_pos"], dtype=np.float64).reshape(3)
            object_rot = _project_to_rotation_matrix(spec["object_rot"])

            self._set_curobo_world_mode(world_mode)
            self._update_inter_arm_worlds()
            stage_targets = self._stage_targets_from_object_pose(
                contact_points_local,
                normals_local,
                object_pos,
                object_rot,
                center_offset=center_offset,
            )
            left_ik = self.solve_arm_ik(
                self.left_arm,
                stage_targets["left_tip_pos"],
                stage_targets["left_tip_rot"],
            )
            right_ik = self.solve_arm_ik(
                self.right_arm,
                stage_targets["right_tip_pos"],
                stage_targets["right_tip_rot"],
            )
            stage_pose_set[stage_name] = {
                "targets": stage_targets,
                "left": left_ik,
                "right": right_ik,
                "world_mode": world_mode,
            }

        self._set_curobo_world_mode(original_world_mode)
        self._update_inter_arm_worlds()
        return stage_pose_set

    def _build_stage2_tracking_rotations(self, left_stage_rot, right_stage_rot):
        left_stage_rot = _tilt_rotation_upward(left_stage_rot, self.args.stage2_up_tilt)
        right_stage_rot = _tilt_rotation_upward(right_stage_rot, self.args.stage2_up_tilt)
        return (
            _project_to_rotation_matrix(left_stage_rot),
            _project_to_rotation_matrix(right_stage_rot),
        )

    def _run_precomputed_dual_arm_plan_stage(
        self,
        label,
        max_steps,
        left_virtual_point,
        right_virtual_point,
        left_contact_ik,
        right_contact_ik,
        object_target_pos,
        object_target_quat,
        left_goal_pos,
        right_goal_pos,
        left_goal_rot,
        right_goal_rot,
        verify_cost_1,
        verify_cost_2,
        pos_tol,
        rot_tol=None,
        success_fn=None,
        world_mode="with_pedestal",
    ):
        rot_tol = self.args.ik_rot_tol if rot_tol is None else float(rot_tol)
        if int(max_steps) <= 0:
            return False, {}

        self._set_curobo_world_mode(world_mode)
        planner_sol_guess = None
        last_report_step = -1
        info = {}

        left_virtual_point = np.asarray(left_virtual_point, dtype=np.float64).reshape(3)
        right_virtual_point = np.asarray(right_virtual_point, dtype=np.float64).reshape(3)
        object_target_pos = np.asarray(object_target_pos, dtype=np.float64).reshape(3)
        object_target_quat = np.asarray(object_target_quat, dtype=np.float64).reshape(4)
        left_goal_pos = np.asarray(left_goal_pos, dtype=np.float64).reshape(3)
        right_goal_pos = np.asarray(right_goal_pos, dtype=np.float64).reshape(3)

        for step in range(max_steps):
            if not self.is_running():
                break

            self._update_inter_arm_worlds()
            left_curr_pos, left_curr_rot = self.get_tip_pose(self.left_arm)
            right_curr_pos, right_curr_rot = self.get_tip_pose(self.right_arm)
            left_step_rot = left_goal_rot(step, left_curr_pos, left_curr_rot) if callable(left_goal_rot) else left_goal_rot
            right_step_rot = right_goal_rot(step, right_curr_pos, right_curr_rot) if callable(right_goal_rot) else right_goal_rot
            left_step_rot = _project_to_rotation_matrix(left_step_rot)
            right_step_rot = _project_to_rotation_matrix(right_step_rot)

            curr_x = self.get_planner_state()
            phi_vec, jac_mat = self._detect_planner_contacts()
            planner_result = self.planner.plan_once(
                object_target_pos,
                object_target_quat,
                curr_x,
                phi_vec,
                jac_mat,
                sol_guess=planner_sol_guess,
                verify_cost_param_1=float(verify_cost_1),
                verify_cost_param_2=float(verify_cost_2),
                virtual_point_1=left_virtual_point,
                virtual_point_2=right_virtual_point,
                contact_point_1=left_contact_ik.solved_tip_pos_world,
                contact_point_2=right_contact_ik.solved_tip_pos_world,
            )
            planner_sol_guess = planner_result["sol_guess"]
            self.plan_params.sol_guess_ = planner_sol_guess
            planner_backend = str(planner_result.get("solver_backend", self.plan_params.planner_solver_))
            planner_status = str(planner_result.get("solve_status", ""))
            action = np.asarray(planner_result["action"], dtype=np.float64).reshape(-1)
            if action.shape[0] != 6:
                raise RuntimeError(f"Expected a 6D dual-arm plan_once action, got shape {action.shape}.")

            self.step_cartesian_action(
                action[:3],
                action[3:6],
                left_step_rot,
                right_step_rot,
                sync_visualization=False,
            )

            self.set_marker("obj_point", object_target_pos)
            mujoco.mj_forward(self.model, self.data)
            self._sync_visualization()

            left_err, left_rot_err = self._tip_target_error(self.left_arm, left_goal_pos, left_step_rot)
            right_err, right_rot_err = self._tip_target_error(self.right_arm, right_goal_pos, right_step_rot)
            contacts = self.extract_object_contacts()
            info = {
                "step": step,
                "contacts": contacts,
                "left_force": self._max_normal_force(contacts["left"]),
                "right_force": self._max_normal_force(contacts["right"]),
                "left_pos_err": float(left_err),
                "right_pos_err": float(right_err),
                "left_rot_err": float(left_rot_err),
                "right_rot_err": float(right_rot_err),
                "left_goal_pos": left_goal_pos.copy(),
                "right_goal_pos": right_goal_pos.copy(),
                "left_goal_rot": left_step_rot.copy(),
                "right_goal_rot": right_step_rot.copy(),
                "left_goal_q_mj": np.asarray(left_contact_ik.q_mj, dtype=np.float64).copy(),
                "right_goal_q_mj": np.asarray(right_contact_ik.q_mj, dtype=np.float64).copy(),
                "left_ik_ok": bool(left_contact_ik.success),
                "right_ik_ok": bool(right_contact_ik.success),
                "left_ik_pos_err": float(left_contact_ik.position_error),
                "right_ik_pos_err": float(right_contact_ik.position_error),
                "left_ik_rot_err": float(left_contact_ik.rotation_error),
                "right_ik_rot_err": float(right_contact_ik.rotation_error),
                "left_ik_constraint_total": float(left_contact_ik.constraint_total),
                "right_ik_constraint_total": float(right_contact_ik.constraint_total),
                "left_ik_failure_reason": str(left_contact_ik.failure_reason),
                "right_ik_failure_reason": str(right_contact_ik.failure_reason),
                "object_pos": self.get_object_pose()[0].copy(),
                "left_planner_cmd": action[:3].copy(),
                "right_planner_cmd": action[3:6].copy(),
                "planner_backend": planner_backend,
                "planner_status": planner_status,
            }

            if bool(self.args.verbose) and (step == 0 or step == max_steps - 1 or step - last_report_step >= 40):
                print(
                    f"[{label}] step={step:04d} "
                    f"left_err={left_err:.4f}/{left_rot_err:.4f} "
                    f"right_err={right_err:.4f}/{right_rot_err:.4f} "
                    f"left_force={info['left_force']:.3f} right_force={info['right_force']:.3f} "
                    f"left_ik={left_contact_ik.success} right_ik={right_contact_ik.success} "
                    f"planner={planner_backend}"
                )
                if step == 0:
                    print(
                        f"  stage_cfg: world_mode={world_mode} "
                        f"verify=({float(verify_cost_1):.1f},{float(verify_cost_2):.1f}) "
                        f"planner=plan_once+impedance[{planner_backend}]"
                    )
                if planner_status:
                    print(f"  planner_status: {planner_status}")
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

    def set_marker(self, name, pos, quat=None):
        body_id = self.marker_body_ids[name]
        self.model.body_pos[body_id] = np.asarray(pos, dtype=np.float64)
        if quat is not None:
            self.model.body_quat[body_id] = np.asarray(quat, dtype=np.float64)

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

    def get_planner_state(self):
        obj_pos, obj_quat, _ = self.get_object_pose()
        left_tip_pos = self.get_tip_pos(self.left_arm)
        right_tip_pos = self.get_tip_pos(self.right_arm)
        return np.hstack([obj_pos, obj_quat, left_tip_pos, right_tip_pos]).astype(np.float32)

    def _build_object_jacobian(self, point_local):
        jacobian = np.zeros((3, self.plan_params.n_qvel_), dtype=np.float64)
        jacobian[:, :3] = np.eye(3, dtype=np.float64)
        jacobian[0, 4] = point_local[2]
        jacobian[0, 5] = -point_local[1]
        jacobian[1, 3] = -point_local[2]
        jacobian[1, 5] = point_local[0]
        jacobian[2, 3] = point_local[1]
        jacobian[2, 4] = -point_local[0]
        return jacobian

    def _build_tip_jacobian_block(self, arm):
        jacobian = np.zeros((3, self.plan_params.n_qvel_), dtype=np.float64)
        if arm.prefix == "left_":
            jacobian[:, 6:9] = np.eye(3, dtype=np.float64)
        else:
            jacobian[:, 9:12] = np.eye(3, dtype=np.float64)
        return jacobian

    def _reformat_planner_contacts(self, con_phi_list=None, con_jac_list=None):
        con_phi_list = [] if con_phi_list is None else con_phi_list
        con_jac_list = [] if con_jac_list is None else con_jac_list
        phi_vec = np.ones((self.plan_params.max_ncon_ * 4,), dtype=np.float32)
        jac_mat = np.zeros((self.plan_params.max_ncon_ * 4, self.plan_params.n_qvel_), dtype=np.float32)
        for i in range(min(len(con_phi_list), self.plan_params.max_ncon_)):
            phi_vec[4 * i : 4 * i + 4] = float(con_phi_list[i])
            jac_mat[4 * i : 4 * i + 4] = np.asarray(con_jac_list[i], dtype=np.float32)
        return phi_vec, jac_mat

    def _detect_planner_contacts(self, return_wrench_maps=False):
        if self.object_is_ghost:
            phi_vec, jac_mat = self._reformat_planner_contacts([], [])
            if return_wrench_maps:
                zero_map = np.zeros((3, self.plan_params.max_ncon_ * 4), dtype=np.float32)
                return phi_vec, jac_mat, zero_map.copy(), zero_map.copy()
            return phi_vec, jac_mat
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

        obj_pos, _, obj_rot = self.get_object_pose()
        left_robot_jacobian = self._build_tip_jacobian_block(self.left_arm)
        right_robot_jacobian = self._build_tip_jacobian_block(self.right_arm)

        phi_vec = np.ones((self.plan_params.max_ncon_ * 4,), dtype=np.float32)
        jac_mat = np.zeros((self.plan_params.max_ncon_ * 4, self.plan_params.n_qvel_), dtype=np.float32)
        robot_contact_force_map = np.zeros((3, self.plan_params.max_ncon_ * 4), dtype=np.float32)
        robot_contact_torque_map = np.zeros((3, self.plan_params.max_ncon_ * 4), dtype=np.float32)

        row_idx = 0
        for i in range(self.data.ncon):
            if row_idx >= self.plan_params.max_ncon_:
                break

            contact_i = self.data.contact[i]
            geom1 = int(contact_i.geom1)
            geom2 = int(contact_i.geom2)
            if self.obj_geom_id not in {geom1, geom2}:
                continue

            body1_id = int(self.model.geom_bodyid[geom1])
            body2_id = int(self.model.geom_bodyid[geom2])
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or ""
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or ""

            object_is_first = body1_name == "obj"
            other_body_name = body2_name if object_is_first else body1_name

            con_pos = np.asarray(contact_i.pos, dtype=np.float64).copy()
            con_dist = float(contact_i.dist) * 0.5
            con_mu = float(self.args.arm_friction)
            con_frame = np.asarray(contact_i.frame, dtype=np.float64).reshape((-1, 3)).T
            con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

            con_pos_local = obj_rot.T @ (con_pos - obj_pos)
            object_jacobian = self._build_object_jacobian(con_pos_local)
            con_jacp_obj = con_frame_pmd.T @ object_jacobian

            is_robot_contact = False
            if other_body_name.startswith("left_"):
                con_jacp_other = con_frame_pmd.T @ left_robot_jacobian
                is_robot_contact = True
            elif other_body_name.startswith("right_"):
                con_jacp_other = con_frame_pmd.T @ right_robot_jacobian
                is_robot_contact = True
            else:
                con_jacp_other = np.zeros((5, self.plan_params.n_qvel_), dtype=np.float64)

            con_jac = con_jacp_obj - con_jacp_other
            con_jac = con_jac[0] + con_mu * con_jac[1:]
            phi_vec[4 * row_idx : 4 * row_idx + 4] = float(con_dist)
            jac_mat[4 * row_idx : 4 * row_idx + 4, :] = np.asarray(con_jac, dtype=np.float32)

            if is_robot_contact:
                force_dirs_world = _contact_force_edge_directions_world_from_frame(con_frame, con_mu)
                moment_arm_world = np.asarray(con_pos - obj_pos, dtype=np.float64).reshape(3)
                torque_dirs_world = np.column_stack(
                    [np.cross(moment_arm_world, force_dirs_world[:, edge_idx]) for edge_idx in range(4)]
                )
                robot_contact_force_map[:, 4 * row_idx : 4 * row_idx + 4] = force_dirs_world.astype(np.float32)
                robot_contact_torque_map[:, 4 * row_idx : 4 * row_idx + 4] = torque_dirs_world.astype(np.float32)

            row_idx += 1

        if return_wrench_maps:
            return phi_vec, jac_mat, robot_contact_force_map, robot_contact_torque_map
        return phi_vec, jac_mat

    def get_current_joint_position(self, arm, update_kinematics=False):
        if update_kinematics:
            mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.qpos[arm.qpos_adr], dtype=np.float64).copy()

    def get_current_joint_velocity(self, arm, update_kinematics=False):
        if update_kinematics:
            mujoco.mj_forward(self.model, self.data)
        return np.asarray(self.data.qvel[arm.dof_adr], dtype=np.float64).copy()

    def get_arm_jacobian(self, arm, update_kinematics=False):
        if update_kinematics:
            mujoco.mj_forward(self.model, self.data)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        tip_pos, _ = self._get_tip_geom_pose(arm)
        tip_body_id = int(self.model.geom_bodyid[arm.tip_geom_id])
        mujoco.mj_jac(
            self.model,
            self.data,
            jacp=jacp,
            jacr=jacr,
            point=tip_pos,
            body=tip_body_id,
        )
        return np.vstack([jacp[:, arm.dof_adr], jacr[:, arm.dof_adr]])

    def set_control_torque(self, arm, torque):
        self.data.ctrl[arm.actuator_ids] = np.clip(
            np.asarray(torque, dtype=np.float64),
            -arm.torque_limits,
            arm.torque_limits,
        )

    def set_desired_tip_pose(self, arm, position, orientation):
        arm.position_d = np.asarray(position, dtype=np.float64).copy()
        arm.orientation_d = _project_to_rotation_matrix(orientation)
        arm.p_d = arm.position_d.copy()
        arm.R_d = arm.orientation_d.copy()

    def _compute_cartesian_impedance_control(self, arm, q=None, dq=None, jacobian=None, tip_pose=None):
        q = self.get_current_joint_position(arm) if q is None else np.asarray(q, dtype=np.float64).copy()
        dq = self.get_current_joint_velocity(arm) if dq is None else np.asarray(dq, dtype=np.float64).copy()
        jacobian = self.get_arm_jacobian(arm) if jacobian is None else np.asarray(jacobian, dtype=np.float64).copy()
        if tip_pose is None:
            p_current, rot_current = self.get_tip_pose(arm)
        else:
            p_current, rot_current = tip_pose
            p_current = np.asarray(p_current, dtype=np.float64).copy()
            rot_current = np.asarray(rot_current, dtype=np.float64).reshape(3, 3).copy()

        error = np.zeros(6, dtype=np.float64)
        error[:3] = p_current - arm.position_d
        rot_error = rot_current.T @ arm.orientation_d
        error_quat = Rotation.from_matrix(rot_error).as_quat()
        if error_quat[3] < 0.0:
            error_quat = -error_quat
        error[3:] = -rot_current @ error_quat[:3]

        velocity = jacobian @ dq
        desired_wrench = -arm.cartesian_stiffness @ error - arm.cartesian_damping @ velocity
        tau_task = jacobian.T @ desired_wrench

        jacobian_pinv = pinv(jacobian.T)
        nullspace_proj = np.eye(7) - jacobian.T @ jacobian_pinv
        tau_nullspace = nullspace_proj @ (
            arm.nullspace_stiffness * (arm.home_q - q)
            - 2.0 * np.sqrt(arm.nullspace_stiffness) * dq
        )
        tau_bias = np.asarray(self.data.qfrc_bias[arm.dof_adr], dtype=np.float64).copy()
        tau = tau_task + tau_nullspace + tau_bias
        return np.clip(tau, -arm.torque_limits, arm.torque_limits)

    def _gather_arm_control_state(self, arm):
        q = np.asarray(self.data.qpos[arm.qpos_adr], dtype=np.float64).copy()
        dq = np.asarray(self.data.qvel[arm.dof_adr], dtype=np.float64).copy()
        tip_pose = self.get_tip_pose(arm)
        jacobian = self.get_arm_jacobian(arm)
        return {
            "q": q,
            "dq": dq,
            "tip_pose": tip_pose,
            "jacobian": jacobian,
        }

    def _compute_dual_arm_cartesian_impedance_control(self):
        mujoco.mj_forward(self.model, self.data)
        left_state = self._gather_arm_control_state(self.left_arm)
        right_state = self._gather_arm_control_state(self.right_arm)
        left_tau = self._compute_cartesian_impedance_control(
            self.left_arm,
            q=left_state["q"],
            dq=left_state["dq"],
            jacobian=left_state["jacobian"],
            tip_pose=left_state["tip_pose"],
        )
        right_tau = self._compute_cartesian_impedance_control(
            self.right_arm,
            q=right_state["q"],
            dq=right_state["dq"],
            jacobian=right_state["jacobian"],
            tip_pose=right_state["tip_pose"],
        )
        return left_tau, right_tau

    def _planner_tracking_steps(self):
        base_steps = max(int(self.args.mj_steps_per_command) * max(int(self.args.command_substeps), 1), 1)
        planner_steps = max(int(np.ceil(float(self.plan_params.h_) / float(self.model.opt.timestep))), 1)
        return max(base_steps, planner_steps)

    def _set_object_wrench_world(self, force_world=None, torque_world=None):
        if self.object_body_id < 0:
            return
        self.data.xfrc_applied[self.object_body_id, :] = 0.0
        if force_world is not None:
            self.data.xfrc_applied[self.object_body_id, :3] = np.asarray(force_world, dtype=np.float64).reshape(3)
        if torque_world is not None:
            self.data.xfrc_applied[self.object_body_id, 3:6] = np.asarray(torque_world, dtype=np.float64).reshape(3)

    def get_object_velocity_world(self):
        if self.object_dof_adr < 0:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        velocity = np.asarray(
            self.data.qvel[self.object_dof_adr : self.object_dof_adr + 6],
            dtype=np.float64,
        ).reshape(6)
        return velocity[:3].copy(), velocity[3:].copy()

    def _get_object_application_point_world(self):
        if self.object_body_id < 0:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.data.xipos[self.object_body_id], dtype=np.float64).reshape(3).copy()

    def _compute_optimizer_torque_from_contact_points_local(self, contact_points_local, object_rot):
        contact_points_local = np.asarray(contact_points_local, dtype=np.float64).reshape(-1, 3)
        object_rot = _project_to_rotation_matrix(object_rot)
        if contact_points_local.shape[0] < 2:
            return None, {
                "bilateral_contact_active": False,
                "optimizer_solver_failed": True,
                "optimizer_torque_body_name": "obj",
                "optimizer_torque_body_id": int(self.object_body_id),
                "optimizer_torque_application_point_world": self._get_object_application_point_world(),
            }

        current_pose_local = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        gravity_local = np.hstack(
            [
                object_rot.T @ np.array([0.0, 0.0, -float(self.args.obj_mass) * 9.81], dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            ]
        )

        _, _, _, objective_value, optimizer_info, optimizer_torque_world = self.optimizer.optimize_control_input(
            current_pose_local,
            current_pose_local,
            gravity_local,
            p_arm=contact_points_local,
            r_obj_to_world=object_rot,
            return_world_torque=True,
        )

        debug = {
            "bilateral_contact_active": True,
            "optimizer_contact_points_local": contact_points_local.copy(),
            "optimizer_objective": float(objective_value),
            "optimizer_solver_failed": bool(optimizer_info.get("solver_failed", False)),
            "optimizer_torque_body_name": "obj",
            "optimizer_torque_body_id": int(self.object_body_id),
            "optimizer_torque_application_point_world": self._get_object_application_point_world(),
        }
        for key in (
            "contact_force_local",
            "contact_torque_local",
            "contact_force_world",
            "contact_torque_world",
            "contact_points_local",
            "contact_force_vectors_local",
        ):
            if key in optimizer_info:
                debug[key] = np.asarray(optimizer_info[key], dtype=np.float64).copy()

        if optimizer_torque_world is None or bool(optimizer_info.get("solver_failed", False)):
            return None, debug

        optimizer_torque_world = np.asarray(optimizer_torque_world, dtype=np.float64).reshape(3)
        if np.linalg.norm(optimizer_torque_world) < 1e-9:
            return None, debug
        debug["applied_optimizer_torque_world"] = optimizer_torque_world.copy()
        return optimizer_torque_world, debug

    @staticmethod
    def _has_active_contact(contact_items, min_normal_force=1e-5):
        return bool(contact_items) and any(float(item.get("normal_force", 0.0)) > float(min_normal_force) for item in contact_items)

    def _classify_fingertip_contact_mode(self, contacts):
        left_active = self._has_active_contact(contacts.get("left", []))
        right_active = self._has_active_contact(contacts.get("right", []))
        if left_active and right_active:
            return "bilateral"
        if left_active:
            return "left_only"
        if right_active:
            return "right_only"
        return "none"

    @staticmethod
    def _best_contact(contact_items):
        if not contact_items:
            return None
        return max(
            contact_items,
            key=lambda item: (
                float(item.get("normal_force", 0.0)),
                -abs(float(item.get("dist", 0.0))),
            ),
        )

    def _build_optimizer_target_pose_local(self, object_pos, object_rot, object_target_pos, object_target_quat):
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_rot = _project_to_rotation_matrix(object_rot)
        object_target_pos = np.asarray(object_target_pos, dtype=np.float64).reshape(3)
        object_target_quat = np.asarray(object_target_quat, dtype=np.float64).reshape(4)
        target_rot_world = quat_wxyz_to_mat(object_target_quat)
        target_rot_local = object_rot.T @ target_rot_world
        target_pos_local = object_rot.T @ (object_target_pos - object_pos)
        return np.hstack([target_pos_local, mat_to_quat_wxyz(target_rot_local)]).astype(np.float64)

    def _compute_optimizer_object_torque_world(
        self,
        contacts,
        object_pos,
        object_rot,
        object_target_pos,
        object_target_quat,
    ):
        del object_pos
        del object_target_pos
        del object_target_quat
        if not self._has_active_contact(contacts.get("left", [])) or not self._has_active_contact(contacts.get("right", [])):
            return None, {"bilateral_contact_active": False}

        left_contact = self._best_contact(contacts.get("left", []))
        right_contact = self._best_contact(contacts.get("right", []))
        if left_contact is None or right_contact is None:
            return None, {"bilateral_contact_active": False}

        contact_points_local = np.vstack(
            [
                np.asarray(left_contact["local_pos"], dtype=np.float64).reshape(3),
                np.asarray(right_contact["local_pos"], dtype=np.float64).reshape(3),
            ]
        )
        return self._compute_optimizer_torque_from_contact_points_local(contact_points_local, object_rot)

    def _compute_single_contact_stabilizing_wrench_world(
        self,
        contacts,
        object_pos,
        object_quat,
        object_rot,
        hold_target_pos,
        hold_target_quat,
    ):
        contact_mode = self._classify_fingertip_contact_mode(contacts)
        if contact_mode not in {"left_only", "right_only"}:
            return None, None, {"single_contact_stabilizer_active": False, "contact_mode": contact_mode}

        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        object_quat = np.asarray(object_quat, dtype=np.float64).reshape(4)
        object_rot = _project_to_rotation_matrix(object_rot)
        hold_target_pos = np.asarray(hold_target_pos, dtype=np.float64).reshape(3)
        hold_target_quat = np.asarray(hold_target_quat, dtype=np.float64).reshape(4)
        hold_target_rot = quat_wxyz_to_mat(hold_target_quat)
        linear_velocity_world, angular_velocity_world = self.get_object_velocity_world()

        pos_error_world = object_pos - hold_target_pos
        rot_error_world = _rotation_error(object_rot, hold_target_rot)

        stabilizing_force_world = (
            -float(self.args.single_contact_pos_kp) * pos_error_world
            - float(self.args.single_contact_lin_damping) * linear_velocity_world
        )
        stabilizing_torque_world = (
            -float(self.args.single_contact_ori_kp) * rot_error_world
            - float(self.args.single_contact_ang_damping) * angular_velocity_world
        )

        force_max = max(float(self.args.single_contact_force_max), 0.0)
        if force_max > 0.0:
            force_norm = float(np.linalg.norm(stabilizing_force_world))
            if force_norm > force_max:
                stabilizing_force_world = stabilizing_force_world * (force_max / max(force_norm, 1e-9))
        else:
            stabilizing_force_world = np.zeros(3, dtype=np.float64)

        torque_max = max(float(self.args.single_contact_torque_max), 0.0)
        if torque_max > 0.0:
            torque_norm = float(np.linalg.norm(stabilizing_torque_world))
            if torque_norm > torque_max:
                stabilizing_torque_world = stabilizing_torque_world * (torque_max / max(torque_norm, 1e-9))
        else:
            stabilizing_torque_world = np.zeros(3, dtype=np.float64)

        debug = {
            "single_contact_stabilizer_active": True,
            "contact_mode": contact_mode,
            "single_contact_side": "left" if contact_mode == "left_only" else "right",
            "single_contact_hold_target_pos": hold_target_pos.copy(),
            "single_contact_hold_target_quat": hold_target_quat.copy(),
            "single_contact_pos_error_world": pos_error_world.copy(),
            "single_contact_rot_error_world": rot_error_world.copy(),
            "single_contact_linear_velocity_world": linear_velocity_world.copy(),
            "single_contact_angular_velocity_world": angular_velocity_world.copy(),
            "applied_single_contact_force_world": stabilizing_force_world.copy(),
            "applied_single_contact_torque_world": stabilizing_torque_world.copy(),
        }
        return stabilizing_force_world, stabilizing_torque_world, debug

    def step_cartesian_action(
        self,
        left_cmd,
        right_cmd,
        left_rot_world,
        right_rot_world,
        object_force_world=None,
        object_torque_world=None,
        sync_visualization=True,
    ):
        left_curr_pos, left_curr_rot = self.get_tip_pose(self.left_arm)
        right_curr_pos, right_curr_rot = self.get_tip_pose(self.right_arm)
        left_target_pos = left_curr_pos + np.asarray(left_cmd, dtype=np.float64)
        right_target_pos = right_curr_pos + np.asarray(right_cmd, dtype=np.float64)

        num_steps = self._planner_tracking_steps()
        for step_idx in range(num_steps):
            alpha = float(step_idx + 1) / float(num_steps)
            left_interp_pos = left_curr_pos + alpha * (left_target_pos - left_curr_pos)
            right_interp_pos = right_curr_pos + alpha * (right_target_pos - right_curr_pos)
            left_interp_rot = _slerp_rotation_matrix(left_curr_rot, left_rot_world, alpha)
            right_interp_rot = _slerp_rotation_matrix(right_curr_rot, right_rot_world, alpha)

            self.set_desired_tip_pose(self.left_arm, left_interp_pos, left_interp_rot)
            self.set_desired_tip_pose(self.right_arm, right_interp_pos, right_interp_rot)
            left_tau, right_tau = self._compute_dual_arm_cartesian_impedance_control()
            self.set_control_torque(self.left_arm, left_tau)
            self.set_control_torque(self.right_arm, right_tau)
            self._set_object_wrench_world(force_world=object_force_world, torque_world=object_torque_world)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if bool(sync_visualization) and step_idx == num_steps - 1:
                self._sync_visualization()
            if self.args.real_time and self.viewer is not None:
                time.sleep(self.model.opt.timestep)
        self._set_object_wrench_world(force_world=None, torque_world=None)

    def hold_current_pose(self, num_steps=1, object_force_world=None, object_torque_world=None, sync_visualization=True):
        for arm in (self.left_arm, self.right_arm):
            tip_pos, tip_rot = self.get_tip_pose(arm)
            self.set_desired_tip_pose(arm, tip_pos, tip_rot)

        total_steps = max(int(num_steps), 1)
        for step_idx in range(total_steps):
            left_tau, right_tau = self._compute_dual_arm_cartesian_impedance_control()
            self.set_control_torque(self.left_arm, left_tau)
            self.set_control_torque(self.right_arm, right_tau)
            self._set_object_wrench_world(force_world=object_force_world, torque_world=object_torque_world)
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if bool(sync_visualization) and step_idx == total_steps - 1:
                self._sync_visualization()
            if self.args.real_time and self.viewer is not None:
                time.sleep(self.model.opt.timestep)
        self._set_object_wrench_world(force_world=None, torque_world=None)

    @staticmethod
    def _clip_cartesian_delta(delta, max_norm):
        delta = np.asarray(delta, dtype=np.float64).reshape(3)
        max_norm = max(float(max_norm), 1e-9)
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= max_norm:
            return delta.copy()
        return delta * (max_norm / delta_norm)

    def _track_desired_tip_targets(
        self,
        left_goal_pos,
        right_goal_pos,
        max_step,
        left_goal_rot=None,
        right_goal_rot=None,
        object_force_world=None,
        object_torque_world=None,
    ):
        left_curr_pos, left_curr_rot = self.get_tip_pose(self.left_arm)
        right_curr_pos, right_curr_rot = self.get_tip_pose(self.right_arm)
        left_goal_pos = np.asarray(left_goal_pos, dtype=np.float64).reshape(3)
        right_goal_pos = np.asarray(right_goal_pos, dtype=np.float64).reshape(3)
        left_goal_rot = left_curr_rot.copy() if left_goal_rot is None else _project_to_rotation_matrix(left_goal_rot)
        right_goal_rot = right_curr_rot.copy() if right_goal_rot is None else _project_to_rotation_matrix(right_goal_rot)

        left_cmd = self._clip_cartesian_delta(left_goal_pos - left_curr_pos, max_step)
        right_cmd = self._clip_cartesian_delta(right_goal_pos - right_curr_pos, max_step)
        self.step_cartesian_action(
            left_cmd,
            right_cmd,
            left_goal_rot,
            right_goal_rot,
            object_force_world=object_force_world,
            object_torque_world=object_torque_world,
            sync_visualization=False,
        )
        return left_cmd, right_cmd

    def _run_optimizer_torque_test(self, contact_targets, num_steps=None):
        num_steps = max(int(self.args.squeeze_steps if num_steps is None else num_steps), 1)
        contact_targets = self._copy_contact_targets(contact_targets)
        contact_points_local = np.asarray(contact_targets["contact_points_local"], dtype=np.float64).reshape(2, 3)
        last_info = {}

        for arm in (self.left_arm, self.right_arm):
            tip_pos, tip_rot = self.get_tip_pose(arm)
            self.set_desired_tip_pose(arm, tip_pos, tip_rot)

        self.set_marker("contact_point1", np.asarray(contact_targets["contact_points_world"][0], dtype=np.float64))
        self.set_marker("contact_point2", np.asarray(contact_targets["contact_points_world"][1], dtype=np.float64))
        mujoco.mj_forward(self.model, self.data)
        self._sync_visualization()

        for _ in range(num_steps):
            if not self.is_running():
                break
            _, _, object_rot = self.get_object_pose()
            applied_torque_world, optimizer_debug = self._compute_optimizer_torque_from_contact_points_local(
                contact_points_local,
                object_rot,
            )
            last_info = dict(optimizer_debug)
            if applied_torque_world is not None:
                last_info["applied_optimizer_torque_world"] = np.asarray(
                    applied_torque_world,
                    dtype=np.float64,
                ).reshape(3).copy()
            self.hold_current_pose(
                num_steps=1,
                object_force_world=None,
                object_torque_world=applied_torque_world,
            )

        return bool(last_info.get("applied_optimizer_torque_world") is not None), last_info

    def _run_realtime_contact_tracking_stage(
        self,
        label,
        max_steps,
        goal_mode,
        pos_tol,
        virtual_offset=None,
        goal_offset=0.0,
        success_fn=None,
        world_mode="with_pedestal",
        initial_contact_targets=None,
        apply_optimizer_object_torque=False,
    ):
        if int(max_steps) <= 0:
            return False, {}
        if initial_contact_targets is None:
            raise ValueError(f"{label} requires cached contact targets from the initial optimizer solve.")
        if goal_mode not in {"virtual", "contact"}:
            raise ValueError(f"Unsupported goal_mode: {goal_mode}")

        self._set_curobo_world_mode(world_mode)
        last_report_step = -1
        info = {}
        cached_contact_targets = self._copy_contact_targets(initial_contact_targets)
        step_limit = float(self.args.cartesian_step)

        for step in range(max_steps):
            if not self.is_running():
                break

            object_pos, object_quat, object_rot = self.get_object_pose()
            solve_t0 = time.perf_counter()
            live_targets = self._get_live_contact_targets(
                object_pos,
                object_rot,
                previous_targets=cached_contact_targets,
                virtual_offset=virtual_offset,
            )
            solve_elapsed = time.perf_counter() - solve_t0
            cached_contact_targets = self._copy_contact_targets(live_targets)

            if goal_mode == "virtual":
                goal_points_world = np.asarray(live_targets["virtual_points_world"], dtype=np.float64).reshape(2, 3)
            else:
                goal_points_world = self._offset_points_along_normals(
                    live_targets["contact_points_world"],
                    live_targets["outward_normals_world"],
                    goal_offset,
                )

            pre_step_contacts = self.extract_object_contacts()
            contact_mode = self._classify_fingertip_contact_mode(pre_step_contacts)
            applied_object_force_world = None
            applied_object_torque_world = None
            stage_debug = {"contact_mode": contact_mode}
            solve_t1 = time.perf_counter()
            if apply_optimizer_object_torque:
                if contact_mode == "bilateral":
                    applied_object_torque_world, object_wrench_debug = self._compute_optimizer_object_torque_world(
                        pre_step_contacts,
                        object_pos,
                        object_rot,
                        object_pos,
                        object_quat,
                    )
                    stage_debug.update(object_wrench_debug)
                elif contact_mode in {"left_only", "right_only"}:
                    (
                        applied_object_force_world,
                        applied_object_torque_world,
                        object_wrench_debug,
                    ) = self._compute_single_contact_stabilizing_wrench_world(
                        pre_step_contacts,
                        object_pos,
                        object_quat,
                        object_rot,
                        object_pos,
                        object_quat,
                    )
                    stage_debug.update(object_wrench_debug)
            solve_t2 = time.perf_counter()

            track_t0 = time.perf_counter()
            left_cmd, right_cmd = self._track_desired_tip_targets(
                goal_points_world[0],
                goal_points_world[1],
                step_limit,
                left_goal_rot=None,
                right_goal_rot=None,
                object_force_world=applied_object_force_world,
                object_torque_world=applied_object_torque_world,
            )
            track_t1 = time.perf_counter()

            self.set_marker("contact_point1", np.asarray(live_targets["contact_points_world"][0], dtype=np.float64))
            self.set_marker("contact_point2", np.asarray(live_targets["contact_points_world"][1], dtype=np.float64))
            mujoco.mj_forward(self.model, self.data)
            self._sync_visualization()

            left_err, left_rot_err = self._tip_target_error(self.left_arm, goal_points_world[0], None)
            right_err, right_rot_err = self._tip_target_error(self.right_arm, goal_points_world[1], None)
            contacts = self.extract_object_contacts()
            info = {
                "step": step,
                "contacts": contacts,
                "left_force": self._max_normal_force(contacts["left"]),
                "right_force": self._max_normal_force(contacts["right"]),
                "left_pos_err": float(left_err),
                "right_pos_err": float(right_err),
                "left_rot_err": float(left_rot_err),
                "right_rot_err": float(right_rot_err),
                "left_goal_pos": goal_points_world[0].copy(),
                "right_goal_pos": goal_points_world[1].copy(),
                "left_planner_cmd": np.asarray(left_cmd, dtype=np.float64).copy(),
                "right_planner_cmd": np.asarray(right_cmd, dtype=np.float64).copy(),
                "object_pos": self.get_object_pose()[0].copy(),
                "contact_points_local": live_targets["contact_points_local"].copy(),
                "contact_points_world": live_targets["contact_points_world"].copy(),
                "normals_local": live_targets["normals_local"].copy(),
                "inward_normals_world": live_targets["inward_normals_world"].copy(),
                "outward_normals_world": live_targets["outward_normals_world"].copy(),
                "planner_contact_points_world": goal_points_world.copy(),
                "planner_virtual_points_world": live_targets["virtual_points_world"].copy(),
                "left_virtual_point": live_targets["virtual_points_world"][0].copy(),
                "right_virtual_point": live_targets["virtual_points_world"][1].copy(),
                "grasp_cost": float(live_targets["total_cost"]),
                "region_score": float(live_targets["region_score"]),
                "antipodal_margin": float(live_targets["antipodal_margin"]),
                "contact_targets": self._copy_contact_targets(live_targets),
                "contact_solve_time": float(live_targets.get("solve_time", solve_elapsed)),
                "contact_solve_wall_time": float(live_targets.get("wall_time", solve_elapsed)),
                "optimizer_wrench_solve_time": float(solve_t2 - solve_t1),
                "track_control_time": float(track_t1 - track_t0),
            }
            for key, value in stage_debug.items():
                if isinstance(value, np.ndarray):
                    info[key] = value.copy()
                else:
                    info[key] = copy.deepcopy(value)

            loop_total_time = float(
                info["contact_solve_time"]
                + info["optimizer_wrench_solve_time"]
                + info["track_control_time"]
            )
            print(
                f"[timing:{label}] step={step:04d} "
                f"contact_solve={info['contact_solve_time']:.4f}s "
                f"optimizer_wrench={info['optimizer_wrench_solve_time']:.4f}s "
                f"track_control={info['track_control_time']:.4f}s "
                f"total={loop_total_time:.4f}s"
            )

            if bool(self.args.verbose) and (step == 0 or step == max_steps - 1 or step - last_report_step >= 40):
                print(
                    f"[{label}] step={step:04d} "
                    f"left_err={left_err:.4f} right_err={right_err:.4f} "
                    f"left_force={info['left_force']:.3f} right_force={info['right_force']:.3f} "
                    f"contact_mode={contact_mode} solve={info['contact_solve_time']:.4f}s "
                    f"grasp_cost={info['grasp_cost']:.4f}"
                )
                last_report_step = step

            pose_ok = left_err < pos_tol and right_err < pos_tol
            if success_fn is None:
                if pose_ok:
                    return True, info
            elif success_fn(info):
                return True, info

        return False, info

    def _run_realtime_bigrasp_loop(
        self,
        label,
        max_steps,
        pos_tol,
        rot_tol=None,
        required_normal_force=0.0,
        world_mode="with_pedestal",
        initial_contact_targets=None,
        optimizer_candidate_cache=None,
    ):
        rot_tol = self.args.ik_rot_tol if rot_tol is None else float(rot_tol)
        if int(max_steps) <= 0:
            return False, {}
        if initial_contact_targets is None:
            raise ValueError(f"{label} requires initial contact targets.")

        self._set_curobo_world_mode(world_mode)
        planner_sol_guess = None
        cached_contact_targets = self._copy_contact_targets(initial_contact_targets)
        verify_state = {"left": False, "right": False}
        stable_contact_steps = 0
        last_report_step = -1
        info = {}
        timing_sum = {
            "contact_plan_wall": 0.0,
            "target_build": 0.0,
            "planner_contacts": 0.0,
            "planner_solve": 0.0,
            "cartesian_step": 0.0,
            "post_update": 0.0,
            "wall_total": 0.0,
        }

        for step in range(int(max_steps)):
            if not self.is_running():
                break

            loop_t0 = time.perf_counter()
            object_pos, object_quat, object_rot = self.get_object_pose()

            step_t1 = time.perf_counter()
            live_targets = self._get_live_contact_targets(
                object_pos,
                object_rot,
                previous_targets=cached_contact_targets,
                virtual_offset=float(self.args.planner_attract_offset),
                candidate_cache=optimizer_candidate_cache,
            )
            cached_contact_targets = self._copy_contact_targets(live_targets)
            step_t2 = time.perf_counter()

            fingertip_targets = self._stage_targets_from_object_pose(
                live_targets["contact_points_local"],
                live_targets["normals_local"],
                object_pos,
                object_rot,
                center_offset=0.0,
            )
            left_contact_goal = np.asarray(fingertip_targets["left_tip_pos"], dtype=np.float64).reshape(3)
            right_contact_goal = np.asarray(fingertip_targets["right_tip_pos"], dtype=np.float64).reshape(3)
            left_goal_rot = _project_to_rotation_matrix(fingertip_targets["left_tip_rot"])
            right_goal_rot = _project_to_rotation_matrix(fingertip_targets["right_tip_rot"])

            left_tip_pos, _ = self.get_tip_pose(self.left_arm)
            right_tip_pos, _ = self.get_tip_pose(self.right_arm)
            pre_step_contacts = self.extract_object_contacts()
            left_verify_cost, verify_state["left"], left_contact_dist, left_has_contact = self._resolve_arm_verify_cost(
                verify_state["left"],
                left_tip_pos,
                left_contact_goal,
                pre_step_contacts.get("left", []),
            )
            right_verify_cost, verify_state["right"], right_contact_dist, right_has_contact = self._resolve_arm_verify_cost(
                verify_state["right"],
                right_tip_pos,
                right_contact_goal,
                pre_step_contacts.get("right", []),
            )
            tracking_goal_points_world = np.vstack(
                [
                    left_contact_goal if left_verify_cost >= 0.5 else live_targets["virtual_points_world"][0],
                    right_contact_goal if right_verify_cost >= 0.5 else live_targets["virtual_points_world"][1],
                ]
            )
            desired_force_world, desired_torque_world, execute_desired_wrench, desired_wrench_source = (
                self._resolve_live_wrench_targets(live_targets)
            )
            step_t3 = time.perf_counter()

            curr_x = self.get_planner_state()
            phi_vec, jac_mat, robot_contact_force_map, robot_contact_torque_map = self._detect_planner_contacts(
                return_wrench_maps=True
            )
            step_t4 = time.perf_counter()
            planner_result = self.planner.plan_once(
                object_pos,
                object_quat,
                curr_x,
                phi_vec,
                jac_mat,
                sol_guess=planner_sol_guess,
                verify_cost_param_1=left_verify_cost,
                verify_cost_param_2=right_verify_cost,
                virtual_point_1=live_targets["virtual_points_world"][0],
                virtual_point_2=live_targets["virtual_points_world"][1],
                contact_point_1=left_contact_goal,
                contact_point_2=right_contact_goal,
                desired_force_world=desired_force_world,
                desired_torque_world=desired_torque_world,
                robot_contact_force_map=robot_contact_force_map,
                robot_contact_torque_map=robot_contact_torque_map,
                execute_desired_wrench=execute_desired_wrench,
            )
            step_t5 = time.perf_counter()

            planner_sol_guess = planner_result["sol_guess"]
            self.plan_params.sol_guess_ = planner_sol_guess
            planner_backend = str(planner_result.get("solver_backend", self.plan_params.planner_solver_))
            planner_status = str(planner_result.get("solve_status", ""))
            action = np.asarray(planner_result["action"], dtype=np.float64).reshape(-1)
            if action.shape[0] != 6:
                raise RuntimeError(f"Expected a 6D dual-arm plan_once action, got shape {action.shape}.")

            self.step_cartesian_action(
                action[:3],
                action[3:6],
                left_goal_rot,
                right_goal_rot,
                sync_visualization=False,
            )
            step_t6 = time.perf_counter()

            self.set_marker("contact_point1", np.asarray(live_targets["contact_points_world"][0], dtype=np.float64))
            self.set_marker("contact_point2", np.asarray(live_targets["contact_points_world"][1], dtype=np.float64))
            self.set_marker("obj_point", object_pos)
            self.set_marker("goal", object_pos, object_quat)
            mujoco.mj_forward(self.model, self.data)
            contacts = self.extract_object_contacts()
            self._sync_visualization()
            step_t7 = time.perf_counter()
            object_pos_after, object_quat_after, _ = self.get_object_pose()

            left_err, left_rot_err = self._tip_target_error(self.left_arm, tracking_goal_points_world[0], left_goal_rot)
            right_err, right_rot_err = self._tip_target_error(self.right_arm, tracking_goal_points_world[1], right_goal_rot)
            left_force = self._max_normal_force(contacts["left"])
            right_force = self._max_normal_force(contacts["right"])
            contact_mode = self._classify_fingertip_contact_mode(contacts)

            force_ok = (
                left_force >= float(required_normal_force)
                and right_force >= float(required_normal_force)
            )
            pose_ok = (
                left_err < pos_tol
                and right_err < pos_tol
                and left_rot_err < rot_tol
                and right_rot_err < rot_tol
            )
            if contact_mode == "bilateral" and pose_ok and force_ok:
                stable_contact_steps += 1
            else:
                stable_contact_steps = 0

            timing = {
                "contact_plan_wall": step_t2 - step_t1,
                "target_build": step_t3 - step_t2,
                "planner_contacts": step_t4 - step_t3,
                "planner_solve": step_t5 - step_t4,
                "cartesian_step": step_t6 - step_t5,
                "post_update": step_t7 - step_t6,
                "wall_total": step_t7 - loop_t0,
            }
            for key, value in timing.items():
                timing_sum[key] += float(value)

            info = {
                "step": step,
                "contacts": contacts,
                "left_force": float(left_force),
                "right_force": float(right_force),
                "left_pos_err": float(left_err),
                "right_pos_err": float(right_err),
                "left_rot_err": float(left_rot_err),
                "right_rot_err": float(right_rot_err),
                "left_goal_pos": np.asarray(tracking_goal_points_world[0], dtype=np.float64).copy(),
                "right_goal_pos": np.asarray(tracking_goal_points_world[1], dtype=np.float64).copy(),
                "left_goal_rot": left_goal_rot.copy(),
                "right_goal_rot": right_goal_rot.copy(),
                "left_contact_goal": left_contact_goal.copy(),
                "right_contact_goal": right_contact_goal.copy(),
                "left_virtual_point": np.asarray(live_targets["virtual_points_world"][0], dtype=np.float64).copy(),
                "right_virtual_point": np.asarray(live_targets["virtual_points_world"][1], dtype=np.float64).copy(),
                "left_verify_cost": float(left_verify_cost),
                "right_verify_cost": float(right_verify_cost),
                "left_contact_distance": float(left_contact_dist),
                "right_contact_distance": float(right_contact_dist),
                "left_has_contact": bool(left_has_contact),
                "right_has_contact": bool(right_has_contact),
                "left_planner_cmd": action[:3].copy(),
                "right_planner_cmd": action[3:6].copy(),
                "planner_backend": planner_backend,
                "planner_status": planner_status,
                "object_pos": object_pos_after.copy(),
                "object_quat": object_quat_after.copy(),
                "contact_points_local": live_targets["contact_points_local"].copy(),
                "contact_points_world": live_targets["contact_points_world"].copy(),
                "best_contact_locations_world": live_targets["contact_points_world"].copy(),
                "normals_local": live_targets["normals_local"].copy(),
                "inward_normals_world": live_targets["inward_normals_world"].copy(),
                "outward_normals_world": live_targets["outward_normals_world"].copy(),
                "grasp_cost": float(live_targets["total_cost"]),
                "region_score": float(live_targets["region_score"]),
                "antipodal_margin": float(live_targets["antipodal_margin"]),
                "contact_targets": self._copy_contact_targets(live_targets),
                "desired_force_world": np.asarray(desired_force_world, dtype=np.float64).copy(),
                "desired_torque_world": np.asarray(desired_torque_world, dtype=np.float64).copy(),
                "desired_wrench_source": str(desired_wrench_source),
                "execute_desired_wrench": bool(execute_desired_wrench),
                "contact_solve_reported_time": float(live_targets.get("solve_time", float("nan"))),
                "contact_solve_reported_wall_time": float(live_targets.get("wall_time", float("nan"))),
                "stable_contact_steps": int(stable_contact_steps),
                "contact_mode": contact_mode,
                "timing": dict(timing),
            }

            loop_hz = 1.0 / max(timing["wall_total"], 1e-9)
            print(
                f"[timing:{label}] step={step:04d} "
                f"contact_plan_wall={timing['contact_plan_wall']:.4f}s "
                f"target_build={timing['target_build']:.4f}s "
                f"planner_contacts={timing['planner_contacts']:.4f}s "
                f"planner_solve={timing['planner_solve']:.4f}s "
                f"cartesian_step={timing['cartesian_step']:.4f}s "
                f"post_update={timing['post_update']:.4f}s "
                f"wall_total={timing['wall_total']:.4f}s "
                f"hz={loop_hz:.2f}"
            )

            if bool(self.args.verbose) and (step == 0 or step == max_steps - 1 or step - last_report_step >= 40):
                print(
                    f"[{label}] step={step:04d} "
                    f"left_err={left_err:.4f}/{left_rot_err:.4f} "
                    f"right_err={right_err:.4f}/{right_rot_err:.4f} "
                    f"left_force={left_force:.3f} right_force={right_force:.3f} "
                    f"verify=({left_verify_cost:.0f},{right_verify_cost:.0f}) "
                    f"contact_mode={contact_mode} planner={planner_backend} "
                    f"wrench_src={desired_wrench_source or 'none'} "
                    f"grasp_cost={info['grasp_cost']:.4f} "
                    f"hz={loop_hz:.2f}"
                )
                if step == 0:
                    print(
                        f"  loop_cfg: world_mode={world_mode} "
                        f"required_normal_force={float(required_normal_force):.3f} "
                        f"viewer={'on' if self.viewer is not None else 'off'} "
                        f"screenshot={'on' if self.screenshot_recorder is not None else 'off'}"
                    )
                if planner_status:
                    print(f"  planner_status: {planner_status}")
                last_report_step = step

            if stable_contact_steps >= int(self.args.contact_stable_steps):
                info["timing_avg"] = {
                    key: value / max(step + 1, 1)
                    for key, value in timing_sum.items()
                }
                return True, info

        completed_steps = int(info.get("step", -1)) + 1 if info else 0
        if completed_steps > 0:
            info["timing_avg"] = {
                key: value / max(completed_steps, 1)
                for key, value in timing_sum.items()
            }
        return False, info

    def extract_object_contacts(self):
        if self.object_is_ghost:
            return {"left": [], "right": []}
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
        waypoint_offset=None,
    ):
        rot_tol = self.args.ik_rot_tol if rot_tol is None else float(rot_tol)
        if int(max_steps) <= 0:
            return False, {}

        self._set_curobo_world_mode(world_mode)
        last_report_step = -1
        info = {}
        cached_targets = None
        cached_ik = None
        planner_sol_guess = None
        waypoint_offset = self.args.planner_attract_offset if waypoint_offset is None else float(waypoint_offset)
        waypoint_done = waypoint_offset <= 1e-9
        if fixed_joint_goals is None:
            left_joint_goal = None
            right_joint_goal = None
        else:
            left_joint_goal, right_joint_goal = fixed_joint_goals
        for step in range(max_steps):
            if not self.is_running():
                break

            self._update_inter_arm_worlds()
            left_curr_pos, left_curr_rot = self.get_tip_pose(self.left_arm)
            right_curr_pos, right_curr_rot = self.get_tip_pose(self.right_arm)
            if solve_ik_once:
                if cached_targets is None:
                    cached_targets = target_fn(step)
                    left_target = cached_targets["left_tip_pos"]
                    left_rot = cached_targets["left_tip_rot"]
                    right_target = cached_targets["right_tip_pos"]
                    right_rot = cached_targets["right_tip_rot"]
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
                else:
                    left_target = cached_targets["left_tip_pos"]
                    left_rot = cached_targets["left_tip_rot"]
                    right_target = cached_targets["right_tip_pos"]
                    right_rot = cached_targets["right_tip_rot"]
                    left_ik, right_ik = cached_ik
                stage_targets = cached_targets
            else:
                stage_targets = target_fn(step)
                left_target = stage_targets["left_tip_pos"]
                left_rot = stage_targets["left_tip_rot"]
                right_target = stage_targets["right_tip_pos"]
                right_rot = stage_targets["right_tip_rot"]
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

            left_attract = left_ik.solved_tip_pos_world + stage_targets["left_outward_normal"] * waypoint_offset
            right_attract = right_ik.solved_tip_pos_world + stage_targets["right_outward_normal"] * waypoint_offset

            if not waypoint_done:
                left_waypoint_dist = float(np.linalg.norm(self.get_tip_pos(self.left_arm) - left_attract))
                right_waypoint_dist = float(np.linalg.norm(self.get_tip_pos(self.right_arm) - right_attract))
                if (
                    left_waypoint_dist < float(self.args.planner_attract_tol)
                    and right_waypoint_dist < float(self.args.planner_attract_tol)
                ):
                    waypoint_done = True
                    planner_sol_guess = None

            waypoint_active = not waypoint_done
            left_step_target = left_attract if waypoint_active else left_ik.solved_tip_pos_world
            right_step_target = right_attract if waypoint_active else right_ik.solved_tip_pos_world
            left_step_rot = left_curr_rot if waypoint_active else left_ik.solved_tip_rot_world
            right_step_rot = right_curr_rot if waypoint_active else right_ik.solved_tip_rot_world

            curr_x = self.get_planner_state()
            phi_vec, jac_mat = self._detect_planner_contacts()
            object_target_pos = np.asarray(stage_targets.get("object_target_pos", curr_x[:3]), dtype=np.float64)
            object_target_quat = np.asarray(stage_targets.get("object_target_quat", curr_x[3:7]), dtype=np.float64)
            planner_result = self.planner.plan_once(
                object_target_pos,
                object_target_quat,
                curr_x,
                phi_vec,
                jac_mat,
                sol_guess=planner_sol_guess,
                verify_cost_param_1=0.0 if waypoint_active else 1.0,
                verify_cost_param_2=0.0 if waypoint_active else 1.0,
                virtual_point_1=left_attract,
                virtual_point_2=right_attract,
                contact_point_1=left_ik.solved_tip_pos_world,
                contact_point_2=right_ik.solved_tip_pos_world,
            )
            planner_sol_guess = planner_result["sol_guess"]
            self.plan_params.sol_guess_ = planner_sol_guess
            planner_backend = str(planner_result.get("solver_backend", self.plan_params.planner_solver_))
            planner_status = str(planner_result.get("solve_status", ""))
            action = np.asarray(planner_result["action"], dtype=np.float64).reshape(-1)
            if action.shape[0] != 6:
                raise RuntimeError(f"Expected a 6D dual-arm plan_once action, got shape {action.shape}.")
            self.step_cartesian_action(
                action[:3],
                action[3:6],
                left_step_rot,
                right_step_rot,
                sync_visualization=False,
            )

            self.set_marker("obj_point", object_target_pos)
            mujoco.mj_forward(self.model, self.data)
            self._sync_visualization()

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
            left_plan_pose_err, _ = self._tip_target_error(
                self.left_arm,
                left_step_target,
                left_ik.solved_tip_rot_world,
            )
            right_plan_pose_err, _ = self._tip_target_error(
                self.right_arm,
                right_step_target,
                right_ik.solved_tip_rot_world,
            )
            contacts = self.extract_object_contacts()
            info = {
                "step": step,
                "contacts": contacts,
                "left_force": self._max_normal_force(contacts["left"]),
                "right_force": self._max_normal_force(contacts["right"]),
                "waypoint_active": waypoint_active,
                "left_waypoint_pos_err": float(np.linalg.norm(self.get_tip_pos(self.left_arm) - left_attract)),
                "right_waypoint_pos_err": float(np.linalg.norm(self.get_tip_pos(self.right_arm) - right_attract)),
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
                "left_mpc_pose_err": float(left_plan_pose_err),
                "right_mpc_pose_err": float(right_plan_pose_err),
                "left_goal_q_mj": left_ik.q_mj.copy(),
                "right_goal_q_mj": right_ik.q_mj.copy(),
                "object_pos": self.get_object_pose()[0].copy(),
                "left_planner_cmd": action[:3].copy(),
                "right_planner_cmd": action[3:6].copy(),
                "planner_backend": planner_backend,
                "planner_status": planner_status,
            }
            if bool(self.args.verbose) and (step == 0 or step == max_steps - 1 or step - last_report_step >= 40):
                print(
                    f"[{label}] step={step:04d} "
                    f"left_err={left_err:.4f}/{left_rot_err:.4f} "
                    f"right_err={right_err:.4f}/{right_rot_err:.4f} "
                    f"left_force={info['left_force']:.3f} right_force={info['right_force']:.3f} "
                    f"left_ik={left_ik.success} right_ik={right_ik.success} "
                    f"left_ik_res={left_ik.position_error:.4f}/{left_ik.rotation_error:.4f} "
                    f"right_ik_res={right_ik.position_error:.4f}/{right_ik.rotation_error:.4f} "
                    f"left_mpc={left_plan_pose_err:.4f} right_mpc={right_plan_pose_err:.4f} "
                    f"planner={planner_backend}"
                )
                if step == 0:
                    print(
                        f"  stage_cfg: world_mode={world_mode} "
                        f"use_ik={use_ik} planner=plan_once[{planner_backend}] "
                        f"waypoint_offset={waypoint_offset:.4f}"
                    )
                if planner_status:
                    print(f"  planner_status: {planner_status}")
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
                if waypoint_active:
                    print(
                        "  waypoint_track: "
                        f"left={info['left_waypoint_pos_err']:.4f} "
                        f"right={info['right_waypoint_pos_err']:.4f}"
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

        optimizer_candidate_cache = self._build_precomputed_optimizer_contact_cache(obj_pos, obj_rot)

        contact_solve_t0 = time.perf_counter()
        initial_contact_targets = self._get_live_contact_targets(
            obj_pos,
            obj_rot,
            previous_targets=None,
            virtual_offset=float(self.args.planner_attract_offset),
            candidate_cache=optimizer_candidate_cache,
        )
        contact_solve_elapsed = time.perf_counter() - contact_solve_t0
        contact_points_local = initial_contact_targets["contact_points_local"]
        normals_local = initial_contact_targets["normals_local"]
        contact_points_world = initial_contact_targets["contact_points_world"]
        inward_normals_world = initial_contact_targets["inward_normals_world"]
        grasp_result = self.optimizer.last_grasp_result
        static_result = None
        if grasp_result is not None:
            static_result = self.optimizer.solve_static_equilibrium(grasp_result, gravity_local)

        self.set_marker("contact_point1", contact_points_world[0])
        self.set_marker("contact_point2", contact_points_world[1])
        self.set_marker("goal", obj_pos, obj_quat)
        mujoco.mj_forward(self.model, self.data)
        self._sync_visualization()

        contact_solve_time = float(initial_contact_targets.get("solve_time", float("nan")))
        if not np.isfinite(contact_solve_time):
            contact_solve_time = float(contact_solve_elapsed)
        print(
            f"Optimal contact solve time: {contact_solve_time:.4f}s "
            f"cost={float(initial_contact_targets['total_cost']):.6f} "
            f"region_score={float(initial_contact_targets['region_score']):.6f} "
            f"antipodal_margin={float(initial_contact_targets['antipodal_margin']):.6f}"
        )

        if bool(self.args.test):
            test_ok, test_info = self._run_optimizer_torque_test(initial_contact_targets)
            if not test_ok:
                print("Optimizer torque test did not produce a valid torque.")
                return
            if "applied_optimizer_torque_world" in test_info:
                print(
                    "Applied optimizer torque in test mode:",
                    np.array2string(np.asarray(test_info["applied_optimizer_torque_world"], dtype=np.float64), precision=4),
                )
            if "optimizer_torque_application_point_world" in test_info:
                print(
                    "Optimizer torque application point world in test mode:",
                    np.array2string(np.asarray(test_info["optimizer_torque_application_point_world"], dtype=np.float64), precision=4),
                )
            return

        required_normal_force = (
            0.35 * self.args.obj_mass * 9.81
            if self.args.min_normal_force is None
            else float(self.args.min_normal_force)
        )
        if static_result is not None and static_result["valid"]:
            modeled_normal = float(np.max(np.asarray(static_result["contact_forces_local"], dtype=np.float64)[:, 0]))
            required_normal_force = max(required_normal_force, 0.5 * modeled_normal)

        cached_contact_targets = self._copy_contact_targets(initial_contact_targets)
        contact_order = np.asarray(cached_contact_targets.get("contact_order", np.array([0, 1], dtype=int)), dtype=int).reshape(-1)

        desired_contact_forces_local = np.zeros((0, 3), dtype=np.float64)
        desired_force_vectors_local = np.zeros((0, 3), dtype=np.float64)
        if static_result is not None and static_result["valid"]:
            desired_contact_forces_local = np.asarray(static_result["contact_forces_local"], dtype=np.float64).reshape(-1, 3)
            desired_force_vectors_local = np.asarray(static_result["force_vectors_local"], dtype=np.float64).reshape(-1, 3)
            if desired_contact_forces_local.shape[0] == contact_order.shape[0] and contact_order.shape[0] == 2:
                desired_contact_forces_local = desired_contact_forces_local[contact_order]
            if desired_force_vectors_local.shape[0] == contact_order.shape[0] and contact_order.shape[0] == 2:
                desired_force_vectors_local = desired_force_vectors_local[contact_order]
        else:
            desired_contact_forces_local = np.asarray(
                cached_contact_targets.get("witness_contact_forces_local", np.zeros((0, 3), dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1, 3)
            desired_force_vectors_local = np.asarray(
                cached_contact_targets.get("witness_force_vectors_local", np.zeros((0, 3), dtype=np.float64)),
                dtype=np.float64,
            ).reshape(-1, 3)

        if desired_contact_forces_local.shape[0] == 2:
            cached_contact_targets["desired_contact_forces_local"] = desired_contact_forces_local.copy()
        if desired_force_vectors_local.shape[0] == 2:
            cached_contact_targets["desired_force_vectors_local"] = desired_force_vectors_local.copy()
        cached_contact_targets = self._project_cached_contact_targets(
            cached_contact_targets,
            obj_pos,
            obj_rot,
            virtual_offset=float(self.args.planner_attract_offset),
        )

        if bool(self.args.verbose):
            print("Generated scene:", self.scene_path)
            print("Mesh:", self.mesh_path)
            print("Scale:", self.mesh_scale)
            print("Object pose:", obj_pos, obj_quat)
            if optimizer_candidate_cache is not None:
                cache_timing = optimizer_candidate_cache.get("timing", {})
                print(
                    "Precomputed contact search cache: "
                    f"candidates={len(optimizer_candidate_cache.get('candidate_entries', []))} "
                    f"regions={len(optimizer_candidate_cache.get('region_groups', []))} "
                    f"build_time={float(cache_timing.get('wall_time', 0.0)):.4f}s"
                )
            if self.screenshot_recorder is not None:
                print(
                    f"Screenshots: interval={float(self.args.screenshot_interval):.2f}s "
                    f"dir={Path(self.args.screenshot_dir).resolve()}"
                )
            raw_contact_points_local = np.asarray(
                initial_contact_targets.get("raw_contact_points_local", contact_points_local),
                dtype=np.float64,
            ).reshape(-1, 3)
            print("Raw optimizer contact points local:\n", raw_contact_points_local)
            print("Arm-assigned contact points local:\n", contact_points_local)
            print("Contact points world:\n", contact_points_world)
            print("Inward normals world:\n", inward_normals_world)
            if desired_contact_forces_local.shape[0] == 2:
                print("Desired contact forces local:\n", desired_contact_forces_local)
            if desired_force_vectors_local.shape[0] == 2:
                print("Desired contact force vectors in object frame:\n", desired_force_vectors_local)
            print(f"Required normal force per fingertip: {required_normal_force:.3f} N")
            if static_result is not None:
                print(
                    f"Static equilibrium: valid={static_result['valid']} "
                    f"residual_norm={static_result['residual_norm']:.6f} "
                    f"solve_time={static_result['solve_time']:.4f}s"
                )

        control_step_budget = max(
            int(self.args.approach_steps)
            + int(self.args.touch_steps)
            + int(self.args.squeeze_steps)
            + max(int(self.args.squeeze_extra_steps), 0),
            1,
        )
        if bool(self.args.verbose):
            print(
                "Realtime bigrasp control: "
                "best_contact_locations -> plan_once -> step_cartesian_action"
            )
            print(
                f"Control step budget: {control_step_budget} "
                f"(approach_steps + touch_steps + squeeze_steps + squeeze_extra_steps)"
            )

        control_ok, control_info = self._run_realtime_bigrasp_loop(
            "realtime_bigrasp",
            control_step_budget,
            pos_tol=self.args.target_tol,
            rot_tol=self.args.ik_rot_tol,
            required_normal_force=required_normal_force,
            world_mode="with_pedestal",
            initial_contact_targets=cached_contact_targets,
            optimizer_candidate_cache=optimizer_candidate_cache,
        )

        final_obj_pos, final_obj_quat, _ = self.get_object_pose()
        print(
            "Realtime bigrasp result:",
            f"success={control_ok}",
            f"final_object_pos={np.array2string(final_obj_pos, precision=4)}",
            f"final_object_quat={np.array2string(final_obj_quat, precision=4)}",
        )
        if control_info:
            print(
                f"Final fingertip forces: left={float(control_info.get('left_force', 0.0)):.4f}N "
                f"right={float(control_info.get('right_force', 0.0)):.4f}N"
            )
            print(
                f"Final contact mode: {control_info.get('contact_mode', 'unknown')} "
                f"stable_steps={int(control_info.get('stable_contact_steps', 0))}"
            )
            if control_info.get("desired_wrench_source"):
                print(
                    "Planner wrench target:",
                    control_info["desired_wrench_source"],
                    "force=",
                    np.array2string(np.asarray(control_info["desired_force_world"], dtype=np.float64), precision=4),
                    "torque=",
                    np.array2string(np.asarray(control_info["desired_torque_world"], dtype=np.float64), precision=4),
                )
            if "timing_avg" in control_info:
                avg_timing = control_info["timing_avg"]
                print(
                    "Average loop timing:",
                    f"contact_plan_wall={avg_timing['contact_plan_wall']:.4f}s",
                    f"planner_contacts={avg_timing['planner_contacts']:.4f}s",
                    f"planner_solve={avg_timing['planner_solve']:.4f}s",
                    f"cartesian_step={avg_timing['cartesian_step']:.4f}s",
                    f"post_update={avg_timing['post_update']:.4f}s",
                    f"wall_total={avg_timing['wall_total']:.4f}s",
                )
            final_contacts = control_info.get("contacts", {"left": [], "right": []})
            print(
                "Measured contacts:",
                {key: len(value) for key, value in final_contacts.items()},
            )
        if not control_ok:
            print("Realtime bigrasp loop reached its step limit before stable bilateral contact was established.")

        if self.args.hold_steps > 0:
            self.hold_current_pose(self.args.hold_steps)


def build_argparser():
    parser = argparse.ArgumentParser(description="Dual Panda MuJoCo grasp demo driven by mlqp_point_v2, cuRobo IK, and plan_once tracking.")
    parser.add_argument("--obj", type=str, default="elephant", help="Object asset name in envs/assets/objects.")
    parser.add_argument("--mesh", type=str, default=None, help="Absolute or relative mesh path. Overrides --obj.")
    parser.add_argument("--scale", type=float, nargs=3, default=None, help="Mesh scale factors sx sy sz.")
    parser.add_argument("--obj-mass", type=float, default=0.01, help="Object mass used by the grasp scoring model and the MuJoCo dynamic object body.")
    parser.add_argument(
        "--arm-friction",
        type=float,
        default=5.0,
        help="Friction coefficient used by the MuJoCo/planner contact model and the Stage 4/5 tangential force estimate.",
    )
    parser.add_argument(
        "--optimizer-arm-friction",
        type=float,
        default=0.9,
        help="Friction coefficient passed to mlqp_point_v2 contact selection. Matches planning/mlqp_point_v2.py default.",
    )
    parser.add_argument("--object-friction", type=float, default=5.0, help="Sliding friction coefficient used by the MuJoCo object geom.")
    parser.add_argument("--contact-stiffness", type=float, default=12.5, help="Contact stiffness passed to mlqp_point_v2.")
    parser.add_argument(
        "--sample-num",
        type=int,
        default=70,
        help="Surface samples used by mlqp_point_v2. Matches planning/mlqp_point_v2.py default.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=("ipopt", "snopt", "acados"),
        default="acados",
        help="Optional solver used by mlqp_point_v2 for grasp scoring and static-equilibrium checks.",
    )
    parser.add_argument(
        "--optimizer-use-support-filter",
        action="store_true",
        help="Enable the old support-aware candidate filtering before calling mlqp_point_v2. Disabled by default to match standalone mlqp_point_v2.py behavior.",
    )
    parser.add_argument("--optimizer-curvature-neighbor-k", type=int, default=8, help="Neighbor count used to estimate local surface curvature in mlqp_point_v2.")
    parser.add_argument("--optimizer-max-region-mean-curvature", type=float, default=0.12, help="Reject candidate contact regions whose mean curvature score exceeds this value.")
    parser.add_argument("--optimizer-max-point-curvature", type=float, default=0.25, help="Reject candidate contact regions that contain points above this curvature score.")
    parser.add_argument("--optimizer-curvature-penalty-weight", type=float, default=1.5, help="Quality penalty weight applied to higher-curvature regions in mlqp_point_v2.")
    parser.add_argument("--pos-coef", type=float, default=1.0, help="Position coefficient for mlqp_point_v2.")
    parser.add_argument("--ori-coef", type=float, default=0.0005, help="Orientation coefficient for mlqp_point_v2.")
    parser.add_argument("--scene-center-x", type=float, default=0.58, help="Midpoint between the two Panda bases.")
    parser.add_argument("--robot-span", type=float, default=0.75, help="Distance between the two Panda bases.") # 間距
    parser.add_argument("--pedestal-pos", type=float, nargs=3, default=(0.58, 0.0, 0.06), help="Central pedestal position.")
    parser.add_argument("--pedestal-size", type=float, nargs=3, default=(0.05, 0.07, 0.06), help="Central pedestal half sizes.")
    parser.add_argument("--object-yaw", type=float, default=0.0, help="Initial object yaw in radians.")
    parser.add_argument("--object-z-offset", type=float, default=0.0, help="Extra object height above the pedestal.")
    parser.add_argument("--obj_init_height", type=float, default=0.0, help="Extra initial object height relative to the pedestal top when the viewer starts.")
    parser.add_argument("--initial-object-lift", type=float, default=0.2, help="Extra height added to the pedestal/support under the object.") # 臺子高度
    parser.add_argument(
        "--camera_free",
        type=_parse_camera_free_arg,
        default=[[-0.1395,  -0.5727,   0.72078], [0.22557, -0.29262,  0.57177]],
        help="Fix the viewer camera using '[[cam_x, cam_y, cam_z], [look_x, look_y, look_z]]' or 6 flat values.",
    )
    parser.add_argument(
        "--show",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        help="If true, disable object collision and gravity so the object is shown only visually.",
    )
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
    parser.add_argument("--approach-steps", type=int, default=1000, help="Simulation steps for the pregrasp stage.")
    parser.add_argument(
        "--touch-steps",
        type=int,
        default=320,
        help="Simulation steps for the Stage 3 approach from attract points to the projected contact points.",
    )
    parser.add_argument("--squeeze-steps", type=int, default=480, help="Simulation steps for the squeeze stage.")
    parser.add_argument("--squeeze-extra-steps", type=int, default=360, help="Extra squeeze steps automatically used if the first squeeze window is not enough.")
    parser.add_argument("--lift-steps", type=int, default=320, help="Simulation steps for the lift stage.")
    parser.add_argument("--hold-steps", type=int, default=0, help="Extra simulation steps after lifting.")
    parser.add_argument("--cartesian-step", type=float, default=0.01, help="Direct end-effector xyz delta limit used by the realtime attract-point tracker.")
    parser.add_argument(
        "--test",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool_arg,
        help="If true, keep both arms fixed at their current pose and only apply the optimizer torque to the object.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed stage-by-stage debug prints.")
    parser.add_argument(
        "--precompute-contact-search",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        help="Precompute get_best_regions and candidate contact combinations once, then reuse them online so each step mainly reevaluates _solve_force_closure.",
    )
    parser.add_argument(
        "--print-contact-timing",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool_arg,
        help="Print timings for _get_live_contact_targets(), get_best_regions(), and get_best_grasp().",
    )
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
    parser.add_argument("--mujoco-dt", type=float, default=0.01, help="MuJoCo simulation timestep in seconds.")
    parser.add_argument("--mj-steps-per-command", type=int, default=1, help="Number of MuJoCo steps executed after each MPC command.")
    parser.add_argument("--command-substeps", type=int, default=1, help="Multiplier used when matching cuRobo MPC dt to the MuJoCo control cadence.")
    parser.add_argument("--curobo-collision-activation-distance", type=float, default=0.06, help="Collision activation distance passed to cuRobo.")
    parser.add_argument("--disable-curobo-self-collision", action="store_true", help="Disable cuRobo self-collision checking.")
    parser.add_argument("--disable-curobo-cuda-graph", action="store_true", help="Disable cuRobo CUDA graph capture.")
    parser.add_argument("--pose-only-mpc", action="store_true", help="Legacy option kept for CLI compatibility; plan_once tracking is pose-based by default.")
    parser.add_argument("--planner-dt", type=float, default=0.01, help="Time step used by the plan_once object-motion model.")
    parser.add_argument("--planner-horizon", type=int, default=20, help="plan_once horizon length.")
    parser.add_argument(
        "--planner-solver",
        type=str,
        choices=("ipopt", "acados"),
        default="acados",
        help="Solver backend used by self.planner.plan_once. Use acados to reduce stage-2 solve latency when a local acados build is available.",
    )
    parser.add_argument("--planner-max-contacts", type=int, default=15, help="Maximum object contacts modeled by plan_once.")
    parser.add_argument("--planner-cmd-limit", type=float, default=0.05, help="Per-step Cartesian delta limit in meters for each arm.")
    parser.add_argument("--planner-attract-offset", type=float, default=0.025, help="Outward offset used to build Stage 2 attract_point_world from contact_points_world along the outward object normal.")
    parser.add_argument("--planner-attract-tol", type=float, default=0.03, help="Distance threshold for switching from attract points to the IK contact pose.")
    parser.add_argument("--stage2-up-tilt", type=float, default=0.25, help="Extra local pitch, in radians, used only for the Stage 2 desired fingertip rotations.")
    parser.add_argument("--planner-attract-coef", type=float, default=0.5, help="Attract cost coefficient for plan_once.")
    parser.add_argument("--planner-reject-coef", type=float, default=0.001, help="Reject cost coefficient for plan_once.")
    parser.add_argument("--planner-contact-coef", type=float, default=0.7, help="Contact cost coefficient for plan_once.")
    parser.add_argument("--planner-contact-cost-param", type=float, default=0.0, help="Blend factor inside the dual-arm contact cost.")
    parser.add_argument(
        "--planner-force-tracking-weight",
        type=float,
        default=1.0,
        help="Weight on the plan_once predicted contact force tracking term when a desired contact wrench is provided.",
    )
    parser.add_argument(
        "--planner-torque-tracking-weight",
        type=float,
        default=1.0,
        help="Weight on the plan_once predicted contact torque tracking term when a desired contact wrench is provided.",
    )
    parser.add_argument("--planner-reject-distance", type=float, default=0.005, help="Reject distance threshold used by plan_once.")
    parser.add_argument("--planner-object-inertia-pos", type=float, default=40.0, help="Translational object inertia weight used by plan_once.")
    parser.add_argument("--planner-object-inertia-rot", type=float, default=0.05, help="Rotational object inertia weight used by plan_once.")
    parser.add_argument("--planner-robot-stiffness", type=float, default=300.0, help="Cartesian point stiffness used by the plan_once robot model.")
    parser.add_argument("--mppi-samples", type=int, default=256, help="Number of sampled trajectories used by plan_once.")
    parser.add_argument("--mppi-iterations", type=int, default=4, help="Number of MPPI update iterations after warm start.")
    parser.add_argument("--mppi-init-iterations", type=int, default=8, help="Number of MPPI iterations used before a warm start exists.")
    parser.add_argument("--mppi-lambda", type=float, default=1.0, help="MPPI temperature.")
    parser.add_argument("--mppi-noise-sigma", type=float, default=0.005, help="Action noise sigma for MPPI.")
    parser.add_argument("--mppi-noise-decay", type=float, default=0.85, help="Per-iteration MPPI noise decay.")
    parser.add_argument("--mppi-elite-frac", type=float, default=0.1, help="Elite fraction used by MPPI weighting.")
    parser.add_argument("--mppi-device", type=str, default=None, help="Torch device used by MPPI, for example cpu or cuda:0.")
    parser.add_argument("--mppi-use-torch-compile", action="store_true", help="Enable torch.compile for the MPPI kernels when available.")
    parser.add_argument("--cartesian-stiffness-pos", type=float, default=500.0, help="Translational stiffness used by the Cartesian impedance controller.")
    parser.add_argument("--cartesian-stiffness-rot", type=float, default=50.0, help="Rotational stiffness used by the Cartesian impedance controller.")
    parser.add_argument("--nullspace-stiffness", type=float, default=10.0, help="Nullspace stiffness used by the Cartesian impedance controller.")
    parser.add_argument("--min-normal-force", type=float, default=None, help="Required normal force per fingertip before lifting. Defaults to a mass-based value.")
    parser.add_argument(
        "--force-control-stiffness",
        type=float,
        default=None,
        help="Normal stiffness used by the Stage 4/5 spring model. Defaults to required_force / squeeze_depth.",
    )
    parser.add_argument(
        "--force-control-dissipation-velocity",
        type=float,
        default=0.1,
        help="Normal dissipation velocity used by the Stage 4/5 spring model.",
    )
    parser.add_argument(
        "--force-control-stiction-velocity",
        type=float,
        default=0.05,
        help="Tangential velocity regularization used by the Stage 4/5 spring model.",
    )
    parser.add_argument(
        "--force-control-smoothing",
        type=float,
        default=0.0,
        help="Optional softplus smoothing used by the Stage 4/5 spring model.",
    )
    parser.add_argument(
        "--single-contact-pos-kp",
        type=float,
        default=0.0,
        help="Position hold gain used when only one fingertip is in contact during Stage 4/5.",
    )
    parser.add_argument(
        "--single-contact-ori-kp",
        type=float,
        default=0.1,
        help="Orientation hold gain used when only one fingertip is in contact during Stage 4/5.",
    )
    parser.add_argument(
        "--single-contact-lin-damping",
        type=float,
        default=0.0,
        help="Linear damping force gain applied to the object when only one fingertip is in contact.",
    )
    parser.add_argument(
        "--single-contact-ang-damping",
        type=float,
        default=0.01,
        help="Angular damping torque gain applied to the object when only one fingertip is in contact.",
    )
    parser.add_argument(
        "--single-contact-force-max",
        type=float,
        default=0.0,
        help="Norm cap for the unilateral-contact stabilizing force. Use 0 to disable force stabilization.",
    )
    parser.add_argument(
        "--single-contact-torque-max",
        type=float,
        default=0.03,
        help="Norm cap for the unilateral-contact stabilizing torque applied before bilateral contact is established.",
    )
    parser.add_argument("--contact-stable-steps", type=int, default=15, help="Number of consecutive squeeze steps that must satisfy the normal-force threshold.")
    parser.add_argument("--lift-success-margin", type=float, default=0.005, help="Allowed height error when deciding whether the lift succeeded.")
    parser.add_argument("--visualize", action="store_true", help="Launch the MuJoCo passive viewer.")
    parser.add_argument("--real-time", action="store_true", help="Sleep to approximate real-time playback when visualizing.")
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=DEFAULT_SCREENSHOT_DIR,
        help="Directory where 1 Hz SVG screenshots will be written.",
    )
    parser.add_argument(
        "--screenshot-interval",
        type=float,
        default=0.0,
        help="Seconds between automatic screenshots. Defaults to 0 to avoid hidden rendering overhead.",
    )
    parser.add_argument("--screenshot-width", type=int, default=1280, help="Screenshot render width.")
    parser.add_argument("--screenshot-height", type=int, default=960, help="Screenshot render height.")
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
