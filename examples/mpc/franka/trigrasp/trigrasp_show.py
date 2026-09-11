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

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import mujoco
import mujoco.viewer
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from planning.mlqp_point_v2 import LambdaContactControlOptimizer
from planning.screenshot import (
    PeriodicSVGScreenshotRecorder,
    build_free_camera_config_from_position,
)


OBJECT_ASSET_DIR = REPO_ROOT / "envs" / "assets" / "objects"
DEFAULT_SCENE_OUTPUT = REPO_ROOT / "envs" / "xmls" / "trigrasp.xml"
DEFAULT_SCREENSHOT_DIR = CURRENT_DIR / "figs"
_SPIDER_ALLEGRO_REL = Path("thirdparty/spider/spider/assets/robots/allegro/right.xml")
_local_spider_xml = REPO_ROOT / _SPIDER_ALLEGRO_REL
SPIDER_ALLEGRO_XML = _local_spider_xml if _local_spider_xml.is_file() else (REPO_ROOT.parent / _SPIDER_ALLEGRO_REL)
SPIDER_ALLEGRO_ASSET_DIR = SPIDER_ALLEGRO_XML.parent / "assets"
DEFAULT_MESH_SCALE_MULTIPLIER = 2.0

FINGER_NAMES = ("thumb", "index", "middle", "ring")
FINGERTIP_SITE_NAMES = tuple(f"right_{finger}_tip" for finger in FINGER_NAMES)
DEFAULT_SCALE_MAP = {
    "stanford_bunny2": np.array([1.5, 1.5, 1.5], dtype=np.float64),
    "rubber_duck": np.array([1.3, 1.4, 1.4], dtype=np.float64),
    "Wolf_Duck": np.array([0.002, 0.002, 0.002], dtype=np.float64),
}
FINGER_POSE_CANDIDATES = (
    np.array(
        [
            0.10,
            0.60,
            0.70,
            0.40,
            0.00,
            0.60,
            0.70,
            0.40,
            -0.10,
            0.60,
            0.70,
            0.40,
            0.90,
            0.50,
            0.40,
            0.50,
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            0.20,
            1.00,
            0.70,
            0.30,
            0.00,
            0.90,
            1.00,
            0.30,
            0.00,
            1.00,
            0.70,
            0.30,
            1.20,
            1.00,
            0.70,
            0.60,
        ],
        dtype=np.float64,
    ),
)


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


def normalize(vec, eps=1e-9):
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec)
    return vec / norm


def project_to_rotation_matrix(rotation_matrix):
    u, _, vh = np.linalg.svd(np.asarray(rotation_matrix, dtype=np.float64))
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


def quat_xyzw_to_wxyz(quat_xyzw):
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def quat_wxyz_to_mat(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_matrix()


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
    scale_multiplier = float(scale_multiplier)
    if scale_override is not None:
        return scale_multiplier * np.asarray(scale_override, dtype=np.float64).reshape(3)
    base_scale = DEFAULT_SCALE_MAP.get(mesh_path.stem, np.ones(3, dtype=np.float64)).copy()
    return scale_multiplier * base_scale


def patch_robot_asset_paths(asset_element, asset_dir):
    asset_element = copy.deepcopy(asset_element)
    for node in asset_element.iter():
        if node.tag == "mesh" and "file" in node.attrib:
            mesh_file = Path(node.attrib["file"])
            if not mesh_file.is_absolute():
                node.attrib["file"] = str((asset_dir / mesh_file).resolve())
    return asset_element


def best_fit_rigid_transform(source_points, target_points):
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    covariance = (source_points - source_center).T @ (target_points - target_center)
    u, _, vh = np.linalg.svd(covariance)
    rotation_matrix = vh.T @ u.T
    if np.linalg.det(rotation_matrix) < 0.0:
        vh[-1, :] *= -1.0
        rotation_matrix = vh.T @ u.T
    translation = target_center - rotation_matrix @ source_center
    return project_to_rotation_matrix(rotation_matrix), translation


def project_to_plane(vec, normal):
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    normal = normalize(normal)
    return vec - np.dot(vec, normal) * normal


def build_trigrasp_scene_xml(
    mesh_path,
    mesh_scale,
    object_pos,
    object_quat,
    scene_output_path=DEFAULT_SCENE_OUTPUT,
    floor_z=-0.12,
    mujoco_timestep=0.01,
):
    robot_root = ET.parse(SPIDER_ALLEGRO_XML).getroot()
    robot_body = robot_root.find("./worldbody/body[@name='right_palm']")
    robot_asset = robot_root.find("asset")
    robot_default = robot_root.find("default")
    robot_actuator = robot_root.find("actuator")

    if robot_body is None or robot_asset is None or robot_default is None or robot_actuator is None:
        raise RuntimeError(f"Unexpected Allegro XML layout in {SPIDER_ALLEGRO_XML}")

    root = ET.Element("mujoco", {"model": "allegro_trigrasp"})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": f"{float(mujoco_timestep):.8f}",
            "iterations": "20",
            "ls_iterations": "50",
        },
    )
    ET.SubElement(root, "statistic", {"center": format_vec(object_pos), "extent": "0.55"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.7 0.7 0.7", "ambient": "0.25 0.25 0.25", "specular": "0.05 0.05 0.05"},
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
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "scene_groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "scene_groundplane",
            "texture": "scene_groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )
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
            "name": "object_mesh",
            "file": str(Path(mesh_path).resolve()),
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
    ET.SubElement(worldbody, "light", {"pos": "0.0 -0.5 1.2", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": "0.32 -0.55 0.42",
            "xyaxes": "0.88 0.48 0.00 -0.19 0.34 0.92",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {"name": "floor", "type": "plane", "pos": f"0 0 {float(floor_z):.8f}", "size": "0 0 0.05", "material": "scene_groundplane"},
    )

    obj_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "obj", "pos": format_vec(object_pos), "quat": format_vec(object_quat)},
    )
    ET.SubElement(
        obj_body,
        "geom",
        {
            "name": "obj_geom",
            "type": "mesh",
            "mesh": "object_mesh",
            "material": "object_visual_mat",
            "contype": "0",
            "conaffinity": "0",
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

    worldbody.append(copy.deepcopy(robot_body))

    actuator = ET.SubElement(root, "actuator")
    for child in list(robot_actuator):
        actuator.append(copy.deepcopy(child))

    indent_xml(root)
    tree = ET.ElementTree(root)
    scene_output_path = Path(scene_output_path)
    scene_output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(scene_output_path, encoding="utf-8", xml_declaration=False)
    return scene_output_path


class AllegroTrigraspDemo:
    def __init__(self, args):
        if int(args.num_grasp_contacts) != 4:
            raise ValueError("This demo currently expects --num-grasp-contacts=4 so the four Allegro fingertips can be assigned.")

        self.args = args
        self.mesh_path = resolve_mesh_path(args.obj, args.mesh)
        self.mesh_scale = resolve_mesh_scale(self.mesh_path, args.scale, args.scale_multiplier)
        self.mesh_bounds = load_mesh_bounds(self.mesh_path, self.mesh_scale)
        z_min = float(self.mesh_bounds[0, 2])
        z_max = float(self.mesh_bounds[1, 2])
        self.local_top_half_z_threshold = z_min + (2.0 / 3.0) * (z_max - z_min)
        self.object_pos = np.asarray(args.object_pos, dtype=np.float64).reshape(3)
        self.object_quat = quat_from_yaw(args.object_yaw)
        self.object_rot = quat_wxyz_to_mat(self.object_quat)

        self.scene_path = build_trigrasp_scene_xml(
            mesh_path=self.mesh_path,
            mesh_scale=self.mesh_scale,
            object_pos=self.object_pos,
            object_quat=self.object_quat,
            scene_output_path=args.scene_output,
            floor_z=args.floor_z,
            mujoco_timestep=args.mujoco_dt,
        )

        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = float(args.mujoco_dt)
        self.data = mujoco.MjData(self.model)
        self.solve_data = mujoco.MjData(self.model)
        self.viewer = None
        self.scene_camera_config = None
        self.screenshot_recorder = None

        self.palm_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_palm")
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj")
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
        self.actuator_qpos_adr = np.array(
            [self.model.jnt_qposadr[self.model.actuator_trnid[actuator_id, 0]] for actuator_id in range(self.model.nu)],
            dtype=np.int32,
        )
        self.joint_qpos_adr = np.array([self.model.jnt_qposadr[joint_id] for joint_id in range(self.model.njnt)], dtype=np.int32)
        self.joint_ranges = np.asarray(self.model.jnt_range, dtype=np.float64).copy()

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
        )

        self.initial_qpos = np.zeros(self.model.nq, dtype=np.float64)
        self.initial_qpos[:3] = np.array([0.0, -0.22, 0.02], dtype=np.float64)
        self.initial_qpos[3:6] = np.array([np.pi, 1.9, -1.2], dtype=np.float64)
        self.initial_qpos[6:] = FINGER_POSE_CANDIDATES[0].copy()
        self.set_hand_qpos(self.initial_qpos)
        self.set_object_pose(self.object_pos, self.object_quat)
        self.sync_viewer()

    def close(self):
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
        lookat = self.object_pos.copy() if lookat is None else np.asarray(lookat, dtype=np.float64).reshape(3)
        lookat = lookat + np.array([0.0, 0.0, 0.02], dtype=np.float64)
        # Tune this offset to move the shared viewer/screenshot camera closer or farther.
        camera_position = lookat + np.array([-0.52, 0.40, 0.20], dtype=np.float64)
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

    def _ensure_screenshot_recorder(self):
        if self.screenshot_recorder is not None or float(self.args.screenshot_interval) <= 0.0:
            return
        try:
            self.screenshot_recorder = self._build_screenshot_recorder()
        except Exception as exc:
            print(f"Warning: failed to initialize SVG screenshot recorder: {exc}")
            self.screenshot_recorder = None

    def sync_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()
        if self.screenshot_recorder is not None:
            self.screenshot_recorder.capture_if_due(self.data)

    def set_object_pose(self, object_pos, object_quat):
        self.object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3).copy()
        self.object_quat = np.asarray(object_quat, dtype=np.float64).reshape(4).copy()
        self.object_quat /= max(np.linalg.norm(self.object_quat), 1e-9)
        self.object_rot = quat_wxyz_to_mat(self.object_quat)
        self.model.body_pos[self.object_body_id] = self.object_pos
        self.model.body_quat[self.object_body_id] = self.object_quat
        mujoco.mj_forward(self.model, self.data)

    def set_marker_pose(self, marker_name, pos, quat=None):
        body_id = self.marker_body_ids[marker_name]
        self.model.body_pos[body_id] = np.asarray(pos, dtype=np.float64).reshape(3)
        if quat is not None:
            quat = np.asarray(quat, dtype=np.float64).reshape(4)
            quat /= max(np.linalg.norm(quat), 1e-9)
            self.model.body_quat[body_id] = quat

    def set_hand_qpos(self, qpos):
        qpos = np.asarray(qpos, dtype=np.float64).reshape(self.model.nq)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        if self.data.ctrl.shape[0] == self.actuator_qpos_adr.shape[0]:
            self.data.ctrl[:] = qpos[self.actuator_qpos_adr]
        mujoco.mj_forward(self.model, self.data)

    def forward_kinematics(self, qpos, data=None):
        data = self.solve_data if data is None else data
        qpos = np.asarray(qpos, dtype=np.float64).reshape(self.model.nq)
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)
        palm_pos = np.asarray(data.body(self.palm_body_id).xpos, dtype=np.float64).copy()
        palm_rot = np.asarray(data.body(self.palm_body_id).xmat, dtype=np.float64).reshape(3, 3).copy()
        tip_positions = np.stack([np.asarray(data.site(site_id).xpos, dtype=np.float64).copy() for site_id in self.tip_site_ids], axis=0)
        return palm_pos, palm_rot, tip_positions

    def local_tip_positions(self, qpos):
        palm_pos, palm_rot, tip_positions = self.forward_kinematics(qpos)
        return (palm_rot.T @ (tip_positions - palm_pos[None, :]).T).T

    def build_world_contact_targets(self):
        visible_idx = np.where(
            np.asarray(self.optimizer.sample_point[:, 2], dtype=np.float64) >= self.local_top_half_z_threshold
        )[0]
        if visible_idx.size == 0:
            visible_idx = np.arange(self.optimizer.sample_num, dtype=int)
        grasp_result = self.optimizer.get_best_grasp(visible_face_idx=visible_idx)
        if grasp_result is None:
            raise RuntimeError("mlqp_point_v2 failed to produce a 4-contact grasp candidate.")

        contact_points_local = np.asarray(grasp_result["contact_points"], dtype=np.float64).reshape(-1, 3)
        contact_normals_local = np.asarray(grasp_result["contact_normals"], dtype=np.float64).reshape(-1, 3)
        contact_points_world = (self.object_rot @ contact_points_local.T).T + self.object_pos[None, :]
        contact_normals_world = (self.object_rot @ contact_normals_local.T).T
        fingertip_targets_world = contact_points_world + np.array(
            [0.0, 0.0, float(self.args.top_down_tip_lift)],
            dtype=np.float64,
        )
        return grasp_result, contact_points_world, contact_normals_world, fingertip_targets_world

    def build_top_down_palm_rotation(self, nominal_local_tip_positions, assigned_targets_world):
        nominal_local_tip_positions = np.asarray(nominal_local_tip_positions, dtype=np.float64).reshape(-1, 3)
        assigned_targets_world = np.asarray(assigned_targets_world, dtype=np.float64).reshape(-1, 3)

        local_centroid = np.mean(nominal_local_tip_positions, axis=0)
        world_centroid = np.mean(assigned_targets_world, axis=0)
        centered_local = nominal_local_tip_positions - local_centroid[None, :]
        _, _, vh = np.linalg.svd(centered_local, full_matrices=False)

        # Allegro's palm-facing axis is not the palm body's local z-axis in this XML.
        # Estimate it directly from the fingertip plane so "top down" means the actual
        # palm normal points toward world -z.
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
        q = np.asarray(q_init, dtype=np.float64).reshape(self.model.nq).copy()
        q_nominal = q.copy()
        best_payload = None
        best_score = np.inf
        identity_nv = np.eye(self.model.nv, dtype=np.float64)
        regularization_weights = np.concatenate(
            [
                0.02 * np.ones(6, dtype=np.float64),
                0.04 * np.ones(self.model.nv - 6, dtype=np.float64),
            ]
        )

        for _ in range(int(self.args.ik_max_iters)):
            palm_pos, palm_rot, tip_positions = self.forward_kinematics(q)
            error_terms = []
            jacobian_terms = []

            palm_jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            palm_jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacBody(self.model, self.solve_data, palm_jacp, palm_jacr, self.palm_body_id)
            error_terms.append(float(self.args.ik_palm_pos_weight) * (target_palm_pos - palm_pos))
            jacobian_terms.append(float(self.args.ik_palm_pos_weight) * palm_jacp)
            error_terms.append(float(self.args.ik_palm_rot_weight) * rotation_error(palm_rot, target_palm_rot))
            jacobian_terms.append(float(self.args.ik_palm_rot_weight) * palm_jacr)

            for site_id, target_tip in zip(self.tip_site_ids, tip_targets_world):
                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, self.solve_data, jacp, jacr, site_id)
                error_terms.append(float(self.args.ik_tip_weight) * (target_tip - self.solve_data.site(site_id).xpos))
                jacobian_terms.append(float(self.args.ik_tip_weight) * jacp)

            regularization_error = regularization_weights * (q_nominal - q)
            error_terms.append(regularization_error)
            jacobian_terms.append(np.diag(regularization_weights))

            error_vector = np.concatenate(error_terms)
            jacobian = np.vstack(jacobian_terms)
            lhs = jacobian.T @ jacobian + float(self.args.ik_damping) * identity_nv
            rhs = jacobian.T @ error_vector
            dq = np.linalg.solve(lhs, rhs)
            q += float(self.args.ik_step_size) * dq

            for joint_id, qpos_adr in enumerate(self.joint_qpos_adr):
                q[qpos_adr] = np.clip(q[qpos_adr], self.joint_ranges[joint_id, 0], self.joint_ranges[joint_id, 1])

            palm_pos, palm_rot, tip_positions = self.forward_kinematics(q)
            tip_error_norms = np.linalg.norm(tip_positions - tip_targets_world, axis=1)
            mean_tip_error = float(np.mean(tip_error_norms))
            max_tip_error = float(np.max(tip_error_norms))
            palm_pos_error = float(np.linalg.norm(target_palm_pos - palm_pos))
            palm_rot_error = float(np.linalg.norm(rotation_error(palm_rot, target_palm_rot)))
            score = (
                mean_tip_error
                + 0.35 * max_tip_error
                + 0.20 * palm_pos_error
                + 0.05 * palm_rot_error
            )
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

        for seed_fingers in FINGER_POSE_CANDIDATES:
            q_seed = np.zeros(self.model.nq, dtype=np.float64)
            q_seed[6:] = seed_fingers.copy()
            nominal_local_tip_positions = self.local_tip_positions(q_seed)
            nominal_local_tip_centroid = np.mean(nominal_local_tip_positions, axis=0)

            for permutation in itertools.permutations(range(len(FINGER_NAMES))):
                assigned_targets_world = fingertip_targets_world[list(permutation)]
                target_palm_rot_world = self.build_top_down_palm_rotation(nominal_local_tip_positions, assigned_targets_world)
                target_palm_pos_world = np.mean(assigned_targets_world, axis=0) - target_palm_rot_world @ nominal_local_tip_centroid

                target_palm_euler = Rotation.from_matrix(target_palm_rot_world).as_euler("xyz")
                q_init = np.zeros(self.model.nq, dtype=np.float64)
                q_init[:3] = target_palm_pos_world
                q_init[3:6] = target_palm_euler
                q_init[6:] = seed_fingers.copy()

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

    def update_markers(self, contact_points_world, ik_result):
        for idx, point in enumerate(contact_points_world, start=1):
            self.set_marker_pose(f"contact_point{idx}", point)
        for finger_name, point in zip(FINGER_NAMES, ik_result.assigned_targets_world):
            self.set_marker_pose(f"{finger_name}_target", point)
        palm_target_quat = quat_xyzw_to_wxyz(Rotation.from_matrix(ik_result.target_palm_rot_world).as_quat())
        self.set_marker_pose("palm_target", ik_result.target_palm_pos_world, palm_target_quat)
        mujoco.mj_forward(self.model, self.data)

    def run(self):
        self.maybe_launch_viewer()
        self._ensure_screenshot_recorder()
        grasp_result, contact_points_world, contact_normals_world, fingertip_targets_world = self.build_world_contact_targets()
        ik_result = self.solve_grasp_ik(fingertip_targets_world)
        self.set_hand_qpos(ik_result.qpos)
        self.update_markers(contact_points_world, ik_result)
        if self.screenshot_recorder is not None:
            self.screenshot_recorder.capture(self.data, label="grasp_pose")
        self.sync_viewer()

        print("\nScene")
        print(f"scene_xml: {self.scene_path}")
        print(f"mesh_path: {self.mesh_path}")
        print(f"mesh_scale: {np.array2string(self.mesh_scale, precision=4)}")
        print(f"object_pos_world: {np.array2string(self.object_pos, precision=4)}")
        print(f"object_quat_wxyz: {np.array2string(self.object_quat, precision=4)}")

        print("\nGrasp Result")
        print(f"num_grasp_contacts: {grasp_result['contact_indices'].shape[0]}")
        print(f"contact_indices: {grasp_result['contact_indices']}")
        print(f"contact_points_local:\n{np.array2string(grasp_result['contact_points'], precision=4)}")
        print(f"contact_normals_local:\n{np.array2string(grasp_result['contact_normals'], precision=4)}")
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
        print(f"solved_qpos:\n{np.array2string(ik_result.qpos, precision=5)}")
        print(f"solved_tip_positions_world:\n{np.array2string(ik_result.solved_tip_positions_world, precision=4)}")

        if self.viewer is not None and not bool(self.args.no_wait):
            while self.viewer.is_running():
                mujoco.mj_step(self.model, self.data, nstep=max(1, int(round(0.02 / max(self.model.opt.timestep, 1e-9)))))
                self.sync_viewer()
                time.sleep(0.02)

        return grasp_result, ik_result


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Single-hand Allegro trigrasp demo: generate 4-contact grasp points with mlqp_point_v2 and solve Allegro IK in MuJoCo."
    )
    parser.add_argument("--obj", type=str, default="stanford_bunny2", help="Object asset name in envs/assets/objects.")
    parser.add_argument("--mesh", type=str, default=None, help="Optional absolute mesh path. Overrides --obj.")
    parser.add_argument("--scene-output", type=Path, default=DEFAULT_SCENE_OUTPUT, help="Path of the generated MuJoCo XML scene.")
    parser.add_argument("--scale", type=float, nargs=3, default=None, help="Optional mesh scale, e.g. --scale 1 1 1.")
    parser.add_argument("--scale-multiplier", type=float, default=DEFAULT_MESH_SCALE_MULTIPLIER, help="Global multiplier applied to the final mesh scale. Default is 1.5.")
    parser.add_argument("--object-pos", type=float, nargs=3, default=(0.0, 0.0, 0.18), help="World position of the object mesh.")
    parser.add_argument("--object-yaw", type=float, default=0.0, help="Object yaw in radians.")
    parser.add_argument("--floor-z", type=float, default=-0.12, help="Scene floor height.")
    parser.add_argument("--mujoco-dt", type=float, default=0.01, help="MuJoCo timestep.")
    parser.add_argument("--obj-mass", type=float, default=0.01, help="Object mass passed to the grasp scorer.")
    parser.add_argument("--contact-stiffness", type=float, default=12.5, help="Contact stiffness passed to mlqp_point_v2.")
    parser.add_argument("--sample-num", type=int, default=70, help="Surface sample count used by mlqp_point_v2.")
    parser.add_argument(
        "--solver",
        type=str,
        choices=("ipopt", "snopt", "acados"),
        default="ipopt",
        help="Optional solver used by mlqp_point_v2 for grasp scoring and static-equilibrium checks.",
    )
    parser.add_argument("--num-grasp-contacts", type=int, default=4, help="Must stay at 4 for Allegro thumb/index/middle/ring.")
    parser.add_argument("--region-contact-samples", type=int, default=5, help="Region sample count passed to mlqp_point_v2.")
    parser.add_argument("--top-region-pairs", type=int, default=3, help="Number of top region groups evaluated by mlqp_point_v2.")
    parser.add_argument("--preselect-region-pairs", type=int, default=200, help="Preselected region group budget passed to mlqp_point_v2.")
    parser.add_argument("--max-point-combination-eval", type=int, default=256, help="Max contact combinations evaluated inside mlqp_point_v2.")
    parser.add_argument("--tip-offset", type=float, default=0.008, help="Offset from the object surface along outward normal for fingertip IK targets.")
    parser.add_argument("--top-down-tip-lift", type=float, default=0.012, help="For top-down IK, lift fingertip targets upward in world z by this amount instead of offsetting along surface normals.")
    parser.add_argument("--ik-max-iters", type=int, default=220, help="Maximum iterations per Allegro IK candidate.")
    parser.add_argument("--ik-step-size", type=float, default=0.35, help="Damped least-squares IK step size.")
    parser.add_argument("--ik-damping", type=float, default=2e-4, help="Damped least-squares regularization.")
    parser.add_argument("--ik-tip-weight", type=float, default=4.0, help="Weight on fingertip position residuals.")
    parser.add_argument("--ik-palm-pos-weight", type=float, default=1.2, help="Weight on palm position residual.")
    parser.add_argument("--ik-palm-rot-weight", type=float, default=0.6, help="Weight on palm rotation residual.")
    parser.add_argument("--ik-tip-tol", type=float, default=0.006, help="Tip position error threshold used to mark IK success.")
    parser.add_argument("--screenshot-dir", type=Path, default=DEFAULT_SCREENSHOT_DIR, help="Directory where SVG screenshots will be written.")
    parser.add_argument("--screenshot-interval", type=float, default=1.0, help="Seconds between automatic SVG screenshots. Use 0 to disable.")
    parser.add_argument("--screenshot-width", type=int, default=1280, help="Screenshot render width.")
    parser.add_argument("--screenshot-height", type=int, default=960, help="Screenshot render height.")
    parser.add_argument("--visualize", dest="visualize", action="store_true", help="Open a passive MuJoCo viewer after setting the solved grasp pose.")
    parser.add_argument("--headless", dest="visualize", action="store_false", help="Run without opening the MuJoCo viewer.")
    parser.add_argument("--no-wait", action="store_true", help="With --visualize, exit immediately after updating the viewer once.")
    parser.set_defaults(visualize=True)
    return parser


def main():
    args = build_argparser().parse_args()
    demo = AllegroTrigraspDemo(args)
    try:
        demo.run()
    finally:
        demo.close()


if __name__ == "__main__":
    main()
