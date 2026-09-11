import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from isaacgym import gymapi, gymtorch

DEFAULT_CARTESIAN_STIFFNESS = np.array([500.0, 500.0, 500.0, 50.0, 50.0, 50.0], dtype=np.float32)
POSE_AXIS_LENGTH = 0.08
POSE_AXIS_RADIUS = 0.003
POSE_AXIS_CENTER_RADIUS = 0.009
HIDDEN_GHOST_POSITION = np.array([0.0, 0.0, -10.0], dtype=np.float32)


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
# Put the repository ahead of third-party packages named ``examples``.  Isaac
# Gym/PyTorch may import such a package before this script reaches this point.
sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir) != parent_dir]
sys.path.insert(0, parent_dir)
loaded_examples = sys.modules.get("examples")
loaded_examples_file = getattr(loaded_examples, "__file__", "") if loaded_examples else ""
if loaded_examples and not os.path.abspath(loaded_examples_file).startswith(parent_dir):
    for module_name in list(sys.modules):
        if module_name == "examples" or module_name.startswith("examples."):
            del sys.modules[module_name]

from examples.mpc.franka.ik2.params import ExplicitMPCParams
from planning.MPPIExplicit import (
    MPPIExplicit,
    _franka_fk_T_jax,
    _franka_jacobian_pos_jax,
    _tangent_basis_from_normal,
    _contact_jacobian,
)
from planning.mpc_implicit import MPCImplicit
from planning.mlqp_point_v1_ip import LambdaContactControlOptimizer
from utils import metrics, rotations


def _extract_vec3(v) -> np.ndarray:
    if isinstance(v, np.void) and getattr(v, "dtype", None) is not None and v.dtype.names is not None:
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=np.float32)
    arr = np.asarray(v)
    if arr.dtype.names is not None:
        return np.array([float(arr["x"]), float(arr["y"]), float(arr["z"])], dtype=np.float32)
    arr = arr.reshape(-1)
    return np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=np.float32)


def _extract_quat_xyzw(q) -> np.ndarray:
    if isinstance(q, np.void) and getattr(q, "dtype", None) is not None and q.dtype.names is not None:
        return np.array([float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])], dtype=np.float32)
    arr = np.asarray(q)
    if arr.dtype.names is not None:
        return np.array([float(arr["x"]), float(arr["y"]), float(arr["z"]), float(arr["w"])], dtype=np.float32)
    arr = arr.reshape(-1)
    return np.array([float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])], dtype=np.float32)


def _parse_bool_arg(value):
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in ("true", "1", "yes", "y", "on"):
        return True
    if value_str in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got: {value}")


class IsaacFrankaSimulator:
    """Isaac Gym version of the MuJoCo MjSimulator interface used in test_mppi.py."""

    def __init__(self, param, headless=False, sim_device="cuda:0", graphics_device_id=0):
        self.param_ = param
        self.break_out_signal_ = False
        self.dyn_paused_ = False
        self.viewer_ = None
        self.show_ghost_object_ = bool(getattr(self.param_, "show_ghost_object_", False))

        self.gym = gymapi.acquire_gym()

        compute_id = int(sim_device.split(":")[-1]) if ":" in sim_device else 0
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.01
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # This script uses CPU-style state APIs (get/set_actor_rigid_body_states),
        # so disable GPU pipeline to avoid invalid resource handle errors.
        sim_params.use_gpu_pipeline = False
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.01
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = (compute_id >= 0)

        self.sim = self.gym.create_sim(compute_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")

        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane)

        env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
        env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)

        self._create_scene_actors()
        self._configure_franka()
        self.gym.prepare_sim(self.sim)
        self._build_body_index_cache()
        self.reset_mj_env()

        if not headless:
            self.viewer_ = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer_ is not None:
                cam_pos = gymapi.Vec3(1.4, 0.8, 1.0)
                cam_target = gymapi.Vec3(0.5, 0.0, 0.35)
                self.gym.viewer_camera_look_at(self.viewer_, self.env, cam_pos, cam_target)

        self.R_d = np.eye(3)
        self.cartesian_stiffness = np.diag(DEFAULT_CARTESIAN_STIFFNESS)
        self.cartesian_damping = (2.0 * np.sqrt(self.cartesian_stiffness)).astype(np.float32)

    def _create_scene_actors(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
        franka_asset_root, franka_asset_file = self._get_franka_asset_info(repo_root)

        franka_opts = gymapi.AssetOptions()
        franka_opts.fix_base_link = True
        franka_opts.disable_gravity = True
        franka_opts.collapse_fixed_joints = False
        franka_opts.flip_visual_attachments = True
        franka_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        self.franka_asset = self.gym.load_asset(self.sim, franka_asset_root, franka_asset_file, franka_opts)

        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, 2.0, 2.0, 0.35, table_opts)

        mesh_path = getattr(self.param_, "mesh_path_", None)
        print("mesh_path = ", mesh_path)
        mesh_obj_urdf_rel, mesh_target_urdf_rel, mesh_asset_root = self._prepare_mesh_urdf_assets(
            repo_root, mesh_path
        )
        obj_opts = gymapi.AssetOptions()
        obj_opts.density = 200.0
        obj_opts.use_mesh_materials = True
        self.obj_asset = self.gym.load_asset(self.sim, mesh_asset_root, mesh_obj_urdf_rel, obj_opts)

        target_obj_opts = gymapi.AssetOptions()
        target_obj_opts.fix_base_link = True
        target_obj_opts.disable_gravity = True
        target_obj_opts.use_mesh_materials = True
        self.target_obj_asset = self.gym.load_asset(self.sim, mesh_asset_root, mesh_target_urdf_rel, target_obj_opts)

        marker_urdf_rel, marker_asset_root = self._prepare_marker_urdf_asset(repo_root)
        marker_opts = gymapi.AssetOptions()
        marker_opts.fix_base_link = True
        marker_opts.disable_gravity = True
        self.marker_asset = self.gym.load_asset(self.sim, marker_asset_root, marker_urdf_rel, marker_opts)

        pose_axes_urdf_rel, pose_axes_asset_root = self._prepare_pose_axes_urdf_asset(
            repo_root, translucent=False
        )
        pose_axes_opts = gymapi.AssetOptions()
        pose_axes_opts.fix_base_link = True
        pose_axes_opts.disable_gravity = True
        self.pose_axes_asset = self.gym.load_asset(
            self.sim, pose_axes_asset_root, pose_axes_urdf_rel, pose_axes_opts
        )

        ghost_pose_axes_urdf_rel, ghost_pose_axes_asset_root = self._prepare_pose_axes_urdf_asset(
            repo_root, translucent=True
        )
        ghost_pose_axes_opts = gymapi.AssetOptions()
        ghost_pose_axes_opts.fix_base_link = True
        ghost_pose_axes_opts.disable_gravity = True
        self.ghost_pose_axes_asset = self.gym.load_asset(
            self.sim, ghost_pose_axes_asset_root, ghost_pose_axes_urdf_rel, ghost_pose_axes_opts
        )

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        franka_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(1.2, 0.0, 0.175)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0.45, 0.0, 0.375)
        obj_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        target_obj_pose = gymapi.Transform()
        target_obj_pose.p = gymapi.Vec3(
            float(self.param_.target_p_[0]),
            float(self.param_.target_p_[1]),
            float(self.param_.target_p_[2]),
        )
        target_obj_pose.r = gymapi.Quat(
            float(self.param_.target_q_[1]),
            float(self.param_.target_q_[2]),
            float(self.param_.target_q_[3]),
            float(self.param_.target_q_[0]),
        )
        self._target_obj_pos_cache = np.array(self.param_.target_p_, dtype=np.float32).copy()
        self._target_obj_quat_xyzw_cache = np.array(
            [
                float(self.param_.target_q_[1]),
                float(self.param_.target_q_[2]),
                float(self.param_.target_q_[3]),
                float(self.param_.target_q_[0]),
            ],
            dtype=np.float32,
        )
        target_obj_visual_pose = gymapi.Transform()
        if self.show_ghost_object_:
            target_obj_visual_pose.p = gymapi.Vec3(
                float(self._target_obj_pos_cache[0]),
                float(self._target_obj_pos_cache[1]),
                float(self._target_obj_pos_cache[2]),
            )
        else:
            target_obj_visual_pose.p = gymapi.Vec3(
                float(HIDDEN_GHOST_POSITION[0]),
                float(HIDDEN_GHOST_POSITION[1]),
                float(HIDDEN_GHOST_POSITION[2]),
            )
        target_obj_visual_pose.r = gymapi.Quat(
            float(self._target_obj_quat_xyzw_cache[0]),
            float(self._target_obj_quat_xyzw_cache[1]),
            float(self._target_obj_quat_xyzw_cache[2]),
            float(self._target_obj_quat_xyzw_cache[3]),
        )

        marker_pose = gymapi.Transform()
        marker_pose.p = gymapi.Vec3(0.3, 0.0, 0.4)
        marker_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        p_arm_marker_pose = gymapi.Transform()
        p_arm_marker_pose.p = gymapi.Vec3(0.3, 0.0, 0.4)
        p_arm_marker_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.franka_actor = self.gym.create_actor(self.env, self.franka_asset, franka_pose, "franka", 0, 0)
        self.table_actor = self.gym.create_actor(self.env, self.table_asset, table_pose, "table", 0, 0)
        metallic_gray = gymapi.Vec3(0.28, 0.30, 0.33)

        num_bodies = self.gym.get_actor_rigid_body_count(self.env, self.table_actor)
        for i in range(num_bodies):
            self.gym.set_rigid_body_color(
                self.env,
                self.table_actor,
                i,
                gymapi.MESH_VISUAL,
                metallic_gray,
            )

        # num_shapes = self.gym.get_actor_rigid_shape_count(self.env, self.table_actor)
        # for i in range(num_shapes):
        #     self.gym.set_rigid_shape_color(
        #         self.env,
        #         self.table_actor,
        #         i,
        #         gymapi.MESH_VISUAL_AND_COLLISION,
        #         silver_color
        #     )
        self.obj_actor = self.gym.create_actor(self.env, self.obj_asset, obj_pose, "obj", 0, 0)
        self.target_obj_actor = self.gym.create_actor(
            self.env, self.target_obj_asset, target_obj_visual_pose, "target_obj", 0, 0
        )
        self.obj_pose_axes_actor = self.gym.create_actor(
            self.env, self.pose_axes_asset, obj_pose, "obj_pose_axes", 0, 0
        )
        self.target_pose_axes_actor = self.gym.create_actor(
            self.env, self.ghost_pose_axes_asset, target_obj_pose, "target_pose_axes", 0, 0
        )
        # self.marker_actor = self.gym.create_actor(self.env, self.marker_asset, marker_pose, "marker", 0, 0)
        # self.p_arm_marker_actor = self.gym.create_actor(
        #     self.env, self.marker_asset, p_arm_marker_pose, "p_arm_marker", 0, 0
        # )

        texture_applied = self._apply_object_surface_texture(repo_root, self.obj_actor)
        if not texture_applied:
            self.gym.set_rigid_body_color(self.env, self.obj_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.6, 1.0))
        self.gym.set_rigid_body_color(
            self.env, self.target_obj_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.9, 0.9, 0.9)
        )
        # self.gym.set_rigid_body_color(self.env, self.marker_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.1, 0.1))
        # self.gym.set_rigid_body_color(
        #     self.env, self.p_arm_marker_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.9, 0.1)
        # )

    def _resolve_model_xml_path(self, repo_root):
        model_path = getattr(self.param_, "model_path_", None)
        if model_path is None:
            return None
        model_abs = model_path if os.path.isabs(model_path) else os.path.join(repo_root, model_path)
        model_abs = os.path.abspath(model_abs)
        return model_abs if os.path.isfile(model_abs) else None

    def _get_object_surface_texture_path(self, repo_root):
        model_xml = self._resolve_model_xml_path(repo_root)
        if model_xml is None:
            return None

        try:
            root = ET.parse(model_xml).getroot()
        except ET.ParseError:
            return None

        asset = root.find("asset")
        if asset is None:
            return None

        texture_by_name = {}
        for texture_elem in asset.findall("texture"):
            tex_name = texture_elem.get("name")
            tex_file = texture_elem.get("file")
            if tex_name and tex_file:
                texture_by_name[tex_name] = tex_file

        material_elem = asset.find("./material[@name='obj_surface']")
        if material_elem is None:
            material_elem = asset.find("./material[@name='obj_material']")
        if material_elem is None:
            return None

        texture_name = material_elem.get("texture")
        texture_rel = texture_by_name.get(texture_name)
        if texture_rel is None:
            return None

        texture_abs = os.path.abspath(os.path.join(os.path.dirname(model_xml), texture_rel))
        return texture_abs if os.path.isfile(texture_abs) else None

    def _apply_object_surface_texture(self, repo_root, actor_handle):
        if not bool(getattr(self.param_, "use_xml_texture_", False)):
            return False
        texture_abs = self._get_object_surface_texture_path(repo_root)
        if texture_abs is None:
            return False
        if not hasattr(self.gym, "create_texture_from_file") or not hasattr(self.gym, "set_rigid_body_texture"):
            return False

        try:
            texture_handle = self.gym.create_texture_from_file(self.sim, texture_abs)
            self.gym.set_rigid_body_texture(
                self.env,
                actor_handle,
                0,
                gymapi.MESH_VISUAL,
                texture_handle,
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _get_franka_asset_info(repo_root):
        return (
            os.path.join(repo_root, "envs/robots/assets/urdf"),
            os.path.join("franka_description", "robots", "franka_panda_gripper.urdf"),
        )

    @staticmethod
    def _prepare_mesh_urdf_assets(repo_root, mesh_path):
        if mesh_path is None:
            raise ValueError("param.mesh_path_ is required for Isaac object asset loading")

        mesh_abs_path = mesh_path if os.path.isabs(mesh_path) else os.path.join(repo_root, mesh_path)
        mesh_abs_path = os.path.abspath(mesh_abs_path)
        if not os.path.isfile(mesh_abs_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_abs_path}")

        mesh_asset_root = os.path.join(repo_root, "envs", "assets", "objects", "_isaac_tmp")
        os.makedirs(mesh_asset_root, exist_ok=True)
        mesh_rel_to_urdf = os.path.relpath(mesh_abs_path, mesh_asset_root)
        # mesh_scale = "0.0025 0.0025 0.0025"
        mesh_scale = "1 1 1"
        obj_urdf_rel = "obj_mesh_dynamic.urdf"
        obj_urdf_abs = os.path.join(mesh_asset_root, obj_urdf_rel)
        target_urdf_rel = "obj_mesh_target_ghost.urdf"
        target_urdf_abs = os.path.join(mesh_asset_root, target_urdf_rel)

        obj_urdf = f"""<?xml version="1.0"?>
<robot name="mesh_obj">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.05"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_rel_to_urdf}" scale="{mesh_scale}"/>
      </geometry>
      <material name="obj_color">
        <color rgba="0.2 0.6 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_rel_to_urdf}" scale="{mesh_scale}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
        ghost_urdf = f"""<?xml version="1.0"?>
<robot name="mesh_target_ghost">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_rel_to_urdf}" scale="{mesh_scale}"/>
      </geometry>
      <material name="ghost_color">
        <color rgba="0.95 0.95 0.95 0.35"/>
      </material>
    </visual>
  </link>
</robot>
"""
        with open(obj_urdf_abs, "w", encoding="ascii") as f:
            f.write(obj_urdf)
        with open(target_urdf_abs, "w", encoding="ascii") as f:
            f.write(ghost_urdf)

        return obj_urdf_rel, target_urdf_rel, mesh_asset_root

    @staticmethod
    def _prepare_marker_urdf_asset(repo_root):
        marker_asset_root = os.path.join(repo_root, "envs", "assets", "objects", "_isaac_tmp")
        os.makedirs(marker_asset_root, exist_ok=True)
        marker_urdf_rel = "marker_visual_only.urdf"
        marker_urdf_abs = os.path.join(marker_asset_root, marker_urdf_rel)

        marker_urdf = """<?xml version="1.0"?>
<robot name="marker_visual_only">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.012"/>
      </geometry>
      <material name="marker_red">
        <color rgba="1.0 0.1 0.1 1.0"/>
      </material>
    </visual>
  </link>
</robot>
"""
        with open(marker_urdf_abs, "w", encoding="ascii") as f:
            f.write(marker_urdf)

        return marker_urdf_rel, marker_asset_root

    @staticmethod
    def _prepare_pose_axes_urdf_asset(repo_root, translucent=False):
        marker_asset_root = os.path.join(repo_root, "envs", "assets", "objects", "_isaac_tmp")
        os.makedirs(marker_asset_root, exist_ok=True)
        pose_axes_urdf_rel = "pose_axes_visual_ghost.urdf" if translucent else "pose_axes_visual.urdf"
        pose_axes_urdf_abs = os.path.join(marker_asset_root, pose_axes_urdf_rel)

        axis_alpha = 0.45 if translucent else 1.0
        sphere_alpha = 0.55 if translucent else 1.0
        pose_axes_urdf = f"""<?xml version="1.0"?>
<robot name="pose_axes_visual">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="{POSE_AXIS_CENTER_RADIUS}"/>
      </geometry>
      <material name="pose_axis_center">
        <color rgba="0.55 0.55 0.55 {sphere_alpha}"/>
      </material>
    </visual>
    <visual>
      <origin xyz="{0.5 * POSE_AXIS_LENGTH} 0 0" rpy="0 1.57079632679 0"/>
      <geometry>
        <cylinder radius="{POSE_AXIS_RADIUS}" length="{POSE_AXIS_LENGTH}"/>
      </geometry>
      <material name="pose_axis_x">
        <color rgba="1.0 0.2 0.2 {axis_alpha}"/>
      </material>
    </visual>
    <visual>
      <origin xyz="0 {0.5 * POSE_AXIS_LENGTH} 0" rpy="-1.57079632679 0 0"/>
      <geometry>
        <cylinder radius="{POSE_AXIS_RADIUS}" length="{POSE_AXIS_LENGTH}"/>
      </geometry>
      <material name="pose_axis_y">
        <color rgba="0.2 1.0 0.2 {axis_alpha}"/>
      </material>
    </visual>
    <visual>
      <origin xyz="0 0 {0.5 * POSE_AXIS_LENGTH}" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="{POSE_AXIS_RADIUS}" length="{POSE_AXIS_LENGTH}"/>
      </geometry>
      <material name="pose_axis_z">
        <color rgba="0.2 0.45 1.0 {axis_alpha}"/>
      </material>
    </visual>
  </link>
</robot>
"""
        with open(pose_axes_urdf_abs, "w", encoding="ascii") as f:
            f.write(pose_axes_urdf)

        return pose_axes_urdf_rel, marker_asset_root

    def _configure_franka(self):
        dof_props = self.gym.get_actor_dof_properties(self.env, self.franka_actor)
        dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][:] = 400.0
        dof_props["damping"][:] = 60.0
        self.gym.set_actor_dof_properties(self.env, self.franka_actor, dof_props)

        self.franka_dof_count = self.gym.get_actor_dof_count(self.env, self.franka_actor)

        self._joint_targets = np.zeros(self.franka_dof_count, dtype=np.float32)
        self._joint_targets[:7] = np.array(self.param_.init_robot_qpos_, dtype=np.float32)
        if self.franka_dof_count >= 9:
            self._joint_targets[7] = 0.04
            self._joint_targets[8] = 0.04

        self.gym.set_actor_dof_position_targets(self.env, self.franka_actor, self._joint_targets)

    def _build_body_index_cache(self):
        self.obj_body_idx = self.gym.get_actor_rigid_body_index(
            self.env, self.obj_actor, 0, gymapi.DOMAIN_SIM
        )
        self.table_body_idx = self.gym.get_actor_rigid_body_index(
            self.env, self.table_actor, 0, gymapi.DOMAIN_SIM
        )

        franka_body_count = self.gym.get_actor_rigid_body_count(self.env, self.franka_actor)
        self.franka_body_indices = set(
            self.gym.get_actor_rigid_body_index(self.env, self.franka_actor, i, gymapi.DOMAIN_SIM)
            for i in range(franka_body_count)
        )
        self.franka_body_names = self.gym.get_actor_rigid_body_names(self.env, self.franka_actor)
        self.franka_body_count = len(self.franka_body_names)
        self.franka_body_name_to_index = {
            self.franka_body_names[i]: self.gym.get_actor_rigid_body_index(
                self.env, self.franka_actor, i, gymapi.DOMAIN_SIM
            )
            for i in range(self.franka_body_count)
        }
        self.sim_body_to_actor = {}
        for i in range(self.franka_body_count):
            sim_idx = self.gym.get_actor_rigid_body_index(self.env, self.franka_actor, i, gymapi.DOMAIN_SIM)
            self.sim_body_to_actor[sim_idx] = (self.franka_actor, i, "franka")
        self.sim_body_to_actor[self.obj_body_idx] = (self.obj_actor, 0, "obj")
        self.sim_body_to_actor[self.table_body_idx] = (self.table_actor, 0, "table")

        jac = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        self._jacobian = gymtorch.wrap_tensor(jac)
        self._jacobian_body_offset = self.franka_body_count - int(self._jacobian.shape[1])
        if self._jacobian_body_offset not in (0, 1):
            raise RuntimeError(
                f"Unexpected Franka Jacobian body shape: rigid_bodies={self.franka_body_count}, "
                f"jacobian_bodies={int(self._jacobian.shape[1])}"
            )

    @staticmethod
    def _contact_field(c, *names, default=None):
        if not isinstance(c, np.void):
            return default
        if getattr(c, "dtype", None) is None or c.dtype.names is None:
            return default
        for n in names:
            if n in c.dtype.names:
                return c[n]
        return default

    def get_physx_contacts(self):
        contacts_raw = self.gym.get_env_rigid_contacts(self.env)
        contacts = []
        if contacts_raw is None:
            return contacts

        for c in contacts_raw:
            body0 = self._contact_field(c, "body0", "bodyA", default=-1)
            body1 = self._contact_field(c, "body1", "bodyB", default=-1)
            if body0 is None or body1 is None:
                continue
            body0 = int(body0)
            body1 = int(body1)

            # Isaac Gym's RigidContact has no ``separation`` field.
            # ``initial_overlap`` is positive for penetration, while the
            # explicit model expects MuJoCo's negative-distance convention.
            separation_raw = self._contact_field(c, "separation", "distance", default=None)
            if separation_raw is None:
                initial_overlap = self._contact_field(c, "initial_overlap", default=0.0)
                separation = -float(initial_overlap if initial_overlap is not None else 0.0)
            else:
                separation = float(separation_raw)

            normal_raw = self._contact_field(c, "normal", default=np.array([0.0, 0.0, 1.0], dtype=np.float32))
            normal = _extract_vec3(normal_raw)
            nrm = np.linalg.norm(normal)
            if nrm > 1e-8:
                normal = normal / nrm
            else:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # RigidContact exposes feature points in each body's local frame.
            # Convert both to world and average them.  A fabricated point near
            # the object center gives the wrong object moment arm and robot
            # point Jacobian.
            pos = None
            pos_field = self._contact_field(c, "pos", "position", default=None)
            if pos_field is not None:
                pos = _extract_vec3(pos_field)
            else:
                local_pos0 = self._contact_field(c, "local_pos0", default=None)
                local_pos1 = self._contact_field(c, "local_pos1", default=None)
                world_points = []
                for body_idx, local_pos in ((body0, local_pos0), (body1, local_pos1)):
                    if local_pos is None:
                        continue
                    body_pos, body_quat_xyzw = self.get_body_pose_by_sim_index(body_idx)
                    if body_pos is None:
                        continue
                    body_rot = Rotation.from_quat(body_quat_xyzw).as_matrix()
                    world_points.append(body_pos + body_rot @ _extract_vec3(local_pos))
                if world_points:
                    pos = np.mean(np.stack(world_points, axis=0), axis=0).astype(np.float32)

            contacts.append(
                {
                    "body0": body0,
                    "body1": body1,
                    "separation": separation,
                    "normal": normal,
                    "pos": pos,
                }
            )

        return contacts

    def _get_actor_body_pose(self, actor_handle, local_body_idx):
        states = self.gym.get_actor_rigid_body_states(self.env, actor_handle, gymapi.STATE_POS)
        p = _extract_vec3(states["pose"]["p"][local_body_idx])
        q_xyzw = _extract_quat_xyzw(states["pose"]["r"][local_body_idx])
        return p, q_xyzw

    def get_body_pose_by_sim_index(self, sim_body_idx):
        info = self.sim_body_to_actor.get(int(sim_body_idx), None)
        if info is None:
            return None, None
        actor_handle, local_idx, _ = info
        return self._get_actor_body_pose(actor_handle, local_idx)

    @staticmethod
    def _skew(v):
        return np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ],
            dtype=np.float32,
        )

    def _jacobian_body_index(self, local_body_idx):
        jac_idx = int(local_body_idx) - int(self._jacobian_body_offset)
        if jac_idx < 0 or jac_idx >= int(self._jacobian.shape[1]):
            raise IndexError(
                f"Rigid body index {local_body_idx} maps to invalid jacobian index {jac_idx}; "
                f"offset={self._jacobian_body_offset}, jacobian_bodies={int(self._jacobian.shape[1])}"
            )
        return jac_idx

    def get_body_point_jacobian(self, sim_body_idx, point_world):
        info = self.sim_body_to_actor.get(int(sim_body_idx), None)
        if info is None:
            return np.zeros((3, self.param_.n_robot_qpos_), dtype=np.float32)
        _, local_idx, actor_kind = info
        if actor_kind != "franka":
            return np.zeros((3, self.param_.n_robot_qpos_), dtype=np.float32)

        self.gym.refresh_jacobian_tensors(self.sim)
        jac_body_idx = self._jacobian_body_index(local_idx)
        jac_body = self._jacobian[0, jac_body_idx]  # (6, dof)
        jv = np.array(jac_body[:3, : self.param_.n_robot_qpos_], dtype=np.float32)
        jw = np.array(jac_body[3:6, : self.param_.n_robot_qpos_], dtype=np.float32)

        p_body, _ = self.get_body_pose_by_sim_index(sim_body_idx)
        if p_body is None:
            return jv
        r = np.asarray(point_world, dtype=np.float32) - np.asarray(p_body, dtype=np.float32)
        return jv - self._skew(r) @ jw

    def _simulate_once(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self._sync_pose_axes()
        if self.viewer_ is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer_, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def reset_mj_env(self):
        # reset object pose according to params.py random init
        obj_state = self.gym.get_actor_rigid_body_states(self.env, self.obj_actor, gymapi.STATE_ALL)
        init_obj = np.array(self.param_.init_obj_qpos_, dtype=np.float32)
        obj_state["pose"]["p"][0] = (float(init_obj[0]), float(init_obj[1]), float(init_obj[2]))
        # Isaac quat is xyzw, params are wxyz
        obj_state["pose"]["r"][0] = (float(init_obj[4]), float(init_obj[5]), float(init_obj[6]), float(init_obj[3]))
        obj_state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
        obj_state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
        self.gym.set_actor_rigid_body_states(self.env, self.obj_actor, obj_state, gymapi.STATE_ALL)

        dof_states = self.gym.get_actor_dof_states(self.env, self.franka_actor, gymapi.STATE_ALL)
        dof_states["pos"][:] = 0.0
        dof_states["vel"][:] = 0.0
        dof_states["pos"][:7] = np.array(self.param_.init_robot_qpos_, dtype=np.float32)
        if self.franka_dof_count >= 9:
            dof_states["pos"][7] = 0.04
            dof_states["pos"][8] = 0.04
        self.gym.set_actor_dof_states(self.env, self.franka_actor, dof_states, gymapi.STATE_ALL)

        self._joint_targets[:7] = np.array(self.param_.init_robot_qpos_, dtype=np.float32)
        if self.franka_dof_count >= 9:
            self._joint_targets[7] = 0.04
            self._joint_targets[8] = 0.04
        self.gym.set_actor_dof_position_targets(self.env, self.franka_actor, self._joint_targets)
        self.show_target_object_pose(self.param_.target_p_, self.param_.target_q_)
        self._sync_pose_axes()

        for _ in range(8):
            self._simulate_once()

    # def set_goal(self, goal_pos=None, goal_quat=None):
    #     self.show_target(goal_pos=goal_pos, goal_quat=goal_quat)

    def set_ghost_object_visibility(self, visible):
        self.show_ghost_object_ = bool(visible)
        self._sync_target_object_visual()

    # def show_target(self, goal_pos=None, goal_quat=None):
    #     marker_state = self.gym.get_actor_rigid_body_states(self.env, self.marker_actor, gymapi.STATE_ALL)
    #     if goal_pos is not None:
    #         marker_state["pose"]["p"][0] = (float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]))
    #     if goal_quat is not None:
    #         # input quat from scipy Rotation.as_quat() is xyzw
    #         marker_state["pose"]["r"][0] = (
    #             float(goal_quat[0]),
    #             float(goal_quat[1]),
    #             float(goal_quat[2]),
    #             float(goal_quat[3]),
    #         )
    #     marker_state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
    #     marker_state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
        # self.gym.set_actor_rigid_body_states(self.env, self.marker_actor, marker_state, gymapi.STATE_ALL)

    def show_point(self, goal_pos=None):
        marker_actor = getattr(self, "p_arm_marker_actor", self.marker_actor)
        marker_state = self.gym.get_actor_rigid_body_states(self.env, marker_actor, gymapi.STATE_ALL)
        if goal_pos is not None:
            marker_state["pose"]["p"][0] = (float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]))
        marker_state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
        marker_state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
        self.gym.set_actor_rigid_body_states(self.env, marker_actor, marker_state, gymapi.STATE_ALL)
        
    def show_target_object_pose(self, goal_pos=None, goal_quat_wxyz=None):
        if goal_pos is not None:
            self._target_obj_pos_cache = np.asarray(goal_pos, dtype=np.float32).copy()
        if goal_quat_wxyz is not None:
            self._target_obj_quat_xyzw_cache = np.array(
                [
                    float(goal_quat_wxyz[1]),
                    float(goal_quat_wxyz[2]),
                    float(goal_quat_wxyz[3]),
                    float(goal_quat_wxyz[0]),
                ],
                dtype=np.float32,
            )
        self._sync_target_object_visual()
        self._sync_pose_axes()

    def _set_visual_actor_pose_xyzw(self, actor_handle, position, quat_xyzw):
        state = self.gym.get_actor_rigid_body_states(self.env, actor_handle, gymapi.STATE_ALL)
        state["pose"]["p"][0] = (float(position[0]), float(position[1]), float(position[2]))
        state["pose"]["r"][0] = (
            float(quat_xyzw[0]),
            float(quat_xyzw[1]),
            float(quat_xyzw[2]),
            float(quat_xyzw[3]),
        )
        state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
        state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
        self.gym.set_actor_rigid_body_states(self.env, actor_handle, state, gymapi.STATE_ALL)

    def _sync_target_object_visual(self):
        if not hasattr(self, "target_obj_actor"):
            return
        target_pos = self._target_obj_pos_cache if self.show_ghost_object_ else HIDDEN_GHOST_POSITION
        self._set_visual_actor_pose_xyzw(
            self.target_obj_actor,
            target_pos,
            self._target_obj_quat_xyzw_cache,
        )

    def _sync_pose_axes(self):
        if not hasattr(self, "obj_pose_axes_actor") or not hasattr(self, "target_pose_axes_actor"):
            return
        obj_pos, obj_quat_xyzw = self._get_actor_body_pose(self.obj_actor, 0)
        self._set_visual_actor_pose_xyzw(self.obj_pose_axes_actor, obj_pos, obj_quat_xyzw)
        self._set_visual_actor_pose_xyzw(
            self.target_pose_axes_actor,
            self._target_obj_pos_cache,
            self._target_obj_quat_xyzw_cache,
        )

    def draw_rollout_lines(self, rollout_q):
        if self.viewer_ is None:
            return
        self.gym.clear_lines(self.viewer_)
        if rollout_q is None:
            return
        rollout_q = np.asarray(rollout_q, dtype=np.float32)
        if rollout_q.ndim != 2 or rollout_q.shape[0] < 2 or rollout_q.shape[1] < 7 + self.param_.n_robot_qpos_:
            return

        ee_points = []
        for i in range(rollout_q.shape[0]):
            q_robot = rollout_q[i, -self.param_.n_robot_qpos_:]
            T = np.array(_franka_fk_T_jax(q_robot), dtype=np.float32)
            ee_points.append(T[:3, 3])

        for i in range(len(ee_points) - 1):
            p0 = ee_points[i]
            p1 = ee_points[i + 1]
            c = 0.2 + 0.8 * (i / max(len(ee_points) - 2, 1))
            self.gym.add_lines(
                self.viewer_,
                self.env,
                1,
                [float(p0[0]), float(p0[1]), float(p0[2]), float(p1[0]), float(p1[1]), float(p1[2])],
                [1.0 - 0.6 * c, 0.3 + 0.5 * c, 0.1],
            )

    def get_current_joint_position(self):
        dof_states = self.gym.get_actor_dof_states(self.env, self.franka_actor, gymapi.STATE_POS)
        return np.array(dof_states["pos"][:7], dtype=np.float32)

    def get_state(self):
        obj_state = self.gym.get_actor_rigid_body_states(self.env, self.obj_actor, gymapi.STATE_POS)
        obj_pos = _extract_vec3(obj_state["pose"]["p"][0])
        quat_xyzw = _extract_quat_xyzw(obj_state["pose"]["r"][0])
        obj_quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)

        q = self.get_current_joint_position()
        return np.hstack([obj_pos, obj_quat_wxyz, q]).astype(np.float32)

    def get_end_effector_pos(self):
        q = self.get_current_joint_position()
        T = np.array(_franka_fk_T_jax(q), dtype=np.float32)
        p = T[:3, 3]
        R = T[:3, :3]
        return p, R

    def get_R(self):
        _, R = self.get_end_effector_pos()
        return R

    def step_joint_delta(self, dq):
        q = self.get_current_joint_position()
        dq = np.asarray(dq, dtype=np.float32).reshape(7)
        q_des = q + dq

        self._joint_targets[:7] = q_des
        if self.franka_dof_count >= 9:
            self._joint_targets[7] = 0.04
            self._joint_targets[8] = 0.04
        self.gym.set_actor_dof_position_targets(self.env, self.franka_actor, self._joint_targets)

        self._simulate_once()

    def step(self, cmd):
        cmd = np.asarray(cmd).reshape(-1)
        if cmd.size != 7:
            raise ValueError(f"Invalid action dimension: {cmd.size}. Expected 7.")
        self.step_joint_delta(cmd)

    def close(self):
        if self.viewer_ is not None:
            self.gym.destroy_viewer(self.viewer_)
            self.viewer_ = None
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None


class ContactIsaac:
    """Isaac replacement for contact.franka_collision_detection2.Contact."""

    def __init__(self, param):
        self.param_ = param

    def detect_once(self, simulator: IsaacFrankaSimulator):
        q = simulator.get_state()
        q = np.asarray(q, dtype=np.float32)
        nv = self.param_.n_qvel_
        max_ncon = self.param_.max_ncon_

        phi_vec = np.ones((max_ncon * 4,), dtype=np.float32)
        jac_mat = np.zeros((max_ncon * 4, nv), dtype=np.float32)
        jac_mat_env = np.zeros((max_ncon * 4, nv), dtype=np.float32)
        con_pos_list = []

        obj_pos = q[0:3]
        obj_quat_wxyz = q[3:7]
        q_robot = q[-self.param_.n_robot_qpos_:]

        mu = float(self.param_.mu_object_)
        contact_sep_threshold = float(getattr(self.param_, "if_contact_separation_threshold_", 0.0))
        contacts = simulator.get_physx_contacts()
        if_contact = False

        row_idx = 0
        row_env_idx = 0
        for c in contacts:
            b0 = c["body0"]
            b1 = c["body1"]
            sep = float(c["separation"])
            n_raw = c["normal"]
            cpos = c["pos"]

            involves_obj = (b0 == simulator.obj_body_idx) or (b1 == simulator.obj_body_idx)
            if not involves_obj:
                continue

            # Flip normal to point from object -> other body.
            if b1 == simulator.obj_body_idx:
                n_raw = -n_raw

            # Use PhysX contact position when available, otherwise a small step along normal.
            if cpos is None:
                cpos = obj_pos + 0.01 * n_raw
            else:
                cpos = np.asarray(cpos, dtype=np.float32)

            # Object-point translational Jacobian in world frame.
            r_obj = cpos - obj_pos
            J_obj = np.zeros((3, nv), dtype=np.float32)
            J_obj[:, 0:3] = np.eye(3, dtype=np.float32)
            J_obj[:, 3:6] = -simulator._skew(r_obj)

            # Other body Jacobian (franka contact body or static table).
            J_other = np.zeros((3, nv), dtype=np.float32)
            other_sim_idx = b1 if b0 == simulator.obj_body_idx else b0
            if other_sim_idx in simulator.franka_body_indices:
                J_body_point = simulator.get_body_point_jacobian(other_sim_idx, cpos)
                J_other[:, 6: 6 + self.param_.n_robot_qpos_] = J_body_point

            # Relative point Jacobian: object - other body.
            J_rel_point = J_obj - J_other

            n, t1, t2 = _tangent_basis_from_normal(n_raw)
            con_jac = np.array(_contact_jacobian(n, t1, t2, J_rel_point, mu), dtype=np.float32)

            # Object-franka contact contributes to main contact set.
            other_is_franka = (b0 in simulator.franka_body_indices) or (b1 in simulator.franka_body_indices)
            fingertip_sim_idx = simulator.franka_body_name_to_index.get("fingertip")
            other_is_fingertip = fingertip_sim_idx is not None and other_sim_idx == fingertip_sim_idx
            if other_is_fingertip and sep <= contact_sep_threshold:
                if_contact = True
            if other_is_franka and row_idx < max_ncon:
                phi_vec[4 * row_idx: 4 * row_idx + 4] = 0.5 * sep
                jac_mat[4 * row_idx: 4 * row_idx + 4, :] = con_jac
                row_idx += 1

            # Table contacts are used by the contact-point optimizer below,
            # but are deliberately not copied into MPPI's main contact set.
            # PhysX reports several table manifold points; the simple explicit
            # penalty model treats every one as independently supporting the
            # full object weight and consequently predicts a large upward
            # jump.  MPPI instead uses a planar tabletop constraint in its
            # rollout while Isaac remains the source of physical friction.
            other_is_table = (b0 == simulator.table_body_idx) or (b1 == simulator.table_body_idx)
            if other_is_table and row_env_idx < max_ncon:
                jac_mat_env[4 * row_env_idx: 4 * row_env_idx + 4, :] = con_jac
                row_env_idx += 1
                # Return table contact point in object-local frame.
                quat_xyzw = np.array(
                    [obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]],
                    dtype=np.float32,
                )
                R_obj_to_world = Rotation.from_quat(quat_xyzw).as_matrix()
                con_pos_local = R_obj_to_world.T @ (cpos - obj_pos)
                con_pos_list.append(np.array(con_pos_local, dtype=np.float32))

        # Fallback to geometric table contact if PhysX did not report one this frame.
        if row_env_idx == 0:
            dist_table = float(obj_pos[2] - float(self.param_.table_height))
            if dist_table < 0.02:
                n_t = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                t1_t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                t2_t = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                J_rel_table = np.concatenate(
                    [np.eye(3), np.zeros((3, 3)), np.zeros((3, self.param_.n_robot_qpos_))], axis=1
                )
                con_jac_table = np.array(_contact_jacobian(n_t, t1_t, t2_t, J_rel_table, mu), dtype=np.float32)
                jac_mat_env[0:4, :] = con_jac_table
                con_pos_list.append(np.array([0.0, 0.0, -0.025], dtype=np.float32))

        return phi_vec, jac_mat, con_pos_list, jac_mat_env, if_contact


def adapt_param_for_joint_mppi(param, args):
    """Configure the explicit rollout model for Panda joint-delta MPPI."""
    param.n_cmd_ = 7
    param.n_robot_qpos_ = 7
    param.n_qpos_ = 14
    param.n_qvel_ = 13
    param.mppi_tabletop_lock_ = True

    joint_stiffness = float(args.joint_model_stiffness)
    param.robot_stiff_ = np.diag(np.full(7, joint_stiffness, dtype=np.float32))
    q_metric = np.zeros((13, 13), dtype=np.float32)
    q_metric[:6, :6] = param.obj_inertia_
    q_metric[6:, 6:] = param.robot_stiff_
    param.Q = q_metric

    joint_step = float(args.joint_step)
    param.mpc_u_lb_ = -joint_step * np.ones(7, dtype=np.float32)
    param.mpc_u_ub_ = joint_step * np.ones(7, dtype=np.float32)
    panda_q_lb = np.array(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        dtype=np.float32,
    )
    panda_q_ub = np.array(
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        dtype=np.float32,
    )
    param.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), panda_q_lb))
    param.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), panda_q_ub))

    for name in (
        "Nsample", "Hsample", "Hnode", "Ndiffuse", "Ndiffuse_init",
        "temp_sample", "sigma_scale", "traj_diffuse_factor", "horizon_diffuse_factor",
    ):
        setattr(param, f"mppi_{name}_", getattr(args, f"mppi_{name}"))
    for name in (
        "base", "object_pos", "ee_ori", "joint_limit", "manip", "cond", "energy", "vel", "acc",
    ):
        setattr(param, f"mppi_w_{name}_", getattr(args, f"mppi_w_{name}"))
    param.mppi_seed_ = int(args.mppi_seed)

    # The shared DyWA setup randomizes mass/friction after ExplicitMPCParams
    # creates this optimizer, so rebuild it with the actual trial parameters.
    param.lambda_optimizer = LambdaContactControlOptimizer(
        mesh_path=param.mesh_path_,
        obj_mass=param.obj_mass_,
        arm_friction=param.mu_object_,
        contact_stiffness=param.model_params,
        time_step=param.h_,
        sample_num=args.sample_num,
        pos_coef=args.pos_coef,
        ori_coef=args.ori_coef,
        scale_factors=[1.0] * 3,
    )
    param.sol_guess_ = None
    return param


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, default='foam_brick', help='name of obj')
    parser.add_argument('--use-xml-texture', action='store_true', help='apply object texture parsed from env_fingertips_*.xml')
    parser.add_argument('--attract_coef', type=float, default=0.5, help='coef of attract function')
    parser.add_argument('--reject_coef', type=float, default=0.01, help='coef of reject function')
    parser.add_argument('--contact_coef', type=float, default=0.5, help='coef of contact function')
    parser.add_argument('--contact_cost_param', type=float, default=0.0, help='mass center or project point attract')
    parser.add_argument('--model_param', type=float, default=7, help='model param')
    parser.add_argument('--reject_dis', type=float, default=0.01, help='reject radius')
    parser.add_argument('--attract_point_comp', type=float, default=0.05, help='normal offset from the surface used only by the hover/detach phase')
    parser.add_argument('--ground_height_threshold', type=float, default=0.33, help='threshold of sample points height')
    parser.add_argument(
        '--hover-switch-distance',
        type=float,
        default=0.02,
        help='switch from the hover point to the selected surface point after reaching this distance',
    )
    parser.add_argument('--sample_num', type=int, default=70, help='number of sample point')
    parser.add_argument('--pos_coef', type=float, default=1, help='coef of position cost in mlqp_point')
    parser.add_argument('--ori_coef', type=float, default=0.005, help='coef of orientation cost in mlqp_point')
    parser.add_argument('--low_err_coef', type=float, default=0.3, help='coef of delta error')
    parser.add_argument('--upper_err_coef', type=float, default=1.0, help='coef of delta error')
    parser.add_argument('--joint-step', type=float, default=0.08, help='maximum joint delta (rad) per MPPI step')
    parser.add_argument('--joint-model-stiffness', type=float, default=300.0, help='joint stiffness in explicit rollout model')
    parser.add_argument(
        '--control-substeps',
        type=int,
        default=3,
        help='number of Isaac/OSC frames used to track each MPPI joint-delta action',
    )
    parser.add_argument(
        '--contact-min-hold-steps',
        type=int,
        default=30,
        help='minimum outer-loop steps to keep the surface-contact phase before allowing a detach',
    )
    parser.add_argument(
        '--contact-release-bad-steps',
        type=int,
        default=8,
        help='consecutive poor-contact assessments required to leave the surface-contact phase',
    )
    parser.add_argument('--mppi_Nsample', type=int, default=128)
    parser.add_argument('--mppi_Hsample', type=int, default=20)
    parser.add_argument('--mppi_Hnode', type=int, default=5)
    parser.add_argument('--mppi_Ndiffuse', type=int, default=1, help='sequential MPPI updates after warm start')
    parser.add_argument('--mppi_Ndiffuse_init', type=int, default=2, help='MPPI updates on the first solve')
    parser.add_argument('--mppi_temp_sample', type=float, default=0.5)
    parser.add_argument('--mppi_sigma_scale', type=float, default=0.6)
    parser.add_argument('--mppi_traj_diffuse_factor', type=float, default=0.5)
    parser.add_argument('--mppi_horizon_diffuse_factor', type=float, default=0.9)
    parser.add_argument('--mppi_seed', type=int, default=0)
    parser.add_argument('--mppi_w_base', type=float, default=1.0, help='weight for the interaction potential')
    parser.add_argument(
        '--mppi_w_object_pos',
        type=float,
        default=50.0,
        help='contact-stage running weight for object XY translation tracking',
    )
    parser.add_argument('--mppi_w_ee_ori', type=float, default=1.0, help='weight for EE orientation hold cost')
    parser.add_argument('--mppi_w_joint_limit', type=float, default=100.0, help='weight for joint limit penalty')
    parser.add_argument('--mppi_w_manip', type=float, default=0.0, help='weight for manipulability penalty')
    parser.add_argument('--mppi_w_cond', type=float, default=0.0, help='weight for Jacobian condition penalty')
    parser.add_argument('--mppi_w_energy', type=float, default=0.01, help='weight for action energy penalty')
    parser.add_argument('--mppi_w_vel', type=float, default=0.0, help='weight for joint velocity penalty')
    parser.add_argument('--mppi_w_acc', type=float, default=0.0, help='weight for joint acceleration penalty')

    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--sim-device', type=str, default='cuda:0')
    parser.add_argument('--graphics-device-id', type=int, default=0)
    parser.add_argument('--osc-pos-stiffness', type=float, default=2000.0)
    parser.add_argument('--osc-ori-stiffness', type=float, default=400.0)
    parser.add_argument('--effort-joint-damping', type=float, default=10.0)
    parser.add_argument('--trial-start', type=int, default=3)
    parser.add_argument('--trial-count', type=int, default=1)
    parser.add_argument('--max-rollout-length', type=int, default=5000)
    parser.add_argument(
        '--debug-motion',
        action='store_true',
        help='print predicted/actual object motion diagnostics (adds host-side overhead)',
    )
    parser.add_argument(
        '--show-ghost-object',
        type=_parse_bool_arg,
        default=False,
        help='whether to render the semi-transparent ghost object mesh at the target pose (true/false)',
    )

    args = parser.parse_args()

    if args.trial_start < 0 or args.trial_count <= 0:
        raise ValueError("trial-start must be non-negative and trial-count must be positive")
    if args.control_substeps <= 0:
        raise ValueError("control-substeps must be positive")
    if args.contact_min_hold_steps < 0 or args.contact_release_bad_steps <= 0:
        raise ValueError("contact hold/release step counts must be non-negative/positive")

    save_flag = False
    if save_flag:
        save_dir = './examples/mpc/trifinger/elephant/save/'
        prefix_data_name = 'ours_'
        save_data = dict()

    trial_start = int(args.trial_start)
    trial_num = int(args.trial_count)
    trial_stop = trial_start + trial_num
    success_pos_threshold = 0.02
    success_quat_threshold = 0.04
    consecutive_success_time_threshold = 20
    if args.max_rollout_length <= 0:
        raise ValueError("max-rollout-length must be positive")
    max_rollout_length = int(args.max_rollout_length)
    success_rate = 0

    # Use the identical Isaac Gym/OSC implementation and randomized DyWA
    # physics configuration as test_mpc_isaac.py.  MPPI remains the only
    # planner difference and emits seven joint deltas to the same controller.
    # Avoid loading a second copy of this module when this file is executed as
    # a script and test_mpc_isaac imports its shared base simulator.
    sys.modules.setdefault("examples.mpc.franka.ik2.test_mppi_isaac", sys.modules[__name__])
    from examples.mpc.franka.ik2.test_mpc_isaac import (
        IsaacFrankaOSCSimulator,
        _apply_dywa_physics_to_param,
    )

    for trial_count in range(trial_start, trial_stop):
        param = ExplicitMPCParams(args, rand_seed=trial_count, target_type='rotation', mpc_model='explicit')
        param = _apply_dywa_physics_to_param(param)
        param.use_jax_contact_ = False
        param = adapt_param_for_joint_mppi(param, args)
        param.osc_pos_stiffness_ = float(args.osc_pos_stiffness)
        param.osc_ori_stiffness_ = float(args.osc_ori_stiffness)
        param.effort_joint_damping_ = float(args.effort_joint_damping)
        param.control_substeps_ = int(args.control_substeps)
        param.show_ghost_object_ = bool(args.show_ghost_object)
        param.svg_screenshot_dir_ = None

        contact = ContactIsaac(param)
        env = IsaacFrankaOSCSimulator(
            param,
            headless=args.headless,
            sim_device=args.sim_device,
            graphics_device_id=args.graphics_device_id,
        )
        env.show_target_object_pose(param.target_p_, param.target_q_)

        mpc = MPPIExplicit(param) if param.mpc_model == 'explicit' else MPCImplicit(param)

        rollout_step = 0
        consecutive_success_time = 0
        verify_cost = 0
        current_x = np.zeros(7)
        current_x[3] = 1

        low_err_coef = args.low_err_coef
        upper_err_coef = args.upper_err_coef
        consecutive_detect_time = 0
        consecutive_contact_time = 0
        contact_phase_steps = 0
        bad_contact_steps = 0
        initial_object_pose = None

        rollout_q_traj = []
        while rollout_step < max_rollout_length:
            if not env.dyn_paused_:
                curr_q = env.get_state()
                rollout_q_traj.append(curr_q)
                if initial_object_pose is None:
                    initial_object_pose = curr_q[:7].copy()

                phi_vec, jac_mat, con_point, jac_mat_env, if_contact = contact.detect_once(env)
                quanternion = [curr_q[4], curr_q[5], curr_q[6], curr_q[3]]  # wxyz -> xyzw
                R_obj_to_world = Rotation.from_quat(quanternion).as_matrix()
                gravity = np.hstack([R_obj_to_world.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3)])

                target_pos_ = param.target_p_ - curr_q[0:3]
                target_pos_[2] = 0

                target_quat_local = rotations.quaternion_multiply(
                    rotations.quaternion_conjugate(curr_q[3:7]),
                    param.target_q_)
                target_pose_local = np.hstack([R_obj_to_world.T @ target_pos_, target_quat_local])

                param.lambda_optimizer.update_Jacobian(jac_mat_env)
                visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                    curr_q[0:3], R_obj_to_world, param.target_p_, args.ground_height_threshold
                )
                best_contact_point, normal, min_error, max_error, curr_ori_coef = param.lambda_optimizer.choose_contact_points(
                    target_pose_local, current_x, gravity, visible_point_idx
                )

                best_contact_point_world = R_obj_to_world @ best_contact_point + curr_q[0:3]
                attract_point = best_contact_point.copy()
                attract_point_world = R_obj_to_world @ attract_point + curr_q[0:3]
                original_height = attract_point_world[2]
                attract_point_world -= args.attract_point_comp * R_obj_to_world @ normal
                attract_point_world[2] = max(attract_point_world[2], original_height)

                ee_pos = env.get_end_effector_pos()[0].copy()
                local_point = R_obj_to_world.T @ (ee_pos - curr_q[0:3])
                p_arm_local, _, x_plus_opt, error, info = param.lambda_optimizer.optimize_control_input(
                    target_pose_local, current_x, gravity, local_point
                )
                # Enforce that p_arm is represented by a vertex of the same
                # scaled mesh used to create the Isaac object (scale 1, no
                # local URDF offset).  optimize_control_input currently does
                # this projection too; repeating it here makes the geometric
                # contract explicit and guards future optimizer changes.
                p_arm_raw_local = np.asarray(p_arm_local, dtype=np.float32).reshape(3)
                p_arm_surface_idx, _, _, _ = param.lambda_optimizer.pp.project_point_to_mesh(
                    p_arm_raw_local
                )
                p_arm_local = np.asarray(
                    param.lambda_optimizer.pp.scaled_mesh.vertices[p_arm_surface_idx],
                    dtype=np.float32,
                )
                p_arm_projection_residual = float(np.linalg.norm(p_arm_raw_local - p_arm_local))
                p_arm_world = R_obj_to_world @ p_arm_local + curr_q[:3]
                surface_distance = float(np.linalg.norm(ee_pos - best_contact_point_world))
                template_to_arm_surface_distance = float(
                    np.linalg.norm(best_contact_point_world - p_arm_world)
                )
                hover_distance = float(np.linalg.norm(ee_pos - attract_point_world))

                if verify_cost:
                    low_err_coef = args.low_err_coef
                else:
                    if np.linalg.norm(ee_pos - attract_point_world) < 5e-2:
                        low_err_coef = min(low_err_coef * 1.1, 1.0)

                delta_error = max(float(max_error - min_error), 1e-6)
                improvement = float(max_error - error) / delta_error
                if not verify_cost:
                    # Phase 0 has exactly one purpose: reach the normal-offset
                    # hover/detach point.  Do not let an abstract object-pose
                    # improvement switch the geometric target prematurely.
                    hover_ready = hover_distance <= args.hover_switch_distance
                    consecutive_detect_time = consecutive_detect_time + 1 if hover_ready else 0
                    if consecutive_detect_time >= 2:
                        verify_cost = 1
                        consecutive_detect_time = 0
                        consecutive_contact_time = 0
                        contact_phase_steps = 0
                        bad_contact_steps = 0
                else:
                    # Contact must be held long enough to transmit momentum to
                    # the object.  The old cumulative five-contact-frame test
                    # could immediately switch back to the hover target and
                    # unload the fingertip before any useful push occurred.
                    contact_phase_steps += 1
                    consecutive_contact_time = consecutive_contact_time + 1 if if_contact else 0
                    bad_contact_steps = bad_contact_steps + 1 if improvement <= low_err_coef else 0
                    may_release = contact_phase_steps >= args.contact_min_hold_steps
                    persistently_bad = bad_contact_steps >= args.contact_release_bad_steps
                    if may_release and persistently_bad:
                        verify_cost = 0
                        consecutive_detect_time = 0
                        consecutive_contact_time = 0
                        contact_phase_steps = 0
                        bad_contact_steps = 0
                print('min error:', min_error, 'max error', max_error, 'actual error:', error, 'low_coef:', low_err_coef)
                print("verify cost:", verify_cost)

                st = time.time()
                sol = mpc.plan_once(
                    param.target_p_,
                    param.target_q_,
                    curr_q,
                    phi_vec,
                    jac_mat,
                    verify_cost_param=verify_cost,
                    virtual_point=attract_point_world,
                    # p_arm_world is the nearest valid surface point.  The
                    # best template point is used only to assess whether the
                    # nearby contact is good enough.
                    contact_point=p_arm_world,
                    curr_ori_coef=curr_ori_coef,
                    sol_guess=param.sol_guess_,
                )
                param.sol_guess_ = sol['sol_guess']
                action = np.asarray(sol['action'], dtype=np.float32)
                predicted_object_delta = None
                predicted_contact_force_norm = None
                if args.debug_motion:
                    # One-step model diagnostic: this must become non-zero when
                    # a candidate action loads an active robot-object contact.
                    q_inv = np.linalg.inv(np.asarray(param.Q, dtype=np.float32))
                    model_gravity = np.asarray(param.gravity_, dtype=np.float32).copy()
                    if bool(getattr(param, "mppi_tabletop_lock_", False)):
                        model_gravity[2] = 0.0
                    b_model = np.hstack(
                        [
                            float(param.obj_mass_) * model_gravity,
                            np.asarray(param.robot_stiff_, dtype=np.float32) @ action,
                        ]
                    )
                    jqb_model = jac_mat @ q_inv @ b_model
                    contact_force_model = np.maximum(
                        -float(param.model_params) * (jqb_model + phi_vec),
                        0.0,
                    )
                    velocity_model = (
                        q_inv @ b_model + q_inv @ jac_mat.T @ contact_force_model
                    ) / float(param.h_)
                    predicted_object_delta = float(param.h_) * velocity_model[:3]
                    if bool(getattr(param, "mppi_tabletop_lock_", False)):
                        predicted_object_delta[2] = 0.0
                    predicted_contact_force_norm = float(np.linalg.norm(contact_force_model))
                # env.draw_rollout_lines(sol.get("rollout_q", None))
                print("time_cost = ", time.time() - st, action.shape)
                print("attract_point_world = ", attract_point_world)
                active_target = p_arm_world if verify_cost else attract_point_world
                print(
                    "surface_distance = ",
                    surface_distance,
                    "template_to_arm_surface_distance = ",
                    template_to_arm_surface_distance,
                    "hover_distance = ",
                    hover_distance,
                    "active_target_distance = ",
                    float(np.linalg.norm(ee_pos - active_target)),
                    "p_arm_projection_residual = ",
                    p_arm_projection_residual,
                    "if_contact = ",
                    bool(if_contact),
                    "contact_phase_steps = ",
                    contact_phase_steps,
                    "action_norm = ",
                    float(np.linalg.norm(action)),
                    "object_displacement = ",
                    float(np.linalg.norm(curr_q[:3] - initial_object_pose[:3])),
                )
                if args.debug_motion:
                    print(
                        "motion_debug: object_rotation_change = ",
                        float(metrics.comp_quat_error(curr_q[3:7], initial_object_pose[3:7])),
                        "target_pos_error = ",
                        float(metrics.comp_pos_error(curr_q[:3], param.target_p_)),
                        "target_quat_error = ",
                        float(metrics.comp_quat_error(curr_q[3:7], param.target_q_)),
                        "predicted_object_delta = ",
                        predicted_object_delta,
                        "predicted_contact_force_norm = ",
                        predicted_contact_force_norm,
                    )

                env.show_point(active_target)
                state_before_execution = curr_q.copy()
                env.step(action)

                rollout_step = rollout_step + 1

                curr_q = env.get_state()
                if args.debug_motion:
                    print(
                        "actual_object_step_delta = ",
                        curr_q[:3] - state_before_execution[:3],
                        "actual_object_step_rotation = ",
                        float(metrics.comp_quat_error(curr_q[3:7], state_before_execution[3:7])),
                    )
                if (metrics.comp_pos_error(curr_q[0:3], param.target_p_) < success_pos_threshold) \
                        and (metrics.comp_quat_error(curr_q[3:7], param.target_q_) < success_quat_threshold):
                    consecutive_success_time = consecutive_success_time + 1
                else:
                    consecutive_success_time = 0

                if consecutive_success_time > consecutive_success_time_threshold:
                    break

        env.close()

        if save_flag:
            save_data.update(target_obj_pos=param.target_p_)
            save_data.update(target_obj_quat=param.target_q_)
            save_data.update(rollout_traj=np.array(rollout_q_traj))
            if rollout_step < max_rollout_length:
                save_data.update(success=True)
            else:
                save_data.update(success=False)
            metrics.save_data(
                save_data,
                data_name=prefix_data_name + 'trial_' + str(trial_count) + '_rollout',
                save_dir=save_dir,
            )

        success_rate = success_rate + (1 if rollout_step < max_rollout_length else 0)
    print(
        f"Success rate over {trial_num} trials "
        f"(trial ids {trial_start} to {trial_stop - 1}): "
        f"{success_rate}/{trial_num} = {success_rate/trial_num:.2%}"
    )


if __name__ == '__main__':
    main()
