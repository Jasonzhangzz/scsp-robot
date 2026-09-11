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
sys.path.append(parent_dir)

from examples.mpc.franka.ik2.params import ExplicitMPCParams
from planning.MPPIExplicit import (
    MPPIExplicit,
    _franka_fk_T_jax,
    _franka_jacobian_pos_jax,
    _tangent_basis_from_normal,
    _contact_jacobian,
)
from planning.mpc_implicit import MPCImplicit
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

        texture_applied = self._apply_object_surface_texture(repo_root, self.obj_actor)
        if not texture_applied:
            self.gym.set_rigid_body_color(self.env, self.obj_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.6, 1.0))
        self.gym.set_rigid_body_color(
            self.env, self.target_obj_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.9, 0.9, 0.9)
        )

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

            # PhysX separation is negative on penetration; keep this convention.
            separation = self._contact_field(c, "separation", "distance", default=0.0)
            separation = float(separation if separation is not None else 0.0)

            normal_raw = self._contact_field(c, "normal", default=np.array([0.0, 0.0, 1.0], dtype=np.float32))
            normal = _extract_vec3(normal_raw)
            nrm = np.linalg.norm(normal)
            if nrm > 1e-8:
                normal = normal / nrm
            else:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            # Many Isaac builds expose local contact points only.
            # For our Jacobian approximation, world contact point is optional.
            pos = None
            pos_field = self._contact_field(c, "pos", "position", default=None)
            if pos_field is not None:
                pos = _extract_vec3(pos_field)

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

    def show_point(self, goal_pos=None):
        return
        
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
        contacts = simulator.get_physx_contacts()

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
            if other_is_franka and row_idx < max_ncon:
                phi_vec[4 * row_idx: 4 * row_idx + 4] = 0.5 * sep
                jac_mat[4 * row_idx: 4 * row_idx + 4, :] = con_jac
                row_idx += 1

            # Object-table contact contributes both to main set and env jacobian set.
            other_is_table = (b0 == simulator.table_body_idx) or (b1 == simulator.table_body_idx)
            if other_is_table and row_idx < max_ncon:
                phi_vec[4 * row_idx: 4 * row_idx + 4] = 0.5 * sep
                jac_mat[4 * row_idx: 4 * row_idx + 4, :] = con_jac
                row_idx += 1

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

        return phi_vec, jac_mat, con_pos_list, jac_mat_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, default='elephant', help='name of obj')
    parser.add_argument('--use-xml-texture', action='store_true', help='apply object texture parsed from env_fingertips_*.xml')
    parser.add_argument('--attract_coef', type=float, default=0.5, help='coef of attract function')
    parser.add_argument('--reject_coef', type=float, default=0.001, help='coef of reject function')
    parser.add_argument('--contact_coef', type=float, default=0.5, help='coef of contact function')
    parser.add_argument('--contact_cost_param', type=float, default=1, help='mass center or project point attract')
    parser.add_argument('--model_param', type=float, default=7, help='model param')
    parser.add_argument('--reject_dis', type=float, default=0.01, help='reject radius')
    parser.add_argument('--attract_point_comp', type=float, default=0.05, help='distance compensation of attract point')
    parser.add_argument('--ground_height_threshold', type=float, default=0.012, help='threshold of sample points height')
    parser.add_argument('--sample_num', type=int, default=70, help='number of sample point')
    parser.add_argument('--pos_coef', type=float, default=1, help='coef of position cost in mlqp_point')
    parser.add_argument('--ori_coef', type=float, default=0.001, help='coef of orientation cost in mlqp_point')
    parser.add_argument('--low_err_coef', type=float, default=0.1, help='coef of delta error')
    parser.add_argument('--upper_err_coef', type=float, default=1, help='coef of delta error')
    parser.add_argument('--mppi_w_ee_ori', type=float, default=10.0, help='weight for EE orientation hold cost')
    parser.add_argument('--mppi_w_joint_limit', type=float, default=5.0, help='weight for joint limit penalty')
    parser.add_argument('--mppi_w_manip', type=float, default=0.1, help='weight for manipulability penalty')
    parser.add_argument('--mppi_w_cond', type=float, default=0.0, help='weight for Jacobian condition penalty')
    parser.add_argument('--mppi_w_energy', type=float, default=0.01, help='weight for action energy penalty')
    parser.add_argument('--mppi_w_vel', type=float, default=0.01, help='weight for joint velocity penalty')
    parser.add_argument('--mppi_w_acc', type=float, default=0.001, help='weight for joint acceleration penalty')

    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--sim-device', type=str, default='cuda:0')
    parser.add_argument('--graphics-device-id', type=int, default=0)
    parser.add_argument(
        '--show-ghost-object',
        type=_parse_bool_arg,
        default=False,
        help='whether to render the semi-transparent ghost object mesh at the target pose (true/false)',
    )

    args = parser.parse_args()

    save_flag = False
    if save_flag:
        save_dir = './examples/mpc/trifinger/elephant/save/'
        prefix_data_name = 'ours_'
        save_data = dict()

    trial_num = 20
    success_pos_threshold = 0.02
    success_quat_threshold = 0.04
    consecutive_success_time_threshold = 20
    max_rollout_length = 5000
    success_rate = 0

    trial_count = 0
    while trial_count < trial_num:
        param = ExplicitMPCParams(args, rand_seed=trial_count, target_type='rotation', mpc_model='explicit')
        param.use_jax_contact_ = False
        param.show_ghost_object_ = bool(args.show_ghost_object)

        contact = ContactIsaac(param)
        env = IsaacFrankaSimulator(
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

        rollout_q_traj = []
        while rollout_step < max_rollout_length:
            if not env.dyn_paused_:
                curr_q = env.get_state()
                rollout_q_traj.append(curr_q)

                phi_vec, jac_mat, con_point, jac_mat_env = contact.detect_once(env)
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
                p_arm_world = R_obj_to_world @ p_arm_local + curr_q[:3]

                if verify_cost:
                    low_err_coef = args.low_err_coef
                else:
                    if np.linalg.norm(ee_pos - attract_point_world) < 5e-2:
                        low_err_coef *= 1.1
                upper_err_coef = max(args.upper_err_coef if not verify_cost else upper_err_coef - 0.002, 0.7)

                delta_error = max_error - min_error
                adaptive = delta_error * upper_err_coef if verify_cost else delta_error * low_err_coef
                verify_cost = 1 if error < (min_error + adaptive) else 0
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
                    contact_point=p_arm_world,
                    curr_ori_coef=curr_ori_coef,
                    sol_guess=param.sol_guess_,
                )
                param.sol_guess_ = sol['sol_guess']
                action = sol['action']
                # env.draw_rollout_lines(sol.get("rollout_q", None))
                print("time_cost = ", time.time() - st, action.shape)
                print("attract_point_world = ", attract_point_world)

                quat = Rotation.from_matrix(env.get_R()).as_quat()  # xyzw
                env.show_target(goal_pos=attract_point_world, goal_quat=quat)
                env.step(action)

                rollout_step = rollout_step + 1

                curr_q = env.get_state()
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
        trial_count = trial_count + 1

    print(f"Success rate over {trial_num} trials: {success_rate}/{trial_num} = {success_rate/trial_num:.2%}")


if __name__ == '__main__':
    main()
