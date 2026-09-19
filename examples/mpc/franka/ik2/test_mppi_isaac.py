import argparse
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__" and "--sync-planner" not in sys.argv:
    os.environ.setdefault("SCSP_PLANNER_ONLY", "1")

if os.environ.get("SCSP_PLANNER_ONLY") == "1":
    gymapi = None
    gymtorch = None
else:
    from isaacgym import gymapi, gymtorch

DEFAULT_CARTESIAN_STIFFNESS = np.array([500.0, 500.0, 500.0, 50.0, 50.0, 50.0], dtype=np.float32)
POSE_AXIS_LENGTH = 0.08
POSE_AXIS_RADIUS = 0.003
POSE_AXIS_CENTER_RADIUS = 0.009
HIDDEN_GHOST_POSITION = np.array([0.0, 0.0, -10.0], dtype=np.float32)


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(current_dir)
while os.path.basename(parent_dir) != "scsp-robot":
    _next_dir = os.path.dirname(parent_dir)
    if _next_dir == parent_dir:
        raise RuntimeError("scsp-robot repo root not found from %s" % current_dir)
    parent_dir = _next_dir
# Put the repository ahead of third-party packages named ``examples``.  Isaac
# Gym/PyTorch may import such a package before this script reaches this point.
sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir) != parent_dir]
sys.path.insert(0, parent_dir)
from planning.acados_env import ensure_acados_env
ensure_acados_env()
loaded_examples = sys.modules.get("examples")
loaded_examples_file = getattr(loaded_examples, "__file__", "") if loaded_examples else ""
if loaded_examples and not os.path.abspath(loaded_examples_file).startswith(parent_dir):
    for module_name in list(sys.modules):
        if module_name == "examples" or module_name.startswith("examples."):
            del sys.modules[module_name]

from examples.mpc.franka.ik2.params import ExplicitMPCParams
from examples.mpc.fingertips.test.test_0902 import (
    add_rollout_via_args,
    _lambda_pose_cost,
    _predicted_object_pose,
    _x_plus_is_usable,
)
from planning.MPPIExplicit import _franka_fk_T_jax
from examples.mpc.franka.ik2.contact_frames import (
    _contact_jacobian_np as _contact_jacobian,
    tangent_basis_from_normal as _tangent_basis_from_normal,
)
from utils import metrics


DYWA_SIM_DT = 0.0125
DYWA_SIM_SUBSTEPS = 1
DYWA_PHYSX_SOLVER_TYPE = 1
DYWA_PHYSX_POSITION_ITERATIONS = 8
DYWA_PHYSX_VELOCITY_ITERATIONS = 1
DYWA_PHYSX_CONTACT_OFFSET = 0.001
DYWA_PHYSX_REST_OFFSET = 0.0
DYWA_PHYSX_FRICTION_OFFSET_THRESHOLD = 0.001
DYWA_PHYSX_FRICTION_CORRELATION_DISTANCE = 0.0005
DYWA_PHYSX_MAX_DEPENETRATION_VELOCITY = 0.25
DYWA_TABLE_FRICTION_RANGE = (0.3, 0.8)
DYWA_OBJECT_FRICTION_RANGE = (0.2, 1.0)
DYWA_OBJECT_MASS_RANGE = (0.1, 0.5)


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


def _set_actor_friction(gym, env, actor_handle, friction):
    shape_props = gym.get_actor_rigid_shape_properties(env, actor_handle)
    for prop in shape_props:
        prop.friction = float(friction)
        prop.torsion_friction = float(friction)
        prop.rolling_friction = float(friction)
    gym.set_actor_rigid_shape_properties(env, actor_handle, shape_props)


def _set_actor_mass(gym, env, actor_handle, mass):
    body_props = gym.get_actor_rigid_body_properties(env, actor_handle)
    for prop in body_props:
        prop.mass = float(mass)
    gym.set_actor_rigid_body_properties(env, actor_handle, body_props, True)


def _apply_dywa_physics_to_param(param):
    param.sim_dt_ = DYWA_SIM_DT
    param.sim_substeps_ = DYWA_SIM_SUBSTEPS
    param.physx_solver_type_ = DYWA_PHYSX_SOLVER_TYPE
    param.physx_position_iterations_ = DYWA_PHYSX_POSITION_ITERATIONS
    param.physx_velocity_iterations_ = DYWA_PHYSX_VELOCITY_ITERATIONS
    param.physx_contact_offset_ = DYWA_PHYSX_CONTACT_OFFSET
    param.physx_rest_offset_ = DYWA_PHYSX_REST_OFFSET
    param.physx_friction_offset_threshold_ = DYWA_PHYSX_FRICTION_OFFSET_THRESHOLD
    param.physx_friction_correlation_distance_ = DYWA_PHYSX_FRICTION_CORRELATION_DISTANCE
    param.physx_max_depenetration_velocity_ = DYWA_PHYSX_MAX_DEPENETRATION_VELOCITY
    param.table_friction_ = float(np.random.uniform(*DYWA_TABLE_FRICTION_RANGE))
    param.object_friction_ = float(np.random.uniform(*DYWA_OBJECT_FRICTION_RANGE))
    param.obj_mass_ = float(np.random.uniform(*DYWA_OBJECT_MASS_RANGE))
    param.mu_object_ = float(param.object_friction_)
    param.gravity_[2] = -9.81
    return param


def _mjcf_quat_wxyz_to_urdf_rpy(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


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
        sim_params.dt = float(getattr(self.param_, "sim_dt_", 0.01))
        sim_params.substeps = int(getattr(self.param_, "sim_substeps_", 2))
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # This script uses CPU-style state APIs (get/set_actor_rigid_body_states),
        # so disable GPU pipeline to avoid invalid resource handle errors.
        sim_params.use_gpu_pipeline = False
        sim_params.physx.solver_type = int(getattr(self.param_, "physx_solver_type_", 1))
        sim_params.physx.num_position_iterations = int(
            getattr(self.param_, "physx_position_iterations_", 8)
        )
        sim_params.physx.num_velocity_iterations = int(
            getattr(self.param_, "physx_velocity_iterations_", 1)
        )
        sim_params.physx.contact_offset = float(getattr(self.param_, "physx_contact_offset_", 0.01))
        sim_params.physx.rest_offset = float(getattr(self.param_, "physx_rest_offset_", 0.0))
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
        visual_mesh_path = getattr(self.param_, "visual_mesh_path_", mesh_path)
        print("mesh_path = ", mesh_path, "visual_mesh_path = ", visual_mesh_path)
        mesh_obj_urdf_rel, mesh_target_urdf_rel, mesh_asset_root = self._prepare_mesh_urdf_assets(
            repo_root,
            mesh_path,
            visual_mesh_path,
            mass=float(getattr(self.param_, "sim_obj_mass_", getattr(self.param_, "obj_mass_", 0.01))),
            inertia_diag=getattr(self.param_, "sim_obj_inertia_diag_", None),
        )
        obj_opts = gymapi.AssetOptions()
        obj_opts.density = 200.0
        obj_opts.use_mesh_materials = True
        if hasattr(obj_opts, "vhacd_enabled"):
            obj_opts.vhacd_enabled = False
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

        self._apply_optional_scene_physics()
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
    def _prepare_mesh_urdf_assets(repo_root, mesh_path, visual_mesh_path=None, mass=0.01, inertia_diag=None):
        if mesh_path is None:
            raise ValueError("param.mesh_path_ is required for Isaac object asset loading")

        def _abs_mesh(path):
            abs_path = path if os.path.isabs(path) else os.path.join(repo_root, path)
            abs_path = os.path.abspath(abs_path)
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"Mesh file not found: {abs_path}")
            return abs_path

        collision_abs = _abs_mesh(mesh_path)
        visual_abs = _abs_mesh(visual_mesh_path or mesh_path)

        mesh_asset_root = os.path.join(repo_root, "envs", "assets", "objects", "_isaac_tmp")
        os.makedirs(mesh_asset_root, exist_ok=True)
        visual_rel = "obj_mesh_visual.stl"
        collision_rel = "obj_mesh_collision.stl"
        shutil.copy2(visual_abs, os.path.join(mesh_asset_root, visual_rel))
        shutil.copy2(collision_abs, os.path.join(mesh_asset_root, collision_rel))
        mesh_scale = "1 1 1"
        obj_urdf_rel = "obj_mesh_dynamic.urdf"
        obj_urdf_abs = os.path.join(mesh_asset_root, obj_urdf_rel)
        target_urdf_rel = "obj_mesh_target_ghost.urdf"
        target_urdf_abs = os.path.join(mesh_asset_root, target_urdf_rel)
        if inertia_diag is None:
            inertia_diag = (1.5e-6, 1.5e-6, 1.5e-6)
        ixx, iyy, izz = [float(v) for v in np.asarray(inertia_diag, dtype=np.float64).reshape(3)]
        mass = float(mass)

        obj_urdf = f"""<?xml version="1.0"?>
<robot name="mesh_obj">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{visual_rel}" scale="{mesh_scale}"/>
      </geometry>
      <material name="obj_color">
        <color rgba="0.2 0.6 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{collision_rel}" scale="{mesh_scale}"/>
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
        <mesh filename="{visual_rel}" scale="{mesh_scale}"/>
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

    def _apply_optional_scene_physics(self):
        if not hasattr(self, "table_actor") or not hasattr(self, "obj_actor"):
            return
        if hasattr(self.param_, "table_friction_"):
            _set_actor_friction(self.gym, self.env, self.table_actor, float(self.param_.table_friction_))
        if hasattr(self.param_, "object_friction_"):
            _set_actor_friction(self.gym, self.env, self.obj_actor, float(self.param_.object_friction_))
        if hasattr(self.param_, "obj_mass_"):
            _set_actor_mass(self.gym, self.env, self.obj_actor, float(self.param_.obj_mass_))

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

    def _contact_bodies(self, c):
        body0 = self._contact_field(c, "body0", "bodyA", default=-1)
        body1 = self._contact_field(c, "body1", "bodyB", default=-1)
        if body0 is None or body1 is None:
            return None, None
        return int(body0), int(body1)

    def _contact_sep_normal(self, c):
        # Isaac Gym's RigidContact has no ``separation`` field.
        # ``initial_overlap`` is positive for penetration.  overlap=0 is
        # either a made touch or the 1 mm contact_offset pair; callers
        # replace this placeholder with the fingertip-sphere signed gap.
        contact_offset = float(getattr(self.param_, "physx_contact_offset_", 0.001))
        separation_raw = self._contact_field(c, "separation", "distance", "minDist", default=None)
        speculative = False
        if separation_raw is None:
            initial_overlap = self._contact_field(c, "initial_overlap", default=0.0)
            overlap = float(initial_overlap if initial_overlap is not None else 0.0)
            if overlap > 1e-8:
                separation = -overlap
            else:
                separation = contact_offset
                speculative = True
        else:
            separation = float(separation_raw)
            speculative = separation > contact_offset + 1e-8
        normal_raw = self._contact_field(c, "normal", default=np.array([0.0, 0.0, 1.0], dtype=np.float32))
        normal = _extract_vec3(normal_raw)
        nrm = float(np.linalg.norm(normal))
        if nrm > 1e-8:
            normal = normal / nrm
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return float(separation), bool(speculative), normal

    def _contact_world_pos(self, c, body0, body1, pose_cache):
        pos_field = self._contact_field(c, "pos", "position", default=None)
        if pos_field is not None:
            return _extract_vec3(pos_field)
        local_pos0 = self._contact_field(c, "localPos0", "local_pos0", default=None)
        local_pos1 = self._contact_field(c, "localPos1", "local_pos1", default=None)
        world_points = []
        for body_idx, local_pos in ((body0, local_pos0), (body1, local_pos1)):
            if local_pos is None:
                continue
            if body_idx not in pose_cache:
                pose_cache[body_idx] = self.get_body_pose_by_sim_index(body_idx)
            body_pos, body_quat_xyzw = pose_cache[body_idx]
            if body_pos is None:
                continue
            body_rot = Rotation.from_quat(body_quat_xyzw).as_matrix()
            world_points.append(body_pos + body_rot @ _extract_vec3(local_pos))
        if world_points:
            return np.mean(np.stack(world_points, axis=0), axis=0).astype(np.float32)
        return None

    def get_physx_contacts(self, need_world_pos=True):
        contacts_raw = self.gym.get_env_rigid_contacts(self.env)
        contacts = []
        if contacts_raw is None:
            return contacts
        pose_cache = {}
        for c in contacts_raw:
            body0, body1 = self._contact_bodies(c)
            if body0 is None:
                continue
            separation, speculative, normal = self._contact_sep_normal(c)
            pos = None
            if need_world_pos:
                pos = self._contact_world_pos(c, body0, body1, pose_cache)
            contacts.append(
                {
                    "body0": body0,
                    "body1": body1,
                    "separation": separation,
                    "normal": normal,
                    "pos": pos,
                    "speculative": bool(speculative),
                }
            )
        return contacts

    def fingertip_object_contact(self):
        """Fingertip/object pair with a MuJoCo-style signed sphere gap."""
        from examples.mpc.franka.ik2.physx_contact import (
            contact_overlap,
            end_effector_position,
            fingertip_body_index,
            fingertip_radius,
            physx_signed_gap,
        )

        fingertip_sim_idx = fingertip_body_index(self)
        obj_idx = getattr(self, "obj_body_idx", None)
        if fingertip_sim_idx is None or obj_idx is None:
            return False, None, float("inf")
        contacts_raw = self.gym.get_env_rigid_contacts(self.env)
        if contacts_raw is None:
            return False, None, float("inf")
        tip_pos = end_effector_position(self, fallback=None)
        radius = fingertip_radius(self)
        in_contact = False
        best_n = None
        best_sep = float("inf")
        pose_cache = {}
        for c in contacts_raw:
            body0, body1 = self._contact_bodies(c)
            if body0 is None:
                continue
            if fingertip_sim_idx not in (body0, body1) or obj_idx not in (body0, body1):
                continue
            _sep, _speculative, normal = self._contact_sep_normal(c)
            if body1 == obj_idx:
                normal = -normal
            cpos = self._contact_world_pos(c, body0, body1, pose_cache)
            dist = physx_signed_gap(contact_overlap(c), cpos, tip_pos, normal, radius)
            if dist < best_sep:
                best_sep = dist
                best_n = np.asarray(normal, dtype=np.float32)
            if dist <= 0.0:
                in_contact = True
        return in_contact, best_n, best_sep

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


class IsaacFrankaJointSimulator(IsaacFrankaSimulator):
    """Position-controlled no-hand Franka with the same fingertip sphere as the OSC branch."""

    @staticmethod
    def _get_franka_asset_info(repo_root):
        asset_root = os.path.join(repo_root, "envs/robots/assets/urdf")
        src_urdf = os.path.join(asset_root, "franka_description", "robots", "franka_panda.urdf")
        dst_rel = os.path.join("franka_description", "robots", "franka_panda_nohand_sphere_tmp.urdf")
        dst_urdf = os.path.join(asset_root, dst_rel)
        attachment_rpy = _mjcf_quat_wxyz_to_urdf_rpy([0.3826834, 0.0, 0.0, 0.9238795])

        with open(src_urdf, "r", encoding="ascii") as f:
            urdf_text = f.read()

        strip_patterns = [
            r'\s*<joint name="panda_hand_joint"[\s\S]*?</joint>',
            r'\s*<link name="panda_hand">[\s\S]*?</link>',
            r'\s*<link name="panda_leftfinger">[\s\S]*?</link>',
            r'\s*<link name="panda_rightfinger">[\s\S]*?</link>',
            r'\s*<joint name="panda_finger_joint1"[\s\S]*?</joint>',
            r'\s*<joint name="panda_finger_joint2"[\s\S]*?</joint>',
        ]
        for pattern in strip_patterns:
            urdf_text = re.sub(pattern, "\n", urdf_text, flags=re.MULTILINE)

        attachment_block = f"""
  <link name="attachment">
    <visual>
      <origin xyz="0 0 0.03" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.005" length="0.06"/>
      </geometry>
      <material name="attachment_dark">
        <color rgba="0.1 0.1 0.1 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.03" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.005" length="0.06"/>
      </geometry>
    </collision>
  </link>
  <joint name="attachment_joint" type="fixed">
    <parent link="panda_link7"/>
    <child link="attachment"/>
    <origin xyz="0 0 0.107" rpy="{attachment_rpy[0]} {attachment_rpy[1]} {attachment_rpy[2]}"/>
  </joint>
  <link name="fingertip">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.01"/>
      </geometry>
      <material name="fingertip_red">
        <color rgba="0.8 0.2 0.2 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.01"/>
      </geometry>
    </collision>
  </link>
  <joint name="fingertip_joint" type="fixed">
    <parent link="attachment"/>
    <child link="fingertip"/>
    <origin xyz="0 0 0.06" rpy="0 0 0"/>
  </joint>
"""
        urdf_text = urdf_text.replace("</robot>", attachment_block + "\n</robot>")
        with open(dst_urdf, "w", encoding="ascii") as f:
            f.write(urdf_text)
        return asset_root, dst_rel

    def _create_scene_actors(self):
        super()._create_scene_actors()
        p_arm_marker_pose = gymapi.Transform()
        p_arm_marker_pose.p = gymapi.Vec3(0.3, 0.0, 0.4)
        p_arm_marker_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.p_arm_marker_actor = self.gym.create_actor(
            self.env, self.marker_asset, p_arm_marker_pose, "p_arm_marker", 0, 0
        )
        self.gym.set_rigid_body_color(
            self.env, self.p_arm_marker_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.9, 0.1)
        )
        best_marker_pose = gymapi.Transform()
        best_marker_pose.p = gymapi.Vec3(0.3, 0.05, 0.4)
        best_marker_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.best_contact_marker_actor = self.gym.create_actor(
            self.env, self.marker_asset, best_marker_pose, "best_contact_marker", 0, 0
        )
        self.gym.set_rigid_body_color(
            self.env, self.best_contact_marker_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.1, 0.1)
        )

    def _build_body_index_cache(self):
        super()._build_body_index_cache()
        self.fingertip_body_idx = (
            self.franka_body_names.index("fingertip") if "fingertip" in self.franka_body_names else 0
        )
        self.task_body_local_idx = self.fingertip_body_idx

    def get_end_effector_pos(self):
        p, q_xyzw = self._get_actor_body_pose(self.franka_actor, self.task_body_local_idx)
        r = Rotation.from_quat(q_xyzw).as_matrix().astype(np.float32)
        return p.astype(np.float32), r

    def get_policy_state(self):
        full_q = self.get_state()
        ee_pos, _ = self.get_end_effector_pos()
        return np.hstack([full_q[:7], ee_pos]).astype(np.float32)

    def _set_marker_pos(self, actor, goal_pos=None):
        marker_state = self.gym.get_actor_rigid_body_states(self.env, actor, gymapi.STATE_ALL)
        if goal_pos is not None:
            marker_state["pose"]["p"][0] = (float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]))
        marker_state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
        marker_state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
        self.gym.set_actor_rigid_body_states(self.env, actor, marker_state, gymapi.STATE_ALL)

    def show_point(self, goal_pos=None):
        self._set_marker_pos(self.p_arm_marker_actor, goal_pos)

    def show_target(self, goal_pos=None, goal_quat=None):
        del goal_quat
        self.show_point(goal_pos)

    def show_best_contact(self, goal_pos=None):
        self._set_marker_pos(self.best_contact_marker_actor, goal_pos)

    def step_joint_delta(self, dq):
        q = self.get_current_joint_position()
        dq = np.asarray(dq, dtype=np.float32).reshape(7)
        q_des = q + dq
        self._joint_targets[:7] = q_des
        if self.franka_dof_count >= 9:
            self._joint_targets[7] = 0.04
            self._joint_targets[8] = 0.04
        self.gym.set_actor_dof_position_targets(self.env, self.franka_actor, self._joint_targets)
        control_substeps = max(int(getattr(self.param_, "control_substeps_", 1)), 1)
        for _ in range(control_substeps):
            self._simulate_once()


class ContactIsaac:
    """Isaac replacement for contact.franka_collision_detection2.Contact."""

    def __init__(self, param):
        self.param_ = param

    @staticmethod
    def _contact_jacobian_body_frame(jacobian, body_mat):
        """Match contact.fingertips_collision_detection2.Contact."""
        jacobian = np.asarray(jacobian, dtype=np.float64).copy()
        if jacobian.ndim != 2 or jacobian.shape[1] < 6:
            return jacobian
        rot = np.asarray(body_mat, dtype=np.float64).reshape(3, 3)
        frame = np.zeros((6, 6), dtype=np.float64)
        frame[:3, :3] = rot
        frame[3:, 3:] = rot
        jacobian[:, :6] = jacobian[:, :6] @ frame
        return jacobian

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
        quat_xyzw = np.array(
            [obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]],
            dtype=np.float32,
        )
        r_obj_to_world = Rotation.from_quat(quat_xyzw).as_matrix()

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
                jac_mat_env[4 * row_env_idx: 4 * row_env_idx + 4, :6] = self._contact_jacobian_body_frame(
                    con_jac[:, :6], r_obj_to_world
                )
                row_env_idx += 1
                con_pos_local = r_obj_to_world.T @ (cpos - obj_pos)
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
                jac_mat_env[0:4, :6] = self._contact_jacobian_body_frame(
                    con_jac_table[:, :6], r_obj_to_world
                )
                con_pos_list.append(np.array([0.0, 0.0, -0.025], dtype=np.float32))

        return phi_vec, jac_mat, con_pos_list, jac_mat_env, if_contact


def adapt_param_for_joint_mppi(param, args):
    """MPPI joint-space model.  Does not overwrite cartesian ranking dims."""
    from planning.MPPIWarp import adapt_param_for_joint_mppi as _adapt
    return _adapt(param, args)


class _ConfView:
    def __init__(self, tightness, accum):
        self.accum = float(accum)
        self._tightness = float(tightness)

    def tightness(self):
        return self._tightness


def _plan_payload(env, contact, table_ground, dwell=None):
    curr_q = env.get_policy_state()
    full_q = env.get_state()
    phi_vec, jac_mat, _con, jac_mat_env, if_contact = contact.detect_once(env)
    payload = {
        "cmd": "plan",
        "policy_q": np.asarray(curr_q, dtype=np.float32),
        "full_q": np.asarray(full_q, dtype=np.float32),
        "phi_vec": np.asarray(phi_vec, dtype=np.float64),
        "jac_mat": np.asarray(jac_mat, dtype=np.float64),
        "jac_mat_env": np.asarray(jac_mat_env, dtype=np.float64),
        "if_contact": bool(if_contact),
        "table_ground": float(table_ground),
    }
    if dwell is not None:
        payload["dwell"] = dwell
    measured = contact.get_actual_fingertip_contact()
    if measured is None:
        contact_distance = float("inf")
    else:
        contact_distance = float(measured.get("dist", float("inf")))
    return payload, curr_q, if_contact, contact_distance


def _apply_plan_result(param, env, result, curr_q, print_fn):
    from planning.MPPIWarp import _apply_opt_snapshot

    _apply_opt_snapshot(param.lambda_optimizer, result.get("opt_snapshot"))
    policy = result["policy"]
    print_fn(
        param, curr_q, policy, policy["value_info"], result["verify_cost"],
        result["verify_chatter"],
        _ConfView(result["model_tightness"], result["model_accum"]),
        None, None, result.get("if_contact", False), policy["escape_on"],
    )
    print(
        "rank_dt:", round(float(result["rank_dt"]), 4),
        "plan_dt:", round(float(result.get("plan_dt", result.get("mppi_dt", 0.0))), 4),
    )
    if env is not None:
        env.show_target(policy["mpc_virtual_point"])
        env.show_best_contact(policy["best_contact_world"])
    return policy


def _plan_from_obs(obs, dwell=None):
    payload = {
        "cmd": "plan",
        "policy_q": np.asarray(obs["policy_q"], dtype=np.float32),
        "full_q": np.asarray(obs.get("full_q", obs["policy_q"]), dtype=np.float32),
        "if_contact": bool(obs.get("if_contact", False)),
        "table_ground": float(obs["table_ground"]),
    }
    for key in ("phi_vec", "jac_mat", "jac_mat_env"):
        if obs.get(key) is not None:
            payload[key] = np.asarray(obs[key], dtype=np.float64)
    if dwell is not None:
        payload["dwell"] = dwell
    return payload, np.asarray(obs["policy_q"], dtype=np.float32)


def _pred_reduction_from_policy(param, curr_q, policy):
    c_now_cost = _lambda_pose_cost(
        curr_q[:3], curr_q[3:7], param.target_p_, param.target_q_,
        param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef,
    )
    pred_reduction = None
    x_plus_opt = policy.get("x_plus_opt")
    info = policy.get("info")
    if x_plus_opt is not None and _x_plus_is_usable(x_plus_opt, info):
        pred_pos, pred_quat = _predicted_object_pose(curr_q[:7], x_plus_opt)
        c_pred = _lambda_pose_cost(
            pred_pos, pred_quat, param.target_p_, param.target_q_,
            param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef,
        )
        pred_reduction = c_now_cost - c_pred
    return c_now_cost, pred_reduction


def _dwell_payload(
    env, args, contact, policy, last_accept_p_arm, escape_on, verify_chatter,
    c_now_cost, pred_reduction, contact_distance, pos_err_now,
    curr_q=None, param=None,
):
    if param is None:
        param = env.param_
    if curr_q is None:
        curr_q = env.get_policy_state()
    r_now = Rotation.from_quat([curr_q[4], curr_q[5], curr_q[6], curr_q[3]]).as_matrix()
    c_after = _lambda_pose_cost(
        curr_q[:3], curr_q[3:7], param.target_p_, param.target_q_,
        param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef,
    )
    return {
        "tip_now": np.asarray(curr_q[7:10], dtype=float),
        "obj_pos": np.asarray(curr_q[:3], dtype=float),
        "r_now": np.asarray(r_now, dtype=float),
        "p_arm_world": np.asarray(policy["p_arm_world"], dtype=float),
        "post_physical": bool(np.isfinite(contact_distance) and contact_distance <= 0.003),
        "last_accept_p_arm": bool(last_accept_p_arm),
        "escape_on": bool(escape_on),
        "verify_chatter": bool(verify_chatter),
        "pos_err_now": float(pos_err_now),
        "c_now_cost": None if c_now_cost is None else float(c_now_cost),
        "pred_reduction": None if pred_reduction is None else float(pred_reduction),
        "c_after": float(c_after),
    }


def _publish_then_sync(env, planner, payload):
    """Publish latest state, wait on the Isaac clock, then take a ready action."""
    if planner is not None:
        planner.publish_state(payload)
    env.sync_realtime()
    if planner is None:
        return None
    return planner.take_action()


def _configure_rollout_param(param, args):
    param.rollout_press_patch = True
    param.quadratic_contact_track = True
    param.attract_coef = max(float(param.attract_coef), 20.0)
    param.field_cost_weight = 0.0
    param.contact_coef = max(float(param.contact_coef), float(param.attract_coef))
    param.lambda_optimizer.lock_contact_patch = False
    param.lambda_optimizer.contact_switch_confirm_steps = max(
        1, int(args.contact_switch_confirm_steps)
    )
    return param


def _add_mppi_policy_args(parser):
    add_rollout_via_args(parser)
    parser.add_argument("--use-xml-texture", action="store_true")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--viewer", dest="headless", action="store_false")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--show-ghost-object", type=_parse_bool_arg, default=False)
    parser.add_argument("--cartesian-joint-stiffness", type=float, default=100.0)
    parser.add_argument("--osc-pos-stiffness", type=float, default=12000.0)
    parser.add_argument("--osc-ori-stiffness", type=float, default=0.0)
    parser.add_argument("--nullspace-stiffness", type=float, default=10.0)
    parser.add_argument("--svg-screenshot-dir", type=str, default="")
    parser.add_argument("--svg-screenshot-interval", type=float, default=0.2)
    parser.add_argument("--svg-screenshot-width", type=int, default=1280)
    parser.add_argument("--svg-screenshot-height", type=int, default=960)
    parser.add_argument("--cartesian_stiffness", type=float, nargs="+", default=None)
    parser.add_argument("--cartesian_damping", type=float, nargs="+", default=None)
    parser.add_argument("--effort-joint-damping", type=float, default=0.0)
    parser.add_argument(
        "--target-type",
        dest="target_type",
        type=str,
        default="ground-rotation",
        choices=("ground-rotation", "rotation"),
        help="ground-rotation matches fingertips --rollout (90 deg pitch flip).",
    )
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument(
        "--control-substeps",
        type=int,
        default=0,
        help="OSC frames per MPPI joint update.  0 uses the 20 ms --rollout interval.",
    )
    parser.add_argument(
        "--joint-step",
        type=float,
        default=0.02,
        help="MPPI joint-delta limit per 20 ms (rad).  Smaller is smoother.",
    )
    parser.add_argument(
        "--action-smooth",
        type=float,
        default=0.35,
        help="EMA blend of the executed joint action.  1 is raw MPPI.",
    )
    parser.add_argument(
        "--via-max-lead",
        type=float,
        default=0.008,
        help="Max via/press offset from the current fingertip (m).",
    )
    parser.add_argument(
        "--via-smooth-rate",
        type=float,
        default=0.05,
        help="SmoothedApproachVia lerp rate.  Smaller interpolates more.",
    )
    parser.add_argument("--joint-model-stiffness", type=float, default=300.0)
    parser.add_argument("--mppi_Nsample", type=int, default=256)
    parser.add_argument("--mppi_Hsample", type=int, default=16)
    parser.add_argument("--mppi_Ndiffuse", type=int, default=2)
    parser.add_argument("--mppi_Ndiffuse_init", type=int, default=3)
    parser.add_argument("--mppi_temp_sample", type=float, default=0.5)
    parser.add_argument("--mppi_sigma_scale", type=float, default=0.6)
    parser.add_argument("--mppi_traj_diffuse_factor", type=float, default=0.5)
    parser.add_argument("--mppi_seed", type=int, default=0)
    parser.add_argument(
        "--mppi_w_energy",
        type=float,
        default=50.0,
        help="0902 plan_once control weight on ||u||^2.",
    )
    parser.add_argument("--mppi_w_nopen", type=float, default=80.0)
    parser.add_argument("--mppi_w_joint_limit", type=float, default=100.0)
    parser.add_argument("--mppi_nopen_margin", type=float, default=0.002)
    parser.add_argument(
        "--async-planner",
        dest="async_planner",
        action="store_true",
        help="ROS-style latest-only state/action topics in a second process.",
    )
    parser.add_argument(
        "--sync-planner",
        dest="async_planner",
        action="store_false",
        help="Rank+MPPI in the Isaac process (debug).",
    )
    parser.add_argument("--viewer-hz", type=float, default=60.0)
    parser.add_argument(
        "--gpu-physx",
        action="store_true",
        help="GPU PhysX in the viewer (fights Warp MPPI on the same device).",
    )
    parser.set_defaults(headless=False, async_planner=True)
    return parser


def run_mppi_planner(bus, args):
    from examples.mpc.franka.ik2.isaac_bus import wait_trial_obs
    from examples.mpc.franka.ik2.test_mpc_isaac import (
        DEFAULT_CARTESIAN_STIFFNESS,
        adapt_param_for_cartesian_solver,
        _apply_dywa_physics_to_param as _apply_mpc_physics_to_param,
        _configure_rollout_param as _configure_mpc_rollout_param,
        _print_rollout_step,
    )
    from planning.MPPIWarp import (
        _build_planner_runtime,
        blend_action,
        handle_planner_request,
        planner_init_payload,
    )

    if args.cartesian_stiffness is None:
        args.cartesian_stiffness = DEFAULT_CARTESIAN_STIFFNESS.tolist()
    if args.trial_start < 0:
        raise ValueError(f"trial_start must be non-negative, got {args.trial_start}")
    if int(args.trial_num) <= 0:
        raise ValueError(f"trial_num must be positive, got {args.trial_num}")

    trial_start = int(args.trial_start)
    trial_num = max(1, int(args.trial_num))
    trial_stop = trial_start + trial_num
    success_pos_threshold = 0.02
    success_quat_threshold = 0.015
    consecutive_success_time_threshold = 0
    max_rollout_length = max(1, int(args.max_rollout_length))
    success_rate = 0
    viewer_quit = False

    for trial_count in range(trial_start, trial_stop):
        param = ExplicitMPCParams(
            args,
            rand_seed=trial_count,
            target_type=getattr(args, "target_type", "ground-rotation"),
            mpc_model="explicit",
        )
        param = _apply_mpc_physics_to_param(param)
        param.use_jax_contact_ = False
        param = adapt_param_for_cartesian_solver(param, args)
        param = _configure_mpc_rollout_param(param, args)
        param = adapt_param_for_joint_mppi(param, args)
        param.control_substeps_ = int(args.control_substeps)
        init = planner_init_payload(args, param, trial_count, args.sim_device)
        _, plan_param, mpc, trackers = _build_planner_runtime(init)

        obs = wait_trial_obs(bus, trial_count)
        if obs is None:
            viewer_quit = True
            break

        prev_action = None
        policy = None
        c_now_cost = None
        pred_reduction = None
        last_accept_p_arm = False
        escape_on = False
        verify_chatter = False
        rollout_step = 0
        consecutive_success_time = 0
        min_pos_err = float("inf")
        min_quat_err = float("inf")
        pose_apply_count = 0
        choose_times = []
        pos_err_now = None
        quat_err_now = None
        action = np.zeros(7, dtype=np.float32)

        while rollout_step < max_rollout_length:
            if obs.get("break_out") or obs.get("viewer_closed") or obs.get("cmd") == "stop":
                viewer_quit = True
                break
            curr_q = np.asarray(obs["policy_q"], dtype=np.float32)
            pos_err_now = float(metrics.comp_pos_error(curr_q[0:3], param.target_p_))
            quat_err_now = float(metrics.comp_quat_error(curr_q[3:7], param.target_q_))
            min_pos_err = min(min_pos_err, pos_err_now)
            min_quat_err = min(min_quat_err, quat_err_now)
            if pos_err_now < success_pos_threshold and quat_err_now < success_quat_threshold:
                consecutive_success_time += 1
            else:
                consecutive_success_time = 0
            if consecutive_success_time > consecutive_success_time_threshold:
                break

            contact_distance = float(obs.get("contact_distance", float("inf")))
            dwell = None
            if policy is not None:
                dwell = _dwell_payload(
                    None, args, None, policy, last_accept_p_arm, escape_on,
                    verify_chatter, c_now_cost, pred_reduction, contact_distance,
                    pos_err_now, curr_q=curr_q, param=param,
                )
            payload, curr_q = _plan_from_obs(obs, dwell)
            result = handle_planner_request(args, plan_param, mpc, trackers, payload)
            policy = _apply_plan_result(param, None, result, curr_q, _print_rollout_step)
            choose_times.append(float(result["rank_dt"]))
            c_now_cost, pred_reduction = _pred_reduction_from_policy(param, curr_q, policy)
            last_accept_p_arm = bool(policy["value_info"].get("accept_p_arm", False))
            escape_on = bool(policy["escape_on"])
            verify_chatter = bool(result["verify_chatter"])
            raw = np.asarray(result["action"], dtype=np.float32).reshape(7)
            action = blend_action(prev_action, raw, float(getattr(args, "action_smooth", 0.35)))
            prev_action = action.copy()

            on_exec_contact = False
            if np.isfinite(contact_distance) and contact_distance <= 0.003:
                post_tip = np.asarray(curr_q[7:10], dtype=float)
                on_exec_contact = (
                    float(np.linalg.norm(post_tip - np.asarray(policy["p_arm_world"], dtype=float))) <= 0.03
                )
                if on_exec_contact:
                    pose_apply_count += 1
            print(
                "contact_distance:",
                None if not np.isfinite(contact_distance) else round(contact_distance, 6),
                "physics_contact:", int(on_exec_contact),
                "mppi_action_norm:", round(float(np.linalg.norm(action)), 6),
            )
            bus.publish_cmd({
                "kind": "joint",
                "action": action,
                "markers": {
                    "target": policy["mpc_virtual_point"],
                    "best_contact": policy["best_contact_world"],
                },
            })
            rollout_step += 1
            obs = bus.wait_obs(timeout=180.0)
            if obs is None:
                viewer_quit = True
                break

        bus.publish_cmd({"cmd": "end_trial"})
        choose_arr = np.asarray(choose_times, dtype=np.float64) if choose_times else np.array([0.0])
        print("trial_summary:", {
            "trial": trial_count,
            "mode": "mppi_planner",
            "success": int(rollout_step < max_rollout_length),
            "steps": rollout_step,
            "pose_applies": pose_apply_count,
            "final_pos_err": None if pos_err_now is None else round(float(pos_err_now), 5),
            "final_quat_err": None if quat_err_now is None else round(float(quat_err_now), 5),
            "min_pos_err": None if not np.isfinite(min_pos_err) else round(float(min_pos_err), 5),
            "min_quat_err": None if not np.isfinite(min_quat_err) else round(float(min_quat_err), 5),
            "choose_dt_mean": round(float(np.mean(choose_arr)), 5),
            "choose_dt_max": round(float(np.max(choose_arr)), 5),
            "choose_dt_p95": round(float(np.percentile(choose_arr, 95)), 5),
        })
        success_rate += 1 if rollout_step < max_rollout_length else 0
        if viewer_quit:
            break

    print(
        f"Success rate over {trial_num} trials "
        f"(trial ids {trial_start} to {trial_stop - 1}): "
        f"{success_rate}/{trial_num} = {success_rate / trial_num:.2%}"
    )
    return viewer_quit


def planner_worker(cmd_q, obs_q, ready_q, init):
    from examples.mpc.franka.ik2.isaac_bus import IsaacBus, args_from_init

    if ready_q is not None:
        ready_q.put({"ok": True, "pid": os.getpid()})
    run_mppi_planner(IsaacBus(cmd_q, obs_q), args_from_init(init))


def main():
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
    os.environ.pop("SCSP_PLANNER_ONLY", None)
    os.execv(sys.executable, [sys.executable, script, "--planner", "mppi", *sys.argv[1:]])


if __name__ == "__main__":
    main()
