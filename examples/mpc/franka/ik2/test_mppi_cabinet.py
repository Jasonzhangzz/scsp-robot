import argparse
import importlib.util
import os
import re
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


from isaacgym import gymapi, gymtorch
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(repo_root)

from planning.MPPIExplicit import _contact_jacobian, _franka_fk_T_jax, _franka_jacobian_pos_jax, _tangent_basis_from_normal
from planning.MPPICabinet import MPPICabinet
from planning.mlqp_point import LambdaContactControlOptimizer
from planning.mpc_implicit import MPCImplicit
from utils import rotations

torch_jit_utils_path = os.path.join(
    repo_root,
    "IsaacGymEnvs",
    "isaacgymenvs",
    "utils",
    "torch_jit_utils.py",
)
torch_jit_utils_spec = importlib.util.spec_from_file_location("torch_jit_utils_local", torch_jit_utils_path)
if torch_jit_utils_spec is None or torch_jit_utils_spec.loader is None:
    raise ImportError(f"Failed to load torch_jit_utils from {torch_jit_utils_path}")
torch_jit_utils = importlib.util.module_from_spec(torch_jit_utils_spec)
torch_jit_utils_spec.loader.exec_module(torch_jit_utils)

get_axis_params = torch_jit_utils.get_axis_params
quat_apply = torch_jit_utils.quat_apply
quat_conjugate = torch_jit_utils.quat_conjugate
quat_mul = torch_jit_utils.quat_mul
tf_combine = torch_jit_utils.tf_combine
tf_vector = torch_jit_utils.tf_vector


def to_torch(x, dtype=None, device="cpu"):
    import torch

    if dtype is None:
        return torch.tensor(x, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


def _quat_xyzw_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def _quat_wxyz_to_xyzw(q_wxyz):
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)


def _mjcf_quat_wxyz_to_urdf_rpy(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=False)


def _skew(v):
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float32,
    )


def compute_grasp_transforms(
    hand_rot,
    hand_pos,
    franka_local_grasp_rot,
    franka_local_grasp_pos,
    drawer_rot,
    drawer_pos,
    drawer_local_grasp_rot,
    drawer_local_grasp_pos,
):
    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot,
        hand_pos,
        franka_local_grasp_rot,
        franka_local_grasp_pos,
    )
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot,
        drawer_pos,
        drawer_local_grasp_rot,
        drawer_local_grasp_pos,
    )
    return global_franka_rot, global_franka_pos, global_drawer_rot, global_drawer_pos


class CabinetMPCParams:
    def __init__(self, args):
        self.contact_cost_param = args.contact_cost_param
        self.attract_coef = args.attract_coef
        self.reject_coef = args.reject_coef
        self.contact_coef = args.contact_coef
        self.reject_dis = args.reject_dis

        self.h_ = 0.05
        self.frame_skip_ = 20

        self.n_robot_qpos_ = 7
        self.n_qpos_ = 14
        self.n_qvel_ = 13
        self.n_cmd_ = 7
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 15

        self.table_height = 0.4
        self.init_robot_qpos_ = np.array([0.0, -1.0, 0.0, -2.2, 0.0, 1.2, 0.8], dtype=np.float32)

        self.target_p_ = np.zeros(3, dtype=np.float32)
        self.target_q_ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.init_obj_qpos_ = np.hstack([self.target_p_, self.target_q_]).astype(np.float32)

        self.mu_object_ = 0.9
        self.obj_inertia_ = np.identity(6, dtype=np.float32)
        self.obj_inertia_[0:3, 0:3] = 20.0 * np.eye(3, dtype=np.float32)
        self.obj_inertia_[3:, 3:] = 0.05 * np.eye(3, dtype=np.float32)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [300.0]).astype(np.float32)

        self.Q = np.zeros((self.n_qvel_, self.n_qvel_), dtype=np.float32)
        self.Q[:6, :6] = self.obj_inertia_
        self.Q[6:, 6:] = self.robot_stiff_

        self.obj_mass_ = 0.2
        self.gravity_ = np.array([0.0, 0.0, -9.8, 0.0, 0.0, 0.0], dtype=np.float32)
        self.model_params = args.model_param
        self.use_jax_contact_ = False
        self.contact_radius_ = 0.02

        self.mpc_horizon_ = 16
        self.ipopt_max_iter_ = 100
        self.mpc_model = "explicit"
        self.mpc_u_lb_ = -0.05
        self.mpc_u_ub_ = 0.05
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), -1e7 * np.ones(7))).astype(np.float32)
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), 1e7 * np.ones(7))).astype(np.float32)

        self.mppi_w_ee_ori_ = getattr(args, "mppi_w_ee_ori", 10.0)
        self.mppi_w_joint_limit_ = getattr(args, "mppi_w_joint_limit", 5.0)
        self.mppi_w_manip_ = getattr(args, "mppi_w_manip", 0.1)
        self.mppi_w_cond_ = getattr(args, "mppi_w_cond", 0.0)
        self.mppi_w_energy_ = getattr(args, "mppi_w_energy", 0.01)
        self.mppi_w_vel_ = getattr(args, "mppi_w_vel", 0.01)
        self.mppi_w_acc_ = getattr(args, "mppi_w_acc", 0.001)

        self.final_cost_mode_ = "drawer_open"
        self.drawer_open_axis_ = 0
        self.drawer_open_weight_ = getattr(args, "drawer_open_weight", 500.0)
        self.drawer_lateral_weight_ = getattr(args, "drawer_lateral_weight", 25.0)
        self.drawer_quat_weight_ = getattr(args, "drawer_quat_weight", 0.0)

        self.sol_guess_ = None
        self.mppi_samples_ = getattr(args, "mppi_samples", 512)
        self.mppi_iterations_ = getattr(args, "mppi_iterations", 4)
        self.mppi_init_iterations_ = getattr(args, "mppi_init_iterations", 8)
        self.mppi_lambda_ = getattr(args, "mppi_lambda", 1.0)
        self.mppi_noise_sigma_ = getattr(args, "mppi_noise_sigma", 0.01)
        self.mppi_noise_decay_ = getattr(args, "mppi_noise_decay", 0.85)
        self.mppi_elite_frac_ = getattr(args, "mppi_elite_frac", 0.1)
        self.mppi_use_torch_compile_ = getattr(args, "mppi_use_torch_compile", False)
        default_mppi_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.mppi_device_ = getattr(args, "mppi_device", default_mppi_device)
        self.mppi_w_ref_q_ = getattr(args, "mppi_w_ref_q", 30.0)
        self.mppi_w_ref_u_ = getattr(args, "mppi_w_ref_u", 3.0)

        self.mesh_path_ = os.path.join(
            repo_root,
            "IsaacGymEnvs",
            "assets",
            "urdf",
            "sektion_cabinet_model",
            "urdf",
            "sektion_cabinet_2.urdf",
        )
        self.handle_link_name = str(getattr(args, "handle_link_name", "drawer_handle_top"))
        self.lambda_optimizer = LambdaContactControlOptimizer(
            mesh_path=self.mesh_path_,
            obj_mass=self.obj_mass_,
            arm_friction=self.mu_object_,
            contact_stiffness=self.model_params,
            time_step=self.h_ * 10.0,
            sample_num=args.sample_num,
            pos_coef=args.pos_coef,
            ori_coef=args.ori_coef,
            scale_factors=[1.0, 1.0, 1.0],
            sample_bounds={
                "x": (args.handle_sample_x_min, args.handle_sample_x_max),
                "y": (args.handle_sample_y_min, args.handle_sample_y_max),
                "z": (args.handle_sample_z_min, args.handle_sample_z_max),
            },
            handle_link_name=self.handle_link_name,
        )
        selected_links = getattr(self.lambda_optimizer.pp, "selected_link_names", [])
        print(
            f"[CabinetMPCParams] requested_handle_link={self.handle_link_name}, "
            f"optimizer_selected_links={selected_links}"
        )


class CabinetSimulator:
    @staticmethod
    def _resolve_drawer_body_name_from_handle(handle_link_name):
        handle_name = str(handle_link_name).lower()
        return "drawer_bottom" if handle_name.endswith("bottom") else "drawer_top"

    @staticmethod
    def _get_franka_asset_info(repo_root_local):
        asset_root = os.path.join(repo_root_local, "envs/robots/assets/urdf")
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

    def _find_first_body_handle(self, candidates):
        for name in candidates:
            handle = self.gym.find_actor_rigid_body_handle(self.env, self.franka_actor, name)
            if handle != -1:
                return handle
        raise RuntimeError(f"Failed to find Franka rigid body handle from candidates: {candidates}")

    def __init__(
        self,
        param,
        headless=False,
        sim_device="cuda:0",
        graphics_device_id=0,
        physx_use_gpu=False,
        save_frames=False,
        frame_dir="outputs/cabinet_frames",
        frame_every=1,
        camera_width=1280,
        camera_height=720,
        camera_pos=(1.9, 1.1, 1.3),
        camera_target=(0.9, 0.0, 0.45),
    ):
        import torch

        self.param_ = param
        self.break_out_signal_ = False
        self.dyn_paused_ = False
        self.viewer_ = None
        self.save_frames_ = bool(save_frames)
        self.frame_dir_ = frame_dir
        self.frame_every_ = max(1, int(frame_every))
        self.camera_width_ = int(camera_width)
        self.camera_height_ = int(camera_height)
        self.camera_pos_ = tuple(float(v) for v in camera_pos)
        self.camera_target_ = tuple(float(v) for v in camera_target)
        self.capture_cam_handle_ = None
        self.device = torch.device(sim_device if "cuda" in sim_device and torch.cuda.is_available() else "cpu")

        self.gym = gymapi.acquire_gym()
        compute_id = int(sim_device.split(":")[-1]) if "cuda" in sim_device and ":" in sim_device else (
            0 if "cuda" in sim_device else -1
        )

        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        # This controller relies on legacy rigid-contact queries, which are unavailable
        # once Isaac Gym runs with the GPU pipeline enabled.
        sim_params.use_gpu_pipeline = False
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 12
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.005
        sim_params.physx.rest_offset = 0.0
        # Keep PhysX on CPU by default when using CPU pipeline contact queries.
        # This avoids extra host/device synchronization every control step.
        sim_params.physx.use_gpu = bool(physx_use_gpu and compute_id >= 0)

        self.sim = self.gym.create_sim(compute_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
        env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)

        self._create_scene_actors()
        self._init_frame_capture()
        self._configure_franka()
        self.gym.prepare_sim(self.sim)
        self._acquire_tensors()
        self._build_body_index_cache()
        self.init_data()
        self.reset_mj_env()

        if not headless:
            self.viewer_ = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer_ is not None:
                cam_pos = gymapi.Vec3(1.9, 1.1, 1.3)
                cam_target = gymapi.Vec3(0.9, 0.0, 0.45)
                self.gym.viewer_camera_look_at(self.viewer_, self.env, cam_pos, cam_target)

    def _init_frame_capture(self):
        if not self.save_frames_:
            return
        os.makedirs(self.frame_dir_, exist_ok=True)
        cam_props = gymapi.CameraProperties()
        cam_props.width = self.camera_width_
        cam_props.height = self.camera_height_
        self.capture_cam_handle_ = self.gym.create_camera_sensor(self.env, cam_props)
        cam_pos = gymapi.Vec3(*self.camera_pos_)
        cam_target = gymapi.Vec3(*self.camera_target_)
        self.gym.set_camera_location(self.capture_cam_handle_, self.env, cam_pos, cam_target)
        print(
            f"[CabinetSimulator] frame capture enabled: dir={self.frame_dir_}, "
            f"every={self.frame_every_}, size={self.camera_width_}x{self.camera_height_}"
        )

    def _create_scene_actors(self):
        franka_asset_root, franka_asset_file = self._get_franka_asset_info(repo_root)
        asset_root = os.path.join(repo_root, "IsaacGymEnvs", "assets")
        cabinet_asset_file = "urdf/sektion_cabinet_model/urdf/sektion_cabinet_2.urdf"

        franka_opts = gymapi.AssetOptions()
        franka_opts.flip_visual_attachments = True
        franka_opts.fix_base_link = True
        franka_opts.collapse_fixed_joints = True
        franka_opts.disable_gravity = True
        franka_opts.thickness = 0.001
        franka_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
        franka_opts.use_mesh_materials = True
        self.franka_asset = self.gym.load_asset(self.sim, franka_asset_root, franka_asset_file, franka_opts)

        cabinet_opts = gymapi.AssetOptions()
        cabinet_opts.flip_visual_attachments = False
        cabinet_opts.collapse_fixed_joints = True
        cabinet_opts.disable_gravity = False
        cabinet_opts.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        cabinet_opts.armature = 0.005
        self.cabinet_asset = self.gym.load_asset(self.sim, asset_root, cabinet_asset_file, cabinet_opts)

        marker_opts = gymapi.AssetOptions()
        marker_opts.fix_base_link = True
        marker_opts.disable_gravity = True
        marker_opts.disable_gravity = True
        self.marker_asset = self.gym.create_sphere(self.sim, 0.01, marker_opts)

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        cabinet_pose = gymapi.Transform()
        cabinet_pose.p = gymapi.Vec3(0.3, 0.0, 0.4)

        marker_pose = gymapi.Transform()
        marker_pose.p = gymapi.Vec3(0.8, 0.0, 0.5)
        marker_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        contact_marker_pose = gymapi.Transform()
        contact_marker_pose.p = gymapi.Vec3(0.8, 0.0, 0.5)
        contact_marker_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.franka_actor = self.gym.create_actor(self.env, self.franka_asset, franka_pose, "franka", 0, 1, 0)
        self.cabinet_actor = self.gym.create_actor(self.env, self.cabinet_asset, cabinet_pose, "cabinet", 0, 2, 0)
        # Put markers in dedicated collision groups so they never collide with cabinet/franka in group 0.
        self.target_marker_actor = self.gym.create_actor(self.env, self.marker_asset, marker_pose, "target_marker", 30, 0, 0)
        self.contact_marker_actor = self.gym.create_actor(
            self.env, self.marker_asset, contact_marker_pose, "contact_marker", 31, 0, 0
        )

        self.gym.set_rigid_body_color(
            self.env, self.target_marker_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.7, 0.9)
        )
        self.gym.set_rigid_body_color(
            self.env, self.contact_marker_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.95, 0.1, 0.1)
        )

        self.hand_handle = self._find_first_body_handle(("attachment", "panda_link7"))
        self.lfinger_handle = self._find_first_body_handle(("fingertip", "attachment", "panda_link7"))
        self.rfinger_handle = self._find_first_body_handle(("fingertip", "attachment", "panda_link7"))
        requested_drawer_body = self._resolve_drawer_body_name_from_handle(self.param_.handle_link_name)
        self.drawer_handle = self.gym.find_actor_rigid_body_handle(self.env, self.cabinet_actor, requested_drawer_body)
        if self.drawer_handle == -1:
            self.drawer_handle = self.gym.find_actor_rigid_body_handle(self.env, self.cabinet_actor, "drawer_top")

    def _configure_franka(self):
        dof_props = self.gym.get_actor_dof_properties(self.env, self.franka_actor)
        dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][:] = 400.0
        dof_props["damping"][:] = 80.0
        if dof_props["effort"].shape[0] >= 9:
            dof_props["effort"][7] = 200.0
            dof_props["effort"][8] = 200.0
        self.gym.set_actor_dof_properties(self.env, self.franka_actor, dof_props)

        cabinet_dof_props = self.gym.get_actor_dof_properties(self.env, self.cabinet_actor)
        cabinet_dof_props["damping"][:] = 10.0
        self.gym.set_actor_dof_properties(self.env, self.cabinet_actor, cabinet_dof_props)

        self.franka_dof_count = self.gym.get_actor_dof_count(self.env, self.franka_actor)
        self._joint_targets = np.zeros(self.franka_dof_count, dtype=np.float32)
        self._joint_targets[:7] = np.array(self.param_.init_robot_qpos_, dtype=np.float32)
        self._joint_targets[7:] = 0.04

    def _acquire_tensors(self):
        import torch

        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        jac = self.gym.acquire_jacobian_tensor(self.sim, "franka")

        self.root_state_tensor = gymtorch.wrap_tensor(root_tensor).view(1, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(1, -1, 13)
        self._jacobian = gymtorch.wrap_tensor(jac)
        self.device = self.rigid_body_states.device

        self.num_franka_dofs = self.gym.get_actor_dof_count(self.env, self.franka_actor)
        self.num_cabinet_dofs = self.gym.get_actor_dof_count(self.env, self.cabinet_actor)
        self.franka_dof_state = self.dof_state.view(1, -1, 2)[:, : self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        self.cabinet_dof_state = self.dof_state.view(1, -1, 2)[:, self.num_franka_dofs :]
        self.cabinet_dof_pos = self.cabinet_dof_state[..., 0]
        self.cabinet_dof_vel = self.cabinet_dof_state[..., 1]
        self.franka_dof_targets = torch.zeros((1, self.num_franka_dofs + self.num_cabinet_dofs), dtype=torch.float32, device=self.device)
        self.franka_default_dof_pos = to_torch(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.04, 0.04],
            dtype=torch.float32,
            device=self.device,
        )

    def _build_body_index_cache(self):
        self.target_marker_idx = self.gym.get_actor_index(self.env, self.target_marker_actor, gymapi.DOMAIN_SIM)
        self.contact_marker_idx = self.gym.get_actor_index(self.env, self.contact_marker_actor, gymapi.DOMAIN_SIM)
        cabinet_body_dict = self.gym.get_actor_rigid_body_dict(self.env, self.cabinet_actor)
        preferred_handle = str(self.param_.handle_link_name).strip()
        preferred_drawer = self._resolve_drawer_body_name_from_handle(preferred_handle)
        if preferred_drawer == "drawer_bottom":
            candidate_names = (
                preferred_handle,
                "drawer_handle_bottom",
                "drawer_bottom",
                "drawer_handle_top",
                "drawer_top",
            )
        else:
            candidate_names = (
                preferred_handle,
                "drawer_handle_top",
                "drawer_top",
                "drawer_handle_bottom",
                "drawer_bottom",
            )

        self.handle_body_name = None
        for candidate in candidate_names:
            if candidate in cabinet_body_dict:
                self.handle_body_name = candidate
                break
        if self.handle_body_name is None:
            raise RuntimeError(
                f"Failed to resolve cabinet handle body from candidates={candidate_names}. "
                f"Available={sorted(cabinet_body_dict.keys())}"
            )
        self.handle_body_local_idx = int(cabinet_body_dict[self.handle_body_name])
        self.handle_body_idx = self.gym.get_actor_rigid_body_index(
            self.env,
            self.cabinet_actor,
            self.handle_body_local_idx,
            gymapi.DOMAIN_SIM,
        )
        cabinet_body_count = self.gym.get_actor_rigid_body_count(self.env, self.cabinet_actor)
        self.cabinet_body_indices = set(
            self.gym.get_actor_rigid_body_index(self.env, self.cabinet_actor, i, gymapi.DOMAIN_SIM)
            for i in range(cabinet_body_count)
        )
        self.franka_body_names = self.gym.get_actor_rigid_body_names(self.env, self.franka_actor)
        self.franka_body_env_dict = self.gym.get_actor_rigid_body_dict(self.env, self.franka_actor)
        franka_body_count = self.gym.get_actor_rigid_body_count(self.env, self.franka_actor)
        self.franka_body_indices = set(
            self.gym.get_actor_rigid_body_index(self.env, self.franka_actor, i, gymapi.DOMAIN_SIM)
            for i in range(franka_body_count)
        )
        self.franka_body_name_to_index = {
            self.franka_body_names[i]: self.gym.get_actor_rigid_body_index(self.env, self.franka_actor, i, gymapi.DOMAIN_SIM)
            for i in range(franka_body_count)
        }
        self.ee_body_name = "attachment" if "attachment" in self.franka_body_names else "panda_link7"
        self.ee_pos_body_name = "fingertip" if "fingertip" in self.franka_body_names else self.ee_body_name
        self.ee_ori_handle = self.gym.find_actor_rigid_body_handle(self.env, self.franka_actor, self.ee_body_name)
        self.ee_pos_handle = self.gym.find_actor_rigid_body_handle(self.env, self.franka_actor, self.ee_pos_body_name)
        print(
            f"[CabinetSimulator] requested_handle={preferred_handle}, "
            f"resolved_handle_body={self.handle_body_name}, drawer_body={preferred_drawer}"
        )
        print(f"[CabinetSimulator] ee_pos_body={self.ee_pos_body_name}, ee_ori_body={self.ee_body_name}")
        self.cabinet_body_indices.discard(self.handle_body_idx)
        self._jacobian_body_offset = len(self.franka_body_names) - int(self._jacobian.shape[1])

    def _resolve_first_existing_body_name(self, candidates):
        for name in candidates:
            if name in self.franka_body_names:
                return name
        raise RuntimeError(
            f"None of candidate end-effector body names exist: {candidates}. "
            f"Available bodies: {self.franka_body_names}"
        )

    def refresh_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

    def init_data(self):
        self.refresh_tensors()

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]

        finger_pos = 0.5 * (lfinger_pos + rfinger_pos)
        finger_rot = lfinger_rot
        hand_rot_inv = quat_conjugate(hand_rot)
        tensor_device = hand_pos.device
        tensor_dtype = hand_pos.dtype

        grasp_pose_axis = 1
        grasp_offset = to_torch([get_axis_params(0.04, grasp_pose_axis)], dtype=tensor_dtype, device=tensor_device)
        self.franka_local_grasp_pos = quat_apply(hand_rot_inv, finger_pos - hand_pos) + grasp_offset
        self.franka_local_grasp_rot = quat_mul(hand_rot_inv, finger_rot)
        self.drawer_local_grasp_pos = to_torch(
            [get_axis_params(0.01, grasp_pose_axis, 0.3)],
            dtype=tensor_dtype,
            device=tensor_device,
        )
        self.drawer_local_grasp_rot = to_torch([[0.0, 0.0, 0.0, 1.0]], dtype=tensor_dtype, device=tensor_device)

        self.gripper_forward_axis = to_torch([[0.0, 0.0, 1.0]], dtype=tensor_dtype, device=tensor_device)
        self.drawer_inward_axis = to_torch([[-1.0, 0.0, 0.0]], dtype=tensor_dtype, device=tensor_device)
        self.gripper_up_axis = to_torch([[0.0, 1.0, 0.0]], dtype=tensor_dtype, device=tensor_device)
        self.drawer_up_axis = to_torch([[0.0, 0.0, 1.0]], dtype=tensor_dtype, device=tensor_device)

        self.franka_grasp_pos = self.franka_local_grasp_pos.clone()
        self.franka_grasp_rot = self.franka_local_grasp_rot.clone()
        self.drawer_grasp_pos = self.drawer_local_grasp_pos.clone()
        self.drawer_grasp_rot = self.drawer_local_grasp_rot.clone()

    def compute_observations(self):
        self.refresh_tensors()
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        drawer_pos = self.rigid_body_states[:, self.drawer_handle][:, 0:3]
        drawer_rot = self.rigid_body_states[:, self.drawer_handle][:, 3:7]
        (
            self.franka_grasp_rot[:],
            self.franka_grasp_pos[:],
            self.drawer_grasp_rot[:],
            self.drawer_grasp_pos[:],
        ) = compute_grasp_transforms(
            hand_rot,
            hand_pos,
            self.franka_local_grasp_rot,
            self.franka_local_grasp_pos,
            drawer_rot,
            drawer_pos,
            self.drawer_local_grasp_rot,
            self.drawer_local_grasp_pos,
        )

    def _simulate_once(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if self.viewer_ is not None or self.save_frames_:
            self.gym.step_graphics(self.sim)
        if self.save_frames_ and self.capture_cam_handle_ is not None:
            self.gym.render_all_camera_sensors(self.sim)
        if self.viewer_ is not None:
            self.gym.draw_viewer(self.viewer_, self.sim, True)
            self.gym.sync_frame_time(self.sim)

    def save_frame_if_needed(self, step_idx):
        if not self.save_frames_ or self.capture_cam_handle_ is None:
            return
        if step_idx % self.frame_every_ != 0:
            return
        out_path = os.path.join(self.frame_dir_, f"frame_{int(step_idx):06d}.png")
        self.gym.write_camera_image_to_file(
            self.sim,
            self.env,
            self.capture_cam_handle_,
            gymapi.IMAGE_COLOR,
            out_path,
        )

    def reset_mj_env(self):
        self.franka_dof_state[:] = 0.0
        self.franka_dof_pos[0, :7] = to_torch(self.param_.init_robot_qpos_, dtype=self.franka_dof_pos.dtype, device=self.device)
        self.franka_dof_pos[0, 7:] = 0.04
        self.cabinet_dof_state[:] = 0.0
        self.franka_dof_targets[0, : self.num_franka_dofs] = self.franka_dof_pos[0]

        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.franka_dof_targets))

        for _ in range(8):
            self._simulate_once()
        self.refresh_tensors()
        self._initial_handle_pos, self._initial_handle_quat = self.get_handle_pose()

    def get_handle_pose(self):
        self.refresh_tensors()
        body_state = self.rigid_body_states[0, self.handle_body_idx]
        pos = body_state[0:3].detach().cpu().numpy().astype(np.float32)
        quat_xyzw = body_state[3:7].detach().cpu().numpy().astype(np.float32)
        quat_wxyz = _quat_xyzw_to_wxyz(quat_xyzw)
        return pos, quat_wxyz

    def get_state(self):
        pos, quat = self.get_handle_pose()
        q = self.franka_dof_pos[0, :7].detach().cpu().numpy().astype(np.float32)
        return np.hstack([pos, quat, q]).astype(np.float32)

    def get_end_effector_pos(self):
        self.refresh_tensors()
        p = self.rigid_body_states[0, self.ee_pos_handle, 0:3].detach().cpu().numpy().astype(np.float32)
        q_xyzw = self.rigid_body_states[0, self.ee_ori_handle, 3:7].detach().cpu().numpy().astype(np.float32)
        R = Rotation.from_quat(q_xyzw).as_matrix().astype(np.float32)
        return p, R

    def get_R(self):
        return self.get_end_effector_pos()[1]

    def get_drawer_open_amount(self):
        self.refresh_tensors()
        return float(self.cabinet_dof_pos[0, 3].item())

    def get_target_handle_pose(self, target_open):
        return (
            self._initial_handle_pos + np.array([target_open, 0.0, 0.0], dtype=np.float32),
            self._initial_handle_quat.copy(),
        )

    def show_target(self, goal_pos=None, goal_quat=None):
        if goal_pos is not None:
            self.root_state_tensor[0, self.target_marker_idx, 0:3] = to_torch(
                goal_pos, dtype=self.root_state_tensor.dtype, device=self.device
            )
        if goal_quat is not None:
            self.root_state_tensor[0, self.target_marker_idx, 3:7] = to_torch(
                goal_quat, dtype=self.root_state_tensor.dtype, device=self.device
            )
        self.root_state_tensor[0, self.target_marker_idx, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))

    def show_best_contact_point(self, point_pos=None):
        if point_pos is None:
            return
        self.root_state_tensor[0, self.contact_marker_idx, 0:3] = to_torch(
            point_pos, dtype=self.root_state_tensor.dtype, device=self.device
        )
        self.root_state_tensor[0, self.contact_marker_idx, 3:7] = to_torch(
            [0.0, 0.0, 0.0, 1.0], dtype=self.root_state_tensor.dtype, device=self.device
        )
        self.root_state_tensor[0, self.contact_marker_idx, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor))

    def draw_rollout_lines(self, rollout_q):
        if self.viewer_ is None or rollout_q is None:
            return
        self.gym.clear_lines(self.viewer_)
        if isinstance(rollout_q, torch.Tensor):
            rollout_q = rollout_q.detach().cpu().numpy()
        rollout_q = np.asarray(rollout_q, dtype=np.float32)
        if rollout_q.ndim != 2 or rollout_q.shape[0] < 2:
            return
        ee_points = []
        for i in range(rollout_q.shape[0]):
            q_robot = rollout_q[i, -self.param_.n_robot_qpos_ :]
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

    def step_joint_delta(self, dq):
        dq_t = to_torch(dq, dtype=self.franka_dof_pos.dtype, device=self.device).reshape(7)
        q_des = self.franka_dof_pos[0, :7] + dq_t
        self.franka_dof_targets[0, :7] = q_des
        self.franka_dof_targets[0, 7 : self.num_franka_dofs] = 0.04
        if self.num_cabinet_dofs > 0:
            self.franka_dof_targets[0, self.num_franka_dofs :] = self.cabinet_dof_pos[0]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.franka_dof_targets))
        self._simulate_once()

    def step(self, cmd):
        if isinstance(cmd, torch.Tensor):
            cmd = cmd.detach().cpu().numpy()
        cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
        if cmd.size != 7:
            raise ValueError(f"Invalid action dimension: {cmd.size}. Expected 7.")
        self.step_joint_delta(cmd)

    def get_physx_contacts(self):
        contacts_raw = self.gym.get_env_rigid_contacts(self.env)
        contacts = []
        if contacts_raw is None:
            return contacts
        for c in contacts_raw:
            if getattr(c, "dtype", None) is None or c.dtype.names is None:
                continue
            field_names = set(c.dtype.names)
            body0 = int(c["body0"] if "body0" in c.dtype.names else c["bodyA"])
            body1 = int(c["body1"] if "body1" in c.dtype.names else c["bodyB"])
            separation_field = None
            for name in ("separation", "distance", "dist", "lambda", "offset"):
                if name in field_names:
                    separation_field = name
                    break
            if separation_field is None:
                continue
            separation = float(c[separation_field])
            normal_raw = c["normal"]
            normal = np.array([float(normal_raw["x"]), float(normal_raw["y"]), float(normal_raw["z"])], dtype=np.float32)
            nrm = np.linalg.norm(normal)
            if nrm > 1e-8:
                normal /= nrm
            pos = None
            if "pos" in c.dtype.names:
                pos_raw = c["pos"]
                pos = np.array([float(pos_raw["x"]), float(pos_raw["y"]), float(pos_raw["z"])], dtype=np.float32)
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

    def get_body_point_jacobian(self, sim_body_idx, point_world):
        info = None
        for name, idx in self.franka_body_name_to_index.items():
            if idx == int(sim_body_idx):
                info = name
                break
        if info is None:
            return np.zeros((3, self.param_.n_robot_qpos_), dtype=np.float32)

        local_idx = self.franka_body_names.index(info)
        env_handle = int(self.franka_body_env_dict[info])
        self.gym.refresh_jacobian_tensors(self.sim)
        jac_idx = local_idx - self._jacobian_body_offset
        if jac_idx < 0 or jac_idx >= int(self._jacobian.shape[1]):
            return np.zeros((3, self.param_.n_robot_qpos_), dtype=np.float32)
        jac_body = self._jacobian[0, jac_idx]
        jv = np.array(jac_body[:3, : self.param_.n_robot_qpos_], dtype=np.float32)
        jw = np.array(jac_body[3:6, : self.param_.n_robot_qpos_], dtype=np.float32)

        self.refresh_tensors()
        body_pos = self.rigid_body_states[0, env_handle, 0:3].detach().cpu().numpy().astype(np.float32)
        r = np.asarray(point_world, dtype=np.float32) - body_pos
        return jv - _skew(r) @ jw

    def close(self):
        if self.viewer_ is not None:
            self.gym.destroy_viewer(self.viewer_)
            self.viewer_ = None
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None


class ContactCabinet:
    def __init__(self, param):
        self.param_ = param

    def detect_once(self, simulator: CabinetSimulator):
        q = simulator.get_state()
        nv = self.param_.n_qvel_
        max_ncon = self.param_.max_ncon_

        phi_vec = np.ones((max_ncon * 4,), dtype=np.float32)
        jac_mat = np.zeros((max_ncon * 4, nv), dtype=np.float32)
        jac_mat_env = np.zeros((max_ncon * 4, nv), dtype=np.float32)
        con_pos_list = []

        handle_pos = q[0:3]
        handle_quat_wxyz = q[3:7]
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

            involves_handle = (b0 == simulator.handle_body_idx) or (b1 == simulator.handle_body_idx)
            if not involves_handle:
                continue

            if b1 == simulator.handle_body_idx:
                n_raw = -n_raw

            if cpos is None:
                cpos = handle_pos + 0.01 * n_raw
            else:
                cpos = np.asarray(cpos, dtype=np.float32)

            r_obj = cpos - handle_pos
            J_obj = np.zeros((3, nv), dtype=np.float32)
            J_obj[:, 0:3] = np.eye(3, dtype=np.float32)
            J_obj[:, 3:6] = -_skew(r_obj)

            J_other = np.zeros((3, nv), dtype=np.float32)
            other_sim_idx = b1 if b0 == simulator.handle_body_idx else b0
            if other_sim_idx in simulator.franka_body_indices:
                J_body_point = simulator.get_body_point_jacobian(other_sim_idx, cpos)
                J_other[:, 6 : 6 + self.param_.n_robot_qpos_] = J_body_point

            J_rel_point = J_obj - J_other
            n, t1, t2 = _tangent_basis_from_normal(n_raw)
            con_jac = np.array(_contact_jacobian(n, t1, t2, J_rel_point, mu), dtype=np.float32)

            other_is_franka = (b0 in simulator.franka_body_indices) or (b1 in simulator.franka_body_indices)
            if other_is_franka and row_idx < max_ncon:
                phi_vec[4 * row_idx : 4 * row_idx + 4] = 0.5 * sep
                jac_mat[4 * row_idx : 4 * row_idx + 4, :] = con_jac
                row_idx += 1

            other_is_env = (b0 in simulator.cabinet_body_indices) or (b1 in simulator.cabinet_body_indices)
            if other_is_env and row_env_idx < max_ncon:
                jac_mat_env[4 * row_env_idx : 4 * row_env_idx + 4, :] = con_jac
                row_env_idx += 1
                quat_xyzw = _quat_wxyz_to_xyzw(handle_quat_wxyz)
                R_handle_to_world = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
                con_pos_local = R_handle_to_world.T @ (cpos - handle_pos)
                con_pos_list.append(con_pos_local.astype(np.float32))

        return phi_vec, jac_mat, con_pos_list, jac_mat_env


def solve_contact_ik(curr_q, target_pos_world, max_iters=80, step_size=0.7, damp=1e-4):
    q = np.asarray(curr_q, dtype=np.float32).copy()
    q_lb = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
    q_ub = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)
    for _ in range(max_iters):
        T = np.array(_franka_fk_T_jax(q), dtype=np.float32)
        pos = T[:3, 3]
        err = np.asarray(target_pos_world, dtype=np.float32) - pos
        if np.linalg.norm(err) < 1e-3:
            break
        J = np.array(_franka_jacobian_pos_jax(q), dtype=np.float32)
        JJt = J @ J.T + damp * np.eye(3, dtype=np.float32)
        dq = J.T @ np.linalg.solve(JJt, err)
        q = np.clip(q + step_size * dq, q_lb, q_ub)
    return q.astype(np.float32)


def build_reference_trajectory(curr_q, q_goal, horizon, u_lb, u_ub):
    curr_q = np.asarray(curr_q, dtype=np.float32)
    q_goal = np.asarray(q_goal, dtype=np.float32)
    q_ref = np.zeros((horizon, curr_q.shape[0]), dtype=np.float32)
    for t in range(horizon):
        alpha = min((t + 1) / max(1, int(0.6 * horizon)), 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        q_ref[t] = (1.0 - alpha) * curr_q + alpha * q_goal
    u_ref = np.zeros_like(q_ref)
    u_ref[0] = np.clip(q_ref[0] - curr_q, u_lb, u_ub)
    for t in range(1, horizon):
        u_ref[t] = np.clip(q_ref[t] - q_ref[t - 1], u_lb, u_ub)
    return q_ref, u_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attract_coef", type=float, default=0.5)
    parser.add_argument("--reject_coef", type=float, default=0.001)
    parser.add_argument("--contact_coef", type=float, default=0.5)
    parser.add_argument("--contact_cost_param", type=float, default=0.0)
    parser.add_argument("--model_param", type=float, default=7.0)
    parser.add_argument("--reject_dis", type=float, default=0.01)
    parser.add_argument("--attract_point_comp", type=float, default=0.05)
    parser.add_argument("--ground_height_threshold", type=float, default=0.0)
    parser.add_argument("--sample_num", type=int, default=70)
    parser.add_argument(
        "--handle-link-name",
        type=str,
        default="drawer_handle_top",
        choices=["drawer_handle_top", "drawer_handle_bottom"],
    )
    parser.add_argument("--handle-sample-x-min", dest="handle_sample_x_min", type=float, default=0.273229)
    parser.add_argument("--handle-sample-x-max", dest="handle_sample_x_max", type=float, default=0.311329)
    parser.add_argument("--handle-sample-y-min", dest="handle_sample_y_min", type=float, default=-0.084135)
    parser.add_argument("--handle-sample-y-max", dest="handle_sample_y_max", type=float, default=0.084135)
    parser.add_argument("--handle-sample-z-min", dest="handle_sample_z_min", type=float, default=-0.002405)
    parser.add_argument("--handle-sample-z-max", dest="handle_sample_z_max", type=float, default=0.021405)
    parser.add_argument("--pos_coef", type=float, default=1.0)
    parser.add_argument("--ori_coef", type=float, default=0.001)
    parser.add_argument("--low_err_coef", type=float, default=0.1)
    parser.add_argument("--upper_err_coef", type=float, default=1.0)
    parser.add_argument("--mppi_w_ee_ori", type=float, default=10.0)
    parser.add_argument("--mppi_w_joint_limit", type=float, default=5.0)
    parser.add_argument("--mppi_w_manip", type=float, default=0.1)
    parser.add_argument("--mppi_w_cond", type=float, default=0.0)
    parser.add_argument("--mppi_w_energy", type=float, default=0.01)
    parser.add_argument("--mppi_w_vel", type=float, default=0.01)
    parser.add_argument("--mppi_w_acc", type=float, default=0.001)
    parser.add_argument("--drawer-open-target", type=float, default=0.3)
    parser.add_argument("--drawer-open-weight", type=float, default=500.0)
    parser.add_argument("--drawer-lateral-weight", type=float, default=25.0)
    parser.add_argument("--drawer-quat-weight", type=float, default=0.0)
    parser.add_argument("--mppi-samples", type=int, default=512)
    parser.add_argument("--mppi-iterations", type=int, default=4)
    parser.add_argument("--mppi-init-iterations", type=int, default=8)
    parser.add_argument("--mppi-lambda", type=float, default=1.0)
    parser.add_argument("--mppi-noise-sigma", type=float, default=0.01)
    parser.add_argument("--mppi-noise-decay", type=float, default=0.85)
    parser.add_argument("--mppi-elite-frac", type=float, default=0.1)
    parser.add_argument("--mppi-use-torch-compile", action="store_true")
    parser.add_argument("--mppi-w-ref-q", type=float, default=30.0)
    parser.add_argument("--mppi-w-ref-u", type=float, default=3.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sim-device", type=str, default="cuda:0")
    parser.add_argument("--mppi-device", type=str, default=None)
    parser.add_argument("--physx-use-gpu", action="store_true")
    parser.add_argument("--graphics-device-id", type=int, default=0)
    parser.add_argument("--max-rollout-length", type=int, default=800)
    parser.add_argument("--use_reference", type=bool, default=False)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-dir", type=str, default="outputs/cabinet_frames")
    parser.add_argument("--frame-every", type=int, default=1)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[1.9, 1.1, 1.3])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.9, 0.0, 0.45])
    args = parser.parse_args()

    param = CabinetMPCParams(args)
    contact = ContactCabinet(param)
    env = CabinetSimulator(
        param,
        headless=args.headless,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        physx_use_gpu=args.physx_use_gpu,
        save_frames=args.save_frames,
        frame_dir=args.frame_dir,
        frame_every=args.frame_every,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_pos=args.camera_pos,
        camera_target=args.camera_target,
    )

    target_p, target_q = env.get_target_handle_pose(args.drawer_open_target)
    param.target_p_ = target_p
    param.target_q_ = target_q
    env.show_target(goal_pos=target_p, goal_quat=_quat_wxyz_to_xyzw(target_q))
    env.save_frame_if_needed(0)

    mpc = MPPICabinet(param) if param.mpc_model == "explicit" else MPCImplicit(param)

    rollout_step = 0
    consecutive_success_time = 0
    consecutive_success_time_threshold = 20
    verify_cost = 0
    current_x = np.zeros(7, dtype=np.float32)
    current_x[0] = 0.0
    current_x[3] = 1.0

    low_err_coef = args.low_err_coef
    upper_err_coef = args.upper_err_coef

    try:
        while rollout_step < args.max_rollout_length:
            if env.dyn_paused_:
                continue

            curr_q = env.get_state()
            phi_vec, jac_mat, _, jac_mat_env = contact.detect_once(env)
            quat_xyzw = _quat_wxyz_to_xyzw(curr_q[3:7])
            R_handle_to_world = Rotation.from_quat(quat_xyzw).as_matrix().astype(np.float32)
            gravity = np.hstack([R_handle_to_world.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3, dtype=np.float32)])

            target_pos_local = R_handle_to_world.T @ (param.target_p_ - curr_q[0:3])
            target_quat_local = rotations.quaternion_multiply(
                rotations.quaternion_conjugate(curr_q[3:7]),
                param.target_q_,
            )
            target_pose_local = np.hstack([target_pos_local, target_quat_local]).astype(np.float32)

            param.lambda_optimizer.update_Jacobian(jac_mat_env)
            visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                curr_q[0:3],
                R_handle_to_world,
                param.target_p_,
                args.ground_height_threshold,
            )
            best_contact_point, normal, min_error, max_error, curr_ori_coef = param.lambda_optimizer.choose_contact_points(
                target_pose_local,
                current_x,
                gravity,
                visible_point_idx,
            )
            best_contact_point_world = R_handle_to_world @ best_contact_point + curr_q[0:3]

            attract_point_world = best_contact_point_world.copy()
            attract_point_world[2] += args.attract_point_comp

            ee_pos = env.get_end_effector_pos()[0].copy()
            local_point = R_handle_to_world.T @ (ee_pos - curr_q[0:3])
            p_arm_local, _, x_plus_opt, error, _ = param.lambda_optimizer.optimize_control_input(
                target_pose_local,
                current_x,
                gravity,
                local_point,
            )
            p_arm_world = R_handle_to_world @ p_arm_local + curr_q[0:3]
            normal_world = R_handle_to_world @ normal
            ref_joint_traj = None
            ref_ctrl_traj = None
            if args.use_reference:
                ik_target_world = p_arm_world - 0.015 * normal_world
                st = time.time()
                q_contact = solve_contact_ik(curr_q[-param.n_robot_qpos_ :], ik_target_world)
                print("ik cost time = ", time.time() - st)
                ref_joint_traj, ref_ctrl_traj = build_reference_trajectory(
                    curr_q[-param.n_robot_qpos_ :],
                    q_contact,
                    param.mpc_horizon_,
                    np.broadcast_to(np.asarray(param.mpc_u_lb_, dtype=np.float32), (param.n_cmd_,)),
                    np.broadcast_to(np.asarray(param.mpc_u_ub_, dtype=np.float32), (param.n_cmd_,)),
                )

            if verify_cost:
                low_err_coef = args.low_err_coef
            elif np.linalg.norm(ee_pos - attract_point_world) < 5e-2:
                low_err_coef *= 1.1
            upper_err_coef = max(args.upper_err_coef if not verify_cost else upper_err_coef - 0.002, 0.7)

            delta_error = max_error - min_error
            adaptive = delta_error * upper_err_coef if verify_cost else delta_error * low_err_coef
            verify_cost = 1 if error < (min_error + adaptive) else 0
            verify_cost = 0
            st = time.time()
            sol = mpc.plan_once(
                param.target_p_,
                param.target_q_,
                curr_q,
                phi_vec,
                jac_mat,
                verify_cost_param=verify_cost,
                virtual_point=attract_point_world,
                contact_point=best_contact_point_world,
                curr_ori_coef=curr_ori_coef,
                sol_guess=param.sol_guess_,
                ref_joint_traj=ref_joint_traj,
                ref_ctrl_traj=ref_ctrl_traj,
            )
            param.sol_guess_ = sol["sol_guess"]
            action = sol["action"]
            # action = torch.zeros_like(action)
            print("time_cost =", time.time() - st, action.shape)
            # env.draw_rollout_lines(sol.get("rollout_q", None))
            
            print("verify cost:", verify_cost, "drawer_open:", env.get_drawer_open_amount())
            print("attract_point_world =", attract_point_world)

            env.step(action)
            rollout_step += 1

            curr_q_post = env.get_state()
            quat_xyzw_post = _quat_wxyz_to_xyzw(curr_q_post[3:7])
            R_handle_to_world_post = Rotation.from_quat(quat_xyzw_post).as_matrix().astype(np.float32)
            
            best_contact_point_world_post = (
                R_handle_to_world_post @ best_contact_point + curr_q_post[0:3]
            )
            print("best_contact_point_world_post = ", best_contact_point_world_post)
            attract_point_world_post = best_contact_point_world_post.copy()
            attract_point_world_post[2] += args.attract_point_comp

            quat = Rotation.from_matrix(env.get_R()).as_quat().astype(np.float32)
            env.show_target(goal_pos=attract_point_world_post, goal_quat=quat)
            env.show_best_contact_point(best_contact_point_world)
            env.save_frame_if_needed(rollout_step)

            drawer_open = env.get_drawer_open_amount()
            if drawer_open > args.drawer_open_target - 0.02:
                consecutive_success_time += 1
            else:
                consecutive_success_time = 0

            if consecutive_success_time > consecutive_success_time_threshold:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
