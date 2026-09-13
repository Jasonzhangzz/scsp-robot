import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
import traceback
import warnings
from argparse import Namespace
from collections import deque
from itertools import product
from typing import Dict, List, Tuple

# Force JAX/XLA onto CPU before any transitively imported module touches JAX.
# Isaac still uses `sim_device=cuda:0` below, so this only moves JAX helpers off GPU.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


warnings.filterwarnings("ignore", category=UserWarning)
signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(current_dir)
while os.path.basename(parent_dir) != "scsp-robot":
    _next_dir = os.path.dirname(parent_dir)
    if _next_dir == parent_dir:
        raise RuntimeError("scsp-robot repo root not found from %s" % current_dir)
    parent_dir = _next_dir
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from examples.mpc.franka.ik2.params import ExplicitMPCParams
from utils import metrics, rotations

ISAAC_IMPORT_ERROR = None
try:
    from isaacgym import gymapi, gymtorch
    import torch
    from examples.mpc.franka.ik2.test_mppi_isaac import IsaacFrankaSimulator, _extract_quat_xyzw, _extract_vec3
    from examples.mpc.franka.ik2.test_mpc_isaac import (
        ContactIsaacCartesian,
        IsaacFrankaOSCSimulator,
        _quat_xyzw_to_matrix,
        adapt_param_for_cartesian_solver,
    )
    from planning.mpc_explicit import MPCExplicitIsaac
    from planning.MPPIExplicit import _contact_jacobian, _tangent_basis_from_normal
    from planning.mpc_implicit import MPCImplicit
except ModuleNotFoundError as exc:
    torch = None
    gymapi = None
    gymtorch = None
    IsaacFrankaSimulator = None
    _extract_quat_xyzw = None
    _extract_vec3 = None
    ContactIsaacCartesian = None
    IsaacFrankaOSCSimulator = None
    _quat_xyzw_to_matrix = None
    adapt_param_for_cartesian_solver = None
    MPCExplicitIsaac = None
    _contact_jacobian = None
    _tangent_basis_from_normal = None
    MPCImplicit = None
    ISAAC_IMPORT_ERROR = exc


OPTIMIZER_ONLY_FIELDS = {
    "num_trials",
    "max_workers",
    "max_param_num",
    "result_dir",
    "parallel_envs",
    "max_rollout_length",
    "consecutive_success_steps",
    "success_pos_threshold",
    "success_quat_threshold",
    "early_stop_trials",
    "min_rate_threshold",
    "max_rate_threshold",
    "delta_threshold",
}
DEFAULT_EFFORT_JOINT_DAMPING = 10.0
SEARCH_PARAMETER_GRID = {
    "attract_coef": [0.5],
    "reject_coef": [0.001, 0.0005],
    "contact_coef": [0.5, 0.3, 0.7],
    "model_param": [7, 5, 10],
    "reject_dis": [0.02, 0.01],
    "attract_point_comp": [0.1],
    "ori_coef": [0.0, 0.005],
    "contact_cost_param": [1, 0],
    "low_err_coef": [0.3, 0.1],
}
SEARCH_PARAM_NAMES = list(SEARCH_PARAMETER_GRID.keys())
SEARCH_VALUE_PRECISION = 6
WORKER_CRASH_RETRY_LIMIT = 1
WORKER_POLL_INTERVAL_SEC = 0.2


SUMMARY_FIELDS = [
    "attract_coef",
    "reject_coef",
    "contact_coef",
    "contact_cost_param",
    "model_param",
    "reject_dis",
    "attract_point_comp",
    "ground_height_threshold",
    "sample_num",
    "pos_coef",
    "ori_coef",
    "low_err_coef",
    "upper_err_coef",
    "cartesian_step",
    "cartesian_joint_stiffness",
    "osc_pos_stiffness",
    "osc_ori_stiffness",
]

# Replace the second rollout seed with a less degenerate tabletop placement.
# The goal orientation for every trial is built below as a multi-axis flip
# relative to the corresponding initial pose.
TRIAL_POSE_OVERRIDES = {
    1: {
        "init_obj_qpos": np.array([0.3009, 0.0954, 0.4, 0.9969, 0.0, 0.0, 0.0791], dtype=np.float32),
        "target_p": np.array([0.4110, 0.0852, 0.38], dtype=np.float32),
    },
}

# [yaw, pitch, roll] in degrees. Each offset includes at least two non-zero
# axes, and the large pitch term makes the task a flip rather than a planar yaw
# rotation.
TRIAL_FLIP_ROTATION_DELTAS_DEG: Tuple[Tuple[float, float, float], ...] = (
    (35.0, 135.0, 20.0),
    (-45.0, -125.0, 30.0),
    (60.0, 145.0, -25.0),
    (-70.0, -135.0, -35.0),
)


def _quat_wxyz_to_rotation(quat_wxyz: np.ndarray) -> Rotation:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
    return Rotation.from_quat(quat_xyzw)


def _rotation_to_quat_wxyz(rotation: Rotation) -> np.ndarray:
    quat_xyzw = rotation.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    quat_norm = np.linalg.norm(quat_wxyz)
    if quat_norm > 0.0:
        quat_wxyz /= quat_norm
    if quat_wxyz[0] < 0.0:
        quat_wxyz = -quat_wxyz
    return quat_wxyz


def _build_trial_flip_delta(trial_count: int) -> Rotation:
    delta_ypr_deg = TRIAL_FLIP_ROTATION_DELTAS_DEG[trial_count % len(TRIAL_FLIP_ROTATION_DELTAS_DEG)]
    if sum(abs(angle_deg) > 1e-3 for angle_deg in delta_ypr_deg) < 2:
        raise ValueError("Flip delta must differ from the start pose on at least two rotation axes.")
    return Rotation.from_euler("ZYX", delta_ypr_deg, degrees=True)


def _apply_trial_flip_goal_pose(param, trial_count: int) -> None:
    init_quat_wxyz = np.asarray(param.init_obj_qpos_[3:7], dtype=np.float64)
    init_rotation = _quat_wxyz_to_rotation(init_quat_wxyz)
    target_rotation = init_rotation * _build_trial_flip_delta(trial_count)
    param.target_q_ = _rotation_to_quat_wxyz(target_rotation)


def _clean_params_dict(params: Namespace) -> Dict[str, object]:
    cleaned = {}
    for key, value in vars(params).items():
        if key.startswith("search_") or key in OPTIMIZER_ONLY_FIELDS:
            continue
        if isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def _build_runtime_base_args(base_args: Namespace) -> Namespace:
    runtime_args = Namespace(**vars(base_args))
    default_values = {
        "sim_device": "cuda:0",
        "graphics_device_id": 0,
        "cartesian_step": 0.05,
        "cartesian_joint_stiffness": 100.0,
        "cartesian_dls_lambda": 1e-4,
        "osc_pos_stiffness": 2000.0,
        "osc_ori_stiffness": 400.0,
        "headless": True,
        "result_dir": "",
        "parallel_envs": 1,
        "max_rollout_length": 3000,
        "consecutive_success_steps": 20,
        "success_pos_threshold": 0.02,
        "success_quat_threshold": 0.015,
        "early_stop_trials": 11,
        "min_rate_threshold": 0.4,
        "max_rate_threshold": 0.55,
        "delta_threshold": 0.01,
    }
    for key, value in default_values.items():
        if not hasattr(runtime_args, key):
            setattr(runtime_args, key, value)
    return runtime_args


if ISAAC_IMPORT_ERROR is None:
    class ParallelTrialIsaacFrankaOSCSimulator:
        def __init__(self, params_list, headless=False, sim_device="cuda:0", graphics_device_id=0):
            self.params_ = list(params_list)
            if not self.params_:
                raise ValueError("params_list must not be empty")

            self.num_envs = len(self.params_)
            self.break_out_signal_ = False
            self.dyn_paused_ = False
            self.viewer_ = None
            self.gym = gymapi.acquire_gym()

            compute_id = int(sim_device.split(":")[-1]) if ":" in sim_device else 0
            sim_params = gymapi.SimParams()
            sim_params.dt = 0.01
            sim_params.substeps = 2
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
            sim_params.use_gpu_pipeline = False
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.contact_offset = 0.01
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.use_gpu = compute_id >= 0

            self.sim = self.gym.create_sim(compute_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
            if self.sim is None:
                raise RuntimeError("Failed to create Isaac Gym sim")

            plane = gymapi.PlaneParams()
            plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.add_ground(self.sim, plane)

            self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
            self._load_assets(self.repo_root, self.params_[0])
            self._create_envs_and_actors()
            self._configure_franka()
            self.gym.prepare_sim(self.sim)
            self._build_body_index_cache()
            self.reset_all_envs(self.params_)

            if not headless:
                self.viewer_ = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
                if self.viewer_ is not None:
                    cam_pos = gymapi.Vec3(3.0, 2.0, 1.6)
                    cam_target = gymapi.Vec3(0.6, 0.0, 0.35)
                    self.gym.viewer_camera_look_at(self.viewer_, self.envs[0], cam_pos, cam_target)

        def _load_assets(self, repo_root, param):
            franka_asset_root, franka_asset_file = IsaacFrankaOSCSimulator._get_franka_asset_info(repo_root)

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

            mesh_obj_urdf_rel, _, mesh_asset_root = IsaacFrankaOSCSimulator._prepare_mesh_urdf_assets(
                repo_root, getattr(param, "mesh_path_", None)
            )
            obj_opts = gymapi.AssetOptions()
            obj_opts.density = 200.0
            obj_opts.use_mesh_materials = True
            self.obj_asset = self.gym.load_asset(self.sim, mesh_asset_root, mesh_obj_urdf_rel, obj_opts)

        def _create_envs_and_actors(self):
            self.envs = []
            self.franka_actors = []
            self.table_actors = []
            self.obj_actors = []

            spacing = 2.5
            env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
            env_upper = gymapi.Vec3(spacing, spacing, 2.0)
            num_per_row = int(np.ceil(np.sqrt(self.num_envs)))

            franka_pose = gymapi.Transform()
            franka_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
            franka_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(1.2, 0.0, 0.175)
            table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            obj_pose = gymapi.Transform()
            obj_pose.p = gymapi.Vec3(0.45, 0.0, 0.375)
            obj_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

            for env_idx in range(self.num_envs):
                env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
                franka_actor = self.gym.create_actor(env, self.franka_asset, franka_pose, "franka", env_idx, 0)
                table_actor = self.gym.create_actor(env, self.table_asset, table_pose, "table", env_idx, 0)
                obj_actor = self.gym.create_actor(env, self.obj_asset, obj_pose, "obj", env_idx, 0)
                self.gym.set_rigid_body_color(env, obj_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.6, 1.0))

                self.envs.append(env)
                self.franka_actors.append(franka_actor)
                self.table_actors.append(table_actor)
                self.obj_actors.append(obj_actor)

        def _configure_franka(self):
            self.franka_dof_count = None
            self._joint_targets = []
            self.torque_limits = None
            self.nullspace_stiffness = 10.0
            self.home_q = np.array(self.params_[0].init_robot_qpos_, dtype=np.float32)
            pos_stiffness = float(getattr(self.params_[0], "osc_pos_stiffness_", 150.0))
            ori_stiffness = float(getattr(self.params_[0], "osc_ori_stiffness_", 400.0))
            self.osc_task_kp = np.array(
                [pos_stiffness, pos_stiffness, pos_stiffness, ori_stiffness, ori_stiffness, ori_stiffness],
                dtype=np.float32,
            )
            self.osc_task_kd = (2.0 * np.sqrt(self.osc_task_kp)).astype(np.float32)

            for env_idx, env in enumerate(self.envs):
                franka_actor = self.franka_actors[env_idx]
                dof_props = self.gym.get_actor_dof_properties(env, franka_actor)
                dof_props["driveMode"][:7] = gymapi.DOF_MODE_EFFORT
                dof_props["stiffness"][:7] = 0.0
                dof_props["damping"][:7] = 0.0

                if dof_props["driveMode"].shape[0] >= 9:
                    dof_props["driveMode"][7:] = gymapi.DOF_MODE_POS
                    dof_props["stiffness"][7:] = 1.0e6
                    dof_props["damping"][7:] = 1.0e2

                self.gym.set_actor_dof_properties(env, franka_actor, dof_props)

                if self.franka_dof_count is None:
                    self.franka_dof_count = self.gym.get_actor_dof_count(env, franka_actor)
                    self.torque_limits = np.array(dof_props["effort"][:7], dtype=np.float32)

                joint_targets = np.zeros(self.franka_dof_count, dtype=np.float32)
                joint_targets[:7] = np.array(self.params_[env_idx].init_robot_qpos_, dtype=np.float32)
                if self.franka_dof_count >= 9:
                    joint_targets[7] = 0.04
                    joint_targets[8] = 0.04
                self._joint_targets.append(joint_targets)

            self.position_d = [np.zeros(3, dtype=np.float32) for _ in range(self.num_envs)]
            self.orientation_d = [np.eye(3, dtype=np.float32) for _ in range(self.num_envs)]
            self.p_d = [np.zeros(3, dtype=np.float32) for _ in range(self.num_envs)]
            self.R_d = [np.eye(3, dtype=np.float32) for _ in range(self.num_envs)]
            self.R_d_hold = [np.eye(3, dtype=np.float32) for _ in range(self.num_envs)]
            self.q_d_nullspace = [
                np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)
                for _ in range(self.num_envs)
            ]

        def _build_body_index_cache(self):
            self.franka_body_names = self.gym.get_actor_rigid_body_names(self.envs[0], self.franka_actors[0])
            self.franka_body_count = len(self.franka_body_names)

            jac = self.gym.acquire_jacobian_tensor(self.sim, "franka")
            mm = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
            dof_states = self.gym.acquire_dof_state_tensor(self.sim)
            self._jacobian = gymtorch.wrap_tensor(jac)
            self._mm = gymtorch.wrap_tensor(mm)
            self._dof_state = gymtorch.wrap_tensor(dof_states)

            self._jacobian_body_offset = self.franka_body_count - int(self._jacobian.shape[1])
            if self._jacobian_body_offset not in (0, 1):
                raise RuntimeError(
                    f"Unexpected Franka Jacobian body shape: rigid_bodies={self.franka_body_count}, "
                    f"jacobian_bodies={int(self._jacobian.shape[1])}"
                )

            self.ee_body_name = "attachment" if "attachment" in self.franka_body_names else "panda_link7"
            self.ee_body_local_idx = self.franka_body_names.index(self.ee_body_name)
            self.fingertip_body_idx = (
                self.franka_body_names.index("fingertip")
                if "fingertip" in self.franka_body_names
                else self.ee_body_local_idx
            )
            self.task_body_local_idx = self.fingertip_body_idx

            self.obj_body_indices = []
            self.table_body_indices = []
            self.franka_body_indices = []
            self.franka_dof_sim_indices = []

            for env_idx, env in enumerate(self.envs):
                obj_body_idx = self.gym.get_actor_rigid_body_index(
                    env, self.obj_actors[env_idx], 0, gymapi.DOMAIN_SIM
                )
                table_body_idx = self.gym.get_actor_rigid_body_index(
                    env, self.table_actors[env_idx], 0, gymapi.DOMAIN_SIM
                )
                franka_body_count = self.gym.get_actor_rigid_body_count(env, self.franka_actors[env_idx])
                franka_body_indices = set(
                    self.gym.get_actor_rigid_body_index(env, self.franka_actors[env_idx], i, gymapi.DOMAIN_SIM)
                    for i in range(franka_body_count)
                )
                dof_sim_indices = np.array(
                    [
                        self.gym.get_actor_dof_index(env, self.franka_actors[env_idx], i, gymapi.DOMAIN_SIM)
                        for i in range(self.franka_dof_count)
                    ],
                    dtype=np.int64,
                )

                self.obj_body_indices.append(obj_body_idx)
                self.table_body_indices.append(table_body_idx)
                self.franka_body_indices.append(franka_body_indices)
                self.franka_dof_sim_indices.append(dof_sim_indices)

            self._effort_control = torch.zeros(
                (self._dof_state.shape[0],), dtype=torch.float32, device=self._dof_state.device
            )

        def _refresh_osc_tensors(self):
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)

        def _jacobian_body_index(self, local_body_idx):
            jac_idx = int(local_body_idx) - int(self._jacobian_body_offset)
            if jac_idx < 0 or jac_idx >= int(self._jacobian.shape[1]):
                raise IndexError(
                    f"Rigid body index {local_body_idx} maps to invalid jacobian index {jac_idx}; "
                    f"offset={self._jacobian_body_offset}, jacobian_bodies={int(self._jacobian.shape[1])}"
                )
            return jac_idx

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

        @staticmethod
        def _orientation_error(r_current, r_desired):
            r_error = r_current.T @ r_desired
            err_quat_xyzw = Rotation.from_matrix(r_error).as_quat()
            if err_quat_xyzw[3] < 0.0:
                err_quat_xyzw = -err_quat_xyzw
            return (r_current @ err_quat_xyzw[:3]).astype(np.float32)

        @staticmethod
        def _project_to_rotation_matrix(rotation_matrix):
            u, _, vh = np.linalg.svd(np.asarray(rotation_matrix, dtype=np.float64))
            r_proj = u @ vh
            if np.linalg.det(r_proj) < 0.0:
                u[:, -1] *= -1.0
                r_proj = u @ vh
            return r_proj.astype(np.float32)

        @staticmethod
        def _safe_inverse(matrix, rcond=1e-5):
            matrix = np.asarray(matrix, dtype=np.float64)
            try:
                inv = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(matrix, rcond=rcond)
            return inv.astype(np.float32)

        def _simulate_once(self):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if self.viewer_ is not None:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer_, self.sim, True)
                self.gym.sync_frame_time(self.sim)

        def reset_env(self, env_idx, param):
            env = self.envs[env_idx]

            obj_state = self.gym.get_actor_rigid_body_states(env, self.obj_actors[env_idx], gymapi.STATE_ALL)
            init_obj = np.array(param.init_obj_qpos_, dtype=np.float32)
            obj_state["pose"]["p"][0] = (float(init_obj[0]), float(init_obj[1]), float(init_obj[2]))
            obj_state["pose"]["r"][0] = (float(init_obj[4]), float(init_obj[5]), float(init_obj[6]), float(init_obj[3]))
            obj_state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
            obj_state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
            self.gym.set_actor_rigid_body_states(env, self.obj_actors[env_idx], obj_state, gymapi.STATE_ALL)

            dof_states = self.gym.get_actor_dof_states(env, self.franka_actors[env_idx], gymapi.STATE_ALL)
            dof_states["pos"][:] = 0.0
            dof_states["vel"][:] = 0.0
            dof_states["pos"][:7] = np.array(param.init_robot_qpos_, dtype=np.float32)
            if self.franka_dof_count >= 9:
                dof_states["pos"][7] = 0.04
                dof_states["pos"][8] = 0.04
            self.gym.set_actor_dof_states(env, self.franka_actors[env_idx], dof_states, gymapi.STATE_ALL)

            self._joint_targets[env_idx][:7] = np.array(param.init_robot_qpos_, dtype=np.float32)
            if self.franka_dof_count >= 9:
                self._joint_targets[env_idx][7] = 0.04
                self._joint_targets[env_idx][8] = 0.04
            self.gym.set_actor_dof_position_targets(env, self.franka_actors[env_idx], self._joint_targets[env_idx])

        def reset_all_envs(self, params_list):
            for env_idx, param in enumerate(params_list):
                self.reset_env(env_idx, param)

            for _ in range(8):
                self._simulate_once()

            self._sync_desired_pose_with_current()

        def _sync_desired_pose_with_current(self):
            self._refresh_osc_tensors()
            for env_idx in range(self.num_envs):
                p, r = self.get_end_effector_pos(env_idx)
                self.position_d[env_idx] = p.copy()
                self.orientation_d[env_idx] = r.copy()
                self.p_d[env_idx] = p.copy()
                self.R_d[env_idx] = r.copy()
                self.R_d_hold[env_idx] = r.copy()
                self.q_d_nullspace[env_idx] = np.array(
                    [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32
                )

        def get_physx_contacts(self, env_idx):
            contacts_raw = self.gym.get_env_rigid_contacts(self.envs[env_idx])
            contacts = []
            if contacts_raw is None:
                return contacts

            for c in contacts_raw:
                body0 = IsaacFrankaSimulator._contact_field(c, "body0", "bodyA", default=-1)
                body1 = IsaacFrankaSimulator._contact_field(c, "body1", "bodyB", default=-1)
                if body0 is None or body1 is None:
                    continue

                separation = IsaacFrankaSimulator._contact_field(c, "separation", "distance", default=0.0)
                normal_raw = IsaacFrankaSimulator._contact_field(
                    c, "normal", default=np.array([0.0, 0.0, 1.0], dtype=np.float32)
                )
                pos_field = IsaacFrankaSimulator._contact_field(c, "pos", "position", default=None)

                normal = _extract_vec3(normal_raw)
                nrm = np.linalg.norm(normal)
                normal = normal / nrm if nrm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
                pos = _extract_vec3(pos_field) if pos_field is not None else None

                contacts.append(
                    {
                        "body0": int(body0),
                        "body1": int(body1),
                        "separation": float(separation if separation is not None else 0.0),
                        "normal": normal.astype(np.float32),
                        "pos": pos,
                    }
                )

            return contacts

        def _get_body_pose(self, env_idx, local_idx):
            states = self.gym.get_actor_rigid_body_states(
                self.envs[env_idx], self.franka_actors[env_idx], gymapi.STATE_POS
            )
            p = _extract_vec3(states["pose"]["p"][local_idx])
            quat_xyzw = _extract_quat_xyzw(states["pose"]["r"][local_idx])
            r = _quat_xyzw_to_matrix(quat_xyzw)
            return p.astype(np.float32), r.astype(np.float32)

        def get_end_effector_pos(self, env_idx):
            return self._get_body_pose(env_idx, self.task_body_local_idx)

        def get_current_joint_position(self, env_idx):
            dof_states = self.gym.get_actor_dof_states(
                self.envs[env_idx], self.franka_actors[env_idx], gymapi.STATE_POS
            )
            return np.array(dof_states["pos"][:7], dtype=np.float32)

        def get_current_joint_velocity(self, env_idx):
            dof_states = self.gym.get_actor_dof_states(
                self.envs[env_idx], self.franka_actors[env_idx], gymapi.STATE_VEL
            )
            return np.array(dof_states["vel"][:7], dtype=np.float32)

        def get_state(self, env_idx):
            obj_state = self.gym.get_actor_rigid_body_states(self.envs[env_idx], self.obj_actors[env_idx], gymapi.STATE_POS)
            obj_pos = _extract_vec3(obj_state["pose"]["p"][0])
            quat_xyzw = _extract_quat_xyzw(obj_state["pose"]["r"][0])
            obj_quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
            q = self.get_current_joint_position(env_idx)
            return np.hstack([obj_pos, obj_quat_wxyz, q]).astype(np.float32)

        def _get_task_jacobian(self, env_idx):
            jac_task_idx = self._jacobian_body_index(self.task_body_local_idx)
            return np.array(self._jacobian[env_idx, jac_task_idx, :, :7], dtype=np.float32)

        def _get_arm_mass_matrix(self, env_idx):
            mm = self._mm
            if mm.ndim == 3:
                mm = mm[env_idx]
            return np.array(mm[:7, :7], dtype=np.float32)

        def begin_control_step(self):
            self._effort_control.zero_()

        def _apply_arm_torque(self, env_idx, tau):
            tau = np.asarray(tau, dtype=np.float32).reshape(-1)
            tau = np.clip(tau, -self.torque_limits, self.torque_limits)
            dof_indices = self.franka_dof_sim_indices[env_idx]
            self._effort_control[dof_indices[:7]] = torch.as_tensor(
                tau, dtype=torch.float32, device=self._effort_control.device
            )
            if self.franka_dof_count >= 9:
                self.gym.set_actor_dof_position_targets(
                    self.envs[env_idx], self.franka_actors[env_idx], self._joint_targets[env_idx]
                )
            return tau

        def _compute_task_space_error(self, env_idx, p_curr, r_curr):
            dpose = np.zeros(6, dtype=np.float32)
            dpose[:3] = self.position_d[env_idx] - p_curr
            dpose[3:] = self._orientation_error(r_curr, self.orientation_d[env_idx])
            return dpose

        def compute_cartesian_impedance_control(self, env_idx):
            q = self.get_current_joint_position(env_idx)
            qd = self.get_current_joint_velocity(env_idx)
            p_curr, r_curr = self.get_end_effector_pos(env_idx)
            jac = self._get_task_jacobian(env_idx)
            mm = self._get_arm_mass_matrix(env_idx)
            mm_inv = self._safe_inverse(mm)
            m_task_inv = jac @ mm_inv @ jac.T
            m_task = self._safe_inverse(m_task_inv)

            dpose = self._compute_task_space_error(env_idx, p_curr, r_curr)
            ee_velocity = jac @ qd
            tau_task = jac.T @ (m_task @ (self.osc_task_kp * dpose - self.osc_task_kd * ee_velocity))

            j_task_inv = m_task @ jac @ mm_inv
            q_error = (self.q_d_nullspace[env_idx] - q + np.pi) % (2.0 * np.pi) - np.pi
            tau_nullspace = (
                2.0 * np.sqrt(self.nullspace_stiffness) * (-qd) + self.nullspace_stiffness * q_error
            ).astype(np.float32)
            tau_nullspace = mm @ tau_nullspace
            tau_nullspace = (np.eye(7, dtype=np.float32) - jac.T @ j_task_inv) @ tau_nullspace

            tau_d = tau_task + tau_nullspace
            return np.clip(tau_d, -self.torque_limits, self.torque_limits)

        def queue_cartesian_action(self, env_idx, cmd):
            cmd = np.asarray(cmd, dtype=np.float32).reshape(-1)
            if cmd.size != 3:
                raise ValueError(f"Invalid action dimension: {cmd.size}. Expected 3.")

            p_curr, _ = self.get_end_effector_pos(env_idx)
            self.p_d[env_idx] = p_curr + cmd
            self.R_d[env_idx] = self.R_d_hold[env_idx].copy()

            p_target = np.asarray(self.p_d[env_idx], dtype=np.float32).copy()
            r_target = self._project_to_rotation_matrix(self.R_d_hold[env_idx])
            self.q_d_nullspace[env_idx] = np.array(
                [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32
            )
            self.position_d[env_idx] = p_target
            self.orientation_d[env_idx] = r_target
            tau = self.compute_cartesian_impedance_control(env_idx)
            self._apply_arm_torque(env_idx, tau)
            return tau

        def advance(self):
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))
            self._simulate_once()

        def close(self):
            if self.viewer_ is not None:
                self.gym.destroy_viewer(self.viewer_)
                self.viewer_ = None
            if self.sim is not None:
                self.gym.destroy_sim(self.sim)
                self.sim = None


def detect_once_parallel(param, simulator, env_idx):
    full_q = simulator.get_state(env_idx)
    full_q = np.asarray(full_q, dtype=np.float32)
    obj_pos = full_q[0:3]
    obj_quat_wxyz = full_q[3:7]

    nv = param.n_qvel_
    max_ncon = param.max_ncon_
    phi_vec = np.ones((max_ncon * 4,), dtype=np.float32)
    jac_mat = np.zeros((max_ncon * 4, nv), dtype=np.float32)
    jac_mat_env = np.zeros((max_ncon * 4, nv), dtype=np.float32)
    con_pos_list = []
    if_contact = False

    contacts = simulator.get_physx_contacts(env_idx)
    mu = float(param.mu_object_)
    contact_sep_threshold = float(getattr(param, "if_contact_separation_threshold_", 0.0))
    row_idx = 0
    row_env_idx = 0
    obj_body_idx = simulator.obj_body_indices[env_idx]
    table_body_idx = simulator.table_body_indices[env_idx]
    franka_body_indices = simulator.franka_body_indices[env_idx]

    for c in contacts:
        b0 = c["body0"]
        b1 = c["body1"]
        sep = float(c["separation"])
        n_raw = c["normal"]
        cpos = c["pos"]

        involves_obj = (b0 == obj_body_idx) or (b1 == obj_body_idx)
        if not involves_obj:
            continue

        if b1 == obj_body_idx:
            n_raw = -n_raw

        if cpos is None:
            cpos = obj_pos + 0.01 * n_raw
        else:
            cpos = np.asarray(cpos, dtype=np.float32)

        r_obj = cpos - obj_pos
        j_obj = np.zeros((3, nv), dtype=np.float32)
        j_obj[:, 0:3] = np.eye(3, dtype=np.float32)
        j_obj[:, 3:6] = -simulator._skew(r_obj)

        j_other = np.zeros((3, nv), dtype=np.float32)
        other_sim_idx = b1 if b0 == obj_body_idx else b0
        if other_sim_idx in franka_body_indices:
            j_other[:, 6:9] = np.eye(3, dtype=np.float32)

        j_rel_point = j_obj - j_other
        n, t1, t2 = _tangent_basis_from_normal(n_raw)
        con_jac = np.array(_contact_jacobian(n, t1, t2, j_rel_point, mu), dtype=np.float32)

        other_is_franka = (b0 in franka_body_indices) or (b1 in franka_body_indices)
        if other_is_franka and sep <= contact_sep_threshold:
            if_contact = True
        if other_is_franka and row_idx < max_ncon:
            phi_vec[4 * row_idx : 4 * row_idx + 4] = 0.5 * sep
            jac_mat[4 * row_idx : 4 * row_idx + 4, :] = con_jac
            row_idx += 1

        other_is_table = (b0 == table_body_idx) or (b1 == table_body_idx)
        if other_is_table and row_idx < max_ncon:
            phi_vec[4 * row_idx : 4 * row_idx + 4] = 0.5 * sep
            jac_mat[4 * row_idx : 4 * row_idx + 4, :] = con_jac
            row_idx += 1
        if other_is_table and row_env_idx < max_ncon:
            jac_mat_env[4 * row_env_idx : 4 * row_env_idx + 4, :6] = con_jac[:, :6]
            row_env_idx += 1
            quat_xyzw = np.array(
                [obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]],
                dtype=np.float32,
            )
            r_obj_to_world = Rotation.from_quat(quat_xyzw).as_matrix()
            con_pos_local = r_obj_to_world.T @ (cpos - obj_pos)
            con_pos_list.append(np.array(con_pos_local, dtype=np.float32))

    if row_env_idx == 0:
        dist_table = float(obj_pos[2] - float(param.table_height))
        if dist_table < 0.02:
            n_t = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            t1_t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            t2_t = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            j_rel_table = np.concatenate([np.eye(3), np.zeros((3, 6))], axis=1)
            con_jac_table = np.array(_contact_jacobian(n_t, t1_t, t2_t, j_rel_table, mu), dtype=np.float32)
            jac_mat_env[0:4, :6] = con_jac_table[:, :6]
            con_pos_list.append(np.array([0.0, 0.0, -0.025], dtype=np.float32))

    return phi_vec, jac_mat, con_pos_list, jac_mat_env, if_contact


class HyperparameterOptimizer:
    def __init__(self, base_args: Namespace, num_trials: int = 100, max_workers: int = 4, max_param_num: int = 20):
        self.base_args = _build_runtime_base_args(base_args)
        self.num_trials = int(num_trials)
        self.max_workers = int(max_workers)
        self.max_param_num = int(max_param_num)
        self.parallel_envs = max(1, int(getattr(self.base_args, "parallel_envs", 1)))
        self.max_rollout_length = int(getattr(self.base_args, "max_rollout_length", 3000))
        self.consecutive_success_steps = int(getattr(self.base_args, "consecutive_success_steps", 20))
        self.success_pos_threshold = float(getattr(self.base_args, "success_pos_threshold", 0.02))
        self.success_quat_threshold = float(getattr(self.base_args, "success_quat_threshold", 0.015))
        self.early_stop_trials = int(getattr(self.base_args, "early_stop_trials", 11))
        self.min_rate_threshold = float(getattr(self.base_args, "min_rate_threshold", 0.4))
        self.max_rate_threshold = float(getattr(self.base_args, "max_rate_threshold", 0.55))
        self.delta_threshold = float(getattr(self.base_args, "delta_threshold", 0.01))
        self.results = []
        self.best_params = None
        self.best_result = None
        self.best_success_rate = -1.0
        self.start_time = None
        self.total_combinations = 0
        self.tested_param_num = 0
        self.search_space = self._build_search_space()

        result_root = getattr(self.base_args, "result_dir", "")
        if not result_root:
            result_root = os.path.join(current_dir, "param_search_results_isaac", self.base_args.obj)
        self.save_dir = os.path.abspath(result_root)
        os.makedirs(self.save_dir, exist_ok=True)

    def _worker_settings(self) -> Dict[str, object]:
        return {
            "num_trials": self.num_trials,
            "max_param_num": self.max_param_num,
            "parallel_envs": self.parallel_envs,
            "max_rollout_length": self.max_rollout_length,
            "consecutive_success_steps": self.consecutive_success_steps,
            "success_pos_threshold": self.success_pos_threshold,
            "success_quat_threshold": self.success_quat_threshold,
            "early_stop_trials": self.early_stop_trials,
            "min_rate_threshold": self.min_rate_threshold,
            "max_rate_threshold": self.max_rate_threshold,
            "delta_threshold": self.delta_threshold,
        }

    @staticmethod
    def _close_queue_handle(queue_handle):
        try:
            queue_handle.close()
        except Exception:
            pass
        try:
            queue_handle.join_thread()
        except Exception:
            pass

    @staticmethod
    def _try_get_queue_message(queue_handle):
        try:
            return queue_handle.get_nowait()
        except queue.Empty:
            return None
        except (EOFError, OSError):
            return None

    def _build_failed_result(self, params: Namespace, error_message: str, elapsed_time: float = 0.0) -> Dict[str, object]:
        elapsed_time = max(float(elapsed_time), 0.0)
        return {
            "params": _clean_params_dict(params),
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "avg_time": elapsed_time,
            "trials": [
                {
                    "trial": -1,
                    "success": False,
                    "steps": 0,
                    "time": elapsed_time,
                    "final_pos_error": None,
                    "final_quat_error": None,
                    "error": error_message,
                }
            ],
        }

    def _finalize_parameter_result(self, result: Dict[str, object], param_pbar):
        self._record_result(result)
        self.tested_param_num += 1
        param_pbar.update(1)
        param_pbar.set_postfix(best=f"{max(self.best_success_rate, 0.0):.1%}")
        if self.tested_param_num % 5 == 0 or self.tested_param_num == self.total_combinations:
            self.save_results()

    def _build_search_space(self) -> Dict[str, object]:
        return {
            "strategy": "grid_search",
            "search_fields": list(SEARCH_PARAM_NAMES),
            "candidate_values": {
                field_name: [
                    self._normalize_search_value(candidate_value)
                    for candidate_value in candidate_values
                ]
                for field_name, candidate_values in SEARCH_PARAMETER_GRID.items()
            },
            "sampling": "random_subset_if_needed",
        }

    @staticmethod
    def _normalize_search_value(value: float) -> float:
        return round(max(float(value), 0.0), SEARCH_VALUE_PRECISION)

    def generate_parameters(self) -> List[Namespace]:
        param_combinations = []
        candidate_lists = [SEARCH_PARAMETER_GRID[field_name] for field_name in SEARCH_PARAM_NAMES]

        for candidate_values in product(*candidate_lists):
            params = Namespace(**vars(self.base_args))
            for field_name, candidate_value in zip(SEARCH_PARAM_NAMES, candidate_values):
                setattr(params, field_name, self._normalize_search_value(candidate_value))
            param_combinations.append(params)

        if len(param_combinations) > self.max_param_num:
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[: self.max_param_num]

        self.total_combinations = len(param_combinations)
        return param_combinations

    @staticmethod
    def _is_better_result(candidate: Dict[str, object], incumbent: Dict[str, object]) -> bool:
        if incumbent is None:
            return True

        candidate_rate = float(candidate["success_rate"])
        incumbent_rate = float(incumbent["success_rate"])
        if candidate_rate != incumbent_rate:
            return candidate_rate > incumbent_rate

        if candidate_rate <= 0.0:
            return False

        candidate_steps = float(candidate["avg_steps"])
        incumbent_steps = float(incumbent["avg_steps"])
        if candidate_steps != incumbent_steps:
            return candidate_steps < incumbent_steps

        return float(candidate["avg_time"]) < float(incumbent["avg_time"])

    def _record_result(self, result: Dict[str, object]):
        self.results.append(result)
        if self._is_better_result(result, self.best_result):
            self.best_result = result
            self.best_success_rate = float(result["success_rate"])
            self.best_params = dict(result["params"])

    def _build_trial_param(self, params: Namespace, trial_count: int):
        param = ExplicitMPCParams(params, rand_seed=trial_count, target_type="rotation", mpc_model="explicit")
        trial_pose_override = TRIAL_POSE_OVERRIDES.get(trial_count)
        if trial_pose_override is not None:
            if "init_obj_qpos" in trial_pose_override:
                param.init_obj_qpos_ = trial_pose_override["init_obj_qpos"].copy()
            if "target_p" in trial_pose_override:
                param.target_p_ = trial_pose_override["target_p"].copy()
        _apply_trial_flip_goal_pose(param, trial_count)
        param.use_jax_contact_ = False
        param = adapt_param_for_cartesian_solver(param, params)
        param.sim_dt_ = 0.01
        param.osc_pos_stiffness_ = float(params.osc_pos_stiffness)
        param.osc_ori_stiffness_ = float(params.osc_ori_stiffness)
        return param

    @staticmethod
    def _init_trial_result(trial_count: int) -> Dict[str, object]:
        return {
            "trial": int(trial_count),
            "success": False,
            "steps": 0,
            "time": 0.0,
            "final_pos_error": None,
            "final_quat_error": None,
            "error": None,
        }

    def _evaluate_single_trial(self, params: Namespace, trial_count: int) -> Dict[str, object]:
        trial_start = time.time()
        env = None
        trial_result = self._init_trial_result(trial_count)

        try:
            if ISAAC_IMPORT_ERROR is not None:
                raise RuntimeError(
                    "Isaac Gym Python package is unavailable. "
                    f"Original import error: {ISAAC_IMPORT_ERROR}"
                )

            param = self._build_trial_param(params, trial_count)

            contact = ContactIsaacCartesian(param)
            env = IsaacFrankaOSCSimulator(
                param,
                headless=bool(params.headless),
                sim_device=params.sim_device,
                graphics_device_id=int(params.graphics_device_id),
            )
            env.show_target_object_pose(param.target_p_, param.target_q_)

            mpc = MPCExplicitIsaac(param) if param.mpc_model == "explicit" else MPCImplicit(param)

            rollout_step = 0
            consecutive_success_time = 0
            verify_cost = 0
            current_x = np.zeros(7, dtype=np.float32)
            current_x[3] = 1.0
            low_err_coef = float(params.low_err_coef)
            upper_err_coef = float(params.upper_err_coef)
            consecutive_detect_time = 0
            consecutive_contact_time = 0

            while rollout_step < self.max_rollout_length:
                curr_q = env.get_state()
                ee_pos = env.get_end_effector_pos()[0].copy()
                curr_x_solver = np.hstack([curr_q[:7], ee_pos]).astype(np.float32)

                phi_vec, jac_mat, _, jac_mat_env, if_contact = contact.detect_once(env)
                quaternion = [curr_q[4], curr_q[5], curr_q[6], curr_q[3]]
                r_obj_to_world = Rotation.from_quat(quaternion).as_matrix()
                gravity = np.hstack([r_obj_to_world.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3)])

                target_pos_ = param.target_p_ - curr_q[0:3]
                target_pos_[2] = 0.0
                target_quat_local = rotations.quaternion_multiply(
                    rotations.quaternion_conjugate(curr_q[3:7]),
                    param.target_q_,
                )
                target_pose_local = np.hstack([r_obj_to_world.T @ target_pos_, target_quat_local])

                param.lambda_optimizer.update_Jacobian(jac_mat_env)
                visible_point_idx = param.lambda_optimizer.get_availble_point_idx(
                    curr_q[0:3],
                    r_obj_to_world,
                    param.target_p_,
                    float(params.ground_height_threshold),
                )

                best_contact_point, normal, min_error, max_error, curr_ori_coef = (
                    param.lambda_optimizer.choose_contact_points(
                        target_pose_local,
                        current_x,
                        gravity,
                        visible_point_idx,
                    )
                )

                attract_point_world = r_obj_to_world @ best_contact_point + curr_q[0:3]
                original_height = attract_point_world[2]
                attract_point_world -= float(params.attract_point_comp) * (r_obj_to_world @ normal)
                attract_point_world[2] = max(attract_point_world[2], original_height)

                local_point = r_obj_to_world.T @ (ee_pos - curr_q[0:3])
                p_arm_local, _, _, error, _ = param.lambda_optimizer.optimize_control_input(
                    target_pose_local,
                    current_x,
                    gravity,
                    local_point,
                )
                p_arm_world = r_obj_to_world @ p_arm_local + curr_q[:3]

                if verify_cost:
                    low_err_coef = float(params.low_err_coef)
                elif np.linalg.norm(ee_pos - attract_point_world) < 5e-2:
                    low_err_coef *= 1.1

                delta_error = max(float(max_error) - float(min_error), 1e-6)
                improvement = (float(max_error) - float(error)) / delta_error

                if verify_cost:
                    upper_err_coef = float(params.upper_err_coef)
                elif np.linalg.norm(ee_pos[:2] - attract_point_world[:2]) < 5e-2:
                    upper_err_coef *= 0.8

                consecutive_contact_time = consecutive_contact_time + int(bool(if_contact)) if verify_cost else 0
                if (not verify_cost and improvement > upper_err_coef) or (
                    verify_cost and improvement <= low_err_coef
                ):
                    consecutive_detect_time += 1
                else:
                    consecutive_detect_time = 0

                if not verify_cost and consecutive_detect_time >= 5:
                    verify_cost = 1
                elif verify_cost and consecutive_detect_time >= 1 and consecutive_contact_time >= 5:
                    verify_cost = 0

                sol = mpc.plan_once(
                    param.target_p_,
                    param.target_q_,
                    curr_x_solver,
                    phi_vec,
                    jac_mat,
                    verify_cost_param=verify_cost,
                    virtual_point=attract_point_world,
                    contact_point=p_arm_world,
                    curr_ori_coef=float(curr_ori_coef),
                    sol_guess=param.sol_guess_,
                )
                param.sol_guess_ = sol["sol_guess"]
                action = np.asarray(sol["action"], dtype=np.float32)

                env.step(action)
                rollout_step += 1

                curr_q = env.get_state()
                pos_err = float(metrics.comp_pos_error(curr_q[0:3], param.target_p_))
                quat_err = float(metrics.comp_quat_error(curr_q[3:7], param.target_q_))
                if pos_err < self.success_pos_threshold and quat_err < self.success_quat_threshold:
                    consecutive_success_time += 1
                else:
                    consecutive_success_time = 0

                if consecutive_success_time > self.consecutive_success_steps:
                    break

            final_q = env.get_state()
            trial_result["success"] = bool(rollout_step < self.max_rollout_length)
            trial_result["steps"] = int(rollout_step)
            trial_result["final_pos_error"] = float(metrics.comp_pos_error(final_q[0:3], param.target_p_))
            trial_result["final_quat_error"] = float(metrics.comp_quat_error(final_q[3:7], param.target_q_))

        except Exception as exc:
            trial_result["error"] = str(exc)

        finally:
            if env is not None:
                env.close()
            trial_result["time"] = float(time.time() - trial_start)

        return trial_result

    def _evaluate_trial_batch(self, params: Namespace, trial_indices: List[int]) -> List[Dict[str, object]]:
        batch_start = time.time()
        results = [self._init_trial_result(trial_idx) for trial_idx in trial_indices]
        if ISAAC_IMPORT_ERROR is not None:
            err = (
                "Isaac Gym Python package is unavailable. "
                f"Original import error: {ISAAC_IMPORT_ERROR}"
            )
            for result in results:
                result["error"] = err
            return results

        env = None
        active = [True] * len(trial_indices)

        try:
            trial_params = [self._build_trial_param(params, trial_idx) for trial_idx in trial_indices]
            mpc_list = [
                MPCExplicitIsaac(param_i) if param_i.mpc_model == "explicit" else MPCImplicit(param_i)
                for param_i in trial_params
            ]
            env = ParallelTrialIsaacFrankaOSCSimulator(
                trial_params,
                headless=bool(params.headless),
                sim_device=params.sim_device,
                graphics_device_id=int(params.graphics_device_id),
            )

            rollout_steps = [0 for _ in trial_indices]
            consecutive_success = [0 for _ in trial_indices]
            verify_cost = [0 for _ in trial_indices]
            current_x = [np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32) for _ in trial_indices]
            low_err_coef = [float(params.low_err_coef) for _ in trial_indices]
            upper_err_coef = [float(params.upper_err_coef) for _ in trial_indices]
            consecutive_detect_time = [0 for _ in trial_indices]
            consecutive_contact_time = [0 for _ in trial_indices]

            while any(active):
                env.begin_control_step()
                env._refresh_osc_tensors()
                queued_envs = 0

                for env_idx, is_active in enumerate(active):
                    if not is_active:
                        continue

                    try:
                        param_i = trial_params[env_idx]
                        curr_q = env.get_state(env_idx)
                        ee_pos = env.get_end_effector_pos(env_idx)[0].copy()
                        curr_x_solver = np.hstack([curr_q[:7], ee_pos]).astype(np.float32)

                        phi_vec, jac_mat, _, jac_mat_env, if_contact = detect_once_parallel(param_i, env, env_idx)
                        quaternion = [curr_q[4], curr_q[5], curr_q[6], curr_q[3]]
                        r_obj_to_world = Rotation.from_quat(quaternion).as_matrix()
                        gravity = np.hstack([r_obj_to_world.T @ param_i.gravity_[:3] * param_i.obj_mass_, np.zeros(3)])

                        target_pos_ = param_i.target_p_ - curr_q[0:3]
                        target_pos_[2] = 0.0
                        target_quat_local = rotations.quaternion_multiply(
                            rotations.quaternion_conjugate(curr_q[3:7]),
                            param_i.target_q_,
                        )
                        target_pose_local = np.hstack([r_obj_to_world.T @ target_pos_, target_quat_local])

                        param_i.lambda_optimizer.update_Jacobian(jac_mat_env)
                        visible_point_idx = param_i.lambda_optimizer.get_availble_point_idx(
                            curr_q[0:3],
                            r_obj_to_world,
                            param_i.target_p_,
                            float(params.ground_height_threshold),
                        )

                        best_contact_point, normal, min_error, max_error, curr_ori_coef = (
                            param_i.lambda_optimizer.choose_contact_points(
                                target_pose_local,
                                current_x[env_idx],
                                gravity,
                                visible_point_idx,
                            )
                        )

                        attract_point_world = r_obj_to_world @ best_contact_point + curr_q[0:3]
                        original_height = attract_point_world[2]
                        attract_point_world -= float(params.attract_point_comp) * (r_obj_to_world @ normal)
                        attract_point_world[2] = max(attract_point_world[2], original_height)

                        local_point = r_obj_to_world.T @ (ee_pos - curr_q[0:3])
                        p_arm_local, _, _, error, _ = param_i.lambda_optimizer.optimize_control_input(
                            target_pose_local,
                            current_x[env_idx],
                            gravity,
                            local_point,
                        )
                        p_arm_world = r_obj_to_world @ p_arm_local + curr_q[:3]

                        if verify_cost[env_idx]:
                            low_err_coef[env_idx] = float(params.low_err_coef)
                        elif np.linalg.norm(ee_pos - attract_point_world) < 5e-2:
                            low_err_coef[env_idx] *= 1.1

                        delta_error = max(float(max_error) - float(min_error), 1e-6)
                        improvement = (float(max_error) - float(error)) / delta_error

                        if verify_cost[env_idx]:
                            upper_err_coef[env_idx] = float(params.upper_err_coef)
                        elif np.linalg.norm(ee_pos[:2] - attract_point_world[:2]) < 5e-2:
                            upper_err_coef[env_idx] *= 0.8

                        consecutive_contact_time[env_idx] = (
                            consecutive_contact_time[env_idx] + int(bool(if_contact))
                            if verify_cost[env_idx]
                            else 0
                        )
                        if (not verify_cost[env_idx] and improvement > upper_err_coef[env_idx]) or (
                            verify_cost[env_idx] and improvement <= low_err_coef[env_idx]
                        ):
                            consecutive_detect_time[env_idx] += 1
                        else:
                            consecutive_detect_time[env_idx] = 0

                        if not verify_cost[env_idx] and consecutive_detect_time[env_idx] >= 5:
                            verify_cost[env_idx] = 1
                        elif (
                            verify_cost[env_idx]
                            and consecutive_detect_time[env_idx] >= 1
                            and consecutive_contact_time[env_idx] >= 5
                        ):
                            verify_cost[env_idx] = 0

                        sol = mpc_list[env_idx].plan_once(
                            param_i.target_p_,
                            param_i.target_q_,
                            curr_x_solver,
                            phi_vec,
                            jac_mat,
                            verify_cost_param=verify_cost[env_idx],
                            virtual_point=attract_point_world,
                            contact_point=p_arm_world,
                            curr_ori_coef=float(curr_ori_coef),
                            sol_guess=param_i.sol_guess_,
                        )
                        param_i.sol_guess_ = sol["sol_guess"]
                        action = np.asarray(sol["action"], dtype=np.float32)
                        env.queue_cartesian_action(env_idx, action)
                        queued_envs += 1

                    except Exception as exc:
                        results[env_idx]["steps"] = int(rollout_steps[env_idx])
                        results[env_idx]["time"] = float(time.time() - batch_start)
                        results[env_idx]["error"] = str(exc)
                        active[env_idx] = False

                if queued_envs == 0:
                    break

                env.advance()
                curr_time = time.time()

                for env_idx, is_active in enumerate(active):
                    if not is_active:
                        continue

                    rollout_steps[env_idx] += 1
                    curr_q = env.get_state(env_idx)
                    pos_err = float(metrics.comp_pos_error(curr_q[0:3], trial_params[env_idx].target_p_))
                    quat_err = float(metrics.comp_quat_error(curr_q[3:7], trial_params[env_idx].target_q_))

                    if pos_err < self.success_pos_threshold and quat_err < self.success_quat_threshold:
                        consecutive_success[env_idx] += 1
                    else:
                        consecutive_success[env_idx] = 0

                    is_success = consecutive_success[env_idx] > self.consecutive_success_steps
                    is_timeout = rollout_steps[env_idx] >= self.max_rollout_length
                    if is_success or is_timeout:
                        results[env_idx]["success"] = bool(is_success)
                        results[env_idx]["steps"] = int(rollout_steps[env_idx])
                        results[env_idx]["time"] = float(curr_time - batch_start)
                        results[env_idx]["final_pos_error"] = pos_err
                        results[env_idx]["final_quat_error"] = quat_err
                        active[env_idx] = False

        except Exception as exc:
            for env_idx, is_active in enumerate(active):
                if is_active:
                    results[env_idx]["time"] = float(time.time() - batch_start)
                    results[env_idx]["error"] = str(exc)

        finally:
            if env is not None:
                env.close()

            elapsed = float(time.time() - batch_start)
            for result in results:
                if result["time"] <= 0.0:
                    result["time"] = elapsed

        return results

    def evaluate_parameters(
        self,
        params: Namespace,
        current_index: int = None,
        total_combinations: int = None,
        show_trial_progress: bool = True,
    ) -> Dict[str, object]:
        if current_index is None:
            current_index = self.tested_param_num + 1
        if total_combinations is None:
            total_combinations = self.total_combinations if self.total_combinations else self.max_param_num

        remaining = max(total_combinations - current_index, 0)

        print(
            f"\n{'=' * 72}\n"
            f"开始测试参数配置 {current_index}/{total_combinations}\n"
            f"剩余搜索预算: {remaining}\n"
            f"参数: {self._format_params(params)}\n"
            f"{'=' * 72}"
        )

        trial_results = []
        success_count = 0
        consecutive_fail_num = 0
        rate_threshold = self.min_rate_threshold
        trial_cursor = 0
        stop_early = False
        trial_desc = f"Trials {current_index}/{total_combinations}"
        with tqdm(
            total=self.num_trials,
            desc=trial_desc,
            leave=False,
            file=sys.stderr,
            disable=not show_trial_progress,
        ) as trial_pbar:
            while trial_cursor < self.num_trials and not stop_early:
                batch_indices = list(
                    range(trial_cursor, min(self.num_trials, trial_cursor + self.parallel_envs))
                )
                if self.parallel_envs > 1:
                    batch_results = self._evaluate_trial_batch(params, batch_indices)
                else:
                    batch_results = [self._evaluate_single_trial(params, trial_idx) for trial_idx in batch_indices]

                for trial_result in batch_results:
                    trial_results.append(trial_result)
                    trial_pbar.update(1)

                    if trial_result["success"]:
                        success_count += 1
                        consecutive_fail_num = 0
                    else:
                        consecutive_fail_num += 1

                    tested_trials = len(trial_results)
                    current_rate = success_count / float(tested_trials)
                    trial_pbar.set_postfix(
                        success=f"{current_rate:.1%}",
                        last="ok" if trial_result["success"] else "fail",
                    )
                    print(
                        f"Trial {trial_result['trial'] + 1}/{self.num_trials} | "
                        f"结果: {'成功' if trial_result['success'] else '失败'} | "
                        f"当前成功率: {current_rate:.1%} | "
                        f"步数: {trial_result['steps']} | "
                        f"耗时: {trial_result['time']:.2f}s"
                    )

                    trial_index = int(trial_result["trial"])
                    if trial_index >= self.early_stop_trials - 1 and current_rate < rate_threshold:
                        print(
                            f"早停: 参数 {self._format_params(params)} 在前 {tested_trials} 次试验中的成功率 "
                            f"{current_rate:.1%} 低于阈值 {rate_threshold:.1%}"
                        )
                        stop_early = True
                        break

                    if consecutive_fail_num >= 5:
                        print("早停: 连续失败次数达到 5")
                        stop_early = True
                        break

                    if trial_index % 10 == 0 and trial_index != 0:
                        rate_threshold = min(rate_threshold + self.delta_threshold, self.max_rate_threshold)

                trial_cursor += len(batch_indices)

        success_rate = success_count / float(len(trial_results)) if trial_results else 0.0
        avg_steps = float(np.mean([t["steps"] for t in trial_results])) if trial_results else 0.0
        avg_time = float(np.mean([t["time"] for t in trial_results])) if trial_results else 0.0

        return {
            "params": _clean_params_dict(params),
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_time": avg_time,
            "trials": trial_results,
        }

    def _format_params(self, params: Namespace) -> str:
        return ", ".join(
            f"{field_name}={self._normalize_search_value(getattr(params, field_name))}"
            for field_name in SEARCH_PARAM_NAMES
        )

    def run_optimization(self):
        self.start_time = time.time()
        params_list = self.generate_parameters()

        print(
            f"\n{'=' * 72}\n"
            f"开始 Isaac 参数搜索\n"
            f"搜索参数: {', '.join(SEARCH_PARAM_NAMES)}\n"
            f"搜索方式: 与 param_detect.py 一致的离散网格搜索\n"
            f"候选值: {self.search_space['candidate_values']}\n"
            f"最大参数配置数: {self.max_param_num} | 实际参数组合数: {self.total_combinations} | 每组试验次数: {self.num_trials}\n"
            f"并行方式: 多进程参数组搜索 | max_workers: {self.max_workers}\n"
            f"每个进程内并行 env 数: {self.parallel_envs}\n"
            f"结果目录: {self.save_dir}\n"
            f"说明: 不同参数组会分发到不同操作系统进程；"
            f"若 parallel_envs=1，则总并发仿真个数与 max_workers 一致。\n"
            f"{'=' * 72}"
        )

        if ISAAC_IMPORT_ERROR is not None:
            print(
                "当前环境缺少 Isaac Gym Python 包，无法真正启动参数搜索。\n"
                f"原始导入错误: {ISAAC_IMPORT_ERROR}"
            )
            return

        try:
            worker_settings = self._worker_settings()
            ctx = mp.get_context("spawn")
            pending_jobs = deque(
                (index, params, 0)
                for index, params in enumerate(params_list, start=1)
            )
            active_jobs = []

            with tqdm(total=self.total_combinations, desc="Parameter Sets", file=sys.stderr) as param_pbar:
                while pending_jobs or active_jobs:
                    while pending_jobs and len(active_jobs) < self.max_workers:
                        index, params, retry_count = pending_jobs.popleft()
                        result_queue = ctx.Queue(maxsize=1)
                        process = ctx.Process(
                            target=_worker_process_entry,
                            args=(
                                result_queue,
                                vars(self.base_args),
                                worker_settings,
                                vars(params),
                                index,
                                self.total_combinations,
                            ),
                        )
                        process.start()
                        active_jobs.append(
                            {
                                "index": index,
                                "params": params,
                                "retry_count": retry_count,
                                "process": process,
                                "queue": result_queue,
                                "start_time": time.time(),
                            }
                        )

                    finished_any = False
                    remaining_jobs = []

                    for job in active_jobs:
                        process = job["process"]
                        result_queue = job["queue"]
                        message = self._try_get_queue_message(result_queue)

                        if message is not None:
                            process.join(timeout=0.1)
                            self._close_queue_handle(result_queue)

                            if message.get("status") == "ok":
                                self._finalize_parameter_result(message["result"], param_pbar)
                            else:
                                error_message = message.get("error", "Worker raised an unknown exception")
                                print(
                                    f"\n参数配置 {job['index']}/{self.total_combinations} 测试失败: "
                                    f"{self._format_params(job['params'])} | 错误: {error_message}"
                                )
                                if message.get("traceback"):
                                    print(message["traceback"])
                                failed_result = self._build_failed_result(
                                    job["params"],
                                    error_message,
                                    elapsed_time=time.time() - job["start_time"],
                                )
                                self._finalize_parameter_result(failed_result, param_pbar)

                            finished_any = True
                            continue

                        if process.is_alive():
                            remaining_jobs.append(job)
                            continue

                        process.join(timeout=0.1)
                        message = self._try_get_queue_message(result_queue)
                        self._close_queue_handle(result_queue)

                        if message is not None:
                            if message.get("status") == "ok":
                                self._finalize_parameter_result(message["result"], param_pbar)
                            else:
                                error_message = message.get("error", "Worker raised an unknown exception")
                                print(
                                    f"\n参数配置 {job['index']}/{self.total_combinations} 测试失败: "
                                    f"{self._format_params(job['params'])} | 错误: {error_message}"
                                )
                                if message.get("traceback"):
                                    print(message["traceback"])
                                failed_result = self._build_failed_result(
                                    job["params"],
                                    error_message,
                                    elapsed_time=time.time() - job["start_time"],
                                )
                                self._finalize_parameter_result(failed_result, param_pbar)
                            finished_any = True
                            continue

                        exitcode = process.exitcode
                        crash_reason = f"Worker process exited with code {exitcode}"
                        if exitcode == -9:
                            crash_reason += " (possible OOM kill or GPU memory exhaustion)"
                        elif exitcode is not None and exitcode < 0:
                            crash_reason += " (terminated by signal)"

                        if job["retry_count"] < WORKER_CRASH_RETRY_LIMIT:
                            next_retry = job["retry_count"] + 1
                            print(
                                f"\n参数配置 {job['index']}/{self.total_combinations} 子进程异常退出，"
                                f"准备重试 {next_retry}/{WORKER_CRASH_RETRY_LIMIT}: "
                                f"{self._format_params(job['params'])} | 原因: {crash_reason}"
                            )
                            pending_jobs.appendleft((job["index"], job["params"], next_retry))
                        else:
                            print(
                                f"\n参数配置 {job['index']}/{self.total_combinations} 测试失败: "
                                f"{self._format_params(job['params'])} | 错误: {crash_reason}"
                            )
                            failed_result = self._build_failed_result(
                                job["params"],
                                crash_reason,
                                elapsed_time=time.time() - job["start_time"],
                            )
                            self._finalize_parameter_result(failed_result, param_pbar)

                        finished_any = True

                    active_jobs = remaining_jobs

                    if not finished_any and active_jobs:
                        time.sleep(WORKER_POLL_INTERVAL_SEC)

        except KeyboardInterrupt:
            print("\n捕获到中断信号，正在保存当前搜索结果...")
            self.save_results()
            return

        self.save_results()
        print(
            f"\n{'=' * 72}\n"
            f"搜索完成 | 总耗时: {time.time() - self.start_time:.1f}s\n"
            f"最佳成功率: {self.best_success_rate:.1%}\n"
            f"最佳参数: {self.best_params}\n"
            f"{'=' * 72}"
        )

    def save_results(self):
        payload = {
            "best_params": self.best_params,
            "best_success_rate": self.best_success_rate,
            "search_space": self.search_space,
            "all_results": self.results,
        }

        with open(os.path.join(self.save_dir, "param_search_results.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        csv_path = os.path.join(self.save_dir, "results_summary.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(SUMMARY_FIELDS + ["success_rate", "avg_steps", "avg_time", "tested_trials"]) + "\n")
            for result in self.results:
                params = result["params"]
                row = [str(params.get(field, "")) for field in SUMMARY_FIELDS]
                row.extend(
                    [
                        f"{result['success_rate']:.6f}",
                        f"{result['avg_steps']:.2f}",
                        f"{result['avg_time']:.4f}",
                        str(len(result["trials"])),
                    ]
                )
                f.write(",".join(row) + "\n")


def _evaluate_parameter_set_worker(
    base_args_dict: Dict[str, object],
    worker_settings: Dict[str, object],
    params_dict: Dict[str, object],
    current_index: int,
    total_combinations: int,
) -> Tuple[Dict[str, object], float]:
    base_args = Namespace(**base_args_dict)
    params = Namespace(**params_dict)
    optimizer = HyperparameterOptimizer(
        base_args,
        num_trials=int(worker_settings["num_trials"]),
        max_workers=1,
        max_param_num=int(worker_settings["max_param_num"]),
    )
    optimizer.parallel_envs = max(1, int(worker_settings["parallel_envs"]))
    optimizer.max_rollout_length = int(worker_settings["max_rollout_length"])
    optimizer.consecutive_success_steps = int(worker_settings["consecutive_success_steps"])
    optimizer.success_pos_threshold = float(worker_settings["success_pos_threshold"])
    optimizer.success_quat_threshold = float(worker_settings["success_quat_threshold"])
    optimizer.early_stop_trials = int(worker_settings["early_stop_trials"])
    optimizer.min_rate_threshold = float(worker_settings["min_rate_threshold"])
    optimizer.max_rate_threshold = float(worker_settings["max_rate_threshold"])
    optimizer.delta_threshold = float(worker_settings["delta_threshold"])
    result = optimizer.evaluate_parameters(
        params,
        current_index=current_index,
        total_combinations=total_combinations,
        show_trial_progress=False,
    )
    return result, float(result["success_rate"])


def _worker_process_entry(
    result_queue,
    base_args_dict: Dict[str, object],
    worker_settings: Dict[str, object],
    params_dict: Dict[str, object],
    current_index: int,
    total_combinations: int,
):
    try:
        result, success_rate = _evaluate_parameter_set_worker(
            base_args_dict,
            worker_settings,
            params_dict,
            current_index,
            total_combinations,
        )
        result_queue.put(
            {
                "status": "ok",
                "result": result,
                "success_rate": float(success_rate),
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


if __name__ == "__main__":
    base_args = Namespace(
        obj="mug",
        attract_coef=1.0,
        reject_coef=0.001,
        contact_coef=0.5,
        contact_cost_param=0,
        model_param=15,
        reject_dis=0.01,
        attract_point_comp=0.1,
        ground_height_threshold=0.012,
        sample_num=70,
        pos_coef=1,
        ori_coef=0.0005,
        low_err_coef=0.3,
        upper_err_coef=1,
    )

    optimizer = HyperparameterOptimizer(
        base_args,
        num_trials=100,
        max_workers=8,
        max_param_num=1000,
    )
    optimizer.run_optimization()
