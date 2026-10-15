"""Stage-1 isolation: Isaac floating fingertip + test_0902 --rollout algorithm.

Same lambda_optimizer, verify, tightness, and MPCExplicit as
``examples/mpc/fingertips/test/test_0902.py --rollout``.  The only swap is
MuJoCo -> PhysX for a 3-DOF sliding sphere (no Franka, no MPPI, no async
planner).  If flip still fails here, the gap is the Isaac contact/physics
stack, not the Franka/OSC/MPPI wrapping.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaacgym import gymapi  # must precede torch
import numpy as np
from scipy.spatial.transform import Rotation

# test_mpc_isaac imports gymtorch unless the planner-only guard is set.
os.environ["SCSP_PLANNER_ONLY"] = "1"

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(current_dir)
while os.path.basename(parent_dir) != "scsp-robot":
    _next_dir = os.path.dirname(parent_dir)
    if _next_dir == parent_dir:
        raise RuntimeError("scsp-robot repo root not found from %s" % current_dir)
    parent_dir = _next_dir
sys.path.insert(0, parent_dir)

from planning.acados_env import ensure_acados_env
ensure_acados_env()

from examples.mpc.fingertips.test.params import ExplicitMPCParams
from examples.mpc.fingertips.test.test_0902 import (
    add_rollout_via_args,
    compute_rollout_contact_via,
    ContactValueTracker,
    ModelCostConfidence,
    SmoothedApproachVia,
    _lambda_pose_cost,
    _predicted_object_pose,
    _protect_destination_dwell,
    _rollout_dwell_assignment,
    _same_contact_patch,
    _should_observe_model_cost,
    _verify_is_chatter,
    _x_plus_is_usable,
)
from examples.mpc.franka.ik2.test_mpc_isaac import ContactIsaacCartesian
from planning.mpc_explicit import MPCExplicit
from utils import metrics


FINGERTIP_RADIUS = 0.01
FINGERTIP_MASS = 4.0 / 3.0 * np.pi * (FINGERTIP_RADIUS ** 3) * 1000.0
FINGERTIP_KP = 100.0
FINGERTIP_KD = 2.0
SIM_DT = 0.002
OBJECT_FRICTION = 0.9
TABLE_FRICTION = 0.5
FINGERTIP_FRICTION = 1.5


def _extract_vec3(v):
    if isinstance(v, np.void) and getattr(v, "dtype", None) is not None and v.dtype.names is not None:
        return np.array([float(v["x"]), float(v["y"]), float(v["z"])], dtype=np.float32)
    arr = np.asarray(v)
    if arr.dtype.names is not None:
        return np.array([float(arr["x"]), float(arr["y"]), float(arr["z"])], dtype=np.float32)
    arr = arr.reshape(-1)
    return np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=np.float32)


def _extract_quat_xyzw(q):
    if isinstance(q, np.void) and getattr(q, "dtype", None) is not None and q.dtype.names is not None:
        return np.array([float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"])], dtype=np.float32)
    arr = np.asarray(q)
    if arr.dtype.names is not None:
        return np.array([float(arr["x"]), float(arr["y"]), float(arr["z"]), float(arr["w"])], dtype=np.float32)
    arr = arr.reshape(-1)
    return np.array([float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])], dtype=np.float32)


def _contact_field(c, *names, default=None):
    if not isinstance(c, np.void):
        return default
    if getattr(c, "dtype", None) is None or c.dtype.names is None:
        return default
    for name in names:
        if name in c.dtype.names:
            return c[name]
    return default


def _set_shape_friction(gym, env, actor, friction):
    props = gym.get_actor_rigid_shape_properties(env, actor)
    for prop in props:
        prop.friction = float(friction)
        prop.torsion_friction = float(friction)
        prop.rolling_friction = float(friction)
    gym.set_actor_rigid_shape_properties(env, actor, props)


def _set_actor_mass(gym, env, actor, mass):
    props = gym.get_actor_rigid_body_properties(env, actor)
    for prop in props:
        prop.mass = float(mass)
    gym.set_actor_rigid_body_properties(env, actor, props, True)


def _skew(v):
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float32,
    )


def _prepare_mesh_urdf(repo_root, mesh_path, visual_mesh_path=None, mass=0.01):
    def _abs_mesh(path):
        abs_path = path if os.path.isabs(path) else os.path.join(repo_root, path)
        abs_path = os.path.abspath(abs_path)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError("Mesh file not found: %s" % abs_path)
        return abs_path

    import shutil

    collision_abs = _abs_mesh(mesh_path)
    visual_abs = _abs_mesh(visual_mesh_path or mesh_path)
    mesh_asset_root = os.path.join(repo_root, "envs", "assets", "objects", "_isaac_tmp")
    os.makedirs(mesh_asset_root, exist_ok=True)
    visual_rel = "stage1_obj_visual.stl"
    collision_rel = "stage1_obj_collision.stl"
    shutil.copy2(visual_abs, os.path.join(mesh_asset_root, visual_rel))
    shutil.copy2(collision_abs, os.path.join(mesh_asset_root, collision_rel))
    obj_urdf_rel = "stage1_obj_dynamic.urdf"
    target_urdf_rel = "stage1_obj_ghost.urdf"
    obj_urdf = f"""<?xml version="1.0"?>
<robot name="mesh_obj">
  <link name="base">
    <inertial>
      <mass value="{float(mass)}"/>
      <inertia ixx="1.5e-6" ixy="0" ixz="0" iyy="1.5e-6" iyz="0" izz="1.5e-6"/>
    </inertial>
    <visual>
      <geometry><mesh filename="{visual_rel}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{collision_rel}" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
"""
    ghost_urdf = f"""<?xml version="1.0"?>
<robot name="mesh_target_ghost">
  <link name="base">
    <visual>
      <geometry><mesh filename="{visual_rel}" scale="1 1 1"/></geometry>
    </visual>
  </link>
</robot>
"""
    with open(os.path.join(mesh_asset_root, obj_urdf_rel), "w", encoding="ascii") as f:
        f.write(obj_urdf)
    with open(os.path.join(mesh_asset_root, target_urdf_rel), "w", encoding="ascii") as f:
        f.write(ghost_urdf)
    return obj_urdf_rel, target_urdf_rel, mesh_asset_root


def _fingertip_urdf(repo_root):
    asset_root = os.path.join(repo_root, "envs", "assets", "objects", "_isaac_tmp")
    os.makedirs(asset_root, exist_ok=True)
    rel = "fingertip_xyz_slide.urdf"
    mass = float(FINGERTIP_MASS)
    inertia = 0.4 * mass * (FINGERTIP_RADIUS ** 2)
    urdf = f"""<?xml version="1.0"?>
<robot name="fingertip_xyz">
  <link name="base"/>
  <link name="slide_x">
    <inertial>
      <mass value="1e-5"/>
      <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>
    </inertial>
  </link>
  <joint name="joint_x" type="prismatic">
    <parent link="base"/>
    <child link="slide_x"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" effort="10" velocity="10"/>
  </joint>
  <link name="slide_y">
    <inertial>
      <mass value="1e-5"/>
      <inertia ixx="1e-9" ixy="0" ixz="0" iyy="1e-9" iyz="0" izz="1e-9"/>
    </inertial>
  </link>
  <joint name="joint_y" type="prismatic">
    <parent link="slide_x"/>
    <child link="slide_y"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="10" velocity="10"/>
  </joint>
  <link name="fingertip">
    <inertial>
      <mass value="{mass}"/>
      <inertia ixx="{inertia}" ixy="0" ixz="0" iyy="{inertia}" iyz="0" izz="{inertia}"/>
    </inertial>
    <visual>
      <geometry><sphere radius="{FINGERTIP_RADIUS}"/></geometry>
      <material name="tip_red"><color rgba="0.8 0.2 0.2 1"/></material>
    </visual>
    <collision>
      <geometry><sphere radius="{FINGERTIP_RADIUS}"/></geometry>
    </collision>
  </link>
  <joint name="joint_z" type="prismatic">
    <parent link="slide_y"/>
    <child link="fingertip"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.02" upper="1" effort="10" velocity="10"/>
  </joint>
</robot>
"""
    with open(os.path.join(asset_root, rel), "w", encoding="ascii") as f:
        f.write(urdf)
    return asset_root, rel


class IsaacFingertipSimulator:
    """PhysX 3-DOF sliding sphere with the MjSimulator control interface."""

    def __init__(self, param, headless=False, realtime=False):
        self.param_ = param
        self.param_.table_height = 0.0
        self.break_out_signal_ = False
        self.dyn_paused_ = False
        self.viewer_ = None
        self.headless_ = bool(headless)
        self.realtime_ = bool(realtime)
        self.sim_dt_ = float(getattr(param, "sim_dt_", SIM_DT))
        self.frame_skip_ = int(getattr(param, "frame_skip_", 10))
        self.fingertip_mass = float(FINGERTIP_MASS)

        self.gym = gymapi.acquire_gym()
        sim_params = gymapi.SimParams()
        sim_params.dt = self.sim_dt_
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = False
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.bounce_threshold_velocity = 2.0 * 9.81 * self.sim_dt_
        sim_params.physx.max_depenetration_velocity = 10.0
        sim_params.physx.use_gpu = False
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym sim")

        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane.static_friction = TABLE_FRICTION
        plane.dynamic_friction = TABLE_FRICTION
        self.gym.add_ground(self.sim, plane)

        self.env = self.gym.create_env(
            self.sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 1.0), 1
        )
        self._create_actors()
        self.gym.prepare_sim(self.sim)
        self._build_index_cache()
        self.reset_mj_env()

        if not self.headless_:
            self.viewer_ = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer_ is not None:
                self.gym.viewer_camera_look_at(
                    self.viewer_, self.env,
                    gymapi.Vec3(0.35, -0.45, 0.28),
                    gymapi.Vec3(0.0, 0.0, 0.03),
                )

    def _create_actors(self):
        repo_root = parent_dir
        mesh_obj_rel, mesh_target_rel, mesh_root = _prepare_mesh_urdf(
            repo_root,
            self.param_.mesh_path_,
            getattr(self.param_, "visual_mesh_path_", self.param_.mesh_path_),
            mass=float(getattr(self.param_, "obj_mass_", 0.01)),
        )
        obj_opts = gymapi.AssetOptions()
        obj_opts.density = 200.0
        obj_opts.use_mesh_materials = True
        if hasattr(obj_opts, "vhacd_enabled"):
            obj_opts.vhacd_enabled = False
        self.obj_asset = self.gym.load_asset(self.sim, mesh_root, mesh_obj_rel, obj_opts)

        target_opts = gymapi.AssetOptions()
        target_opts.fix_base_link = True
        target_opts.disable_gravity = True
        self.target_asset = self.gym.load_asset(self.sim, mesh_root, mesh_target_rel, target_opts)

        tip_root, tip_rel = _fingertip_urdf(repo_root)
        tip_opts = gymapi.AssetOptions()
        tip_opts.fix_base_link = True
        tip_opts.disable_gravity = True
        tip_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        self.tip_asset = self.gym.load_asset(self.sim, tip_root, tip_rel, tip_opts)

        marker_opts = gymapi.AssetOptions()
        marker_opts.fix_base_link = True
        marker_opts.disable_gravity = True
        self.marker_asset = self.gym.create_sphere(self.sim, 0.008, marker_opts)

        identity = gymapi.Transform()
        self.tip_actor = self.gym.create_actor(self.env, self.tip_asset, identity, "fingertip", 0, 0)
        self.obj_actor = self.gym.create_actor(self.env, self.obj_asset, identity, "obj", 0, 0)
        self.target_actor = self.gym.create_actor(self.env, self.target_asset, identity, "target", 1, 1)
        self.via_actor = self.gym.create_actor(self.env, self.marker_asset, identity, "via", 1, 1)
        self.best_actor = self.gym.create_actor(self.env, self.marker_asset, identity, "best", 1, 1)

        _set_actor_mass(self.gym, self.env, self.obj_actor, float(getattr(self.param_, "obj_mass_", 0.01)))
        _set_shape_friction(self.gym, self.env, self.obj_actor, OBJECT_FRICTION)
        _set_shape_friction(self.gym, self.env, self.tip_actor, FINGERTIP_FRICTION)
        self.gym.set_rigid_body_color(
            self.env, self.obj_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.6, 1.0)
        )
        self.gym.set_rigid_body_color(
            self.env, self.via_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.1, 0.1)
        )
        self.gym.set_rigid_body_color(
            self.env, self.best_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 1.0, 0.1)
        )

        dof_props = self.gym.get_actor_dof_properties(self.env, self.tip_actor)
        dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][:] = FINGERTIP_KP
        dof_props["damping"][:] = FINGERTIP_KD
        # MuJoCo applies F=100*cmd-2*vel (~0.5 N at 5 mm).  The URDF effort
        # cap of 10 N lets PhysX PD shove the mesh once the target sits
        # inside the object.
        if "effort" in dof_props.dtype.names:
            dof_props["effort"][:] = 2.0
        self.gym.set_actor_dof_properties(self.env, self.tip_actor, dof_props)
        self.tip_dof_count = int(self.gym.get_actor_dof_count(self.env, self.tip_actor))

    def _build_index_cache(self):
        self.obj_body_idx = self.gym.get_actor_rigid_body_index(
            self.env, self.obj_actor, 0, gymapi.DOMAIN_SIM
        )
        tip_names = list(self.gym.get_actor_rigid_body_names(self.env, self.tip_actor))
        self.tip_local_idx = tip_names.index("fingertip") if "fingertip" in tip_names else len(tip_names) - 1
        self.tip_body_idx = self.gym.get_actor_rigid_body_index(
            self.env, self.tip_actor, self.tip_local_idx, gymapi.DOMAIN_SIM
        )
        self.tip_body_env_idx = self.gym.get_actor_rigid_body_index(
            self.env, self.tip_actor, self.tip_local_idx, gymapi.DOMAIN_ENV
        )
        self.table_body_idx = -1
        self.fingertip_radius = FINGERTIP_RADIUS
        self.franka_body_name_to_index = {"fingertip": int(self.tip_body_idx)}

    def _set_actor_pose(self, actor, pos, quat_wxyz=None):
        state = self.gym.get_actor_rigid_body_states(self.env, actor, gymapi.STATE_ALL)
        state["pose"]["p"][0] = (float(pos[0]), float(pos[1]), float(pos[2]))
        if quat_wxyz is not None:
            state["pose"]["r"][0] = (
                float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]), float(quat_wxyz[0]),
            )
        state["vel"]["linear"][0] = (0.0, 0.0, 0.0)
        state["vel"]["angular"][0] = (0.0, 0.0, 0.0)
        self.gym.set_actor_rigid_body_states(self.env, actor, state, gymapi.STATE_ALL)

    def reset_mj_env(self):
        init_obj = np.asarray(self.param_.init_obj_qpos_, dtype=np.float64)
        self._set_actor_pose(self.obj_actor, init_obj[:3], init_obj[3:7])
        self._set_actor_pose(self.target_actor, self.param_.target_p_, self.param_.target_q_)
        dof = self.gym.get_actor_dof_states(self.env, self.tip_actor, gymapi.STATE_ALL)
        dof["pos"][:] = np.asarray(self.param_.init_robot_qpos_, dtype=np.float32)
        dof["vel"][:] = 0.0
        self.gym.set_actor_dof_states(self.env, self.tip_actor, dof, gymapi.STATE_ALL)
        for _ in range(4):
            self._simulate_once(draw=False)

    def set_goal(self, goal_pos=None, goal_quat=None):
        if goal_pos is None:
            goal_pos = self.param_.target_p_
        if goal_quat is None:
            goal_quat = self.param_.target_q_
        self._set_actor_pose(self.target_actor, goal_pos, goal_quat)

    def _simulate_once(self, draw=False):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if draw and self.viewer_ is not None:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer_, self.sim, True)
            if self.realtime_:
                self.gym.sync_frame_time(self.sim)
            if self.gym.query_viewer_has_closed(self.viewer_):
                self.break_out_signal_ = True

    def _tip_dof(self):
        dof = self.gym.get_actor_dof_states(self.env, self.tip_actor, gymapi.STATE_ALL)
        return np.array(dof["pos"][:3], dtype=np.float64), np.array(dof["vel"][:3], dtype=np.float64)

    def get_state(self):
        obj_state = self.gym.get_actor_rigid_body_states(self.env, self.obj_actor, gymapi.STATE_POS)
        obj_pos = _extract_vec3(obj_state["pose"]["p"][0])
        quat_xyzw = _extract_quat_xyzw(obj_state["pose"]["r"][0])
        obj_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        tip, _ = self._tip_dof()
        return np.hstack([obj_pos, obj_quat, tip]).astype(np.float32)

    def get_policy_state(self):
        return self.get_state()

    def get_end_effector_pos(self):
        tip, _ = self._tip_dof()
        return tip.astype(np.float32), np.eye(3, dtype=np.float32)

    @staticmethod
    def _skew(v):
        return _skew(v)

    def step(self, fts_pos_cmd):
        cmd = np.asarray(fts_pos_cmd, dtype=np.float64).reshape(3)
        for i in range(self.frame_skip_):
            tip, _ = self._tip_dof()
            # Re-anchor the PD target each substep so F ≈ kp*cmd − kd*vel,
            # matching MjSimulator's constant-force command over frame_skip.
            self.gym.set_actor_dof_position_targets(
                self.env, self.tip_actor, (tip + cmd).astype(np.float32)
            )
            self._simulate_once(draw=(not self.headless_) and (i + 1 == self.frame_skip_))

    def show_target(self, goal_pos=None):
        if goal_pos is not None:
            self._set_actor_pose(self.via_actor, goal_pos)

    def show_best_contact(self, point=None):
        if point is not None:
            self._set_actor_pose(self.best_actor, point)

    def _contact_bodies(self, c):
        body0 = _contact_field(c, "body0", "bodyA", default=-1)
        body1 = _contact_field(c, "body1", "bodyB", default=-1)
        if body0 is None or body1 is None:
            return None, None
        return int(body0), int(body1)

    def _contact_sep_normal(self, c):
        contact_offset = 0.001
        separation_raw = _contact_field(c, "separation", "distance", "minDist", default=None)
        speculative = False
        if separation_raw is None:
            overlap = float(_contact_field(c, "initial_overlap", default=0.0) or 0.0)
            if overlap > 1e-8:
                separation = -overlap
            else:
                # PhysX lists the pair.  overlap=0 is either touch or the
                # 1 mm offset; MuJoCo would report dist≈0 for a made contact.
                # Marking this speculative made 800 flip steps look contact-free.
                separation = 0.0
                speculative = False
        else:
            separation = float(separation_raw)
            speculative = separation > contact_offset + 1e-8
        normal = _extract_vec3(_contact_field(c, "normal", default=np.array([0.0, 0.0, 1.0])))
        nrm = float(np.linalg.norm(normal))
        normal = normal / nrm if nrm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return float(separation), bool(speculative), normal

    def get_body_pose_by_sim_index(self, sim_body_idx):
        if int(sim_body_idx) == int(self.obj_body_idx):
            state = self.gym.get_actor_rigid_body_states(self.env, self.obj_actor, gymapi.STATE_POS)
            return _extract_vec3(state["pose"]["p"][0]), _extract_quat_xyzw(state["pose"]["r"][0])
        if int(sim_body_idx) == int(self.tip_body_idx):
            tip, _ = self._tip_dof()
            return tip.astype(np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return None, None

    def _contact_world_pos(self, c, body0=None, body1=None, pose_cache=None):
        pos_field = _contact_field(c, "pos", "position", default=None)
        if pos_field is not None:
            return _extract_vec3(pos_field)
        if pose_cache is None:
            pose_cache = {}
        world_points = []
        for body_idx, name in ((body0, "localPos0"), (body1, "localPos1")):
            local_pos = _contact_field(c, name, "local_pos0" if name.endswith("0") else "local_pos1", default=None)
            if local_pos is None or body_idx is None:
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

    def close(self):
        if self.viewer_ is not None:
            self.gym.destroy_viewer(self.viewer_)
            self.viewer_ = None
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None


def _configure_rollout_param(param, args):
    param.rollout_press_patch = True
    param.quadratic_contact_track = True
    param.attract_coef = max(float(param.attract_coef), 20.0)
    param.field_cost_weight = 0.0
    param.contact_coef = max(float(param.contact_coef), float(param.attract_coef))
    param.lambda_optimizer.lock_contact_patch = False
    param.lambda_optimizer.contact_switch_confirm_steps = max(1, int(args.contact_switch_confirm_steps))
    param.lambda_optimizer.solver = "acados"
    param.table_height = 0.0
    param.physx_contact_offset_ = 0.001
    param.sim_dt_ = SIM_DT
    return param


def _contact_distance(contact, env):
    contact.detect_once(env)
    measured = contact.get_actual_fingertip_contact()
    if measured is None:
        return float("inf")
    return float(measured.get("dist", float("inf")))


def build_parser():
    parser = argparse.ArgumentParser(description="Stage-1 Isaac fingertip flip isolation.")
    add_rollout_via_args(parser)
    parser.add_argument("--viewer", dest="headless", action="store_false")
    parser.add_argument("--headless", dest="headless", action="store_true")
    parser.add_argument("--realtime", action="store_true", help="Lock the Isaac viewer to sim time.")
    parser.set_defaults(headless=False, rollout=True)
    return parser


def main(args=None):
    if args is None:
        args = build_parser().parse_args()
    args.solver = "acados"
    args.rollout = True
    trial_num = max(1, int(args.trial_num))
    success_pos_threshold = 0.02
    success_quat_threshold = 0.015
    max_rollout_length = max(1, int(args.max_rollout_length))
    success_rate = 0
    env = None
    contact = None

    print(
        "stage1_isaac_fingertip: same lambda/verify/tightness/MPC as test_0902 --rollout; "
        "PhysX 3-DOF sphere only; flip target=ground-rotation",
        flush=True,
    )

    for trial_count in range(trial_num):
        param = ExplicitMPCParams(args, rand_seed=trial_count, target_type="ground-rotation", model="explicit")
        param.torch_solver = "acados"
        param = _configure_rollout_param(param, args)
        mpc = MPCExplicit(param)
        if env is None:
            contact = ContactIsaacCartesian(param)
            env = IsaacFingertipSimulator(param, headless=args.headless, realtime=args.realtime)
        else:
            contact.param_ = param
            env.param_ = param
            env.set_goal(param.target_p_, param.target_q_)
            env.reset_mj_env()

        value_tracker = ContactValueTracker(
            tau=float(args.value_tau), rel_scale=float(args.value_rel_scale),
            rho=float(args.value_rho), alpha=float(args.value_alpha),
            beta=float(args.verify_beta), window_size=int(args.verify_window_size),
            confirm_steps=int(args.verify_enter_steps), min_hold_steps=int(args.verify_hold_steps),
            release_steps=int(args.verify_release_steps), accept_margin_ratio=0.05, accept_margin_abs=0.02,
        )
        model_cost_conf = ModelCostConfidence(
            threshold=float(args.model_cost_error_threshold),
            eps=float(args.model_cost_error_eps),
            min_steps=int(args.model_cost_error_min_steps),
        )
        mpc_step = max(1e-4, float(args.mpc_step_limit))
        approach_via = SmoothedApproachVia(rate=0.10, max_step=mpc_step, max_lead=mpc_step)
        arrived_hold = False
        arrived_dest_idx = None
        last_verify_cost = None
        last_accept_p_arm = False
        pred_reduction = None
        act_reduction = None
        c_now_cost = None
        pose_apply_count = 0
        choose_times = []
        rollout_step = 0
        consecutive_success_time = 0
        min_pos_err = float("inf")
        min_quat_err = float("inf")
        curr_q = env.get_state()

        while rollout_step < max_rollout_length and not env.break_out_signal_:
            curr_q = env.get_state()
            phi_vec, jac_mat, _con, jac_mat_env, if_contact = contact.detect_once(env)
            r_obj = Rotation.from_quat([curr_q[4], curr_q[5], curr_q[6], curr_q[3]]).as_matrix()
            gravity = np.hstack([r_obj.T @ param.gravity_[:3] * param.obj_mass_, np.zeros(3)])
            t0 = time.perf_counter()
            policy = compute_rollout_contact_via(
                param, args, curr_q, r_obj, gravity, jac_mat_env,
                FINGERTIP_RADIUS, value_tracker, model_cost_conf, approach_via,
                arrived_hold, arrived_dest_idx, floor_ground=0.012, floor_z=0.0,
            )
            rank_dt = time.perf_counter() - t0
            choose_times.append(rank_dt)
            arrived_hold = policy["arrived_hold"]
            arrived_dest_idx = policy["arrived_dest_idx"]
            verify_cost = policy["verify_cost"]
            verify_chatter = _verify_is_chatter(last_verify_cost, verify_cost)
            last_verify_cost = float(verify_cost)
            last_accept_p_arm = bool(policy["value_info"].get("accept_p_arm", False))
            escape_on = bool(policy["escape_on"])
            env.show_target(policy["mpc_virtual_point"])
            env.show_best_contact(policy["best_contact_world"])

            c_now_cost = _lambda_pose_cost(
                curr_q[:3], curr_q[3:7], param.target_p_, param.target_q_,
                param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef,
            )
            x_plus_opt = policy["x_plus_opt"]
            info = policy["info"]
            pred_reduction = None
            if x_plus_opt is not None and _x_plus_is_usable(x_plus_opt, info):
                pred_pos, pred_quat = _predicted_object_pose(curr_q[:7], x_plus_opt)
                pred_reduction = c_now_cost - _lambda_pose_cost(
                    pred_pos, pred_quat, param.target_p_, param.target_q_,
                    param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef,
                )

            t1 = time.perf_counter()
            sol = mpc.plan_once(
                param.target_p_, param.target_q_, curr_q, phi_vec, jac_mat,
                verify_cost_param=verify_cost,
                virtual_point=policy["mpc_virtual_point"],
                contact_point=policy["mpc_contact_point"],
                sol_guess=param.sol_guess_,
            )
            plan_dt = time.perf_counter() - t1
            param.sol_guess_ = sol["sol_guess"]
            action = np.asarray(sol["action"], dtype=np.float64).reshape(3)
            action_norm = float(np.linalg.norm(action))
            if action_norm > mpc_step and action_norm > 1e-9:
                action = action * (mpc_step / action_norm)
            env.step(action)

            contact_distance = _contact_distance(contact, env)
            post_tip = np.asarray(env.get_state()[7:10], dtype=float)
            on_exec_contact = False
            if np.isfinite(contact_distance) and contact_distance <= 0.003:
                on_exec_contact = float(np.linalg.norm(post_tip - policy["p_arm_world"])) <= 0.03
                if on_exec_contact:
                    pose_apply_count += 1

            print(
                f"step={rollout_step:04d} rank_dt={rank_dt:.4f} plan_dt={plan_dt:.4f} "
                f"verify={verify_cost:.3f} tight={model_cost_conf.tightness():.3f} "
                f"contact={int(if_contact)} dist={None if not np.isfinite(contact_distance) else round(float(contact_distance), 5)} "
                f"pos={metrics.comp_pos_error(curr_q[:3], param.target_p_):.4f} "
                f"quat={metrics.comp_quat_error(curr_q[3:7], param.target_q_):.4f} "
                f"tip={np.round(np.asarray(curr_q[7:10], dtype=float), 4).tolist()} "
                f"via={np.round(np.asarray(policy['mpc_virtual_point'], dtype=float), 4).tolist()} "
                f"du={round(float(np.linalg.norm(action)), 5)}",
                flush=True,
            )

            rollout_step += 1
            curr_q = env.get_state()
            pos_err_now = float(metrics.comp_pos_error(curr_q[:3], param.target_p_))
            quat_err_now = float(metrics.comp_quat_error(curr_q[3:7], param.target_q_))
            act_reduction = None
            if c_now_cost is not None:
                c_after = _lambda_pose_cost(
                    curr_q[:3], curr_q[3:7], param.target_p_, param.target_q_,
                    param.lambda_optimizer.pos_coef, param.lambda_optimizer.ori_coef,
                )
                act_reduction = c_now_cost - c_after
                opt = param.lambda_optimizer
                post_physical = bool(np.isfinite(contact_distance) and contact_distance <= 0.003)
                if post_physical and _should_observe_model_cost(
                    opt.has_delta_span(), getattr(opt, "last_pose_cost_now", None)
                ):
                    pred_delta = pred_reduction
                    if pred_delta is None or not np.isfinite(float(pred_delta)):
                        pred_delta = getattr(opt, "last_best_delta", None)
                    if pred_delta is None:
                        model_cost_conf.observe_unusable_prediction()
                    else:
                        pred_n = opt.normalize_cost_delta(pred_delta)
                        act_n = opt.normalize_cost_delta(act_reduction)
                        if pred_n is None or act_n is None:
                            model_cost_conf.observe_unusable_prediction()
                        else:
                            model_cost_conf.observe(pred_n, act_n)
            min_pos_err = min(min_pos_err, pos_err_now)
            min_quat_err = min(min_quat_err, quat_err_now)
            if pos_err_now < success_pos_threshold and quat_err_now < success_quat_threshold:
                consecutive_success_time += 1
            else:
                consecutive_success_time = 0

            r_now = Rotation.from_quat([curr_q[4], curr_q[5], curr_q[6], curr_q[3]]).as_matrix()
            tip_now = np.asarray(curr_q[7:10], dtype=float)
            tip_local_now = r_now.T @ (tip_now - curr_q[:3])
            post_physical = bool(np.isfinite(contact_distance) and contact_distance <= 0.003)
            (progress_idx, dwell_active, dwell_dead, occupied_for_log,
             on_exec_patch, _dist) = _rollout_dwell_assignment(
                param.lambda_optimizer, tip_local_now, tip_now, curr_q[:3], r_now,
                getattr(param.lambda_optimizer, "last_executed_idx", None),
                policy["p_arm_world"], post_physical,
                prev_dwell=getattr(param.lambda_optimizer, "_dwell_idx", None),
            )
            if dwell_active and not on_exec_patch:
                dwell_dead = True
            elif dwell_active and on_exec_patch:
                dwell_dead = False
                dwell_active = False
            if (not last_accept_p_arm) and not on_exec_patch and escape_on:
                dwell_dead = False
                dwell_active = False
            if verify_chatter:
                progress_idx = occupied_for_log or progress_idx
                dwell_active = True
                dwell_dead = True
            dest_idx = getattr(param.lambda_optimizer, "last_global_idx", None)
            dest_protected = bool(
                dest_idx is not None and progress_idx is not None
                and _same_contact_patch(param.lambda_optimizer, progress_idx, dest_idx, radius=0.03)
            )
            dwell_active, dwell_dead = _protect_destination_dwell(
                param.lambda_optimizer, progress_idx, dest_idx, dwell_active, dwell_dead
            )
            if progress_idx is None:
                progress_idx = dest_idx or getattr(param.lambda_optimizer, "last_executed_idx", None)
            param.lambda_optimizer.note_contact_progress(
                progress_idx, pos_err_now, active=bool(dwell_active),
                gamma=float(args.contact_dwell_gamma), min_dwell_steps=int(args.contact_dwell_steps),
                improve_eps=0.002, dead_increment=bool(dwell_dead), merge_radius=0.03,
                block_radius=0.03, block_cycles=80,
                time_decay=bool(last_accept_p_arm and not dest_protected),
            )
            if consecutive_success_time > 0:
                break

        choose_arr = np.asarray(choose_times, dtype=np.float64) if choose_times else np.array([0.0])
        success = int(rollout_step < max_rollout_length)
        success_rate += success
        print("trial_summary:", {
            "trial": trial_count,
            "mode": "stage1_isaac_fingertip",
            "success": success,
            "steps": rollout_step,
            "pose_applies": pose_apply_count,
            "final_pos_err": round(float(metrics.comp_pos_error(curr_q[:3], param.target_p_)), 5),
            "final_quat_err": round(float(metrics.comp_quat_error(curr_q[3:7], param.target_q_)), 5),
            "min_pos_err": None if not np.isfinite(min_pos_err) else round(float(min_pos_err), 5),
            "min_quat_err": None if not np.isfinite(min_quat_err) else round(float(min_quat_err), 5),
            "choose_dt_mean": round(float(np.mean(choose_arr)), 5),
            "choose_dt_max": round(float(np.max(choose_arr)), 5),
            "lambda_failures": int(getattr(param.lambda_optimizer, "acados_failure_count", 0)),
            "mpc_failures": int(getattr(mpc, "acados_failure_count", 0)),
        }, flush=True)
        if env.break_out_signal_:
            break

    if env is not None:
        env.close()
    print(
        f"Stage-1 success rate over {trial_num} flip trials: "
        f"{success_rate}/{trial_num} = {success_rate / trial_num:.2%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
