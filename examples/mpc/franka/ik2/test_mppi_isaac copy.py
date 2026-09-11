import argparse
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from isaacgym import gymapi


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


class IsaacFrankaSimulator:
    """Isaac Gym version of the MuJoCo MjSimulator interface used in test_mppi.py."""

    def __init__(self, param, headless=False, sim_device="cuda:0", graphics_device_id=0):
        self.param_ = param
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
        self.reset_mj_env()

        if not headless:
            self.viewer_ = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            if self.viewer_ is not None:
                cam_pos = gymapi.Vec3(1.4, 0.8, 1.0)
                cam_target = gymapi.Vec3(0.5, 0.0, 0.35)
                self.gym.viewer_camera_look_at(self.viewer_, self.env, cam_pos, cam_target)

        self.R_d = np.eye(3)

    def _create_scene_actors(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
        franka_asset_root = os.path.join(repo_root, "envs/robots/assets/urdf")
        franka_asset_file = os.path.join("franka_description", "robots", "franka_panda_gripper.urdf")

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

        obj_opts = gymapi.AssetOptions()
        obj_opts.density = 200.0
        # 近似 scene.xml 里的 mesh 体积，作为动态方块
        self.obj_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, obj_opts)

        marker_opts = gymapi.AssetOptions()
        marker_opts.fix_base_link = True
        self.marker_asset = self.gym.create_sphere(self.sim, 0.012, marker_opts)

        franka_pose = gymapi.Transform()
        franka_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        franka_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(1.2, 0.0, 0.175)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        obj_pose = gymapi.Transform()
        obj_pose.p = gymapi.Vec3(0.45, 0.0, 0.375)
        obj_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        marker_pose = gymapi.Transform()
        marker_pose.p = gymapi.Vec3(0.3, 0.0, 0.4)
        marker_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.franka_actor = self.gym.create_actor(self.env, self.franka_asset, franka_pose, "franka", 0, 0)
        self.table_actor = self.gym.create_actor(self.env, self.table_asset, table_pose, "table", 0, 0)
        self.obj_actor = self.gym.create_actor(self.env, self.obj_asset, obj_pose, "obj", 0, 0)
        self.marker_actor = self.gym.create_actor(self.env, self.marker_asset, marker_pose, "marker", 0, 0)

        self.gym.set_rigid_body_color(self.env, self.obj_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.6, 1.0))
        self.gym.set_rigid_body_color(self.env, self.marker_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1.0, 0.1, 0.1))

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

    def _simulate_once(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
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

        for _ in range(8):
            self._simulate_once()

    def set_goal(self, goal_pos=None, goal_quat=None):
        self.show_target(goal_pos=goal_pos, goal_quat=goal_quat)

    def show_target(self, goal_pos=None, goal_quat=None):
        marker_state = self.gym.get_actor_rigid_body_states(self.env, self.marker_actor, gymapi.STATE_ALL)
        if goal_pos is not None:
            marker_state["pose"]["p"][0] = (float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]))
        if goal_quat is not None:
            # input quat from scipy Rotation.as_quat() is xyzw
            marker_state["pose"]["r"][0] = (
                float(goal_quat[0]),
                float(goal_quat[1]),
                float(goal_quat[2]),
                float(goal_quat[3]),
            )
        self.gym.set_actor_rigid_body_states(self.env, self.marker_actor, marker_state, gymapi.STATE_POS)

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
        phi_vec, jac_mat = self._compute_contacts_from_state(q)

        jac_mat_env = np.zeros_like(jac_mat)
        con_pos_list = []

        # Table-contact block in compute_contact_jax is rows [4:8].
        # Keep this consistent with lambda_optimizer.update_Jacobian usage.
        obj_pos = q[0:3]
        table_dist = obj_pos[2] - float(self.param_.table_height)
        if table_dist < 0.02:
            jac_mat_env[0:4, :] = jac_mat[4:8, :]
            con_pos_list.append(np.array([0.0, 0.0, -0.025], dtype=np.float32))

        return phi_vec, jac_mat, con_pos_list, jac_mat_env

    def _compute_contacts_from_state(self, q):
        q = np.asarray(q, dtype=np.float32)
        nv = self.param_.n_qvel_
        max_ncon = self.param_.max_ncon_

        phi_vec = np.ones((max_ncon * 4,), dtype=np.float32)
        jac_mat = np.zeros((max_ncon * 4, nv), dtype=np.float32)

        obj_pos = q[0:3]
        q_robot = q[-self.param_.n_robot_qpos_:]
        ee_pos = np.array(_franka_fk_T_jax(q_robot), dtype=np.float32)[:3, 3]
        J_pos = np.array(_franka_jacobian_pos_jax(q_robot), dtype=np.float32)  # (3,7)
        J_rel = np.concatenate([np.eye(3), np.zeros((3, 3)), -J_pos], axis=1)  # (3,13)

        # object-EE contact
        d = ee_pos - obj_pos
        dist = float(np.linalg.norm(d) - float(getattr(self.param_, "contact_radius_", 0.02)))
        n, t1, t2 = _tangent_basis_from_normal(d)
        con_jac = np.array(_contact_jacobian(n, t1, t2, J_rel, float(self.param_.mu_object_)), dtype=np.float32)
        if dist < 0.0:
            phi_vec[0:4] = dist
            jac_mat[0:4, :] = con_jac

        # object-table contact
        dist_table = float(obj_pos[2] - float(self.param_.table_height))
        n_t = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        t1_t = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        t2_t = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        J_rel_table = np.concatenate(
            [np.eye(3), np.zeros((3, 3)), np.zeros((3, self.param_.n_robot_qpos_))], axis=1
        )  # (3,13)
        con_jac_table = np.array(
            _contact_jacobian(n_t, t1_t, t2_t, J_rel_table, float(self.param_.mu_object_)),
            dtype=np.float32,
        )
        if dist_table < 0.0:
            phi_vec[4:8] = dist_table
            jac_mat[4:8, :] = con_jac_table

        return phi_vec, jac_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj', type=str, default='elephant', help='name of obj')
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

    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--sim-device', type=str, default='cuda:0')
    parser.add_argument('--graphics-device-id', type=int, default=0)

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

        contact = ContactIsaac(param)
        env = IsaacFrankaSimulator(
            param,
            headless=args.headless,
            sim_device=args.sim_device,
            graphics_device_id=args.graphics_device_id,
        )

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
