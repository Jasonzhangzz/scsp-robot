import importlib.util
from pathlib import Path

import casadi as cs
import numpy as np

from utils import rotations


def _load_allegro_fk():
    """Load the CasADi Allegro fingertip FK used by the original Allegro MPC examples."""
    try:
        import envs.allegro_fkin as fk_module

        return fk_module
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[4]
    fk_path = repo_root / "Complementarity-Free-Dexterous-Manipulation" / "envs" / "allegro_fkin.py"
    if not fk_path.exists():
        raise ModuleNotFoundError("No module named 'envs.allegro_fkin' and fallback allegro_fkin.py was not found.")

    spec = importlib.util.spec_from_file_location("cf_allegro_fkin", fk_path)
    fk_module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Failed to load Allegro FK module from {fk_path}")
    spec.loader.exec_module(fk_module)
    return fk_module


allegro_fk = _load_allegro_fk()


def allegro_tip_positions(base_pos, finger_q, base_rot=None):
    ff_qpos = finger_q[0:4]
    mf_qpos = finger_q[4:8]
    rf_qpos = finger_q[8:12]
    th_qpos = finger_q[12:16]

    local_tips = [
        allegro_fk.fftp_pos_fd_fn(ff_qpos),
        allegro_fk.mftp_pos_fd_fn(mf_qpos),
        allegro_fk.rftp_pos_fd_fn(rf_qpos),
        allegro_fk.thtp_pos_fd_fn(th_qpos),
    ]
    if base_rot is None:
        return [base_pos + tip for tip in local_tips]

    base_rot = cs.reshape(base_rot, 3, 3)
    return [base_pos + base_rot @ tip for tip in local_tips]


def grasp_closure_cost(object_pos, tip_positions, eps=1e-8):
    unit_sum = 0.0
    for tip_position in tip_positions:
        obj_to_tip = tip_position - object_pos
        unit_sum += obj_to_tip / cs.sqrt(cs.sumsqr(obj_to_tip) + eps)
    return cs.sumsqr(unit_sum)


class ExplicitMPCParams:
    def __init__(self, rand_seed=1, target_type='rotation'):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters 
        # ---------------------------------------------------------------------------------------------
        self.model_path_ = 'envs/xmls/env_allegro_bunny.xml'
        self.object_names_ = ['obj']

        self.h_ = 0.1
        self.frame_skip_ = int(50)

        # system dimensions:
        self.n_robot_qpos_ = 16
        self.n_qpos_ = 23
        self.n_qvel_ = 22
        self.n_cmd_ = 16

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(100 + rand_seed)

        self.init_robot_qpos_ = np.array([
            0.125, 1.13, 1.45, 1.24,
            -0.02, 0.445, 1.17, 1.5,
            -0.459, 1.54, 1.11, 1.23,
            0.638, 1.85, 1.5, 1.26
        ])

        # random init and target pose for object
        if target_type == 'rotation':
            init_obj_xy = np.array([-0.03, 0.0]) + 0.005 * np.random.randn(2)
            init_obj_pos = np.hstack([init_obj_xy, 0.04])
            init_yaw_angle = np.pi * np.random.rand(1) - np.pi / 2
            init_obj_quat_rand = rotations.rpy_to_quaternion(np.hstack([init_yaw_angle, 0, 0]))
            self.init_obj_qpos_ = np.hstack((init_obj_pos, init_obj_quat_rand))

            self.target_p_ = np.array([-0.00, -0.0, 0.04])
            yaw_angle = init_yaw_angle + np.random.choice([np.pi / 2, -np.pi / 2])
            self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))

        else:
            raise ValueError(f'Target type {target_type} not supported')

        # ---------------------------------------------------------------------------------------------
        #      contact parameters
        # ---------------------------------------------------------------------------------------------
        self.mu_object_ = 0.5
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 20

        # ---------------------------------------------------------------------------------------------
        #      models parameters
        # ---------------------------------------------------------------------------------------------
        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.1 * np.eye(3)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [1])

        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q

        self.obj_mass_ = 0.01
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])

        self.model_params = 0.5

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_horizon_ = 4
        self.ipopt_max_iter_ = 50
        self.mpc_model = 'explicit'
        self.mpc_contact_weight_ = 1.0
        self.mpc_grasp_closure_weight_ = 0.05
        self.mpc_u_weight_ = 0.1

        self.mpc_u_lb_ = -0.2
        self.mpc_u_ub_ = 0.2
        obj_pos_lb = np.array([-0.99, -0.99, 0])
        obj_pos_ub = np.array([0.99, 0.99, 0.99])
        self.mpc_q_lb_ = np.hstack((obj_pos_lb, -1e7 * np.ones(4), -1e7 * np.ones(16)))
        self.mpc_q_ub_ = np.hstack((obj_pos_ub, 1e7 * np.ones(4), 1e7 * np.ones(16)))

        self.sol_guess_ = None

    # ---------------------------------------------------------------------------------------------
    #      cost functions for MPC
    # ---------------------------------------------------------------------------------------------
    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)

        obj_pose = x[0:7]
        ff_qpos = x[7:11]
        mf_qpos = x[11:15]
        rf_qpos = x[15:19]
        tm_qpos = x[19:23]

        # forward kinematics to compute the position of fingertip
        finger_q = cs.vertcat(ff_qpos, mf_qpos, rf_qpos, tm_qpos)
        tip_positions = allegro_tip_positions(cs.DM.zeros(3), finger_q)

        # target cost
        target_position = cs.SX.sym('target_position', 3)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        position_cost = cs.sumsqr(obj_pose[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(obj_pose[3:7], target_quaternion) ** 2
        contact_cost = sum(cs.sumsqr(obj_pose[0:3] - tip_position) for tip_position in tip_positions)

        # grasp cost
        grasp_closure = grasp_closure_cost(obj_pose[0:3], tip_positions)

        # control cost
        control_cost = cs.sumsqr(u)

        # cost params
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat])

        # base cost
        base_cost = self.mpc_contact_weight_ * contact_cost + self.mpc_grasp_closure_weight_ * grasp_closure
        final_cost = 100 * position_cost + 5.0 * quaternion_cost

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + self.mpc_u_weight_ * control_cost])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost])

        return path_cost_fn, final_cost_fn
