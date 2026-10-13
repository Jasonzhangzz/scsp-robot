import casadi as cs
import numpy as np

from utils import rotations
from planning.attract_function import compute_scalar_potential_and_gradient
from planning.mlqp_point import LambdaContactControlOptimizer

class ExplicitMPCParams:
    def __init__(self, args, rand_seed=1, target_type='ground-rotation', model='explicit'):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters
        # ---------------------------------------------------------------------------------------------
        self.contact_cost_param = args.contact_cost_param
        self.attract_coef = args.attract_coef
        self.reject_coef = args.reject_coef
        self.contact_coef = args.contact_coef
        self.reject_dis = args.reject_dis

        self.model_path_ = 'envs/xmls/env_football.xml'
        self.mesh_path_ = "envs/assets/objects/"+args.obj+".stl"
        self.object_names_ = ['obj']

        self.h_ = 0.1
        self.frame_skip_ = int(10)

        # system dimensions:
        self.n_robot_qpos_ = 6
        self.n_qpos_ = 13
        self.n_qvel_ = 12
        self.n_cmd_ = 6

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(100 + rand_seed)

        # random initial pose for object
        init_height = 0.021
        init_xy_rand = 0.05 * np.random.rand(2) - 0.025

        yaw_angle = 2 * np.pi * np.random.rand(1) - np.pi
        # yaw_angle = 0
        pitch_angle = 0
        roll_angle = 0
        init_obj_quat_rand = rotations.rpy_to_quaternion(np.hstack([yaw_angle, pitch_angle, roll_angle]))

        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))
        self.init_robot_qpos_ = np.array([0.2, 0.0, 0.0, -0.2, 0.0, 0.0])

        if target_type == 'ground-rotation':
            target_xy_rand = 0.2 * np.random.rand(2) - 0.1
            self.target_p_ = np.hstack([target_xy_rand, init_height])
            yaw_angle = 2 * np.pi * np.random.rand(1) - np.pi
            self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))
        elif target_type == 'in-air':
            target_height = 0.03 + 0.05 * np.random.rand(1)
            target_xy_rand = 0.2 * np.random.rand(2) - 0.1
            self.target_p_ = np.hstack([target_xy_rand, target_height])
            angle = 2 * np.pi * np.random.rand(1) - np.pi
            axis = np.array([0, 1, 1]) + np.random.randn(3) * 0.1
            self.target_q_ = rotations.axisangle2quat(np.hstack((axis, angle)))
        else:
            raise ValueError('Invalid target type')

        # ---------------------------------------------------------------------------------------------
        #      contact parameters
        # ---------------------------------------------------------------------------------------------
        self.mu_object_ = 0.5
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 10

        # ---------------------------------------------------------------------------------------------
        #      models parameters
        # ---------------------------------------------------------------------------------------------
        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.06 * np.eye(3)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [100])

        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])

        self.obj_mass_ = 0.01
        self.model_params = 1

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------f
        self.mpc_model = model
        self.mpc_horizon_ = 20
        self.ipopt_max_iter_ = 500
        self.comple_relax = 0.01

        self.mpc_u_lb_ = -0.005
        self.mpc_u_ub_ = 0.005
        fts_q_lb = np.array([-100, -100, 0.0, -100, -100, 0.0])
        fts_q_ub = np.array([100, 100, 100, 100, 100, 100])
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), fts_q_lb))
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), fts_q_ub))

        self.sol_guess_ = None

        self.lambda_optimizer = LambdaContactControlOptimizer(
                                                mesh_path=self.mesh_path_,
                                                obj_mass=self.obj_mass_,
                                                arm_friction=1.0,
                                                contact_stiffness=self.model_params,
                                                time_step=self.h_ * 10,
                                                sample_num=args.sample_num,
                                                pos_coef=args.pos_coef,
                                                ori_coef=args.ori_coef,
                                                # scale_factors=[0.0008]*3
                                            )

    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)

        # target cost
        target_position = cs.SX.sym('target_position', 3)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        position_cost = cs.sumsqr(x[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(x[3:7], target_quaternion) ** 2
        contact_cost_1 = cs.sumsqr(x[0:3] - x[7:10])
        contact_cost_2 = cs.sumsqr(x[0:3] - x[10:13])
        control_cost = cs.sumsqr(u)
        virtual_point_1 = cs.SX.sym('virtual_point_1', 3)
        virtual_point_2 = cs.SX.sym('virtual_point_2', 3)
        contact_point_1 = cs.SX.sym('contact_point_1', 3)
        contact_point_2 = cs.SX.sym('contact_point_2', 3)
        curr_ori_coef_1 = cs.SX.sym('curr_ori_coef_1', 1)
        curr_ori_coef_2 = cs.SX.sym('curr_ori_coef_2', 1)

        # cost params
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        verify_cost_param_1 = cs.SX.sym('verify_cost_1', 1)
        verify_cost_param_2 = cs.SX.sym('verify_cost_2', 1)
        virtual_point_cost_1 = self.log_barrier_function(x, virtual_point_1, slice(7, 10))
        virtual_point_cost_2 = self.log_barrier_function(x, virtual_point_2, slice(10, 13))

        reject_cost_1 = cs.if_else(
            cs.sumsqr(x[7:10] - contact_point_1) < self.reject_dis,
            -self.log_barrier_function(x, contact_point_1, slice(7, 10)),
            0.0,
        )
        reject_cost_2 = cs.if_else(
            cs.sumsqr(x[10:13] - contact_point_2) < self.reject_dis,
            -self.log_barrier_function(x, contact_point_2, slice(10, 13)),
            0.0,
        )
        attract_cost_1 = self.attract_coef * virtual_point_cost_1 + self.reject_coef * reject_cost_1
        attract_cost_2 = self.attract_coef * virtual_point_cost_2 + self.reject_coef * reject_cost_2

        cost_param = cs.vvcat([
            target_position,
            target_quaternion,
            phi_vec,
            jac_mat,
            verify_cost_param_1,
            verify_cost_param_2,
            virtual_point_1,
            virtual_point_2,
            contact_point_1,
            contact_point_2,
            curr_ori_coef_1,
            curr_ori_coef_2,
        ])

        # base cost
        contact_point_cost_1 = self.log_barrier_function(x, contact_point_1, slice(7, 10))
        contact_point_cost_2 = self.log_barrier_function(x, contact_point_2, slice(10, 13))
        contact_cost_param = self.contact_cost_param
        base_cost_1 = (1 - verify_cost_param_1) * attract_cost_1 + self.contact_coef * verify_cost_param_1 * (
            contact_cost_param * contact_cost_1 + (1 - contact_cost_param) * contact_point_cost_1
        )
        base_cost_2 = (1 - verify_cost_param_2) * attract_cost_2 + self.contact_coef * verify_cost_param_2 * (
            contact_cost_param * contact_cost_2 + (1 - contact_cost_param) * contact_point_cost_2
        )
 
        final_cost = 500 * position_cost + 5.0 * quaternion_cost * 1

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost_1 + base_cost_2 + 50 * control_cost])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost])

        return path_cost_fn, final_cost_fn

    @staticmethod
    def log_barrier_function(x, virtual_point, contact_slice, epsilon=1e-3):
        """
        对数障碍函数：当 x 接近 virtual_point 时，成本急剧增加。
        Args:
            x: 状态变量（部分或全部）
            virtual_point: 虚拟点坐标（3维）
            epsilon: 避免除零的小正数
        """
        diff = x[contact_slice] - virtual_point
        squared_norm = cs.dot(diff, diff) + epsilon
        barrier_cost = cs.log(squared_norm)
        return barrier_cost
