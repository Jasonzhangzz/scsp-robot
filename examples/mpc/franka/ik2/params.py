import numpy as np
import casadi as cs

from utils import rotations
from planning.mlqp_point_v1_ip import LambdaContactControlOptimizer
from planning.attract_function import compute_scalar_potential_and_gradient

class ExplicitMPCParams:
    def __init__(self, args, rand_seed=1, target_type='rotation', mpc_model='explicit'):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters 
        # ---------------------------------------------------------------------------------------------
        self.contact_cost_param = args.contact_cost_param
        self.attract_coef = args.attract_coef
        self.reject_coef = args.reject_coef
        self.contact_coef = args.contact_coef
        self.reject_dis = args.reject_dis

        self.model_path_ = './envs/xmls/env_fingertips_'+args.obj+'.xml'
        self.mesh_path_ = "envs/assets/objects/"+args.obj+".stl"
        self.object_names_ = ['obj']

        self.h_ = 0.01
        self.frame_skip_ = int(10)

        # system dimensions:
        self.n_robot_qpos_ = 3
        self.n_qpos_ = 10
        self.n_qvel_ = 9
        self.n_cmd_ = 3

        # internal joint controller for each finger
        self.jc_kp_ = 200
        self.jc_damping_ = 10
        self.proximity_threshold_ = 0.1
        self.fingertip_geoms = [
            "left_finger_tip_pad_1", "left_finger_tip_pad_2", "left_finger_tip_pad_3",
            "left_finger_tip_pad_4", "left_finger_tip_pad_5",
            "right_finger_tip_pad_1", "right_finger_tip_pad_2", "right_finger_tip_pad_3",
            "right_finger_tip_pad_4", "right_finger_tip_pad_5"
        ]
        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(100 + rand_seed)

        # random initial pose for object
        self.table_height = 0.35
        init_height = 0.05 + self.table_height
        init_xy_rand = 0.1 * np.random.rand(2)
        init_xy_rand[0] += 0.3

        yaw_angle = - np.pi * np.random.rand(1) + np.pi / 2

        init_obj_quat_rand = rotations.rpy_to_quaternion(np.hstack([yaw_angle, 0, 0]))

        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))
        self.init_robot_qpos_ = np.array([0.0, -0.785, 0.0, -2.356, 0, 1.571, 0.785])

        # random target pose for object
        if target_type == 'rotation':
            target_xy_rand = 0.1 * np.random.rand(2)
            target_xy_rand[0] += 0.35

            self.target_p_ = np.hstack([target_xy_rand, init_height -0.02])
            yaw_angle = np.pi * np.random.rand(1) - np.pi / 2
            pitch = 0
            yaw = 0
            self.target_q_ = rotations.rpy_to_quaternion(np.hstack([yaw_angle, pitch, yaw]))
        else:
            raise ValueError(f'Target type {target_type} not supported')

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
        self.obj_inertia_[3:, 3:] = 0.05 * np.eye(3)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [300])

        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q

        self.obj_mass_ = 0.1
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])

        self.model_params = args.model_param

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_horizon_ = 5
        self.ipopt_max_iter_ = 100
        self.mpc_model = mpc_model

        self.mpc_u_lb_ = -0.02
        self.mpc_u_ub_ = 0.02
        # obj_pos_lb = np.array([-10, -10, 0])
        # obj_pos_ub = np.array([10, 10, 0.99])
        fts_q_lb = np.array([-3, -3, self.table_height+0.02])
        fts_q_ub = np.array([3, 3, 3])
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), fts_q_lb))
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), fts_q_ub))
        self.sol_guess_ = None
        self.comple_relax = 0.1
        self.max_env_contacts_ = 4
        self.lambda_optimizer = LambdaContactControlOptimizer(
                                                mesh_path=self.mesh_path_,
                                                obj_mass=self.obj_mass_,
                                                arm_friction=self.mu_object_,
                                                contact_stiffness=self.model_params,
                                                time_step=self.h_*10,
                                                sample_num=args.sample_num,
                                                pos_coef=args.pos_coef,
                                                ori_coef=args.ori_coef,
                                                # scale_factors=[0.01]*3
                                            )
    @staticmethod
    def calculate_rotation_quaternion(x, target_position):
        # 计算方向向量
        direction = x[:3] - target_position
        direction = direction / cs.norm_2(direction)
        
        # 计算绕Z轴的旋转角度（使用atan2计算y和x分量的角度）
        angle = cs.arctan2(direction[1], direction[0])  # 加上90度，使得x轴指向目标
        
        # 计算四元数（绕Z轴旋转）
        # 四元数公式: q = [cos(θ/2), 0, 0, sin(θ/2)]
        half_angle = angle / 2.0
        w = cs.cos(half_angle)
        z = cs.sin(half_angle)
        return [w, 0, 0, z]

    def init_cost_fns(self):
        x = cs.SX.sym('x', self.n_qpos_)
        u = cs.SX.sym('u', self.n_cmd_)
        
        # target cost
        target_position = cs.SX.sym('target_position', 3)
        target_quaternion = cs.SX.sym('target_quaternion', 4)
        position_cost = cs.sumsqr(x[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(x[3:7], target_quaternion) ** 2
        contact_cost = cs.sumsqr(x[0:3] - x[7:10])
        control_cost = cs.sumsqr(u)
        virtual_point = cs.SX.sym('virtual_point', 3)
        contact_point = cs.SX.sym('contact point', 3)
        curr_ori_coef = cs.SX.sym('curr_ori_coef', 1)

        # cost params
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        verify_cost_param = cs.SX.sym('verify_cost_param', 1)
        virtual_point_cost = self.log_barrier_function(x, virtual_point)
        contact_point_cost = self.log_barrier_function(x, contact_point)

        reject_cost = cs.if_else(cs.sumsqr(x[7:10]-contact_point)<self.reject_dis, -self.log_barrier_function(x, contact_point), 0.0)
        attract_cost = self.attract_coef * virtual_point_cost + self.reject_coef * reject_cost  # 世界坐标系

        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat, verify_cost_param, virtual_point, contact_point, curr_ori_coef])

        # base cost
        base_cost = (1 - verify_cost_param) * attract_cost + self.contact_coef * verify_cost_param * (self.contact_cost_param * contact_cost + (1-self.contact_cost_param) * contact_point_cost)
        final_cost = 500 * position_cost * curr_ori_coef + 5.0 * quaternion_cost * 1

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + 50 * control_cost * verify_cost_param])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost * verify_cost_param])

        return path_cost_fn, final_cost_fn
    
    @staticmethod
    def log_barrier_function(x, virtual_point, epsilon=1e-3):
        """
        对数障碍函数：当 x 接近 virtual_point 时，成本急剧增加。
        Args:
            x: 状态变量（部分或全部）
            virtual_point: 虚拟点坐标（3维）
            epsilon: 避免除零的小正数
        """
        diff = x[7:10] - virtual_point
        squared_norm = cs.dot(diff, diff) + epsilon
        barrier_cost = cs.log(squared_norm)
        return barrier_cost
    