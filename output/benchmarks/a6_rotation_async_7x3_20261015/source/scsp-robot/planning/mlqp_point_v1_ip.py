import time
import numpy as np
import casadi as cs
try:
    from project_point import ProjectionPoint
except:
    from planning.project_point import ProjectionPoint

class LambdaContactControlOptimizer:
    def __init__(self, mesh_path, obj_mass=0.01, arm_friction=0.9, 
                 contact_stiffness=12.5, time_step=0.01, max_contacts=10, sample_num=70,
                 pos_coef=1, ori_coef=0.0005, scale_factors=[1.0, 1.0, 1.0]):
        # 系统参数
        self.m = obj_mass
        self.mu_arm_obj = arm_friction
        self.K_contact = contact_stiffness
        self.h = time_step
        self.max_contacts = max_contacts
        self.pp = ProjectionPoint(mesh_path, scale_factors)

        self.sample_num = sample_num
        # self.sampling_frame = self.pp.sample_contacts_uniform_patches(5e-05, 1e-05)
        self.sampling_frame = self.pp.sample_vertices_with_normals(num_samples=self.sample_num)

        # self.pp.visualize_with_normals(sampled_frames=None,
        #                         normal_scale=None,
        #                         show_face_normals=False)
        self.sample_point = self.sampling_frame['points']
        self.normal = self.sampling_frame['normals']
        self.t1 = self.sampling_frame['tangent1']
        self.t2 = self.sampling_frame['tangent2']

        self.J_tilde = np.zeros([4 * self.max_contacts, 6])

        # 构建系统刚度矩阵Q
        self.obj_inertia = np.eye(6)
        self.obj_inertia[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia[3:, 3:] = 0.05 * np.eye(3)
        Q = np.zeros((6,6))
        Q[:6, :6] = self.obj_inertia
        self.Q_inv = np.linalg.inv(Q + 1e-8 * np.eye(Q.shape[0]))

        self.pos_coef = pos_coef
        self.ori_coef = ori_coef
        self.point_idx = np.arange(self.sample_num)
        self.init_utils()
        self._precompile_optimization_function()

    def update_Jacobian(self, J_tilde=None):
        required_rows = 4 * self.max_contacts
        """更新环境接触雅可比矩阵"""
        if J_tilde is None:
            pass
        else:
            J_tilde = J_tilde[:, :6]
            current_rows = self.J_tilde.shape[0]
            if current_rows < required_rows:
                padding = np.zeros([required_rows - current_rows, 6])
                self.J_tilde = np.concatenate(self.J_tilde, padding)
            else:
                self.J_tilde = J_tilde[:required_rows, :6]

    def _precompile_optimization_function(self):
        """预编译优化函数 - 同时优化接触力和接触点位置"""
        opti = cs.Opti()
        
        # 定义优化变量和参数
        x_d = opti.parameter(7)
        current_x = opti.parameter(7)
        J_tilde = opti.parameter(4 * self.max_contacts, 6)
        tau_o_np = opti.parameter(6)
        p_arm = opti.parameter(3)    # 接触点位置
        n_arm = opti.parameter(3)
        t1 = opti.parameter(3)
        t2 = opti.parameter(3)
        curr_ori_coef = opti.parameter(1)
        R_contact = cs.horzcat(n_arm, t1, t2)

        regularization_weight = opti.parameter()
        opti.set_value(regularization_weight, 0.01)

        # 机械臂接触力作为优化变量
        lam_arm = opti.variable(3)  # fn, ft1, ft2
        
        # 初始猜测
        opti.set_initial(lam_arm, [0.01, 0, 0])
        
        # 计算世界坐标系下的接触雅可比
        J_arm_world = self.compute_contact_jacobian(p_arm)
        
        # 构建b向量
        b = tau_o_np + cs.transpose(J_arm_world) @ (R_contact @ lam_arm)
        
        # 构造接触刚度矩阵K
        K = (self.K_contact * self.h) * cs.MX.eye(4 * self.max_contacts)
        
        # 计算接触力
        Q_inv_b = cs.MX(self.Q_inv) @ b
        J_tilde_Q_inv_b = J_tilde @ Q_inv_b
        contact_force = -K @ J_tilde_Q_inv_b
        contact_force = cs.fmax(contact_force, 0)
       
        # 计算预测速度v+
        v_plus = Q_inv_b / self.h + cs.MX(self.Q_inv) @ J_tilde.T @ contact_force / self.h
        
        # 计算预测位姿x+
        x_plus = self.cs_qposInteg_(current_x, v_plus)
        
        # 目标函数
        position_error = x_plus[:3] - x_d[:3]
        orientation_error = 1 - cs.dot(x_plus[3:7], x_d[3:7]) ** 2
        objective = (self.pos_coef * cs.norm_2(position_error)+
                     self.ori_coef * orientation_error)
        opti.minimize(objective)
        
        # 摩擦锥约束
        mu = self.mu_arm_obj
        opti.subject_to(lam_arm[1] <= mu * lam_arm[0])
        opti.subject_to(lam_arm[1] >= -mu * lam_arm[0])
        opti.subject_to(lam_arm[2] <= mu * lam_arm[0])
        opti.subject_to(lam_arm[2] >= -mu * lam_arm[0])
        opti.subject_to(lam_arm[0] >= 0.001)
        opti.subject_to(lam_arm[0] <= 2)

        
        # ----------------- IPOPT 配置 -----------------
        p_opts = {
            "print_time": False,
            "jit": False,
        }
        s_opts = {
            "max_iter": 300,
            "tol": 1e-6,
            "acceptable_tol": 1e-5,
            "linear_solver": "mumps",
            "print_level": 0,
            "sb": "yes",
        }
        opti.solver("ipopt", p_opts, s_opts)

        # 构建优化函数
        self.optimization_fn = opti.to_function(
            'optimization_fn_joint',
            [x_d, current_x, J_tilde, tau_o_np, n_arm, t1, t2, p_arm, curr_ori_coef],
            [lam_arm, x_plus, objective],
            ['x_d', 'current_x', 'J_tilde', 'tau_o_np', 'n_arm', 't1', 't2', 'p_arm', 'curr_ori_coef'],
            ['lam_arm_opt', 'x_plus_opt', 'cost']
        )

    def init_utils(self):
        # -------------------------------
        #    quaternion integration fn
        # -------------------------------
        quat = cs.SX.sym('quat', 4)
        H_q_body = cs.vertcat(cs.horzcat(-quat[1], quat[0], quat[3], -quat[2]),
                              cs.horzcat(-quat[2], -quat[3], quat[0], quat[1]),
                              cs.horzcat(-quat[3], quat[2], -quat[1], quat[0]))
        self.cs_qmat_body_fn_ = cs.Function('cs_qmat_body_fn', [quat], [H_q_body.T])

        # -------------------------------
        #    state integration fn
        # -------------------------------
        qvel = cs.SX.sym('qvel', 6)
        qpos = cs.SX.sym('qpos', 7)
        next_obj_pos = qpos[0:3] + self.h * qvel[0:3]
        next_obj_quat = (qpos[3:7] + 0.5 * self.h * self.cs_qmat_body_fn_(qpos[3:7]) @ qvel[3:6])
        next_obj_quat = next_obj_quat / cs.norm_2(next_obj_quat)
        next_qpos = cs.vertcat(next_obj_pos, next_obj_quat)
        self.cs_qposInteg_ = cs.Function('cs_qposInte', [qpos, qvel], [next_qpos])

    @staticmethod
    def compute_contact_jacobian(p):
        """优化后的接触雅可比计算 - MX 版本"""
        J_c = cs.MX.zeros(3, 6)  # 改为 MX 类型
        J_c[:3, :3] = cs.MX.eye(3)
        # 使用 CasADi 构建斜对称矩阵
        J_c[0, 4], J_c[0, 5] = p[2], -p[1]
        J_c[1, 3], J_c[1, 5] = -p[2], p[0]
        J_c[2, 3], J_c[2, 4] = p[1], -p[0]
        return J_c
    
    def optimize_control_input(self, x_d, current_x, tau_o, p_arm=None):
        """优化控制输入 - 更新接口"""
        if p_arm is None:
            p_arm = np.array([-1, 0, 0])

        ori_align_sq = cs.dot(current_x[3:7], x_d[3:7]) ** 2
        th = 0.85
        scale = 10
        curr_ori_coef = (1.0 + cs.tanh(scale * (ori_align_sq - th)))

        closest_idx, n, t1, t2  = self.pp.project_point_to_mesh(p_arm)
        p_obj_local = self.pp.scaled_mesh.vertices[closest_idx]
        normal_obj_local = self.pp.scaled_mesh.vertex_normals[closest_idx]

        start_time = time.time()
        sol = self.optimization_fn(
            x_d=x_d, 
            current_x=current_x, 
            J_tilde=self.J_tilde, 
            tau_o_np=tau_o,
            n_arm=n,
            t1=t1,
            t2=t2,
            p_arm=p_obj_local,
            curr_ori_coef=curr_ori_coef
        )
        lam_arm = sol['lam_arm_opt']
        x_plus_opt = sol['x_plus_opt']

        info = {
            "solve_time": time.time() - start_time,
            "control_input": lam_arm,
            "resulting_pose": x_plus_opt,
        }
        
        return p_obj_local, -normal_obj_local, x_plus_opt, float(sol['cost']), info

    def choose_contact_points(self, x_d, current_x, tau_o, visible_face_idx):
        if not len(visible_face_idx):
            # 确保返回不是None
            return self.sample_point[0], self.normal[0], 1.5, 1, 0

        ori_align_sq = cs.dot(current_x[3:7], x_d[3:7]) ** 2
        th = 0.85
        scale = 10
        curr_ori_coef = (1.0 + cs.tanh(scale * (ori_align_sq - th)))
        # curr_ori_coef=1
        
        error_list = cs.MX.zeros(visible_face_idx.shape[0])
        pos_buffer = []
        force_buffer = []
        # start_time = time.time()    
        for i, idx in enumerate(visible_face_idx):
            # start_t = time.time()
            sol = self.optimization_fn(
                    x_d=x_d, 
                    current_x=current_x, 
                    J_tilde=self.J_tilde, 
                    tau_o_np=tau_o,
                    n_arm=self.normal[idx],
                    t1=self.t1[idx],
                    t2=self.t2[idx],
                    p_arm=self.sample_point[idx],
                    curr_ori_coef=curr_ori_coef
                )
            
            error_list[i] = sol['cost']
            pos_buffer.append(sol['x_plus_opt'][3:])
            force_buffer.append(sol['lam_arm_opt'])
            # print(f"Contact point {i+1}/{self.sample_num} evaluation time: {time.time() - start_t}")
        # print("Contact point selection time:", time.time() - start_time)

        min_error = cs.mmin(error_list)
        max_error = cs.mmax(error_list)
        min_idx = visible_face_idx[int(cs.evalf(cs.find(error_list == min_error)[0]))]

        min_error = float(cs.evalf(min_error))
        max_error = float(cs.evalf(max_error))
        return self.sample_point[min_idx], self.normal[min_idx], min_error, max_error, curr_ori_coef
    
    def get_availble_point_idx(self, pos, R, target_pos, threshold=0.025):
        centers_world = (R @ self.sample_point.T).T + pos
        # height_point_indices = np.where(z_coords > theshold)[0]
        # common_mask = (centers_world[:, 2] > threshold) & (centers_world[:, 2] < threshold+0.02)
        common_mask = (centers_world[:, 2] > threshold)

        direction = target_pos - pos
        dis = np.linalg.norm(direction[:2])
        
        # direction/=np.linalg.norm(direction)
        # face_normals_world = (R @ self.normal.T).T
        # face_dot_products = face_normals_world @ direction
        # common_mask = common_mask & (face_dot_products > 0.)

        # if dis > 0.05:
        #     direction/=np.linalg.norm(direction)
        #     face_normals_world = (R @ self.normal.T).T
        #     face_dot_products = face_normals_world @ direction
        #     common_mask = common_mask & (face_dot_products > 0.5)

        return np.where(common_mask)[0]
    
# 使用示例
if __name__ == "__main__":
    # 创建优化器实例 (10cm x 10cm x 10cm 的方块)
    optimizer = LambdaContactControlOptimizer(
        box_size=(0.1, 0.1, 0.1),
        obj_mass=0.01,
        arm_stiffness=200,
        ground_friction=0.9,
        arm_friction=0.9,
        contact_stiffness=10,
        time_step=0.05,
        max_contacts=5
    )
    
    # 更新接触点 (底面和机械臂接触点)
    new_contact_points = [
        {'type': 'ground', 'position': np.array([0.05, 0.05, -0.05]), 'face': 'bottom'},
        {'type': 'ground', 'position': np.array([-0.05, 0.05, -0.05]), 'face': 'bottom'},
        {'type': 'ground', 'position': np.array([0.05, -0.05, -0.05]), 'face': 'bottom'},
        {'type': 'ground', 'position': np.array([-0.05, -0.05, -0.05]), 'face': 'bottom'},
    ]
    optimizer.update_contact_points(new_contact_points)
    
    # 设置目标位姿和当前位姿
    target_pose = np.array([0.1, -0., 0.0, 0.0, 0.0, 0.5])  # [x,y,z, rx,ry,rz]
    current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 
    tau_o = np.array([0.0, 0.0, -0.01 * 9.81, 0.0, 0.0, 0.0])  # 重力
    p_arm = np.array([0.0, -0.05, -0.0])  # 左侧
    n_arm = np.array([-0, 1, 0])  # 法线方向指向右侧

    lam_arm, x_plus_opt, info = optimizer.optimize_control_input(
        target_pose, current_pose, tau_o, n_arm=n_arm, p_arm=p_arm
    )
    
    # 转换结果为NumPy数组
    lam_arm_np = np.array(cs.evalf(lam_arm)).flatten()
    x_plus_opt_np = np.array(cs.evalf(x_plus_opt)).flatten()
    
    # 打印结果
    print("\n优化结果:")
    print(f"求解时间: {info['solve_time']:.6f}s")
    # print(f"位置误差: {info['position_error']:.6f}")
    # print(f"姿态误差: {info['orientation_error']:.6f}")
    print(f"最优控制输入: {lam_arm_np}")
    print(f"预测位姿: {x_plus_opt_np}")
    print(f"目标位姿: {target_pose}")
