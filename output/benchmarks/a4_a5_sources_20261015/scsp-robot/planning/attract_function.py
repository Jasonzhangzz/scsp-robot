import casadi as ca

# 定义符号变量
x, y, z = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('z')
mx, my, mz = ca.SX.sym('mx'), ca.SX.sym('my'), ca.SX.sym('mz')  # 磁矩分量

# 定义磁偶极子场的符号表达式（通用方向）
r = ca.sqrt(x**2 + y**2 + z**2)

phi_m = (mx*x + my*y + mz*z) / (4 * ca.pi * r**3)
phi_m_field = ca.Function('phi_m_field', [x, y, z, mx, my, mz], [phi_m])

def compute_circular_scalar_potential_and_gradient(pos, center=[0,0,0], virtual_point=[0,0,-1], radius=1,
                                                 distance=1, m_magnitude=1, sample_rate=8):
    """
    计算磁标势及其梯度（即负的磁场）
    使用两组圆形排列的磁偶极子，法向方向由center和virtual_point确定
    每个面上的偶极子按照圆形方式平行于切面排列
    
    参数:
        x_val, y_val, z_val: 计算点的坐标
        center: 两面磁偶极子的中心点
        virtual_point: 负极法向上的一点，用于确定法向方向
        distance: 两平面之间的距离
        m_magnitude: 磁矩大小
        sample_rate: 每个面上的磁偶极子数量
    """
    # 计算法向方向 (从center指向virtual_point)
    x_val, y_val, z_val = pos[0], pos[1], pos[2]
    normal_direction = ca.vertcat(virtual_point[0] - center[0],
                                 virtual_point[1] - center[1],
                                 0)
                                 
    normal_direction = normal_direction / ca.norm_2(normal_direction)
    
    # 创建切平面内的两个正交基向量
    # 选择一个任意向量来找到第一个切向方向
    temp_vec = ca.if_else(ca.fabs(normal_direction[1]) > 1e-9, 
                        ca.vertcat(1, 0, 0), 
                        ca.vertcat(0, 1, 0))
    tangent_direction1 = ca.cross(temp_vec, normal_direction)
    tangent_direction1 = tangent_direction1 / ca.norm_2(tangent_direction1)
    
    # 第二个切向方向是与法向和第一个切向都正交的向量
    tangent_direction2 = ca.cross(normal_direction, tangent_direction1)
    tangent_direction2 = tangent_direction2 / ca.norm_2(tangent_direction2)
    
    # 计算两个平面的中心
    plane1_center = center + normal_direction * (distance / 2)
    plane2_center = center - normal_direction * (distance / 2)
    
    # 初始化总磁标势
    phi_m_total = 0
    
    # 在每个平面上创建圆形排列的偶极子
    for plane_center in [plane1_center, plane2_center]:
        for i in range(sample_rate):
            # 计算当前角度
            angle = 2 * ca.pi * i / sample_rate
            
            # 计算偶极子位置 (在切平面内的圆形排列)
            dipole_pos = plane_center + \
                        radius * ca.cos(angle) * tangent_direction1 + \
                        radius * ca.sin(angle) * tangent_direction2
            
            # 偶极子磁矩方向 (沿法线方向)
            m_vec = normal_direction * m_magnitude
            
            # 计算相对于偶极子的坐标
            x_rel = x_val - dipole_pos[0]
            y_rel = y_val - dipole_pos[1]
            z_rel = z_val - dipole_pos[2]
            
            # 计算该偶极子的磁标势并累加
            phi_m_total += phi_m_field(x_rel, y_rel, z_rel, m_vec[0], m_vec[1], m_vec[2])
    
    return phi_m_total

def compute_scalar_potential_and_gradient(x_val, y_val, z_val, center=[0,0,0], quaternion=[1,0,0,0], distance=1, m_magnitude=1):
    """
    计算磁标势及其梯度（即负的磁场）
    """
    # 旋转矩阵计算（与您原有代码相同）
    rotation_matrix = quaternion_to_rotation_matrix_casadi(quaternion)
    direction = rotation_matrix[:, 0]
    
    # 计算两个磁偶极子的位置
    offset = direction * distance / 2
    dipole1_pos = center - offset
    dipole2_pos = center + offset
    
    # 计算两个磁偶极子的磁矩方向
    m_vec1 = direction * m_magnitude
    
    # 计算相对于每个偶极子的坐标
    x1, y1, z1 = x_val - dipole1_pos[0], y_val - dipole1_pos[1], z_val - dipole1_pos[2]
    x2, y2, z2 = x_val - dipole2_pos[0], y_val - dipole2_pos[1], z_val - dipole2_pos[2]
    
    # 计算每个偶极子的磁标势
    phi_m1 = phi_m_field(x1, y1, z1, m_vec1[0], m_vec1[1], m_vec1[2])
    phi_m2 = phi_m_field(x2, y2, z2, m_vec1[0], m_vec1[1], m_vec1[2])  # 第二个偶极子方向相反
    
    # 合成磁标势
    phi_m_total = phi_m1 + phi_m2
    
    return ca.tanh(phi_m_total)

def potential_field_obstacle_avoidance(position, goal, obstacle_center, 
                                     k_att=1.0,       # 吸引增益
                                     k_rep=0.5,       # 排斥增益
                                     influence_dist=0.1,  # 障碍影响范围
                                     min_dist=0.01):  # 最小有效距离
    """
    基于障碍物中心的势场避障函数
    
    参数:
        position: 当前位置 [x,y,z] (casadi.SX)
        goal: 目标点 [x,y,z]
        obstacle_center: 障碍物中心 [x,y,z]
        k_att: 吸引力系数
        k_rep: 排斥力系数
        influence_dist: 障碍物影响半径
        min_dist: 最小计算距离（防止除零）

    返回:
        total_cost: 总势场值
        forces: [吸引力, 排斥力] (用于分析)
    """
    # ========== 吸引势场 ==========
    to_goal = position - goal
    dist_to_goal = ca.sumsqr(to_goal)
    U_att = k_att * dist_to_goal**2

    # ========== 排斥势场 ==========
    to_obstacle = position - obstacle_center
    dist_to_obs = ca.sumsqr(to_obstacle)
    
    # 排斥势场（在影响范围内生效）
    U_rep = ca.if_else(
        dist_to_obs < influence_dist,
        k_rep / (dist_to_obs + 10),
        0.0
    )

    # ========== 总势场 ==========
    total_cost = U_att + U_rep
    
    return total_cost

def quaternion_to_rotation_matrix_casadi(quaternion):
    # 现在假设输入是 [w, x, y, z] 顺序
    x, y, z, w = quaternion[1], quaternion[2], quaternion[3], quaternion[0]
    
    R = ca.SX(3, 3)
    R[0, 0] = 1 - 2*(y**2 + z**2)
    R[0, 1] = 2*(x*y - w*z)
    R[0, 2] = 2*(x*z + w*y)
    
    R[1, 0] = 2*(x*y + w*z)
    R[1, 1] = 1 - 2*(x**2 + z**2)
    R[1, 2] = 2*(y*z - w*x)
    
    R[2, 0] = 2*(x*z - w*y)
    R[2, 1] = 2*(y*z + w*x)
    R[2, 2] = 1 - 2*(x**2 + y**2)
    
    return R


