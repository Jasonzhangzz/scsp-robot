import numpy as np
import casadi as cs

FINGERTIP_GEOM_OFFSET_Z = 0.06
FINGERTIP_SITE_OFFSET_Z = 0.10


def franka_fk_modified_dh(q):
    # 修正后的Modified DH参数（与官方URDF对齐）
    DH_params = [
        (0.0,      0.0,       0.333,  q[0]),  # Joint 1
        (0.0,      -np.pi/2,   0.0,    q[1]),  # Joint 2
        (0.0,      np.pi/2,    0.316,  q[2]),  # Joint 3
        (0.0825,   np.pi/2,    0.0,    q[3]),  # Joint 4
        (-0.0825,  -np.pi/2,   0.384,  q[4]),  # Joint 5
        (0.0,      np.pi/2,    0.0,    q[5]),  # Joint 6
        (0.088,    np.pi/2,    0.107,  q[6])   # Joint 7
    ]
    
    T = cs.SX.eye(4)
    for a, alpha, d, theta in DH_params:
        # 正确的Modified DH变换矩阵
        Ti = cs.vertcat(
            cs.horzcat(cs.cos(theta),      -cs.sin(theta),       0,       a),
            cs.horzcat(cs.sin(theta)*cs.cos(alpha), cs.cos(theta)*cs.cos(alpha), -cs.sin(alpha), -d*cs.sin(alpha)),
            cs.horzcat(cs.sin(theta)*cs.sin(alpha), cs.cos(theta)*cs.sin(alpha),  cs.cos(alpha),  d*cs.cos(alpha)),
            cs.horzcat(0,                          0,                          0,       1)
        )
        T = cs.mtimes(T, Ti)
    # This chain already lands at the "attachment" body origin in panda_nohand.xml.
    return T


def _offset_point_from_attachment(offset_z):
    q = cs.SX.sym("q", 7)
    T_attachment = franka_fk_modified_dh(q)
    T_tool = cs.SX.eye(4)
    T_tool[2, 3] = offset_z
    T_point = cs.mtimes(T_attachment, T_tool)
    return cs.Function(f"franka_point_fk_{str(offset_z).replace('.', '_')}", [q], [T_point[:3, 3]])


franka_attachment_fk = _offset_point_from_attachment(0.0)
franka_fingertip_fk = _offset_point_from_attachment(FINGERTIP_GEOM_OFFSET_Z)
franka_fingertip_site_fk = _offset_point_from_attachment(FINGERTIP_SITE_OFFSET_Z)

# Default FK matches the "fingertip" geom center in envs/xmls/panda_nohand.xml.
franka_fk = franka_fingertip_fk

if __name__ == "__main__":
    import roboticstoolbox as rtb

    # 测试函数
    q_test = np.array([0.0, -0.785, 1.0, -2.356, 0, 1.571, 0.785])  # 示例关节角度
    T_result = franka_fk(q_test)

    panda = rtb.models.Panda()
    fk = panda.fkine(q_test)

    print("末端执行器的变换矩阵：")
    print(T_result.full())

    print("rtb的fk",fk)
    print(panda.links)  # 查看 Robotics Toolbox 使用的 DH 参数
