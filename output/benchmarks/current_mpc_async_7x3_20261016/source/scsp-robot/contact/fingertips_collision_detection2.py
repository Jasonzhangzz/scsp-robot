import mujoco
import casadi as cs
import numpy as np

np.set_printoptions(suppress=True)

from envs.fingertips_env import MjSimulator

class Contact:
    def __init__(self, param):
        self.param_ = param
        # Contacts involving the actuated fingertip and the object, retained
        # from the latest ``detect_once`` call.  The optimizer historically
        # only needed table contacts, while the rollout now also needs the
        # *physical* object-side contact point when applying an ideal pose
        # update.  Each item contains ``point_world`` (object surface point),
        # ``point_local`` (same point in the object frame), ``normal_world``
        # (object -> fingertip), ``dist`` and the geom names.
        self.last_fingertip_contacts = []
        # World-space contacts between the fingertip geoms and the object,
        # refreshed by every detect_once call.  The controller uses this
        # physical contact location when applying an ideal pose update.
        self.object_contacts_world = []

    @staticmethod
    def _contact_jacobian_body_frame(jacobian, body_mat):
        """Express an object contact Jacobian in the object body frame.

        MuJoCo free-joint translational/angular qvel components are world
        frame quantities.  The lambda optimizer uses body-frame contact
        points and wrenches, so its object columns must be right-multiplied
        by diag(R_body, R_body) before they are passed downstream.
        """
        jacobian = np.asarray(jacobian, dtype=np.float64).copy()
        if jacobian.ndim != 2 or jacobian.shape[1] < 6:
            return jacobian
        R = np.asarray(body_mat, dtype=np.float64).reshape(3, 3)
        frame = np.zeros((6, 6), dtype=np.float64)
        frame[:3, :3] = R
        frame[3:, 3:] = R
        jacobian[:, :6] = jacobian[:, :6] @ frame
        return jacobian

    def get_actual_fingertip_contact(self):
        """Return the closest object/fingertip contact from the last pass.

        The returned dictionary is one of :attr:`last_fingertip_contacts` and
        contains both world and object-local coordinates.  ``None`` means the
        latest collision pass had no fingertip/object contact.  Keeping the
        helper here avoids callers accidentally treating the legacy
        ``object_contacts_world`` midpoint as a surface point.
        """
        if not self.last_fingertip_contacts:
            return None
        return min(self.last_fingertip_contacts,
                   key=lambda item: abs(float(item.get('dist', 0.0))))

    def detect_once(self, simulator: MjSimulator):
        """
        检测当前仿真环境中的接触点，计算接触距离和接触雅可比矩阵
        :param simulator: MjSimulator 仿真环境对象
        :return:
            phi_vec: 接触距离向量
            jac_mat: 接触雅可比矩阵
            con_pos_list: 物体坐标系中的接触点位置列表 (仅包含物体与环境的接触点)
        """
        mujoco.mj_forward(simulator.model_, simulator.data_)
        mujoco.mj_collision(simulator.model_, simulator.data_)

        # extract the contacts
        n_con = simulator.data_.ncon
        contacts = simulator.data_.contact
        self.last_fingertip_contacts = []

        # solve the contact Jacobian
        con_phi_list = []
        con_frame_list = []
        con_pos_list = []  # 物体坐标系中的接触点位置
        con_jac_list = []
        con_jac_env_list = []
        con_phi_env_list = []
        if_contact = False
        self.object_contacts_world = []

        for i in range(n_con):
            contact_i = contacts[i]

            geom1_name = mujoco.mj_id2name(simulator.model_, mujoco.mjtObj.mjOBJ_GEOM, contact_i.geom1)
            body1_id = simulator.model_.geom_bodyid[contact_i.geom1]
            geom2_name = mujoco.mj_id2name(simulator.model_, mujoco.mjtObj.mjOBJ_GEOM, contact_i.geom2)
            body2_id = simulator.model_.geom_bodyid[contact_i.geom2]

            # contact between balls and object
            if (geom1_name in self.param_.object_names_):
                # 世界坐标系中的接触点位置
                con_pos_world = contact_i.pos
                # ``contact.dist`` is the signed gap between the two geom
                # surfaces.  The midpoint reconstruction below uses half of
                # it, but the constraint residual passed to lambda must keep
                # the full MuJoCo gap.
                con_dist = float(contact_i.dist)
                con_mu = self.param_.mu_object_

                # 接触帧的旋转矩阵 (3x3)
                con_frame = contact_i.frame.reshape((-1, 3)).T
                
                # 或者如果需要更精确的物体坐标系转换：
                # 获取物体的世界坐标系位姿
                body_pos = simulator.data_.body(body1_id).xpos
                body_mat = simulator.data_.body(body1_id).xmat.reshape(3, 3)
                con_pos_body = body_mat.T @ (con_pos_world - body_pos)

                # MuJoCo stores ``pos`` at the midpoint of the two nearest
                # points.  Its contact-frame first axis points from geom1 to
                # geom2, so recover the object-side surface point by moving
                # half the signed gap toward geom1.  Keep only fingertip /
                # object contacts here; table contacts remain in the legacy
                # environment-contact arrays below.
                if geom2_name.startswith('fingertip'):
                    n_world = con_frame[:, 0].copy()
                    object_surface_world = con_pos_world - 0.5 * float(contact_i.dist) * n_world
                    self.last_fingertip_contacts.append({
                        'point_world': object_surface_world.copy(),
                        'point_local': body_mat.T @ (object_surface_world - body_pos),
                        'midpoint_world': con_pos_world.copy(),
                        'normal_world': n_world,
                        'dist': float(contact_i.dist),
                        'geom_object': geom1_name,
                        'geom_fingertip': geom2_name,
                    })

                con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

                jacp1 = np.zeros((3, self.param_.n_qvel_))
                mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp1, jacr=None, point=con_pos_world, body=body1_id)
                con_jacp1 = con_frame_pmd.T @ jacp1

                if geom2_name != 'table':
                    jacp2 = np.array([[0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
                    if_contact = True
                    # Preserve the historical midpoint representation for
                    # this legacy attribute.  Callers needing the physical
                    # object surface should use ``last_fingertip_contacts``.
                    self.object_contacts_world.append(
                        np.asarray(con_pos_world, dtype=np.float64).copy())
                else:
                    jacp2 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
                con_jacp2 = con_frame_pmd.T @ jacp2

                con_jacp = -(con_jacp2 - con_jacp1)
                con_jacp_n = con_jacp[0]
                con_jacp_f = con_jacp[1:]
                con_jac = con_jacp_n + con_mu * con_jacp_f

                # 存储物体坐标系中的接触点位置
                if geom2_name == 'table':
                    con_pos_list.append(con_pos_body)  # 或者使用 con_pos_local
                    con_jac_env_list.append(
                        self._contact_jacobian_body_frame(con_jac, body_mat))
                    con_phi_env_list.append(con_dist)
                con_phi_list.append(con_dist)
                con_frame_list.append(con_frame)
                con_jac_list.append(con_jac)

            elif (geom2_name in self.param_.object_names_):
                # 世界坐标系中的接触点位置
                con_pos_world = contact_i.pos
                con_dist = float(contact_i.dist)
                con_mu = self.param_.mu_object_

                # 接触帧的旋转矩阵 (3x3)
                con_frame = contact_i.frame.reshape((-1, 3)).T
                
                # 精确的物体坐标系转换
                body_pos = simulator.data_.body(body2_id).xpos
                body_mat = simulator.data_.body(body2_id).xmat.reshape(3, 3)
                con_pos_body = body_mat.T @ (con_pos_world - body_pos)

                # Here geom2 is the object and the contact normal points from
                # the fingertip (geom1) toward the object.  Move half the
                # signed gap toward geom2 to recover the object-side point.
                if geom1_name.startswith('fingertip'):
                    n_world = con_frame[:, 0].copy()
                    object_surface_world = con_pos_world + 0.5 * float(contact_i.dist) * n_world
                    self.last_fingertip_contacts.append({
                        'point_world': object_surface_world.copy(),
                        'point_local': body_mat.T @ (object_surface_world - body_pos),
                        'midpoint_world': con_pos_world.copy(),
                        'normal_world': -n_world,
                        'dist': float(contact_i.dist),
                        'geom_object': geom2_name,
                        'geom_fingertip': geom1_name,
                    })

                con_frame_pmd = np.hstack((con_frame, -con_frame[:, -2:]))

                if geom1_name != 'table':
                    jacp1 = np.array([[0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 1.]])
                    if_contact = True
                    self.object_contacts_world.append(
                        np.asarray(con_pos_world, dtype=np.float64).copy())
                else:
                    jacp1 = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
                con_jacp1 = con_frame_pmd.T @ jacp1

                jacp2 = np.zeros((3, self.param_.n_qvel_))
                mujoco.mj_jac(simulator.model_, simulator.data_, jacp=jacp2, jacr=None, point=con_pos_world, body=body2_id)
                con_jacp2 = con_frame_pmd.T @ jacp2

                con_jacp = (con_jacp2 - con_jacp1)
                con_jacp_n = con_jacp[0]
                con_jacp_f = con_jacp[1:]
                con_jac = con_jacp_n + con_mu * con_jacp_f

                # 存储物体坐标系中的接触点位置
                if geom1_name == 'table':
                    con_pos_list.append(con_pos_body)  # 或者使用 con_pos_local
                    con_jac_env_list.append(
                        self._contact_jacobian_body_frame(con_jac, body_mat))
                    con_phi_env_list.append(con_dist)

                con_phi_list.append(con_dist)
                con_frame_list.append(con_frame)
                con_jac_list.append(con_jac)

        phi_vec, jac_mat = self.reformat(
            dict(
                con_pos_list=con_pos_list,
                con_phi_list=con_phi_list,
                con_frame_list=con_frame_list,
                con_jac_list=con_jac_list)
        )
        _, jac_mat_env = self.reformat(
            dict(
                con_phi_list=con_phi_env_list,
                con_jac_list=con_jac_env_list)
        )
        return phi_vec, jac_mat, con_pos_list, jac_mat_env, if_contact

    def reformat(self, contacts=None):
        # parse the input
        con_jac_list = contacts['con_jac_list']
        con_phi_list = contacts['con_phi_list']

        # fill the phi_vec
        phi_vec = np.ones((self.param_.max_ncon_ * 4,))
        jac_mat = np.zeros((self.param_.max_ncon_ * 4, self.param_.n_mj_v_))
        for i in range(len(con_phi_list)):
            phi_vec[4 * i: 4 * i + 4] = con_phi_list[i]
            jac_mat[4 * i: 4 * i + 4] = con_jac_list[i]

        return phi_vec, jac_mat
