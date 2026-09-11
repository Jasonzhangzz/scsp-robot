import casadi as cs
import numpy as np
import os
import tempfile
import hashlib
import trimesh


def _mujoco_collision_mesh(mesh_path, model_path):
    """Extract MuJoCo's compiled convex mesh into a cached STL.

    MuJoCo retains the full STL for rendering but compiles a convex polygon
    set for mesh collisions.  Sampling the source STL can therefore select
    concave points (for example the inside of the trunk) that the simulator
    will never report as contact.  The ``mesh_poly*`` arrays are the exact
    compiled collision hull, so exporting them keeps lambda and MuJoCo on the
    same geometry.  Return the original path when extraction is unavailable.
    """
    try:
        import mujoco
        model_path = os.path.abspath(model_path)
        mesh_path = os.path.abspath(mesh_path)
        model = mujoco.MjModel.from_xml_path(model_path)
        # MuJoCo does not expose the original asset file name through
        # MjModel.  The fingertip XML contains one object mesh (index zero);
        # selecting that compiled mesh also avoids accidentally using the
        # visual duplicate geoms.
        mesh_idx = 0 if model.nmesh == 1 else None
        if mesh_idx is None or int(model.mesh_polynum[mesh_idx]) <= 0:
            return mesh_path
        v0 = int(model.mesh_vertadr[mesh_idx]); nv = int(model.mesh_vertnum[mesh_idx])
        vertices = np.asarray(model.mesh_vert[v0:v0 + nv], dtype=np.float64)
        # ``mesh_vert`` is in MuJoCo's compiled mesh frame.  Convert it to the
        # object geom/body frame, which is the frame used by the optimizer and
        # by the STL asset.  The mesh and geom transforms are equivalent for
        # this XML; use the geom transform so this remains correct if an asset
        # transform is moved from ``<mesh>`` to ``<geom>`` later.
        geom_id = model.geom('obj').id
        rot_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_flat, np.asarray(model.geom_quat[geom_id], dtype=np.float64))
        geom_rot = rot_flat.reshape(3, 3)
        vertices = np.asarray(model.geom_pos[geom_id], dtype=np.float64) + vertices @ geom_rot.T
        p0 = int(model.mesh_polyadr[mesh_idx]); npoly = int(model.mesh_polynum[mesh_idx])
        polys = []
        for i in range(p0, p0 + npoly):
            a = int(model.mesh_polyvertadr[i]); n = int(model.mesh_polyvertnum[i])
            ids = np.asarray(model.mesh_polyvert[a:a + n], dtype=np.int64)
            # mesh_polyvert stores indices in the mesh-local vertex array.
            ids = ids - v0 if ids.size and ids.max() >= nv else ids
            if ids.size >= 3:
                polys.extend([ids[[0, j, j + 1]] for j in range(1, len(ids) - 1)])
        if not polys:
            return mesh_path
        hull = trimesh.Trimesh(vertices=vertices, faces=np.asarray(polys), process=False)
        hull.remove_unreferenced_vertices()
        # mesh_poly facets can be emitted with mixed winding after fan
        # triangulation.  Reorient the closed hull before exporting so vertex
        # normals used by projection are nonzero and consistently outward.
        hull.fix_normals()
        stat = os.stat(mesh_path)
        model_stat = os.stat(model_path)
        key = hashlib.sha1(f'bodyframe-v6:{os.path.abspath(mesh_path)}:{stat.st_mtime_ns}:{os.path.abspath(model_path)}:{model_stat.st_mtime_ns}:{len(polys)}'.encode()).hexdigest()[:16]
        cached = os.path.join(tempfile.gettempdir(), f'mujoco_collision_{key}.stl')
        if not os.path.exists(cached):
            hull.export(cached)
        return cached
    except Exception:
        # Keep the experiment usable in environments without mujoco's Python
        # bindings; ProjectionPoint will then use its normal source mesh.
        return mesh_path

from utils import rotations
from planning.attract_function import compute_scalar_potential_and_gradient
from planning.mlqp_point import LambdaContactControlOptimizer

class ExplicitMPCParams:
    def __init__(self, args, rand_seed=1, target_type='ground-rotation', model='explicit'):
        # ---------------------------------------------------------------------------------------------
        #      simulation parameters
        # ---------------------------------------------------------------------------------------------
        # The best-contact diagnostic is intended to test whether the
        # selected surface point itself can drive the object.  With the
        # historical default of 1.0 the verify objective uses only the
        # object-to-fingertip center term and the supplied contact point has
        # zero weight, which would make that diagnostic inconclusive.  Keep
        # explicit non-default values available for controlled sweeps.
        requested_contact_cost = float(args.contact_cost_param)
        if (getattr(args, 'ideal_contact_best_contact', False)
                and np.isclose(requested_contact_cost, 1.0)):
            requested_contact_cost = 0.0
        self.contact_cost_param = requested_contact_cost
        self.attract_coef = float(args.attract_coef)
        # The diagnostic tracks a surface point.  The default field / center-
        # reject terms pull the fingertip toward or away from the object
        # center and make that experiment hit the wrong face.
        self.field_cost_weight = 0.05
        self.quadratic_contact_track = False
        if getattr(args, 'ideal_contact_best_contact', False):
            self.attract_coef *= max(
                1.0, float(getattr(args, 'ideal_contact_tracking_coef', 10.0)))
            self.field_cost_weight = 0.0
            self.quadratic_contact_track = True
        self.reject_coef = args.reject_coef
        # Escape experiments replace the conflicting object-center rejection
        # term with the moving cubic-spline virtual-point cost.
        self.spline_escape_cost = bool(getattr(args, 'spline_escape_cost', 0))
        if getattr(args, 'ideal_contact_best_contact', False):
            self.spline_escape_cost = True
        self.contact_coef = args.contact_coef
        self.reject_dis = args.reject_dis

        self.model_path_ = './envs/xmls/env_fingertips_'+args.obj+'.xml'
        self.mesh_path_ = "envs/assets/objects/"+args.obj+".stl"
        self.object_names_ = ['obj']
        # The MuJoCo mesh geom is rendered from the full STL but collisions
        # are evaluated on its convex representation.  Use that same hull for
        # elephant contact candidates; this removes unreachable concave
        # samples (inside the trunk/feet) from lambda optimisation.
        requested_hull = getattr(args, 'collision_hull', None)
        self.collision_hull = (args.obj == 'elephant' if requested_hull is None
                               else bool(requested_hull))
        self._collision_mesh_extracted = False
        if self.collision_hull:
            source_mesh = self.mesh_path_
            self.mesh_path_ = _mujoco_collision_mesh(source_mesh, self.model_path_)
            self._collision_mesh_extracted = (self.mesh_path_ != source_mesh)

        # Keep the calibrated outer MPC discretization.  Lambda uses this
        # same step below; the previous h*10 setting produced 0.5 s pose
        # increments that were written every 0.02 s simulation cycle.
        self.h_ = 0.05
        self.frame_skip_ = int(10)

        # system dimensions:
        self.n_robot_qpos_ = 9 - 6
        self.n_qpos_ = 16 - 6
        self.n_qvel_ = 15 - 6
        self.n_cmd_ = 9 - 6

        # ---------------------------------------------------------------------------------------------
        #      initial state and target state
        # ---------------------------------------------------------------------------------------------
        np.random.seed(100 + rand_seed)

        # random initial pose for object
        # Keep this as the target height even when a tilted start needs to be
        # lifted clear of the MuJoCo ground plane.  In particular, changing
        # the initial orientation must not silently change target_p_.
        target_height = 0.03
        init_height = target_height
        init_xy_rand = -0.2 * np.random.rand(2) + 0.1
        # init_xy_rand = np.zeros(2)
        # init_xy_rand[1] = -0.1

        yaw_angle = float(2 * np.pi * np.random.rand() - np.pi)
        # Draw the target roll before any optional initial-tilt draws.  This
        # preserves the target quaternion sequence for a given rand_seed, so
        # enabling the experiment changes only the initial state.
        target_roll_sample = float(np.random.rand())

        # Keep the original yaw randomization, and optionally add a bounded
        # tilt around a random horizontal axis.  Sampling the tilt as an
        # angle/axis pair avoids the corner bias introduced by independently
        # drawing pitch and roll, while still producing genuinely toppled
        # starts when ``--init_tilt_deg`` is large (for example 60--75 deg).
        # The target pose below is intentionally computed independently and is
        # therefore unchanged by this experiment.
        pitch_angle = 0.0
        roll_angle = 0.0
        if getattr(args, 'random_init_tilt', False):
            max_tilt = np.deg2rad(max(0.0, float(getattr(args, 'init_tilt_deg', 65.0))))
            min_tilt = np.deg2rad(max(0.0, float(getattr(args, 'init_tilt_min_deg', 0.0))))
            min_tilt = min(min_tilt, max_tilt)
            # Uniformly sample tilt magnitude in area (rather than clustering
            # all starts at the upright orientation), and choose its horizontal
            # axis uniformly.
            tilt = np.sqrt(min_tilt ** 2 + np.random.rand() *
                           (max_tilt ** 2 - min_tilt ** 2))
            axis_angle = 2.0 * np.pi * np.random.rand()
            pitch_angle = float(tilt * np.cos(axis_angle))
            roll_angle = float(tilt * np.sin(axis_angle))
        init_obj_quat_rand = rotations.rpy_to_quaternion(np.hstack([yaw_angle, pitch_angle, roll_angle]))

        if getattr(args, 'random_init_tilt', False):
            # A rotated mesh whose centre stays at z=.03 can intersect the
            # table by several centimetres.  MuJoCo then resolves the deep
            # penetration with a large impulse, often undoing the intended
            # toppled start before the first MPC step.  Lift only the initial
            # body until its conservative local AABB is just above z=0; the
            # target remains at ``target_height`` below.
            qw, qx, qy, qz = np.asarray(init_obj_quat_rand, dtype=float)
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),
                 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),
                 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw),
                 1 - 2 * (qx * qx + qy * qy)],
            ])
            try:
                import trimesh
                local_bounds = np.asarray(trimesh.load_mesh(
                    self.mesh_path_, process=False).bounds, dtype=float)
            except Exception:
                # The six bounds are enough for this safety lift and avoid
                # making initialization depend on trimesh in unit tests.
                local_bounds = np.array([[-0.06, -0.04, -0.04],
                                         [0.06, 0.04, 0.06]], dtype=float)
            corners = np.array(np.meshgrid(*zip(local_bounds[0], local_bounds[1]))).T.reshape(-1, 3)
            min_world_z = float(np.min(corners @ R[2, :]))
            init_height = max(init_height, -min_world_z + 0.002)

        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))
        self.init_robot_qpos_ = np.array([0.2, 0.0, 0.0])

        # random target pose for object
        if target_type == 'ground-rotation':
            # target_xy_rand = 0.05 * np.random.rand(2) - 0.1
            target_xy_rand = np.zeros(2)
            target_xy_rand[1] = 0.1

            self.target_p_ = np.hstack([target_xy_rand, target_height])
            yaw_angle = 0
            pitch_angle = -np.pi/2
            roll_angle =  np.pi * target_roll_sample - np.pi / 2
            body_target_q = rotations.rpy_to_quaternion(
                np.hstack([yaw_angle, pitch_angle, roll_angle]))
            # The elephant goal mesh historically carried an extra +90deg Y
            # geom rotation while the real object geom did not.  Fold that
            # visual offset into the target body quaternion, then keep the
            # goal geom at identity so the displayed target and pose metric
            # refer to the same physical object orientation.
            goal_geom_q = np.array([np.sqrt(0.5), 0., np.sqrt(0.5), 0.])
            self.target_q_ = rotations.quaternion_multiply(body_target_q, goal_geom_q)

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
        self.obj_mass_ = 0.01
        self.obj_inertia_ = np.identity(6)
        # Keep the historical effective inertia for now.  Replacing it with
        # the raw XML mass changes the contact-force scale by orders of
        # magnitude and requires a separate force-model calibration.
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.05 * np.eye(3)
        self.robot_stiff_ = np.diag(self.n_cmd_ * [200])

        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])

        self.model_params = args.model_param

        # ---------------------------------------------------------------------------------------------
        #      planner parameters
        # ---------------------------------------------------------------------------------------------
        self.mpc_model = model
        self.torch_solver = getattr(args, 'solver', 'ipopt')
        self.mpc_horizon_ = 5
        self.ipopt_max_iter_ = 100
        self.comple_relax = 0.01

        # Expose the fingertip step limit for trajectory-tracking studies.
        # Larger limits reduce lag to the lambda-predicted contact trajectory.
        self.mpc_u_lb_ = -float(getattr(args, 'mpc_step_limit', 0.005))
        self.mpc_u_ub_ = -self.mpc_u_lb_
        fts_q_lb = np.array([-10, -10, -0.01])
        fts_q_ub = np.array([10, 10, 1])
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), fts_q_lb))
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), fts_q_ub))

        self.sol_guess_ = None

        self.lambda_optimizer = LambdaContactControlOptimizer(
                                                mesh_path=self.mesh_path_,
                                                obj_mass=self.obj_mass_,
                                                arm_friction=self.mu_object_,
                                                contact_stiffness=self.model_params,
                                                # Use the calibrated lambda
                                                # discretization.  The old
                                                # h*10 value (0.5 s) caused
                                                # severe pose teleportation.
                                                time_step=self.h_,
                                                sample_num=args.sample_num,
                                                pos_coef=args.pos_coef,
                                                ori_coef=args.ori_coef,
                                                friction_reg_coef=getattr(args, 'friction_reg_coef', 0.0),
                                                force_reg_coef=getattr(args, 'force_reg_coef', 0.01),
                                                max_contact_force=getattr(args, 'max_contact_force', 10.0),
                                                contact_switch_radius=getattr(args, 'contact_switch_radius', 0.03),
                                                contact_switch_margin_ratio=getattr(args, 'contact_switch_margin_ratio', 0.2),
                                                contact_switch_margin_abs=getattr(args, 'contact_switch_margin_abs', 1e-3),
                                                fingertip_clearance=getattr(args, 'fingertip_clearance', 0.011),
                                                normal_stability_cos=getattr(args, 'normal_stability_cos', 0.90),
                                                solver=getattr(args, 'solver', 'ipopt'),
                                                torch_max_iter=getattr(args, 'torch_max_iter', 100),
                                                # An extracted mesh already is
                                                # MuJoCo's hull; only apply the
                                                # trimesh fallback hull when
                                                # extraction was unavailable.
                                                collision_hull=(self.collision_hull and
                                                                 not self._collision_mesh_extracted)
                                            )

    @staticmethod
    def calculate_rotation_quaternion(x, target_position):
        direction = x[:2] - target_position[:2]
        direction = direction / cs.sqrt(cs.sumsqr(direction) + 1e-9)
        angle = cs.arctan2(direction[1], direction[0])
        half_angle = angle / 2.0
        return [cs.cos(half_angle), 0, 0, cs.sin(half_angle)]

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

        # cost params
        phi_vec = cs.SX.sym('phi_vec', self.max_ncon_ * 4)
        jac_mat = cs.SX.sym('jac_mat', self.max_ncon_ * 4, self.n_qvel_)
        verify_cost_param = cs.SX.sym('verify_cost', 1)
        direction_quat = self.calculate_rotation_quaternion(x, target_position)
        field_cost = compute_scalar_potential_and_gradient(
            x[7], x[8], x[9],
            center=x[:3],
            quaternion=direction_quat,
            distance=0.5,
            m_magnitude=0.5,
        )[0]
        if getattr(self, 'quadratic_contact_track', False):
            virtual_point_cost = cs.sumsqr(x[7:10] - virtual_point)
        else:
            virtual_point_cost = self.log_barrier_function(x, virtual_point)

        reject_distance = cs.sumsqr(x[0:2] - x[7:9]) + 1e-3
        obstacle_cost = (cs.DM(0) if self.spline_escape_cost else
                         cs.if_else(reject_distance < self.reject_dis, 1 / (reject_distance), 0.0))
        attract_cost = (self.attract_coef * virtual_point_cost
                        + float(getattr(self, 'field_cost_weight', 0.05)) * field_cost
                        + self.reject_coef * obstacle_cost)

        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat, verify_cost_param, virtual_point, contact_point])

        # base cost
        contact_point_cost = (cs.sumsqr(x[7:10] - contact_point)
                              if getattr(self, 'quadratic_contact_track', False)
                              else self.log_barrier_function(x, contact_point))
        base_cost = (1 - verify_cost_param) * attract_cost + self.contact_coef * verify_cost_param * (self.contact_cost_param * contact_cost + (1-self.contact_cost_param) * contact_point_cost)
 
        final_cost = 500 * position_cost + 5.0 * quaternion_cost * 4

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + 50 * control_cost])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost])

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
