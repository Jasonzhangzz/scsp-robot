import casadi as cs
import numpy as np
import os
import tempfile
import hashlib
import trimesh


def _mujoco_collision_mesh(mesh_path, model_path):
    """Extract MuJoCo's compiled convex mesh into a cached STL.

    MuJoCo retains the full STL for rendering but uses its convex hull for
    mesh collisions.  Sampling the source STL can therefore select
    concave points (for example the inside of the trunk) that the simulator
    will never report as contact.  Rebuild that hull from compiled vertices
    in the object body frame.  Return the original path when extraction is
    unavailable.
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
        if mesh_idx is None:
            return mesh_path
        v0 = int(model.mesh_vertadr[mesh_idx]); nv = int(model.mesh_vertnum[mesh_idx])
        vertices = np.asarray(model.mesh_vert[v0:v0 + nv], dtype=np.float64)
        # The compiled geom pose already incorporates mesh centering and
        # principal-axis alignment.  ``mesh_vert`` therefore needs only
        # geom_quat/geom_pos to reach the body frame.  Applying mesh_pos/quat
        # as well repeats that transform and shifts/rotates the contact hull.
        geom_id = model.geom('obj').id
        rot_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rot_flat, np.asarray(model.geom_quat[geom_id], dtype=np.float64))
        geom_rot = rot_flat.reshape(3, 3)
        vertices = np.asarray(model.geom_pos[geom_id], dtype=np.float64) + vertices @ geom_rot.T
        # Fan triangulation of mesh_poly* can create inward-facing or
        # non-supporting triangles on nearly coplanar polygon groups.
        # Rebuilding the convex hull gives closed, outward-facing facets;
        # each facet's radius-offset sphere then touches MuJoCo's geom.
        hull = trimesh.convex.convex_hull(vertices)
        stat = os.stat(mesh_path)
        model_stat = os.stat(model_path)
        key = hashlib.sha1(f'bodyframe-v8-single-transform:{os.path.abspath(mesh_path)}:{stat.st_mtime_ns}:{os.path.abspath(model_path)}:{model_stat.st_mtime_ns}:{len(hull.faces)}'.encode()).hexdigest()[:16]
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
        self.contact_cost_param = float(args.contact_cost_param)
        self.attract_coef = float(args.attract_coef)
        self.field_cost_weight = 0.05
        self.quadratic_contact_track = False
        self.reject_coef = args.reject_coef
        self.spline_escape_cost = bool(getattr(args, 'spline_escape_cost', 0))
        self.contact_coef = args.contact_coef
        self.reject_dis = args.reject_dis
        # --ideal_contact_pose replaces the verify_cost 0/1 switch with a
        # single C-inf detour: always attract to the selected patch, and
        # stay outside the object except in a cone around that patch.
        self.smooth_contact_detour = bool(getattr(args, 'ideal_contact_pose', False))
        # Quadratic tracking needs a much larger weight than the old
        # log-barrier attract_coef=0.5; otherwise 50||u||^2 freezes the ball
        # after the lift term has already raised it.
        self.detour_attract_coef = float(getattr(args, 'detour_attract_coef', 80.0))
        self.detour_repel_coef = float(getattr(args, 'detour_repel_coef', 40.0))
        self.detour_lift_coef = float(getattr(args, 'detour_lift_coef', 25.0))
        self.detour_align_thresh = float(getattr(args, 'detour_align_thresh', 0.50))
        self.detour_align_sharpness = float(getattr(args, 'detour_align_sharpness', 8.0))
        self.object_circumradius = 0.08
        self.object_aabb_lo = np.array([-0.06, -0.04, -0.04], dtype=np.float64)
        self.object_aabb_hi = np.array([0.06, 0.04, 0.06], dtype=np.float64)

        self.model_path_ = './envs/xmls/env_fingertips_'+args.obj+'.xml'
        self.mesh_path_ = "envs/assets/objects/"+args.obj+".stl"
        self.object_names_ = ['obj']
        # The MuJoCo mesh geom is rendered from the full STL but collisions
        # are evaluated on its compiled convex representation.  Lambda must
        # sample that same representation (including MuJoCo's mesh/geom frame
        # transform), otherwise it can select a point that mj_forward can
        # never contact.  Keep an explicit override for compatibility.
        requested_hull = getattr(args, 'collision_hull', None)
        self.collision_hull = (True if requested_hull is None
                               else bool(requested_hull))
        self._collision_mesh_extracted = False
        if self.collision_hull:
            source_mesh = self.mesh_path_
            self.mesh_path_ = _mujoco_collision_mesh(source_mesh, self.model_path_)
            self._collision_mesh_extracted = (self.mesh_path_ != source_mesh)
        try:
            bounds = np.asarray(trimesh.load_mesh(self.mesh_path_, process=False).bounds,
                                dtype=np.float64)
            self.object_aabb_lo = bounds[0].copy()
            self.object_aabb_hi = bounds[1].copy()
            self.object_circumradius = float(np.linalg.norm(0.5 * (bounds[1] - bounds[0])))
        except Exception:
            self.object_circumradius = 0.08

        # MPC / MuJoCo execution can stay on the 20 ms control interval in
        # rollout.  Lambda contact *ranking* must not: a 20 ms step with a
        # force-scaled wrench against the historical Q=50 inertia predicts
        # micrometre-scale x_plus for every sample, flattens the pose-cost
        # landscape, and traps the switch policy on the nearest patch.
        self.frame_skip_ = int(10)
        try:
            import mujoco
            _model_dt = float(mujoco.MjModel.from_xml_path(self.model_path_).opt.timestep)
        except Exception:
            _model_dt = 0.002
        self.h_ = (_model_dt * self.frame_skip_
                   if bool(getattr(args, 'rollout', False)) else 0.05)
        # Calibrated one-step ranking horizon, shared with
        # --ideal_contact_switch.  Do not couple this to the MuJoCo control
        # interval; object motion in rollout still comes from physics.
        self.lambda_h_ = 0.05

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
        self.init_robot_qpos_ = np.array([0.2, 0.0, 0.02])
        if getattr(args, 'random_init_tilt', False):
            self.init_robot_qpos_[:2] += 0.06 * (2.0 * np.random.rand(2) - 1.0)
            self.init_robot_qpos_[2] = 0.02 + 0.03 * float(np.random.rand())

        # Rollout uses MuJoCo's free-body mass matrix instead of the old
        # hand-tuned lambda inertia.  Keep both representations: MPC's state
        # uses world-frame free-joint qvel, while lambda contact points and
        # wrenches are expressed in the object body frame.
        lambda_obj_inertia = None
        if bool(getattr(args, 'rollout', False)):
            try:
                import mujoco
                _mj_model = mujoco.MjModel.from_xml_path(self.model_path_)
                _mj_data = mujoco.MjData(_mj_model)
                _mj_data.qpos[:7] = self.init_obj_qpos_
                mujoco.mj_forward(_mj_model, _mj_data)
                _full_mass = np.zeros((_mj_model.nv, _mj_model.nv), dtype=np.float64)
                mujoco.mj_fullM(_mj_model, _mj_data, _full_mass)
                _mass_world = np.asarray(_full_mass[:6, :6], dtype=np.float64)
                _body_rot = np.asarray(
                    _mj_data.xmat[_mj_model.body('obj').id], dtype=np.float64).reshape(3, 3)
                _frame = np.zeros((6, 6), dtype=np.float64)
                _frame[:3, :3] = _body_rot
                _frame[3:, 3:] = _body_rot
                lambda_obj_inertia = _frame.T @ _mass_world @ _frame
                self._mujoco_object_inertia_world = _mass_world.copy()
            except Exception:
                lambda_obj_inertia = None

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
        # Keep the explicit MPC in its historically conditioned numerical
        # coordinates.  The physical MuJoCo mass matrix is passed separately
        # to LambdaContactControlOptimizer below; putting values around 1e-6
        # directly into this Q makes Q^{-1} ill-conditioned and causes the
        # acados planner to fail before it can generate lateral motion.
        # MjSimulator drives the fingertip with -100*dpos - 2*dvel.  The
        # simplified MPC uses the command as a position increment, so its
        # stiffness term should match that 100 N/m rollout actuator.
        self.robot_stiff_ = np.diag(
            self.n_cmd_ * [100 if bool(getattr(args, 'rollout', False)) else 200])

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
                                                time_step=self.lambda_h_,
                                                sample_num=args.sample_num,
                                                pos_coef=args.pos_coef,
                                                ori_coef=args.ori_coef,
                                                friction_reg_coef=getattr(args, 'friction_reg_coef', 0.0),
                                                force_reg_coef=getattr(args, 'force_reg_coef', 0.01),
                                                # Keep the ranking force cap on the same impulse
                                                # scale as --ideal_contact_switch.  A 0.75 N
                                                # physical cap made every candidate look equally
                                                # powerless against Q=50, so nearest-patch
                                                # hysteresis always won.
                                                max_contact_force=float(getattr(args, 'max_contact_force', 10.0)),
                                                contact_switch_radius=getattr(args, 'contact_switch_radius', 0.03),
                                                contact_switch_margin_ratio=getattr(args, 'contact_switch_margin_ratio', 0.2),
                                                contact_switch_margin_abs=getattr(args, 'contact_switch_margin_abs', 1e-3),
                                                fingertip_clearance=getattr(args, 'fingertip_clearance', 0.011),
                                                normal_stability_cos=getattr(args, 'normal_stability_cos', 0.95),
                                                solver=getattr(args, 'solver', 'ipopt'),
                                                torch_max_iter=getattr(args, 'torch_max_iter', 100),
                                                # The raw MuJoCo rotational inertia is only a few
                                                # 1e-6 kg m^2.  The reduced one-step contact model
                                                # has no compliant contact state, so using it directly
                                                # turns the table reaction into an enormous angular
                                                # impulse.  Keep the calibrated effective inertia for
                                                # candidate ranking; MuJoCo remains the execution
                                                # model and the measured mass is retained above for
                                                # diagnostics.
                                                obj_inertia=None,
                                                # Ranking uses the historical impulse-like lambda.
                                                # Interpreting the wrench as a force (times h)
                                                # is a physical-unit conversion for diagnostics
                                                # only and must not be used to score patches:
                                                # it collapses the cost gaps that let the
                                                # switch policy reject a nearby local optimum.
                                                wrench_is_force=False,
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
        cost_param = cs.vvcat([target_position, target_quaternion, phi_vec, jac_mat, verify_cost_param, virtual_point, contact_point])
        if getattr(self, 'smooth_contact_detour', False):
            base_cost = self._smooth_contact_detour_cost(x, virtual_point, contact_point)
            final_cost = (500 * position_cost + 20.0 * quaternion_cost
                          + 20.0 * self.detour_attract_coef * cs.sumsqr(x[7:10] - virtual_point))
            control_weight = 8.0
        else:
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
            contact_point_cost = (cs.sumsqr(x[7:10] - contact_point)
                                  if getattr(self, 'quadratic_contact_track', False)
                                  else self.log_barrier_function(x, contact_point))
            base_cost = ((1 - verify_cost_param) * attract_cost
                         + self.contact_coef * verify_cost_param
                         * (self.contact_cost_param * contact_cost
                            + (1 - self.contact_cost_param) * contact_point_cost))
            final_cost = 500 * position_cost + 5.0 * quaternion_cost * 4
            control_weight = 50.0

        path_cost_fn = cs.Function('path_cost_fn', [x, u, cost_param], [base_cost + control_weight * control_cost])
        final_cost_fn = cs.Function('final_cost_fn', [x, cost_param], [10 * final_cost])

        return path_cost_fn, final_cost_fn

    @staticmethod
    def _smooth_relu(z, eps=1e-4):
        return 0.5 * (z + cs.sqrt(z * z + eps))

    @staticmethod
    def _smooth_gate(value, threshold, sharpness=8.0):
        return 0.5 * (1.0 + cs.tanh(float(sharpness) * (value - threshold)))

    @staticmethod
    def _quat_wxyz_to_rot(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        return cs.vertcat(
            cs.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            cs.horzcat(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            cs.horzcat(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )

    def _object_top_z(self, obj_pos, obj_quat):
        lo = np.asarray(self.object_aabb_lo, dtype=np.float64)
        hi = np.asarray(self.object_aabb_hi, dtype=np.float64)
        half = cs.DM(0.5 * (hi - lo))
        center_local = cs.DM(0.5 * (hi + lo))
        R = self._quat_wxyz_to_rot(obj_quat)
        center_world = obj_pos + R @ center_local
        z_extent = (cs.fabs(R[2, 0]) * half[0]
                    + cs.fabs(R[2, 1]) * half[1]
                    + cs.fabs(R[2, 2]) * half[2])
        return center_world[2] + z_extent

    def _smooth_contact_detour_cost(self, x, virtual_point, contact_point):
        """Always attract to the selected patch; lift only while blocked.

        Attracting to a sky waypoint on the object's circumsphere created a
        hover equilibrium: the elephant keep-out sphere is ~8 cm, the via
        sat on top of it, and gravity compensation held the fingertip there.
        Lift is now a one-sided floor (too-low penalty) that turns off once
        the ball clears the mesh top, so the goal term can pull it down.
        """
        tip = x[7:10]
        obj = x[0:3]
        goal = virtual_point
        surface = contact_point

        patch_n = goal - surface
        patch_n = patch_n / cs.sqrt(cs.sumsqr(patch_n) + 1e-9)
        radial = tip - obj
        r = cs.sqrt(cs.sumsqr(radial) + 1e-9)
        align = cs.dot(radial / r, patch_n)
        dist_goal = cs.sqrt(cs.sumsqr(tip - goal) + 1e-12)
        # Arrival must kill lift/clearance.  A short tip→goal chord makes
        # d_line tiny, so blocked stays on and lift holds the ball just
        # above a side patch.  The AABB ellipsoid is also larger than the
        # mesh, so the track point itself sits inside it.
        arrive = self._smooth_gate(0.03 - dist_goal, 0.0, 20.0)
        approach_gate = cs.fmax(
            self._smooth_gate(align, self.detour_align_thresh, self.detour_align_sharpness),
            arrive)

        r_core = 0.025
        lo = np.asarray(self.object_aabb_lo, dtype=np.float64)
        hi = np.asarray(self.object_aabb_hi, dtype=np.float64)
        half = cs.DM(0.5 * (hi - lo) + 0.008)
        center_local = cs.DM(0.5 * (hi + lo))
        R = self._quat_wxyz_to_rot(x[3:7])
        tip_local = R.T @ (tip - obj) - center_local
        rho = cs.sqrt(cs.sumsqr(tip_local / half) + 1e-9)
        U_clear = (1.0 - approach_gate) * (
            self._smooth_relu(1.0 - rho) ** 2 + self._smooth_relu(r_core - r) ** 2)

        chord = goal - tip
        chord_len = cs.sqrt(cs.sumsqr(chord) + 1e-9)
        d_line = cs.sqrt(cs.sumsqr(cs.cross(chord, obj - tip)) + 1e-12) / chord_len
        blocked = (1.0 - arrive) * self._smooth_gate(0.04 - d_line, 0.0, 12.0)
        z_clear = self._object_top_z(obj, x[3:7]) + 0.015
        U_att = cs.sumsqr(tip - goal)
        U_lift = blocked * self._smooth_relu(z_clear - tip[2]) ** 2
        return (self.detour_attract_coef * U_att
                + self.detour_repel_coef * U_clear
                + self.detour_lift_coef * U_lift)

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
