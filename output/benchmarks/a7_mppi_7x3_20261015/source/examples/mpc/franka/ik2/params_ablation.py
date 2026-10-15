import casadi as cs
import re
from pathlib import Path
import numpy as np
import trimesh

from utils import rotations
from planning.attract_function import compute_scalar_potential_and_gradient
from planning.mlqp_point import LambdaContactControlOptimizer
from models.soft_scm_contact import (
    SOFT_SCM_RESPONSE_TIME, SOFT_SCM_DAMPING_RATIO, SOFT_SCM_REGULARIZATION,
    normalize_contact_model, soft_scm_coefficients,
)
from examples.mpc.fingertips.test.params import canonical_object_meshes
from models.object_collision_mesh import decomposed_collision_mesh
from planning.runtime_compat import resolve_solver_backend
from planning.contact_limits import contact_force_limit, contact_command_force_limit
from examples.mpc.franka.ik2.scene_friction import (
    DEFAULT_TANGENTIAL_FORCE_PENALTY, configure_scene_friction,
)

# Legacy yaw sampler's factorization; physical standing is handled separately.
_GOAL_GEOM_Q = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0], dtype=np.float64)


def add_via_init_pose_args(parser):
    """Same tilted-start flags as examples/mpc/fingertips/test/test_0902.py."""
    parser.add_argument(
        "--random_init_tilt",
        dest="random_init_tilt",
        action="store_true",
        help="Randomize the initial object tilt so flip starts are not upright.",
    )
    parser.add_argument("--no_random_init_tilt", dest="random_init_tilt", action="store_false")
    parser.set_defaults(random_init_tilt=True)
    parser.add_argument("--init_tilt_deg", type=float, default=75.0)
    parser.add_argument("--init_tilt_min_deg", type=float, default=35.0)
    return parser


def _tilted_init_quaternion(args, yaw_angle, target_type="rotation"):
    """Sample tilt only for flipping; rotation preserves the target's up axis."""
    yaw = float(np.asarray(yaw_angle, dtype=np.float64).reshape(-1)[0])
    if target_type == "rotation":
        return rotations.rpy_to_quaternion([yaw, 0.0, 0.0])
    tilt = np.pi / 2 if target_type == "ground-rotation" else 0.0
    axis_angle = 0.0
    if getattr(args, "random_init_tilt", False):
        max_deg = float(getattr(args, "init_tilt_deg", 90.0))
        min_deg = float(getattr(args, "init_tilt_min_deg", 90.0))
        if not np.isfinite([min_deg, max_deg]).all() or not 0 <= min_deg <= max_deg <= 180:
            raise ValueError("initial tilt requires 0 <= init_tilt_min_deg <= init_tilt_deg <= 180")
        min_tilt, max_tilt = np.deg2rad([min_deg, max_deg])
        tilt = np.sqrt(min_tilt ** 2 + np.random.rand() * (max_tilt ** 2 - min_tilt ** 2))
        axis_angle = 2.0 * np.pi * np.random.rand()
    # A single horizontal-axis rotation gives the requested physical tilt.
    # Splitting it into Euler pitch/roll changes the actual inclination.
    tilt_q = rotations.angle_dir_to_quat(tilt, [np.sin(axis_angle), np.cos(axis_angle), 0.0])
    yaw_q = rotations.rpy_to_quaternion([yaw, 0.0, 0.0])
    return rotations.quaternion_multiply(yaw_q, tilt_q)


def _quat_wxyz_to_R(quat_wxyz):
    qw, qx, qy, qz = np.asarray(quat_wxyz, dtype=float).reshape(4)
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _lift_init_height_for_tilt(init_height, quat_wxyz, local_bounds, table_height):
    rot = _quat_wxyz_to_R(quat_wxyz)
    corners = np.array(np.meshgrid(*zip(local_bounds[0], local_bounds[1]))).T.reshape(-1, 3)
    min_world_z = float(np.min(corners @ rot[2, :]))
    return max(float(init_height), float(table_height) + 0.002 - min_world_z)


def _xml_obj_geom_quat_wxyz(model_path):
    """Authored geom-to-body offset, not a physical standing pose (wxyz)."""
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    try:
        text = open(model_path, "r", encoding="ascii").read()
    except OSError:
        return identity
    tag = None
    for match in re.finditer(r"<geom\b[^>]*>", text):
        chunk = match.group(0)
        if re.search(r'\bname="obj"', chunk):
            tag = chunk
            break
    if tag is None:
        return identity
    quat_m = re.search(r'\bquat="([^"]+)"', tag)
    if quat_m is None:
        return identity
    quat = np.fromstring(quat_m.group(1), sep=" ", dtype=np.float64)
    if quat.size != 4:
        return identity
    nrm = float(np.linalg.norm(quat))
    if nrm < 1e-9:
        return identity
    return quat / nrm


def _source_standing_quat_wxyz(model_path):
    """Physical standing pose of the audited source STL, before XML baking.

    All seven fingertip source meshes stand with +Z up. Five XML files
    rotate that geometry by +90 Y, so their *actor* meshes stand with +X up.
    Do not mistake the authored frame transform for a standing rotation.
    """
    name = Path(model_path).stem.replace('env_fingertips_', '', 1)
    if name not in ('elephant', 'stanford_bunny2', 'piggy_bank', 'mug',
                    'teapot', 'rubber_duck', 'foam_brick'):
        raise ValueError('Source standing pose has not been audited for ' + name)
    return np.array([1., 0., 0., 0.], dtype=np.float64)


def _actor_standing_quat_wxyz(model_path, extracted):
    """Undo the baked geom rotation to stand the actor's actual mesh."""
    source_q = _source_standing_quat_wxyz(model_path)
    if not extracted:
        return source_q
    geom_q = _xml_obj_geom_quat_wxyz(model_path)
    return rotations.quaternion_multiply(source_q, geom_q * [1., -1., -1., -1.])


def _supported_mesh_height(quat_wxyz, vertices, table_height, clearance=0.):
    """Place the lowest collision vertex on the table in the target pose."""
    return float(table_height) + float(clearance) - float(
        np.min(np.asarray(vertices, dtype=np.float64) @ _quat_wxyz_to_R(quat_wxyz)[2, :]))


def _ground_rotation_target_q(target_roll_sample):
    """Legacy ground-rotation sample, equivalent to a world-Z yaw only.

    The -90 Y and +90 Y factors cancel, conjugating the sampled roll into
    yaw. The caller must compose the actual actor standing orientation.
    """
    body_target_q = rotations.rpy_to_quaternion(
        np.hstack([0.0, -0.5 * np.pi, np.pi * float(target_roll_sample) - 0.5 * np.pi])
    )
    return rotations.quaternion_multiply(body_target_q, _GOAL_GEOM_Q)


def _actor_quat_from_body(body_q, standing_q, extracted):
    """Map a MuJoCo body quaternion into the Isaac actor frame."""
    body_q = np.asarray(body_q, dtype=np.float64).reshape(4)
    if extracted:
        return body_q
    return rotations.quaternion_multiply(body_q, np.asarray(standing_q, dtype=np.float64).reshape(4))


def _box_inertia_diag(mass, aabb_lo, aabb_hi):
    """Principal box inertia for a uniform solid of the given AABB."""
    extents = np.asarray(aabb_hi, dtype=np.float64).reshape(3) - np.asarray(aabb_lo, dtype=np.float64).reshape(3)
    extents = np.maximum(np.abs(extents), 1e-4)
    mass = float(mass)
    return np.array(
        [
            mass / 12.0 * (extents[1] ** 2 + extents[2] ** 2),
            mass / 12.0 * (extents[0] ** 2 + extents[2] ** 2),
            mass / 12.0 * (extents[0] ** 2 + extents[1] ** 2),
        ],
        dtype=np.float64,
    )


def build_lambda_optimizer(param, args):
    """Same acados contact NLP as examples/mpc/fingertips/test/params.py."""
    runtime_cost = bool(getattr(param, 'runtime_object_cost_', False))
    grouped_response = bool(getattr(param, 'grouped_contact_response_', False))
    extracted = bool(getattr(param, "_collision_mesh_extracted", False))
    shared = None
    if getattr(param, 'shared_contact_dynamics_', False):
        shared = dict(mass=np.r_[np.full(3, param.sim_obj_mass_), param.sim_obj_inertia_diag_,
                                  np.full(3, getattr(args, 'mpc_tip_effective_mass', 2.5))].tolist(),
            h=param.h_, tip_mass=float(getattr(args, 'mpc_tip_effective_mass', 2.5)),
            damping=float(getattr(args, 'mpc_tip_damping', 40.)),
            response_time=param.soft_scm_response_time, rigid_response_time=.01,
            damping_ratio=param.soft_scm_damping_ratio, regularization=param.soft_scm_regularization,
            blend_speed=.005, variant=param.contact_dynamics_variant_)
    opt = LambdaContactControlOptimizer(
        mesh_path=param.mesh_path_,
        obj_mass=float(getattr(param, "lambda_obj_mass_", 0.01)),
        arm_friction=getattr(param, 'mu_fingertip_object_', param.mu_object_),
        friction_cone='pyramid',
        contact_stiffness=param.model_params,
        time_step=(param.h_ if runtime_cost else getattr(param, 'lambda_h_', .05)),
        max_contacts=4 if grouped_response else 10,
        sample_num=args.sample_num,
        top_k=getattr(args, "top_k", 2),
        pos_coef=args.pos_coef,
        ori_coef=args.ori_coef,
        friction_reg_coef=getattr(args, "friction_reg_coef", DEFAULT_TANGENTIAL_FORCE_PENALTY),
        force_reg_coef=getattr(args, "force_reg_coef", 0.01),
        max_contact_force=(contact_command_force_limit(args, 600.) if runtime_cost else contact_force_limit(args)),
        contact_switch_radius=getattr(args, "contact_switch_radius", 0.03),
        contact_switch_margin_ratio=getattr(args, "contact_switch_margin_ratio", 0.2),
        contact_switch_margin_abs=getattr(args, "contact_switch_margin_abs", 1e-3),
        fingertip_clearance=getattr(args, "fingertip_clearance", 0.011),
        normal_stability_cos=getattr(args, "normal_stability_cos", 0.95),
        solver=resolve_solver_backend(
            getattr(args, "solver_backend", getattr(args, "solver", "auto"))
        ),
        torch_max_iter=getattr(args, "torch_max_iter", 100),
        obj_inertia=param.obj_inertia_ if runtime_cost else None,
        wrench_is_force=runtime_cost,
        contact_dynamics='quasi-dynamic' if runtime_cost else None,
        grouped_contact_response=grouped_response,
        robot_stiffness=600. if grouped_response else None,
        contact_model=getattr(param, 'contact_model', 'soft-scm'),
        soft_scm_response_time=getattr(param, 'soft_scm_response_time', SOFT_SCM_RESPONSE_TIME),
        soft_scm_damping_ratio=getattr(param, 'soft_scm_damping_ratio', SOFT_SCM_DAMPING_RATIO),
        soft_scm_regularization=getattr(param, 'soft_scm_regularization', SOFT_SCM_REGULARIZATION),
        collision_hull=(bool(getattr(param, "collision_hull", True)) and not extracted),
        surface_sampling=getattr(param, 'contact_sampling_', 'vertices') == 'surface',
        collision_part_paths=getattr(param, 'collision_mesh_paths_', None),
        shared_response=shared,
    )
    guidance = str(getattr(args, "cso_guidance", "cso"))
    if guidance not in ("cso", "pose", "random", "nearest"):
        raise ValueError("cso_guidance must be cso, pose, random, or nearest")
    opt.cso_guidance = guidance
    # One stream per trial so random guidance is reproducible across matched runs.
    opt._cso_rng = np.random.default_rng(int(getattr(param, "trial_seed_", 0)))
    return opt


class ExplicitMPCParams:
    def __init__(self, args, rand_seed=1, target_type="ground-rotation", mpc_model="explicit",
                 build_optimizer=True):
        self.ablation_mode = str(getattr(args, 'ablation_mode', 'full'))
        self.ablation_collision_weight = float(getattr(args, 'ablation_collision_weight', 10000.0))
        self.ablation_collision_radius = float(getattr(args, 'ablation_collision_radius', 0.01))
        self.ablation_collision_tolerance = float(getattr(args, 'ablation_collision_tolerance', 0.0005))
        if self.ablation_mode == 'a3' and not (
                np.isfinite(self.ablation_collision_weight) and self.ablation_collision_weight > 0
                and np.isfinite(self.ablation_collision_radius) and self.ablation_collision_radius > 0
                and 0 <= self.ablation_collision_tolerance < self.ablation_collision_radius):
            raise ValueError('A3 requires positive finite collision weight/radius and 0 <= tolerance < radius')
        self.contact_cost_param = float(args.contact_cost_param)
        self.cpo_variant = str(getattr(args, "cpo_variant", "full"))
        if self.cpo_variant not in ("full", "no_object_center", "direct"):
            raise ValueError("cpo_variant must be full, no_object_center, or direct")
        self.cpo_direct_trajectory = self.cpo_variant == "direct"
        self.trial_seed_ = int(rand_seed)
        self.attract_coef = float(args.attract_coef)
        self.field_cost_weight = 0.05
        self.quadratic_contact_track = False
        self.reject_coef = args.reject_coef
        self.spline_escape_cost = bool(getattr(args, "spline_escape_cost", 0))
        self.contact_coef = args.contact_coef
        self.reject_dis = args.reject_dis
        self.pos_coef = float(getattr(args, "pos_coef", 500.0))
        self.ori_coef = float(getattr(args, "ori_coef", 20.0))
        self.contact_dynamics_variant_ = getattr(args, 'contact_dynamics_variant', 'legacy')
        self.shared_contact_dynamics_ = bool(getattr(args, 'shared_contact_dynamics', False))
        if (self.shared_contact_dynamics_ or self.contact_dynamics_variant_ != 'legacy') and (
                not getattr(args, 'two_stage_object_cost', False) or getattr(args, 'contact_model', 'soft-scm') != 'soft-scm'
                or getattr(args, 'online_scm', False) or not getattr(args, 'mpc_carry_velocity', True)):
            raise ValueError('contact dynamics experiments require two-stage soft-SCM, velocity state and no online-SCM')
        self.runtime_object_cost_ = bool(getattr(args, 'two_stage_object_cost', False))
        if self.runtime_object_cost_ and (not np.isfinite(self.ori_coef) or self.ori_coef <= 0):
            raise ValueError('two-stage object cost requires ori_coef > 0')
        self.smooth_contact_detour = bool(getattr(args, "ideal_contact_pose", False))
        self.detour_attract_coef = float(getattr(args, "detour_attract_coef", 80.0))
        self.detour_repel_coef = float(getattr(args, "detour_repel_coef", 40.0))
        self.detour_lift_coef = float(getattr(args, "detour_lift_coef", 25.0))
        self.detour_align_thresh = float(getattr(args, "detour_align_thresh", 0.50))
        self.detour_align_sharpness = float(getattr(args, "detour_align_sharpness", 8.0))
        self.object_circumradius = 0.08
        self.object_aabb_lo = np.array([-0.06, -0.04, -0.04], dtype=np.float64)
        self.object_aabb_hi = np.array([0.06, 0.04, 0.06], dtype=np.float64)

        self.model_path_ = "./envs/xmls/env_fingertips_" + args.obj + ".xml"
        self.source_mesh_path_ = "envs/assets/objects/" + args.obj + ".stl"
        self.mesh_path_ = self.source_mesh_path_
        self.visual_mesh_path_ = self.source_mesh_path_
        self.object_names_ = ["obj"]
        requested_hull = getattr(args, "collision_hull", None)
        self.collision_hull = True if requested_hull is None else bool(requested_hull)
        # Use one authored, version-independent convex hull so ranking,
        # PhysX, and the pose metric share the same object frame.
        self._collision_mesh_extracted = False
        self.collision_mesh_paths_ = []
        geometry = getattr(args, 'collision_geometry', 'auto')
        if geometry not in ('auto', 'hull', 'decomposed'):
            raise ValueError('collision_geometry must be auto, hull, or decomposed')
        use_parts = (geometry == 'decomposed' or
                     (geometry == 'auto' and args.obj in ('mug', 'rubber_duck')))
        if self.collision_hull:
            # Do not consume MuJoCo's compiled mesh arrays here.  Their frame
            # and centering conventions differ between MuJoCo releases and
            # change both ranking contacts and PhysX geometry.
            self.mesh_path_, self.visual_mesh_path_ = canonical_object_meshes(
                self.source_mesh_path_, self.model_path_
            )
            self._collision_mesh_extracted = True
            if use_parts:
                self.mesh_path_, self.collision_mesh_paths_ = decomposed_collision_mesh(
                    self.visual_mesh_path_, manifest_path=getattr(args, 'collision_manifest', None))
            elif getattr(args, 'collision_manifest', None):
                raise ValueError('--collision-manifest requires decomposed collision geometry')
        self.collision_geometry_ = 'decomposed' if self.collision_mesh_paths_ else 'hull'
        sampling = getattr(args, 'contact_sampling', 'auto')
        if sampling not in ('auto', 'surface', 'vertices'):
            raise ValueError('contact_sampling must be auto, surface, or vertices')
        self.contact_sampling_ = ('surface' if args.obj in ('mug', 'rubber_duck') else 'vertices') if sampling == 'auto' else sampling
        try:
            mesh = trimesh.load_mesh(self.mesh_path_, process=False)
            self.object_vertices_ = np.asarray(mesh.vertices, dtype=np.float64).copy()
            hull = mesh.convex_hull
            self.object_hull_normals_ = np.asarray(hull.face_normals, dtype=np.float64)
            self.object_hull_offsets_ = np.sum(self.object_hull_normals_ * hull.triangles_center, axis=1)
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            self.object_aabb_lo = bounds[0].copy()
            self.object_aabb_hi = bounds[1].copy()
            self.object_circumradius = float(np.linalg.norm(0.5 * (bounds[1] - bounds[0])))
        except Exception:
            self.object_circumradius = 0.08
            self.object_vertices_ = np.array(np.meshgrid(*zip(
                self.object_aabb_lo, self.object_aabb_hi))).T.reshape(-1, 3)

        self.frame_skip_ = int(10)
        # Match the fingertip --rollout planner interval.  Isaac physics keeps
        # its own sim_dt_ (DyWA 12.5 ms); this h_ is only the acados model step.
        self.h_ = 0.02
        self.lambda_h_ = 0.05

        self.n_robot_qpos_ = 3
        self.n_qpos_ = 10
        self.n_qvel_ = 9
        self.n_cmd_ = 3

        self.jc_kp_ = 200
        self.jc_damping_ = 10
        self.proximity_threshold_ = 0.1
        self.fingertip_geoms = [
            "left_finger_tip_pad_1", "left_finger_tip_pad_2", "left_finger_tip_pad_3",
            "left_finger_tip_pad_4", "left_finger_tip_pad_5",
            "right_finger_tip_pad_1", "right_finger_tip_pad_2", "right_finger_tip_pad_3",
            "right_finger_tip_pad_4", "right_finger_tip_pad_5",
        ]

        np.random.seed(100 + rand_seed)

        self.table_height = 0.35
        extracted = bool(self._collision_mesh_extracted)
        self.standing_q_ = _actor_standing_quat_wxyz(self.model_path_, extracted)
        init_xy_rand = 0.1 * np.random.rand(2)
        init_xy_rand[0] += 0.3

        yaw_angle = float(2.0 * np.pi * np.random.rand() - np.pi)
        # Draw the target roll before the optional tilt so enabling
        # --random_init_tilt only changes the initial state.
        target_roll_sample = float(np.random.rand())
        target_type = str(getattr(args, "target_type", target_type) or target_type)
        if target_type == "ground-rotation":
            # Preserve the legacy world-yaw distribution, with the actual
            # actor-frame standing rotation rather than an elephant offset.
            self.target_q_ = rotations.quaternion_multiply(
                _ground_rotation_target_q(target_roll_sample), self.standing_q_)
            target_xy = np.array([0.40, 0.10], dtype=np.float64)
        elif target_type == "rotation":
            target_xy_rand = 0.1 * np.random.rand(2)
            target_xy_rand[0] += 0.35
            target_yaw = np.pi * np.random.rand(1) - np.pi / 2
            target_yaw_q = rotations.rpy_to_quaternion(np.hstack([target_yaw, 0, 0]))
            self.target_q_ = rotations.quaternion_multiply(target_yaw_q, self.standing_q_)
            target_xy = target_xy_rand
        else:
            raise ValueError(f"Target type {target_type} not supported")

        self.target_p_ = np.r_[target_xy, _supported_mesh_height(
            self.target_q_, self.object_vertices_, self.table_height)]

        from planning.object_pose_cost import target_up_axis
        self.object_up_axis_ = target_up_axis(self.target_q_)
        self.flip_enter_angle_deg_ = float(getattr(args, 'flip_enter_angle_deg', 10.))
        self.flip_stable_time_ = float(getattr(args, 'flip_stable_time', .3))

        init_tilt_q = _tilted_init_quaternion(args, yaw_angle, target_type)
        init_obj_quat_rand = rotations.quaternion_multiply(init_tilt_q, self.standing_q_)
        init_height = _supported_mesh_height(
            init_obj_quat_rand, self.object_vertices_, self.table_height, clearance=0.002)

        self.init_obj_qpos_ = np.hstack((init_xy_rand, init_height, init_obj_quat_rand))
        self.init_robot_qpos_ = np.array([0.0, -0.785, 0.0, -2.356, 0, 1.571, 0.785])
        # Same start as fingertips --rollout: beside the object, just above
        # the table.  The Franka ready pose parks the sphere above the
        # elephant, so the keep-out via hovers and never drops onto press.
        toward_base = -np.asarray(self.init_obj_qpos_[:2], dtype=np.float64)
        toward_norm = float(np.linalg.norm(toward_base))
        if toward_norm < 1e-6:
            toward_base = np.array([-1.0, 0.0], dtype=np.float64)
        else:
            toward_base = toward_base / toward_norm
        self.init_fingertip_pos_ = np.array(
            [
                float(self.init_obj_qpos_[0] + 0.10 * toward_base[0]),
                float(self.init_obj_qpos_[1] + 0.10 * toward_base[1]),
                float(self.table_height + 0.02),
            ],
            dtype=np.float64,
        )
        if target_type == "ground-rotation" and getattr(args, "random_init_tilt", False):
            self.init_fingertip_pos_[:2] += 0.06 * (2.0 * np.random.rand(2) - 1.0)
            self.init_fingertip_pos_[2] += 0.03 * float(np.random.rand())

        combine_mode = ('max' if getattr(args, 'physics_backend', 'isaac') == 'mjwarp'
                        and getattr(args, 'mjwarp_scene', 'isaac') == 'legacy' else 'average')
        configure_scene_friction(self, combine_mode)
        self.n_mj_q_ = self.n_qpos_
        self.n_mj_v_ = self.n_qvel_
        self.max_ncon_ = 10

        self.obj_inertia_ = np.identity(6)
        self.obj_inertia_[0:3, 0:3] = 50 * np.eye(3)
        self.obj_inertia_[3:, 3:] = 0.05 * np.eye(3)
        self.grouped_contact_response_ = self.runtime_object_cost_ and getattr(args, 'contact_model', 'soft-scm') == 'soft-scm'
        self.robot_stiff_ = np.diag(self.n_cmd_ * [600 if self.runtime_object_cost_ else 100])

        Q = np.zeros((self.n_qvel_, self.n_qvel_))
        Q[:6, :6] = self.obj_inertia_
        Q[6:, 6:] = self.robot_stiff_
        self.Q = Q

        self.obj_mass_ = 0.01
        # Planner, ranking, and PhysX actor mass all match fingertips --rollout.
        self.lambda_obj_mass_ = 0.01
        self.sim_obj_mass_ = 0.01
        self.sim_obj_inertia_diag_ = _box_inertia_diag(
            self.sim_obj_mass_, self.object_aabb_lo, self.object_aabb_hi
        )
        self.gravity_ = np.array([0.00, 0.00, -9.8, 0.0, 0.0, 0.0])
        self.model_params = args.model_param
        # This planner commands Cartesian displacements and Q_robot=K_robot,
        # exactly as main. Using h*M^-1*f here shrinks dq by h^2 (=1/2500
        # at h=.02) and makes the approaching arm crawl.
        self.contact_dynamics = 'quasi-dynamic'
        self.contact_model = normalize_contact_model(getattr(args, 'contact_model', 'soft-scm'))
        self.online_scm_ = bool(getattr(args, 'online_scm', False) or getattr(args, 'scm_parameters', '') or getattr(args, 'scm_record_data', False)) and self.contact_model == 'soft-scm'
        self.soft_scm_response_time = float(getattr(args, 'soft_scm_response_time', SOFT_SCM_RESPONSE_TIME))
        self.soft_scm_damping_ratio = float(getattr(args, 'soft_scm_damping_ratio', SOFT_SCM_DAMPING_RATIO))
        self.soft_scm_regularization = float(getattr(args, 'soft_scm_regularization', SOFT_SCM_REGULARIZATION))
        soft_scm_coefficients(self.h_, self.soft_scm_response_time,
                              self.soft_scm_damping_ratio, self.soft_scm_regularization)

        self.mpc_model = mpc_model
        requested_solver = getattr(
            args, "solver_backend", getattr(args, "solver", "auto")
        )
        self.solver_backend_ = str(requested_solver)
        self.torch_solver = resolve_solver_backend(requested_solver)
        self.planner_solver_ = self.torch_solver
        self.mpc_horizon_ = 5
        self.ipopt_max_iter_ = 100
        self.comple_relax = 0.01

        self.mpc_u_lb_ = -float(getattr(args, "mpc_step_limit", 0.005))
        self.mpc_u_ub_ = -self.mpc_u_lb_
        # Same fingertip box as fingertips --rollout, shifted by the table.
        fts_q_lb = np.array([-10, -10, self.table_height - 0.01])
        fts_q_ub = np.array([10, 10, self.table_height + 1.0])
        self.mpc_q_lb_ = np.hstack((-1e7 * np.ones(7), fts_q_lb))
        self.mpc_q_ub_ = np.hstack((1e7 * np.ones(7), fts_q_ub))
        self.sol_guess_ = None
        self.max_env_contacts_ = 4
        self.lambda_optimizer = build_lambda_optimizer(self, args) if build_optimizer else None

    @staticmethod
    def calculate_rotation_quaternion(x, target_position):
        direction = x[:2] - target_position[:2]
        direction = direction / cs.sqrt(cs.sumsqr(direction) + 1e-9)
        angle = cs.arctan2(direction[1], direction[0])
        half_angle = angle / 2.0
        return [cs.cos(half_angle), 0, 0, cs.sin(half_angle)]

    def init_cost_fns(self):
        symbol = cs.MX if getattr(self, 'ablation_mode', 'full') == 'a3' else cs.SX
        x = symbol.sym("x", self.n_qpos_)
        u = symbol.sym("u", self.n_cmd_)

        target_position = symbol.sym("target_position", 3)
        target_quaternion = symbol.sym("target_quaternion", 4)
        position_cost = cs.sumsqr(x[0:3] - target_position)
        quaternion_cost = 1 - cs.dot(x[3:7], target_quaternion) ** 2
        contact_cost = cs.sumsqr(x[0:3] - x[7:10])
        control_cost = cs.sumsqr(u)
        virtual_point = symbol.sym("virtual_point", 3)
        contact_point = symbol.sym("contact point", 3)

        phi_vec = symbol.sym("phi_vec", self.max_ncon_ * 4)
        jac_mat = symbol.sym("jac_mat", self.max_ncon_ * 4, self.n_qvel_)
        verify_cost_param = symbol.sym("verify_cost", 1)
        cost_param = cs.vvcat(
            [target_position, target_quaternion, phi_vec, jac_mat, verify_cost_param, virtual_point, contact_point]
        )
        object_config = None
        if getattr(self, 'runtime_object_cost_', False):
            from planning.object_pose_cost import OBJECT_COST_CONFIG_SIZE
            object_config = symbol.sym('object_config', OBJECT_COST_CONFIG_SIZE)
            contact_reference = symbol.sym('contact_reference', 7)
            cost_param = cs.vertcat(cost_param, contact_reference, object_config)
            from models.contact_geometry import attached_point_cs
            # Approach is a world path; contact tracking follows the predicted
            # object pose instead of pulling the finger toward a stale point.
            contact_point = attached_point_cs(x, contact_reference, contact_point)
        from planning.mpc_costs import cpo_stage_weights
        center_w, ref_w, phased = cpo_stage_weights(
            getattr(self, "contact_cost_param", 0.0), getattr(self, "cpo_variant", "full"))
        if getattr(self, 'ablation_mode', 'full') == 'mppi':
            # Replace the rejection/approach potentials and surface reference
            # with attraction to the predicted object center at every stage.
            # Contact geometry and the object-pose/control objectives are kept.
            base_cost = self.contact_coef * contact_cost
            final_cost = self.pos_coef * position_cost + self.ori_coef * quaternion_cost
            control_weight = 0.0
            phased = False
        elif getattr(self, 'ablation_mode', 'full') == 'a3':
            # Direct tracking of the selected p_arm_world. No object-center,
            # rejection, field, lift/place or virtual-point cost.
            base_cost = self.contact_coef * cs.sumsqr(x[7:10] - contact_point)
            base_cost += self.ablation_collision_weight * self.object_collision_cost_cs(x)
            final_cost = self.pos_coef * position_cost + self.ori_coef * quaternion_cost
            control_weight = 50.0
            phased = False
        elif not phased:
            # Direct trajectory optimization keeps reference tracking and
            # object-center attraction on every stage. The ranking trigger
            # does not switch in a lift or place potential.
            contact_point_cost = (
                cs.sumsqr(x[7:10] - contact_point)
                if getattr(self, "quadratic_contact_track", False)
                else self.log_barrier_function(x, contact_point)
            )
            base_cost = self.contact_coef * (
                center_w * contact_cost + ref_w * contact_point_cost)
            final_cost = (
                float(getattr(self, "pos_coef", 500.0)) * position_cost
                + float(getattr(self, "ori_coef", 20.0)) * quaternion_cost
            )
            control_weight = 50.0
        elif getattr(self, "smooth_contact_detour", False):
            base_cost = self._smooth_contact_detour_cost(x, virtual_point, contact_point)
            final_cost = (
                500 * position_cost
                + 20.0 * quaternion_cost
                + 20.0 * self.detour_attract_coef * cs.sumsqr(x[7:10] - virtual_point)
            )
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
            if getattr(self, "quadratic_contact_track", False):
                virtual_point_cost = cs.sumsqr(x[7:10] - virtual_point)
            else:
                virtual_point_cost = self.log_barrier_function(x, virtual_point)

            reject_distance = cs.sumsqr(x[0:2] - x[7:9]) + 1e-3
            obstacle_cost = (
                cs.DM(0)
                if self.spline_escape_cost
                else cs.if_else(reject_distance < self.reject_dis, 1 / reject_distance, 0.0)
            )
            attract_cost = (
                self.attract_coef * virtual_point_cost
                + float(getattr(self, "field_cost_weight", 0.05)) * field_cost
                + self.reject_coef * obstacle_cost
            )
            contact_point_cost = (
                cs.sumsqr(x[7:10] - contact_point)
                if getattr(self, "quadratic_contact_track", False)
                else self.log_barrier_function(x, contact_point)
            )
            if getattr(self, "rollout_press_patch", False):
                press_cost = contact_point_cost
            else:
                press_cost = center_w * contact_cost + ref_w * contact_point_cost
            base_cost = (1 - verify_cost_param) * attract_cost + self.contact_coef * verify_cost_param * press_cost
            # Lambda pose: pos_coef||Δp||² + ori_coef(1 − q·q*)².
            final_cost = (
                float(getattr(self, "pos_coef", 500.0)) * position_cost
                + float(getattr(self, "ori_coef", 20.0)) * quaternion_cost
            )
            control_weight = 50.0

        if object_config is not None:
            from planning.object_pose_cost import object_cost_cs
            final_cost = object_cost_cs(x[:7], cs.vertcat(target_position, target_quaternion), object_config)
            # Score the whole predicted trajectory: terminal-only tracking
            # neither prices intermediate tipping nor rewards early progress.
            base_cost += final_cost
            if phased and getattr(self, 'smooth_contact_detour', False):
                final_cost += 20.0 * self.detour_attract_coef * cs.sumsqr(x[7:10] - virtual_point)
        path_cost_fn = cs.Function("path_cost_fn", [x, u, cost_param], [base_cost + control_weight * control_cost])
        final_cost_fn = cs.Function("final_cost_fn", [x, cost_param], [10 * final_cost])
        return path_cost_fn, final_cost_fn

    def _collision_planes(self):
        """Use the same body-frame convex parts as PhysX, including cavities."""
        if not hasattr(self, '_ablation_planes'):
            from scipy.spatial import ConvexHull
            parts = getattr(self, 'collision_mesh_paths_', [])
            vertices = ([np.asarray(trimesh.load_mesh(path, process=False).vertices)
                         for path in parts] if parts else [self.object_vertices_])
            self._ablation_planes = []
            for points in vertices:
                # Qhull merges coplanar triangles; plane distances have metre units.
                equations = ConvexHull(points).equations
                equations = np.unique(np.round(equations, 12), axis=0)
                self._ablation_planes.append((equations[:, :3], -equations[:, 3]))
        return self._ablation_planes

    def _collision_max_function(self, size):
        """Exact max with the same averaged subgradient, without AD expansion.

        The gradient is the uniform average of maximizers, matching CasADi's
        mmax including ties. Its derivative is zero, as in mmax's branch AD.
        Providing that Jacobian explicitly prevents an O(size**2) graph on
        CasADi 3.6; the collision objective and all its derivatives are retained.
        """
        if not hasattr(self, '_ablation_max_functions'):
            self._ablation_max_functions = {}
        if size not in self._ablation_max_functions:
            values = cs.SX.sym('max_values', size)
            output = cs.SX.sym('max_output')
            mask = cs.if_else(values == output, 1., 0.)
            name = 'ablation_collision_max_%d' % size
            jacobian = cs.Function('jac_' + name, [values, output],
                [(mask / cs.sum1(mask)).T], {'cse': True})
            self._ablation_max_functions[size] = cs.Function(name, [values],
                [cs.mmax(values)], {'custom_jacobian': jacobian, 'jac_penalty': 0.,
                                   'never_inline': True})
        return self._ablation_max_functions[size]

    def object_collision_cost_cs(self, state):
        from models.contact_geometry import rotation_cs
        local_tip = rotation_cs(state[3:7]).T @ (state[7:10] - state[:3])
        if not hasattr(self, '_ablation_collision_fn'):
            # MX preserves the max call boundaries on both CasADi 3.6 and 3.7.
            tip = cs.MX.sym('collision_tip_local', 3)
            radius = self.ablation_collision_radius - self.ablation_collision_tolerance
            penalties = []
            for normals, offsets in self._collision_planes():
                distances = cs.DM(normals) @ tip - cs.DM(offsets)
                gap = self._collision_max_function(len(offsets))(distances)
                penalties.append(cs.fmax(radius - gap, 0)**2)
            cost = (penalties[0] if len(penalties) == 1 else
                    self._collision_max_function(len(penalties))(cs.vertcat(*penalties)))
            self._ablation_collision_fn = cs.Function('ablation_object_collision',
                [tip], [cost], {'never_inline': True, 'cse': True})
        # Exact chain rule transports the penalty with object pose. The
        # inflated convex parts permit legal sphere contact without penalty.
        return self._ablation_collision_fn(local_tip)

    def object_collision_cost_torch(self, state):
        import torch
        from models.contact_geometry import rotation_torch
        local_tip = rotation_torch(state[3:7]).T @ (state[7:10] - state[:3])
        radius = self.ablation_collision_radius - self.ablation_collision_tolerance
        penalties = []
        for normals, offsets in self._collision_planes():
            n = torch.as_tensor(normals, dtype=state.dtype, device=state.device)
            b = torch.as_tensor(offsets, dtype=state.dtype, device=state.device)
            penalties.append(torch.clamp(radius - torch.max(n @ local_tip - b), min=0).square())
        return torch.max(torch.stack(penalties))

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
        z_extent = (
            cs.fabs(R[2, 0]) * half[0]
            + cs.fabs(R[2, 1]) * half[1]
            + cs.fabs(R[2, 2]) * half[2]
        )
        return center_world[2] + z_extent

    def _smooth_contact_detour_cost(self, x, virtual_point, contact_point):
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
        arrive = self._smooth_gate(0.03 - dist_goal, 0.0, 20.0)
        approach_gate = cs.fmax(
            self._smooth_gate(align, self.detour_align_thresh, self.detour_align_sharpness),
            arrive,
        )

        r_core = 0.025
        lo = np.asarray(self.object_aabb_lo, dtype=np.float64)
        hi = np.asarray(self.object_aabb_hi, dtype=np.float64)
        half = cs.DM(0.5 * (hi - lo) + 0.008)
        center_local = cs.DM(0.5 * (hi + lo))
        R = self._quat_wxyz_to_rot(x[3:7])
        tip_local = R.T @ (tip - obj) - center_local
        rho = cs.sqrt(cs.sumsqr(tip_local / half) + 1e-9)
        U_clear = (1.0 - approach_gate) * (
            self._smooth_relu(1.0 - rho) ** 2 + self._smooth_relu(r_core - r) ** 2
        )

        chord = goal - tip
        chord_len = cs.sqrt(cs.sumsqr(chord) + 1e-9)
        d_line = cs.sqrt(cs.sumsqr(cs.cross(chord, obj - tip)) + 1e-12) / chord_len
        blocked = (1.0 - arrive) * self._smooth_gate(0.04 - d_line, 0.0, 12.0)
        z_clear = self._object_top_z(obj, x[3:7]) + 0.015
        U_att = cs.sumsqr(tip - goal)
        U_lift = blocked * self._smooth_relu(z_clear - tip[2]) ** 2
        return (
            self.detour_attract_coef * U_att
            + self.detour_repel_coef * U_clear
            + self.detour_lift_coef * U_lift
        )

    @staticmethod
    def log_barrier_function(x, virtual_point, epsilon=1e-3):
        diff = x[7:10] - virtual_point
        squared_norm = cs.dot(diff, diff) + epsilon
        return cs.log(squared_norm)
