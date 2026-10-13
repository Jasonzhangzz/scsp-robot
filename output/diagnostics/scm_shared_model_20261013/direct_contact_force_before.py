"""Physical execution of a lambda contact force at a body-fixed surface point.

MuJoCo free joints store translation velocity in world coordinates and
angular velocity in body coordinates. Lambda uses body coordinates for both.
The ideal actuator has no colliding fingertip; only the object and its table
contacts generate motion. No object pose or velocity is overwritten.
"""
import numpy as np
import mujoco


def add_direct_force_args(parser):
    parser.add_argument('--execution-force-cap', type=float, default=0.2,
                        help='Force bound in N, imposed inside lambda optimization; 0 uses max_contact_force.')
    parser.add_argument('--execution-torque-cap', type=float, default=0.0,
                        help='Optional N*m bound; scales the entire point force, preserving its moment arm.')
    parser.add_argument('--force-response-time', type=float, default=0.008,
                        help='SCM support response time in seconds for direct physical execution.')
    parser.add_argument('--success-hold-steps', type=int, default=10,
                        help='Require this many consecutive successful control intervals, then a settled object.')
    parser.add_argument('--force-output', default=None,
                        help='Write physical rollout traces and trial summaries to this JSON file.')
    return parser


def object_frame(env):
    body = env.model_.body('obj').id
    return np.asarray(env.data_.xmat[body]).reshape(3, 3).copy()


def object_velocity_body(env):
    """Convert this scene's object free-joint velocity to lambda coordinates."""
    return np.r_[object_frame(env).T @ env.data_.qvel[:3], env.data_.qvel[3:6]]


def object_mass_body(env):
    """Full generalized mass, including inertial-axis rotation and COM offset."""
    model, data = env.model_, env.data_
    mass = np.zeros((model.nv, model.nv))
    try:
        mujoco.mj_fullM(model, data, mass)
    except TypeError:  # MuJoCo <= 3.3
        mujoco.mj_fullM(model, mass, data.qM)
    frame = np.eye(6)
    frame[:3, :3] = object_frame(env)
    # The angular free-joint coordinates already are in the body frame.
    # Round only floating-point rotation noise to reuse the same solver cache
    # across initial orientations of one object.
    return np.round(frame.T @ mass[:6, :6] @ frame, decimals=14)


def environment_contacts_body(env):
    """Table contact rays, using MuJoCo's effective friction and signed gaps."""
    model, data = env.model_, env.data_
    body = model.body('obj').id
    table = model.geom('table').id
    frame = np.eye(6)
    frame[:3, :3] = object_frame(env)
    rows, gaps = [], []
    for con in data.contact[:data.ncon]:
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 == table and int(model.geom_bodyid[g2]) == body:
            sign = 1.0
        elif g2 == table and int(model.geom_bodyid[g1]) == body:
            sign = -1.0
        else:
            continue
        normal, t1, t2 = np.asarray(con.frame).reshape(3, 3)
        normal = sign * normal  # table -> object
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, con.pos, body)
        mu = float(con.friction[0])
        rays = np.stack((normal + mu*t1, normal + mu*t2,
                         normal - mu*t1, normal - mu*t2))
        rows.extend(rays @ jac[:, :6] @ frame)
        gaps.extend([float(con.dist)] * 4)
    return np.asarray(rows).reshape(-1, 6), np.asarray(gaps)


def gravity_wrench_body(env):
    body = env.model_.body('obj').id
    force = object_frame(env).T @ (env.model_.body_mass[body] * env.model_.opt.gravity)
    return np.r_[force, np.cross(env.model_.body_ipos[body], force)]


def configure_direct_force_optimizer(param, env, args):
    """Configure lambda in N with the same mass and timestep as mj_step."""
    from planning.mlqp_point import LambdaContactControlOptimizer

    model = env.model_
    tip = model.geom('fingertip0').id
    model.geom_contype[tip] = 0
    model.geom_conaffinity[tip] = 0
    env.data_.ctrl[:] = 0.0
    mujoco.mj_forward(model, env.data_)
    env.direct_force_ = True
    mass = object_mass_body(env)
    force_cap = float(args.execution_force_cap)
    if not np.isfinite(force_cap) or force_cap < 0:
        raise ValueError('execution_force_cap must be finite and nonnegative')
    limit = min(float(args.max_contact_force), force_cap) if force_cap else float(args.max_contact_force)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError('max_contact_force must be finite and positive')
    param.obj_mass_ = float(model.body_mass[model.body('obj').id])
    param.obj_inertia_ = mass
    param.lambda_optimizer = LambdaContactControlOptimizer(
        param.mesh_path_, obj_mass=param.obj_mass_, arm_friction=param.mu_object_,
        time_step=float(model.opt.timestep) * int(param.frame_skip_),
        obj_inertia=mass, dynamics_scale=param.dynamics_scale_,
        wrench_is_force=True, sample_num=args.sample_num,
        max_contact_force=limit, pos_coef=args.pos_coef, ori_coef=args.ori_coef,
        friction_reg_coef=args.friction_reg_coef, force_reg_coef=args.force_reg_coef,
        solver=getattr(args, 'solver', 'acados'), grouped_contact_response=True,
        soft_scm_response_time=args.force_response_time,
        soft_scm_damping_ratio=1.0, surface_sampling=True)
    return param.lambda_optimizer


def step_point_force(env, point_body, force_body, nstep=None):
    """Apply a body-fixed force at its surface point on every physics substep.

    mj_applyFT computes the torque about MuJoCo's actual COM. Updating the
    world point/force each substep keeps the contact attached as the object
    rotates, instead of applying a stale world wrench over the interval.
    """
    point = np.asarray(point_body, dtype=float).reshape(3)
    force = np.asarray(force_body, dtype=float).reshape(3)
    if not np.isfinite(np.r_[point, force]).all():
        raise ValueError('point and force must be finite')
    nstep = int(env.param_.frame_skip_ if nstep is None else nstep)
    if nstep < 1:
        raise ValueError('nstep must be positive')
    model, data = env.model_, env.data_
    body = model.body('obj').id
    torque = np.cross(point - model.body_ipos[body], force)
    try:
        for _ in range(nstep):
            mujoco.mj_forward(model, data)
            rot = object_frame(env)
            point_world = data.xpos[body] + rot @ point
            data.qfrc_applied[:] = 0.0
            data.xfrc_applied[:] = 0.0
            mujoco.mj_applyFT(model, data, rot @ force, np.zeros(3),
                             point_world, body, data.qfrc_applied)
            tip_body = getattr(env, 'fingertip_body_id', None)
            if tip_body is not None:
                data.xfrc_applied[tip_body, :3] = (
                    -model.body_mass[tip_body] * model.opt.gravity)
            mujoco.mj_step(model, data)
        # Refresh geometry for the next selection and the displayed markers.
        mujoco.mj_forward(model, data)
        unstable = (not np.isfinite(data.qpos).all() or
                    not np.isfinite(data.qvel).all() or
                    not np.isfinite(data.qacc).all() or
                    np.max(np.abs(data.qacc)) > 1e6)
        env._sync_viewer()
        dt = float(model.opt.timestep) * nstep
        wrench = np.r_[force, torque]
        return dict(h_exec=dt, nstep=nstep, force_body=wrench,
                    actual_applied_impulse_body=wrench*dt,
                    mujoco_unstable=bool(unstable),
                    max_abs_qacc=float(np.max(np.abs(data.qacc))))
    finally:
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
