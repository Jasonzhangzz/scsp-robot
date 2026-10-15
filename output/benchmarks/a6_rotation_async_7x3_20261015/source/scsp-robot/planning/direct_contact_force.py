"""Physical execution of a lambda contact force at a body-fixed surface point.

MuJoCo free joints store translation velocity in world coordinates and
angular velocity in body coordinates. Lambda uses body coordinates for both.
The ideal actuator has no colliding fingertip; only the object and its table
contacts generate motion. No object pose or velocity is overwritten.
"""
import numpy as np
import mujoco


def add_direct_force_args(parser):
    from examples.mpc.franka.ik2.params import add_lambda_predictor_args
    add_lambda_predictor_args(parser)
    parser.add_argument('--execution-force-cap', type=float, default=0.0,
                        help='Optional applied force cap in N; 0 executes the full optimized force.')
    parser.add_argument('--execution-torque-cap', type=float, default=0.0,
                        help='Optional N*m bound; scales the entire point force, preserving its moment arm.')
    parser.add_argument('--success-hold-steps', type=int, default=10,
                        help='Require this many consecutive successful control intervals, then a settled object.')
    parser.add_argument('--force-apply-duration', type=float, default=0.0005,
                        help='Apply the full optimized force for this many seconds per policy interval.')
    parser.add_argument('--force-physics-substeps', type=int, default=40,
                        help='MuJoCo physics substeps per shared 20 ms prediction interval.')
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


def environment_contacts_body(env, friction=None):
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
        mu = float(con.friction[0]) if friction is None else float(friction)
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
    """Reuse the Isaac parameter object and factory, without physical retuning."""
    from examples.mpc.franka.ik2.params import ExplicitMPCParams, build_lambda_optimizer

    model = env.model_
    tip = model.geom('fingertip0').id
    model.geom_contype[tip] = 0
    model.geom_conaffinity[tip] = 0
    env.data_.ctrl[:] = 0.0
    mujoco.mj_forward(model, env.data_)
    env.direct_force_ = True
    force_cap = float(args.execution_force_cap)
    if not np.isfinite(force_cap) or force_cap < 0:
        raise ValueError('execution_force_cap must be finite and nonnegative')
    param.obj_mass_ = float(model.body_mass[model.body('obj').id])
    param.lambda_source_param_ = ExplicitMPCParams(
        args, rand_seed=int(getattr(param, 'trial_seed_', 0)), build_optimizer=False)
    param.lambda_optimizer = build_lambda_optimizer(param.lambda_source_param_, args)
    if (param.lambda_optimizer.contact_model != 'soft-scm' or
            param.lambda_optimizer.robot_stiffness is None):
        raise ValueError('Direct SCM execution requires two-stage coupled soft-SCM')
    from planning.object_pose_cost import ObjectCostStages, target_up_axis
    param.force_cost_stages_ = ObjectCostStages(args, target_up_axis(param.target_q_))
    # PhysX averages the object/table materials. MuJoCo uses their maximum;
    # assign the effective pair value to both to realize the same friction.
    for name in ('obj', 'table'):
        geom = model.geom(name).id
        model.geom_friction[geom, 0] = param.lambda_source_param_.mu_table_object_
        # The shared SCM represents normal/sliding reactions only. Avoid
        # adding unmodeled MuJoCo torsional/rolling constraints.
        model.geom_friction[geom, 1:] = 0.
        model.geom_condim[geom] = 3
    # Match the reference Isaac URDF, whose inertial origin and axes are
    # explicitly zero/identity and whose diagonal comes from _box_inertia_diag.
    # MuJoCo's automatic mesh COM/axes otherwise change the contact moment arm.
    body = model.body('obj').id
    source = param.lambda_source_param_
    substeps = int(args.force_physics_substeps)
    if substeps < 1:
        raise ValueError('force_physics_substeps must be positive')
    model.opt.timestep = param.lambda_optimizer.h / substeps
    param.frame_skip_ = substeps
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    saved = {name: getattr(env.data_, name).copy() for name in
             ('qpos', 'qvel', 'ctrl', 'xfrc_applied', 'qfrc_applied')}
    model.body_ipos[body] = 0.
    model.body_iquat[body] = [1., 0., 0., 0.]
    model.body_inertia[body] = (source.sim_obj_inertia_diag_ *
                              model.body_mass[body] / source.sim_obj_mass_)
    model.opt.gravity[:] = source.gravity_[:3]
    mujoco.mj_setConst(model, env.data_)
    for name, value in saved.items():
        getattr(env.data_, name)[:] = value
    mujoco.mj_forward(model, env.data_)
    return param.lambda_optimizer


def step_point_force(env, point_body, force_body, nstep=None, force_at_substep=None):
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
    impulse = np.zeros(6)
    max_force = 0.0
    try:
        for _ in range(nstep):
            mujoco.mj_forward(model, data)
            if force_at_substep is not None:
                force = np.asarray(force_at_substep(), dtype=float).reshape(3)
                if not np.isfinite(force).all():
                    raise ValueError('Substep force must be finite')
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
            torque = np.cross(point - model.body_ipos[body], force)
            impulse += float(model.opt.timestep) * np.r_[force, torque]
            max_force = max(max_force, float(np.linalg.norm(force)))
        # Refresh geometry for the next selection and the displayed markers.
        mujoco.mj_forward(model, data)
        unstable = (not np.isfinite(data.qpos).all() or
                    not np.isfinite(data.qvel).all() or
                    not np.isfinite(data.qacc).all() or
                    np.max(np.abs(data.qacc)) > 1e6)
        env._sync_viewer()
        dt = float(model.opt.timestep) * nstep
        wrench = impulse / dt
        return dict(h_exec=dt, nstep=nstep, force_body=wrench,
                    actual_applied_impulse_body=impulse,
                    max_substep_force=max_force,
                    mujoco_unstable=bool(unstable),
                    max_abs_qacc=float(np.max(np.abs(data.qacc))))
    finally:
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0


def step_actuator_contact_force(env, solved):
    """Transmit the optimized actuator force through an ideal point actuator.

    The reference predictor is quasi-dynamic, whereas MuJoCo retains physical
    inertia. Use a short force pulse followed by unforced physics within the
    same policy interval. The optimized force magnitude/direction is retained;
    only its activation duration changes. No simulator oracle reranks actions.
    """
    opt = env.param_.lambda_optimizer
    frame = np.column_stack((solved['n_arm'], solved['t1'], solved['t2']))
    command = np.asarray(solved['lam'], dtype=float).reshape(3)
    force = frame @ command
    point = np.asarray(solved['p_local'], dtype=float).reshape(3)
    requested = float(env.force_args_.force_apply_duration)
    if not np.isfinite(requested) or requested < 0 or requested > opt.h:
        raise ValueError('force_apply_duration must be in [0, prediction interval]')
    active = min(env.param_.frame_skip_, int(round(requested/env.model_.opt.timestep)))
    if requested > 0 and active < 1:
        raise ValueError('force_apply_duration must be at least one physics substep')
    if solved.get('settling', False):
        active = 0
    cap = float(solved.get('execution_force_cap', 0.))
    torque_cap = float(solved.get('execution_torque_cap', 0.))
    com = env.model_.body_ipos[env.model_.body('obj').id]
    raw = np.r_[force, np.cross(point-com, force)]
    scale = 1.0
    if cap > 0 and np.linalg.norm(force) > cap:
        scale = min(scale, cap/np.linalg.norm(force))
    if torque_cap > 0 and np.linalg.norm(raw[3:]) > torque_cap:
        scale = min(scale, torque_cap/np.linalg.norm(raw[3:]))
    count = 0

    def substep_force():
        nonlocal count
        count += 1
        return scale*force if count <= active else np.zeros(3)

    info = step_point_force(env, point, np.zeros(3), force_at_substep=substep_force)
    applied_dt = active*float(env.model_.opt.timestep)
    return dict(info, execution='wrench', actuator_force_contact=command,
                applied_force_body=scale*raw if active else np.zeros(6), active_substeps=active,
                requested_force_duration=requested, applied_force_duration=applied_dt,
                force_body_unclipped=raw*applied_dt/info['h_exec'],
                wrench_impulse_body=info['actual_applied_impulse_body'].copy(),
                wrench_impulse_raw_body=raw*applied_dt,
                force_cap=cap, torque_cap=torque_cap,
                force_was_clipped=bool(scale < 1.-1e-6), force_scale=scale,
                force_units='N', impulse_units='N*s')
