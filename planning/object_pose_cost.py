"""Shared object objective and simulation-time flip/pose stage controller.

Configuration layout: [position, quaternion, upright weights, body axis(3),
reference-frame up(3), optional XY region radius, 1-cos(tilt limit),
upright barrier weight]. Nine-value legacy configurations are supported.
Quaternions use wxyz. Lambda's reference frame is
the observed object frame; MPC and measured rewards use the world frame.
"""
import numpy as np

OBJECT_COST_CONFIG_SIZE = 12


def extended_cost_config(config):
    c = np.asarray(config, dtype=float).reshape(-1)
    if c.size == 9:
        c = np.r_[c, np.zeros(3)]
    if c.size != OBJECT_COST_CONFIG_SIZE or not np.isfinite(c).all() or np.any(c[9:] < 0):
        raise ValueError('invalid object region/upright configuration')
    return c


def pose_only_config(pos_weight, ori_weight, body_axis=(0., 0., 1.),
                     reference_up=(0., 0., 1.)):
    """Direct pose objective ell_pose: position plus full quaternion error.

    The stable-pose objective ell_cso replaces the quaternion term with an
    upright / placement-mode penalty during the flip stage. This config does
    not.
    """
    return cost_config(pos_weight, ori_weight, 0.0,
                       body_axis=body_axis, reference_up=reference_up)


def cost_config(pos_weight, ori_weight, up_weight=0.0,
                body_axis=(0., 0., 1.), reference_up=(0., 0., 1.)):
    weights = np.asarray([pos_weight, ori_weight, up_weight], dtype=float)
    axes = [np.asarray(a, dtype=float).reshape(3) for a in (body_axis, reference_up)]
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('object cost weights must be finite and nonnegative')
    for a in axes:
        if not np.isfinite(a).all() or np.linalg.norm(a) < 1e-12:
            raise ValueError('object cost axes must be finite and nonzero')
    return np.concatenate([weights] + [a / np.linalg.norm(a) for a in axes])


def rotate_axis_np(q, axis):
    q = np.asarray(q, dtype=float).reshape(4)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('invalid object quaternion')
    q = q / np.linalg.norm(q)
    a = np.asarray(axis, dtype=float)
    return a + 2 * np.cross(q[1:], np.cross(q[1:], a) + q[0] * a)



def target_up_axis(target_quaternion):
    """Object-local direction that target pose maps to world +Z (wxyz)."""
    q = np.asarray(target_quaternion, dtype=float).reshape(4)
    axis = rotate_axis_np(q * np.array([1., -1., -1., -1.]), [0., 0., 1.])
    # Exact cardinal axes avoid injecting numerical noise into pure-yaw goals.
    axis[np.abs(axis) < 1e-12] = 0.
    return axis / np.linalg.norm(axis)


def object_cost_np(pose, target, config):
    pose, target = np.asarray(pose, dtype=float), np.asarray(target, dtype=float)
    c = np.asarray(config, dtype=float)
    if not np.isfinite(pose[:7]).all() or not np.isfinite(target[:7]).all():
        return float('inf')
    q, qd = pose[3:7], target[3:7]
    if min(np.linalg.norm(q), np.linalg.norm(qd)) < 1e-12:
        return float('inf')
    q, qd = q / np.linalg.norm(q), qd / np.linalg.norm(qd)
    up = .5 * (1 - np.clip(np.dot(rotate_axis_np(q, c[3:6]), c[6:9]), -1, 1))
    delta = pose[:3] - target[:3]
    position = np.sum(delta**2)
    barrier = 0.
    if c.size >= OBJECT_COST_CONFIG_SIZE:
        if c[9] > 0:
            planar = delta - c[6:9] * np.dot(delta, c[6:9])
            position = max(np.linalg.norm(planar)-c[9], 0.)**2
        barrier = c[11] * max(2*up-c[10], 0.)**2
    return float(c[0]*position + c[1]*(1-np.clip(np.dot(q, qd), -1, 1)**2)
                 + c[2]*up + barrier)


def object_cost_batch_np(poses, target, config):
    """Evaluate the same objective on candidate rows without Python calls."""
    poses = np.asarray(poses, dtype=float).reshape(-1, 7)
    target, c = np.asarray(target, dtype=float), np.asarray(config, dtype=float)
    qnorm = np.linalg.norm(poses[:, 3:7], axis=1)
    tnorm = np.linalg.norm(target[3:7])
    valid = np.isfinite(poses).all(axis=1) & (qnorm >= 1e-12)
    if not np.isfinite(target[:7]).all() or tnorm < 1e-12:
        return np.full(len(poses), np.inf)
    q = poses[:, 3:7] / np.maximum(qnorm[:, None], 1e-12)
    axis = c[3:6] + 2*np.cross(q[:, 1:], np.cross(q[:, 1:], c[3:6]) + q[:, :1]*c[3:6])
    up = .5*(1-np.clip(np.sum(axis*c[6:9], axis=1), -1, 1))
    delta = poses[:, :3]-target[:3]
    position, barrier = np.sum(delta**2, axis=1), 0.
    if c.size >= OBJECT_COST_CONFIG_SIZE:
        if c[9] > 0:
            planar = delta-np.sum(delta*c[6:9], axis=1, keepdims=True)*c[6:9]
            position = np.maximum(np.linalg.norm(planar, axis=1)-c[9], 0.)**2
        barrier = c[11]*np.maximum(2*up-c[10], 0.)**2
    cost = (c[0]*position + c[1]*(1-np.clip(np.sum(q*(target[3:7]/tnorm), axis=1), -1, 1)**2)
            + c[2]*up + barrier)
    return np.where(valid, cost, np.inf)


def object_cost_cs(pose, target, config):
    import casadi as cs
    q = pose[3:7] / cs.sqrt(cs.sumsqr(pose[3:7]) + 1e-12)
    a = config[3:6]
    axis = a + 2 * cs.cross(q[1:4], cs.cross(q[1:4], a) + q[0] * a)
    delta = pose[:3] - target[:3]
    position, barrier = cs.sumsqr(delta), 0
    if int(config.numel()) >= OBJECT_COST_CONFIG_SIZE:
        planar = delta-config[6:9]*cs.dot(delta, config[6:9])
        region = cs.fmax(cs.sqrt(cs.sumsqr(planar)+1e-20)-config[9], 0)**2
        position = cs.if_else(config[9] > 0, region, position)
        barrier = config[11]*cs.fmax(1-cs.dot(axis, config[6:9])-config[10], 0)**2
    return (config[0] * position
            # Preserve the legacy terminal-state expression and its RTI
            # Hessian. Dynamics constrain these quaternions to unit length;
            # normalizing again changes derivatives off that manifold.
            + config[1] * (1 - cs.dot(pose[3:7], target[3:7])**2)
            + config[2] * .5 * (1 - cs.dot(axis, config[6:9])) + barrier)


def object_cost_torch(position, quaternion, target, config):
    import torch
    c = torch.as_tensor(config, device=quaternion.device, dtype=quaternion.dtype)
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
    qd = target[3:7] / torch.linalg.vector_norm(target[3:7]).clamp_min(1e-12)
    a = c[3:6].expand_as(q[..., 1:])
    axis = a + 2 * torch.linalg.cross(q[..., 1:],
        torch.linalg.cross(q[..., 1:], a) + q[..., :1] * a)
    delta = position-target[:3]
    pos_cost, barrier = (delta**2).sum(dim=-1), 0.
    if c.numel() >= OBJECT_COST_CONFIG_SIZE:
        planar = delta-(delta*c[6:9]).sum(dim=-1, keepdim=True)*c[6:9]
        region = (torch.linalg.vector_norm(planar, dim=-1)-c[9]).clamp_min(0)**2
        pos_cost = torch.where(c[9] > 0, region, pos_cost)
        barrier = c[11]*(1-(axis*c[6:9]).sum(dim=-1)-c[10]).clamp_min(0)**2
    return (c[0] * pos_cost
            + c[1] * (1 - (q * qd).sum(dim=-1)**2)
            + c[2] * .5 * (1 - (axis * c[6:9]).sum(dim=-1)) + barrier)


def param_cost_config(param):
    config = getattr(param, 'object_cost_config_', None)
    return cost_config(param.pos_coef, param.ori_coef) if config is None else config


def measured_object_cost(param, pose):
    return object_cost_np(pose, np.r_[param.target_p_, param.target_q_], param_cost_config(param))


class ObjectCostStages:
    def __init__(self, args, body_axis=(0., 0., 1.)):
        self.enabled = bool(getattr(args, 'two_stage_object_cost', False))
        self.pos_weight, self.ori_weight = float(args.pos_coef), float(args.ori_coef)
        pose_up = getattr(args, 'pose_upright_coef', None)
        self.pose_up_weight = self.ori_weight if pose_up is None else float(pose_up)
        self.body_axis = cost_config(0, 0, body_axis=body_axis)[3:6]
        self.enter_angle = float(getattr(args, 'flip_enter_angle_deg', 10.))
        self.exit_angle = float(getattr(args, 'flip_exit_angle_deg', 25.))
        self.linear_limit = float(getattr(args, 'flip_linear_speed', .02))
        self.angular_limit = float(getattr(args, 'flip_angular_speed', .25))
        self.enter_hold = float(getattr(args, 'flip_stable_time', .3))
        self.exit_hold = float(getattr(args, 'flip_fallback_time', .1))
        # Keep the existing position objective for uncalibrated objects. The
        # bunny needs room to raise its COM while rolling about its long side;
        # applying that region to elephant changes its successful contact prior.
        region = getattr(args, 'flip_position_radius', None)
        if region is None:
            region = .08 if getattr(args, 'obj', '') == 'stanford_bunny2' else 0.
        self.flip_region = float(region)
        self.pose_tilt = float(getattr(args, 'pose_tilt_limit_deg', 10.))
        self.pose_barrier = float(getattr(args, 'pose_tilt_barrier_coef', 2000.))
        if (not np.isfinite([self.flip_region, self.pose_tilt, self.pose_barrier]).all()
                or self.flip_region < 0 or not 0 < self.pose_tilt < 90 or self.pose_barrier < 0):
            raise ValueError('invalid flip region or pose tilt limits')
        values = [self.enter_angle, self.exit_angle, self.linear_limit,
                  self.angular_limit, self.enter_hold, self.exit_hold]
        if not np.isfinite(values).all() or min(values) < 0 or not 0 <= self.enter_angle < self.exit_angle <= 180:
            raise ValueError('invalid flip thresholds; require 0 <= enter < exit <= 180')
        cost_config(self.pos_weight, self.ori_weight)
        cost_config(0, 0, self.pose_up_weight)
        if self.enabled and self.ori_weight <= 0:
            raise ValueError('two-stage object cost requires ori_coef > 0')
        self.stage = 'flip' if self.enabled else 'pose'
        self.last_time = self.start_time = self.since = None
        self.first_flip_time = None
        self.switches = self.fallbacks = 0
        self.angle = self.linear_speed = self.angular_speed = None

    def config(self, reference_rotation=None):
        weights = ((self.pos_weight, 0, self.ori_weight) if self.stage == 'flip'
                   else (self.pos_weight, self.ori_weight,
                         self.pose_up_weight if self.enabled else 0))
        up = np.array([0., 0., 1.])
        if reference_rotation is not None:
            up = np.asarray(reference_rotation).T @ up
        config = cost_config(*weights, body_axis=self.body_axis, reference_up=up)
        if not self.enabled:
            return config
        return np.r_[config, self.flip_region if self.stage == 'flip' else 0.,
                     1-np.cos(np.radians(self.pose_tilt)),
                     self.pose_barrier if self.stage == 'pose' else 0.]

    def progress_scale(self):
        return max(self.pos_weight * .02**2 + self.ori_weight * (1. if self.stage == 'flip' else .015)
                   + (self.pose_up_weight * .015 if self.enabled and self.stage == 'pose' else 0), 1e-12)

    def update(self, obs):
        """Return whether objective changed. Never count wall clock or duplicate frames."""
        try:
            t = float(obs['sim_time'])
            q = np.asarray(obs['policy_q'])[3:7]
            self.angle = float(np.degrees(np.arccos(np.clip(
                rotate_axis_np(q, self.body_axis)[2], -1, 1))))
            self.linear_speed = float(np.linalg.norm(np.asarray(obs['object_linear_velocity'], dtype=float).reshape(3)))
            self.angular_speed = float(np.linalg.norm(np.asarray(obs['object_angular_velocity'], dtype=float).reshape(3)))
            if not np.isfinite([t, self.angle, self.linear_speed, self.angular_speed]).all():
                raise ValueError('non-finite observation')
        except (KeyError, TypeError, ValueError):
            self.since = None
            self.angle = self.linear_speed = self.angular_speed = None
            return False
        if self.last_time is not None and t <= self.last_time:
            if t < self.last_time:
                self.since = None
            return False
        self.last_time = t
        if self.start_time is None:
            self.start_time = t
        # Also monitor stable flipping in legacy runs for a paired comparison.
        stable = (self.angle <= self.enter_angle and self.linear_speed <= self.linear_limit
                  and self.angular_speed <= self.angular_limit)
        condition = stable if self.stage == 'flip' or not self.enabled else self.angle > self.exit_angle
        if not condition:
            self.since = None
            return False
        if self.since is None:
            self.since = t
        hold = self.enter_hold if self.stage == 'flip' or not self.enabled else self.exit_hold
        if t - self.since < hold - 1e-9:
            return False
        if self.stage == 'flip' or not self.enabled:
            if self.first_flip_time is None:
                self.first_flip_time = t - self.start_time
            if not self.enabled:
                return False
            self.stage = 'pose'
        else:
            self.stage = 'flip'
            self.fallbacks += 1
        self.switches += 1
        self.since = None
        return True

    def diagnostics(self):
        config = self.config()
        return dict(object_cost_stage=self.stage,
                    object_cost_weights=config[:3].tolist(),
                    object_target_up_axis=self.body_axis.tolist(),
                    flip_position_radius_m=self.flip_region,
                    pose_tilt_limit_deg=self.pose_tilt,
                    pose_tilt_barrier_coef=self.pose_barrier,
                    object_tilt_deg=self.angle,
                    object_linear_speed=self.linear_speed, object_angular_speed=self.angular_speed,
                    first_stable_flip_time_s=self.first_flip_time,
                    object_cost_switches=self.switches, object_cost_fallbacks=self.fallbacks,
                    stable_flip_success=self.first_flip_time is not None)


def reset_objective_history(opt, mpc, trackers):
    """Discard objective-dependent evidence, retaining mesh/dynamics/SCM state."""
    for name in list(vars(opt)):
        if name.startswith(('last_candidate_', 'last_best_', 'last_global_', 'last_executed_',
                            'last_selected_', '_rank_score_', '_pending_selected_')):
            setattr(opt, name, None)
    for name in ('last_cost_lo', 'last_cost_hi', 'last_delta_lo', 'last_delta_hi',
                 'last_delta_center', 'last_delta_scale', 'last_pose_cost_now', '_dwell_steps',
                 '_pending_selected_count'):
        setattr(opt, name, 0)
    opt.last_topk_ids = np.zeros(0, dtype=np.int32)
    opt.last_topk_costs = np.zeros(0)
    opt.last_best_idx = opt.last_best_force = opt.last_selected_local = None
    opt._dwell_idx = opt._dwell_best_cost = opt._dwell_last_cost = None
    opt._dwell_blocked = opt._dwell_was_active = False
    opt._blocked_contact_indices = {}
    opt._contact_patch_failures = {}
    opt.lock_contact_patch = False
    opt.contact_switch_confidence = 1.0
    trackers['value_tracker'].reset_contact()
    trackers['value_tracker'].v_best = trackers['value_tracker'].v_arm = None
    trackers['value_tracker'].last_arm_idx = None
    trackers['model_cost_conf'].reset()
    trackers['model_cost_conf']._idx = None
    trackers.update(sol_guess=None, arrived_hold=False, arrived_dest_idx=None, last_verify_cost=None)
    if trackers.get('approach_via') is not None:
        trackers['approach_via'].reset()
    mpc._reset_acados_solver()
