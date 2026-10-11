"""Bounded Cartesian approach and post-flip pressure release for Franka MPC."""
import numpy as np

from planning.object_pose_cost import rotate_axis_np


def bounded_step(tip, goal, limit):
    delta = np.asarray(goal, dtype=float) - np.asarray(tip, dtype=float)
    return delta * min(1., float(limit) / max(np.linalg.norm(delta), 1e-12))


class FrankaApproachVia:
    """Lift, cross above the mesh, then descend; never select an XY orbit.

    Latch the destination during transit so changing candidate rankings cannot
    reverse a half-completed crossing. Clearance is based on the mesh top,
    rather than on the moving fingertip height.
    """
    def __init__(self, rate=.1, max_step=.005, max_lead=.005):
        self.rate = float(rate)
        self.max_step, self.max_lead = max_step, max_lead
        self.reset()

    def reset(self):
        self.via = self.goal = self.center = None
        self.height = None
        self.phase = 'drop'
        self.blocked = False

    def update(self, tip, obj, best_surface, best_track, object_top_z,
               keepout, arrived, exec_press):
        from examples.mpc.fingertips.test.test_0902 import _press_path_blocked
        tip, obj, press = [np.asarray(x, dtype=float).reshape(3)
                           for x in (tip, obj, exec_press)]
        # Release the latch after arrival, or when object motion invalidates it.
        invalid = self.center is not None and np.linalg.norm(obj - self.center) > .015
        done = self.goal is not None and np.linalg.norm(tip - self.goal) <= .012
        if self.goal is None or invalid or done:
            self.goal, self.center = press.copy(), obj.copy()
            self.height = max(float(object_top_z) + .025, float(press[2]) + .015)
            self.phase = 'lift' if _press_path_blocked(tip, obj, press, keepout) else 'drop'
        if self.phase == 'lift' and tip[2] >= self.height - .004:
            self.phase = 'cross'
        if self.phase == 'cross' and np.linalg.norm(tip[:2] - self.goal[:2]) <= .008:
            self.phase = 'drop'
        if self.phase == 'lift':
            desired = np.r_[tip[:2], self.height]
        elif self.phase == 'cross':
            desired = np.r_[self.goal[:2], self.height]
        else:
            desired = self.goal.copy()
        self.blocked = self.phase in ('lift', 'cross')
        limit = min(float(self.max_step or .005), float(self.max_lead or .005))
        # Do not blend across corners: blending lift/cross/drop can cut through
        # the mesh and can retain a horizontal orbit after the target changes.
        self.via = tip + bounded_step(tip, desired, limit)
        return bool(self.blocked or np.linalg.norm(tip - self.goal) > .018), self.via.copy(), self.phase


class UprightRelease:
    """Withdraw pressure while the flipped body settles, using a fixed goal."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.goal = None
        self.until = None
        self.last_stage = 'flip'
        self.last_time = self.last_angle = None
        self.releases = self.rejected_tilts = 0

    def update(self, stages, obs, policy):
        if stages is None or not stages.enabled:
            return None
        try:
            tip = np.asarray(obs['policy_q'], dtype=float)[7:10]
            t = float(obs['sim_time'])
            if tip.size != 3 or not np.isfinite(np.r_[tip, t]).all():
                raise ValueError('invalid motion observation')
        except (KeyError, TypeError, ValueError):
            self.goal = self.until = self.last_time = self.last_angle = None
            return None
        if self.last_stage == 'flip' and stages.stage == 'pose':
            self.until = t + stages.enter_hold
        self.last_stage = stages.stage
        tilt_rate = 0.
        if self.last_time is not None and t > self.last_time and stages.angle is not None:
            tilt_rate = (stages.angle - self.last_angle) / (t - self.last_time)
        if stages.angle is not None:
            self.last_time, self.last_angle = t, stages.angle
        tipping = (stages.stage == 'pose' and stages.angle is not None and
                   ((stages.angle > stages.enter_angle and tilt_rate > 1.) or
                    (stages.angle > 2. and tilt_rate > 15.)))
        if tipping:
            self.until = t + stages.enter_hold
        settle = (stages.stage == 'flip' and stages.angle is not None
                  and stages.angle <= stages.enter_angle)
        if stages.stage == 'flip' and stages.angle is not None and stages.angle > stages.enter_angle + 3:
            self.goal = self.until = None
        active = settle or (self.until is not None and t < self.until)
        if not active:
            self.goal = None
            return None
        if self.goal is None:
            surface = np.asarray(policy['p_arm_surface_world'], dtype=float)
            track = np.asarray(policy['p_arm_track_world'], dtype=float)
            outward = np.asarray(policy.get('withdrawal_axis', track - surface), dtype=float).copy()
            if np.linalg.norm(outward) < 1e-6:
                outward = tip - np.asarray(obs['policy_q'])[:3]
            outward /= max(np.linalg.norm(outward), 1e-9)
            self.goal = tip + .020 * outward
            # Sideways withdrawal keeps a contacted, tall object from being
            # lifted with the fingertip. Never command below the current tip.
            self.goal[2] = max(self.goal[2], tip[2])
            self.releases += 1
        return self.goal.copy()

    def unsafe_prediction(self, stages, rollout):
        if stages is None or not stages.enabled or stages.stage != 'pose' or stages.angle is None:
            return False
        poses = np.asarray(rollout, dtype=float)
        if poses.size == 0:
            return False
        poses = poses.reshape(-1, poses.shape[-1])
        try:
            angles = [np.degrees(np.arccos(np.clip(rotate_axis_np(x[3:7], stages.body_axis)[2], -1, 1))) for x in poses]
        except ValueError:
            self.rejected_tilts += 1
            return True
        # Do not trade a stable standing posture for position/yaw progress.
        unsafe = max(angles) > min(stages.exit_angle, max(stages.enter_angle, stages.angle + 2.))
        self.rejected_tilts += int(unsafe)
        return unsafe


class UprightExecutionGuard:
    """PhysX-frame withdrawal, independent of asynchronous planner latency."""
    def __init__(self, body_axis, enter_angle=10., hold=.3):
        self.body_axis = np.asarray(body_axis, dtype=float)
        self.enter_angle, self.hold = float(enter_angle), float(hold)
        self.until = self.goal = None
        self.releases = 0

    def update(self, pose, omega_world, tip, normal_out, in_contact, sim_time, pose_stage):
        up = rotate_axis_np(np.asarray(pose)[3:7], self.body_axis)
        cosine = np.clip(up[2], -1., 1.)
        angle = float(np.degrees(np.arccos(cosine)))
        rate = float(np.degrees(-np.cross(omega_world, up)[2]
                                / max(np.sqrt(max(0., 1. - cosine*cosine)), 1e-6)))
        # A stable intermediate tilt can still be corrected by MPC. Release
        # on growing tilt, rather than locking forever at a stationary angle.
        tipping = pose_stage and ((angle > 2. and rate > 15.) or
                                  (angle > self.enter_angle and rate > 1.))
        settling = not pose_stage and angle <= self.enter_angle and in_contact
        if tipping or settling:
            if self.until is None or sim_time >= self.until:
                n = np.asarray(normal_out, dtype=float).reshape(3) if normal_out is not None else (
                    np.asarray(tip) - np.asarray(pose)[:3])
                n = n / max(np.linalg.norm(n), 1e-9)
                self.goal = np.asarray(tip, dtype=float) + .020*n
                self.goal[2] = max(self.goal[2], np.asarray(tip)[2])
                self.releases += 1
            self.until = float(sim_time) + self.hold
        if self.until is not None and sim_time < self.until:
            return self.goal.copy()
        self.until = self.goal = None
        return None


def outward_release_force(force, normal_out):
    """While touching, brake along the actual outward normal without pressing."""
    if normal_out is None:
        return np.zeros(3)
    n = np.asarray(normal_out, dtype=float).reshape(3)
    n /= max(np.linalg.norm(n), 1e-9)
    return max(0., float(np.dot(force, n))) * n
