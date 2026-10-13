"""Bounded Cartesian approach and explicit withdrawal helpers for Franka MPC."""
import numpy as np



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


class SurfaceNormalApproachVia(FrankaApproachVia):
    """Enter a concave object's patch from its outward side after transit.

    Descending straight to a recessed patch can hit a duck's head/back first
    and slide the body instead of reaching the ranked contact. Transit ends
    at an outward standoff; the last leg approaches along the patch normal.
    """
    def __init__(self, standoff=.035, floor_z=0., **kwargs):
        self.standoff, self.floor_z = float(standoff), float(floor_z)
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.press_goal = self.entry = self.anchor = None
        self.normal_approach = False

    def update(self, tip, obj, best_surface, best_track, object_top_z,
               keepout, arrived, exec_press):
        tip, obj, surface, track, press = [np.asarray(x, dtype=float).reshape(3)
            for x in (tip, obj, best_surface, best_track, exec_press)]
        limit = min(float(self.max_step or .005), float(self.max_lead or .005))
        outward = track-surface
        length = np.linalg.norm(outward)
        if length < 1e-9:
            return super().update(tip, obj, surface, track, object_top_z, keepout, arrived, press)
        # A reached contact follows the moving patch directly; restarting the
        # transit here would withdraw the support during an ongoing flip.
        if np.linalg.norm(tip-track) < .018:
            self.reset()
            self.via = tip+bounded_step(tip, press, limit)
            return False, self.via.copy(), 'drop'
        moved = self.anchor is not None and np.linalg.norm(obj-self.anchor) > .015
        completed = self.press_goal is not None and np.linalg.norm(tip-self.press_goal) <= .012
        changed = self.press_goal is not None and np.linalg.norm(press-self.press_goal) > .012
        # Release a completed latch when ranking or the flip/pose stage picks
        # a new destination; otherwise the old waypoint holds forever.
        if self.press_goal is None or moved or (completed and changed):
            self.reset()
            self.press_goal, self.anchor = press.copy(), obj.copy()
            self.entry = self.press_goal+self.standoff*outward/length
            self.entry[2] = max(self.floor_z, self.entry[2])
        if np.linalg.norm(tip-self.entry) < .01:
            self.normal_approach = True
        if self.normal_approach:
            self.blocked = np.linalg.norm(tip-self.press_goal) > .018
            self.phase = 'normal'
            self.via = tip+bounded_step(tip, self.press_goal, limit)
            return self.blocked, self.via.copy(), self.phase
        return super().update(tip, obj, surface, self.entry, object_top_z,
                              keepout, False, self.entry)


def outward_release_force(force, normal_out):
    """While touching, brake along the actual outward normal without pressing."""
    if normal_out is None:
        return np.zeros(3)
    n = np.asarray(normal_out, dtype=float).reshape(3)
    n /= max(np.linalg.norm(n), 1e-9)
    return max(0., float(np.dot(force, n))) * n
