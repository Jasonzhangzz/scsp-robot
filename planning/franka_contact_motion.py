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


class ForceAlignedApproachVia(FrankaApproachVia):
    """Prepare outside the patch, then enter along the optimized force.

    The cubic's final tangent is the Lambda force, not a world-axis rule.
    A latched arc advances from measured tip progress and ends at contact.
    """
    def __init__(self, floor_z=0., surface_normal=False, **kwargs):
        self.floor_z = float(floor_z)
        route = SurfaceNormalApproachVia if surface_normal else FrankaApproachVia
        self.transit = route(**dict(kwargs, **({'floor_z': floor_z} if surface_normal else {})))
        self.context = (np.zeros(3), np.zeros(3), False, False)
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.transit.reset()
        self.arc = self.arc_goal = self.arc_center = None
        self.arc_prepare_rotation = self.arc_curl_rotation = np.zeros(3)
        self.progress = 0.
        self.contact_rotation_world = np.zeros(3)
        self.contact_prepare_rotation = np.zeros(3)
        self.contact_rotation_steps = 10
        self.force_approach_active = False
        self.tool_rotation_world = np.zeros(3)

    def set_force_context(self, force_world, normal_outward, *, active, touching):
        self.force_world = np.asarray(force_world, dtype=float).copy()
        self.context = (np.asarray(force_world, dtype=float),
                        np.asarray(normal_outward, dtype=float), bool(active), bool(touching))

    @staticmethod
    def point(controls, u):
        a, b, c, d = np.asarray(controls)
        return (1-u)**3*a+3*(1-u)**2*u*b+3*(1-u)*u*u*c+u**3*d

    def update(self, tip, obj, best_surface, best_track, object_top_z,
               keepout, arrived, exec_press):
        tip, obj, press = [np.asarray(x, dtype=float).reshape(3) for x in (tip, obj, exec_press)]
        force, normal, active, touching = self.context
        self.force_approach_active = False
        limit = min(float(self.max_step or .005), float(self.max_lead or .005))
        finishing_hook = (active and touching and self.arc is not None and self.phase == 'hook'
            and not (self.progress >= 1. and np.linalg.norm(tip-self.arc[-1]) <= .004)
            and np.linalg.norm(obj-self.arc_center) <= .015)
        if finishing_hook:
            # A made touch must not cancel the upward part of the entry.
            # Continue this same bounded arc under the OSC contact force cap;
            # never redirect it to a newly ranked candidate while loaded.
            self.force_approach_active = True
            self.blocked = False
            self.tool_rotation_world = ((1-self.progress)*self.arc_prepare_rotation
                                       +self.progress*self.arc_curl_rotation)
            self.via = self._hook_step(tip, limit)
            return True, self.via.copy(), 'hook'
        if active and touching:
            # The cached best candidate can differ from the measured patch.
            # Keep the attitude prepared for this entry, rather than curling
            # toward another candidate's force after contact has been made.
            if self.arc is not None:
                self.contact_prepare_rotation = self.tool_rotation_world.copy()
                self.contact_rotation_world = self.arc_curl_rotation.copy()
                self.contact_rotation_steps = 0
            if self.contact_rotation_steps < 10:
                self.contact_rotation_steps += 1
                u = self.contact_rotation_steps/10.
                self.tool_rotation_world = ((1-u)*self.contact_prepare_rotation
                                            +u*self.contact_rotation_world)
            else:
                # The entry curl is a short follow-through. Holding a fixed
                # wrist attitude throughout the flip suppresses useful rolling.
                self.tool_rotation_world = np.zeros(3)
            self.arc = None
            self.blocked, self.phase = False, 'contact'
            self.via = tip+bounded_step(tip, press, min(self.max_step, self.max_lead))
            return False, self.via.copy(), self.phase
        size, nsize = np.linalg.norm(force), np.linalg.norm(normal)
        eligible = active and np.isfinite(np.r_[force, normal]).all() and size > 1e-3 and nsize > 1e-6
        if eligible:
            normal = normal/nsize
            inward = -float(np.dot(force, normal))
            tangent = force+inward*normal
            tsize = float(np.linalg.norm(tangent))
            eligible = (inward > 1e-4 and force[2] > .05*size and tangent[2] > .05*size
                        and np.linalg.norm(normal[:2]) > .25 and tsize > 1e-6)
        if not eligible:
            self.arc = None
            self.tool_rotation_world = np.zeros(3)
            if not active:
                self.contact_rotation_world = np.zeros(3)
            self.transit.max_step, self.transit.max_lead = self.max_step, self.max_lead
            use, self.via, self.phase = self.transit.update(
                tip, obj, best_surface, best_track, object_top_z, keepout, arrived, press)
            self.blocked = self.transit.blocked
            return use, self.via.copy(), self.phase
        axis = np.cross([0., 0., 1.], normal)
        axis /= np.linalg.norm(axis)
        # Scale the preparation to actual upward loading. A steep patch
        # normal can imply a large friction angle even for a nearly planar
        # force, which must not receive a full-depth undercut.
        elevation = float(np.arctan2(force[2], np.linalg.norm(force[:2])))
        angle = elevation
        prep = min(.5*angle, np.deg2rad(15.))
        curl = min(angle, np.deg2rad(20.))
        invalid = self.arc_center is not None and np.linalg.norm(obj-self.arc_center) > .015
        changed = self.arc_goal is not None and np.linalg.norm(press-self.arc_goal) > .012
        if self.arc is None or invalid or changed:
            # Floor clearance limits the amount of undercut, never the final
            # force tangent. An infeasible upward entry uses the normal route.
            room = float(press[2]-self.floor_z)
            direction = force/size
            lead = min(.016, .8*room/max(direction[2], 1e-9))
            depth = min(.012, .025*direction[2], .7*room/max(tangent[2]/tsize, 1e-9))
            if lead < .0015 or depth < .001:
                self.arc = None
                self.tool_rotation_world = np.zeros(3)
                self.transit.max_step, self.transit.max_lead = self.max_step, self.max_lead
                use, self.via, self.phase = self.transit.update(
                    tip, obj, best_surface, best_track, object_top_z, keepout, arrived, press)
                self.blocked = self.transit.blocked
                return use, self.via.copy(), self.phase
            start = press+.025*normal-depth*tangent/tsize
            start[2] = max(self.floor_z, start[2])
            second = start+.005*normal
            second[2] = max(self.floor_z, second[2])
            self.arc = np.array([start, second, press-lead*direction, press])
            self.arc_prepare_rotation = -prep*axis
            self.arc_curl_rotation = curl*np.cross(tangent/tsize, normal)
            self.arc_goal, self.arc_center = press.copy(), obj.copy()
            self.progress = 0.
            self.phase = 'prepare'
            self.transit.reset()
        self.force_approach_active = True
        self.tool_rotation_world = ((1-self.progress)*self.arc_prepare_rotation
                                    +self.progress*self.arc_curl_rotation)
        self.blocked = False
        limit = min(float(self.max_step or .005), float(self.max_lead or .005))
        if self.phase != 'hook' and np.linalg.norm(tip-self.arc[0]) > .004:
            self.transit.max_step, self.transit.max_lead = self.max_step, self.max_lead
            _, self.via, _ = self.transit.update(tip, obj, best_surface, self.arc[0],
                object_top_z, keepout, False, self.arc[0])
            self.blocked = self.transit.blocked
            self.phase = 'prepare'
            return True, self.via.copy(), self.phase
        self.phase = 'hook'
        self.via = self._hook_step(tip, limit)
        return True, self.via.copy(), self.phase

    def _hook_step(self, tip, limit):
        current = self.point(self.arc, self.progress)
        if np.linalg.norm(tip-current) <= .004:
            u = self.progress
            derivative = (3*(1-u)**2*(self.arc[1]-self.arc[0])
                +6*(1-u)*u*(self.arc[2]-self.arc[1])+3*u*u*(self.arc[3]-self.arc[2]))
            self.progress = min(1., u+min(.15, .7*limit/max(np.linalg.norm(derivative), .001)))
        return tip+bounded_step(tip, self.point(self.arc, self.progress), limit)


class OSCContactCurl:
    """Outward wrist preparation and one bounded curl per made contact.

    This helper returns only a rotation reference. XYZ commands are owned by
    MPC, including deciding when to load or withdraw from the patch.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.loaded = self.completed = self.prepared = False
        self.angle = self.curl_time = 0.
        self.velocity_reference = np.zeros(3)
        self.rotation = np.zeros(3)
        self.preparation = self.axis = np.zeros(3)

    def update(self, force, normal, action, *, touching, gap, active, dt, base_tool_axis=None, actual_tool_axis=None, tool_angular_velocity=None):
        self.velocity_reference = np.zeros(3)
        if not active:
            self.reset()
            return np.zeros(3)
        force,normal,action = [np.asarray(x,dtype=float).reshape(3) for x in (force,normal,action)]
        if not np.isfinite(np.r_[force,normal,action]).all() or np.linalg.norm(normal)<1e-8:
            return np.zeros(3)
        normal = normal/np.linalg.norm(normal)
        # Do not restart the preparation on tiny positive PhysX margin gaps.
        made = bool(touching or (self.loaded and np.isfinite(gap) and gap <= .002))
        upward = force[2] > .05*max(np.linalg.norm(force),1e-6)
        tangent = force-(force@normal)*normal
        tangent_up = upward and tangent[2] > 1e-5 and np.linalg.norm(normal[:2])>.25
        if not made:
            self.loaded = self.completed = self.prepared = False
            self.angle = self.curl_time = 0.
            axis = np.cross([0.,0.,1.],normal)
            if not tangent_up or np.linalg.norm(axis)<1e-8:
                self.rotation = np.zeros(3)
                return self.rotation.copy()
            axis /= np.linalg.norm(axis)
            elevation = np.arctan2(force[2],np.linalg.norm(force[:2]))
            base = np.array([0.,0.,-1.]) if base_tool_axis is None else np.asarray(base_tool_axis,dtype=float)
            horizontal_normal = normal.copy(); horizontal_normal[2] = 0.
            horizontal_normal /= np.linalg.norm(horizontal_normal)
            placed_tilt = np.arctan2(base@horizontal_normal,-base[2])
            desired_tilt = min(.5*elevation,np.deg2rad(10.))
            # Preserve an already useful outward placement instead of making
            # the successful natural hook first reorient to a different pose.
            deficit = max(0.,desired_tilt-placed_tilt)
            self.rotation = -deficit*axis
            return self.rotation.copy()
        if self.completed:
            return np.zeros(3)
        if not self.loaded:
            self.loaded = True
            self.preparation = self.rotation.copy()
            if tangent_up:
                self.axis = np.cross([0.,0.,1.],normal)
                self.axis /= np.linalg.norm(self.axis)
            else:
                self.axis = np.zeros(3)
        # Wait only in the angular task. Never pause or replace MPC's XYZ
        # command while a previously inward-facing wrist prepares outward.
        if not tangent_up or -float(action@normal) <= 1e-5:
            return np.zeros(3)
        if not self.prepared and actual_tool_axis is not None and np.linalg.norm(self.preparation)>1e-8:
            from scipy.spatial.transform import Rotation
            base = np.array([0.,0.,-1.]) if base_tool_axis is None else np.asarray(base_tool_axis)
            desired_axis = Rotation.from_rotvec(self.preparation).apply(base)
            error = np.arccos(np.clip(np.dot(desired_axis,actual_tool_axis),-1.,1.))
            if error > np.deg2rad(5.):
                return self.preparation.copy()
        self.prepared = True
        # A planned withdrawal or no-load command must not add a rotary push.
        if -float(action@normal) <= 1e-5:
            return np.zeros(3)
        if tangent_up:
            if np.linalg.norm(self.axis)<1e-8:
                self.axis = np.cross([0.,0.,1.],normal)
                self.axis /= np.linalg.norm(self.axis)
            speed = min(1.,max(0.,float(force[2])/max(np.linalg.norm(force),1e-9)))
            measured_speed = speed if tool_angular_velocity is None else max(0.,float(np.asarray(tool_angular_velocity)@self.axis))
            self.angle = min(np.deg2rad(30.),self.angle+max(float(dt),0.)*measured_speed)
            self.curl_time += max(float(dt),0.)
            self.rotation = self.preparation+self.angle*self.axis
            if self.angle >= np.deg2rad(30.) or self.curl_time >= 1.5:
                self.completed = True
                return np.zeros(3)
            self.velocity_reference = speed*self.axis
        return self.rotation.copy()
