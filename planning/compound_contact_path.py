"""Sphere-clear approach paths through a union of attached convex solids."""
import time

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from planning.franka_contact_motion import FrankaApproachVia, bounded_step


class CompoundApproachVia(FrankaApproachVia):
    """Route around collision solids, including through an object's opening.

    Paths live in the object frame and follow its measured pose. The policy
    latches ranking during travel; pose-mode accepted execution targets can
    replace that route. Flip mode passes retarget=False to keep its transit.
    Collision checks use
    triangle distances to the attached convex parts, so a short chord
    through a thin wall cannot count as arrival. No fixed normal standoff
    is used in a cavity.
    """
    def __init__(self, parts, floor_z=0., resolution=.004, tool_aware=False, table_z=0., tool_rotation=None, **kwargs):
        self.parts = parts
        self.floor_z = float(floor_z)
        self.resolution = float(resolution)
        self.planes = []
        for part in parts:
            n = np.asarray(part.face_normals)
            offsets = np.sum(n * part.triangles_center, axis=1)
            planes = np.unique(np.round(np.column_stack([n, offsets]), 8), axis=0)
            self.planes.append((planes[:, :3], planes[:, 3]))
        vertices = np.concatenate([p.vertices for p in parts])
        self.bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
        self.vertices = vertices
        self.tool_aware = bool(tool_aware)
        self.table_z = float(table_z)
        self.extra_planes = []
        self.extra_margin = .001
        if self.tool_aware:
            from models.compound_tool_geometry import cavity_configuration_parts
            obstacles,self.tool_vertices = cavity_configuration_parts(parts,tool_rotation)
            for solid in obstacles:
                n = np.asarray(solid.face_normals)
                offsets = np.sum(n*solid.triangles_center,axis=1)
                planes = np.unique(np.round(np.column_stack([n,offsets]),8),axis=0)
                self.extra_planes.append((np.asarray(solid.bounds),planes[:,:3],planes[:,3]))
            all_vertices = np.concatenate([vertices]+[o.vertices for o in obstacles])
            self.bounds = np.array([all_vertices.min(axis=0),all_vertices.max(axis=0)])
        self.effective_floor_z = self.floor_z
        self._lattice = None
        self._settled_frames = 0
        self.last_plan_seconds = 0.
        self.grid_builds = 0
        self.position, self.rotation = np.zeros(3), np.eye(3)
        self.radius = .0095
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self.route = None
        self.destination_local = None
        self.destination_idx = None
        self.route_failed = False
        self._contact_track = self._contact_surface = None
        self._missed_contact_frames = 0

    def remember_contact(self, surface, track):
        self._contact_surface = self.rotation.T @ (np.asarray(surface)-self.position)
        self._contact_track = self.rotation.T @ (np.asarray(track)-self.position)
        self._missed_contact_frames = 0

    def reacquire_contact(self, tip):
        """Try the same patch briefly after contact flicker during a flip."""
        self._missed_contact_frames += 1
        if self._contact_track is None or self._missed_contact_frames > 10:
            return None
        track = self.position + self.rotation @ self._contact_track
        if np.linalg.norm(np.asarray(tip)-track) > .015:
            return None
        surface = self.position + self.rotation @ self._contact_surface
        return surface, track

    def set_geometry_context(self, position, rotation, radius):
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        self.radius = float(radius)
        self.effective_floor_z = self.floor_z
        if self.tool_aware:
            bottom = float(np.min(self.tool_vertices @ self.rotation[2]))
            self.effective_floor_z = max(self.floor_z, self.table_z-bottom+.001)

    def protect_flip_contact(self, velocity, table_z):
        """Release a made contact only after six supported, quiet observations.

        A low reward confidence does not make it safe to unload a tipping
        or airborne object. This guard applies after approach has ended;
        incidental contact along a route does not latch the wrong wall.
        """
        velocity = np.asarray(velocity, dtype=float).reshape(6)
        self.support_gap = float(np.min(np.einsum('ij,j->i', self.vertices, self.rotation[2])
            + self.position[2]) - table_z)
        settled = (self.support_gap <= .003
            and np.linalg.norm(velocity[:3]) < .02
            and np.linalg.norm(velocity[3:]) < .25)
        self._settled_frames = self._settled_frames + 1 if settled else 0
        return self._settled_frames < 6

    def hold_contact(self, tip, press):
        self.route = self.destination_local = None
        self.destination_idx = None
        self.route_failed = self.blocked = False
        self.phase = 'hold'
        self.last_plan_seconds = 0.
        limit = min(float(self.max_step or .005), float(self.max_lead or .005))
        self.via = np.asarray(tip) + bounded_step(tip, press, limit)
        return False, self.via.copy(), self.phase

    def prepare_lattice(self):
        """Cache body-frame obstacle clearance; only the floor moves in it."""
        step = self.resolution
        key = (step, self.radius)
        if self._lattice is None or self._lattice[0] != key:
            axes = [np.arange(a-.025, b+.025+step, step)
                    for a,b in zip(*self.bounds)]
            shape = tuple(len(a) for a in axes)
            points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
            radius = self.radius
            try:
                self.radius += .5 * step + .0001
                self.extra_margin += .5 * step + .0001
                free = self.free(points, floor=False)
            finally:
                self.radius = radius
                self.extra_margin = .001
            self._lattice = (key, points, shape, free)
            self.grid_builds += 1
        _, points, shape, free = self._lattice
        return points, shape, free.copy()

    def free(self, points, floor=True):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        keep = np.ones(len(points), dtype=bool)
        if floor:
            keep &= np.einsum('ij,j->i', points, self.rotation[2]) + self.position[2] >= self.effective_floor_z - 1e-6
        for part, (normals, offsets) in zip(self.parts, self.planes):
            bounds = np.asarray(part.bounds)
            check = np.flatnonzero(keep & np.all(points >= bounds[0]-self.radius,axis=1)
                & np.all(points <= bounds[1]+self.radius,axis=1))
            if not len(check):
                continue
            separation = np.max(np.einsum('ij,kj->ik', points[check], normals) - offsets, axis=1)
            keep[check] &= separation >= -1e-7
            # Shifted planes overestimate the rounded sphere offset at edges
            # (especially the narrow sector caps). Resolve only that band
            # with actual triangle distance, preserving valid cavity targets.
            near = check[keep[check] & (separation < self.radius - 1e-6)]
            if len(near):
                _, distance, _ = trimesh.proximity.closest_point(part, points[near])
                keep[near] &= distance >= self.radius - 1e-6
        for bounds,normals,offsets in self.extra_planes:
            check = np.flatnonzero(keep & np.all(points >= bounds[0]-self.extra_margin,axis=1)
                & np.all(points <= bounds[1]+self.extra_margin,axis=1))
            if len(check):
                keep[check] &= np.max(np.einsum('ij,kj->ik',points[check],normals)-offsets,axis=1) >= self.extra_margin
        return keep

    def segment_clear(self, start, goal):
        """Broad-phase convex sweep followed by 1 mm sphere clearance probes."""
        start, goal = np.asarray(start), np.asarray(goal)
        delta = goal - start
        if self.tool_aware:
            count = max(2,int(np.ceil(np.linalg.norm(delta)/.001))+1)
            return bool(np.all(self.free(np.linspace(start,goal,count))))
        for normals, offsets in self.planes:
            value = normals @ start - offsets - self.radius + 1e-6
            speed = normals @ delta
            parallel = np.abs(speed) < 1e-12
            if np.any(parallel & (value >= 0.)):
                continue
            lower, upper = 0., 1.
            negative, positive = speed < -1e-12, speed > 1e-12
            if np.any(negative):
                lower = max(lower, float(np.max(-value[negative] / speed[negative])))
            if np.any(positive):
                upper = min(upper, float(np.min(-value[positive] / speed[positive])))
            if lower < upper - 1e-8:
                count = max(2, int(np.ceil(np.linalg.norm(delta) / .001)) + 1)
                return bool(np.all(self.free(np.linspace(start, goal, count))))
        return True

    def path_obstructed(self, tip, target):
        start = self.rotation.T @ (np.asarray(tip) - self.position)
        goal = self.rotation.T @ (np.asarray(target) - self.position)
        return not self.segment_clear(start, goal)

    def _plan(self, start, goal):
        if not self.free([goal])[0]:
            return None
        if self.segment_clear(start, goal):
            return [goal.copy()]
        step = self.resolution
        # Six-connected lattice edges are covered by the endpoint clearance
        # plus half a cell. This avoids thousands of repeated mesh queries
        # while ensuring the edge interiors are sphere-clear too.
        points, shape, free = self.prepare_lattice()
        free &= np.einsum('ij,j->i', points, self.rotation[2]) + self.position[2] >= self.effective_floor_z + .5*step
        choices = np.flatnonzero(free)
        if not len(choices):
            return None
        # Connect endpoints to nearby free lattice nodes without crossing a
        # wall. A small contact penetration is allowed only at the start.
        def connect(endpoint, allow_escape=False):
            distances = np.linalg.norm(points[choices] - endpoint, axis=1)
            for k in np.argsort(distances)[:64]:
                node = int(choices[k])
                if self.segment_clear(endpoint, points[node]):
                    return node
                if allow_escape and distances[k] < .006:
                    line = np.linspace(endpoint, points[node], 9)
                    valid = self.free(line)
                    if valid[-1] and np.all(valid[np.flatnonzero(valid)[0]:]):
                        return node
            return None
        first, last = connect(start, True), connect(goal)
        if first is None or last is None:
            return None
        # Search the lattice in compiled sparse-graph code. The previous
        # Python A* loop caused hundreds of milliseconds of viewer stalls.
        indices = np.arange(len(points)).reshape(shape)
        rows, cols = [], []
        for axis in range(3):
            left, right = [slice(None)]*3, [slice(None)]*3
            left[axis], right[axis] = slice(None,-1), slice(1,None)
            a,b = indices[tuple(left)].ravel(), indices[tuple(right)].ravel()
            valid = free[a] & free[b]
            rows.append(a[valid]); cols.append(b[valid])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = coo_matrix((np.full(len(rows),step), (rows,cols)),
                           shape=(len(points),len(points))).tocsr()
        distances, previous = dijkstra(graph, directed=False, indices=first,
                                       return_predecessors=True)
        if not np.isfinite(distances[last]):
            return None
        path, node = [goal.copy(), points[last]], last
        while node != first:
            node = int(previous[node])
            path.append(points[node])
        path.reverse()
        # Visibility smoothing keeps only collision-clear chords.
        smooth, anchor, i = [], start, 0
        while i < len(path):
            # Probe increasingly distant chords, then bisect the first
            # blocked interval. Every accepted shortcut is still checked;
            # visibility need not be monotone for path validity.
            j, stride = i, 1
            blocked_at = len(path)
            while j < len(path)-1:
                probe = min(len(path)-1, i+stride)
                if not self.segment_clear(anchor,path[probe]):
                    blocked_at = probe
                    break
                j = probe
                stride *= 2
            while blocked_at-j > 1:
                probe = (blocked_at+j)//2
                if self.segment_clear(anchor,path[probe]):
                    j = probe
                else:
                    blocked_at = probe
            smooth.append(path[j]); anchor = path[j]; i = j+1
        return smooth

    def update(self, tip, obj, best_surface, best_track, object_top_z,
               keepout, arrived, exec_press, retarget=True):
        self.last_plan_seconds = 0.
        tip = np.asarray(tip, dtype=float)
        start = self.rotation.T @ (tip - self.position)
        goal = self.rotation.T @ (np.asarray(best_track) - self.position)
        done = (self.destination_local is not None
                and np.linalg.norm(start - self.destination_local) < .012
                and self.segment_clear(start, self.destination_local))
        # Ranking latches its guide in the contact policy. Once that policy
        # accepts another execution patch, this route must follow the supplied
        # target instead of finishing an obsolete trip into the cavity.
        changed = (retarget and self.destination_local is not None
                   and np.linalg.norm(goal - self.destination_local) > 1e-6)
        if self.route is None or done or changed:
            self.destination_local = goal.copy()
            started = time.perf_counter()
            self.route = self._plan(start, goal)
            self.last_plan_seconds = time.perf_counter() - started
            self.route_failed = self.route is None
        if self.route_failed:
            self.blocked, self.phase, self.via = True, 'no-path', tip.copy()
            return True, self.via.copy(), self.phase
        while len(self.route) > 1 and np.linalg.norm(start - self.route[0]) < .001:
            self.route.pop(0)
        if len(self.route) > 1:
            # Every skip is checked against all collision parts.
            while len(self.route) > 1 and self.segment_clear(start, self.route[1]):
                self.route.pop(0)
        destination = self.position + self.rotation @ self.route[0]
        limit = min(float(self.max_step or .005), float(self.max_lead or .005))
        self.via = tip + bounded_step(tip, destination, limit)
        self.blocked = (len(self.route) > 1
            or np.linalg.norm(start - self.destination_local) > .012
            or not self.segment_clear(start, self.destination_local))
        self.phase = 'cavity-transit' if self.blocked else 'drop'
        if not self.blocked:
            self.via = tip + bounded_step(tip, np.asarray(exec_press), limit)
        return bool(self.blocked), self.via.copy(), self.phase
