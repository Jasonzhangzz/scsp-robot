"""Independent FCL contact geometry and unsmoothed rigid NCP predictions.

MuJoCo supplies compiled mass/inertia, smooth dynamics, Jacobians and pose
integration only. No mj_collision/mj_forward call enters either predictor.
Plane contacts enumerate source convex vertices rather than FCL's single
support point, so a flat box retains all four support corners. Cylinders use
an explicit 96-sided convex surface for BOTH pair and plane collision.
"""
from __future__ import annotations

import time
import numpy as np
import fcl
import mujoco
from scipy.spatial import ConvexHull

from examples.mpc.franka.scm.contact_dynamics import full_mass
from examples.mpc.fingertips.test.collision_prediction_external_contacts import (
    ExternalContactPredictors, smooth_data)
from examples.mpc.fingertips.test.collision_prediction_isaacgym_native import (
    contact_frame, _cylinder)
from models.ncp_contact import NCPSettings, solve_ncp


class FCLContacts:
    """All allowed floor/object/ball shape pairs, without a whole-body hull.

    Positive-gap candidates come from closest points or plane vertices within
    candidate_margin; no future reference state or force is consumed. Contacts
    are not deduplicated across convex components. Reaching a capacity is an
    error, not silent truncation. Plane contact positions are mid-gap points.
    """
    def __init__(self, model, candidate_margin=.01, max_contacts=2048):
        if not np.isfinite(candidate_margin) or candidate_margin < 0 or max_contacts < 1:
            raise ValueError("Invalid contact candidate margin/capacity")
        self.model, self.margin, self.capacity = model, candidate_margin, max_contacts
        self.independent_gjk_retries = 0
        self.sphere_witness_repairs = 0
        self.shapes, self.vertices, self.sphere_radius = {}, {}, {}
        self.hulls = {}
        self.geoms = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        for g in self.geoms:
            kind, size = model.geom_type[g], model.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_PLANE:
                if model.geom_bodyid[g] != 0:
                    raise ValueError("Only static planes are supported")
                continue
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                self.shapes[g] = fcl.Sphere(float(size[0]))
                self.sphere_radius[g] = float(size[0])
                continue
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                vertices = np.array([[x,y,z] for x in (-size[0], size[0])
                    for y in (-size[1], size[1]) for z in (-size[2], size[2])])
                self.shapes[g] = fcl.Box(*(2*size))
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                vertices, _ = _cylinder(size[0], size[1], 96)
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                vertices = np.asarray(model.mesh_vert[start:start+count], float)
            else:
                raise ValueError(f"Unsupported FCL shape {kind}; no geometry fallback")
            self.vertices[g] = vertices.copy()
            self.hulls[g] = ConvexHull(vertices)
            if g not in self.shapes:
                hull = self.hulls[g]
                faces = hull.simplices.copy()
                # FCL Convex requires outward CCW polygon winding.
                for i, face in enumerate(faces):
                    normal = np.cross(vertices[face[1]]-vertices[face[0]], vertices[face[2]]-vertices[face[0]])
                    if normal @ hull.equations[i, :3] < 0:
                        faces[i] = face[::-1]
                polygons = np.column_stack((np.full(len(faces), 3), faces)).ravel()
                self.shapes[g] = fcl.Convex(vertices, len(faces), polygons)

    def _sphere_witness(self, first, second, data):
        """Exact surface witness on the same shapes for degenerate FCL output.

        A sphere-center to triangle distance is well-conditioned when the two
        *surfaces* touch; FCL's coincident surface witnesses lose their normal.
        This geometric repair uses no dynamics or displaced/inflated shape.
        """
        self.sphere_witness_repairs += 1
        sphere = second if second in self.sphere_radius else first
        other = first if sphere == second else second
        center = data.geom_xpos[sphere]
        if other in self.sphere_radius:
            delta=center-data.geom_xpos[other]; length=np.linalg.norm(delta)
            if length < 1e-14: raise RuntimeError('Coincident sphere centers have no unique normal')
            normal=delta/length
            surface=data.geom_xpos[other]+normal*self.sphere_radius[other]
            gap=length-self.sphere_radius[other]-self.sphere_radius[sphere]
        else:
            from trimesh.triangles import closest_point
            R=data.geom_xmat[other].reshape(3,3)
            local=R.T@(center-data.geom_xpos[other])
            hull=self.hulls[other]
            triangles=self.vertices[other][hull.simplices]
            witnesses=closest_point(triangles,np.broadcast_to(local,(len(triangles),3)).copy())
            squared=np.sum((witnesses-local)**2,axis=1)
            nearest=witnesses[np.argmin(squared)]
            delta=local-nearest;length=np.linalg.norm(delta)
            if length < 1e-14: raise RuntimeError('Sphere center on convex surface has no unique witness normal')
            inside=np.max(hull.equations[:,:3]@local+hull.equations[:,3]) < 0
            normal=R@(delta/length*(-1. if inside else 1.))
            surface=R@nearest+data.geom_xpos[other]
            gap=(-length if inside else length)-self.sphere_radius[sphere]
        position=surface+.5*gap*normal
        return position, normal if other==first else -normal, gap

    def manifold(self, data):
        model, rows = self.model, []
        objects = {g: fcl.CollisionObject(shape, fcl.Transform(
            data.geom_xmat[g].reshape(3,3), data.geom_xpos[g])) for g, shape in self.shapes.items()}

        def append(g1, g2, point, normal, gap):
            if len(rows) >= self.capacity:
                raise RuntimeError("FCL manifold capacity exceeded; no contacts truncated")
            normal = np.asarray(normal, float)
            if not np.isfinite(normal).all() or np.linalg.norm(normal) < .5:
                raise RuntimeError("Invalid FCL contact normal")
            mu = max(model.geom_friction[g1, 0], model.geom_friction[g2, 0])
            rows.append((g1, g2, np.asarray(point), contact_frame(normal/np.linalg.norm(normal)), float(gap), mu))

        for i, first in enumerate(self.geoms):
            for second in self.geoms[i+1:]:
                if model.geom_bodyid[first] == model.geom_bodyid[second]:
                    continue
                if not ((model.geom_contype[first] & model.geom_conaffinity[second]) or
                        (model.geom_contype[second] & model.geom_conaffinity[first])):
                    continue
                planes = [g for g in (first, second) if g not in objects]
                if planes:
                    if len(planes) != 1:
                        raise ValueError("Plane-plane interaction is unsupported")
                    plane = planes[0]
                    shape = second if plane == first else first
                    normal = data.geom_xmat[plane].reshape(3,3)[:, 2]
                    origin = data.geom_xpos[plane]
                    if shape in self.sphere_radius:
                        point = data.geom_xpos[shape]-normal*self.sphere_radius[shape]
                        points = point[None]
                    else:
                        points = self.vertices[shape] @ data.geom_xmat[shape].reshape(3,3).T + data.geom_xpos[shape]
                    gaps = (points-origin) @ normal
                    for point, gap in zip(points[gaps <= self.margin], gaps[gaps <= self.margin]):
                        append(plane, shape, point-.5*gap*normal, normal, gap)
                    continue
                result = fcl.CollisionResult()
                fcl.collide(objects[first], objects[second], fcl.CollisionRequest(
                    num_max_contacts=self.capacity+1, enable_contact=True), result)
                if result.is_collision:
                    if not result.contacts:
                        raise RuntimeError("FCL collision reported no contact geometry")
                    for contact in result.contacts:
                        # FCL can internally swap object order for shape dispatch.
                        normal = contact.normal if contact.o1 is self.shapes[first] else -contact.normal
                        if np.linalg.norm(normal)<.5 and (first in self.sphere_radius or second in self.sphere_radius):
                            point,normal,gap=self._sphere_witness(first,second,data)
                            append(first,second,point,normal,gap)
                        else:
                            append(first, second, contact.pos, normal, -contact.penetration_depth)
                else:
                    distance = fcl.DistanceResult()
                    gap = fcl.distance(objects[first], objects[second], fcl.DistanceRequest(
                        enable_nearest_points=True), distance)
                    if 0 < gap <= self.margin:
                        p1, p2 = map(np.asarray, distance.nearest_points)
                        if np.linalg.norm(p2-p1)<1e-10 and (first in self.sphere_radius or second in self.sphere_radius):
                            point,normal,gap=self._sphere_witness(first,second,data)
                            append(first,second,point,normal,gap)
                        else:
                            append(first, second, .5*(p1+p2), (p2-p1)/gap, gap)
                    elif gap <= 0:
                        # libccd occasionally fails at a near-touching convex
                        # surface (distance=-1 sentinel). Retry FCL's independent
                        # GJK implementation on the SAME shapes, never substitute
                        # MuJoCo geometry or fabricate a zero-gap contact.
                        self.independent_gjk_retries += 1
                        retry = fcl.CollisionResult()
                        fcl.collide(objects[first],objects[second],fcl.CollisionRequest(
                            num_max_contacts=self.capacity+1,enable_contact=True,
                            gjk_solver_type=fcl.GJKSolverType.GST_INDEP),retry)
                        if not retry.is_collision or not retry.contacts:
                            if first in self.sphere_radius or second in self.sphere_radius:
                                point,normal,gap=self._sphere_witness(first,second,data)
                                if gap<=self.margin: append(first,second,point,normal,gap)
                                continue
                            raise RuntimeError("FCL distance/collision disagreement in both GJK implementations")
                        for contact in retry.contacts:
                            normal = contact.normal if contact.o1 is self.shapes[first] else -contact.normal
                            if np.linalg.norm(normal)<.5 and (first in self.sphere_radius or second in self.sphere_radius):
                                point,normal,gap=self._sphere_witness(first,second,data)
                                append(first,second,point,normal,gap)
                            else:
                                append(first,second,contact.pos,normal,-contact.penetration_depth)
        rows.sort(key=lambda r: (r[0], r[1], *r[2], r[4]))
        n = len(rows)
        return dict(gap=np.array([r[4] for r in rows]),
            position=np.array([r[2] for r in rows]).reshape(n,3),
            frame=np.array([r[3] for r in rows]).reshape(n,3,3),
            geom=np.array([[r[0], r[1]] for r in rows], int).reshape(n,2),
            body=np.array([[model.geom_bodyid[r[0]], model.geom_bodyid[r[1]]] for r in rows], int).reshape(n,2),
            friction=np.array([[r[5], r[5], 0.,0.,0.] for r in rows]).reshape(n,5),
            include_margin=np.full(n, self.margin))


class NCPPredictor(ExternalContactPredictors):
    def __init__(self, model_path, settings, ncp_settings=NCPSettings(), candidate_margin=.01):
        super().__init__(model_path, settings)
        self.ncp_settings = ncp_settings
        self.collision = FCLContacts(self.model, candidate_margin, settings.nconmax)

    def initial_manifold(self, sample):
        return self.collision.manifold(smooth_data(self.scm_model, self._sample(sample)))

    def ncp(self, sample, manifold=None, steps=1):
        began = time.perf_counter()
        diagnostics, records = [], []
        try:
            if not isinstance(steps, int) or steps < 1:
                raise ValueError("steps must be a positive integer")
            current = self._sample(sample)
            initial_q, initial_v = current['qpos'].copy(), current['qvel'].copy()
            trajectory_q, trajectory_v = [initial_q], [initial_v]
            h = self.settings.horizon/steps
            for step in range(steps):
                data = smooth_data(self.scm_model, current)
                contacts = self._manifold(manifold if step == 0 and manifold is not None
                                          else self.collision.manifold(data))
                mass = full_mass(self.scm_model, data)
                inverse = np.linalg.inv(mass)
                free = data.qvel+h*data.qacc_smooth
                jac = self._jacobians(data, contacts)
                velocity, impulse, diagnostic = solve_ncp(free, jac, contacts['gap'], inverse,
                    h, contacts['friction'][:,0], self.ncp_settings)
                diagnostics.append(diagnostic)
                records.append(dict(J=jac, inverse_mass=inverse, free_velocity=free,
                                    impulse=impulse, **contacts))
                if not diagnostic['converged']:
                    raise RuntimeError(f"NCP did not converge at step {step}: {diagnostic}")
                data.qvel[:] = velocity
                mujoco.mj_integratePos(self.scm_model, data.qpos, velocity, h)
                current.update(qpos=data.qpos.copy(), qvel=velocity.copy())
                trajectory_q.append(current['qpos']); trajectory_v.append(current['qvel'])
            if not np.isfinite(np.r_[current['qpos'], current['qvel']]).all():
                raise RuntimeError("Nonfinite NCP prediction")
            return dict(status='ok', failure_reason='', qpos=current['qpos'], qvel=current['qvel'],
                        time=self.settings.horizon, seconds=time.perf_counter()-began,
                        diagnostics=diagnostics, problems=records,
                        trajectory_qpos=np.asarray(trajectory_q), trajectory_qvel=np.asarray(trajectory_v))
        except Exception as exc:
            return dict(self._failed(exc, time.perf_counter()-began), diagnostics=diagnostics, problems=records)
