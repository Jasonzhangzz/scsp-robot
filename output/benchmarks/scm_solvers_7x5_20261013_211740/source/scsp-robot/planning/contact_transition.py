"""Compare continued support with a reachable contact relocation.

Lambda's single-step solve assumes the candidate is already touching. This
rollout removes the finger during travel and refreshes the table footprint
after each object step. It uses the existing object cost and SCM mobility;
there is no angle-based contact lock or additional switching penalty.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class ContactTransitionPredictor:
    def __init__(self, optimizer, vertices, horizon=5, max_prediction_time=.5):
        self.optimizer = optimizer
        self.horizon = max(1, int(horizon))
        self.max_steps = max(self.horizon, int(np.ceil(max_prediction_time/optimizer.h)))
        # Only extreme vertices can touch a plane. The hull also bounds the
        # per-frame work for high-resolution collision meshes.
        import trimesh
        self.vertices = np.asarray(trimesh.Trimesh(
            vertices=np.asarray(vertices), faces=[], process=False).convex_hull.vertices)
        self._build_velocity_function()

    def _build_velocity_function(self):
        import casadi as cs
        opt = self.optimizer
        v, tau, force, point = [cs.SX.sym(name, n) for name, n in
                                [('v', 6), ('tau', 6), ('force', 3), ('point', 3)]]
        frame = cs.SX.sym('frame', 3, 3)
        jac = cs.SX.sym('jac', 4*opt.max_contacts, 6)
        phi = cs.SX.sym('phi', 4*opt.max_contacts)
        inverse, theta = cs.SX.sym('inverse', 6, 6), cs.SX.sym('theta', 6)
        arm_gap = cs.SX.sym('arm_gap')
        velocity = opt._arm_velocity_cs(v, tau, force, point, frame, jac, phi, inverse, theta,
                                        arm_gap=arm_gap, full_velocity=True)
        self.velocity_fn = cs.Function('contact_transition_velocity',
            [v, tau, force, point, frame, jac, phi, inverse, theta, arm_gap], [velocity])
        # Compile the same map once, before simulation starts. Replaying many
        # short contact steps in the SX interpreter would delay async plans.
        import hashlib
        import os
        from pathlib import Path
        import subprocess
        import tempfile
        import shutil
        if shutil.which('cc') is None:
            return
        key = hashlib.sha256(self.velocity_fn.serialize().encode()).hexdigest()[:20]
        cache = Path(tempfile.gettempdir()) / ('scsp_contact_transition_%d' % os.getuid()) / key
        cache.mkdir(parents=True, exist_ok=True)
        library = cache / 'transition.so'
        if not library.exists():
            generator = cs.CodeGenerator('transition.c')
            generator.add(self.velocity_fn)
            generator.generate(str(cache) + '/')
            pending = cache / ('transition_%d.so' % os.getpid())
            subprocess.run(['cc', '-O2', '-shared', '-fPIC', str(cache/'transition.c'),
                            '-o', str(pending), '-lm'], check=True, capture_output=True)
            pending.replace(library)
        self.velocity_fn = cs.external('contact_transition_velocity', str(library))

    def _support_batch(self, positions, rotations, floor_z, mu):
        # Batch the mesh query and lever-arm construction for all proposed
        # trajectories. This keeps the forecast inside the async plan budget.
        # Query heights for the whole hull, then compute XY distances only
        # for vertices inside the contact band. Transforming and measuring
        # every non-supporting vertex dominated the bunny forecast latency.
        heights = rotations[:,2,:] @ self.vertices.T + positions[:,2,None]
        count, ncon = len(positions), self.optimizer.max_contacts
        row = np.arange(count)
        indices = np.zeros((count,ncon), dtype=int)
        active = np.zeros((count,ncon), dtype=bool)
        for i in range(count):
            cloud = np.flatnonzero(heights[i] <= heights[i].min()+.001)
            xy = self.vertices[cloud] @ rotations[i,:2,:].T
            idx = int(heights[i,cloud].argmin())
            distances = np.full(len(cloud),np.inf)
            for k in range(ncon):
                if k and distances[idx] < .002**2:
                    break
                indices[i,k],active[i,k] = cloud[idx],True
                distances = np.minimum(distances,((xy-xy[idx])**2).sum(axis=1))
                idx = int(distances.argmax())
        points = self.vertices[indices]
        skew = np.zeros((count,ncon,3,3))
        skew[:,:,0,1], skew[:,:,0,2] = -points[:,:,2], points[:,:,1]
        skew[:,:,1,0], skew[:,:,1,2] = points[:,:,2], -points[:,:,0]
        skew[:,:,2,0], skew[:,:,2,1] = -points[:,:,1], points[:,:,0]
        relative = np.concatenate([np.broadcast_to(np.eye(3), skew.shape), -skew], axis=3)
        n, t1, t2 = rotations[:,2,:], rotations[:,1,:], -rotations[:,0,:]
        pyramid = np.stack([n+mu*t1, n+mu*t2, n-mu*t1, n-mu*t2], axis=1)
        jac = (pyramid[:,None,:,:] @ relative) * active[:,:,None,None]
        gaps = heights[row[:,None],indices] - floor_z
        phi = np.repeat(np.where(active, gaps, 1.), 4, axis=1)
        return jac.reshape(count,4*ncon,6), phi

    def _rollout_batch(self, pose, target, config, gravity_world, floor_z, mu,
                       points, frames, forces, delays, steps, velocity_world=None,
                       tip_world=None, radius=0., destination_radius=None):
        opt = self.optimizer
        count = len(points)
        poses = np.tile(np.asarray(pose, dtype=float), (count,1))
        totals, costs = np.zeros(count), []
        mapped = getattr(self, '_mapped_velocity', {})
        if count not in mapped:
            mapped[count] = self.velocity_fn.map(count)
            self._mapped_velocity = mapped
        # Model parameters change online; pack the current values every cycle.
        inverse = np.tile(opt.Q_inv, (1,count))
        theta = np.tile(opt.scm_theta[:,None], (1,count))
        tips = None if tip_world is None else np.tile(tip_world,(count,1))
        initial_rotation = Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_matrix()
        destination_radius = radius if destination_radius is None else destination_radius
        destinations = (np.asarray(pose)[:3] + points @ initial_rotation.T
                        - destination_radius*(frames[:,:,0] @ initial_rotation.T))
        for k in range(int(steps)):
            rotations = Rotation.from_quat(poses[:,[4,5,6,3]]).as_matrix()
            jac, phi = self._support_batch(poses[:,:3], rotations, floor_z, mu)
            tau = np.c_[np.einsum('nji,j->ni', rotations, gravity_world), np.zeros((count,3))]
            velocity = np.zeros((count,6))
            if k == 0 and velocity_world is not None:
                velocity[:,:3] = np.einsum('nji,j->ni', rotations, velocity_world[:3])
                velocity[:,3:] = np.einsum('nji,j->ni', rotations, velocity_world[3:])
            touching = k >= np.asarray(delays)
            surface = poses[:,:3] + np.einsum('nij,nj->ni',rotations,points)
            inward = np.einsum('nij,nj->ni',rotations,frames[:,:,0])
            arrival = k == np.asarray(delays)
            centers = surface - radius*inward
            if tips is None:
                tips = centers.copy()
            # New contacts become reachable only after the predicted travel;
            # existing contact starts at the measured fingertip position.
            tips[arrival & (np.asarray(delays) > 0)] = destinations[arrival & (np.asarray(delays) > 0)]
            arm_gaps = ((surface-tips)*inward).sum(axis=1) - radius
            arm_gaps = np.where(touching,arm_gaps,1.)
            current_frames = frames * touching[:,None,None]
            wrench = np.einsum('nij,nj->ni', current_frames, forces)
            body_v = np.asarray(mapped[count](velocity.T, tau.T, wrench.T, points.T,
                np.concatenate(current_frames, axis=1), np.concatenate(jac, axis=1),
                phi.T, inverse, theta,arm_gaps[None,:])).T
            poses[:,:3] += opt.h * np.einsum('nij,nj->ni', rotations, body_v[:,:3])
            tips += opt.h * np.einsum('nij,nj->ni',rotations,body_v[:,6:9])
            q, w = poses[:,3:7].copy(), body_v[:,3:6]
            poses[:,3:7] += .5 * opt.h * np.c_[-(q[:,1:]*w).sum(axis=1),
                q[:,:1]*w + np.cross(q[:,1:],w)]
            poses[:,3:7] /= np.linalg.norm(poses[:,3:7],axis=1)[:,None]
            q = poses[:,3:7]
            axis = config[3:6] + 2*np.cross(q[:,1:],
                np.cross(q[:,1:],config[3:6]) + q[:,:1]*config[3:6])
            cost = (config[0]*((poses[:,:3]-target[:3])**2).sum(axis=1)
                + config[1]*(1-np.clip(q @ (target[3:7]/np.linalg.norm(target[3:7])), -1,1)**2)
                + config[2]*.5*(1-np.clip(axis @ config[6:9],-1,1)))
            totals += cost
            costs.append(cost)
        costs = np.asarray(costs)
        return [dict(cost=float(totals[i]+10*costs[-1,i]), pose=poses[i], costs=costs[:,i])
                for i in range(count)]

    def rollout(self, pose, target, config, gravity_world, floor_z, mu,
                point, frame, force, delay_steps, steps, velocity_world=None):
        """Object path + 10x terminal cost, matching MPC's object terms.

        Before arrival there is no finger Jacobian. A zero force at a held
        patch still retains unilateral support; withdrawal removes it.
        """
        return self._rollout_batch(pose, target, config, gravity_world, floor_z, mu,
            np.asarray(point)[None,:], np.asarray(frame)[None,:,:],
            np.asarray(force)[None,:], [delay_steps], steps, velocity_world)[0]

    def evaluate(self, curr_q, target_world, config_world, actual_contact,
                 measured_force, candidate_ids, approach_via, radius, margin,
                 floor_z, mu, gravity_world, velocity_world=None):
        """Evaluate stay and sampled destinations over the SAME time window."""
        from examples.mpc.fingertips.test.test_0902 import (
            _keepout_radius, _patch_press_point, _press_path_blocked,
        )
        opt = self.optimizer
        curr_q = np.asarray(curr_q, dtype=float)
        rotation = Rotation.from_quat(curr_q[[4,5,6,3]]).as_matrix()
        obj, tip = curr_q[:3], curr_q[7:10]
        actual_point = rotation.T @ (np.asarray(actual_contact['point_world']) - obj)
        inward = -rotation.T @ np.asarray(actual_contact['normal_world'])
        inward /= max(np.linalg.norm(inward), 1e-12)
        ref = np.array([0.,0.,1.]) if abs(inward[2]) < .9 else np.array([0.,1.,0.])
        tangent = np.cross(inward, ref); tangent /= np.linalg.norm(tangent)
        actual_frame = np.column_stack([inward, tangent, np.cross(inward, tangent)])
        top = float((self.vertices @ rotation.T + obj)[:,2].max())
        keepout = _keepout_radius(self.vertices.min(axis=0), self.vertices.max(axis=0))
        step = min(float(approach_via.max_step or .005), float(approach_via.max_lead or .005))
        plans = []
        for idx in np.asarray(candidate_ids, dtype=int):
            force, point = None, opt.sample_point[idx]
            hits = np.flatnonzero(np.asarray(opt.last_candidate_ids) == idx)
            if hits.size:
                force = np.asarray(opt.last_candidate_forces[int(hits[0])], dtype=float)
            if force is None or not np.isfinite(force).all():
                continue
            frame = np.column_stack([opt.normal[idx], opt.t1[idx], opt.t2[idx]])
            surface = rotation @ point + obj
            track = surface - max(1e-4, radius+margin)*(rotation @ frame[:,0])
            press = _patch_press_point(track, surface)
            distance = float(np.linalg.norm(tip - press))
            if _press_path_blocked(tip, obj, press, keepout):
                height = max(top+.025, press[2]+.015)
                distance = abs(height-tip[2]) + np.linalg.norm(tip[:2]-press[:2]) + abs(height-press[2])
            # The live measured patch is evaluated separately with zero delay.
            # Every other destination pays for all bounded approach increments.
            delay = max(1, int(np.ceil(distance / step)))
            plans.append((int(idx), point, frame, force, delay))
        # Recede over a bounded time window. A destination beyond this window
        # stays detached throughout; it must not teleport into the terminal
        # state. This also bounds latency and long-range SCM extrapolation.
        steps = min(self.max_steps, self.horizon + max([p[-1] for p in plans] + [0]))
        common = dict(pose=curr_q[:7], target=target_world, config=config_world,
            gravity_world=gravity_world, floor_z=floor_z, mu=mu, steps=steps,
            velocity_world=velocity_world,tip_world=tip,radius=radius,
            destination_radius=max(1e-4,radius+margin))
        # Releasing pressure while retaining contact is also a feasible stay
        # command. This avoids evaluating only a constant maximum press that
        # overshoots the same objective over the longer window.
        points = [actual_point]*3
        frames = [actual_frame]*3
        forces = [scale*np.asarray(measured_force) for scale in (0., .5, 1.)]
        delays = [0]*3
        for idx, point, frame, force, delay in plans:
            points.extend([point]*3); frames.extend([frame]*3)
            forces.extend([scale*force for scale in (0., .5, 1.)]); delays.extend([delay]*3)
        results = self._rollout_batch(points=np.asarray(points), frames=np.asarray(frames),
            forces=np.asarray(forces), delays=np.asarray(delays), **common)
        stay = min(results[:3], key=lambda p: p['cost'])
        switched = []
        for i, (idx, point, frame, force, delay) in enumerate(plans):
            result = min(results[3+3*i:6+3*i], key=lambda p: p['cost'])
            switched.append(dict(idx=idx, delay_steps=delay, **result))
        best = min(switched, key=lambda p: p['cost']) if switched else None
        # No positive one-step-gain requirement: retaining support can be the
        # best trajectory even if the immediate pose has not improved yet.
        switch = bool(best is not None and best['cost'] < stay['cost'] - 1e-9)
        return dict(switch=switch, best_idx=None if best is None else best['idx'],
            stay_cost=stay['cost'], switch_cost=None if best is None else best['cost'],
            steps=steps, travel_steps=0 if best is None else best['delay_steps'])
