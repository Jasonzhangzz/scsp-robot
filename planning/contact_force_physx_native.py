"""Warp population search + native GPU PhysX; loaded only in the SDK worker.

All samples restart from the same pose/velocity. A separated, unloaded step
before each rollout clears the previous sample's contact manifold/history.
No SCM prediction, contact impulse, or pose integration kernel is executed.
"""
from pathlib import Path
import tempfile
import time
import xml.etree.ElementTree as ET

from isaacgym import gymapi, gymtorch
import numpy as np
import warp as wp

from planning.contact_force_gpu_cem import GPUContactCEM
from planning.point_force_gpu import (ForceJob, V6, bounded_force, initialize,
                                      sample, update, pose_cost)


@wp.kernel
def reset_states(jobs: wp.array(dtype=ForceJob), states: wp.array2d(dtype=wp.float32),
                 initial: wp.array(dtype=wp.float32), support: wp.array(dtype=wp.float32),
                 origins: wp.array(dtype=wp.vec3f), clear: int):
    j = jobs[0]
    c, s = wp.tid()
    body = 2*(c*j.population+s)
    for k in range(13):
        states[body, k] = initial[k]
        states[body+1, k] = support[k]
    for k in range(3):
        states[body, k] += origins[c*j.population+s][k]
        states[body+1, k] += origins[c*j.population+s][k]
    if clear == 1 or j.valid[c] == 0:
        states[body, 2] += wp.float32(10.)
        for k in range(7, 13):
            states[body, k] = wp.float32(0.)


@wp.kernel
def wrenches(jobs: wp.array(dtype=ForceJob), force: wp.array(dtype=wp.vec3f),
             torque: wp.array(dtype=wp.vec3f), external: V6):
    j = jobs[0]
    c, s = wp.tid()
    body = 2*(c*j.population+s)
    f = bounded_force(j, c, j.samples[c, s])
    local_force = j.frame[c]*f
    rotation = wp.transform_get_rotation(j.pose)
    world_force = wp.quat_rotate(rotation, local_force)
    world_torque = wp.quat_rotate(rotation, wp.cross(j.point[c], local_force))
    force[body] = wp.vec3f(world_force+wp.vec3d(external[0], external[1], external[2]))
    torque[body] = wp.vec3f(world_torque+wp.vec3d(external[3], external[4], external[5]))
    if j.valid[c] == 0:
        force[body] = wp.vec3f(0.)
        torque[body] = wp.vec3f(0.)


@wp.kernel
def score(jobs: wp.array(dtype=ForceJob), states: wp.array2d(dtype=wp.float32),
          origins: wp.array(dtype=wp.vec3f)):
    j = jobs[0]
    c, s = wp.tid()
    body = 2*(c*j.population+s)
    origin = origins[c*j.population+s]
    p = wp.vec3d(wp.float64(states[body, 0]-origin[0]), wp.float64(states[body, 1]-origin[1]),
                 wp.float64(states[body, 2]-origin[2]))
    q = wp.quatd(wp.float64(states[body, 3]), wp.float64(states[body, 4]),
                 wp.float64(states[body, 5]), wp.float64(states[body, 6]))
    pose = wp.transformd(p, wp.normalize(q))
    f = bounded_force(j, c, j.samples[c, s])
    cost = pose_cost(pose, j.target, j.config)+j.force_reg*wp.dot(f, f)+j.tangent_reg*(f[1]*f[1]+f[2]*f[2])
    if j.valid[c] == 0 or not wp.isfinite(cost):
        cost = wp.float64(1.e100)
    j.costs[c, s] = cost
    j.poses[c, s] = pose
    j.velocities[c, s] = V6(wp.float64(states[body, 7]), wp.float64(states[body, 8]),
        wp.float64(states[body, 9]), wp.float64(states[body, 10]),
        wp.float64(states[body, 11]), wp.float64(states[body, 12]))
    j.alphas[c, s] = wp.float64(1.)
    j.residuals[c, s] = wp.float64(0.)


class GPUPhysXContactCEM(GPUContactCEM):
    def _warm_up(self):
        # Parent's warm-up executes SCM. Native warm-up is done by the worker.
        pass

    def __init__(self, optimizer, mass, inertia_diag, *, scene, physics_dt=.002,
                 position_iterations=8, velocity_iterations=1, contact_offset=.001, **settings):
        if optimizer.friction_cone != 'pyramid':
            raise ValueError('PhysX CEM currently requires the Franka pyramid force cone')
        pop = int(settings.get('population_size', 24))
        iterations = int(settings.get('population_iterations', 12))
        elite = float(settings.get('elite_fraction', .25))
        if pop < 2 or iterations < 1 or not 0 < elite <= 1:
            raise ValueError('invalid CEM population/iterations/elite fraction')
        if not np.isfinite(physics_dt) or physics_dt <= 0:
            raise ValueError('physics_dt must be positive and finite')
        if not np.isfinite(mass) or mass <= 0 or not np.all(np.asarray(inertia_diag) > 0):
            raise ValueError('PhysX requires positive physical mass and inertia')
        steps = int(round(optimizer.h/physics_dt))
        if steps < 1 or abs(steps*physics_dt-optimizer.h) > 1e-9:
            raise ValueError('policy interval must be an integer multiple of physics_dt')
        self.steps = steps
        self.scene = scene
        self.support = None
        self.sim = None
        super().__init__(optimizer, mass, inertia_diag, **settings)
        self.gym = gymapi.acquire_gym()
        params = gymapi.SimParams()
        params.dt, params.substeps = float(physics_dt), 1
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(0., 0., 0.)  # tau_o includes gravity
        params.use_gpu_pipeline = True
        params.physx.use_gpu = True
        params.physx.solver_type = 1
        params.physx.num_position_iterations = int(position_iterations)
        params.physx.num_velocity_iterations = int(velocity_iterations)
        params.physx.contact_offset = float(contact_offset)
        params.physx.rest_offset = 0.
        params.physx.max_depenetration_velocity = 10.
        params.physx.bounce_threshold_velocity = 2*9.81*physics_dt
        params.physx.max_gpu_contact_pairs = 8*1024*1024
        params.physx.default_buffer_size_multiplier = 16.
        self.sim = self.gym.create_sim(self.gpu.device.ordinal, -1, gymapi.SIM_PHYSX, params)
        if self.sim is None:
            raise RuntimeError('Could not create GPU PhysX simulation')
        try:
            self._create_scene(mass, inertia_diag, scene, pop)
            self.gym.prepare_sim(self.sim)
            descriptor = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.root_torch = gymtorch.wrap_tensor(descriptor)
            if self.root_torch.device.type != 'cuda':
                raise RuntimeError('PhysX silently disabled the GPU pipeline')
            self.root = wp.from_torch(self.root_torch, dtype=wp.float32)
            self.root_descriptor = descriptor
            bodies = 2*(self.count+1)*pop
            self.forces = wp.zeros(bodies, dtype=wp.vec3f, device=self.gpu.device)
            self.torques = wp.zeros_like(self.forces)
            self.force_torch, self.torque_torch = wp.to_torch(self.forces), wp.to_torch(self.torques)
            self.force_descriptor = gymtorch.unwrap_tensor(self.force_torch)
            self.torque_descriptor = gymtorch.unwrap_tensor(self.torque_torch)
            self.initial = wp.zeros(13, dtype=wp.float32, device=self.gpu.device)
            self.support_state = wp.zeros(13, dtype=wp.float32, device=self.gpu.device)
            wp.load_module(module=__name__, device=self.gpu.device)
            self.metadata = dict(engine='Isaac Gym Preview 4 GPU PhysX', gpu_pipeline=True,
                gpu=self.gpu.device.name, worlds=(self.count+1)*pop, population=pop,
                generations=iterations, horizon_s=optimizer.h, physics_dt_s=physics_dt,
                steps=steps, solver='TGS', position_iterations=position_iterations,
                velocity_iterations=velocity_iterations, contact_offset=contact_offset,
                expected_shapes=self.expected_shapes, reset='unloaded separated step before every evaluation',
                environment_layout='2 m grid; independent collision groups; subtract float32 origins before scoring',
                force_semantics='bounded external point force, fixed reference-frame wrench over horizon; native support contacts',
                scene=scene)
        except Exception:
            self.close()
            raise

    def _create_scene(self, mass, inertia_diag, scene, population):
        self.assets_dir = tempfile.TemporaryDirectory(prefix='scsp_physx_cem_')
        robot = ET.Element('robot', name='object')
        link = ET.SubElement(robot, 'link', name='object')
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'mass', value=str(mass))
        ET.SubElement(inertial, 'inertia', ixx=str(inertia_diag[0]), iyy=str(inertia_diag[1]),
                      izz=str(inertia_diag[2]), ixy='0', ixz='0', iyz='0')
        parts = scene.get('collision_mesh_paths') or [scene['mesh_path']]
        for path in parts:
            absolute = Path(path).resolve()
            if not absolute.is_file():
                raise FileNotFoundError(absolute)
            geometry = ET.SubElement(ET.SubElement(link, 'collision'), 'geometry')
            ET.SubElement(geometry, 'mesh', filename=str(absolute), scale='1 1 1')
        ET.ElementTree(robot).write(str(Path(self.assets_dir.name)/'object.urdf'))
        options = gymapi.AssetOptions()
        options.disable_gravity = True
        options.linear_damping, options.angular_damping = 0., 0.
        options.override_com, options.override_inertia = False, False
        options.vhacd_enabled = False
        options.enable_gyroscopic_forces = True
        asset = self.gym.load_asset(self.sim, self.assets_dir.name, 'object.urdf', options)
        if asset is None or self.gym.get_asset_rigid_body_count(asset) != 1:
            raise RuntimeError('PhysX CEM needs a one-body collision asset')
        self.expected_shapes = len(parts)
        if self.gym.get_asset_rigid_shape_count(asset) != len(parts):
            raise RuntimeError('PhysX collision shape count mismatch')
        fixed = gymapi.AssetOptions()
        fixed.fix_base_link = True
        support_asset = self.gym.create_box(self.sim, 1., 1., .1, fixed)
        origins = []
        width = int(np.ceil(np.sqrt((self.count+1)*population)))
        for index in range((self.count+1)*population):
            env = self.gym.create_env(self.sim, gymapi.Vec3(-1., -1., 0.), gymapi.Vec3(1., 1., 0.), width)
            origin = self.gym.get_env_origin(env)
            origins.append([origin.x, origin.y, origin.z])
            for native, name, mu in ((asset, 'object', scene.get('object_friction', .7)),
                                     (support_asset, 'support', scene.get('table_friction', .7))):
                pose = gymapi.Transform()
                pose.p.z = 10. if name == 'object' else -10.
                actor = self.gym.create_actor(env, native, pose, name, index, 0)
                if actor < 0:
                    raise RuntimeError('PhysX actor creation failed')
                props = self.gym.get_actor_rigid_shape_properties(env, actor)
                for prop in props:
                    prop.friction, prop.restitution = float(mu), 0.
                    prop.rolling_friction, prop.torsion_friction = 0., 0.
                self.gym.set_actor_rigid_shape_properties(env, actor, props)
                if index == 0 and name == 'object':
                    actual = self.gym.get_actor_rigid_body_properties(env, actor)[0]
                    diagonal = [actual.inertia.x.x, actual.inertia.y.y, actual.inertia.z.z]
                    if not np.allclose(diagonal, inertia_diag, rtol=1e-5) or not np.isclose(actual.mass, mass):
                        raise RuntimeError('PhysX asset mass/inertia mismatch')
        self.origins = wp.array(origins, dtype=wp.vec3f, device=self.gpu.device)

    def _environment_rows(self, env_phi=None):
        return np.zeros((1, 6)), np.zeros(1)

    def _prepare(self, v_last, tau_o, current_x, *args, **kwargs):
        self.velocity = np.asarray(v_last, float).reshape(6)
        self.external = np.asarray(tau_o, float).reshape(6)
        self.current = np.asarray(current_x, float).reshape(7)
        super()._prepare(v_last, tau_o, current_x, *args, **kwargs)

    def set_support_plane(self, point, normal):
        self.support = (np.asarray(point, float), np.asarray(normal, float))

    def _step(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def _run(self):
        from scipy.spatial.transform import Rotation
        started = time.perf_counter()
        if self.support is None:
            raise ValueError('PhysX requires an explicit support plane in the evaluation frame')
        point, normal = self.support
        normal = normal/np.linalg.norm(normal)
        point = normal*np.dot(point, normal)  # nearest point on the plane
        # Rotation taking support-box +Z to its outward plane normal.
        q = np.r_[np.cross([0., 0., 1.], normal), 1.+normal[2]]
        if np.linalg.norm(q) < 1e-10:
            q = np.array([1., 0., 0., 0.])
        q /= np.linalg.norm(q)
        self.support_state.assign(np.r_[point-.05*normal, q, np.zeros(6)].astype(np.float32))
        initial_q = self.current[[4, 5, 6, 3]]
        omega = Rotation.from_quat(initial_q).apply(self.velocity[3:])
        self.initial.assign(np.r_[self.current[:3], initial_q, self.velocity[:3], omega].astype(np.float32))
        gpu, wp_device = self.gpu, self.gpu.device
        dim = (self.count+1, gpu.job.population)
        wp.launch(initialize, self.count+1, inputs=[gpu.jobs], device=wp_device)
        for generation in range(gpu.search_iterations):
            wp.launch(sample, dim, inputs=[gpu.jobs, generation], device=wp_device)
            wp.launch(reset_states, dim, inputs=[gpu.jobs, self.root, self.initial, self.support_state, self.origins, 1], device=wp_device)
            wp.synchronize_device(wp_device)
            self.gym.set_actor_root_state_tensor(self.sim, self.root_descriptor)
            self._step()  # no external force: clear all prior contact history
            wp.launch(reset_states, dim, inputs=[gpu.jobs, self.root, self.initial, self.support_state, self.origins, 0], device=wp_device)
            wp.launch(wrenches, dim, inputs=[gpu.jobs, self.forces, self.torques, V6(self.external)], device=wp_device)
            wp.synchronize_device(wp_device)
            self.gym.set_actor_root_state_tensor(self.sim, self.root_descriptor)
            for _ in range(self.steps):
                self.gym.apply_rigid_body_force_tensors(self.sim, self.force_descriptor,
                                                       self.torque_descriptor, gymapi.ENV_SPACE)
                self._step()
            self.gym.refresh_actor_root_state_tensor(self.sim)
            # Gym and Warp use different streams. Explicit synchronization
            # keeps root reads and subsequent setters ordered across engines.
            wp.synchronize_device(wp_device)
            wp.launch(score, dim, inputs=[gpu.jobs, self.root, self.origins], device=wp_device)
            wp.launch(update, self.count+1, inputs=[gpu.jobs], device=wp_device)
        wp.synchronize_device(wp_device)
        result = gpu.job.best_cost.numpy(), gpu.job.best.numpy(), gpu.job.best_pose.numpy()
        self.last_timing['gpu_ms'] = 1000*(time.perf_counter()-started)
        self.last_timing['physics_steps'] = gpu.search_iterations*(self.steps+1)
        return result

    def close(self):
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
        if hasattr(self, 'assets_dir'):
            self.assets_dir.cleanup()
