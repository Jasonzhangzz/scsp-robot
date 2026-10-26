"""Python 3.8 Isaac Gym worker: native PhysX contacts, no MuJoCo imports.

Every request creates fresh scenes. Collision-only diagnostics execute a real
2 ms solve in an independent scene; all resulting motion is discarded. Native
records describe pre-integration geometry, although their local anchors are
expressed using the returned post-step body poses. Reference endpoint records
are diagnostic forces over [20,22] ms, never an extension of the evaluated pose.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback
import contextlib
import ctypes
import hashlib
import platform
import re
import tempfile

# Native SDK banners and printf must not corrupt the JSONL protocol.
_PROTOCOL = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

from isaacgym import gymapi  # noqa: E402  (must precede all numeric imports)
import numpy as np  # noqa: E402
from collision_prediction_isaacgym_native import contact_frame, quaternion_matrix  # noqa: E402


def vector(value):
    if isinstance(value, np.void):
        return np.array([float(value[x]) for x in ("x", "y", "z")])
    return np.array([float(value.x), float(value.y), float(value.z)])


def json_value(value):
    if isinstance(value, np.void):
        return {key: json_value(value[key]) for key in value.dtype.names}
    if isinstance(value, np.ndarray):
        return [json_value(x) for x in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def scalar_properties(value):
    result = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        item = getattr(value, name)
        if isinstance(item, (bool, int, float, str)):
            result[name] = item
        elif name == "contact_collection":
            result[name] = int(item)
    return result


def runtime_fingerprint():
    binding = Path(sys.modules[gymapi.Gym.__module__].__file__).resolve()
    libraries = [binding, binding.parent/"libcarb.gym.plugin.so", binding.parent/"libPhysXGpu_64.so"]
    library_info = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in libraries if path.is_file()]
    driver_path = Path("/proc/driver/nvidia/version")
    payload = dict(python_executable=sys.executable, python_version=sys.version,
                   python_implementation=platform.python_implementation(), numpy_version=np.__version__,
                   platform=platform.platform(), machine=platform.machine(), libc=list(platform.libc_ver()),
                   sdk_python_path=str(Path(gymapi.__file__).resolve()), native_libraries=library_info,
                   nvidia_driver=driver_path.read_text().strip() if driver_path.is_file() else None,
                   isaacgym_sdk_label="Isaac Gym Preview 4",
                   sdk_release_label_source="installed IsaacGym_Preview_4_Package; hashes identify exact binary build")
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


@contextlib.contextmanager
def capture_native_log():
    """Preserve native output in the log and expose warnings to the caller."""
    result = {}
    sys.stderr.flush()
    ctypes.CDLL(None).fflush(None)
    saved_one, saved_two = os.dup(1), os.dup(2)
    with tempfile.TemporaryFile() as capture:
        os.dup2(capture.fileno(), 1)
        os.dup2(capture.fileno(), 2)
        try:
            yield result
        finally:
            sys.stderr.flush()
            ctypes.CDLL(None).fflush(None)
            os.dup2(saved_one, 1)
            os.dup2(saved_two, 2)
            os.close(saved_one)
            os.close(saved_two)
            capture.seek(0)
            data = capture.read()
            if data:
                # Use a file object to handle partial low-level writes safely.
                with os.fdopen(os.dup(2), "wb") as log:
                    log.write(data)
            lines = data.decode("utf-8", errors="replace").splitlines()
            result["warnings"] = [line for line in lines if re.search(
                r"warning|error|invalid|capacity|miss interactions|need to increase|buffer overflow|failed|fall.?back", line, re.I)]
            result["physics_device_log"] = [line.strip() for line in lines if "Physics Device:" in line]


class NativeScene:
    def __init__(self, spec, nworld, use_gpu, gravity):
        started = time.perf_counter()
        self.spec, self.nworld = spec, nworld
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.environments, self.actors, self.body_role, self.handles = [], [], [], []
        self.physics_seconds = 0.
        self.validation = []
        params = gymapi.SimParams()
        params.dt, params.substeps = .002, 1
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(*map(float, gravity))
        params.use_gpu_pipeline = False
        params.physx.use_gpu = bool(use_gpu)
        params.physx.solver_type = 1
        params.physx.num_position_iterations = int(spec["position_iterations"])
        params.physx.num_velocity_iterations = int(spec["velocity_iterations"])
        params.physx.contact_offset = float(spec["contact_offset"])
        params.physx.rest_offset = float(spec["rest_offset"])
        params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
        params.physx.bounce_threshold_velocity = 2*9.81*params.dt
        params.physx.max_depenetration_velocity = float(spec.get("max_depenetration_velocity", 10.))
        params.physx.num_threads = 0
        # Pin the SDK defaults as part of the experiment configuration rather
        # than silently inheriting them from an unspecified installation.
        defaults = spec.get("physx_parameters", {})
        for name, default in (("num_subscenes", 0), ("always_use_articulations", False),
                ("friction_correlation_distance", .025), ("friction_offset_threshold", .04),
                ("default_buffer_size_multiplier", 16.)):
            setattr(params.physx, name, defaults.get(name, default))
        params.physx.max_gpu_contact_pairs = max(int(defaults.get("max_gpu_contact_pairs", 8388608)), nworld*16384)
        self.sim = self.gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
        if self.sim is None:
            raise RuntimeError("Unable to create requested native PhysX scene")
        try:
            actual = self.gym.get_sim_params(self.sim)
            if bool(actual.physx.use_gpu) != bool(use_gpu) or actual.use_gpu_pipeline:
                raise RuntimeError("PhysX silently changed the requested device/pipeline")
            self.actual_configuration = dict(physx=scalar_properties(actual.physx),
                dt=float(actual.dt), substeps=int(actual.substeps), gravity=vector(actual.gravity).tolist(),
                up_axis=int(actual.up_axis), use_gpu_pipeline=bool(actual.use_gpu_pipeline),
                requested_physx_gpu=bool(use_gpu), enable_gyroscopic_forces=True,
                environment_origins="zero", environment_isolation="distinct collision group per query")
            plane = gymapi.PlaneParams()
            plane.normal = gymapi.Vec3(0., 0., 1.)
            plane.distance = 0.
            plane.static_friction = plane.dynamic_friction = float(spec["bodies"][0]["friction"])
            plane.restitution = 0.
            self.gym.add_ground(self.sim, plane)
            assets = []
            for body in spec["bodies"]:
                options = gymapi.AssetOptions()
                options.fix_base_link = False
                options.disable_gravity = False
                options.override_com = options.override_inertia = False
                options.vhacd_enabled = False
                options.convex_decomposition_from_submeshes = False
                options.replace_cylinder_with_capsule = False
                options.slices_per_cylinder = 96
                options.thickness = options.armature = 0.
                options.linear_damping = options.angular_damping = 0.
                options.max_linear_velocity = options.max_angular_velocity = 1e6
                options.enable_gyroscopic_forces = True
                path = Path(body["urdf"])
                asset = self.gym.load_asset(self.sim, str(path.parent), path.name, options)
                if asset is None:
                    raise RuntimeError("Failed to load native collision asset " + str(path))
                if self.gym.get_asset_rigid_shape_count(asset) != body["expected_shapes"]:
                    raise RuntimeError("PhysX changed convex-piece count; no fallback permitted")
                assets.append(asset)
            for world in range(nworld):
                # Collision groups isolate queries while identical origins avoid
                # float32 translation/cancellation differences against nworld=1.
                env = self.gym.create_env(self.sim, gymapi.Vec3(0., 0., 0.),
                                          gymapi.Vec3(0., 0., 0.), 1)
                if np.any(vector(self.gym.get_env_origin(env)) != 0.):
                    raise RuntimeError("Native environment origin must remain exactly zero")
                actors, role, handles = [], {-1: 0}, []
                for body, asset in zip(spec["bodies"], assets):
                    pose = gymapi.Transform()
                    # Keep preparation bodies separated; complete reset follows prepare_sim.
                    pose.p = gymapi.Vec3(.3*body["role_id"], 0., .5)
                    actor = self.gym.create_actor(env, asset, pose, body["name"], world, 0)
                    if actor < 0:
                        raise RuntimeError("Failed to create PhysX actor")
                    actors.append(actor)
                    handles.append(self.gym.get_actor_rigid_body_handle(env, actor, 0))
                    role[self.gym.get_actor_rigid_body_index(env, actor, 0, gymapi.DOMAIN_ENV)] = body["role_id"]
                    props = self.gym.get_actor_rigid_shape_properties(env, actor)
                    for prop in props:
                        prop.friction = float(body["friction"])
                        prop.rolling_friction = prop.torsion_friction = prop.restitution = 0.
                    self.gym.set_actor_rigid_shape_properties(env, actor, props)
                    checked_shapes = self.gym.get_actor_rigid_shape_properties(env, actor)
                    if any(abs(float(p.friction)-body["friction"]) > 1e-7 or
                           abs(float(p.rolling_friction))+abs(float(p.torsion_friction))+abs(float(p.restitution)) > 1e-10
                           for p in checked_shapes):
                        raise RuntimeError("PhysX collision material mismatch")
                    prop = self.gym.get_actor_rigid_body_properties(env, actor)[0]
                    mass = float(prop.mass)
                    com = vector(prop.com)
                    inertia = np.column_stack([vector(getattr(prop.inertia, key)) for key in ("x", "y", "z")])
                    expected = np.asarray(body["inertia"], float)
                    if not np.isclose(mass, body["mass"], rtol=2e-5, atol=1e-10):
                        raise RuntimeError("PhysX asset mass mismatch")
                    if not np.allclose(com, 0., rtol=0., atol=2e-7):
                        raise RuntimeError("PhysX asset COM is not at exported link origin")
                    if not np.allclose(inertia, expected, rtol=2e-4, atol=1e-10):
                        raise RuntimeError("PhysX asset inertia mismatch: actual=%s expected=%s" % (inertia, expected))
                    if world == 0:
                        self.validation.append(dict(body=body["name"], mass=mass, com=com.tolist(),
                            inertia=inertia.tolist(), shapes=len(props), expected_shapes=body["expected_shapes"],
                            friction=float(checked_shapes[0].friction), enable_gyroscopic_forces=True,
                            physics_configuration=self.actual_configuration))
                self.environments.append(env)
                self.actors.append(actors)
                self.handles.append(handles)
                self.body_role.append(role)
            self.gym.prepare_sim(self.sim)
            self.startup_seconds = time.perf_counter()-started
        except Exception:
            self.close()
            raise

    def _read_link(self, world, local):
        # Actor getters/setters use ENV coordinates, not the world/env origin.
        state = self.gym.get_actor_rigid_body_states(self.environments[world], self.actors[world][local],
                                                    gymapi.STATE_ALL)[0]
        p = vector(state["pose"]["p"])
        r = state["pose"]["r"]
        quat = np.array([float(r[x]) for x in ("w", "x", "y", "z")])
        quat /= np.linalg.norm(quat)
        velocity = vector(state["vel"]["linear"])
        omega = vector(state["vel"]["angular"])
        return p, quat, velocity, omega

    def read(self):
        qpos, qvel, links = np.zeros((self.nworld, 14)), np.zeros((self.nworld, 12)), []
        for world in range(self.nworld):
            local_links = {}
            for local, body in enumerate(self.spec["bodies"]):
                p, quat, velocity, omega = self._read_link(world, local)
                rotation = quaternion_matrix(quat)
                offset = rotation @ np.asarray(body["ipos"], float)
                q, v = 7*local, 6*local
                qpos[world, q:q+7] = np.r_[p-offset, quat]
                qvel[world, v:v+6] = np.r_[velocity-np.cross(omega, offset), rotation.T @ omega]
                local_links[body["role_id"]] = (p, quat, velocity, omega)
            links.append(local_links)
        if not np.isfinite(np.r_[qpos.reshape(-1), qvel.reshape(-1)]).all():
            raise RuntimeError("Nonfinite native PhysX state")
        return qpos, qvel, links

    def reset(self, qpos, qvel):
        expected_world = []
        for world, (poses, velocities) in enumerate(zip(qpos, qvel)):
            local_expected = {}
            for local, body in enumerate(self.spec["bodies"]):
                q, v = 7*local, 6*local
                quat = np.asarray(poses[q+3:q+7], float)
                rotation = quaternion_matrix(quat)
                quat = quat/np.linalg.norm(quat)
                offset = rotation @ np.asarray(body["ipos"], float)
                omega = rotation @ velocities[v+3:v+6]
                p = poses[q:q+3]+offset
                velocity = velocities[v:v+3]+np.cross(omega, offset)
                local_expected[body["role_id"]] = (velocity.copy(), omega.copy())
                state = self.gym.get_actor_rigid_body_states(self.environments[world],
                    self.actors[world][local], gymapi.STATE_ALL)
                state["pose"]["p"][0] = tuple(map(float, p))
                state["pose"]["r"][0] = tuple(map(float, quat[[1, 2, 3, 0]]))
                state["vel"]["linear"][0] = tuple(map(float, velocity))
                state["vel"]["angular"][0] = tuple(map(float, omega))
                if not self.gym.set_actor_rigid_body_states(self.environments[world],
                        self.actors[world][local], state, gymapi.STATE_ALL):
                    raise RuntimeError("PhysX rejected native query state")
            expected_world.append(local_expected)
        read_q, read_v, links = self.read()
        position_difference = np.max(np.abs(read_q[:, [0, 1, 2, 7, 8, 9]]-qpos[:, [0, 1, 2, 7, 8, 9]]))
        if position_difference > 3e-6:
            raise RuntimeError("Initial native PhysX position round-trip mismatch")
        # Verify the native quantities actually uploaded: COM linear velocity
        # and world angular velocity. Comparing body-angular components after
        # a second quaternion rotation contaminates near-zero components by
        # eps32*|omega| (e.g. 72 rad/s ball spin), despite a correct native state.
        linear_error, angular_error = 0., 0.
        for world in range(self.nworld):
            for body in self.spec["bodies"]:
                role = body["role_id"]
                expected_linear, expected_angular = expected_world[world][role]
                _, _, actual_linear, actual_angular = links[world][role]
                linear_error = max(linear_error, float(np.max(np.abs(actual_linear-expected_linear))))
                angular_error = max(angular_error, float(np.max(np.abs(actual_angular-expected_angular))))
                if (not np.allclose(actual_linear, expected_linear, rtol=2e-5, atol=3e-6) or
                        not np.allclose(actual_angular, expected_angular, rtol=2e-5, atol=3e-6)):
                    raise RuntimeError("Initial native PhysX world COM/angular velocity round-trip mismatch")
        for start in (3, 10):
            normalized = qpos[:, start:start+4]/np.linalg.norm(qpos[:, start:start+4], axis=1)[:, None]
            if np.max(1-np.abs(np.sum(normalized*read_q[:, start:start+4], axis=1))) > 1e-9:
                raise RuntimeError("Initial native PhysX orientation mismatch")
        self.initial_readback = dict(qpos=read_q.tolist(), qvel=read_v.tolist(),
            max_position_roundtrip_m=float(position_difference),
            velocity_guard_frame="native world COM linear and world angular velocity",
            velocity_guard_atol=3e-6, velocity_guard_rtol=2e-5,
            max_world_linear_velocity_roundtrip_m_s=linear_error,
            max_world_angular_velocity_roundtrip_rad_s=angular_error,
            max_body_velocity_roundtrip_component=float(np.max(np.abs(read_v-qvel))))

    def step(self, wrench):
        for world, local_forces in enumerate(wrench):
            for local, body in enumerate(self.spec["bodies"]):
                force = local_forces[body["role_id"]]
                if np.any(force):
                    self.gym.apply_body_forces(self.environments[world], self.handles[world][local],
                        gymapi.Vec3(*map(float, force[:3])), gymapi.Vec3(*map(float, force[3:])), gymapi.ENV_SPACE)
        started = time.perf_counter()
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        elapsed = time.perf_counter()-started
        self.physics_seconds += elapsed
        return elapsed

    def contacts(self, links, include_force):
        output = []
        for world, env in enumerate(self.environments):
            contacts = []
            for row in self.gym.get_env_rigid_contacts(env):
                raw = {name: json_value(row[name]) for name in row.dtype.names}
                a, b = int(row["body0"]), int(row["body1"])
                if a not in self.body_role[world] or b not in self.body_role[world]:
                    raise RuntimeError("Native contact contains unrecognized environment body")
                ra, rb = self.body_role[world][a], self.body_role[world][b]
                for name in ("env0", "env1"):
                    if int(row[name]) not in (-1, world):
                        raise RuntimeError("Cross-environment PhysX contact detected")
                if ra == rb:
                    raise RuntimeError("Unexpected same-body/self-contact")
                # Native normal is force-on-native-body0. Canonical normal is
                # lower-role -> higher-role, force on the higher-role body.
                normal = vector(row["normal"])
                normal /= np.linalg.norm(normal)
                if ra < rb:
                    normal = -normal
                lower, higher = sorted((ra, rb))
                # Anchors are encoded in POST-step link poses, although the
                # force point and initialOverlap belong to initial geometry.
                role, field = (ra, "localPos0") if ra else (rb, "localPos1")
                post_p, post_q, _, _ = links[world][role]
                point = post_p + quaternion_matrix(post_q) @ vector(row[field])
                gap = float(row["initialOverlap"])
                # SDK RigidContact.friction is reported as zero when the
                # friction patch is inactive, even with verified .5 materials.
                # All scene materials are equal, so the configured coefficient
                # is the unambiguous effective PhysX pair coefficient.
                mu = float(self.spec["bodies"][0]["friction"])
                record = dict(body0=lower, body1=higher, normal_world=normal.tolist(),
                              point_world=point.tolist(), gap=gap, friction=mu,
                              include_margin=float(row["minDist"]), raw=raw)
                if include_force:
                    record["force_N"] = float(row["lambda"])
                else:
                    # Predictor snapshots retain raw collision geometry only.
                    record["raw"] = {key: value for key, value in raw.items()
                                     if key not in ("lambda", "lambdaFriction")}
                contacts.append(record)
            contacts.sort(key=lambda x: (x["body0"], x["body1"], *x["point_world"], *x["normal_world"], x["gap"]))
            output.append(contacts)
        return output

    def close(self):
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None


def point_velocity(spec, qpos, qvel, role, point):
    if role == 0:
        return np.zeros(3)
    local = role-1
    q, v = 7*local, 6*local
    rotation = quaternion_matrix(qpos[q+3:q+7])
    omega = rotation @ qvel[v+3:v+6]
    return qvel[v:v+3]+np.cross(omega, point-qpos[q:q+3])


def snapshot(spec, contacts, qpos, qvel, time_value, endpoint=False):
    result = dict(time=float(time_value), gap=[], position=[], frame=[], body=[], pair=[],
                  force=[], relative_velocity=[],
                  force_interval_s=[float(time_value), float(time_value+.002)],
                  endpoint_diagnostic=bool(endpoint),
                  force_components="normal native force; tangential force retained only in raw records")
    for c in contacts:
        if c["gap"] > 0. or 1 not in (c["body0"], c["body1"]):
            continue
        frame = contact_frame(c["normal_world"])
        point = np.asarray(c["point_world"])
        relative = point_velocity(spec, qpos, qvel, c["body1"], point)-point_velocity(
            spec, qpos, qvel, c["body0"], point)
        result["gap"].append(c["gap"])
        result["position"].append(c["point_world"])
        result["frame"].append(frame.tolist())
        result["body"].append([c["body0"], c["body1"]])
        result["pair"].append(1 if c["body1"] == 2 else 0)
        result["force"].append([c["force_N"], 0., 0., 0., 0., 0.])
        result["relative_velocity"].append([*(frame @ relative).tolist(), 0., 0., 0.])
    return result


def geometry(spec, qpos, qvel, use_gpu):
    started = time.perf_counter()
    scene = NativeScene(spec, len(qpos), use_gpu, gravity=[0., 0., 0.])
    try:
        scene.reset(qpos, np.zeros_like(qvel))
        scene.step(np.zeros((len(qpos), 3, 6)))
        _, _, links = scene.read()
        contacts = scene.contacts(links, include_force=False)
        return dict(contacts=contacts, initial_readback=scene.initial_readback,
                    diagnostic=dict(steps=1, dt=.002, zero_velocities=True, zero_gravity=True,
                                    zero_external_forces=True, integrated_state_discarded=True,
                                    native_forces_discarded=True),
                    asset_validation=scene.validation,
                    physics_configuration=scene.actual_configuration,
                    timings=dict(startup_seconds=scene.startup_seconds, physics_seconds=scene.physics_seconds,
                                 end_to_end_seconds=time.perf_counter()-started))
    finally:
        scene.close()


def reference(spec, qpos, qvel, wrench, use_gpu):
    started = time.perf_counter()
    scene = NativeScene(spec, len(qpos), use_gpu, gravity=spec["gravity"])
    endpoint_scene = None
    try:
        scene.reset(qpos, qvel)
        initial_readback = scene.initial_readback
        qs, vs = [qpos.copy()], [qvel.copy()]
        snapshots, raw_contacts = [[] for _ in qpos], [[] for _ in qpos]
        initial_time = scene.gym.get_sim_time(scene.sim)
        for step in range(10):
            scene.step(wrench)
            new_q, new_v, links = scene.read()
            contacts = scene.contacts(links, include_force=True)
            for world, rows in enumerate(contacts):
                snapshots[world].append(snapshot(spec, rows, qs[-1][world], vs[-1][world], .002*step))
                raw_contacts[world].append(rows)
            qs.append(new_q)
            vs.append(new_v)
        elapsed = scene.gym.get_sim_time(scene.sim)-initial_time
        if abs(elapsed-.02) > 1e-7:
            raise RuntimeError("PhysX reference did not advance exactly 10x2ms")
        # Preserve the evaluated endpoint before a separate diagnostic solve.
        final_q, final_v = qs[-1].copy(), vs[-1].copy()
        # The legacy SDK supports one live PhysX foundation per process.
        # Destroy the reference scene before creating its independent clone.
        scene.close()
        endpoint_scene = NativeScene(spec, len(qpos), use_gpu, gravity=spec["gravity"])
        endpoint_scene.reset(final_q, final_v)
        endpoint_scene.step(wrench)
        _, _, endpoint_links = endpoint_scene.read()
        endpoint_contacts = endpoint_scene.contacts(endpoint_links, include_force=True)
        for world, rows in enumerate(endpoint_contacts):
            snapshots[world].append(snapshot(spec, rows, final_q[world], final_v[world], .02, endpoint=True))
            raw_contacts[world].append(rows)
        return dict(qpos=final_q.tolist(), qvel=final_v.tolist(),
                    trajectory_qpos=np.stack(qs, axis=1).tolist(), trajectory_qvel=np.stack(vs, axis=1).tolist(),
                    snapshots=snapshots, raw_contacts=raw_contacts, initial_readback=initial_readback,
                    asset_validation=scene.validation, elapsed_seconds=float(elapsed),
                    physics_configuration=scene.actual_configuration,
                    diagnostic=dict(reference_steps=10, dt=.002, horizon=.02,
                        endpoint_separate_scene=True, endpoint_force_interval_s=[.02, .022],
                        endpoint_integrated_state_discarded=True, fresh_reference_scene=True),
                    timings=dict(startup_seconds=scene.startup_seconds,
                        physics_seconds=scene.physics_seconds,
                        endpoint_startup_seconds=endpoint_scene.startup_seconds,
                        endpoint_physics_seconds=endpoint_scene.physics_seconds,
                        end_to_end_seconds=time.perf_counter()-started))
    finally:
        scene.close()
        if endpoint_scene is not None:
            endpoint_scene.close()


def main():
    spec, use_gpu, max_batch = None, True, 16
    for line in sys.stdin:
        captured = None
        try:
            request = json.loads(line)
            cmd = request["cmd"]
            if cmd == "close":
                _PROTOCOL.write(json.dumps(dict(ok=True))+"\n")
                return
            if cmd == "init":
                spec, use_gpu, max_batch = request["spec"], request["use_gpu"], request["batch_size"]
                for filename, expected in spec["asset_sha256"].items():
                    if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                        raise ValueError("PhysX exported asset hash mismatch: " + filename)
                reply = dict(worker_python=sys.version, device="cuda:0" if use_gpu else "cpu",
                             physx_gpu=bool(use_gpu), gpu_pipeline=False, sdk="Isaac Gym Preview 4",
                             fresh_scenes=True, max_batch=max_batch,
                             runtime_fingerprint=runtime_fingerprint(),
                             physx_sdk_defaults=scalar_properties(gymapi.SimParams().physx),
                             configured_parameters=spec.get("physx_parameters", {}))
            elif cmd in ("geometry", "reference"):
                if spec is None:
                    raise RuntimeError("Worker must be initialized first")
                qpos, qvel, wrench = (np.asarray(request[key], float) for key in ("qpos", "qvel", "wrench"))
                if (qpos.ndim != 2 or qpos.shape[1] != 14 or qvel.shape != (len(qpos), 12)
                        or wrench.shape != (len(qpos), 3, 6) or not 0 < len(qpos) <= max_batch
                        or not np.isfinite(np.r_[qpos.ravel(), qvel.ravel(), wrench.ravel()]).all()):
                    raise ValueError("Invalid native query batch")
                with capture_native_log() as captured:
                    reply = geometry(spec, qpos, qvel, use_gpu) if cmd == "geometry" else reference(
                        spec, qpos, qvel, wrench, use_gpu)
                if use_gpu and any("cpu" in line.lower() for line in captured["physics_device_log"]):
                    raise RuntimeError("Native PhysX log reports CPU despite requested GPU")
            else:
                raise ValueError("Unknown native worker command " + str(cmd))
            reply["ok"] = True
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = dict(ok=False, error=type(exc).__name__+": "+str(exc))
        if captured is not None:
            reply["native_warnings"] = captured.get("warnings", [])
            reply["physics_device_log"] = captured.get("physics_device_log", [])
        _PROTOCOL.write(json.dumps(reply, allow_nan=False)+"\n")
        _PROTOCOL.flush()


if __name__ == "__main__":
    main()
