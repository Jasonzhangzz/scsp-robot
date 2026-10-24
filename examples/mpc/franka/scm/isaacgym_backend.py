"""Isaac Gym (PhysX) adapter used by the one-step contact benchmark.

The benchmark scene contains one free rigid body and a plane.  The adapter
rebuilds the SAME body that MuJoCo simulates: the compiled MuJoCo geometry
(primitive or convex mesh hull), the MuJoCo mass, centre of mass and inertia
tensor, the same friction coefficient, zero restitution, ``dt = h`` and one
PhysX sub-step per prediction.  The external wrench is applied through PhysX
(``apply_body_forces`` at the centre of mass, world frame), exactly as MuJoCo's
``xfrc_applied``.

State conventions that differ between the two engines are converted here:

* MuJoCo free-joint ``qvel[3:6]`` is the body-frame angular velocity; PhysX
  reports the world-frame angular velocity.
* MuJoCo ``qpos[:3]`` / ``qvel[:3]`` are the body-frame origin; the Isaac actor
  link frame is placed at the centre of mass so that PhysX pose/velocity are
  unambiguous (``p_com``, ``v_com``) and mapped back to the body frame.
* MuJoCo quaternions are wxyz; Isaac Gym quaternions are xyzw.

The legacy Isaac Gym SDK only exists for Python 3.8 (MuJoCo <= 3.2.3), while
the benchmark reference is the current MuJoCo.  ``IsaacGymBackend`` therefore
needs no MuJoCo: it is built from a plain ``spec`` dictionary
(``object_spec(model, ...)``) and can run inside a worker process
(``isaacgym_worker.py``) driven by ``IsaacGymProxy`` from the main process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

PRIMITIVES = ("box", "sphere", "cylinder", "capsule", "ellipsoid")
WORKER_PYTHON_ENV = "SCSP_ISAACGYM_PYTHON"
_DEFAULT_WORKER_PYTHONS = (Path.home() / "miniconda3/envs/acc/bin/python",
                           Path.home() / "anaconda3/envs/acc/bin/python")


def quat_to_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _rpy_from_matrix(rotation):
    """URDF roll-pitch-yaw (extrinsic xyz) of a rotation matrix."""
    sy = -rotation[2, 0]
    pitch = np.arcsin(np.clip(sy, -1., 1.))
    if abs(sy) < 1 - 1e-9:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:  # gimbal lock
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.
    return roll, pitch, yaw


def object_spec(model, shape, *, friction=.5, body=1, geom_name="object"):
    """Everything PhysX needs about the MuJoCo body, as JSON-serialisable data."""
    import mujoco
    geom = model.geom(geom_name).id
    spec = {"shape": shape, "friction": float(friction), "timestep": float(model.opt.timestep),
            "gravity": [float(v) for v in model.opt.gravity],
            "mass": float(model.body_mass[body]),
            "ipos": [float(v) for v in model.body_ipos[body]],
            "iquat": [float(v) for v in model.body_iquat[body]],
            "inertia": [float(v) for v in model.body_inertia[body]],
            "geom_pos": [float(v) for v in model.geom_pos[geom]],
            "geom_quat": [float(v) for v in model.geom_quat[geom]],
            "geom_size": [float(v) for v in model.geom_size[geom]]}
    if shape == "mesh":
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            raise ValueError("object geom is not a mesh")
        mesh_id = int(model.geom_dataid[geom])
        adr, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        fadr, fcount = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        spec["mesh_vertices"] = np.asarray(model.mesh_vert[adr:adr + count], dtype=float).tolist()
        spec["mesh_faces"] = np.asarray(model.mesh_face[fadr:fadr + fcount], dtype=int).tolist()
    return spec


def shape_of(model, geom_name="object"):
    import mujoco
    geom_type = int(model.geom_type[model.geom(geom_name).id])
    shape = {int(mujoco.mjtGeom.mjGEOM_BOX): "box",
             int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
             int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
             int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
             int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
             int(mujoco.mjtGeom.mjGEOM_MESH): "mesh"}.get(geom_type)
    if shape is None:
        raise ValueError(f"Isaac Gym backend does not support MuJoCo geom type {geom_type}")
    return shape


def worker_python():
    """Interpreter with the legacy Isaac Gym SDK, or None."""
    candidate = os.environ.get(WORKER_PYTHON_ENV, "")
    if candidate:
        return candidate if Path(candidate).exists() else None
    for path in _DEFAULT_WORKER_PYTHONS:
        if path.exists():
            return str(path)
    return None


def isaacgym_importable():
    try:
        import isaacgym  # noqa: F401
    except Exception:
        return False
    return True


def create_backend(model, shape, *, friction=.5, mass=None, headless=True):
    """In-process PhysX when the SDK is importable here, else an acc worker."""
    spec = object_spec(model, shape, friction=friction)
    if mass is not None and abs(float(mass) - spec["mass"]) > 1e-9:
        raise ValueError(f"Isaac Gym mass {mass} differs from MuJoCo body mass {spec['mass']}")
    if isaacgym_importable():
        return IsaacGymBackend(spec, headless=headless)
    python = worker_python()
    if python is None:
        raise RuntimeError(
            "Isaac Gym backend requested but isaacgym is unavailable in this interpreter and no "
            f"worker interpreter was found. Set {WORKER_PYTHON_ENV}=<python with isaacgym> "
            "(for example the conda `acc` environment) or omit isaacgym.")
    return IsaacGymProxy(spec, python=python, headless=headless)


class IsaacGymBackend:
    """One-world PhysX backend with MuJoCo-compatible state readback."""

    def __init__(self, spec, *, headless=True, position_iterations=8,
                 velocity_iterations=2, contact_offset=.002):
        try:
            from isaacgym import gymapi
        except Exception as exc:  # pragma: no cover - depends on optional SDK
            raise RuntimeError(
                "Isaac Gym backend requested but isaacgym is unavailable. "
                "Use `conda run -n acc ...` or omit isaacgym."
            ) from exc
        self.gymapi = gymapi
        self.spec = spec
        self.shape = spec["shape"]
        self.timestep = float(spec["timestep"])
        self.mass = float(spec["mass"])
        friction = float(spec["friction"])
        # Centre of mass in the MuJoCo body frame; the Isaac link frame sits there.
        self.ipos = np.asarray(spec["ipos"], dtype=float)
        self.iquat = np.asarray(spec["iquat"], dtype=float)
        self.inertia = np.asarray(spec["inertia"], dtype=float)
        self.gym = gymapi.acquire_gym()
        params = gymapi.SimParams()
        params.dt = self.timestep
        params.substeps = 1
        params.up_axis = gymapi.UP_AXIS_Z
        params.gravity = gymapi.Vec3(*[float(v) for v in spec["gravity"]])
        params.use_gpu_pipeline = False
        params.physx.use_gpu = False
        params.physx.solver_type = 1
        params.physx.num_position_iterations = int(position_iterations)
        params.physx.num_velocity_iterations = int(velocity_iterations)
        params.physx.contact_offset = float(contact_offset)
        params.physx.rest_offset = 0.
        params.physx.bounce_threshold_velocity = 2 * 9.81 * params.dt
        params.physx.max_depenetration_velocity = 100.
        params.physx.num_threads = 0
        self.sim = self.gym.create_sim(0, -1 if headless else 0, gymapi.SIM_PHYSX, params)
        if self.sim is None:
            raise RuntimeError("Isaac Gym could not create a PhysX simulation")
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0., 0., 1.)
        plane.distance = 0.
        plane.static_friction = friction
        plane.dynamic_friction = friction
        plane.restitution = 0.
        self.gym.add_ground(self.sim, plane)
        self.env = self.gym.create_env(self.sim, gymapi.Vec3(-2, -2, -2),
                                       gymapi.Vec3(2, 2, 2), 1)
        origin = self.gym.get_env_origin(self.env)
        self.env_origin = np.array([origin.x, origin.y, origin.z], dtype=float)
        self.fallback_geometry = False
        self._workdir = tempfile.TemporaryDirectory(prefix="isaacgym_object_")
        asset = self._load_asset()
        self.actor = self.gym.create_actor(self.env, asset, gymapi.Transform(), "object", 0, 0)
        props = self.gym.get_actor_rigid_shape_properties(self.env, self.actor)
        for prop in props:
            prop.friction = friction
            prop.rolling_friction = 0.
            prop.torsion_friction = 0.
            prop.restitution = 0.
        self.gym.set_actor_rigid_shape_properties(self.env, self.actor, props)
        # Verify the inertial model PhysX actually uses.
        body_props = self.gym.get_actor_rigid_body_properties(self.env, self.actor)[0]
        self.physx_mass = float(body_props.mass)
        com = body_props.com
        self.physx_com = np.array([com.x, com.y, com.z], dtype=float)
        if abs(self.physx_mass - self.mass) > 1e-6 * max(1., self.mass):
            raise RuntimeError(f"PhysX mass {self.physx_mass} differs from MuJoCo mass {self.mass}")
        if np.linalg.norm(self.physx_com) > 1e-6:
            raise RuntimeError(f"PhysX centre of mass {self.physx_com} is not at the link origin")
        self.body_handle = self.gym.get_actor_rigid_body_handle(self.env, self.actor, 0)
        self.last_step_seconds = None
        self.last_contacts = None
        self.collision = self._collision_description

    # ------------------------------------------------------------------ assets
    def _load_asset(self):
        """URDF with MuJoCo's compiled geometry and exact inertial parameters."""
        gymapi = self.gymapi
        spec = self.spec
        shape = self.shape
        geom_pos = np.asarray(spec["geom_pos"], dtype=float)
        rotation = quat_to_matrix(spec["geom_quat"])
        # Collision frame expressed in the link (= centre of mass) frame.
        collision_origin = geom_pos - self.ipos
        if shape == "mesh":
            vertices = np.asarray(spec["mesh_vertices"], dtype=float).reshape(-1, 3)
            faces = np.asarray(spec["mesh_faces"], dtype=int).reshape(-1, 3)
            # Bake the compiled geom pose into the vertices (link frame).
            vertices = vertices @ rotation.T + collision_origin
            self._write_obj(Path(self._workdir.name) / "object.obj", vertices, faces)
            geometry = '<mesh filename="object.obj" scale="1 1 1"/>'
            collision = f'<origin xyz="0 0 0" rpy="0 0 0"/><geometry>{geometry}</geometry>'
            self._collision_description = "PhysX convex hull of the MuJoCo compiled mesh (single hull)"
        else:
            size = np.asarray(spec["geom_size"], dtype=float)
            if shape == "box":
                geometry = f'<box size="{2 * size[0]} {2 * size[1]} {2 * size[2]}"/>'
            elif shape == "sphere":
                geometry = f'<sphere radius="{size[0]}"/>'
            elif shape == "cylinder":
                geometry = f'<cylinder radius="{size[0]}" length="{2 * size[1]}"/>'
            elif shape in ("capsule", "ellipsoid"):
                # URDF has neither primitive: tessellate the MuJoCo surface and
                # let PhysX build its convex hull (recorded in metadata).
                vertices, faces = self._revolved_mesh(shape, size)
                self._write_obj(Path(self._workdir.name) / "object.obj", vertices, faces)
                geometry = '<mesh filename="object.obj" scale="1 1 1"/>'
                self.fallback_geometry = True
            else:
                raise RuntimeError(f"Isaac Gym adapter does not support shape {shape}")
            rpy = _rpy_from_matrix(rotation)
            collision = (f'<origin xyz="{collision_origin[0]} {collision_origin[1]} {collision_origin[2]}" '
                         f'rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/><geometry>{geometry}</geometry>')
            self._collision_description = f"PhysX {shape} primitive" if not self.fallback_geometry else \
                f"PhysX convex hull of a tessellated MuJoCo {shape}"
        irpy = _rpy_from_matrix(quat_to_matrix(self.iquat))
        ixx, iyy, izz = self.inertia
        urdf = f'''<?xml version="1.0"?>
<robot name="object">
  <link name="object">
    <inertial>
      <origin xyz="0 0 0" rpy="{irpy[0]} {irpy[1]} {irpy[2]}"/>
      <mass value="{self.mass}"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
    <collision>{collision}</collision>
  </link>
</robot>
'''
        (Path(self._workdir.name) / "object.urdf").write_text(urdf)
        options = gymapi.AssetOptions()
        options.fix_base_link = False
        options.disable_gravity = False
        options.override_com = False
        options.override_inertia = False
        options.use_mesh_materials = False
        options.vhacd_enabled = False
        options.convex_decomposition_from_submeshes = False
        options.thickness = 0.
        options.armature = 0.
        options.linear_damping = 0.
        options.angular_damping = 0.
        options.max_angular_velocity = 1e6
        options.max_linear_velocity = 1e6
        asset = self.gym.load_asset(self.sim, self._workdir.name, "object.urdf", options)
        if asset is None:
            raise RuntimeError("Isaac Gym failed to load the object URDF")
        return asset

    @staticmethod
    def _write_obj(path, vertices, faces):
        with Path(path).open("w") as stream:
            stream.write("# MuJoCo compiled geometry in the centre-of-mass frame\n")
            for v in vertices:
                stream.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
            for f in faces:
                stream.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")

    @staticmethod
    def _revolved_mesh(shape, size, rings=24, segments=32):
        """Closed triangle mesh of a MuJoCo capsule/ellipsoid (local frame)."""
        vertices, faces = [], []
        if shape == "capsule":
            radius, half = size[0], size[1]
            upper = [(radius * np.cos(a), half + radius * np.sin(a))
                     for a in np.linspace(0., np.pi / 2, rings + 1)]
            profile = [(r, -z) for r, z in reversed(upper)] + upper
        else:
            profile = [(np.cos(t), np.sin(t)) for t in np.linspace(-np.pi / 2, np.pi / 2, 2 * rings + 1)]
        rows = []
        for r, z in profile:
            rows.append(len(vertices))
            for k in range(segments):
                angle = 2 * np.pi * k / segments
                if shape == "capsule":
                    vertices.append((r * np.cos(angle), r * np.sin(angle), z))
                else:
                    vertices.append((size[0] * r * np.cos(angle), size[1] * r * np.sin(angle), size[2] * z))
        for base, nxt in zip(rows[:-1], rows[1:]):
            for k in range(segments):
                k1 = (k + 1) % segments
                faces.append((base + k, nxt + k, nxt + k1))
                faces.append((base + k, nxt + k1, base + k1))
        return np.asarray(vertices), np.asarray(faces)

    # ------------------------------------------------------------------- state
    def _write_state(self, qpos, qvel):
        """Body-frame MuJoCo state -> PhysX link (COM) state in env space."""
        gymapi = self.gymapi
        qpos = np.asarray(qpos, dtype=float)
        qvel = np.asarray(qvel, dtype=float)
        rotation = quat_to_matrix(qpos[3:7])
        offset = rotation @ self.ipos
        omega_world = rotation @ qvel[3:6]
        p_com = qpos[:3] + offset
        v_com = qvel[:3] + np.cross(omega_world, offset)
        q = qpos[3:7] / np.linalg.norm(qpos[3:7])
        state = self.gym.get_actor_rigid_body_states(self.env, self.actor, gymapi.STATE_ALL)
        state[0]["pose"]["p"] = tuple(np.float32(p_com + self.env_origin))
        state[0]["pose"]["r"] = (np.float32(q[1]), np.float32(q[2]), np.float32(q[3]), np.float32(q[0]))
        state[0]["vel"]["linear"] = tuple(np.float32(v_com))
        state[0]["vel"]["angular"] = tuple(np.float32(omega_world))
        if not self.gym.set_actor_rigid_body_states(self.env, self.actor, state, gymapi.STATE_ALL):
            raise RuntimeError("Isaac Gym rejected the rigid body state")

    def _read_state(self):
        gymapi = self.gymapi
        out = self.gym.get_actor_rigid_body_states(self.env, self.actor, gymapi.STATE_ALL)[0]
        p, r, vl, va = out["pose"]["p"], out["pose"]["r"], out["vel"]["linear"], out["vel"]["angular"]
        p_com = np.array([p["x"], p["y"], p["z"]], dtype=float) - self.env_origin
        quat = np.array([r["w"], r["x"], r["y"], r["z"]], dtype=float)
        quat /= np.linalg.norm(quat)
        v_com = np.array([vl["x"], vl["y"], vl["z"]], dtype=float)
        omega_world = np.array([va["x"], va["y"], va["z"]], dtype=float)
        rotation = quat_to_matrix(quat)
        offset = rotation @ self.ipos
        qpos = np.r_[p_com - offset, quat]
        qvel = np.r_[v_com - np.cross(omega_world, offset), rotation.T @ omega_world]
        return qpos, qvel

    def predict(self, qpos, qvel, wrench):
        """One PhysX step from (qpos, qvel) under MuJoCo's world-frame COM wrench."""
        gymapi = self.gymapi
        started = time.perf_counter()
        self._write_state(qpos, qvel)
        wrench = np.asarray(wrench, dtype=float).reshape(6)
        if np.any(wrench):
            # Applied for the immediate time step only; the call returns None.
            self.gym.apply_body_forces(self.env, self.body_handle,
                                       gymapi.Vec3(*wrench[:3]), gymapi.Vec3(*wrench[3:]),
                                       gymapi.ENV_SPACE)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        qpos, qvel = self._read_state()
        self.last_step_seconds = time.perf_counter() - started
        return qpos, qvel, self.last_step_seconds

    # MjData-facing interface shared with IsaacGymProxy
    def reset(self, data):
        """Load the shared query state; PhysX has no persistent warm start here."""
        self._write_state(data.qpos, data.qvel)

    def step(self, data):
        qpos, qvel, _ = self.predict(data.qpos, data.qvel, data.xfrc_applied[1])
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.time += self.timestep

    def close(self):
        if getattr(self, "sim", None) is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None
        workdir = getattr(self, "_workdir", None)
        if workdir is not None:
            workdir.cleanup()
            self._workdir = None


class IsaacGymProxy:
    """Same interface as IsaacGymBackend; PhysX runs in ``isaacgym_worker.py``.

    One worker process per backend. Requests and replies are JSON lines over
    the worker's stdin/stdout; the worker's stderr (PhysX banner) is kept.
    """

    def __init__(self, spec, *, python, headless=True):
        script = Path(__file__).with_name("isaacgym_worker.py")
        self.timestep = float(spec["timestep"])
        self.shape = spec["shape"]
        env = dict(os.environ)
        env.pop("SCSP_PLANNER_ONLY", None)
        self.process = subprocess.Popen([python, "-u", str(script)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
                                        bufsize=1, env=env)
        reply = self._call({"cmd": "init", "spec": spec, "headless": bool(headless)})
        self.fallback_geometry = bool(reply["fallback_geometry"])
        self.collision = reply["collision"]
        self.physx_mass = float(reply["physx_mass"])
        self.worker_python = python
        self.last_step_seconds = None
        self.last_contacts = None

    def _call(self, request):
        if self.process.poll() is not None:
            raise RuntimeError("Isaac Gym worker exited unexpectedly")
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Isaac Gym worker closed its output; see stderr for the PhysX error")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"Isaac Gym worker: {reply['error']}")
        return reply

    def predict(self, qpos, qvel, wrench):
        reply = self._call({"cmd": "step", "qpos": np.asarray(qpos, dtype=float).tolist(),
                            "qvel": np.asarray(qvel, dtype=float).tolist(),
                            "wrench": np.asarray(wrench, dtype=float).reshape(6).tolist()})
        self.last_step_seconds = float(reply["seconds"])
        return np.asarray(reply["qpos"], dtype=float), np.asarray(reply["qvel"], dtype=float), self.last_step_seconds

    def reset(self, data):
        self._call({"cmd": "reset", "qpos": np.asarray(data.qpos, dtype=float).tolist(),
                    "qvel": np.asarray(data.qvel, dtype=float).tolist()})

    def step(self, data):
        qpos, qvel, _ = self.predict(data.qpos, data.qvel, data.xfrc_applied[1])
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.time += self.timestep

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        try:
            if process.poll() is None:
                process.stdin.write(json.dumps({"cmd": "close"}) + "\n")
                process.stdin.flush()
                process.wait(timeout=30)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            process.kill()
        self.process = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
