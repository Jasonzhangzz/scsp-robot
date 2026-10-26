"""Shared-state, 20-ms predictions; reference-only labels and diagnostics."""
from __future__ import annotations

import copy
from contextlib import contextmanager
import ctypes
from dataclasses import asdict, dataclass
import os
import sys
import tempfile

import mujoco
import numpy as np

from examples.mpc.franka.scm.benchmark_utils import Backend
from examples.mpc.franka.scm.fidelity_metrics import contact_observations
from examples.mpc.fingertips.test.collision_prediction_assets import (
    object_vertices, rotation, support_height)

MODES = ("separation", "contact_establishment", "sticking", "sliding")
ALL_MODES = MODES + ("mixed_transition", "invalid_reference", "sampling_failure")
METHODS = ("scm", "comfree", "mjwarp")
GROUPS = ("0", "1", "2", ">=3")
FORCES = (0., .05, .1, .2, .5)
SPEEDS = (0., .05, .2, .5)
METRICS = ("position_error_mm", "orientation_error_deg", "pose_error",
           "quaternion_error", "linear_velocity_error_m_s", "angular_velocity_error_rad_s")
RELATIVE_METRICS = ("relative_position_error", "relative_orientation_error", "relative_pose_error",
                    "relative_linear_velocity_error", "relative_angular_velocity_error")
RELATIVE_FLOORS = dict(position_m=1e-6, orientation_rad=1e-4, pose=1e-4,
                       linear_velocity_m_s=1e-4, angular_velocity_rad_s=1e-3)


def relative_errors(absolute, reference_motion, reference_velocity):
    """Dimensionless ratios; near-zero reference denominators stay undefined.

    Pose denominators are reference displacement/rotation over the query,
    never distance/orientation relative to the arbitrary world origin.
    Velocity denominators are magnitudes of the reference terminal velocities.
    """
    denominators = (reference_motion["position_error_mm"]/1000,
                    np.radians(reference_motion["orientation_error_deg"]), reference_motion["pose_error"],
                    np.linalg.norm(reference_velocity[:3]), np.linalg.norm(reference_velocity[3:]))
    numerators = (absolute["position_error_mm"]/1000,
                  np.radians(absolute["orientation_error_deg"]), absolute["pose_error"],
                  absolute["linear_velocity_error_m_s"], absolute["angular_velocity_error_rad_s"])
    result = {}
    for metric, numerator, denominator, floor in zip(RELATIVE_METRICS, numerators, denominators, RELATIVE_FLOORS.values()):
        result[metric] = float(numerator/denominator) if denominator >= floor else None
        result[metric + "_denominator"] = float(denominator)
    return result


@contextmanager
def native_diagnostics():
    """Collect CUDA printf warnings after synchronization, without patching physics.

    CUDA's printf bypasses Python redirect_stdout. Calls are serialized in this
    single-query runner; restore fd 1 even when a backend raises an exception.
    Each caller synchronizes the device before leaving this context.
    """
    warnings = []
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    with tempfile.TemporaryFile() as capture:
        original = os.dup(1)
        os.dup2(capture.fileno(), 1)
        try:
            yield warnings
        finally:
            sys.stdout.flush()
            libc.fflush(None)
            os.dup2(original, 1)
            os.close(original)
            capture.seek(0)
            output = capture.read().decode("utf-8", errors="replace")
            warnings.extend(line for line in output.splitlines() if "Warning:" in line)


@dataclass(frozen=True)
class Settings:
    horizon: float = .02
    reference_dt: float = .002
    reference_steps: int = 10
    friction: float = .5
    primitive_mass: float = .01
    ball_mass: float = .01
    ball_radius: float = .01
    force_threshold: float = 1e-4
    slip_threshold: float = 1e-3
    scm_temperature: float = 1e-4
    scm_regularization: float = 1e-8
    scm_sweeps: int = 3
    contact_stiffness: float = .1
    contact_damping: float = .001
    nconmax: int = 2048
    njmax: int = 16384
    device: str = "cuda:0"

    def to_dict(self):
        return asdict(self)


def contact_group(count):
    return str(count) if count < 3 else ">=3"


def classify_contacts(snapshots, force_threshold=1e-4, slip_threshold=1e-3):
    """Interface-level events, not changing manifold-point identities.

    Snapshot forces are native MJWarp forces; velocities are incoming velocities
    at that snapshot's solve pose. No CPU constraint forces enter the labels.
    """
    active = []
    slip = []
    counts = []
    for snapshot in snapshots:
        forces = np.asarray(snapshot["force"], float).reshape(-1, 6)
        pairs = np.asarray(snapshot["pair"], int)
        loaded = forces[:, 0] > force_threshold
        active.append([bool(np.any(loaded & (pairs == pair))) for pair in (0, 1)])
        velocity = np.asarray(snapshot["relative_velocity"], float).reshape(-1, 6)
        slip.extend(np.linalg.norm(velocity[loaded, 1:3], axis=1).tolist())
        counts.append([int(np.count_nonzero(pairs == pair)) for pair in (0, 1)])
    active = np.asarray(active, bool)
    change = np.diff(active.astype(int), axis=0)
    established, separated = bool(np.any(change > 0)), bool(np.any(change < 0))
    if established and separated:
        mode = "mixed_transition"
    elif established:
        mode = "contact_establishment"
    elif separated or not np.any(active):
        mode = "separation"
    else:
        mode = "sliding" if max(slip, default=0.) > slip_threshold else "sticking"
    total = np.sum(counts, axis=1)
    return dict(mode=mode, established=established, separated=separated,
                free_flight=not bool(np.any(active)), max_slip_m_s=max(slip, default=0.),
                ncontact_start=int(total[0]), ncontact_end=int(total[-1]),
                ncontact_max=int(np.max(total)), ncontact_floor_start=counts[0][0],
                ncontact_ball_start=counts[0][1], contact_group=contact_group(int(total[0])))


def pose_errors(predicted, reference, length_scale):
    predicted, reference = np.asarray(predicted, float), np.asarray(reference, float)
    if (predicted.shape != (7,) or reference.shape != (7,) or length_scale <= 0
            or not np.isfinite(np.r_[predicted, reference, length_scale]).all()):
        raise ValueError("Pose and positive length scale must be finite")
    q, r = predicted[3:], reference[3:]
    if min(np.linalg.norm(q), np.linalg.norm(r)) < 1e-12:
        raise ValueError("Invalid zero quaternion")
    cosine = float(np.clip(abs(np.dot(q/np.linalg.norm(q), r/np.linalg.norm(r))), 0, 1))
    if 1-cosine < 1e-15:
        cosine = 1.
    theta = 2*np.arccos(cosine)
    distance = float(np.linalg.norm(predicted[:3] - reference[:3]))
    return dict(position_error_mm=1000*distance, orientation_error_deg=float(np.degrees(theta)),
                pose_error=float(np.hypot(distance/length_scale, theta)),
                quaternion_error=max(0., 1-cosine*cosine))


def make_data(model, qpos=None, qvel=None, wrench=None):
    data = mujoco.MjData(model)
    if qpos is not None:
        data.qpos[:] = qpos
    if qvel is not None:
        data.qvel[:] = qvel
    if wrench is not None:
        data.xfrc_applied[:] = wrench
    mujoco.mj_forward(model, data)
    data.qacc_warmstart[:] = 0.
    return data


def world_velocity(model, data):
    # mj_objectVelocity returns [world angular, COM linear], at each state's
    # own orientation. This also handles offset body inertial frames.
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                            model.body("object").id, velocity, 0)
    return np.r_[velocity[3:], velocity[:3]]


def native_reference_contacts(data):
    """Decode global contacts using *every* world-local pyramidal row index.

    In the pinned MJWarp constraint kernel each edge obtains its own atomic
    address. Edges of a single contact need not occupy consecutive rows (even
    with nworld=1). A contiguous slice starting at edge zero silently mixes
    forces from unrelated contacts. Keep this fix local to the new benchmark.
    """
    count = int(data.nacon.numpy()[0])
    collisions = int(data.ncollision.numpy()[0])
    constraints = data.nefc.numpy()
    if count > data.naconmax or collisions > data.naconmax or np.any(constraints > data.njmax):
        raise RuntimeError(f"MJWarp buffer overflow: contacts={count}, collisions={collisions}, "
                           f"max_constraints={constraints.max()}")
    contact = data.contact
    worlds = contact.worldid.numpy()[:count].astype(int)
    addresses = contact.efc_address.numpy()[:count].astype(int)
    dimensions = contact.dim.numpy()[:count]
    friction = contact.friction.numpy()[:count]
    coefficients = data.efc.force.numpy()
    force = np.zeros((count, 6))
    for i, dimension in enumerate(dimensions):
        rows = 1 if dimension == 1 else 2*(int(dimension)-1)
        indices = addresses[i, :rows]
        if np.all(indices < 0):
            continue
        world = worlds[i]
        if (len(indices) != rows or np.any(indices < 0)
                or np.any(indices >= constraints[world])):
            raise RuntimeError("Incomplete or out-of-range native contact constraint addresses")
        edges = coefficients[world, indices]
        force[i, 0] = edges.sum()
        if dimension > 1:
            edges = edges.reshape(-1, 2)
            force[i, 1:int(dimension)] = (edges[:, 0]-edges[:, 1])*friction[i, :int(dimension)-1]
    native = dict(gap=contact.dist.numpy()[:count], position=contact.pos.numpy()[:count],
                  frame=contact.frame.numpy()[:count], geom=contact.geom.numpy()[:count],
                  friction=friction, dim=dimensions, force=force)
    return native, worlds, addresses[:, 0]


def reference_snapshot(model, pose, native, addresses, time):
    """A world-local native snapshot; global batch buffers are split upstream."""
    bodies = model.geom_bodyid[np.asarray(native["geom"], int)]
    obj, ball = model.body("object").id, model.body("ball").id
    keep = (np.any(bodies == obj, axis=1) & (np.asarray(addresses) >= 0)
            & (np.asarray(native["gap"]) <= 0.))
    native = {key: np.asarray(value)[keep].copy() for key, value in native.items()}
    other = np.where(bodies[keep, 0] == obj, bodies[keep, 1], bodies[keep, 0])
    observations = contact_observations(model, pose, pose.qvel, native)
    return dict(time=float(time), pair=np.where(other == ball, 1, 0).tolist(),
                relative_velocity=observations["relative_velocity"].tolist(),
                **{key: value.tolist() for key, value in native.items()})


class Experiment:
    def __init__(self, model_path, info, settings=Settings()):
        self.info, self.settings = info, settings
        reference = mujoco.MjModel.from_xml_path(str(model_path))
        self.models, self.backends = {}, {}
        for method in METHODS:
            model = copy.copy(reference)
            model.opt.timestep = settings.reference_dt if method == "mjwarp" else settings.horizon
            model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
            self.models[method] = model
            self.backends[method] = Backend(
                "scm_box" if method == "scm" else method, model, make_data(model),
                device=settings.device, collect_contacts=True,
                stiffness=settings.contact_stiffness, damping=settings.contact_damping,
                scm_softplus_temperature=settings.scm_temperature,
                scm_regularization=settings.scm_regularization, scm_sweeps=settings.scm_sweeps,
                scm_soft_contact="none", nconmax=settings.nconmax, njmax=settings.njmax)
        self.vertices = object_vertices(reference)
        self.equilibria = {}
        self.support_pose = None

    def settle(self, sample, steps=500):
        """Reference-only preparation, outside the measured 20-ms query."""
        backend, model = self.backends["mjwarp"], self.models["mjwarp"]
        data = make_data(model, sample["qpos"], sample["qvel"], sample["wrench"])
        backend.reset(data)
        for index in range(steps):
            backend.step(data)
            if index >= 100 and np.linalg.norm(data.qvel) < 1e-5:
                break
        return data.qpos.copy(), data.qvel.copy()

    def snapshot(self, pose, native, time):
        model = self.models["mjwarp"]
        backend = self.backends["mjwarp"]
        # Do not use legacy Backend.solver_contacts' contiguous edge slicing.
        native, _, addresses = native_reference_contacts(backend.wdata)
        return reference_snapshot(model, pose, native, addresses, time)

    def reference_batch(self, samples):
        """Same object/model, independent states and forces, native nworld batch."""
        if not hasattr(self, "parallel_reference"):
            from examples.mpc.fingertips.test.collision_prediction_batch import BatchedMjwarpReference
            self.parallel_reference = BatchedMjwarpReference(self)
        return self.parallel_reference.evaluate(samples)

    def reference(self, sample):
        settings, backend = self.settings, self.backends["mjwarp"]
        model = self.models["mjwarp"]
        data = make_data(model, sample["qpos"], sample["qvel"], sample["wrench"])
        backend.reset(data)
        states_q, states_v, snapshots = [data.qpos.copy()], [data.qvel.copy()], []
        for index in range(settings.reference_steps):
            pose = make_data(model, data.qpos, data.qvel, sample["wrench"])
            backend.step(data)
            snapshots.append(self.snapshot(pose, backend.last_contacts, index*settings.reference_dt))
            states_q.append(data.qpos.copy())
            states_v.append(data.qvel.copy())
        # An extra forward SOLVE diagnoses the endpoint; it does not integrate
        # or change qpos/qvel and its forces never feed any prediction query.
        with backend.wp.ScopedDevice(backend.device):
            backend.api.forward(backend.wmodel, backend.wdata)
            backend.wp.synchronize_device(backend.device)
        snapshots.append(self.snapshot(data, backend.solver_contacts(data), settings.horizon))
        if abs(data.time-settings.horizon) > 1e-7:
            raise RuntimeError(f"Reference horizon mismatch: {data.time}")
        return data, snapshots, np.asarray(states_q), np.asarray(states_v)

    def predict(self, sample, reference):
        outputs = {}
        reference_velocity = world_velocity(self.models["mjwarp"], reference)
        reference_motion = pose_errors(reference.qpos[:7], sample["qpos"][:7], self.info["length_scale_m"])
        for method in METHODS:
            try:
                model = self.models[method]
                if method == "mjwarp":
                    prediction = reference
                else:
                    with native_diagnostics() as warnings:
                        prediction = make_data(model, sample["qpos"], sample["qvel"], sample["wrench"])
                        self.backends[method].reset(prediction)
                        self.backends[method].step(prediction)
                    if warnings:
                        raise RuntimeError("Native collision warning: " + "; ".join(warnings))
                if abs(prediction.time-self.settings.horizon) > 1e-7:
                    raise RuntimeError(f"{method} horizon mismatch: {prediction.time}")
                errors = pose_errors(prediction.qpos[:7], reference.qpos[:7], self.info["length_scale_m"])
                difference = world_velocity(model, prediction) - reference_velocity
                errors.update(linear_velocity_error_m_s=float(np.linalg.norm(difference[:3])),
                              angular_velocity_error_rad_s=float(np.linalg.norm(difference[3:])))
                errors.update(relative_errors(errors, reference_motion, reference_velocity))
                outputs[method] = dict(status="ok", failure_reason="", **errors,
                                       qpos=prediction.qpos.copy(), qvel=prediction.qvel.copy())
            except Exception as exc:
                outputs[method] = dict(status="failed", failure_reason=f"{type(exc).__name__}: {exc}",
                                       **{key: None for key in METRICS})
        return outputs

    def sample(self, name, mode, rng, attempt):
        """Recipes produce candidates only; native reference determines labels."""
        model = self.models["mjwarp"]
        qpos, qvel = model.qpos0.copy(), np.zeros(model.nv)
        wrench = np.zeros((model.nbody, 6))
        angle = rng.uniform(-np.pi, np.pi)
        yaw = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        sustained = mode == "sticking" or (mode == "sliding" and attempt % 2 == 0)
        top_impact = mode == "contact_establishment" and attempt % 3 == 0
        if name == "cube":
            kind = 0 if sustained or top_impact else attempt % 3
            # Bottom face, edge and vertex have exact support height.
            from scipy.spatial.transform import Rotation
            tilt = Rotation.from_euler("xy", [(0, np.pi/4, np.pi/4)[kind],
                                               (0, 0, np.arctan(1/np.sqrt(2)))[kind]]).as_quat()[[3, 0, 1, 2]]
            geometry = ("face", "edge", "vertex")[kind]
        elif name == "cylinder":
            tilt = np.array([1., 0, 0, 0]) if attempt % 2 == 0 else np.array([np.sqrt(.5), np.sqrt(.5), 0, 0])
            geometry = "face" if attempt % 2 == 0 else "line"
        else:
            tilt, geometry = np.array([1., 0, 0, 0]), "point" if name == "sphere" else "coacd"
        if geometry == "coacd":
            if self.support_pose is None:
                initial = dict(qpos=model.qpos0.copy(), qvel=np.zeros(model.nv),
                               wrench=np.zeros((model.nbody, 6)))
                initial["qpos"][:3] = [0, 0, support_height(name, tilt, self.vertices) + .0005]
                initial["qpos"][3:7] = tilt
                self.support_pose = self.settle(initial, steps=1500)[0][:7]
            tilt = self.support_pose[3:7].copy()
        mujoco.mju_mulQuat(qpos[3:7], yaw, tilt)
        qpos[:3] = [0, 0, support_height(name, qpos[3:7], self.vertices)-2e-5]
        force = FORCES[attempt % len(FORCES)]
        speed = SPEEDS[(attempt // len(FORCES)) % len(SPEEDS)]
        direction_angle = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(direction_angle), np.sin(direction_angle), 0.])
        tangent = np.array([-direction[1], direction[0], 0.])
        world_omega = np.zeros(3)
        if mode == "sliding":
            speed = SPEEDS[1 + (attempt // len(FORCES)) % 3]
            qvel[:3] = speed*tangent
            world_omega[2] = rng.uniform(-3, 3)
        elif mode in ("separation", "contact_establishment"):
            qvel[:3] = speed*tangent
            world_omega = rng.uniform(-1, 1, 3)*rng.choice([0., 1., 3.])/np.sqrt(3)
        if top_impact or sustained:
            world_omega[:] = 0.
        if top_impact:
            speed = 0.
            qvel[:3] = 0.
        qvel[3:6] = rotation(qpos[3:7]).T @ world_omega
        # Find the EXTERIOR surface along a ray using signed geom distance.
        # Works for primitives and the union of disconnected CoACD parts.
        center = qpos[:3].copy()
        if sustained or top_impact:
            # Downward loading on the top surface permits genuine sustained
            # contact at every force level, without a gravity-driven side slip.
            direction, tangent = np.array([0., 0., 1.]), np.array([1., 0., 0.])
        else:
            center[2] += rng.uniform(-.1, .15)*self.info["length_scale_m"]
            center[2] = max(center[2], self.settings.ball_radius + .002)
        qpos[7:10] = center + direction*self.info["length_scale_m"]*1.5
        qpos[10:14] = [1, 0, 0, 0]
        probe = make_data(model, qpos, qvel)
        ball_geom = model.geom("ball_geom").id
        target_geoms = np.flatnonzero((model.geom_bodyid == model.body("object").id) & (model.geom_contype != 0))

        def gap_at(distance):
            probe.qpos[7:10] = center + direction*distance
            mujoco.mj_kinematics(model, probe)
            return min(mujoco.mj_geomDistance(model, probe, int(g), ball_geom, 10., None)
                       for g in target_geoms)

        # Scan from outside to avoid choosing an interior cavity's far wall.
        # A concave object's first material surface can lie beyond its body
        # origin (e.g. the bottom inside a mug). Traverse the entire object.
        distances = np.linspace(1.5*self.info["length_scale_m"], -1.5*self.info["length_scale_m"], 128)
        outside, inside = None, None
        for distance in distances:
            if gap_at(distance) <= 0:
                inside = distance
                break
            outside = distance
        if inside is None or outside is None:
            raise ValueError("Ray did not intersect collision geometry")
        for _ in range(25):
            middle = (outside+inside)/2
            if gap_at(middle) > 0:
                outside = middle
            else:
                inside = middle
        separation = rng.uniform(.0003, .002) if mode == "contact_establishment" else -2e-5
        qpos[7:10] = center + direction*((outside+inside)/2 + separation)
        if mode == "contact_establishment":
            qvel[6:9] = -direction*rng.uniform(.1, .4) + qvel[:3]
            wrench[model.body("ball").id, :3] = -force*direction
        elif mode == "separation":
            qvel[6:9] = direction*rng.uniform(.1, .5) + qvel[:3]
            wrench[model.body("ball").id, :3] = force*direction
            # Also include true free flight without creating floor impact.
            if attempt % 4 == 0:
                qpos[2] += .04
                qpos[9] += .04
                qpos[7:10] += .02*direction
                geometry += "_airborne"
        else:
            wrench[model.body("ball").id, :3] = -force*direction
            qvel[6:9] = qvel[:3]
        if sustained:
            key = (name, geometry, force)
            if key not in self.equilibria:
                self.equilibria[key] = self.settle(dict(qpos=qpos, qvel=np.zeros(model.nv), wrench=wrench))
            qpos, qvel = [value.copy() for value in self.equilibria[key]]
            # Plane/yaw symmetry supplies different orientations from the same
            # equilibrium without introducing settling transients into queries.
            Rz = rotation(yaw)
            for qadr, vadr in ((0, 0), (7, 6)):
                qpos[qadr:qadr+3] = Rz @ qpos[qadr:qadr+3]
                orientation = qpos[qadr+3:qadr+7].copy()
                mujoco.mju_mulQuat(qpos[qadr+3:qadr+7], yaw, orientation)
                qvel[vadr:vadr+3] = Rz @ qvel[vadr:vadr+3]
            if mode == "sliding":
                qvel[:3] += speed*tangent
                qvel[6:9] += speed*tangent
        return dict(qpos=qpos, qvel=qvel, wrench=wrench, requested_mode=mode,
                    geometry=geometry, force_N=force, speed_level_m_s=0. if mode == "sticking" else speed,
                    initial_speed_m_s=float(np.linalg.norm(qvel[:3])),
                    initial_angular_speed_rad_s=float(np.linalg.norm(qvel[3:6])),
                    ball_initial_speed_m_s=float(np.linalg.norm(qvel[6:9])))
