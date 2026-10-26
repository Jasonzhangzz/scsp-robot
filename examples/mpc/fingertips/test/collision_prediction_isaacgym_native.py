"""Matched PhysX collision assets and a process-isolated Isaac Gym client.

The client uses the benchmark's current Python/MuJoCo environment. The worker
uses legacy Python 3.8, imports Isaac Gym first, and never imports MuJoCo.
No PhysX contact force is exposed by ``normalize_manifold`` to a predictor.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np


def quaternion_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected a finite nonzero wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def contact_frame(normal):
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
        raise ValueError("Invalid native contact normal")
    n = n / np.linalg.norm(n)
    axis = np.eye(3)[np.argmin(np.abs(n))]
    tangent = np.cross(axis, n)
    tangent /= np.linalg.norm(tangent)
    return np.array([n, tangent, np.cross(n, tangent)])


def normalize_manifold(contacts):
    """Canonical, deterministic geometry only; body IDs 0=floor/1=object/2=ball.

    Input contacts have ascending body IDs and a normal from the lower toward
    the higher ID (the force direction on the latter). Native forces and raw
    records are deliberately excluded from this predictor-facing interface.
    """
    rows = []
    for c in contacts:
        bodies = [int(c["body0"]), int(c["body1"])]
        normal = np.asarray(c["normal_world"], float)
        if bodies[0] > bodies[1]:
            bodies.reverse()
            normal = -normal
        if bodies[0] == bodies[1] or min(bodies) < 0 or max(bodies) > 2:
            raise ValueError("Unrecognized PhysX body pair")
        point = np.asarray(c["point_world"], float)
        gap, mu = float(c["gap"]), float(c["friction"])
        if point.shape != (3,) or not np.isfinite(np.r_[point, gap, mu]).all() or mu < 0:
            raise ValueError("Invalid native contact geometry")
        frame = contact_frame(normal)
        margin = float(c.get("include_margin", .002))
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Invalid native candidate margin")
        rows.append((bodies, point, frame, gap, mu, margin))
    rows.sort(key=lambda r: tuple(r[0]) + tuple(r[1]) + tuple(r[2][0]) + (r[3],))
    n = len(rows)
    return dict(gap=np.asarray([r[3] for r in rows], float),
                position=np.asarray([r[1] for r in rows], float).reshape(n, 3),
                frame=np.asarray([r[2] for r in rows], float).reshape(n, 3, 3),
                body=np.asarray([r[0] for r in rows], int).reshape(n, 2),
                friction=np.asarray([[r[4], r[4], 0., 0., 0.] for r in rows], float).reshape(n, 5),
                include_margin=np.asarray([r[5] for r in rows], float))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_obj(path, vertices, faces):
    with Path(path).open("w") as stream:
        stream.write("# Explicit collision geometry in the body COM link frame\n")
        for v in vertices:
            stream.write("v " + " ".join(format(float(x), ".17g") for x in v) + "\n")
        for f in faces:
            stream.write("f " + " ".join(str(int(x)+1) for x in f) + "\n")


def _cylinder(radius, half_height, slices):
    angle = np.arange(slices) * (2*np.pi/slices)
    ring = np.c_[radius*np.cos(angle), radius*np.sin(angle)]
    vertices = np.r_[np.c_[ring, np.full(slices, -half_height)],
                     np.c_[ring, np.full(slices, half_height)]]
    faces = []
    for k in range(slices):
        j = (k+1) % slices
        faces.extend([(k, j, slices+j), (k, slices+j, slices+k)])
    for k in range(1, slices-1):
        faces.extend([(0, k+1, k), (slices, slices+k, slices+k+1)])
    return vertices, np.asarray(faces, int)


def export_scene_spec(model_path, destination, cylinder_slices=96):
    """Export all compiled collision parts and explicit COM/inertia to URDF.

    Mesh geoms are exported separately; neither VHACD nor a whole-object hull
    fallback is used. Compiled MuJoCo mesh centering and geom transforms are
    baked into every exported part. Cylinder tessellation is explicit.
    """
    import mujoco
    if cylinder_slices != 96:
        raise ValueError("This benchmark fixes cylinder geometry to 96 slices")
    model_path, destination = Path(model_path).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if (model.nq, model.nv, model.nbody) != (14, 12, 3):
        raise ValueError("Expected exactly object+ball free bodies and a floor")
    spec = dict(schema_version=1, source_model=str(model_path), source_model_sha256=_sha(model_path),
                timestep=.002, reference_steps=10, horizon=.02,
                gravity=np.asarray(model.opt.gravity).tolist(), cylinder_slices=96,
                cylinder_max_radial_deficit_m=0., bodies=[], asset_sha256={},
                contact_offset=.001, rest_offset=0., position_iterations=8, velocity_iterations=1,
                max_depenetration_velocity=10.,
                collision_source="native PhysX pre-integration manifold; independent diagnostic scene",
                predictor_contact_forces=False,
                native_convex_cooking="PhysX asset loader; one shape per source convex part; cooked vertices not exposed",
                environment_isolation="All environment origins are zero; distinct actor collision group per query",
                enable_gyroscopic_forces=True,
                physx_parameters=dict(solver_type=1, num_position_iterations=8, num_velocity_iterations=1,
                    contact_offset=.001, rest_offset=0., max_depenetration_velocity=10.,
                    bounce_threshold_velocity=2*9.81*.002, num_threads=0, num_subscenes=0,
                    always_use_articulations=False, friction_correlation_distance=.025,
                    friction_offset_threshold=.04, default_buffer_size_multiplier=16.,
                    max_gpu_contact_pairs=8388608, contact_collection=2,
                    use_gpu="selected at worker initialization", use_gpu_pipeline=False),
                geometry_equivalence=dict(within_isaac_methods="Shared exported assets and native PhysX cooked manifolds",
                    source_caveat="Cylinder is 96-sided; PhysX may simplify individual convex parts during cooking; "
                                  "cooked vertices are not exposed. Whole-object single-hull fallback is forbidden."))
    if model.body("object").id != 1 or model.body("ball").id != 2:
        raise ValueError("Expected canonical free-body order object=1, ball=2")
    for name in ("object", "ball"):
        body = model.body(name).id
        ipos = np.asarray(model.body_ipos[body]).copy()
        inertia_rotation = quaternion_matrix(model.body_iquat[body])
        inertia = inertia_rotation @ np.diag(model.body_inertia[body]) @ inertia_rotation.T
        urdf = ET.Element("robot", name=name)
        link = ET.SubElement(urdf, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=format(float(model.body_mass[body]), ".17g"))
        ET.SubElement(inertial, "inertia", **{key: format(float(inertia[i, j]), ".17g")
            for key, i, j in (("ixx", 0, 0), ("iyy", 1, 1), ("izz", 2, 2),
                              ("ixy", 0, 1), ("ixz", 0, 2), ("iyz", 1, 2))})
        geoms = np.flatnonzero((model.geom_bodyid == body) &
                              ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
        coefficients = model.geom_friction[geoms, 0]
        if not len(geoms) or not np.allclose(coefficients, coefficients[0], atol=1e-12):
            raise ValueError("Expected uniformly frictional collision geometry")
        descriptions = []
        for local_index, geom in enumerate(geoms):
            kind = int(model.geom_type[geom])
            size = np.asarray(model.geom_size[geom])
            transform = quaternion_matrix(model.geom_quat[geom])
            translation = np.asarray(model.geom_pos[geom]) - ipos
            collision = ET.SubElement(link, "collision", name=f"part_{local_index:04d}")
            geometry = ET.SubElement(collision, "geometry")
            if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                # Sphere orientation has no geometric effect.
                ET.SubElement(collision, "origin", xyz=" ".join(map(str, translation)), rpy="0 0 0")
                ET.SubElement(geometry, "sphere", radius=str(float(size[0])))
                description = dict(kind="sphere", radius=float(size[0]))
            elif kind == int(mujoco.mjtGeom.mjGEOM_BOX) and np.allclose(transform, np.eye(3), atol=1e-12):
                ET.SubElement(collision, "origin", xyz=" ".join(map(str, translation)), rpy="0 0 0")
                ET.SubElement(geometry, "box", size=" ".join(map(str, 2*size)))
                description = dict(kind="box", half_extents=size.tolist())
            else:
                if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                    mesh = int(model.geom_dataid[geom])
                    adr, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                    fadr, nface = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
                    vertices = np.asarray(model.mesh_vert[adr:adr+count], float)
                    faces = np.asarray(model.mesh_face[fadr:fadr+nface], int)
                    description = dict(kind="convex_mesh", source_geom=int(geom), vertices=count)
                elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                    vertices, faces = _cylinder(size[0], size[1], 96)
                    deficit = float(size[0]*(1-np.cos(np.pi/96)))
                    spec["cylinder_max_radial_deficit_m"] = deficit
                    description = dict(kind="cylinder_convex_96", radius=float(size[0]),
                                       half_height=float(size[1]), max_radial_deficit_m=deficit)
                elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                    vertices = np.array([[x, y, z] for x in (-size[0], size[0])
                        for y in (-size[1], size[1]) for z in (-size[2], size[2])])
                    from scipy.spatial import ConvexHull
                    faces = ConvexHull(vertices).simplices
                    description = dict(kind="transformed_box_convex")
                else:
                    raise ValueError(f"Unsupported PhysX collision geom {kind}; no fallback permitted")
                vertices = vertices @ transform.T + translation
                filename = f"{name}_part_{local_index:04d}.obj"
                path = destination / filename
                _write_obj(path, vertices, faces)
                spec["asset_sha256"][str(path)] = _sha(path)
                ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
                ET.SubElement(geometry, "mesh", filename=filename, scale="1 1 1")
                description["bounds_com"] = [vertices.min(0).tolist(), vertices.max(0).tolist()]
            descriptions.append(description)
        path = destination / f"{name}.urdf"
        ET.ElementTree(urdf).write(path, encoding="unicode", xml_declaration=True)
        spec["asset_sha256"][str(path)] = _sha(path)
        spec["bodies"].append(dict(name=name, role_id=body, urdf=str(path), mass=float(model.body_mass[body]),
                                   ipos=ipos.tolist(), inertia=inertia.tolist(),
                                   principal_inertia=np.asarray(model.body_inertia[body]).tolist(),
                                   friction=float(coefficients[0]), expected_shapes=len(geoms), geometry=descriptions))
    (destination / "scene_spec.json").write_text(json.dumps(spec, indent=2)+"\n")
    return spec


class IsaacGymNative:
    """JSONL proxy; each query batch owns fresh native scenes, without warm starts."""
    def __init__(self, spec, batch_size=16, use_gpu=True, python=None, log_path=None):
        self.spec = spec
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.python = str(python or os.environ.get("SCSP_ISAACGYM_PYTHON") or
                          Path.home()/"miniconda3/envs/acc/bin/python")
        if not Path(self.python).is_file():
            raise RuntimeError(f"Isaac Gym Python interpreter missing: {self.python}")
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        library = str(Path(self.python).resolve().parent.parent / "lib")
        environment["LD_LIBRARY_PATH"] = library + (":"+environment["LD_LIBRARY_PATH"]
                                                   if environment.get("LD_LIBRARY_PATH") else "")
        self._log = Path(log_path).open("a") if log_path is not None else None
        self.process = subprocess.Popen([self.python, "-u", str(Path(__file__).with_name(
            "collision_prediction_isaacgym_worker.py"))], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, text=True, bufsize=1, env=environment)
        self.metadata = self._request(dict(cmd="init", spec=spec, use_gpu=bool(use_gpu), batch_size=self.batch_size))

    def _request(self, request):
        if self.process.poll() is not None:
            raise RuntimeError(f"Isaac Gym worker exited with status {self.process.returncode}")
        self.process.stdin.write(json.dumps(request, allow_nan=False)+"\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Isaac Gym worker closed protocol (exit={self.process.poll()})")
        result = json.loads(line)
        if not result.get("ok", False):
            raise RuntimeError(result.get("error", "Unknown Isaac Gym worker failure"))
        return result

    def _evaluate(self, cmd, qpos, qvel=None, wrench=None):
        qpos = np.asarray(qpos, float)
        if qpos.ndim == 1:
            qpos = qpos[None]
        qvel = np.zeros((len(qpos), 12)) if qvel is None else np.asarray(qvel, float).reshape(-1, 12)
        wrench = np.zeros((len(qpos), 3, 6)) if wrench is None else np.asarray(wrench, float).reshape(-1, 3, 6)
        if qpos.shape != (len(qpos), 14) or len(qvel) != len(qpos) or len(wrench) != len(qpos):
            raise ValueError("Expected matching qpos[N,14], qvel[N,12], wrench[N,3,6]")
        if len(qpos) > self.batch_size or not len(qpos):
            raise ValueError("Expected 1..batch_size independent queries")
        return self._request(dict(cmd=cmd, qpos=qpos.tolist(), qvel=qvel.tolist(), wrench=wrench.tolist()))

    def geometry(self, qpos, qvel=None):
        return self._evaluate("geometry", qpos, qvel)

    def reference(self, qpos, qvel, wrench):
        return self._evaluate("reference", qpos, qvel, wrench)

    def geometry_samples(self, samples):
        return self.geometry([x["qpos"] for x in samples], [x["qvel"] for x in samples])

    def reference_samples(self, samples):
        return self.reference([x["qpos"] for x in samples], [x["qvel"] for x in samples],
                              [x["wrench"] for x in samples])

    def close(self):
        if getattr(self, "process", None) is not None:
            try:
                if self.process.poll() is None:
                    try:
                        self._request(dict(cmd="close"))
                        self.process.wait(timeout=10)
                    except (RuntimeError, BrokenPipeError):
                        pass
            finally:
                if self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait(timeout=10)
                self.process.stdin.close()
                self.process.stdout.close()
                self.process = None
                if self._log is not None:
                    self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
