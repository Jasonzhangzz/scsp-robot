"""Cached convex parts shared by the simulator and contact planner."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import trimesh


_MUG_DECOMPOSITION_VERSION = "mug-wall-sectors-v3"


def _looks_like_canonical_mug(mesh):
    """Recognize the canonical, geom-transformed mug when no object name is passed."""
    extents = np.asarray(mesh.extents, dtype=np.float64).reshape(3)
    # The mug's canonical body frame is stable across STL export order.  The
    # tight bounds also keep generic CoACD callers on their existing path.
    return bool(np.all(np.abs(extents - np.array([.08272726, .065, .09172209])) < .003))


def _mug_convex_parts(mesh):
    """Build convex pieces that retain the hollow interior of the mug.

    CoACD is a good general-purpose decomposition, but its convex hulls can
    bridge the bottom and the rim of this thin-walled mug.  In the canonical
    body frame the mug axis is ``x``.  The cylindrical wall is therefore
    split into narrow sectors including the bevels and the top rim.
    Bottom-cap faces are kept in a separate convex piece, so
    they cannot close the cavity along the whole height.  The handle is
    outside the cup radius and is decomposed independently.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    centers = triangles.mean(axis=1)
    radial = np.linalg.norm(centers[:, 1:3], axis=1)
    angles = np.mod(np.arctan2(centers[:, 2], centers[:, 1]), 2.0 * np.pi)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)

    parts = []
    n_sectors = 32
    cup_radius = 0.042  # outer cup radius in the canonical mug frame

    # Include sloped wall/handle shoulders and the annular top rim. The old
    # |n_x| < .7 gate omitted bevels by up to 1.2 mm, leaving the rendered cup
    # outside its collision solids. Only axial floor caps must stay separate:
    # joining them to a wall sector would bridge the hollow interior.
    wall_mask = (radial <= cup_radius) & (
        (np.abs(normals[:, 0]) < .99)
        | (triangles[:, :, 0].min(axis=1) > .03))
    for sector in range(n_sectors):
        middle = 2.0 * np.pi * (sector + 0.5) / n_sectors
        angular_distance = np.abs(
            np.angle(np.exp(1j * (angles - middle))))
        selected = wall_mask & (
            angular_distance <= np.pi / n_sectors + 0.01)
        points = triangles[selected].reshape(-1, 3)
        if len(points) < 4:
            continue
        part = trimesh.convex.convex_hull(points)
        if part.volume > 1e-10:
            parts.append(part)

    # Keep the lower floor as one short convex solid.  Restricting every
    # vertex to the bottom layer prevents the large cap triangles in some STL
    # exports from connecting the floor to the rim.
    bottom = (
        (triangles[:, :, 0].max(axis=1) <= -0.031)
        & (radial <= cup_radius))
    floor_points = triangles[bottom].reshape(-1, 3)
    if len(floor_points) >= 4:
        floor = trimesh.convex.convex_hull(floor_points)
        if floor.volume > 1e-10:
            parts.append(floor)

    # The handle is a separate exterior component.  Splitting it along the
    # cup axis keeps its loop from creating a long convex bridge back through
    # the cavity near the attachment points.
    handle = radial > cup_radius
    x_edges = np.linspace(-0.042, 0.042, 5)
    for lo, hi in zip(x_edges[:-1], x_edges[1:]):
        selected = handle & (centers[:, 0] >= lo) & (centers[:, 0] <= hi)
        points = triangles[selected].reshape(-1, 3)
        if len(points) < 4:
            continue
        part = trimesh.convex.convex_hull(points)
        if part.volume > 1e-10:
            parts.append(part)

    if len(parts) < n_sectors:
        raise RuntimeError("Mug convex decomposition produced too few parts")
    return parts


def decomposed_collision_mesh(visual_mesh_path, manifest_path=None,
                               object_name=None):
    """Return a surface STL and convex-part STLs in the authored body frame.

    The surface STL is for sampling only. Dynamic PhysX actors must load each
    part as a separate collision element, otherwise PhysX fills the cavity
    again. Publication of the manifest follows all mesh writes.
    """
    if manifest_path:
        payload = json.loads(Path(manifest_path).read_text())
        stale_mug_manifest = (
            str(object_name or '').lower() == 'mug'
            and payload.get('decomposition') != _MUG_DECOMPOSITION_VERSION)
        if not stale_mug_manifest:
            if payload.get('source_sha256') != hashlib.sha256(Path(visual_mesh_path).read_bytes()).hexdigest():
                raise ValueError('Collision manifest does not match the canonical object mesh')
            paths = [payload['surface'], *payload['parts']]
            if not payload['parts']:
                raise ValueError('Collision manifest has no convex parts')
            for path in paths:
                expected = payload.get('asset_sha256', {}).get(path)
                if not expected or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                    raise ValueError('Collision manifest asset changed: '+path)
            return payload['surface'], payload['parts']
    mesh_probe = None
    if object_name is None:
        try:
            mesh_probe = trimesh.load_mesh(visual_mesh_path)
        except Exception:
            mesh_probe = None
    mug_special = (str(object_name or '').lower() == 'mug'
                   or (mesh_probe is not None and _looks_like_canonical_mug(mesh_probe)))
    try:
        import coacd
        from importlib.metadata import version
        coacd_version = version('coacd')
    except ImportError as exc:
        if not mug_special:
            raise RuntimeError("Concave collision geometry requires coacd; install it in the simulation environment") from exc
        coacd = None
        coacd_version = 'not-required'
    settings = dict(threshold=.015, max_convex_hull=64, preprocess_mode='off',
                    resolution=2000, mcts_nodes=20, mcts_iterations=100,
                    mcts_max_depth=3, merge=True, decimate=True,
                    max_ch_vertex=128, seed=0)
    source = Path(visual_mesh_path).read_bytes()
    decomposition_tag = _MUG_DECOMPOSITION_VERSION if mug_special else 'coacd-v1'
    key = hashlib.sha256(source + json.dumps(settings, sort_keys=True).encode()
                         + coacd_version.encode() + decomposition_tag.encode()
                         + b'compound-v2').hexdigest()[:24]
    directory = Path(tempfile.gettempdir()) / 'scsp_robot_convex_parts' / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'collision.json'
    # Multiple trial/planner workers may initialize the same object together.
    import fcntl
    with (directory / 'build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not manifest.exists():
            mesh = mesh_probe if mesh_probe is not None else trimesh.load_mesh(visual_mesh_path)
            if not mesh.is_watertight or not mesh.is_winding_consistent:
                raise ValueError('Convex decomposition requires a closed, consistently wound object mesh')
            if mug_special:
                parts = _mug_convex_parts(mesh)
            else:
                coacd.set_log_level('warn')
                result = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **settings)
                if not result:
                    raise RuntimeError('Convex decomposition produced no collision parts')
                parts = [trimesh.Trimesh(vertices, faces, process=True).convex_hull
                         for vertices, faces in result]
            paths = []
            for index, part in enumerate(parts):
                if not part.is_watertight or part.volume <= 0:
                    raise ValueError('Invalid convex collision part')
                path = directory / ('part_%03d.stl' % index)
                part.export(str(path))
                paths.append(str(path))
            surface = directory / 'surface.stl'
            trimesh.util.concatenate(parts).export(str(surface))
            payload = dict(surface=str(surface), parts=paths, settings=settings,
                           coacd_version=coacd_version, decomposition=decomposition_tag)
            pending = directory / ('collision.%d.tmp' % os.getpid())
            pending.write_text(json.dumps(payload, indent=2) + '\n')
            pending.replace(manifest)
        payload = json.loads(manifest.read_text())
    return payload['surface'], payload['parts']
