"""Offline oriented box fitting for A3's Lambda contact geometry only.

The exported box vertices are in the original actor frame, including the
box's offset and rotation. Current/target actor poses therefore transport
the box without changing Lambda's pose, wrench or inertia conventions.
"""
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import trimesh


_VERSION = 'actor-frame-obb-v1-angle2'


@dataclass(frozen=True)
class OBBContactGeometry:
    source_mesh: Path
    mesh_path: Path
    extents: np.ndarray
    box_to_actor: np.ndarray
    vertices: np.ndarray
    source_hash: str

    def world_transform(self, actor_pose):
        from scipy.spatial.transform import Rotation
        pose = np.asarray(actor_pose, dtype=float).reshape(7)
        actor_to_world = np.eye(4)
        actor_to_world[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
        actor_to_world[:3, 3] = pose[:3]
        return actor_to_world @ self.box_to_actor

    def world_vertices(self, actor_pose):
        transform = self.world_transform(actor_pose)
        corners = self.box_local_vertices()
        return corners @ transform[:3, :3].T + transform[:3, 3]

    def box_local_vertices(self):
        return (self.vertices-self.box_to_actor[:3, 3]) @ self.box_to_actor[:3, :3]

    def diagnostics(self):
        return dict(model='obb', source_mesh=str(self.source_mesh),
            proxy_mesh=str(self.mesh_path), source_sha256=self.source_hash,
            extents=self.extents.tolist(), box_to_actor=self.box_to_actor.tolist(),
            frame='actor', fit='trimesh.bounds.oriented_bounds', version=_VERSION)


def _fit_obb(mesh):
    to_box, _ = trimesh.bounds.oriented_bounds(mesh, angle_digits=2, ordered=True)
    # Recompute the enclosing bounds from every source vertex, including
    # vertices which were not needed by the oriented convex-hull search.
    local = np.asarray(mesh.vertices) @ to_box[:3, :3].T + to_box[:3, 3]
    low, high = local.min(axis=0), local.max(axis=0)
    extents = high-low
    if not np.isfinite(extents).all() or np.any(extents <= 1e-10):
        raise ValueError('A3 requires a nondegenerate 3-D object mesh')
    box_to_actor = np.linalg.inv(to_box)
    box_to_actor[:3, 3] += box_to_actor[:3, :3] @ (.5*(low+high))
    return trimesh.creation.box(extents=extents, transform=box_to_actor), extents, box_to_actor


def _atomic_write(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def cached_mesh_obb(mesh_path, cache_dir=''):
    """Fit once per mesh content; reuse the on-disk box across trials/workers."""
    source = Path(mesh_path).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    key = hashlib.sha256((_VERSION+trimesh.__version__+source_hash).encode()).hexdigest()
    default_cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home()/'.cache')))
    directory = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache/'scsp_robot/a3_obb'
    directory.mkdir(parents=True, exist_ok=True)
    asset, descriptor = directory/(key+'.obj'), directory/(key+'.json')
    # Spawned Isaac planners can request the same object simultaneously.
    with (directory/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        payload = None
        if asset.is_file() and descriptor.is_file():
            try:
                candidate = json.loads(descriptor.read_text())
                if candidate['version'] == _VERSION and candidate['source_hash'] == source_hash:
                    payload = candidate
            except (OSError, ValueError, KeyError):
                pass
        if payload is None:
            mesh = trimesh.load_mesh(source, process=True)
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError('A3 expects a triangle mesh: '+str(source))
            box, extents, transform = _fit_obb(mesh)
            payload = dict(version=_VERSION, source_hash=source_hash,
                extents=extents.tolist(), box_to_actor=transform.tolist(), vertices=box.vertices.tolist())
            # OBJ retains double precision and the box's flat triangle faces.
            _atomic_write(asset, box.export(file_type='obj', digits=17).encode())
            _atomic_write(descriptor, json.dumps(payload, indent=2).encode())
    return OBBContactGeometry(source, asset, np.asarray(payload['extents']),
        np.asarray(payload['box_to_actor']), np.asarray(payload['vertices']), source_hash)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', required=True)
    parser.add_argument('--cache-dir', default='')
    args = parser.parse_args()
    print(json.dumps(cached_mesh_obb(args.mesh, args.cache_dir).diagnostics(), indent=2))


if __name__ == '__main__':
    main()
