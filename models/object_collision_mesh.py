"""Cached convex parts shared by the simulator and contact planner."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import trimesh


def decomposed_collision_mesh(visual_mesh_path):
    """Return a surface STL and convex-part STLs in the authored body frame.

    The surface STL is for sampling only. Dynamic PhysX actors must load each
    part as a separate collision element, otherwise PhysX fills the cavity
    again. Publication of the manifest follows all mesh writes.
    """
    try:
        import coacd
        from importlib.metadata import version
    except ImportError as exc:
        raise RuntimeError("Concave collision geometry requires coacd; install it in the simulation environment") from exc
    settings = dict(threshold=.015, max_convex_hull=64, preprocess_mode='off',
                    resolution=2000, mcts_nodes=20, mcts_iterations=100,
                    mcts_max_depth=3, merge=True, decimate=True,
                    max_ch_vertex=128, seed=0)
    source = Path(visual_mesh_path).read_bytes()
    key = hashlib.sha256(source + json.dumps(settings, sort_keys=True).encode()
                         + version('coacd').encode() + b'compound-v1').hexdigest()[:24]
    directory = Path(tempfile.gettempdir()) / 'scsp_robot_convex_parts' / key
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'collision.json'
    # Multiple trial/planner workers may initialize the same object together.
    import fcntl
    with (directory / 'build.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not manifest.exists():
            mesh = trimesh.load_mesh(visual_mesh_path)
            if not mesh.is_watertight or not mesh.is_winding_consistent:
                raise ValueError('Convex decomposition requires a closed, consistently wound object mesh')
            coacd.set_log_level('warn')
            result = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **settings)
            if not result:
                raise RuntimeError('Convex decomposition produced no collision parts')
            parts = []
            paths = []
            for index, (vertices, faces) in enumerate(result):
                part = trimesh.Trimesh(vertices, faces, process=True).convex_hull
                if not part.is_watertight or part.volume <= 0:
                    raise ValueError('Invalid convex collision part')
                path = directory / ('part_%03d.stl' % index)
                part.export(str(path))
                parts.append(part)
                paths.append(str(path))
            surface = directory / 'surface.stl'
            trimesh.util.concatenate(parts).export(str(surface))
            payload = dict(surface=str(surface), parts=paths, settings=settings,
                           coacd_version=version('coacd'))
            pending = directory / ('collision.%d.tmp' % os.getpid())
            pending.write_text(json.dumps(payload, indent=2) + '\n')
            pending.replace(manifest)
        payload = json.loads(manifest.read_text())
    return payload['surface'], payload['parts']
