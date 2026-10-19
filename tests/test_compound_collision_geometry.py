"""A compound object must keep its cavity and share reachable contact facets."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import trimesh

from examples.mpc.fingertips.test.params import canonical_object_meshes, _xml_obj_geom_transform
from examples.mpc.franka.runtime.simulator import IsaacFrankaSimulator
from models.object_collision_mesh import decomposed_collision_mesh
from planning.project_point import ProjectionPoint


ROOT = Path(__file__).resolve().parents[1]


def test_mug_cavity_is_open_in_collision_parts():
    source = ROOT / 'envs/assets/objects/mug.stl'
    xml = ROOT / 'envs/xmls/env_fingertips_mug.xml'
    hull, visual = canonical_object_meshes(str(source), str(xml))
    _, parts = decomposed_collision_mesh(visual)
    raw = trimesh.load_mesh(str(source))
    bottom = raw.vertices[raw.vertices[:, 2] < raw.bounds[0, 2]+.001]
    probe = np.r_[np.mean(bottom[:, :2], axis=0), raw.bounds[1, 2]+.02]
    rot, pos, scale = _xml_obj_geom_transform(str(xml))
    origin = (probe*scale) @ rot.T + pos
    direction = np.array([0., 0., -1.]) @ rot.T

    def ray_distance(paths):
        distances = []
        for path in paths:
            mesh = trimesh.load_mesh(path)
            hits, _, _ = mesh.ray.intersects_location(origin[None], direction[None])
            if len(hits):
                distances.extend(np.linalg.norm(hits-origin, axis=1))
        return min(distances)

    assert ray_distance(parts)-ray_distance([hull]) > .06
    assert abs(ray_distance(parts)-ray_distance([visual])) < .001


def test_mug_collision_solids_cover_rendered_bevels_and_rim():
    _, visual = canonical_object_meshes(str(ROOT/'envs/assets/objects/mug.stl'),
                                      str(ROOT/'envs/xmls/env_fingertips_mug.xml'))
    _, paths = decomposed_collision_mesh(visual, object_name='mug')
    mesh = trimesh.load_mesh(visual)
    # Test vertices and face interiors of the actual rendered surface, rather
    # than sampling from the collision model (which cannot expose missing faces).
    probes = np.vstack([mesh.vertices, mesh.triangles_center])
    outside = np.full(len(probes), np.inf)
    for path in paths:
        part = trimesh.load_mesh(path)
        normals = part.face_normals
        offsets = np.sum(normals * part.triangles_center, axis=1)
        outside = np.minimum(outside, np.max(probes @ normals.T - offsets, axis=1))
    assert np.max(outside) < 1e-6


def test_compound_urdf_and_planner_use_identical_convex_parts():
    source = ROOT / 'envs/assets/objects/mug.stl'
    xml = ROOT / 'envs/xmls/env_fingertips_mug.xml'
    _, visual = canonical_object_meshes(str(source), str(xml))
    surface, parts = decomposed_collision_mesh(visual)
    dynamic, _, directory = IsaacFrankaSimulator._prepare_mesh_urdf_assets(
        str(ROOT), surface, visual, collision_mesh_paths=parts)
    elements = ET.parse(str(Path(directory)/dynamic)).findall('.//collision/geometry/mesh')
    assert len(elements) == len(parts) > 1
    for element, part in zip(elements, parts):
        assert (Path(directory)/element.attrib['filename']).read_bytes() == Path(part).read_bytes()

    pp = ProjectionPoint(surface, surface_sampling=True, collision_part_paths=parts)
    np.random.seed(104)
    frame = pp.sample_surface_with_normals(70, fingertip_radius=.01)
    # Check collision distance against actual convex mesh geoms, independent
    # of the sampler's halfspace and triangle-distance eligibility logic.
    assets = ''.join('<mesh name="m%d" file="%s"/>' % (i, p) for i, p in enumerate(parts))
    geoms = ''.join('<geom name="g%d" type="mesh" mesh="m%d"/>' % (i, i) for i in range(len(parts)))
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><option ccd_tolerance="1e-9" ccd_iterations="100"><flag nativeccd="enable"/></option>'
        '<asset>'+assets+'</asset><worldbody><body>'+geoms+
        '</body><body><freejoint/><geom name="tip" type="sphere" size=".01" mass=".01"/>'
        '</body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    tip = model.geom('tip').id
    for point, inward in zip(frame['points'], frame['normals']):
        data.qpos[:3] = point-.0098*inward
        data.qpos[3:7] = [1., 0., 0., 0.]
        mujoco.mj_forward(model, data)
        gaps = [float(c.dist) for c in data.contact if tip in (c.geom1, c.geom2)]
        assert gaps
        assert min(gaps) == pytest.approx(-.0002, abs=1e-4)
