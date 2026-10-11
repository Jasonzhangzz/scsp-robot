"""Contact sampling stays on low-curvature faces, not sharp corners."""

import os
import tempfile

import numpy as np
import pytest
import trimesh

from planning.project_point import ProjectionPoint


def _write_mesh(mesh):
    fd, path = tempfile.mkstemp(suffix='.stl')
    os.close(fd)
    mesh.export(path)
    return path


def test_cube_corners_have_high_incident_face_curvature():
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    defect = ProjectionPoint._vertex_angle_defect(mesh.vertices, mesh.faces)
    max_c, mean_c = ProjectionPoint._incident_face_curvature(
        len(mesh.vertices), mesh.faces, mesh.face_normals)
    # Eight cube corners: Gaussian curvature π/2.  Each corner sees a
    # 90° pair of faces (max 0.5); coplanar triangle splits pull the
    # mean down but it still clears the 0.10 sampling gate.
    assert np.all(defect > 1.0)
    assert np.all(max_c == pytest.approx(0.5))
    assert np.all(mean_c > 0.10)


def test_subdivided_box_keeps_face_interiors_and_drops_corners():
    mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.6])
    mesh = mesh.subdivide().subdivide()
    path = _write_mesh(mesh)
    try:
        pp = ProjectionPoint(path)
        corners = np.array([
            [sx, sy, sz]
            for sx in (-0.5, 0.5)
            for sy in (-0.5, 0.5)
            for sz in (-0.3, 0.3)
        ], dtype=np.float64)
        pool = np.asarray(pp.vertices[pp.stable_vertex_indices], dtype=np.float64)
        assert len(pool) >= 8
        dist_pool = np.linalg.norm(pool[:, None, :] - corners[None, :, :], axis=2)
        assert np.all(dist_pool.min(axis=1) > 0.08)

        frames = pp.sample_vertices_with_normals(20, strategy='farthest_point')
        pts = np.asarray(frames['points'], dtype=np.float64)
        assert len(pts) == 20
        dist_samp = np.linalg.norm(pts[:, None, :] - corners[None, :, :], axis=2)
        assert np.all(dist_samp.min(axis=1) > 0.08)
        assert np.all(pp.vertex_curvature[pp.stable_vertex_indices]
                      <= pp.max_vertex_curvature + 1e-9)
    finally:
        os.remove(path)


def test_sampling_drops_a_cone_tip():
    cone = trimesh.creation.cone(radius=0.4, height=1.0)
    tip = np.asarray(cone.vertices[np.argmax(cone.vertices[:, 2])], dtype=np.float64)
    path = _write_mesh(cone)
    try:
        pp = ProjectionPoint(path)
        pool = np.asarray(pp.vertices[pp.stable_vertex_indices], dtype=np.float64)
        assert np.min(np.linalg.norm(pool - tip[None, :], axis=1)) > 0.12
        frames = pp.sample_vertices_with_normals(
            min(16, len(pp.stable_vertex_indices)), strategy='farthest_point')
        pts = np.asarray(frames['points'], dtype=np.float64)
        assert np.min(np.linalg.norm(pts - tip[None, :], axis=1)) > 0.12
    finally:
        os.remove(path)


def test_surface_sampling_covers_all_faces_of_an_unsubdivided_box():
    # Vertex-only eligibility has no face interiors on this common mesh.
    path = _write_mesh(trimesh.creation.box(extents=[.10, .06, .04]))
    try:
        pp = ProjectionPoint(path, surface_sampling=True)
        np.random.seed(104)
        frames = pp.sample_surface_with_normals(70)
        assert len(frames['points']) == 70
        directions = np.unique(np.round(frames['normals'], 6), axis=0)
        assert len(directions) == 6
        assert np.max(frames['curvature']) < .25
        # Samples are on actual facets, and not all forced onto their corners.
        _, distance, _ = trimesh.proximity.closest_point(pp.scaled_mesh, frames['points'])
        assert np.max(distance) < 1e-8
        assert np.any(np.linalg.norm(frames['points'][:, :2], axis=1) < .025)
    finally:
        os.remove(path)


def test_surface_projection_does_not_snap_flat_face_to_a_vertex():
    path = _write_mesh(trimesh.creation.box(extents=[.10, .06, .04]))
    try:
        pp = ProjectionPoint(path, surface_sampling=True)
        point, n, t1, t2, _ = pp.project_surface([.011, -.007, .03])
        np.testing.assert_allclose(point, [.011, -.007, .02], atol=1e-8)
        np.testing.assert_allclose(n, [0., 0., -1.], atol=1e-12)
        np.testing.assert_allclose(np.stack([n, t1, t2]) @ np.stack([n, t1, t2]).T, np.eye(3), atol=1e-12)
    finally:
        os.remove(path)


def test_compound_sampler_rejects_buried_caps_and_keeps_cavity_walls(tmp_path):
    # Three boxes form a U cross-section with a gap wide enough for a 1 cm tip.
    parts = []
    for i, (extents, centre) in enumerate([
        ([.06, .06, .01], [0., 0., -.025]),
        ([.01, .06, .06], [-.025, 0., 0.]),
        ([.01, .06, .06], [.025, 0., 0.]),
    ]):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(centre)
        path = tmp_path / ('part%d.stl' % i)
        mesh.export(str(path)); parts.append(str(path))
    pp = ProjectionPoint(parts[0], surface_sampling=True, collision_part_paths=parts)
    np.random.seed(104)
    frames = pp.sample_surface_with_normals(70, fingertip_radius=.01)
    points, outward = frames['points'], -frames['normals']
    for part in pp.collision_parts:
        probe = points+5e-5*outward
        planes = probe @ part.face_normals.T - np.sum(part.face_normals*part.triangles_center, axis=1)
        assert np.all(np.max(planes, axis=1) >= -1e-7)
        _, distance, _ = trimesh.proximity.closest_point(part, points+.01*outward)
        assert np.min(distance) >= .01-5e-5
    assert np.any((outward[:, 0] > .9) & (points[:, 0] < -.019))
    assert np.any((outward[:, 0] < -.9) & (points[:, 0] > .019))
