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
