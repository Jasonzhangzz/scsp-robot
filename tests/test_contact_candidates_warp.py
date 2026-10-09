"""Physical surface patches and ordered online contact cones on real Warp CPU."""

import numpy as np
import pytest
import trimesh
from pathlib import Path

from planning.contact_candidates_warp import (
    SurfaceContactCandidates, _area_samples, _patch_statistics,
)


@pytest.mark.parametrize("scale", [1.0, 1.2])
@pytest.mark.parametrize("seed", [0, 7])
def test_elephant_body_sides_exclude_head_tail_and_longitudinal_normals(seed, scale):
    mesh = Path(__file__).resolve().parents[1] / "envs/assets/objects/elephant.stl"
    surface = SurfaceContactCandidates(mesh, scale_factors=(scale,) * 3,
        sample_num=70, seed=seed, device="cpu", body_sides=True)
    # Independent anatomical check for this asset, whose long axis is X.
    assert surface.sample_num == 70
    assert np.min(surface.points_local[:, 0]) > -.025 * scale
    assert np.max(surface.points_local[:, 0]) < .0375 * scale
    assert np.min(np.abs(surface.points_local[:, 1])) > .0108 * scale
    assert np.min(np.abs(surface.normals_local[:, 1])) > .84
    assert surface.global_pairs().count > 0
    batch = surface.global_pairs().to_host()
    normals = batch["normals"][batch["valid"]]
    assert np.all(normals[:, 0, 1] * normals[:, 1, 1] < 0)


@pytest.fixture(scope="module")
def candidates():
    mesh = trimesh.creation.box(extents=[.2, .12, .1])
    mesh.apply_translation([.22, -.15, .1])
    return SurfaceContactCandidates(mesh, sample_num=96, dense_sample_num=8192,
                                    seed=7, device="cpu")


def test_area_samples_and_original_mesh_patches_are_repeatable(candidates):
    second = SurfaceContactCandidates(candidates.mesh, sample_num=96,
                                      dense_sample_num=8192, seed=7, device="cpu")
    np.testing.assert_array_equal(candidates.points_local, second.points_local)
    np.testing.assert_array_equal(candidates.normals_local, second.normals_local)
    np.testing.assert_allclose(candidates.geometric_center_local, [.22, -.15, .1])
    assert candidates.preprocessing_diagnostics["original_faces"] == 12
    assert candidates.preprocessing_diagnostics["rejected_curvature"] > 0
    assert np.max(candidates.curvature_max) <= .25
    assert np.max(candidates.curvature_mean) <= .12
    assert np.min(candidates.patch_support_area) >= candidates.preprocessing_diagnostics["support_area_threshold"]
    # Every retained flat-face normal points into the translated solid.
    toward_center = candidates.geometric_center_local - candidates.points_local
    assert np.all(np.einsum("ij,ij->i", toward_center, candidates.normals_local) > 0)
    np.testing.assert_allclose(np.linalg.norm(candidates.normals_local, axis=1), 1)


def test_thin_opposite_faces_do_not_pollute_local_curvature_but_edges_do():
    mesh = trimesh.creation.box(extents=[.2, .12, .002])
    dense, faces = _area_samples(mesh, 65536, np.random.default_rng(4))
    top_face = np.flatnonzero(mesh.face_normals[:, 2] > .99)[0]
    points = np.array([[0., 0., .001], [.098, 0., .001]])
    maximum, mean, support = _patch_statistics(
        mesh, points, np.repeat(top_face, 2), dense, faces, .01)
    assert maximum[0] == pytest.approx(0.)
    assert mean[0] == pytest.approx(0.)
    assert support[0] == pytest.approx(np.pi * .01**2, rel=.15)
    assert maximum[1] >= .5  # Real sidewall inside the spherical pad footprint.
    assert mean[1] > 0


def test_disconnected_close_sheet_does_not_supply_missing_patch_area():
    # The narrow strip lies beside a broad sheet, closer than the pad radius.
    # Area on another connected component cannot support a pad on the strip.
    broad = trimesh.Trimesh(vertices=[[-.1, -.1, 0], [.1, -.1, 0], [.1, .1, 0], [-.1, .1, 0]],
                            faces=[[0, 1, 2], [0, 2, 3]], process=False)
    strip = trimesh.Trimesh(vertices=[[-.1, -.0015, .001], [.1, -.0015, .001],
                                    [.1, .0015, .001], [-.1, .0015, .001]],
                            faces=[[0, 1, 2], [0, 2, 3]], process=False)
    mesh = trimesh.util.concatenate([broad, strip])
    dense, faces = _area_samples(mesh, 65536, np.random.default_rng(9))
    _, _, support = _patch_statistics(mesh, np.array([[0., 0., .001]]),
                                      np.array([2]), dense, faces, .01)
    assert support[0] < .30 * np.pi * .01**2
    assert support[0] == pytest.approx(.003 * .02, rel=.25)


def test_global_pairs_keep_original_ids_and_antipodal_friction_bounds(candidates):
    batch = candidates.global_pairs(arm_friction=.9).to_host()
    valid = batch["valid"]
    assert valid.any()
    ids = batch["ids"][valid]
    assert np.all(ids[:, 0] < ids[:, 1])
    np.testing.assert_array_equal(batch["points"][valid], candidates.points_local[ids])
    np.testing.assert_array_equal(batch["normals"][valid], candidates.normals_local[ids])
    chord = batch["points"][valid, 1] - batch["points"][valid, 0]
    lengths = np.linalg.norm(chord, axis=1)
    assert np.all(lengths >= .03)
    axis = chord / lengths[:, None]
    cosine = 1 / np.sqrt(1 + .9**2)
    assert np.all(np.einsum("ij,ij->i", batch["normals"][valid, 0], axis) >= cosine - 1e-6)
    assert np.all(np.einsum("ij,ij->i", batch["normals"][valid, 1], -axis) >= cosine - 1e-6)


@pytest.mark.parametrize("half_angle", [12., 35.])
def test_online_cones_use_translated_mesh_center_and_keep_left_right_assignment(candidates, half_angle):
    center = candidates.geometric_center_local
    tips = center + np.array([[-.3, 0, 0], [.3, 0, 0]])
    batch = candidates.online_pairs(tips, cone_half_angle_deg=half_angle,
                                    max_per_arm=12, max_pairs=144).to_host()
    points, normals = batch["points"][batch["valid"]], batch["normals"][batch["valid"]]
    assert len(points) > 0
    assert np.all(points[:, 0, 0] < center[0])
    assert np.all(points[:, 1, 0] > center[0])
    # Far faces in the same cone are excluded by their facing normal.
    np.testing.assert_allclose(normals[:, 0, 0], 1.)
    np.testing.assert_allclose(normals[:, 1, 0], -1.)
    delta = points - tips
    headings = (center - tips) / np.linalg.norm(center - tips, axis=1)[:, None]
    cone_cosine = np.einsum("baj,aj->ba", delta, headings) / np.linalg.norm(delta, axis=2)
    assert np.all(cone_cosine >= np.cos(np.deg2rad(half_angle)) - 1e-6)
    expected_approach = np.sum(delta * delta, axis=(1, 2))
    np.testing.assert_allclose(batch["approach_cost"][batch["valid"]], expected_approach, rtol=2e-6)


def test_support_plane_clears_entire_spherical_pad(candidates):
    center = candidates.geometric_center_local
    radius = .01
    support_point = center + [0., 0., -.05]
    batch = candidates.global_pairs(support_point_local=support_point,
                                    support_normal_local=[0., 0., 3.],
                                    fingertip_radius=radius, support_clearance=.002).to_host()
    assert batch["valid"].any()
    points = batch["points"][batch["valid"]]
    normals = batch["normals"][batch["valid"]]
    pad_center = points - radius * normals
    clearance = pad_center[:, :, 2] - support_point[2] - radius
    assert np.all(clearance >= .002 - 1e-7)


def test_device_tip_and_support_plane_filter_matches_host_inputs(candidates):
    import warp as wp

    center = candidates.geometric_center_local
    tips = center + np.array([[-.3, 0., 0.], [.3, 0., 0.]])
    plane = np.array([center + [0., 0., -.05], [0., 0., 3.]])
    host = candidates.online_pairs(tips, cone_half_angle_deg=35., max_per_arm=12,
        support_point_local=plane[0], support_normal_local=plane[1], support_clearance=.002).to_host()
    device = candidates.online_pairs(wp.array(tips, dtype=wp.vec3, device="cpu"),
        cone_half_angle_deg=35., max_per_arm=12,
        support_plane_device=wp.array(plane, dtype=wp.vec3, device="cpu"), support_clearance=.002).to_host()
    assert device["valid"].any()
    for name in ("valid", "ids", "points", "normals", "approach_cost"):
        np.testing.assert_array_equal(device[name], host[name])
    global_host = candidates.global_pairs(support_point_local=plane[0], support_normal_local=plane[1]).to_host()
    global_device = candidates.global_pairs(support_plane_device=wp.array(plane, dtype=wp.vec3, device="cpu")).to_host()
    np.testing.assert_array_equal(global_device["valid"], global_host["valid"])


@pytest.mark.parametrize("normal", [[0., 0., 0.], [np.nan, 0., 1.], [0., 0., np.inf]])
def test_invalid_device_support_plane_never_yields_admissible_pair(candidates, normal):
    import warp as wp

    plane = wp.array([[0., 0., 0.], normal], dtype=wp.vec3, device="cpu")
    assert candidates.global_pairs(support_plane_device=plane).count == 0


def test_empty_or_degenerate_cones_report_no_pair_without_geometry_fallback(candidates):
    center = candidates.geometric_center_local
    batch = candidates.online_pairs(np.stack([center, center]), max_per_arm=4).to_host()
    assert not batch["valid"].any()
    assert np.all(batch["ids"] == -1)
    assert not np.any(batch["points"])
    # Global fallback remains independently available to the live selector.
    assert candidates.global_pairs().count > 0


@pytest.mark.parametrize("angle", [0., 90., np.nan])
def test_invalid_cone_angles_rejected(candidates, angle):
    with pytest.raises(ValueError, match="cone_half_angle"):
        candidates.online_pairs(candidates.geometric_center_local + [[-.3, 0, 0], [.3, 0, 0]],
                                 cone_half_angle_deg=angle)
