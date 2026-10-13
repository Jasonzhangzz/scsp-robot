"""Live contact quality, assignment and hysteresis on real Warp CPU kernels."""

import numpy as np
import pytest

from planning.lambda_contact_lm import ensure_selection_kernels


def _array(values, dtype=float):
    wp = ensure_selection_kernels().wp
    wp.init()
    return wp.array(np.asarray(values), dtype=dtype, device="cpu")


def _select_online(reductions, *, valid=None, incumbent_extra_distance=.0015,
                   switch_margin=.002, quality_fraction=.015, current_cost=.0025):
    k, wp = ensure_selection_kernels(), ensure_selection_kernels().wp
    tips = np.array([[-.2, 0., 0.], [.2, 0., 0.]])
    incumbent_y = np.sqrt((.1 + incumbent_extra_distance / 2)**2 - .1**2)
    pairs = np.array([[[-.1, .10, 0], [.1, .10, 0]],
                      [[-.1, incumbent_y, 0], [.1, incumbent_y, 0]],
                      [[-.1, 0., 0.], [.1, 0., 0.]],
                      [[-.195, 0., 0.], [.195, 0., 0.]]])
    pairs = pairs[:len(reductions)]
    selected = wp.zeros(2, dtype=int, device="cpu")
    diagnostics = wp.zeros(5, dtype=float, device="cpu")
    current = np.array([0., 0., 0., 1., 0., 0., 0.])
    target = np.array([0., 0., np.sqrt(current_cost), 1., 0., 0., 0.])
    wp.launch(k.select_online, dim=1, inputs=[
        _array(reductions), _array(np.ones(len(reductions)) if valid is None else valid, int),
        _array(pairs.reshape(-1, 3), wp.vec3), _array(tips, wp.vec3),
        _array(current), _array(target), 1., .0005, quality_fraction, switch_margin,
        selected, diagnostics], device="cpu")
    return int(selected.numpy()[0]), diagnostics.numpy()


def test_near_quality_selects_close_contacts_and_holds_eligible_incumbent():
    reductions = [.001, .000988, .000990, .000800]
    selected, diagnostics = _select_online(reductions)
    assert selected == 1  # Only 1.5 mm farther; within 2 mm switch margin.
    np.testing.assert_allclose(diagnostics, [.001, .000015, 2., 0., .0025], rtol=1e-6)
    selected, _ = _select_online(reductions, switch_margin=.001)
    assert selected == 2
    # The closest pair is excluded for failing true predicted pose reduction.
    assert selected != 3


@pytest.mark.parametrize("valid,reduction", [([1, 0, 1, 1], .000988),
                                             ([1, 1, 1, 1], .00095)])
def test_stale_or_lower_quality_incumbent_cannot_be_held(valid, reduction):
    selected, diagnostics = _select_online([.001, reduction, .000990, .000800],
                                           valid=valid, switch_margin=1.)
    assert selected == 2
    assert diagnostics[2] == 1


def test_quality_floor_tracks_current_pose_cost_in_small_native_cost_units():
    selected, diagnostics = _select_online([1e-8, .85e-8, .5e-8, .2e-8], switch_margin=0.)
    assert selected == 1
    # Legacy absolute 1e-3 would accept all four of these physically tiny costs.
    assert diagnostics[1] == pytest.approx(1e-6 * .0025, rel=1e-6)
    assert diagnostics[2] == 1
    # Raising pose cost raises the relative absolute floor without normalizing Δ.
    selected, diagnostics = _select_online([.001, .00099, .00098, .0005],
                                           current_cost=100., switch_margin=0.)
    assert selected == 2
    assert diagnostics[1] == pytest.approx(.0001, rel=1e-6)


def test_online_reference_uses_best_actual_reduction_and_global_fallback():
    selected, diagnostics = _select_online([.001, .0009, .0008, .0007])
    assert selected == 0
    assert diagnostics[2] == 0 and diagnostics[3] == 1
    selected, diagnostics = _select_online([.001, .0009, .0012, .001185], switch_margin=0.)
    assert selected == 3  # New better online reference; nearest within quality band.
    assert diagnostics[0] == pytest.approx(.0012)
    selected, diagnostics = _select_online([.001, np.nan, np.inf, -np.inf])
    assert selected == 0
    assert diagnostics[2] == 0
    selected, _ = _select_online([np.nan, np.nan, np.inf, -np.inf], valid=[0, 1, 1, 1])
    assert selected == -1


def test_positive_global_improvement_rejects_nonimproving_cone_contact():
    selected, diagnostics = _select_online([1.1e-9, 0., 0., 0.], current_cost=.01)
    assert selected == 0
    assert diagnostics[2] == 0
    # A negative best reduction still has a finite near-quality policy.
    selected, diagnostics = _select_online([-.01, -.0101, -.01, -.012], switch_margin=0.)
    assert selected == 2
    assert diagnostics[0] == pytest.approx(-.01)


def test_global_selection_uses_curvature_and_balance_only_inside_quality_band():
    k, wp = ensure_selection_kernels(), ensure_selection_kernels().wp
    points = np.array([[[-.1, 0., 0.], [.1, 0., 0.]],
                       [[-.1, .02, 0.], [.1, .02, 0.]],
                       [[-.1, 0., 0.], [.1, 0., 0.]]])
    selected = wp.zeros(2, dtype=int, device="cpu")
    wp.launch(k.select_global, dim=1, inputs=[
        _array([.1, .0995, .095]), _array([1, 1, 1], int),
        _array([[0, 1], [2, 3], [4, 5]], int), _array(points.reshape(-1, 3), wp.vec3),
        _array([.1, .1, .0, .0, .0, .0]), wp.vec3(0.), .15,
        _array([0., 0., 0., 1., 0., 0., 0.]),
        _array([0., 0., .4, 1., 0., 0., 0.]),
        _array([[.2, 0., 0.], [-.2, 0., 0.]], wp.vec3),
        1., .0005, .015, 1.5, selected], device="cpu")
    np.testing.assert_array_equal(selected.numpy(), [1, 1])


def test_online_assembly_reassigns_offline_but_excludes_incumbent_outside_cones():
    k, wp = ensure_selection_kernels(), ensure_selection_kernels().wp
    source = np.array([[-.1, 0., .05], [.1, 0., .05], [-.1, .09, .05], [.1, .09, .05]])
    source_normals = np.array([[1., 0., 0.], [-1., 0., 0.], [1., 0., 0.], [-1., 0., 0.]])
    points, normals = wp.zeros(6, dtype=wp.vec3, device="cpu"), wp.zeros(6, dtype=wp.vec3, device="cpu")
    ids, valid = wp.zeros((3, 2), dtype=int, device="cpu"), wp.zeros(3, dtype=int, device="cpu")
    wp.launch(k.assemble_online, dim=3, inputs=[
        _array(source, wp.vec3), _array(source_normals, wp.vec3),
        _array([1, 0], int), _array([2, 3], int),
        _array([[0., 0., 0.], [0., 0., 0.]], wp.vec3),
        _array([[0., 0., 0.], [0., 0., 0.]], wp.vec3),
        _array([[-1, -1]], int), _array([0], int),
        _array([[-.2, 0., .05], [.2, 0., .05]], wp.vec3),
        wp.vec3(0., 0., .05), .15, float(np.cos(np.deg2rad(10.))),
        1, wp.vec3(0.), wp.vec3(0., 0., 1.), .01, .0,
        points, normals, ids, valid], device="cpu")
    np.testing.assert_array_equal(valid.numpy(), [1, 0, 0])
    np.testing.assert_array_equal(ids.numpy()[0], [0, 1])
    np.testing.assert_array_equal(points.numpy()[:2], source[:2].astype(np.float32))
    # The offline reference remains usable even when live cones are empty.
    selected, _ = _select_online([.001, .001, .001], valid=valid.numpy())
    assert selected == 0


def test_gather_preserves_row_identity_when_global_arm_assignment_flips():
    k, wp = ensure_selection_kernels(), ensure_selection_kernels().wp
    points = np.array([[[-.1, .01, 0], [.1, .01, 0]], [[-.1, .02, 0], [.1, .02, 0]]])
    normals = np.tile([[[1., 0., 0.], [-1., 0., 0.]]], (2, 1, 1))
    forces = np.array([[1., 2., 3., 4., 5., 6.], [.7, .1, .2, .9, .3, .4]])
    poses = np.array([[.1, .2, .3, 1., 0., 0., 0.], [.4, .5, .6, 1., 0., 0., 0.]])
    velocities = np.arange(12).reshape(2, 6).astype(float)
    summary, ids = wp.zeros(40, dtype=float, device="cpu"), wp.zeros(2, dtype=int, device="cpu")
    wp.launch(k.gather, dim=1, inputs=[_array([1, 1], int), _array(points.reshape(-1, 3), wp.vec3),
        _array(normals.reshape(-1, 3), wp.vec3), _array([[10, 11], [20, 21]], int),
        _array(forces), _array(poses), _array(velocities), _array([123., 321.]),
        _array([.003, .001]), _array([.005, .007]), summary, ids], device="cpu")
    actual = summary.numpy()
    np.testing.assert_array_equal(ids.numpy(), [21, 20])
    np.testing.assert_array_equal(actual[:6].reshape(2, 3), points[1, ::-1].astype(np.float32))
    np.testing.assert_array_equal(actual[6:12].reshape(2, 3), normals[1, ::-1].astype(np.float32))
    np.testing.assert_array_equal(actual[12:18].reshape(2, 3), forces[1].reshape(2, 3)[::-1].astype(np.float32))
    np.testing.assert_array_equal(actual[24:31], poses[1].astype(np.float32))
    np.testing.assert_array_equal(actual[31:37], velocities[1].astype(np.float32))
    np.testing.assert_array_equal(actual[37:40], np.array([321., .001, .007], np.float32))


@pytest.fixture
def policy():
    import trimesh
    from planning.lambda_contact_lm import LambdaContactBatchedLM

    return LambdaContactBatchedLM(trimesh.creation.box(extents=[.2, .12, .1]),
        sample_num=48, obj_mass=.1, device="cpu", lm_iterations=12, max_per_arm=8, seed=3)


def test_real_facade_replans_matching_pose_and_forces_from_moving_world_state(policy):
    from scipy.spatial.transform import Rotation
    from tests.test_lambda_batched_lm import _predict, _pose_cost

    rotation = Rotation.from_euler("xyz", [.15, -.10, .2])
    current = np.r_[[.1, -.2, .5], rotation.as_quat()[[3, 0, 1, 2]]]
    target = current.copy()
    target[2] += .03
    inertia = np.diag([.1, .1, .1, .003, .003, .003])
    tips = current[:3] + np.array([[-.3, 0., 0.], [.3, 0., 0.]]) @ rotation.as_matrix().T
    first = policy.initialize_offline(target, current, tips_world=tips,
        object_inertia=inertia, prediction_dt=.06)
    assert first[4]["source"] == "offline_global"
    assert policy.lm.mass == pytest.approx(.1)
    original_points = first[0].copy()
    current[:3] += [.005, -.003, .002]
    tips = current[:3] + np.array([[-.3, 0., 0.], [.3, 0., 0.]]) @ rotation.as_matrix().T
    velocity = np.array([.01, -.02, .03, .02, -.01, .0])
    points, normals, x_plus, cost, info = policy.update_contacts(target, current,
        tips_world=tips, object_inertia=inertia, prediction_dt=.06, v_obj=velocity)
    assert policy.offline_solve_count == 1 and policy.online_solve_count == 1
    assert info["prediction_dt"] == pytest.approx(.06)
    np.testing.assert_array_equal(points, policy.sample_point[info["contact_indices"]])
    np.testing.assert_array_equal(normals, policy.normal[info["contact_indices"]])
    expected_pose, expected_velocity = _predict(current, points, info["force_vectors_local"],
        inertia, .06, velocity, np.array([0., 0., -.981, 0., 0., 0.]))
    np.testing.assert_allclose(x_plus, expected_pose, rtol=3e-5, atol=2e-6)
    np.testing.assert_allclose(info["velocity_next"], expected_velocity, rtol=3e-5, atol=2e-6)
    assert cost == pytest.approx(info["pose_cost"], abs=1e-8)
    assert info["cost_reduction"] == pytest.approx(_pose_cost(current, target) - _pose_cost(x_plus, target), abs=1e-8)
    if info["source"] == "online_cones":
        assert info["cost_reduction"] >= info["reference_cost_reduction"] - info["quality_margin"] - 1e-8
        # Each contact is still associated with its original arm after rotation.
        points_world = points @ rotation.as_matrix().T + current[:3]
        heading = rotation.apply(np.array([[1., 0., 0.], [-1., 0., 0.]]))
        cosine = np.sum((points_world - tips) * heading, axis=1) / np.linalg.norm(points_world - tips, axis=1)
        assert np.all(cosine >= np.cos(np.deg2rad(policy.cone_half_angle_deg)) - 1e-6)
    np.testing.assert_array_equal(first[0], original_points)  # Reused device buffers do not alias host output.
    np.testing.assert_array_equal(policy.last_grasp_result["x_plus_opt"], x_plus)


def test_real_facade_empty_live_cones_keep_offline_fallback_and_reset(policy):
    current = np.array([0., 0., .5, 1., 0., 0., 0.])
    target = current.copy()
    target[2] += .03
    inertia = np.diag([.1, .1, .1, .003, .003, .003])
    tips = current[:3] + np.array([[-.3, 0., 0.], [.3, 0., 0.]])
    first = policy.initialize_offline(target, current, tips_world=tips, object_inertia=inertia, prediction_dt=.06)
    fallback = policy.update_contacts(target, current,
        tips_world=np.tile(current[:3] + policy.surface.geometric_center_local, (2, 1)),
        object_inertia=inertia, prediction_dt=.06)
    assert fallback[4]["source"] == "offline_fallback"
    assert fallback[4]["near_quality_pairs"] == 0
    np.testing.assert_array_equal(fallback[4]["contact_indices"], first[4]["contact_indices"])
    assert policy.offline_solve_count == 1
    policy.reset()
    assert policy.last_grasp_result is None
    assert policy.offline_solve_count == policy.online_solve_count == 0
    np.testing.assert_array_equal(policy._offline_ids.numpy(), [-1, -1])
    np.testing.assert_array_equal(policy._active_ids.numpy(), [-1, -1])
    restarted = policy.update_contacts(target, current, tips_world=tips, object_inertia=inertia, prediction_dt=.06)
    assert restarted[4]["source"] == "offline_global"
    assert policy.offline_solve_count == 1 and policy.online_solve_count == 0


@pytest.mark.parametrize("reverse_assignment", [False, True])
def test_online_empty_admissible_set_keeps_assigned_pair_and_refreshes_prediction(policy, reverse_assignment):
    current = np.array([0., 0., .5, 1., 0., 0., 0.])
    target = current.copy()
    target[2] += .03
    inertia = np.diag([.1, .1, .1, .003, .003, .003])
    tips = current[:3] + np.array([[-.3, 0., 0.], [.3, 0., 0.]])
    points, normals, previous_pose, _, first = policy.initialize_offline(
        target, current, tips_world=tips, object_inertia=inertia, prediction_dt=.06)
    expected_ids = first["contact_indices"].copy()
    if reverse_assignment:
        expected_ids = expected_ids[::-1].copy()
        points, normals = points[::-1].copy(), normals[::-1].copy()
        policy._active_ids.assign(expected_ids)
    # Changed support geometry rejects global fallback, incumbent and cones.
    # This used to leave _choice=-1 and crash in _result.
    policy.set_support_surface([0., 0., .7], [0., 0., 1.])
    current[0] += .02
    next_points, next_normals, predicted, cost, info = policy.update_contacts(
        target, current, tips_world=tips, object_inertia=inertia, prediction_dt=.06)
    np.testing.assert_array_equal(info["contact_indices"], expected_ids)
    np.testing.assert_array_equal(next_points, points)
    np.testing.assert_array_equal(next_normals, normals)
    assert info["source"] == "incumbent_fallback"
    assert info["selection_failed"] and not info["pair_switch_approved"]
    assert not info["solver_failed"]
    assert np.isfinite(predicted).all() and np.isfinite(cost)
    assert not np.allclose(predicted, previous_pose)
    assert policy.offline_solve_count == policy.online_solve_count == 1
    # A later explicit freeze must use the initial pair, not the fallback's
    # potentially different active assignment cached in the one-pair buffer.
    frozen = policy.update_contacts(target, current, tips_world=tips,
        object_inertia=inertia, prediction_dt=.06, freeze_contact_pair=True)
    np.testing.assert_array_equal(frozen[4]["contact_indices"], first["contact_indices"])


def test_real_facade_support_clearance_and_device_support_response(policy):
    current = np.array([0., 0., .5, 1., 0., 0., 0.])
    target = current.copy()
    target[2] += .03
    policy.set_support_surface([0., 0., .45], [0., 0., 2.])
    points, normals, x_plus, cost, info = policy.initialize_offline(target, current,
        tips_world=current[:3] + np.array([[-.3, 0., 0.], [.3, 0., 0.]]),
        object_inertia=np.diag([.1, .1, .1, .003, .003, .003]), prediction_dt=.06)
    pad_centers = points - policy.tip_radius * normals + current[:3]
    assert np.all(pad_centers[:, 2] - .45 - policy.tip_radius >= -1e-7)
    assert np.isfinite(x_plus).all() and np.isfinite(cost)
    assert np.isfinite(policy._support_gap.numpy()).all()
    assert info["cost_reduction"] > 0


def test_loaded_contact_pair_keeps_identity_while_forces_are_reoptimized(policy):
    current = np.array([0., 0., .5, 1., 0., 0., 0.])
    target = current.copy()
    target[2] += .03
    inertia = np.diag([.1, .1, .1, .003, .003, .003])
    points, normals, _, _, first = policy.initialize_offline(target, current,
        tips_world=[[-.3, 0., .5], [.3, 0., .5]], object_inertia=inertia, prediction_dt=.06)
    tips = current[:3] + points - .012 * normals
    points_next, normals_next, predicted, cost, info = policy.update_contacts(target, current,
        tips_world=tips, object_inertia=inertia, prediction_dt=.06, lock_contact_pair=True)
    assert info["contact_pair_locked"]
    np.testing.assert_array_equal(info["contact_indices"], first["contact_indices"])
    np.testing.assert_array_equal(points_next, points)
    np.testing.assert_array_equal(normals_next, normals)
    assert np.isfinite(predicted).all() and np.isfinite(cost)


def test_frozen_pair_survives_contact_loss_and_crossed_arm_assignment(policy, monkeypatch):
    current = np.array([0., 0., .5, 1., 0., 0., 0.])
    target = current.copy()
    target[2] += .03
    inertia = np.diag([.1, .1, .1, .003, .003, .003])
    points, normals, _, _, first = policy.initialize_offline(target, current,
        tips_world=[[-.3, 0., .5], [.3, 0., .5]], object_inertia=inertia, prediction_dt=.06)

    def forbid_search(*args, **kwargs):
        raise AssertionError("Frozen pair must not invoke candidate search")

    monkeypatch.setattr(policy.surface, "online_pairs", forbid_search)
    for tips in ([[.3, 0., .5], [-.3, 0., .5]], [[0., 0., .5], [0., 0., .5]]):
        next_points, next_normals, predicted, cost, info = policy.update_contacts(
            target, current, tips_world=tips, object_inertia=inertia,
            prediction_dt=.06, freeze_contact_pair=True, joint_verify=True)
        np.testing.assert_array_equal(info["contact_indices"], first["contact_indices"])
        np.testing.assert_array_equal(next_points, points)
        np.testing.assert_array_equal(next_normals, normals)
        assert info["source"] == "fixed_pair" and info["contact_pair_frozen"]
        assert not info["pair_switch_approved"] and info["candidate_capacity"] == 1
        assert np.isfinite(predicted).all() and np.isfinite(cost)
    assert policy.offline_solve_count == 1 and policy.online_solve_count == 2
    policy.reset()
    assert policy._fixed is None


@pytest.mark.parametrize("candidate, reduction, expected", [
    ([[.09, 0., 0.], [.3, 0., 0.]], .002, 1),  # summed gain, right arm veto
    ([[.09, 0., 0.], [.11, 0., 0.]], .002, 2),  # both approve
    ([[.09, 0., 0.], [.11, 0., 0.]], .0005, 1),  # worse object response
])
def test_verify_pair_requires_both_arms_and_shared_object_gain(candidate, reduction, expected):
    k, wp = ensure_selection_kernels(), ensure_selection_kernels().wp
    points = np.array([[[.2, 0., 0.], [.2, 0., 0.]],
                       [[.2, 0., 0.], [.2, 0., 0.]], candidate])
    selected = _array([2, 0], int)
    verification = wp.zeros(3, dtype=float, device="cpu")
    wp.launch(k.verify_pair, dim=1, inputs=[_array(points.reshape(-1, 3), wp.vec3),
        _array(np.zeros((2, 3)), wp.vec3), _array([.001, .001, reduction]),
        .002, selected, verification], device="cpu")
    assert selected.numpy()[0] == expected
    assert bool(verification.numpy()[2]) == (expected == 2)
