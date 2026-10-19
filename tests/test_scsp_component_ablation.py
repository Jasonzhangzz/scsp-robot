"""SCSP component ablations: CSO guidance and CPO stage cost."""

import numpy as np
import pytest

from examples.mpc.franka.experiments.ik2.run_ablation import VARIANTS, ablation_forward_argv
from planning.mlqp_point import LambdaContactControlOptimizer
from planning.mpc_costs import cpo_stage_weights
from planning.object_pose_cost import cost_config, object_cost_np, pose_only_config


def _selector():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.sample_point = np.array([
        [0.0, 0.0, 0.0],
        [0.10, 0.0, 0.0],
        [0.20, 0.0, 0.0],
    ])
    opt.normal = np.tile(np.array([0.0, 0.0, 1.0]), (3, 1))
    opt.sample_geodesic = np.linalg.norm(
        opt.sample_point[:, None] - opt.sample_point[None, :], axis=2)
    opt.sample_vertex_indices = np.array([0, 1, 2], dtype=np.int32)
    opt.contact_switch_radius = 0.03
    opt.contact_switch_confidence = 1.0
    opt.contact_switch_margin_abs = 0.001
    opt.contact_switch_margin_ratio = 0.2
    opt.contact_switch_confirm_steps = 5
    opt.lock_contact_patch = True
    opt.last_selected_idx = 0
    opt.last_selected_local = opt.sample_point[0].copy()
    opt.last_global_idx = 0
    opt._pending_selected_idx = None
    opt._pending_selected_count = 0
    opt._blocked_contact_indices = {}
    opt.last_best_idx = 0
    opt.last_best_force = None
    opt.rank_query_local = np.array([0.20, 0.01, 0.0])
    return opt


def _costs():
    # Index 0 is the cost minimum. Index 2 is nearest the query.
    return np.array([0.0, 0.4, 1.0], dtype=np.float64)


def test_stage_weights_match_the_component_table():
    assert cpo_stage_weights(1.0, "full") == (1.0, 0.0, True)
    assert cpo_stage_weights(1.0, "no_object_center") == (0.0, 1.0, True)
    center, reference, phased = cpo_stage_weights(1.0, "direct")
    assert center == pytest.approx(1.0)
    assert reference == pytest.approx(1.0)
    assert phased is False
    assert cpo_stage_weights(0.25, "full") == (0.25, 0.75, True)


def test_pose_only_keeps_yaw_that_stable_pose_drops():
    yaw = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    pose = np.r_[[0.4, 0.1, 0.4], yaw]
    target = np.r_[[0.4, 0.1, 0.4], [1.0, 0.0, 0.0, 0.0]]
    stable = cost_config(500.0, 0.0, 20.0)
    plain = pose_only_config(500.0, 20.0)
    assert object_cost_np(pose, target, stable) == pytest.approx(0.0, abs=1e-8)
    assert object_cost_np(pose, target, plain) == pytest.approx(10.0, abs=1e-6)


def test_nearest_guidance_ignores_the_cost_argmin_and_the_incumbent():
    opt = _selector()
    opt.cso_guidance = "nearest"
    chosen, min_cost, max_cost, local = opt._select_contact_candidate(
        [0, 1, 2], _costs(), force_buffer=np.ones((3, 3), dtype=np.float32),
        contact_anchor_local=opt.sample_point[0], force_required=False)
    assert int(chosen) == 2
    assert int(local) == 2
    assert min_cost == pytest.approx(0.0)
    assert max_cost == pytest.approx(1.0)
    assert opt.choose_nearby_topk_idx(opt.rank_query_local) == 2


def test_random_guidance_draws_from_the_feasible_set():
    opt = _selector()
    opt.cso_guidance = "random"
    opt._cso_rng = np.random.default_rng(0)
    expected = int(np.random.default_rng(0).integers(3))
    chosen, _, _, _ = opt._select_contact_candidate(
        [0, 1, 2], _costs(), force_buffer=np.ones((3, 3), dtype=np.float32),
        force_required=False)
    assert int(chosen) == expected
    assert opt.choose_nearby_topk_idx(opt.rank_query_local) == expected


def test_cost_guidance_still_uses_the_argmin():
    opt = _selector()
    opt.cso_guidance = "cso"
    chosen, _, _, _ = opt._select_contact_candidate(
        [0, 1, 2], _costs(), force_buffer=np.ones((3, 3), dtype=np.float32),
        force_required=False)
    assert int(chosen) == 0


def _cost_param(variant):
    from examples.mpc.franka.runtime.params import ExplicitMPCParams
    param = ExplicitMPCParams.__new__(ExplicitMPCParams)
    param.__dict__.update(
        n_qpos_=10, n_qvel_=9, n_cmd_=3, max_ncon_=10,
        smooth_contact_detour=False, quadratic_contact_track=True,
        attract_coef=20.0, reject_dis=0.001, reject_coef=0.0,
        contact_coef=20.0, contact_cost_param=1.0, spline_escape_cost=True,
        field_cost_weight=0.0, pos_coef=500.0, ori_coef=20.0,
        runtime_object_cost_=True, cpo_variant=variant,
        object_cost_config_=cost_config(500.0, 20.0),
    )
    return param


def _stage_cost(param, ee, contact, verify, virtual):
    from planning.mpc_costs import pack_cost_params
    path, _final = param.init_cost_fns()
    obj = np.array([0.40, 0.10, 0.40, 1.0, 0.0, 0.0, 0.0])
    packed = pack_cost_params(
        "param", param, path, target_p=obj[:3], target_q=obj[3:],
        verify_cost_param=verify, virtual_point=virtual, contact_point=contact,
        contact_reference=obj)
    return float(path(np.r_[obj, ee], np.zeros(3), packed))


def test_cpo_ablations_change_only_the_named_stage_term():
    full = _cost_param("full")
    no_center = _cost_param("no_object_center")
    direct = _cost_param("direct")
    near = np.array([0.42, 0.10, 0.40])
    far = np.array([0.70, 0.10, 0.40])
    # Place phase, reference sitting on the fingertip: only object-center remains.
    full_near = _stage_cost(full, near, near, 1.0, near)
    full_far = _stage_cost(full, far, far, 1.0, far)
    plain_near = _stage_cost(no_center, near, near, 1.0, near)
    plain_far = _stage_cost(no_center, far, far, 1.0, far)
    assert full_far > full_near + 1.0
    assert plain_near == pytest.approx(plain_far, abs=1e-6)
    # Lift still tracks the virtual point when the object-center term is off.
    lifted = _stage_cost(no_center, near, near, 0.0, far)
    held = _stage_cost(no_center, near, near, 0.0, near)
    assert lifted != pytest.approx(held, abs=1e-4)
    # Direct cost ignores the lift/place trigger and still tracks the reference.
    on_reference = _stage_cost(direct, near, near, 0.0, far)
    off_reference = _stage_cost(direct, near, far, 1.0, near)
    assert _stage_cost(direct, near, near, 1.0, far) == pytest.approx(on_reference, abs=1e-6)
    assert off_reference > on_reference + 1.0


def test_forwarded_ablation_command_selects_one_table_cell():
    argv = ablation_forward_argv(
        "cso-pose-only", "flipping", ["--headless", "--trial-num", "2", "--cso-guidance", "random"])
    assert argv[argv.index("--target-type") + 1] == "ground-rotation"
    assert argv[-8:] == [
        "--planner", "mpc",
        "--cso-guidance", "pose",
        "--cpo-variant", "full",
        "--ablation-variant", "cso-pose-only"]
    rotation = ablation_forward_argv("cpo-direct", "rotation", [])
    assert "--no_random_init_tilt" in rotation
    assert rotation[rotation.index("--target-type") + 1] == "rotation"
    assert rotation[rotation.index("--cpo-variant") + 1] == "direct"
    assert set(VARIANTS) == {
        "full", "cso-pose-only", "cso-random", "cso-nearest",
        "cpo-no-object-center", "cpo-direct"}
