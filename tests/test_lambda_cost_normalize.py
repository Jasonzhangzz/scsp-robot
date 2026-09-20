"""Cycle-normalized lambda scores keep ranking and verify on [0, 1]."""

import numpy as np
import pytest

from planning.mlqp_point import LambdaContactControlOptimizer


def _stub_optimizer(ema_rate=1.0):
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.rank_score_ema_rate = float(ema_rate)
    opt._rank_score_ids = None
    opt._rank_score_values = None
    opt.last_global_idx = None
    opt.contact_switch_margin_abs = 0.001
    opt.contact_switch_margin_ratio = 0.08
    opt.contact_switch_confidence = 1.0
    opt.pos_coef = 500.0
    opt.ori_coef = 20.0
    opt.last_candidate_deltas = None
    opt.last_candidate_ids = None
    opt.last_delta_lo = 0.0
    opt.last_delta_hi = 0.0
    opt.last_delta_center = 0.0
    opt.last_delta_scale = 0.0
    opt.last_pose_cost_now = 0.0
    opt.last_best_delta = None
    opt.last_candidate_delta_norms = None
    opt.last_candidate_pose_costs = None
    opt.last_cost_lo = 0.0
    opt.last_cost_hi = 0.0
    return opt


def test_unit_range_puts_best_at_zero():
    scaled = LambdaContactControlOptimizer._unit_range_costs(
        [1.0, 3.0, 5.0, 7.0])
    assert scaled[0] == pytest.approx(0.0)
    assert 0.0 < scaled[1] < 1.0
    assert scaled[-1] == pytest.approx(1.0)


def test_unit_range_clips_an_outlier_without_flattening_the_cluster():
    # A raw min/max map would send 1.2 to ~0.004.  The IQR fence keeps
    # the typical cluster spread out and clips the unreachable sample.
    scaled = LambdaContactControlOptimizer._unit_range_costs(
        [1.0, 1.1, 1.2, 1.3, 50.0])
    assert scaled[0] == pytest.approx(0.0)
    assert scaled[1] < scaled[2] < scaled[3]
    assert scaled[2] > 0.25
    assert scaled[-1] == pytest.approx(1.0)


def test_unit_range_preserves_argmin():
    raw = np.array([4.2, 1.7, 9.0, 2.1], dtype=np.float64)
    scaled = LambdaContactControlOptimizer._unit_range_costs(raw)
    assert int(np.argmin(scaled)) == int(np.argmin(raw))


def test_pose_residuals_ignore_force_regularization():
    opt = _stub_optimizer()
    x_d = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # Sample 0 is closer in pose; sample 1 only looks cheaper because the
    # raw NLP cost includes a large force/friction term on sample 0.
    x_plus = [
        np.array([0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.04, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    raw = np.array([20.0, 1.0], dtype=np.float64)
    pos, ori = opt._decompose_pose_residuals(x_d, x_plus, raw)
    assert pos[0] < pos[1]
    assert ori[0] == pytest.approx(0.0)
    assert ori[1] == pytest.approx(0.0)


def test_rescore_ranks_x_plus_cost_reduction_not_raw_nlp():
    opt = _stub_optimizer(ema_rate=1.0)
    x_d = np.array([0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    current = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # Sample 0 moves farthest toward the target.  Raw NLP prefers sample 2
    # because of force regularizers; ranking must follow C(now)-C(x_plus).
    x_plus = [
        np.array([0.08, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.04, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    raw = np.array([9.0, 2.0, 1.0], dtype=np.float64)
    scores = opt._rescore_candidate_costs([0, 1, 2], raw, x_plus, x_d, current)
    assert int(np.argmin(scores)) == 0
    assert scores[0] < scores[1] < scores[2]
    assert opt.has_improving_delta()
    assert opt.has_delta_span()
    assert opt.normalize_cost_delta(opt.last_best_delta) > 0.0
    # ΔC=0 is C(now), which sits above the predicted C(x_plus) band.
    assert opt.normalize_cost_delta(0.0) <= 0.0


def test_small_raw_deltas_still_rank_after_cost_normalize():
    # Near the goal the raw ΔC band is tiny; unit-ranging C(x_plus)
    # still keeps the closer predicted pose first.
    opt = _stub_optimizer(ema_rate=1.0)
    x_d = np.array([0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    current = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    x_plus = [
        np.array([0.010, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.011, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    scores = opt._rescore_candidate_costs(
        [0, 1], np.array([1.0, 0.9]), x_plus, x_d, current)
    n0 = opt.normalize_cost_delta(opt.last_candidate_deltas[0])
    n1 = opt.normalize_cost_delta(opt.last_candidate_deltas[1])
    assert n1 > n0
    assert n1 == pytest.approx(1.0)
    assert n0 == pytest.approx(0.0)
    assert int(np.argmin(scores)) == 1
    assert scores[1] < scores[0]


def test_confidence_norm_uses_the_ranking_cost_band():
    # Ranking unit-ranges C(x_plus) on [8, 12].  C(now)=10, so ΔC=2
    # is the best predicted cost and must map to +1.  The same map
    # sends airborne ΔC=0 (C=10, mid-band) to 0.5, a 0.2 close to
    # 0.1, and a pose-worsening actual below 0.
    opt = _stub_optimizer()
    opt.last_pose_cost_now = 10.0
    opt.last_cost_lo = 8.0
    opt.last_cost_hi = 12.0
    opt.last_delta_center = 2.0
    opt.last_delta_scale = 0.1
    assert opt.normalize_cost_delta(2.0) == pytest.approx(1.0)
    assert opt.normalize_cost_delta(0.0) == pytest.approx(0.5)
    assert opt.normalize_cost_delta(0.2) == pytest.approx(0.55)
    assert opt.normalize_cost_delta(-2.0) == pytest.approx(0.0)
    assert opt.normalize_cost_delta(-6.0) == pytest.approx(-1.0)
    # ΔC / C(now) would have mapped the best close to 0.2.
    assert opt.normalize_cost_delta(2.0) != pytest.approx(0.2)


def test_actual_reduction_uses_the_same_cost_band_as_ranking():
    opt = _stub_optimizer(ema_rate=1.0)
    x_d = np.array([0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    current = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    x_plus = [
        np.array([0.08, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.04, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    scores = opt._rescore_candidate_costs(
        [0, 1], np.array([1.0, 1.0]), x_plus, x_d, current)
    # Ranking score 0 ↔ normalized reduction 1, and the inverse.
    best = int(np.argmin(scores))
    worst = 1 - best
    assert opt.normalize_cost_delta(opt.last_candidate_deltas[best]) == pytest.approx(1.0)
    assert opt.normalize_cost_delta(opt.last_candidate_deltas[worst]) == pytest.approx(0.0)
    # A MuJoCo step that did not move (ΔC=0) must use this band, not
    # ΔC/C(now)≈0, so it is comparable to the predicted close.
    airborne = opt.normalize_cost_delta(0.0)
    assert airborne < opt.normalize_cost_delta(opt.last_candidate_deltas[best])
    # Implied C = C(now) - Δ_act uses the same lo/hi.  The reduction
    # is clipped to [-1, 1]; the raw cost score may sit above 1.
    c_after = opt.last_pose_cost_now - 0.0
    raw = opt.normalize_pose_cost(c_after)
    assert raw == pytest.approx((c_after - opt.last_cost_lo) /
                                (opt.last_cost_hi - opt.last_cost_lo))
    assert airborne == pytest.approx(max(-1.0, min(1.0, 1.0 - raw)))


def test_nn_normalize_is_centered_and_order_preserving():
    vals = np.array([0.03, 0.08, 0.10, 0.13, 0.90])
    center, scale = LambdaContactControlOptimizer._delta_norm_stats(vals)
    assert center == pytest.approx(float(np.median(vals)))
    signed = LambdaContactControlOptimizer._nn_normalize_deltas(vals, center, scale)
    assert np.all(np.diff(signed) > 0.0)
    assert np.max(np.abs(signed)) <= 1.0 + 1e-12
    assert signed[2] == pytest.approx(0.0, abs=0.15)


def test_improving_sample_beats_a_non_improving_neighbour():
    opt = _stub_optimizer(ema_rate=1.0)
    x_d = np.array([0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    current = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    scores = opt._rescore_candidate_costs(
        [0, 1], np.array([1.0, 0.1]),
        [np.array([0.08, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
         np.array([-0.04, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])],
        x_d, current)
    assert int(np.argmin(scores)) == 0
    assert np.isfinite(scores[0])
    assert not np.isfinite(scores[1])


def test_flat_face_near_a_crease_stays_rankable():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.region_max_point_curvature = 0.25
    opt.region_max_mean_curvature = 0.10
    # High worst-neighbour (sees the crease) but low mean (rest of the
    # neighbourhood is coplanar).  This is a usable side face.
    opt.point_curvature = np.array([0.50, 0.05], dtype=np.float64)
    opt.point_curvature_mean = np.array([0.06, 0.02], dtype=np.float64)
    opt.sample_vertex_indices = np.array([0, 1], dtype=np.int32)
    opt.pp = type('P', (), {'vertex_normal_stability': None})()
    kept = set(int(i) for i in opt.filter_rankable_indices([0, 1]))
    assert kept == {0, 1}


def test_no_improving_sample_leaves_confidence_idle():
    opt = _stub_optimizer(ema_rate=1.0)
    x_d = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    current = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    x_plus = [
        np.array([0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.02, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    opt._rescore_candidate_costs(
        [0, 1], np.array([1.0, 2.0]), x_plus, x_d, current)
    assert not opt.has_improving_delta()


def test_plateau_hold_keeps_last_global_on_a_near_tie():
    opt = _stub_optimizer(ema_rate=1.0)
    opt.last_global_idx = 3
    scores = opt._apply_plateau_hold(
        np.array([1, 3, 5], dtype=np.int32),
        np.array([0.00, 0.04, 0.80], dtype=np.float64))
    assert int(np.argmin(scores)) == 1
    assert int(np.asarray([1, 3, 5])[int(np.argmin(scores))]) == 3


def test_unit_range_cost_scores_spread_a_compressed_delta_band():
    # ΔC of 3.5 vs 3.0 looks tied after tanh((Δ-median)/IQR) and the
    # 8% plateau hold would keep the incumbent.  Unit-ranging C(x_plus)
    # (C_now=30) puts them 0.125 apart, so the better patch wins.
    opt = _stub_optimizer(ema_rate=1.0)
    opt.last_global_idx = 1
    x_d = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    current = np.array([0.24495, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # C(now) ≈ 500 * 0.24495**2 ≈ 30.  x_plus 0 / 1 / 2 close 3.5 / 3.0 / 0.1.
    x_plus = [
        np.array([0.22136, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.23238, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.24454, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    scores = opt._rescore_candidate_costs(
        [0, 1, 2], np.array([1.0, 1.0, 1.0]), x_plus, x_d, current)
    assert int(np.argmin(scores)) == 0
    assert float(scores[1] - scores[0]) > 0.08


def test_plateau_hold_releases_when_the_gap_is_clear():
    opt = _stub_optimizer(ema_rate=1.0)
    opt.last_global_idx = 3
    scores = opt._apply_plateau_hold(
        np.array([1, 3, 5], dtype=np.int32),
        np.array([0.00, 0.40, 0.80], dtype=np.float64))
    assert int(np.argmin(scores)) == 0


def test_rescore_follows_the_current_best_delta():
    opt = _stub_optimizer(ema_rate=0.35)
    x_d = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    close = np.array([0.010, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    far = np.array([0.040, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    first = opt._rescore_candidate_costs(
        [0, 1], np.array([1.0, 2.0]), [close, far], x_d)
    assert int(np.argmin(first)) == 0
    opt.last_global_idx = 0
    # A better predicted close must win immediately.  EMA / hold used
    # to keep sample 0, so best_contact lagged argmax ΔC.
    slightly_better = np.array([0.009, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    slightly_worse = np.array([0.011, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    second = opt._rescore_candidate_costs(
        [0, 1], np.array([1.1, 1.0]), [slightly_worse, slightly_better], x_d)
    assert int(np.argmin(second)) == 1
    assert opt.last_best_delta == pytest.approx(opt.last_delta_hi)


def test_ranking_winner_is_the_max_cost_reduction():
    opt = _stub_optimizer(ema_rate=0.35)
    opt.last_global_idx = 2
    x_d = np.array([0.10, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    current = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    x_plus = [
        np.array([0.07, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.09, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.04, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    ]
    scores = opt._rescore_candidate_costs(
        [0, 1, 2], np.array([1.0, 1.0, 1.0]), x_plus, x_d, current)
    assert int(np.argmin(scores)) == 1
    assert opt.last_best_delta == pytest.approx(float(np.nanmax(opt.last_candidate_deltas)))
    assert opt.normalize_cost_delta(opt.last_best_delta) == pytest.approx(1.0)


def test_select_uses_score_span_for_switch_margin():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.sample_point = np.array([
        [0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.10, 0.0, 0.0]])
    opt.normal = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    opt.sample_geodesic = np.linalg.norm(
        opt.sample_point[:, None] - opt.sample_point[None, :], axis=2)
    opt.sample_vertex_indices = np.array([0, 1, 2], dtype=np.int32)
    opt.contact_switch_radius = 0.03
    opt.contact_switch_confidence = 1.0
    opt.contact_switch_margin_abs = 0.001
    opt.contact_switch_margin_ratio = 0.08
    opt.contact_switch_confirm_steps = 1
    opt.lock_contact_patch = False
    opt.last_selected_idx = 0
    opt.last_selected_local = opt.sample_point[0]
    opt.last_global_idx = 0
    opt._pending_selected_idx = None
    opt._pending_selected_count = 0
    opt._blocked_contact_indices = {}
    opt.last_best_idx = 0
    opt.last_best_force = None
    opt.last_best_x_plus = None
    opt.last_best_cost = 0.0
    opt.last_anchor_sample_idx = None
    # Normalized scores: incumbent 0 is 0.05, true best is 0.  Span is 1,
    # so the 8% margin (0.081) keeps the incumbent.
    chosen, min_e, max_e, local = opt._select_contact_candidate(
        [0, 1, 2],
        np.array([0.05, 0.00, 1.00], dtype=np.float64),
        force_buffer=np.ones((3, 3), dtype=np.float32),
        contact_anchor_local=opt.sample_point[0],
        force_required=False)
    assert int(chosen) == 0
    assert min_e == pytest.approx(0.0)
    assert max_e == pytest.approx(1.0)
    assert int(local) == 0


def test_ranking_drops_the_same_high_curvature_tips_as_execution():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.point_curvature = np.array([0.05, 0.40, 0.10], dtype=np.float64)
    opt.point_curvature_mean = np.array([0.02, 0.20, 0.04], dtype=np.float64)
    opt.region_max_point_curvature = 0.25
    opt.region_max_mean_curvature = 0.10
    opt.sample_vertex_indices = np.array([0, 1, 2], dtype=np.int32)
    opt.pp = type('P', (), {'vertex_normal_stability': None})()
    kept = opt.filter_rankable_indices([0, 1, 2])
    assert 1 not in set(int(i) for i in kept)
    assert set(int(i) for i in kept) == {0, 2}
    # Execution uses the same helper.
    exec_kept = opt._filter_contact_policy_indices(
        [0, 1, 2], drop_blocked=False, drop_high_curvature=True)
    assert set(int(i) for i in exec_kept) == set(int(i) for i in kept)


def test_last_global_prefers_stable_face_over_foot_crease():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.point_curvature = np.array([0.18, 0.45, 0.12], dtype=np.float64)
    opt.point_curvature_mean = np.array([0.04, 0.08, 0.03], dtype=np.float64)
    opt.region_max_point_curvature = 0.25
    opt.region_max_mean_curvature = 0.10
    ids = np.array([0, 1, 2], dtype=np.int32)
    costs = np.array([0.02, 0.00, 0.10], dtype=np.float64)
    finite = np.array([True, True, True])
    # Sample 1 is cheaper but a crease/foot tip; keep it rankable
    # (low mean) while last_global must stay on the flank.
    assert not opt._high_curvature_mask(ids)[1]
    assert opt._destination_crease_mask(ids)[1]
    chosen = opt._prefer_stable_ranking_local(ids, costs, finite, 1)
    assert int(ids[chosen]) == 0


def test_nearby_topk_does_not_replace_stable_best_with_a_foot():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.top_k = 2
    opt.point_curvature = np.array([0.18, 0.45], dtype=np.float64)
    opt.region_max_point_curvature = 0.25
    opt.sample_point = np.array([[0.05, 0.00, 0.04], [0.02, 0.00, 0.00]], dtype=np.float64)
    opt.last_topk_ids = np.array([1, 0], dtype=np.int32)
    opt.last_topk_costs = np.array([0.00, 0.02], dtype=np.float64)
    opt.last_candidate_ids = np.array([0, 1], dtype=np.int32)
    opt.last_candidate_deltas = np.array([3.10, 3.12], dtype=np.float64)
    opt.last_global_idx = 0
    # Unit-range put the foot first and the query sits on it.  The
    # destination must stay on the stable flank.
    got = opt.choose_nearby_topk_idx(np.array([0.02, 0.00, 0.00]))
    assert int(got) == 0


def test_last_global_prefers_same_side_face_over_far_com_winner():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.point_curvature = np.array([0.18, 0.04], dtype=np.float64)
    opt.region_max_point_curvature = 0.25
    opt.sample_point = np.array(
        [[0.05, 0.03, 0.03], [-0.05, -0.02, 0.01]], dtype=np.float64)
    opt.rank_query_local = np.array([0.04, 0.03, 0.03], dtype=np.float64)
    ids = np.array([0, 1], dtype=np.int32)
    costs = np.array([0.02, 0.00], dtype=np.float64)
    finite = np.array([True, True])
    opt.last_candidate_ids = ids
    opt.last_candidate_deltas = np.array([3.06, 3.08], dtype=np.float64)
    # Sample 1 is cheaper but only a table-J near-tie on the far face.
    chosen = opt._prefer_stable_ranking_local(ids, costs, finite, 1)
    assert int(ids[chosen]) == 0


def test_last_global_switches_when_far_face_is_clearly_better():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.point_curvature = np.array([0.18, 0.04], dtype=np.float64)
    opt.region_max_point_curvature = 0.25
    opt.sample_point = np.array(
        [[0.05, 0.03, 0.03], [-0.05, -0.02, 0.01]], dtype=np.float64)
    opt.rank_query_local = np.array([0.04, 0.03, 0.03], dtype=np.float64)
    opt.last_candidate_ids = np.array([0, 1], dtype=np.int32)
    opt.last_candidate_deltas = np.array([2.40, 4.20], dtype=np.float64)
    ids = np.array([0, 1], dtype=np.int32)
    costs = np.array([0.40, 0.00], dtype=np.float64)
    finite = np.array([True, True])
    chosen = opt._prefer_stable_ranking_local(ids, costs, finite, 1)
    assert int(ids[chosen]) == 1


def test_nearby_topk_does_not_replace_same_side_best_with_far_face():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.top_k = 2
    opt.point_curvature = np.array([0.18, 0.04], dtype=np.float64)
    opt.region_max_point_curvature = 0.25
    opt.sample_point = np.array(
        [[0.05, 0.03, 0.03], [-0.05, -0.02, 0.01]], dtype=np.float64)
    opt.last_topk_ids = np.array([1, 0], dtype=np.int32)
    opt.last_topk_costs = np.array([0.00, 0.02], dtype=np.float64)
    opt.last_candidate_ids = np.array([0, 1], dtype=np.int32)
    opt.last_candidate_deltas = np.array([3.06, 3.08], dtype=np.float64)
    opt.last_global_idx = 0
    got = opt.choose_nearby_topk_idx(np.array([0.04, 0.03, 0.03]))
    assert int(got) == 0


def test_nearby_topk_keeps_a_clearly_better_far_face():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.top_k = 2
    opt.point_curvature = np.array([0.18, 0.04], dtype=np.float64)
    opt.region_max_point_curvature = 0.25
    opt.sample_point = np.array(
        [[0.05, 0.03, 0.03], [-0.05, -0.02, 0.01]], dtype=np.float64)
    opt.last_topk_ids = np.array([1, 0], dtype=np.int32)
    opt.last_topk_costs = np.array([0.00, 0.40], dtype=np.float64)
    opt.last_candidate_ids = np.array([0, 1], dtype=np.int32)
    opt.last_candidate_deltas = np.array([2.40, 4.20], dtype=np.float64)
    opt.last_global_idx = 1
    got = opt.choose_nearby_topk_idx(np.array([0.04, 0.03, 0.03]))
    assert int(got) == 1


def test_point_curvature_uses_worst_neighbor_on_a_crease():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.curvature_neighbor_k = 4
    # Four samples on one face and one just across a 90° dihedral.
    opt.sample_point = np.array([
        [0.00, 0.00, 0.00],
        [0.01, 0.00, 0.00],
        [0.00, 0.01, 0.00],
        [-0.01, 0.00, 0.00],
        [0.00, 0.00, 0.01],
        [0.20, 0.00, 0.00],
        [0.21, 0.00, 0.00],
        [0.20, 0.01, 0.00],
        [0.19, 0.00, 0.00],
    ], dtype=np.float64)
    opt.normal = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    curv = opt._estimate_point_curvature()
    # Mean of the four neighbours is 0.125 and would pass a 0.25 gate;
    # the orthogonal neighbour must still mark the junction as sharp.
    assert float(opt.point_curvature_mean[0]) == pytest.approx(0.125)
    assert float(curv[0]) == pytest.approx(0.5)
    opt.point_curvature = curv
    opt.region_max_point_curvature = 0.25
    opt.region_max_mean_curvature = 0.10
    opt.sample_vertex_indices = np.arange(len(curv), dtype=np.int32)
    opt.pp = type('P', (), {'vertex_normal_stability': None})()
    kept = set(int(i) for i in opt.filter_rankable_indices(np.arange(len(curv))))
    assert 0 not in kept
    assert 4 not in kept
    assert 5 in kept


def test_crease_samples_are_removed_from_the_contact_set():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.region_max_point_curvature = 0.25
    opt.region_max_mean_curvature = 0.10
    opt.sample_point = np.zeros((4, 3), dtype=np.float64)
    opt.normal = np.zeros((4, 3), dtype=np.float64)
    opt.t1 = np.zeros((4, 3), dtype=np.float64)
    opt.t2 = np.zeros((4, 3), dtype=np.float64)
    opt.point_curvature = np.array([0.05, 0.40, 0.10, 0.08], dtype=np.float64)
    opt.point_curvature_mean = np.array([0.02, 0.20, 0.04, 0.03], dtype=np.float64)
    assert opt._drop_crease_samples()
    assert opt.sample_num == 3
    assert np.allclose(opt.point_curvature, [0.05, 0.10, 0.08])


def test_available_points_drop_a_downward_sole_near_the_floor():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.sample_point = np.array([
        [0.00, 0.00, -0.03],
        [0.00, 0.00, 0.04],
        [0.05, 0.00, 0.00],
    ], dtype=np.float64)
    # Inward normals: sole +z, back -z, side -x.
    opt.normal = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    opt.fingertip_clearance = 0.011
    idx = set(int(i) for i in opt.get_availble_point_idx(
        np.array([0.0, 0.0, 0.04]), np.eye(3),
        np.array([0.10, 0.0, 0.04]), 0.012, heading_filter=False))
    assert 0 not in idx
    assert 1 in idx
    assert 2 in idx


def test_available_points_use_support_plane_for_a_ramp():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.sample_point = np.array([
        [0.00, 0.00, -0.02],
        [0.00, 0.00, 0.04],
        [0.05, 0.00, 0.00],
    ], dtype=np.float64)
    opt.normal = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    opt.fingertip_clearance = 0.011
    pos = np.array([0.0, 0.0, 0.38])
    target = np.array([0.10, 0.0, 0.38])
    # World-up floor_z=0 keeps the raised sole.  A ramp through the object
    # origin puts that same sample below the support plane.
    idx0 = set(int(i) for i in opt.get_availble_point_idx(
        pos, np.eye(3), target, 0.012, heading_filter=False, floor_z=0.0))
    assert 0 in idx0
    idx = set(int(i) for i in opt.get_availble_point_idx(
        pos, np.eye(3), target, 0.012, heading_filter=False,
        support_point=np.array([0.0, 0.0, 0.38]),
        support_normal=np.array([0.0, 0.6, 0.8])))
    assert 0 not in idx
    assert 1 in idx


def test_available_points_use_floor_z_for_a_raised_table():
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.sample_point = np.array([
        [0.00, 0.00, -0.02],
        [0.00, 0.00, 0.04],
        [0.05, 0.00, 0.00],
    ], dtype=np.float64)
    opt.normal = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)
    opt.fingertip_clearance = 0.011
    pos = np.array([0.0, 0.0, 0.38])
    target = np.array([0.10, 0.0, 0.38])
    # Default floor_z=0 leaves the sole in (world z=0.36 > 0.045).
    idx0 = set(int(i) for i in opt.get_availble_point_idx(
        pos, np.eye(3), target, 0.012, heading_filter=False, floor_z=0.0))
    assert 0 in idx0
    idx = set(int(i) for i in opt.get_availble_point_idx(
        pos, np.eye(3), target, 0.012, heading_filter=False, floor_z=0.35))
    assert 0 not in idx
    assert 1 in idx


def test_pose_delta_for_sample_uses_candidate_ids():
    opt = _stub_optimizer()
    opt.last_candidate_ids = np.array([4, 9, 1], dtype=np.int32)
    opt.last_candidate_deltas = np.array([0.02, -0.01, 0.08])
    opt.last_global_idx = 1
    opt.last_best_delta = 0.08
    assert opt.pose_delta_for_sample(9) == pytest.approx(-0.01)
    assert opt.pose_delta_for_sample(1) == pytest.approx(0.08)
    assert opt.pose_delta_for_sample(7) is None
    assert opt.pose_delta_for_sample(None) is None
