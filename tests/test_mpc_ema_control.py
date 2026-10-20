"""The default staged Franka controller must consume its continuous EMA score."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from planning.verify_control import VerifyControlStats


def test_audit_measures_applied_weights_and_detects_a_binary_override():
    stats = VerifyControlStats()
    stats.observe(dict(verify_cost=.2, verify_cost_applied=None))
    stats.observe(dict(verify_cost=.2, verify_cost_applied=.2))
    stats.observe(dict(verify_cost=.4, verify_cost_applied=1.))
    row = stats.summary()
    assert row['verify_mpc_steps'] == 2
    assert row['verify_mpc_weight_min'] == .2
    assert row['verify_mpc_weight_max'] == 1.
    assert row['verify_mpc_max_abs_error'] == .6


def test_real_staged_contact_mpc_receives_beta_dependent_scores(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from planning.mpc_explicit import _build_mpc_planner_runtime, handle_mpc_request
    record = json.loads((Path(__file__).parent / 'fixtures/bunny_pose_contact.json').read_text())
    weights, objectives = [], []
    for beta in (0., .18, 1.):
        init = copy.deepcopy(record)
        init['args'].update(verify_beta=beta, solver_backend='acados', rollout_trace_jsonl='')
        args, param, mpc, trackers = _build_mpc_planner_runtime(init)
        trackers['object_cost_stages'].stage = 'pose'
        trackers['model_cost_conf'].accum = .8 * trackers['model_cost_conf'].threshold
        original = mpc.plan_once
        captured = []
        def solve(**kwargs):
            captured.append(kwargs['verify_cost_param'])
            costs = mpc._cost_params(**kwargs, contact_reference=kwargs['curr_x'][:7])
            # The world-fixed approach and object-attached press targets can
            # coincide at the observation. Probe a moving-object horizon state,
            # where the two terms must respond differently to the EMA weight.
            predicted = kwargs['curr_x'].copy()
            predicted[0] += .01
            predicted[7] += .005
            objectives.append(float(mpc._path_cost(predicted, np.zeros(3), costs)))
            return original(**kwargs)
        monkeypatch.setattr(mpc, 'plan_once', solve)
        result = handle_mpc_request(args, param, mpc, trackers, dict(init['msg'], dwell=None))
        assert not result['mpc_skipped'] and result['solver_backend'] == 'acados'
        assert captured == pytest.approx([.8 * beta])
        assert result['verify_cost_applied'] == pytest.approx(.8 * beta)
        stats = VerifyControlStats()
        stats.observe(result)
        assert stats.summary()['verify_mpc_steps'] == 1
        assert stats.summary()['verify_mpc_max_abs_error'] == 0.
        assert np.isfinite(result['action']).all()
        weights.append(captured[0])
    assert weights == pytest.approx([0., .144, .8])
    assert abs(objectives[-1] - objectives[0]) > 1e-6
