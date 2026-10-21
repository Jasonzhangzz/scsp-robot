"""Exercise the final quality gate, including staged contact-retention bypasses."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.mpc.franka.experiments.mpc_ablation import sensitivity
from examples.mpc.franka.experiments.mpc_ablation.config import selection_parser
from planning.verify_control import VerifyControlStats, acceptance_decision


def test_quality_center_matches_tightness_and_retention_boundaries():
    options = dict(same_patch=True, near_arm=True)
    assert acceptance_decision(.55, .44, **options)
    assert not acceptance_decision(.55, .45, **options)
    assert acceptance_decision(.55, .64, holding=True, **options)
    assert not acceptance_decision(.55, .65, holding=True, **options)
    # Endpoints must also override best-sample exceptions and hold hysteresis.
    assert acceptance_decision(0, 1)
    assert not acceptance_decision(1, 0, is_best_sample=True, holding=True, **options)
    for value in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError):
            acceptance_decision(value, .5)


def test_staged_controller_applies_final_gate_and_preserves_ema(monkeypatch):
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from planning.mpc_explicit import _build_mpc_planner_runtime, handle_mpc_request
    record = json.loads((Path(__file__).parent / 'fixtures/bunny_pose_contact.json').read_text())
    for threshold in (0., .55, 1.):
        init = copy.deepcopy(record)
        init['args'].update(verify_quality_threshold=threshold, solver_backend='acados',
                            rollout_trace_jsonl='')
        args, param, mpc, trackers = _build_mpc_planner_runtime(init)
        trackers['object_cost_stages'].stage = 'pose'
        result = handle_mpc_request(args, param, mpc, trackers, dict(init['msg'], dwell=None))
        info = result['policy']['value_info']
        assert info['acceptance_gate_applied']
        assert info['acceptance_threshold'] == threshold
        if threshold in (0., 1.):
            assert info['accept_p_arm'] is (threshold == 0.)
        if threshold == 1.:
            np.testing.assert_allclose(result['policy']['p_arm_world'],
                                       result['policy']['best_contact_track_world'])
        stats = VerifyControlStats()
        stats.observe(result)
        row = stats.summary()
        assert row['acceptance_gate_steps'] == 1
        assert row['acceptance_endpoint_violations'] == 0
        assert row['verify_mpc_max_abs_error'] == 0.


@pytest.mark.parametrize('workers', [1, 3])
def test_acceptance_sweep_reruns_center_and_pairs_exact_ema_cases(tmp_path, capsys, monkeypatch, workers):
    source = tmp_path / 'baseline.jsonl'
    cases = [dict(object=obj, trial=i, success=success, verify_beta=.18)
             for i, (obj, success) in enumerate(sensitivity.MIXED_CASES)]
    source.write_text(''.join(json.dumps(row)+'\n' for row in cases))
    output = tmp_path / 'sweep'
    args = selection_parser().parse_args(['--acceptance', '--dry-run', '--baseline-results',
                                         str(source), '--sensitivity-output', str(output),
                                         '--sensitivity-workers', str(workers)])
    sensitivity.run_sensitivity(args, [])
    manifest = json.loads(capsys.readouterr().out)
    assert [c['value'] for c in manifest['cells']] == [
        .55, .53, .57, .5, .6, .45, .65, .3, .8, 0., 1.]
    assert manifest['trial_num'] == 5 and manifest['center_rerun']
    assert sum(len(c['runs']) for c in manifest['cells']) == 55
    assert not output.exists()

    def simulator(command, **kwargs):
        def flag(name):
            return command[command.index(name)+1]
        threshold = float(flag('--verify-quality-threshold'))
        assert command[2].endswith('/ik2/test_mpc_isaac.py')
        assert float(flag('--verify_beta')) == .18
        Path(flag('--ranking-metrics-jsonl')).write_text(json.dumps(dict(
            object=flag('--obj'), trial=int(flag('--trial-start')), success=threshold < 1,
            verify_quality_threshold=threshold, acceptance_threshold=threshold,
            acceptance_gate_version='quality_gate_v1', acceptance_gate_steps=10,
            acceptance_endpoint_violations=0))+'\n')
        return SimpleNamespace(returncode=0)
    args.dry_run = False
    monkeypatch.setattr(sensitivity.subprocess, 'run', simulator)
    sensitivity.run_sensitivity(args, [])
    rows = json.loads((output / 'sensitivity_summary.json').read_text())
    assert len(rows) == 11
    assert rows[0]['successes'] == 5  # fresh center, saved selection had 4
    assert rows[-1]['success_rate'] == 0
    assert all(row['trials'] == 5 for row in rows)
    assert rows[-1]['success_rate_delta_vs_baseline'] == -1
    with pytest.raises(ValueError, match='Unverified acceptance'):
        sensitivity.validate_acceptance_trials([dict(trial=0)], 0)
