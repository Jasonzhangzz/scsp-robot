"""Verify sweep orchestration without requiring an Isaac/GPU installation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.mpc.franka.experiments.mpc_ablation.config import selection_parser
from examples.mpc.franka.experiments.mpc_ablation import sensitivity


def selections(*flags):
    return selection_parser().parse_args(list(flags))


def baseline_results(tmp_path, *, beta=.18, omit_failure=False):
    path = tmp_path / 'baseline.jsonl'
    rows = [dict(object=obj, trial=(1 if obj == 'stanford_bunny2' and success else 0),
                 success=success, verify_beta=beta, target_type='ground-rotation',
                 physics_backend='isaac', async_planner=False,
                 verify_control_version=sensitivity.VERIFY_CONTROL_VERSION)
            for obj, success in sensitivity.MIXED_CASES if success or not omit_failure]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return str(path)


@pytest.mark.parametrize('spec,kwargs', [
    ('', {}), ('nan', {}), ('inf', {}), ('-.1', {}), ('1.1', {}),
    ('.7', {'release': .65}),
])
def test_invalid_grids(spec, kwargs):
    with pytest.raises(ValueError):
        sensitivity.parse_grid(spec, **kwargs)


@pytest.mark.parametrize('mode,option,grid', [
    ('ema', '--verify_beta', '.36,.18,.18'),
    ('threshold', '--verify-accept-threshold', '.6,.45,.45'),
])
def test_dry_run_forwards_runtime_flags_and_matched_cohort(tmp_path, capsys, mode, option, grid):
    output = tmp_path / mode
    args = selections('--' + mode, '--' + mode + '-grid', grid,
                      '--sensitivity-cohort', 'range',
                      '--sensitivity-output', str(output), '--dry-run')
    sensitivity.run_sensitivity(args, ['--obj', 'elephant', '--trial-start', '7',
                                     '--trial_num', '3', '--viewer', '--async-planner'])
    manifest = json.loads(capsys.readouterr().out)
    assert len(manifest['cells']) == 2
    assert not output.exists()
    for cell in manifest['cells']:
        command = cell['command']
        assert command[command.index('--trial-start') + 1] == '7'
        assert command[command.index('--trial_num') + 1] == '3'
        assert command[command.index('--obj') + 1] == 'elephant'
        assert command.index('--headless') > command.index('--viewer')
        if mode == 'ema':
            assert command[2].endswith('/ik2/test_mpc_isaac.py')
            assert '--verify-accept-threshold' not in command
            assert '--verify-release-threshold' not in command
        else:
            assert command[command.index('--verify-release-threshold') + 1] == '0.65'
        assert '--async-planner' in command and '--sync-planner' not in command
        assert '--' + mode not in command
        index = max(i for i, flag in enumerate(command) if flag == option)
        assert float(command[index + 1]) == cell['value']


def fake_trial_run(command, **kwargs):
    path = Path(command[command.index('--ranking-metrics-jsonl') + 1])
    beta = float(command[max(i for i, flag in enumerate(command) if flag == '--verify_beta') + 1])
    rows = [dict(trial=trial, success=success, verify_beta=beta,
                 verify_control_version=sensitivity.VERIFY_CONTROL_VERSION,
                 verify_mpc_steps=10, verify_mpc_max_abs_error=0.)
            for trial, success in ((7, True), (8, beta < .3))]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    kwargs['stdout'].write('simulator output\n')
    return SimpleNamespace(returncode=0)


def test_sweep_logs_all_outcomes_and_baseline_deltas(tmp_path, monkeypatch):
    output = tmp_path / 'ema'
    args = selections('--ema', '--ema-grid', '.18,.36', '--sensitivity-cohort', 'range',
                      '--sensitivity-output', str(output))
    monkeypatch.setattr(sensitivity.subprocess, 'run', fake_trial_run)
    assert sensitivity.run_sensitivity(args, ['--trial-start', '7', '--trial_num', '2']) == 0
    rows = json.loads((output / 'sensitivity_summary.json').read_text())
    assert [r['success_rate'] for r in rows] == [1., .5]
    assert [r['success_rate_delta_vs_baseline'] for r in rows] == [0., -.5]
    assert all(r['trials'] == 2 for r in rows)
    assert (output / 'sensitivity_summary.csv').read_text().count('\n') == 3
    assert [json.loads(line) for line in (output / 'sensitivity.log').read_text().splitlines()] == rows
    assert (output / 'grid.json').is_file()
    for index in (0, 1):
        directory = output / ('cell_%03d' % index)
        assert 'simulator output' in (directory / 'run.log').read_text()
        assert (directory / 'trials.jsonl').is_file()
    with pytest.raises(ValueError, match='empty directory'):
        sensitivity.run_sensitivity(args, [])


@pytest.mark.parametrize('rows', [
    [dict(trial=7, success=True)],
    [dict(trial=7, success=True), dict(trial=7, success=False)],
    [dict(trial=7, success=True), dict(trial=8, success=False, aborted=True)],
    [dict(trial=7, success=True), dict(trial=8)],
])
def test_incomplete_duplicate_aborted_or_missing_outcomes_are_rejected(rows):
    with pytest.raises(ValueError):
        sensitivity.summarize_trials(rows, range(7, 9))


def test_failed_process_is_not_reported_as_a_task_failure(tmp_path, monkeypatch):
    args = selections('--ema', '--baseline-results', baseline_results(tmp_path),
                      '--sensitivity-output', str(tmp_path / 'sweep'))
    monkeypatch.setattr(sensitivity.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=2))
    with pytest.raises(RuntimeError, match='run.log'):
        sensitivity.run_sensitivity(args, [])
    assert not (tmp_path / 'sweep/cell_001/trials.jsonl').exists()


def test_verify_bypassing_modes_and_invalid_thresholds_are_rejected(tmp_path):
    for flags in [('--ema', '--a1'), ('--threshold', '--a6'),
                  ('--threshold', '--verify-accept-threshold', '.7'),
                  ('--threshold', '--verify-release-threshold', 'nan')]:
        with pytest.raises(ValueError):
            sensitivity.run_sensitivity(selections(*flags, '--dry-run'), [])
    with pytest.raises(SystemExit):
        selections('--ema', '--threshold')


def test_scatter_can_be_generated_from_saved_data_only(tmp_path):
    pytest.importorskip('matplotlib')
    sensitivity.plot_results([dict(mode='ema', value=.18, success_rate=.5),
                              dict(mode='ema', value=.36, success_rate=.75)], tmp_path)
    assert '<svg' in (tmp_path / 'success_rate.svg').read_text()
    assert (tmp_path / 'success_rate.png').read_bytes().startswith(b'\x89PNG')


def test_ema_grid_expands_symmetrically_from_current_baseline(tmp_path, capsys):
    args = selections('--ema', '--dry-run', '--baseline-results', baseline_results(tmp_path, beta=.25),
                      '--sensitivity-output', str(tmp_path / 'dry'))
    sensitivity.run_sensitivity(args, ['--verify_beta', '.25'])
    manifest = json.loads(capsys.readouterr().out)
    assert [cell['value'] for cell in manifest['cells']] == [
        .25, .23, .27, .21, .29, .17, .33, .09, .41]
    assert manifest['trial_num'] == 5
    assert manifest['baseline_successes'] == 4
    assert manifest['entrypoint'].endswith('/ik2/test_mpc_isaac.py')
    assert manifest['selection']['method'] == 'saved_results'
    assert manifest['cells'][0]['runs'] == []
    for run in manifest['cells'][1]['runs']:
        command = run['command']
        assert command[2] == manifest['entrypoint']
        assert '--verify-accept-threshold' not in command
        assert '--verify-release-threshold' not in command
    assert manifest['selection']['required'] == [
        dict(object=obj, baseline_success=success) for obj, success in sensitivity.MIXED_CASES]
    assert not (tmp_path / 'dry').exists()
    assert sensitivity.ema_grid(.18, '.16,.02,.02,.04,.08') == [
        .18, .16, .20, .14, .22, .10, .26, .02, .34]
    assert sensitivity.ema_grid(.05, '.1,.5,1') == [.05, .15, .55]
    for offsets in ('', 'nan', 'inf', '-.1', '0', '2'):
        with pytest.raises(ValueError):
            sensitivity.ema_grid(.18, offsets)


def mixed_simulator(calls):
    def run(command, **kwargs):
        assert command[2].endswith('/ik2/test_mpc_isaac.py')
        assert '--verify-accept-threshold' not in command
        assert '--verify-release-threshold' not in command
        def flag(name):
            return command[max(i for i, v in enumerate(command) if v == name) + 1]
        obj, trial, beta = flag('--obj'), int(flag('--trial-start')), float(flag('--verify_beta'))
        # The first bunny trial fails; later bunny trial succeeds. Other objects
        # deliberately reuse trial 0, so aggregating by trial alone is incorrect.
        success = obj != 'stanford_bunny2' or trial > 0
        if beta > .18:
            success = not success
        calls.append((obj, trial, beta))
        Path(flag('--ranking-metrics-jsonl')).write_text(json.dumps(
            dict(object=obj, trial=trial, success=success, aborted=False, verify_beta=beta,
                 verify_control_version=sensitivity.VERIFY_CONTROL_VERSION,
                 verify_mpc_steps=10, verify_mpc_max_abs_error=0.)) + '\n')
        return SimpleNamespace(returncode=0)
    return run


def test_mixed_cohort_is_selected_once_and_retains_failed_case(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sensitivity.subprocess, 'run', mixed_simulator(calls))
    # Explicit grids still include the baseline first for selection and comparison.
    output = tmp_path / 'sweep'
    args = selections('--ema', '--ema-grid', '.2', '--baseline-results', baseline_results(tmp_path),
                      '--sensitivity-output', str(output))
    assert sensitivity.run_sensitivity(args, []) == 0
    cases = json.loads((output / 'cohort.json').read_text())
    assert [(c['object'], c['trial'], c['baseline_success']) for c in cases] == [
        ('stanford_bunny2', 1, True), ('stanford_bunny2', 0, False),
        ('elephant', 0, True), ('piggy_bank', 0, True), ('rubber_duck', 0, True)]
    assert len(calls) == 5  # only the perturbed five; the baseline is never executed
    assert all(beta != .18 for _, _, beta in calls)
    assert [(o, t) for o, t, b in calls if b == .2] == [(c['object'], c['trial']) for c in cases]
    summaries = json.loads((output / 'sensitivity_summary.json').read_text())
    assert [r['success_rate'] for r in summaries] == [.8, .2]
    assert summaries[1]['success_rate_delta_vs_baseline'] == pytest.approx(-.6)
    assert summaries[1]['success_to_failure'] == 4
    assert summaries[1]['failure_to_success'] == 1
    rows = [json.loads(line) for line in (output / 'cell_001/trials.jsonl').read_text().splitlines()]
    assert len(rows) == 5
    assert len({r['cohort_id'] for r in rows}) == 5
    assert sum(r['baseline_success'] for r in rows) == 4
    manifest = json.loads((output / 'grid.json').read_text())
    assert len(manifest['cells'][0]['baseline_sources']) == 5
    assert len(manifest['cells'][1]['runs']) == 5
    with pytest.raises(ValueError, match='exactly once'):
        sensitivity.summarize_mixed(rows[:-1] + [rows[0]], cases)


def test_missing_baseline_failure_stops_before_sweep(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sensitivity.subprocess, 'run', mixed_simulator(calls))
    args = selections('--ema', '--baseline-results', baseline_results(tmp_path, omit_failure=True),
                      '--sensitivity-output', str(tmp_path / 'sweep'))
    with pytest.raises(ValueError, match='Missing saved baseline outcomes'):
        sensitivity.run_sensitivity(args, [])
    assert calls == []
    assert not (tmp_path / 'sensitivity_summary.json').exists()
    assert not list(tmp_path.glob('cell_*'))


@pytest.mark.parametrize('flags', [
    ['--obj', 'elephant'], ['--trial_num', '5'], ['--async-planner'], ['--trial'],
])
def test_mixed_cohort_does_not_silently_ignore_single_object_options(flags):
    with pytest.raises(ValueError, match='Mixed cohort'):
        sensitivity.run_sensitivity(selections('--ema', '--dry-run'), flags)


def test_ema_commands_use_baseline_parser_and_solver_defaults(tmp_path, monkeypatch):
    import argparse
    monkeypatch.setenv('SCSP_PLANNER_ONLY', '1')
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    parser = _add_rollout_policy_args(argparse.ArgumentParser())
    defaults = parser.parse_args([])
    args = selections('--ema')
    command = sensitivity._trial_command(args, [], tmp_path, 'stanford_bunny2', 3,
                                          '--verify_beta', .16, .18)
    parsed = parser.parse_args(command[3:])
    assert parsed.verify_beta == .16
    assert parsed.obj == 'stanford_bunny2'
    assert parsed.trial_start == 3 and parsed.trial_num == 1
    for name in ('solver_backend', 'mpc_horizon', 'contact_model', 'max_rollout_length',
                 'control_substeps', 'target_type'):
        assert getattr(parsed, name) == getattr(defaults, name)
    with pytest.raises(ValueError, match='only perturbs'):
        sensitivity.run_sensitivity(selections('--ema', '--verify-accept-threshold', '.3'), [])


def test_saved_baseline_metadata_and_beta_mismatch_never_trigger_reruns(tmp_path, monkeypatch):
    source = Path(baseline_results(tmp_path))
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        row.pop('verify_beta')
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (tmp_path / 'config.json').write_text(json.dumps(dict(verify_beta=.18)))
    saved = source.read_bytes()
    monkeypatch.setattr(sensitivity.subprocess, 'run', lambda *a, **kw: pytest.fail('baseline was rerun'))
    args = selections('--ema', '--ema-grid', '.18', '--baseline-results', str(source),
                      '--sensitivity-output', str(tmp_path / 'reuse_only'))
    assert sensitivity.run_sensitivity(args, []) == 0
    assert source.read_bytes() == saved
    assert json.loads((tmp_path / 'reuse_only/sensitivity_summary.json').read_text())[0]['success_rate'] == .8
    with pytest.raises(ValueError, match='Missing saved baseline outcomes'):
        sensitivity.run_sensitivity(args, ['--verify_beta', '.25'])


@pytest.mark.parametrize('changes', [
    {'verify_beta': .5}, {'verify_control_version': 'binary_gate'},
    {'verify_mpc_steps': 0}, {'verify_mpc_max_abs_error': .2},
    {'verify_mpc_max_abs_error': float('nan')},
])
def test_sweep_rejects_missing_or_bypassed_live_ema(changes):
    row = dict(trial=0, verify_beta=.18,
               verify_control_version=sensitivity.VERIFY_CONTROL_VERSION,
               verify_mpc_steps=10, verify_mpc_max_abs_error=0.)
    row.update(changes)
    with pytest.raises(ValueError, match='Unverified EMA control'):
        sensitivity.validate_ema_trials([row], .18)


def test_beta_zero_and_one_are_valid_sweep_endpoints(tmp_path, capsys):
    args = selections('--ema', '--ema-grid', '0,1', '--dry-run',
                      '--baseline-results', baseline_results(tmp_path))
    sensitivity.run_sensitivity(args, [])
    manifest = json.loads(capsys.readouterr().out)
    assert [c['value'] for c in manifest['cells']] == [.18, 0., 1.]


def test_legacy_baseline_is_reference_only_and_never_rerun(tmp_path, monkeypatch):
    source = Path(baseline_results(tmp_path))
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        row.pop('verify_control_version')
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    calls = []
    monkeypatch.setattr(sensitivity.subprocess, 'run', mixed_simulator(calls))
    output = tmp_path / 'sweep'
    args = selections('--ema', '--ema-grid', '.2', '--baseline-results', str(source),
                      '--sensitivity-output', str(output))
    sensitivity.run_sensitivity(args, [])
    summary = json.loads((output / 'sensitivity_summary.json').read_text())
    assert summary[0]['result_role'] == 'historical_reference'
    assert summary[0]['verify_control_version'] is None
    assert summary[1]['verify_control_version'] == sensitivity.VERIFY_CONTROL_VERSION
    assert all(not row['baseline_comparable'] for row in summary)
    assert all(row['success_rate_delta_vs_baseline'] is None for row in summary)
    assert len(calls) == 5
