"""Verify sweep orchestration without requiring an Isaac/GPU installation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.mpc.franka.experiments.mpc_ablation.config import selection_parser
from examples.mpc.franka.experiments.mpc_ablation import sensitivity


def selections(*flags):
    return selection_parser().parse_args(list(flags))


@pytest.mark.parametrize('spec,kwargs', [
    ('', {}), ('nan', {}), ('inf', {}), ('-.1', {}), ('1.1', {}),
    ('0', {'ema': True}), ('.7', {'release': .65}),
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
        assert command[command.index('--verify-release-threshold') + 1] == '0.65'
        assert '--async-planner' in command and '--sync-planner' not in command
        assert '--' + mode not in command
        index = max(i for i, flag in enumerate(command) if flag == option)
        assert float(command[index + 1]) == cell['value']


def fake_trial_run(command, **kwargs):
    path = Path(command[command.index('--ranking-metrics-jsonl') + 1])
    beta = float(command[max(i for i, flag in enumerate(command) if flag == '--verify_beta') + 1])
    rows = [dict(trial=7, success=True), dict(trial=8, success=beta < .3)]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    kwargs['stdout'].write('simulator output\n')
    return SimpleNamespace(returncode=0)


def test_sweep_logs_all_outcomes_and_baseline_deltas(tmp_path, monkeypatch):
    output = tmp_path / 'ema'
    args = selections('--ema', '--ema-grid', '.18,.36', '--sensitivity-output', str(output))
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
    args = selections('--ema', '--sensitivity-output', str(tmp_path))
    monkeypatch.setattr(sensitivity.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=2))
    with pytest.raises(RuntimeError, match='run.log'):
        sensitivity.run_sensitivity(args, [])
    assert not (tmp_path / 'sensitivity_summary.json').exists()


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
