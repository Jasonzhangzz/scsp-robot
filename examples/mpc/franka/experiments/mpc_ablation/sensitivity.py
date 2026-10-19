"""Matched one-dimensional verify sensitivity sweeps; no simulator imports."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


def add_sensitivity_args(parser):
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--ema', action='store_true',
                       help='Sweep verify_cost EMA beta with matched trial seeds (headless).')
    modes.add_argument('--threshold', action='store_true',
                       help='Sweep p_arm acceptance tightness; keep release threshold fixed (headless).')
    parser.add_argument('--ema-grid', default='0.045,0.09,0.18,0.36,0.72', metavar='B1,B2,...')
    parser.add_argument('--threshold-grid', default='0.15,0.30,0.45,0.60', metavar='T1,T2,...')
    parser.add_argument('--sensitivity-output', default='', metavar='DIR',
                        help='Empty output directory; default output/mpc_sensitivity/<mode>_<timestamp>.')
    parser.add_argument('--plot', action='store_true',
                        help='Save success_rate.svg and success_rate.png after the sweep (Matplotlib).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print sweep commands without loading simulators or creating files.')
    return parser


def parse_grid(spec, *, ema=False, release=.65):
    try:
        values = sorted(set(float(part.strip()) for part in spec.split(',')))
    except ValueError as exc:
        raise ValueError('Grid must be a comma-separated list of numbers') from exc
    for value in values:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Grid values must be finite and in [0, 1]')
        if ema and value == 0:
            raise ValueError('EMA beta must be in (0, 1]')
        if not ema and value > release:
            raise ValueError('Acceptance grid values must be <= --verify-release-threshold')
    return values


def summarize_trials(rows, expected_ids):
    """Require every requested trial exactly once, including unsuccessful trials."""
    if sorted(row['trial'] for row in rows) != list(expected_ids):
        raise ValueError('Incomplete or duplicate trials: expected IDs %s' % list(expected_ids))
    if any(row.get('aborted') for row in rows):
        raise ValueError('Aborted trials cannot be included in success-rate statistics')
    if any(row.get('success') not in (True, False, 0, 1) for row in rows):
        raise ValueError('Each trial must have a boolean success outcome')
    successes = sum(int(row['success']) for row in rows)
    return dict(trials=len(rows), successes=successes, success_rate=successes / len(rows))


def plot_results(rows, output):
    """Use the saved summary alone; physics is never required for plotting."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.scatter([row['value'] for row in rows], [row['success_rate'] for row in rows])
        if len(rows) <= 12:
            ax.set_xticks([row['value'] for row in rows])
        ax.set_xlabel('EMA beta' if rows[0]['mode'] == 'ema' else 'Acceptance tightness threshold')
        ax.set_ylabel('Success rate')
        ax.set_ylim(-.03, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(True, alpha=.25)
        fig.tight_layout()
        for suffix in ('svg', 'png'):
            fig.savefig(Path(output) / ('success_rate.' + suffix), dpi=180)
    finally:
        plt.close(fig)


def _write_results(output, rows, baseline_value):
    baseline = next((row['success_rate'] for row in rows if row['value'] == baseline_value), None)
    for row in rows:
        row['success_rate_delta_vs_baseline'] = (
            None if baseline is None else row['success_rate'] - baseline)
    (output / 'sensitivity_summary.json').write_text(
        json.dumps(rows, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'sensitivity_summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # JSONL .log is convenient for scripts and remains readable without Python.
    with (output / 'sensitivity.log').open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, allow_nan=False) + '\n')


def run_sensitivity(args, rest):
    from .config import configure_ablation
    configure_ablation(args)
    if args.ablation_mode != 'full':
        raise ValueError('--ema / --threshold require the full baseline; remove --a1 ... --a8')
    if '--floating-fingertip' in rest or '--sampling-c3' in rest:
        raise ValueError('Verify sensitivity requires the Franka MPC planner')
    # Read only cohort/beta options here, before importing Torch or Isaac Gym.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--trial-start', type=int, default=0)
    parser.add_argument('--trial_num', type=int, default=100)
    parser.add_argument('--verify_beta', type=float, default=.18)
    cohort, _ = parser.parse_known_args(rest)
    if cohort.trial_start < 0 or cohort.trial_num <= 0:
        raise ValueError('--trial-start must be nonnegative and --trial_num must be positive')
    if not math.isfinite(cohort.verify_beta) or not 0 < cohort.verify_beta <= 1:
        raise ValueError('--verify_beta must be finite and in (0, 1]')
    mode = 'ema' if args.ema else 'threshold'
    parameter = 'verify_beta' if args.ema else 'verify_accept_threshold'
    option = '--verify_beta' if args.ema else '--verify-accept-threshold'
    values = parse_grid(args.ema_grid if args.ema else args.threshold_grid,
                        ema=args.ema, release=args.verify_release_threshold)
    baseline_value = cohort.verify_beta if args.ema else args.verify_accept_threshold
    # Use the client timezone for automatically named experiment folders.
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = Path(args.sensitivity_output or f'output/mpc_sensitivity/{mode}_{stamp}').resolve()
    script = Path(__file__).resolve().parents[1] / 'test_mpc_ablation.py'
    cells = []
    for index, value in enumerate(values):
        directory = output / ('cell_%03d' % index)
        flags = [*rest, '--headless',
                 '--verify-accept-threshold', str(args.verify_accept_threshold),
                 '--verify-release-threshold', str(args.verify_release_threshold),
                 option, str(value),
                 '--ranking-metrics-jsonl', str(directory / 'trials.jsonl'),
                 '--rollout-summary-json', str(directory / 'rollout_summary.json')]
        if not any(flag in rest for flag in ('--sync-planner', '--async-planner', '--trial')):
            flags.append('--sync-planner')
        cells.append(dict(index=index, value=value, directory=str(directory),
                          command=[sys.executable, '-u', str(script), *flags]))
    manifest = dict(mode=mode, parameter=parameter, baseline_value=baseline_value,
                    verify_beta=cohort.verify_beta,
                    verify_accept_threshold=args.verify_accept_threshold,
                    verify_release_threshold=args.verify_release_threshold,
                    trial_start=cohort.trial_start, trial_num=cohort.trial_num,
                    output=str(output), cells=cells)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return 0
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Sensitivity output must be an empty directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'grid.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    expected_ids = range(cohort.trial_start, cohort.trial_start + cohort.trial_num)
    summaries = []
    for cell in cells:
        directory = Path(cell['directory'])
        directory.mkdir()
        print('[%s] %d/%d %s=%g; log: %s' % (
            mode, cell['index'] + 1, len(cells), parameter, cell['value'], directory / 'run.log'), flush=True)
        with (directory / 'run.log').open('w', encoding='utf-8') as log:
            log.write(shlex.join(cell['command']) + '\n')
            log.flush()
            result = subprocess.run(cell['command'], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Sensitivity cell failed (exit %s); see %s' % (
                result.returncode, directory / 'run.log'))
        path = directory / 'trials.jsonl'
        if not path.is_file():
            raise ValueError('Missing trial outcomes: %s; see %s' % (path, directory / 'run.log'))
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        record = dict(mode=mode, parameter=parameter, value=cell['value'],
                      baseline_value=baseline_value, trial_start=cohort.trial_start,
                      **summarize_trials(rows, expected_ids))
        summaries.append(record)
        _write_results(output, summaries, baseline_value)
        print('[%s] %s=%g: %d/%d = %.2f%%' % (
            mode, parameter, cell['value'], record['successes'], record['trials'],
            100 * record['success_rate']), flush=True)
    if args.plot:
        plot_results(summaries, output)
    print('[%s] results: %s' % (mode, output), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replot a saved sensitivity_summary.json without simulation.')
    parser.add_argument('summary', type=Path)
    args = parser.parse_args()
    plot_results(json.loads(args.summary.read_text(encoding='utf-8')), args.summary.parent)
