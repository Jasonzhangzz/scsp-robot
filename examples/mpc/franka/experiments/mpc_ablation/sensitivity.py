"""Matched one-dimensional verify sensitivity sweeps; no simulator imports."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

from planning.verify_control import VERIFY_CONTROL_VERSION


def add_sensitivity_args(parser):
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--ema', action='store_true',
                       help='Sweep verify_cost EMA beta with matched trial seeds (headless).')
    modes.add_argument('--threshold', action='store_true',
                       help='Sweep p_arm acceptance tightness; keep release threshold fixed (headless).')
    modes.add_argument('--acceptance', action='store_true',
                       help='Sweep final verify quality threshold: 0 accepts all, 1 selects best only.')
    parser.add_argument('--acceptance-grid', default=None, metavar='T1,T2,...')
    parser.add_argument('--acceptance-offsets', default='0.02,0.05,0.10,0.25', metavar='D1,D2,...')
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument('--ema-grid', default=None, metavar='B1,B2,...',
                      help='Explicit beta values; default: expand around the current --verify_beta.')
    grid.add_argument('--ema-offsets', default='0.02,0.04,0.08,0.16', metavar='D1,D2,...',
                      help='Absolute +/- beta offsets, smallest first; omit values outside [0, 1].')
    parser.add_argument('--sensitivity-cohort', choices=('mixed', 'range'), default=None,
                        help='mixed: reuse 4 saved successes and 1 failure across four objects; '
                             'range: use the forwarded object and trial range. '
                             'Default: mixed for EMA/acceptance, range for legacy threshold sweeps.')
    parser.add_argument('--baseline-results', action='append', default=[], metavar='FILE_OR_DIR',
                        help='Saved baseline JSONL/JSON file or directory; repeat in priority order. '
                             'Default: existing baseline/RS results in this repository. Never rerun baseline.')
    parser.add_argument('--threshold-grid', default='0.15,0.30,0.45,0.60', metavar='T1,T2,...')
    parser.add_argument('--sensitivity-output', default='', metavar='DIR',
                        help='Empty output directory; default output/mpc_sensitivity/<mode>_<timestamp>.')
    parser.add_argument('--sensitivity-workers', type=int, default=1,
                        help='Independent synchronous simulations per mixed-cohort cell (default 1).')
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
        if not ema and value > release:
            raise ValueError('Acceptance grid values must be <= --verify-release-threshold')
    return values


def ema_grid(baseline, offsets):
    """Run the baseline first, then increasingly distant additive perturbations."""
    try:
        radii = sorted(set(float(part.strip()) for part in offsets.split(',')))
    except ValueError as exc:
        raise ValueError('EMA offsets must be a comma-separated list of numbers') from exc
    if any(not math.isfinite(radius) or radius <= 0 for radius in radii):
        raise ValueError('EMA offsets must be finite and positive')
    values = [baseline]
    for radius in radii:
        for value in (round(baseline - radius, 12), round(baseline + radius, 12)):
            if 0 <= value <= 1 and value not in values:
                values.append(value)
    if len(values) == 1:
        raise ValueError('EMA offsets produce no valid perturbations around the baseline')
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
    rows = [row for row in rows if row.get('result_role') != 'historical_reference']
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        ax.scatter([row['value'] for row in rows], [row['success_rate'] for row in rows])
        if len(rows) <= 12:
            ax.set_xticks([row['value'] for row in rows])
        ax.set_xlabel({'ema': 'EMA beta', 'threshold': 'Acceptance tightness threshold',
                       'acceptance': 'Verify acceptance quality threshold'}[rows[0]['mode']])
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
            None if baseline is None or not row.get('baseline_comparable', True)
            else row['success_rate'] - baseline)
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


MIXED_CASES = (
    ('stanford_bunny2', True), ('stanford_bunny2', False),
    ('elephant', True), ('piggy_bank', True), ('rubber_duck', True),
)


def _experiment_script(args):
    if args.ema or args.acceptance:
        return Path(__file__).resolve().parents[2] / 'ik2' / 'test_mpc_isaac.py'
    return Path(__file__).resolve().parents[1] / 'test_mpc_ablation.py'


def _threshold_flags(args):
    # EMA must use the actual baseline runtime, whose acceptance thresholds
    # are internal defaults. Only threshold sweeps need the ablation hooks.
    if args.ema or args.acceptance:
        return []
    return ['--verify-accept-threshold', str(args.verify_accept_threshold),
            '--verify-release-threshold', str(args.verify_release_threshold)]


def _trial_command(args, rest, directory, obj, trial, option, value, beta):
    script = _experiment_script(args)
    return [sys.executable, '-u', str(script), *rest, '--headless',
            '--sync-planner', '--num-envs', '1',
            '--obj', obj, '--trial-start', str(trial), '--trial_num', '1',
            '--verify_beta', str(beta),
            *_threshold_flags(args),
            option, str(value),
            '--ranking-metrics-jsonl', str(directory / 'trials.jsonl'),
            '--rollout-summary-json', str(directory / 'rollout_summary.json')]


def _execute(command, directory):
    directory.mkdir(parents=True)
    with (directory / 'run.log').open('w', encoding='utf-8') as log:
        log.write(shlex.join(command) + '\n')
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Sensitivity cell failed (exit %s); see %s' % (
            result.returncode, directory / 'run.log'))
    path = directory / 'trials.jsonl'
    if not path.is_file():
        raise ValueError('Missing trial outcomes: %s; see %s' % (path, directory / 'run.log'))
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def validate_ema_trials(rows, beta):
    """Fail rather than report a sweep whose EMA never reached contact MPC."""
    for row in rows:
        if (row.get('verify_control_version') != VERIFY_CONTROL_VERSION
                or row.get('verify_beta') != beta
                or row.get('verify_mpc_steps', 0) <= 0
                or not math.isfinite(row.get('verify_mpc_max_abs_error', float('inf')))
                or not 0 <= row['verify_mpc_max_abs_error'] <= 1e-9):
            raise ValueError('Unverified EMA control for trial %s: require the requested beta, '
                             'contact MPC steps, and equality of EMA score and applied MPC weight'
                             % row.get('trial'))


def _mixed_trial(command, directory, obj, trial, expected_beta=None, expected_acceptance=None):
    rows = _execute(command, directory)
    summarize_trials(rows, [trial])
    if rows[0].get('object') != obj:
        raise ValueError('Unexpected object in %s: expected %s' % (directory, obj))
    if expected_beta is not None:
        validate_ema_trials(rows, expected_beta)
    if expected_acceptance is not None:
        validate_acceptance_trials(rows, expected_acceptance)
    return rows[0]


def validate_acceptance_trials(rows, threshold):
    for row in rows:
        if (row.get('acceptance_gate_version') != 'quality_gate_v1'
                or row.get('acceptance_threshold') != threshold
                or row.get('verify_quality_threshold') != threshold
                or row.get('acceptance_gate_steps', 0) <= 0
                or row.get('acceptance_endpoint_violations') != 0):
            raise ValueError('Unverified acceptance gate for trial %s' % row.get('trial'))


def summarize_mixed(rows, cases):
    """Trial numbers repeat across objects; validate the full paired identity."""
    expected = [(case['object'], case['trial']) for case in cases]
    actual = [(row['object'], row['trial']) for row in rows]
    if sorted(actual) != sorted(expected) or len(set(actual)) != len(actual):
        raise ValueError('Mixed cohort must contain each selected object/trial exactly once')
    # Validate outcomes without conflating trial 0 of two different objects.
    stats = summarize_trials([dict(row, trial=i) for i, row in enumerate(rows)], range(len(cases)))
    baseline = {(case['object'], case['trial']): case['baseline_success'] for case in cases}
    stats['success_to_failure'] = sum(baseline[key] and not row['success']
                                      for key, row in zip(actual, rows))
    stats['failure_to_success'] = sum(not baseline[key] and row['success']
                                      for key, row in zip(actual, rows))
    return stats


def default_baseline_sources():
    root = Path(__file__).resolve().parents[5]
    sources = []
    for pattern in ('output/ablation/ema_baseline_full_*/selection',
                    'outputs/rs_table_vi_contact5_*/flipping',
                    'outputs/rs_current_*/baseline'):
        sources.extend(sorted(root.glob(pattern), reverse=True))
    return sources


def load_mixed_cases(sources, cohort):
    """Choose saved outcomes only. Missing cases must never launch a simulator."""
    selected, seen = {}, set()
    for source in sources:
        source = Path(source).resolve()
        if not source.exists():
            raise ValueError('Missing baseline results: ' + str(source))
        paths = ([source] if source.is_file() else
                 sorted([*source.rglob('trials.jsonl'), *source.rglob('baseline_summary.json')]))
        for path in paths:
            if path.suffix not in ('.jsonl', '.json'):
                raise ValueError('Baseline results must be JSONL or JSON: ' + str(path))
            if path.suffix == '.jsonl':
                rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
                        if line.strip()]
            else:
                data = json.loads(path.read_text(encoding='utf-8'))
                rows = data if isinstance(data, list) else [data]
            # Older outcomes omit beta; recover it from the saved run config.
            metadata = {}
            for parent in path.parents:
                grid = parent / 'grid.json'
                if grid.is_file():
                    data = json.loads(grid.read_text(encoding='utf-8'))
                    if isinstance(data, dict):
                        metadata.update(data)
                    break
            for name in ('source_config.json', 'config.json'):
                config = path.parent / name
                if config.is_file():
                    metadata.update(json.loads(config.read_text(encoding='utf-8')))
            for index, original in enumerate(rows):
                row = dict(original)
                obj = row.get('object', row.get('obj'))
                trial = row.get('trial', row.get('seed'))
                context = {**metadata, **row}
                if (obj not in {name for name, _ in MIXED_CASES}
                        or not isinstance(trial, int) or trial < 0
                        or row.get('aborted') or row.get('success') not in (True, False, 0, 1)
                        or context.get('verify_beta') != cohort.verify_beta
                        or context.get('target_type', 'ground-rotation') != cohort.target_type
                        or context.get('physics_backend', 'isaac') != cohort.physics_backend
                        or context.get('async_planner', False)
                        or context.get('ablation_mode', 'full') != 'full'
                        or context.get('cso_guidance', 'cso') != 'cso'
                        or context.get('cpo_variant', 'full') != 'full'):
                    continue
                identity = (obj, trial)
                if identity in seen:
                    continue
                seen.add(identity)
                key = (obj, bool(row['success']))
                if key in MIXED_CASES and key not in selected:
                    row.update(object=obj, trial=trial)
                    selected[key] = dict(object=obj, trial=trial, baseline_success=key[1],
                                         verify_control_version=row.get('verify_control_version'),
                                         source=str(path), source_record=index, row=row,
                                         id='%s_%s' % (obj, 'success' if key[1] else 'failed'))
    missing = [key for key in MIXED_CASES if key not in selected]
    if missing:
        raise ValueError('Missing saved baseline outcomes %s for beta=%g; '
                         'provide matching --baseline-results. Baseline will not be rerun.' % (
                             missing, cohort.verify_beta))
    return [selected[key] for key in MIXED_CASES]


def _run_mixed(args, rest, cohort, output, manifest, option, values):
    # Mixing object seeds only makes sense when the caller does not also request
    # a single-object cohort. Fail explicitly instead of silently ignoring flags.
    conflicts = {'--obj', '--trial_num', '--async-planner', '--trial'}
    if any(flag.split('=')[0] in conflicts for flag in rest):
        raise ValueError('Mixed cohort selects objects/seeds and uses synchronous planning; '
                         'remove --obj/--trial_num/--async-planner/--trial, '
                         'or use --sensitivity-cohort range')
    baseline = manifest['baseline_value']
    sources = args.baseline_results or default_baseline_sources()
    cases = load_mixed_cases(sources, cohort)
    baseline_comparable = not args.ema or all(
        case['verify_control_version'] == VERIFY_CONTROL_VERSION for case in cases)
    # Include the saved baseline for comparison, without executing it again.
    values = [baseline, *[value for value in values if value != baseline]]
    manifest.update(cohort='mixed', trial_num=5, baseline_successes=4,
                    verify_control_version=VERIFY_CONTROL_VERSION if args.ema else None,
                    baseline_comparable=baseline_comparable,
                    selection=dict(method='saved_results', sources=[str(Path(p).resolve()) for p in sources],
                                   required=[dict(object=obj, baseline_success=success)
                                             for obj, success in MIXED_CASES]),
                    cells=[dict(index=i, value=value) for i, value in enumerate(values)])
    if args.acceptance:
        manifest.update(baseline_source_role='cohort_selection_only',
                        center_rerun=True, acceptance_gate_version='quality_gate_v1',
                        acceptance_hysteresis_width=.20,
                        endpoint_semantics={'0': 'always accept p_arm', '1': 'always select best contact'})
    manifest['cases'] = [{k: v for k, v in case.items() if k != 'row'} for case in cases]
    for cell in manifest['cells']:
        directory = output / ('cell_%03d' % cell['index'])
        cell['directory'] = str(directory)
        cell['runs'] = [dict(case_id=case['id'], command=_trial_command(
            args, rest, directory / case['id'], case['object'], case['trial'],
            option, cell['value'], cohort.verify_beta)) for case in cases] if cell['index'] or args.acceptance else []
        if not cell['index']:
            cell['baseline_sources'] = [case['source'] for case in cases]
    if args.dry_run:
        print(json.dumps(manifest, indent=2, allow_nan=False))
        return 0
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Sensitivity output must be an empty directory: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'grid.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (output / 'cohort.json').write_text(json.dumps(manifest['cases'], indent=2) + '\n', encoding='utf-8')
    for case in cases:
        print('[cohort] reused %s trial=%d success=%s from %s' % (
            case['object'], case['trial'], case['baseline_success'], case['source']), flush=True)
    summaries = []
    for cell in manifest['cells']:
        directory = Path(cell['directory'])
        directory.mkdir()
        rows = []
        parallel_rows = None
        if cell['runs'] and args.sensitivity_workers > 1:
            def evaluate(entry):
                case, run = entry
                return _mixed_trial(run['command'], directory / case['id'],
                    case['object'], case['trial'],
                    expected_beta=cell['value'] if args.ema else None,
                    expected_acceptance=cell['value'] if args.acceptance else None)
            with ThreadPoolExecutor(max_workers=args.sensitivity_workers) as pool:
                parallel_rows = list(pool.map(evaluate, zip(cases, cell['runs'])))
        for case, run in zip(cases, cell['runs'] or [None] * len(cases)):
            print('[%s] %d/%d %s=%g %s trial=%d' % (
                manifest['mode'], cell['index'] + 1, len(values), manifest['parameter'], cell['value'],
                case['object'], case['trial']), flush=True)
            row = parallel_rows[len(rows)] if parallel_rows is not None else case['row'] if run is None else _mixed_trial(
                run['command'], directory / case['id'], case['object'], case['trial'],
                expected_beta=cell['value'] if args.ema else None,
                expected_acceptance=cell['value'] if args.acceptance else None)
            rows.append(dict(row, cohort_id=case['id'], baseline_success=case['baseline_success'],
                             verify_control_version=(case['verify_control_version'] if run is None
                                                     else row.get('verify_control_version'))))
        (directory / 'trials.jsonl').write_text(
            ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
        record = dict(mode=manifest['mode'], parameter=manifest['parameter'], value=cell['value'],
                      baseline_value=baseline, baseline_comparable=baseline_comparable,
                      verify_control_version=(None if cell['index'] == 0 and not baseline_comparable
                                              else VERIFY_CONTROL_VERSION if args.ema else None),
                      result_role=('historical_reference' if cell['index'] == 0 and not baseline_comparable
                                   else 'sweep'), **summarize_mixed(rows, cases))
        summaries.append(record)
        _write_results(output, summaries, baseline)
        print('[%s] %s=%g: %d/5 = %.2f%%' % (
            manifest['mode'], manifest['parameter'], cell['value'], record['successes'],
            100 * record['success_rate']), flush=True)
    if args.plot:
        plot_results(summaries, output)
    print('[%s] results: %s' % (manifest['mode'], output), flush=True)
    return 0


def run_sensitivity(args, rest):
    from .config import configure_ablation
    configure_ablation(args)
    if args.sensitivity_workers < 1:
        raise ValueError('--sensitivity-workers must be positive')
    if args.ablation_mode != 'full':
        raise ValueError('--ema / --threshold require the full baseline; remove --a1 ... --a8')
    if args.ema and args.verify_quality_threshold is not None:
        raise ValueError('--ema cannot change the final acceptance gate')
    if args.ema and (args.verify_accept_threshold != .45 or args.verify_release_threshold != .65):
        raise ValueError('--ema only perturbs --verify_beta in test_mpc_isaac.py; '
                         'use --threshold to vary acceptance thresholds')
    if '--floating-fingertip' in rest or '--sampling-c3' in rest:
        raise ValueError('Verify sensitivity requires the Franka MPC planner')
    # Read only cohort/beta options here, before importing Torch or Isaac Gym.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--trial-start', type=int, default=0)
    parser.add_argument('--trial_num', type=int, default=100)
    parser.add_argument('--verify_beta', type=float, default=.18)
    parser.add_argument('--target-type', dest='target_type', default='ground-rotation')
    parser.add_argument('--physics-backend', default='isaac')
    parser.add_argument('--cpo-variant', default='full')
    parser.add_argument('--ideal_contact_pose', action='store_true')
    cohort, _ = parser.parse_known_args(rest)
    if cohort.trial_start < 0 or cohort.trial_num <= 0:
        raise ValueError('--trial-start must be nonnegative and --trial_num must be positive')
    if not math.isfinite(cohort.verify_beta) or not 0 <= cohort.verify_beta <= 1:
        raise ValueError('--verify_beta must be finite and in [0, 1]')
    if args.ema and (cohort.cpo_variant == 'direct' or cohort.ideal_contact_pose):
        raise ValueError('--ema requires an objective that uses the continuous verify weight; '
                         'remove --cpo-variant direct / --ideal_contact_pose')
    mode = 'ema' if args.ema else 'acceptance' if args.acceptance else 'threshold'
    parameter = 'verify_beta' if args.ema else 'verify_accept_threshold'
    option = '--verify_beta' if args.ema else '--verify-accept-threshold'
    baseline_value = cohort.verify_beta if args.ema else args.verify_accept_threshold
    values = (ema_grid(baseline_value, args.ema_offsets) if args.ema and args.ema_grid is None
              else parse_grid(args.ema_grid if args.ema else args.threshold_grid,
                              ema=args.ema, release=args.verify_release_threshold))
    if args.acceptance:
        parameter, option = 'verify_quality_threshold', '--verify-quality-threshold'
        baseline_value = .55 if args.verify_quality_threshold is None else args.verify_quality_threshold
        if not math.isfinite(baseline_value) or not 0 <= baseline_value <= 1:
            raise ValueError('verify quality threshold must be finite and in [0, 1]')
        if args.acceptance_grid is None:
            values = ema_grid(baseline_value, args.acceptance_offsets)
            values += [v for v in (0., 1.) if v not in values]
        else:
            values = parse_grid(args.acceptance_grid, ema=True)
            values = [baseline_value, *sorted((v for v in values if v != baseline_value),
                      key=lambda v: (abs(v-baseline_value), v))]
    # Use the client timezone for automatically named experiment folders.
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = Path(args.sensitivity_output or f'output/mpc_sensitivity/{mode}_{stamp}').resolve()
    manifest = dict(mode=mode, parameter=parameter, baseline_value=baseline_value,
                    verify_control_version=VERIFY_CONTROL_VERSION if args.ema else None,
                    entrypoint=str(_experiment_script(args)),
                    verify_beta=cohort.verify_beta,
                    verify_accept_threshold=args.verify_accept_threshold,
                    verify_release_threshold=args.verify_release_threshold,
                    trial_start=cohort.trial_start, trial_num=cohort.trial_num,
                    sensitivity_workers=args.sensitivity_workers, output=str(output), cells=[])
    if args.sensitivity_cohort == 'mixed' or ((args.ema or args.acceptance) and args.sensitivity_cohort is None):
        return _run_mixed(args, rest, cohort, output, manifest, option, values)
    script = _experiment_script(args)
    cells = []
    for index, value in enumerate(values):
        directory = output / ('cell_%03d' % index)
        flags = [*rest, '--headless',
                 *_threshold_flags(args),
                 option, str(value),
                 '--ranking-metrics-jsonl', str(directory / 'trials.jsonl'),
                 '--rollout-summary-json', str(directory / 'rollout_summary.json')]
        if not any(flag in rest for flag in ('--sync-planner', '--async-planner', '--trial')):
            flags.append('--sync-planner')
        cells.append(dict(index=index, value=value, directory=str(directory),
                          command=[sys.executable, '-u', str(script), *flags]))
    manifest['cells'] = cells
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
        if args.ema:
            validate_ema_trials(rows, cell['value'])
        if args.acceptance:
            validate_acceptance_trials(rows, cell['value'])
        record = dict(mode=mode, parameter=parameter, value=cell['value'],
                      baseline_value=baseline_value, trial_start=cohort.trial_start,
                      verify_control_version=VERIFY_CONTROL_VERSION if args.ema else None,
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
