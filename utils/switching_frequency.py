"""Offline switching statistics from original rollout caches; no simulation.

python -m utils.switching_frequency outputs/rs_cached_success_20261013

The primary count statistic is completed physical MPC contact relocations /
original rollouts. RS trigger counts are never a substitute for actual
execution. Neither capped CDR samples nor archived attempts enter the count.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from utils.ranking_metrics import _mean_std


def integer(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be a nonnegative integer')
    if not math.isfinite(value) or value < 0 or int(value) != value:
        raise ValueError(f'{field} must be a nonnegative integer')
    return int(value)


def task_name(row):
    task = row.get('target_type', 'unknown')
    return 'flipping' if task == 'ground-rotation' else task


def statistics(rows, *, successful_only=True, expected=None):
    eligible = [r for r in rows if not r.get('aborted')
                and (not successful_only or bool(r['success']))]
    measured = [r for r in eligible if r.get('mpc_contact_switches') is not None]
    counts = [integer(r['mpc_contact_switches'], 'mpc_contact_switches') for r in measured]
    durations, rates, evaluation_counts = [], [], []
    for row, count in zip(measured, counts):
        duration = row.get('execution_time_s')
        if duration is None and row['success']:
            duration = row.get('completion_time_s')
        if (duration is None and not row.get('async_planner', False)
                and row.get('steps') is not None and row.get('policy_interval_s') is not None):
            steps = integer(row['steps'], 'steps')
            dt = float(row['policy_interval_s'])
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError('policy_interval_s must be finite and positive')
            duration = steps*dt
        if duration is not None:
            duration = float(duration)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError('Cached execution time must be finite and nonnegative')
            durations.append(duration)
            if duration > 0:
                rates.append(count/duration)
            elif count:
                raise ValueError('Nonzero switches in a zero-duration rollout')
        if row.get('rs_evaluations') is not None:
            evaluations = integer(row['rs_evaluations'], 'rs_evaluations')
            if count > evaluations:
                raise ValueError('Switch count exceeds ranking evaluations')
            evaluation_counts.append(evaluations)
    complete_time = bool(measured) and len(durations) == len(measured)
    complete_evaluations = bool(measured) and len(evaluation_counts) == len(measured)
    total = sum(counts) if measured else None
    seconds = sum(durations) if complete_time else None
    evaluations = sum(evaluation_counts) if complete_evaluations else None
    return dict(expected_rollouts=expected, available_rollouts=len(rows),
        successful_rollouts=sum(bool(r['success']) and not r.get('aborted') for r in rows),
        excluded_aborted=sum(bool(r.get('aborted')) for r in rows),
        eligible_rollouts=len(eligible), measured_rollouts=len(measured),
        missing_switch_count_rollouts=len(eligible)-len(measured),
        unavailable_rollouts=max(0, expected-len(rows)) if expected is not None else None,
        total_switches=total,
        switches_per_rollout=_mean_std(counts),
        total_sim_time_s=seconds,
        switching_frequency_hz=total/seconds if seconds else None,
        per_rollout_frequency_hz=_mean_std(rates),
        total_ranking_evaluations=evaluations,
        switch_fraction_per_ranking=total/evaluations if evaluations else None)


def summarize(rows, *, successful_only=True, expected_cases=None):
    """Count each original object/seed/task once, with zero-switch trials included."""
    seen = set()
    for row in rows:
        key = (row['object'], row.get('seed', row.get('trial')), task_name(row))
        if key in seen:
            raise ValueError(f'Duplicate original rollout: {key}')
        seen.add(key)
        if row.get('success') not in (0, 1, False, True):
            raise ValueError('success must be a boolean or 0/1')
    expected_cases = None if expected_cases is None else list(expected_cases)
    objects = sorted({r['object'] for r in rows} |
                     {r['object'] for r in expected_cases or []})
    by_object = {}
    for obj in objects:
        expected = (sum(c['object'] == obj for c in expected_cases)
                    if expected_cases is not None else None)
        by_object[obj] = statistics([r for r in rows if r['object'] == obj],
                                   successful_only=successful_only, expected=expected)
    by_task = {task: statistics([r for r in rows if task_name(r) == task],
                               successful_only=successful_only)
               for task in sorted({task_name(r) for r in rows})}
    return dict(source='offline original-rollout summaries',
        scope='successful original rollouts' if successful_only else 'non-aborted original rollouts',
        definitions=dict(switches_per_rollout='sum(mpc_contact_switches) / measured_rollouts',
            switching_frequency_hz='sum(mpc_contact_switches) / sum(cached execution seconds; synchronous step-duration fallback)',
            switch_fraction_per_ranking='sum(mpc_contact_switches) / sum(rs_evaluations)',
            switch_event='executed far MPC target, physical contact loss, then stable distinct measured contact at target'),
        overall=statistics(rows, successful_only=successful_only,
                           expected=len(expected_cases) if expected_cases is not None else None),
        by_object=by_object, by_task=by_task)


def read_records(path):
    """Read only complete JSONL lines; a concurrent writer may have a partial tail."""
    records = []
    lines = path.read_text(encoding='utf-8').splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines)-1 and not line.endswith('\n'):
                continue
            raise
    return records


def load_cache(path):
    path = Path(path)
    expected = None
    if path.is_file():
        files = [path]
    elif (path/'cases.json').exists():
        expected = json.loads((path/'cases.json').read_text())
        files = [path/('%s_seed%d' % (c['object'], c['seed']))/'trials.jsonl'
                 for c in expected]
    elif (path/'trials.jsonl').exists():
        files = [path/'trials.jsonl']
    else:
        # Exactly one level: never recurse into archived attempts/branches.
        files = sorted(path.glob('*/trials.jsonl'))
        if not files:
            files = sorted(path.glob('*.jsonl'))
    records, sources = [], []
    for file in files:
        if file.exists():
            rows = read_records(file)
            for row in rows:
                if row.get('mpc_contact_switches') is None:
                    trial = row.get('trial', row.get('seed'))
                    trace_path = file.parent/f'execution_contact_trace_trial{trial}.jsonl'
                    if trace_path.exists():
                        from utils.executed_contact_switches import completed_contact_switches
                        trace = read_records(trace_path)
                        if (trace and trace[0]['step'] == 0 and
                                trace[-1]['step'] == row['steps'] and len(trace) == row['steps']+1):
                            try:
                                result = completed_contact_switches(trace)
                                row['mpc_contact_switches'] = result['mpc_contact_switches']
                                row['mpc_contact_switch_events'] = result['events']
                                row['mpc_contact_switch_parameters'] = result['parameters']
                            except ValueError as error:
                                row['mpc_contact_switches_unavailable_reason'] = str(error)
                        else:
                            row['mpc_contact_switches_unavailable_reason'] = 'Incomplete execution contact trace'
                    else:
                        row['mpc_contact_switches_unavailable_reason'] = 'No execution contact trace; RS triggers cannot determine physical relocations'
            records.extend(rows)
            sources.append(str(file))
    if expected is not None:
        allowed = {(c['object'], c['seed']) for c in expected}
        if any((r['object'], r.get('seed', r.get('trial'))) not in allowed for r in records):
            raise ValueError('Cached rollout does not belong to the case manifest')
    return records, expected, sources


def write_csv(path, report):
    fields = ['group', 'name', 'expected_rollouts', 'available_rollouts', 'measured_rollouts',
              'total_switches', 'switches_per_rollout_mean', 'switches_per_rollout_std',
              'total_sim_time_s', 'switching_frequency_hz', 'switch_fraction_per_ranking']
    with Path(path).open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        groups = [('overall', 'all', report['overall'])]
        groups += [('object', name, stats) for name, stats in report['by_object'].items()]
        groups += [('task', name, stats) for name, stats in report['by_task'].items()]
        for group, name, stats in groups:
            row = {key: stats.get(key) for key in fields if key not in ('group', 'name')}
            row.update(group=group, name=name,
                       switches_per_rollout_mean=stats['switches_per_rollout']['mean'],
                       switches_per_rollout_std=stats['switches_per_rollout']['std'])
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--include-failures', action='store_true')
    parser.add_argument('--output', type=Path, help='JSON path; defaults to cache/switching_frequency.json')
    args = parser.parse_args()
    rows, cases, sources = load_cache(args.cache)
    report = summarize(rows, successful_only=not args.include_failures, expected_cases=cases)
    report['input_files'] = sources
    path = args.output or ((args.cache if args.cache.is_dir() else args.cache.parent)/'switching_frequency.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.tmp')
    pending.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    pending.replace(path)
    write_csv(path.with_suffix('.csv'), report)
    print(json.dumps(report['overall'], indent=2, allow_nan=False))
    print('Offline switching statistics saved to '+str(path))


if __name__ == '__main__':
    main()
