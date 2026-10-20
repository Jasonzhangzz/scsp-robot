"""Recover Table IV planning times from local records without running experiments.

Run from any directory. Writes an auditable JSON/Markdown snapshot; --apply
updates Planning Time and the completed archived A6/A7 task metrics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'tro_revision'
METHODS = ('SCSP', 'DyWA', 'ContactSDF', 'STOCS-3D', 'Sampling', 'CF-MPC',
           'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7')
HASHES = {}


def read(relative):
    path = ROOT / relative
    data = path.read_bytes()
    HASHES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode()


def load(relative):
    return json.loads(read(relative))


def jsonlines(relative):
    return [json.loads(line) for line in read(relative).splitlines() if line.strip()]


def moments(rows):
    assert rows and not any(row.get('aborted') for row in rows)
    assert all(row['planning_wall_count'] == row['steps'] for row in rows)
    count = sum(row['planning_wall_count'] for row in rows)
    total = math.fsum(row['planning_wall_total_s'] for row in rows)
    squares = math.fsum(row['planning_wall_sum_squares_s2'] for row in rows)
    variance = max(0., (squares - total * total / count) / (count - 1))
    return dict(trials=len(rows), samples=count, mean_ms=total / count * 1000,
                std_ms=math.sqrt(variance) * 1000, status='cached_cycle_moments')


def recorder_moments(method):
    values, raw_values, excluded, trials = [], [], [], 0
    peak = (0., None, None)
    for path in sorted((ROOT / 'output/trials' / method).glob('*/*/metrics.json')):
        row = load(path.relative_to(ROOT))
        times = row['planning_time_s']
        assert not row['aborted'] and len(times) == row['steps']
        assert math.isclose(math.fsum(times), row['planning_time_total_s'], rel_tol=1e-10)
        raw_values.extend(times)
        for index, seconds in enumerate(times):
            if (method == 'isaac' and path.relative_to(ROOT).as_posix() ==
                    'output/trials/isaac/mug/002/metrics.json' and index == 1502):
                assert math.isclose(seconds, 175.64558509900235, rel_tol=1e-12)
                excluded.append(dict(source=str(path.relative_to(ROOT)),
                    planning_time_index=index, value_s=seconds,
                    reason='Explicit exclusion of the identified timing outlier requested on 2026-10-21.'))
            else:
                values.append(seconds)
        index = max(range(len(times)), key=times.__getitem__)
        if times[index] > peak[0]:
            peak = (times[index], str(path.relative_to(ROOT)), index)
        trials += 1
    assert trials == 35
    assert len(excluded) == (1 if method == 'isaac' else 0)
    return dict(trials=trials, samples=len(values),
                mean_ms=statistics.mean(values) * 1000,
                std_ms=statistics.stdev(values) * 1000,
                max_cycle_ms=peak[0] * 1000, max_cycle_source=peak[1],
                max_cycle_index=peak[2],
                max_retained_cycle_ms=max(values) * 1000,
                excluded_samples=excluded,
                unfiltered=dict(samples=len(raw_values),
                    mean_ms=statistics.mean(raw_values) * 1000,
                    std_ms=statistics.stdev(raw_values) * 1000),
                status='cached_cycle_samples',
                source=f'output/trials/{method}/*/*/metrics.json',
                note='Flipping, seven objects x five seeds, synchronous planner.')


def task_metrics(rows):
    """Use the same completed cohort for SR, successful ET, and terminal errors."""
    def stats(values):
        return dict(n=len(values), mean=statistics.mean(values) if values else None,
                    std=statistics.stdev(values) if len(values) > 1 else None)
    successful = [row for row in rows if row['success']]
    assert not any(row.get('aborted') for row in rows)
    return dict(trials=len(rows), successes=len(successful),
        success_rate=len(successful) / len(rows),
        execution_time_s=stats([row['completion_time_s'] for row in successful]),
        final_pos_error_m=stats([row['final_pos_err'] for row in rows]),
        final_quat_error=stats([row['final_quat_err'] for row in rows]))


def recover():
    result = {task: {} for task in ('Rotation', 'Flipping')}
    rotation, flipping = result.values()
    for method in ('isaac', 'a1', 'a2', 'a3', 'a4', 'a5'):
        flipping['SCSP' if method == 'isaac' else method.upper()] = recorder_moments(method)

    directory = 'output/table4_rotation_20261020'
    rows = [row for path in sorted((ROOT / directory).glob('*/trials.jsonl'))
            for row in jsonlines(path.relative_to(ROOT))]
    assert len(rows) == 35 and all(row['target_type'] == 'rotation' for row in rows)
    rotation['SCSP'] = dict(moments(rows), source=directory + '/*/trials.jsonl')

    source = 'output/benchmarks/a3_rotation_20261013/results/summary.json'
    summary = load(source)['overall']
    assert summary['complete'] and summary['completed_trials'] == 35
    timing = summary['planning_cycle_ms']
    rotation['A3'] = dict(trials=35, samples=timing['n'], mean_ms=timing['mean'],
                          std_ms=timing['std'], source=source,
                          status='cached_pooled_summary',
                          note='Asynchronous planner; 5000-cycle limit in the archived commands.')

    source = 'output/benchmarks/a4_a5_async_7x3_20261015/trials.json'
    rows = load(source)
    for method in ('a4', 'a5'):
        selected = [row for row in rows if row['variant'] == method and row['task'] == 'rotation']
        assert len(selected) == 21
        rotation[method.upper()] = dict(moments(selected), source=source,
                                        note='Rotation only; asynchronous planner, three seeds per object.')

    source = 'output/benchmarks/a6_rotation_async_7x3_20261015/trials.json'
    rows = load(source)
    assert len(rows) == 21 and all(row['cpo_variant'] == 'direct' for row in rows)
    rotation['A6'] = dict(moments(rows), source=source,
                          task_metrics=task_metrics(rows),
                          note='Archived acados direct-CPO variant; not the later AdamW implementation.')

    source = 'output/benchmarks/a7_mppi_7x3_20261015/trials.json'
    rows = load(source)
    for task, cells in result.items():
        selected = [row for row in rows if row['task'] == task.lower()]
        assert len(selected) == 21
        cells['A7'] = dict(moments(selected), source=source,
                           task_metrics=task_metrics(selected),
                           note='MPPI, three seeds per object.')

    source = 'output/benchmarks/contactsdf_official_single_tip_15_20261020/summary.json'
    summary = load(source)
    assert summary['complete'] and summary['trials_complete'] == 15
    directory = str(Path(source).parent)
    rows = [row for case in summary['cases']
            for row in jsonlines(f"{directory}/{case['case_directory']}/trials.jsonl")]
    flipping['ContactSDF'] = dict(moments(rows), source=directory + '/cases/*/trials.jsonl',
                                  note='Detection plus MPC callback; 15 trials, 7500 cycles.')
    aggregate = load(directory + '/aggregate_metrics.json')['planning_time_s']
    assert math.isclose(flipping['ContactSDF']['mean_ms'], aggregate['mean'] * 1000, rel_tol=1e-10)
    assert math.isclose(flipping['ContactSDF']['std_ms'], aggregate['sample_std'] * 1000, rel_tol=1e-10)

    source = 'output/benchmarks/contactsdf_official_rotation_15_20261020/summary.json'
    summary = load(source)
    assert summary['complete'] and summary['trials_complete'] == 15
    rows = [dict(case, steps=case['planning_wall_count']) for case in summary['cases']]
    rotation['ContactSDF'] = dict(moments(rows), source=source,
                                  note='Detection plus MPC callback; 15 trials, 15000 cycles; 0.1 s policy interval.')

    source = 'output/benchmarks/stocs3d_table4_15_20261020/table4_metrics.json'
    summary = load(source)
    excluded = '05_elephant_seed2'
    assert len(summary['cases']) == 15
    # Preserve the exclusion ALREADY stated in experiment.tex. Do not introduce
    # a new post-hoc filter or drop any other slow/offline planning outcome.
    paper = read('tro_revision/experiment.tex')
    assert 'one planning-time outlier removed (elephant, 3690' in paper
    values = [row['planning_time_s'] for row in summary['cases'] if row['trial_key'] != excluded]
    assert len(values) == 14
    flipping['STOCS-3D'] = dict(trials=14, samples=14,
        mean_ms=statistics.mean(values) * 1000, std_ms=statistics.stdev(values) * 1000,
        status='cached_offline_trial_times', source=source,
        note='Offline trajectory planning; retains the paper\'s existing exclusion of ' + excluded,
        full_15_trial_mean_ms=summary['planning_time_s']['mean'] * 1000,
        full_15_trial_std_ms=summary['planning_time_s']['sample_std'] * 1000)
    rotation['STOCS-3D'] = dict(mean_ms=None, std_ms=None, status='unavailable',
                               note='No completed Rotation STOCS-3D cohort found.')

    source = 'tro_revision/experiment_copy.tex'
    archive = read(source).split('\\label{tab:all_data}', 1)[1]
    lines = [line for line in archive.splitlines() if ' & Planning Time (s) & ' in line][:2]
    assert len(lines) == 2
    for task, line in zip(result, lines):
        cells = line.rsplit(' \\\\', 1)[0].split(' & ')
        assert len(cells) == 21
        for method, index in (('DyWA', 9), ('Sampling', 12), ('CF-MPC', 13)):
            result[task][method] = dict(mean_ms=float(cells[index]) * 1000, std_ms=None,
                status='historical_rounded_summary', source=source,
                note='Archived Franka column, originally reported to 0.001 s; raw cycle samples unavailable.')
        if task == 'Rotation':
            # Current Rotation A1 explicitly reuses historical w/o RS, whose
            # archived success/time pair (0.36, 19.36) is in the old A2 column.
            preceding = archive.split(' & Planning Time (s)', 1)[0].splitlines()
            success = next(line for line in preceding if 'Rotation} & Success Rate' in line)
            execution = next(line for line in preceding if ' & Execution Time (s)' in line)
            assert float(success.split(' & ')[15]) == .36
            assert float(execution.split(' & ')[15]) == 19.36
            for method in ('A1', 'A2'):
                rotation[method] = dict(mean_ms=float(cells[15]) * 1000, std_ms=None,
                    status='historical_rounded_summary', source=source,
                    note='Historical w/o RS / old A2 column; no Rotation raw timing samples recovered.')

    source = 'output/diagnostics/cso_table_solver_update_20261014/experiment_before.tex'
    archive = read(source)
    matched = re.search(r'Planning mean: ([0-9.]+) s / ([0-9]+) control cycles', archive)
    assert matched is not None
    total, count = float(matched[1]), int(matched[2])
    flipping['A6'] = dict(mean_ms=total / count * 1000, std_ms=None,
        trials=15, samples=count, total_s=total, status='historical_total_and_count', source=source,
        note='Archived A3, later renamed A4 then A6; 6/15 interim successes. No squared times/raw samples available.')
    assert all(set(cells) == set(METHODS) for cells in result.values())
    return result


def latex_cell(method, value, bold=False):
    if value['mean_ms'] is None:
        return '--'
    mean, std = value['mean_ms'], value['std_ms']
    if method == 'STOCS-3D':
        text = f'({mean / 1e5:.2f}\\pm{std / 1e5:.2f})\\times10^{{5}}'
    else:
        text = f'{mean:.2f}' + (f'\\pm{std:.2f}' if std is not None else '')
    text = '$' + text + '$'
    return '\\textbf{' + text + '}' if bold else text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = recover()
    for task, cells in result.items():
        fastest = min((value['mean_ms'], method) for method, value in cells.items()
                      if value['mean_ms'] is not None)[1]
        for method, value in cells.items():
            value['latex'] = latex_cell(method, value, method == fastest)
    snapshot = dict(unit='ms', std_definition='Sample standard deviation, ddof=1; absent when unavailable.',
                    online_definition='Pooled planning callback wall times over all cycles, including failures.',
                    tasks=result, source_sha256=HASHES)
    (PAPER / 'table_iv_planning_sources.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    report = ['| Task | Method | Planning time (ms) | Source status |', '|---|---|---:|---|']
    for task, cells in result.items():
        for method in METHODS:
            value = cells[method]
            number = '--' if value['mean_ms'] is None else f"{value['mean_ms']:.2f}"
            if value['std_ms'] is not None:
                number += f" ± {value['std_ms']:.2f}"
            report.append(f"| {task} | {method} | {number} | {value['status']} |")
    report += ['', 'SCSP Flipping excludes exactly one timing sample: mug trial 002, planning_time_s[1502] = 175.64558509900235 s. All 35 trials and their outcomes remain included.',
               'Unfiltered timing statistics and the exclusion record are preserved in the JSON snapshot; original data files are unchanged.',
               'A6 Rotation: archived direct-CPO cohort complete (21 trials). A6 Flipping: only historical 15-trial interim summary available; its values are preserved.',
               'A7: archived Rotation and Flipping cohorts complete (21 trials each).',
               'Raw precision, sample counts, source paths, caveats, and SHA256 hashes: table_iv_planning_sources.json.',
               'Historical rounded summaries do not regain precision through unit conversion.',
               'STOCS-3D reports offline time per trajectory; online methods report time per callback.']
    (PAPER / 'table_iv_planning_sources.md').write_text('\n'.join(report) + '\n')
    if args.apply:
        path = PAPER / 'experiment.tex'
        text = path.read_text()
        start = text.index('\\label{tab:all_data}')
        end = text.index('\\end{table*}', start)
        prefix, suffix = text[:start], text[end:]
        section = text[start:end]
        rows = iter(result.values())
        def replace(match):
            cells = next(rows)
            return ' & Planning Time (ms) & ' + ' & '.join(cells[m]['latex'] for m in METHODS) + r' \\'
        section, count = re.subn(r'^ & Planning Time \(ms\).*$', replace, section, flags=re.M)
        assert count == 2
        task = None
        lines = []
        keys = {'Success Rate': 'success_rate', 'Execution Time (s)': 'execution_time_s',
                'Obj. Pos. Error (m)': 'final_pos_error_m', 'Obj. Quat. Error': 'final_quat_error'}
        for line in section.splitlines(keepends=True):
            if r'\multirow{5}{*}{Rotation}' in line:
                task = 'Rotation'
            elif r'\multirow{5}{*}{Flipping}' in line:
                task = 'Flipping'
            if task and ' & ' in line:
                cells = line.rstrip('\n').rsplit(' \\\\', 1)[0].split(' & ')
                key = keys.get(cells[1])
                if key:
                    assert len(cells) == len(METHODS) + 2
                    for method in ('A6', 'A7'):
                        metrics = result[task][method].get('task_metrics')
                        if metrics is None:
                            continue
                        value = metrics[key]
                        if isinstance(value, dict):
                            cell_text = '--' if value['mean'] is None else f"{value['mean']:.2f}"
                            if value['std'] is not None:
                                cell_text = '$' + cell_text + f"\\pm{value['std']:.2f}$"
                        else:
                            cell_text = f'{value:.2f}'
                        cells[METHODS.index(method) + 2] = cell_text
                    line = ' & '.join(cells) + r' \\' + '\n'
            lines.append(line)
        section = ''.join(lines)
        updated = prefix + section + suffix
        assert updated.startswith(prefix) and updated.endswith(suffix)
        assert abs(len(updated.splitlines()) - len(text.splitlines())) < 5
        temporary = path.with_suffix('.tex.tmp')
        temporary.write_text(updated)
        temporary.replace(path)
    print('\n'.join(report))


if __name__ == '__main__':
    main()
