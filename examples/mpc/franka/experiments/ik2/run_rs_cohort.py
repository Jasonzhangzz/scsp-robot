"""Run Table VI cases in isolated processes, then export JSON/CSV/LaTeX.

CSV inputs select historically successful seeds. Each worker reconstructs its
own synchronous factual trajectory; asynchronous CSV completion times are never
used as paired rollout values. A failed/mismatched case remains explicitly failed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from utils.ranking_cdr import summarize_cdr
from examples.mpc.franka.experiments.ik2.test_rs import write_json, csv_source_candidates

OBJECTS = ['foam_brick', 'stanford_bunny2', 'elephant', 'piggy_bank',
           'teapot', 'mug', 'rubber_duck']


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latex_row(task, metrics):
    def number(value, digits=3):
        return '--' if value is None else f'{value:.{digits}g}'
    a = metrics['advantage']
    advantage = ('--' if a['mean'] is None or a['std'] is None else
                 f"${a['mean']:.3g} \\pm {a['std']:.3g}$")
    return ' & '.join([task.capitalize(), number(metrics['contact_switch_count']),
        number(metrics['p_near']), number(metrics['p_optimal']), number(metrics['f_near_hz']),
        number(metrics['f_p_star_hz']),
        '--' if metrics['p_advantage_positive'] is None else f"{100*metrics['p_advantage_positive']:.1f}",
        advantage]) + r' \\'


def export_cohort(output, tasks, cases, gamma):
    result = dict(gamma=gamma, cases=cases, by_task={})
    rows, tex = [], []
    for task in tasks:
        states, trials = [], []
        complete = [c for c in cases if c['task'] == task and c['status'] == 'complete']
        for case in complete:
            directory = Path(case['output'])
            states.extend(read_jsonl(directory/'states.jsonl'))
            trials.extend(read_jsonl(directory/'trials.jsonl'))
        summary = summarize_cdr(states, trials, gamma=gamma)
        result['by_task'][task] = summary
        m = summary['table_vi']
        row = dict(task=task, completed_objects=len(complete), **{
            k: m[k] for k in ('contact_switch_count', 'p_near', 'p_optimal',
                              'f_near_hz', 'f_p_star_hz', 'p_advantage_positive')})
        row.update(advantage_mean=m['advantage']['mean'], advantage_std=m['advantage']['std'],
                   sampled_points=len(states))
        rows.append(row)
        tex.append(latex_row(task, m))
    write_json(output/'summary.json', result)
    with (output/'table_vi.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/'table_vi_rows.tex').write_text('\n'.join(tex)+'\n')
    return result, tex


def update_paper(path, rows):
    text = path.read_text()
    label = text.index(r'\label{tab:ranking_advantage}')
    end = text.index(r'\end{table}', label)
    table = text[label:end]
    for row in rows:
        name = row.split(' & ', 1)[0]
        table, count = re.subn(r'^'+name+r' & .*$', lambda _: row, table, flags=re.M)
        if count != 1:
            raise ValueError(f'Cannot identify unique {name} row in Table VI')
    updated = text[:label]+table+text[end:]
    temp = path.with_suffix('.tex.tmp')
    temp.write_text(updated)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-csv', required=True, nargs='+', type=Path)
    parser.add_argument('--objects', nargs='+', default=OBJECTS)
    parser.add_argument('--tasks', nargs='+', choices=('rotation', 'flipping'),
                        default=['rotation', 'flipping'])
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--max-rollout-length', type=int, default=5000)
    parser.add_argument('--paper', type=Path, default=ROOT/'tro_revision/experiment.tex')
    parser.add_argument('--no-update-paper', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        raise ValueError('Use a new cohort directory; existing results must not be mixed')
    paths = [str(p.resolve()) for p in args.source_csv]
    cases = []
    for task in args.tasks:
        for obj in args.objects:
            seeds = csv_source_candidates(paths, obj=obj, task=task, seed=args.sample_seed)
            cases.append(dict(task=task, object=obj, output=str(output/task/obj),
                              successful_csv_seeds=seeds, status='pending'))
    source_files = [Path(__file__), ROOT/'utils/ranking_cdr.py', ROOT/'utils/rollout_cache.py',
                    ROOT/'examples/mpc/franka/experiments/ik2/test_rs.py',
                    ROOT/'examples/mpc/fingertips/test/test_0902.py',
                    *sorted((ROOT/'examples/mpc/franka/runtime').glob('*.py')),
                    *sorted((ROOT/'planning').glob('*.py'))]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source_files}
    manifest = dict(status='running', pid=os.getpid(), samples_per_trajectory=args.samples,
                    successful_trajectories_per_object_per_task=1, gamma=args.gamma,
                    sample_scope='eligible physical-contact decisions only; transit excluded',
                    source_csv=paths, source_sha256=hashes, cases=cases,
                    source_policy='CSV selects seeds; current synchronous reconstruction must succeed with enough decisions')
    write_json(output/'manifest.json', manifest)
    for case in cases:
        directory = Path(case['output'])
        directory.mkdir(parents=True, exist_ok=True)
        if not case['successful_csv_seeds']:
            case.update(status='failed', error='No historical successful CSV seed for this case')
            continue
        command = [sys.executable, '-u', str(ROOT/'test_rs.py'), '--obj', case['object'],
                   '--rs-cohort', 'single', '--rs-contact-sampling', 'touching',
                   '--'+case['task'], '--rs-output', str(directory),
                   '--rs-source-csv', *paths,
                   '--rs-samples-per-trajectory', str(args.samples),
                   '--rs-successful-trajectories', '1', '--rs-repeats', '1',
                   '--rs-sample-seed', str(args.sample_seed), '--rs-event-stage', 'all',
                   '--ranking-metric-gamma', str(args.gamma),
                   '--max_rollout_length', str(args.max_rollout_length), '--compute-threads', '1']
        case.update(status='running', command=command)
        write_json(output/'manifest.json', manifest)
        print(f"START {case['task']}/{case['object']}", flush=True)
        try:
            with (directory/'run.log').open('w') as log:
                completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
            if completed.returncode:
                raise RuntimeError(f'Worker exit {completed.returncode}; see {directory}/run.log')
            summary = json.loads((directory/'summary.json').read_text())
            if summary['n_states'] != args.samples or summary['table_vi']['source_trajectories'] != 1:
                raise RuntimeError('Worker did not produce exactly one source and requested samples')
            if any(summary['table_vi'][key] is None for key in (
                    'contact_switch_count', 'p_near', 'p_optimal', 'f_near_hz',
                    'f_p_star_hz', 'p_advantage_positive')):
                raise RuntimeError('Worker is missing one or more Table VI measurements')
            case['status'] = 'complete'
        except Exception as error:
            case.update(status='failed', error=str(error))
            traceback.print_exc()
        print(f"{case['status'].upper()} {case['task']}/{case['object']}", flush=True)
        write_json(output/'manifest.json', manifest)
        export_cohort(output, args.tasks, cases, args.gamma)
    result, rows = export_cohort(output, args.tasks, cases, args.gamma)
    manifest['status'] = 'complete' if all(c['status'] == 'complete' for c in cases) else 'incomplete'
    paper_protocol = (set(args.objects) == set(OBJECTS)
                      and set(args.tasks) == {'rotation', 'flipping'}
                      and args.samples == 5 and args.gamma == .99)
    if manifest['status'] == 'complete' and not args.no_update_paper and paper_protocol:
        update_paper(args.paper, rows)
        manifest['updated_paper'] = str(args.paper)
    write_json(output/'manifest.json', manifest)
    print(f"{manifest['status'].upper()}: {output}", flush=True)
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
