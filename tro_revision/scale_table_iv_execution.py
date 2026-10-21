"""Estimate execution SD at each reported mean by retaining a cached proxy's CV.

The target has 200 trials. This does not create those trials or recover their
measured SD; execution statistics remain conditional on successful execution.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from tro_revision.recover_table_iv_execution import parse_paper, METHODS

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'tro_revision'
SOURCE = PAPER / 'table_iv_execution_sources.json'
PROXY_METHODS = ('DyWA', 'Sampling', 'CF-MPC')
NOTE = (
    r'\parbox{\linewidth}{\scriptsize Execution times use successful trajectories only. '
    r'Unmarked $m\pm s$ entries give a cohort-matched mean and sample SD. '
    r'$\dagger$: SDs for DyWA, Sampling, and CF-MPC are estimates for the reported '
    r'200-trial evaluation, obtained as '
    r'$\widehat{s}=s_{\mathrm{proxy}}m/\bar{t}_{\mathrm{proxy}}$, assuming an equal '
    r'coefficient of variation. Proxy samples are disjoint sets of cached controller '
    r'trajectories with approximately matched success rates; they are not measurements '
    r'of those methods or observed SDs across 200 trials. No-success cases remain --; '
    r'a bare mean indicates an unavailable SD.}'
)


def estimate(rows, target_mean):
    values = [r['execution_time_s'] for r in rows if r['success']]
    if len(values) < 2 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('At least two finite, nonnegative successful durations required')
    source_mean = statistics.mean(values)
    if source_mean <= 0 or not math.isfinite(target_mean) or target_mean <= 0:
        raise ValueError('Positive source and target means required')
    source_sd = statistics.stdev(values)
    scale = target_mean / source_mean
    return dict(source_successes=len(values), source_mean_s=source_mean,
                source_sample_std_s=source_sd, retained_mean_s=target_mean,
                scale_factor=scale, coefficient_of_variation=source_sd/source_mean,
                estimated_std_s=source_sd*scale)


def update(text, source):
    current = parse_paper(text)
    estimates = {}
    for task in ('Rotation', 'Flipping'):
        estimates[task] = {}
        for method in PROXY_METHODS:
            group = source['proxy_groups'][task][method]
            value = estimate(group['selected_trajectories'], current[task][method]['retained_mean_s'])
            value.update(target_total_trials=200,
                         target_success_rate=current[task][method]['target_success_rate'],
                         observed_proxy_trials=group['trials'],
                         status='mean_scaled_proxy_estimate_not_observed_200_trial_sd')
            estimates[task][method] = value
    start = text.index(r'\label{tab:all_data}')
    end = text.index(r'\end{table*}', start)
    section = text[start:end]
    lines = []
    task = None
    for line in section.splitlines(keepends=True):
        for name in estimates:
            if r'\multirow{5}{*}{' + name + '}' in line:
                task = name
        if ' & Execution Time (s) & ' in line:
            cells = line.rstrip('\n').rsplit(r' \\', 1)[0].split(' & ')
            for method, value in estimates[task].items():
                index = METHODS.index(method) + 2
                bold = r'\textbf{' in cells[index]
                cell = '$%.2f\\pm%.2f^{\\dagger}$' % (value['retained_mean_s'], value['estimated_std_s'])
                cells[index] = r'\textbf{' + cell + '}' if bold else cell
            line = ' & '.join(cells) + r' \\' + '\n'
        lines.append(line)
    section = ''.join(lines)
    note_start = section.index(r'\parbox{\linewidth}{\scriptsize Execution times')
    section = section[:note_start] + NOTE + '\n'
    return text[:start] + section + text[end:], estimates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    selected = [r for task in ('Rotation', 'Flipping') for method in PROXY_METHODS
                for r in source['proxy_groups'][task][method]['selected_trajectories']]
    for name in {r['source'] for r in selected}:
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == source['source_sha256'][name], name
    paper = PAPER/'experiment.tex'
    updated, estimates = update(paper.read_text(), source)
    snapshot = dict(formula='estimated_std = source_sample_std * retained_mean / source_mean',
                    assumption='Equal coefficient of variation; no sample-count scaling.',
                    target_total_trials=200, observed_target_trials_generated=0,
                    source_snapshot=SOURCE.name,
                    source_snapshot_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    estimates=estimates)
    (PAPER/'table_iv_execution_scaled_sources.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    records = [dict(task=task, method=method, **value) for task, methods in estimates.items()
               for method, value in methods.items()]
    with (PAPER/'table_iv_execution_scaled_sources.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    if args.apply:
        paper.write_text(updated)
    for record in records:
        print('%s %s: %.2f +/- %.2f (estimated)' % (record['task'], record['method'],
              record['retained_mean_s'], record['estimated_std_s']))


if __name__ == '__main__':
    main()
