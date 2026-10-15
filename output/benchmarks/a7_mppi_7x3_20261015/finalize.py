"""Validate the complete, unfiltered cohort and populate only the A7 column."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.mpc.franka.ik2.evaluate_a7 import OBJECTS, TASKS, load_cell, summarize

out = Path(__file__).resolve().parent
manifest = json.loads((out/'manifest.json').read_text())
for path, expected in manifest['source_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
all_rows, results = [], {}
for task in TASKS:
    rows = []
    per_object = {}
    for obj in OBJECTS:
        cell = load_cell(out/task/obj/'trials.jsonl', obj, task)
        for row in cell:
            config = dict(row['a7_mppi'])
            config.pop('solves')
            assert config == dict(samples=128, horizon=10, iterations=1, temperature=.005, noise=.003)
        rows.extend(cell)
        per_object[obj] = summarize(cell)
    assert len(rows) == 21
    summary = summarize(rows)
    # Independent array reductions check weighting and the inclusion of failures.
    assert np.isclose(summary['planning_time_s'], np.average(
        [r['planning_wall_total_s']/r['planning_wall_count'] for r in rows],
        weights=[r['planning_wall_count'] for r in rows]))
    assert np.isclose(summary['final_pos_error_m']['mean'], np.mean([r['final_pos_err'] for r in rows]))
    assert np.isclose(summary['final_quat_error']['mean'], np.mean([r['final_quat_err'] for r in rows]))
    assert summary['successes'] == sum(r['success'] for r in rows)
    results[task] = dict(overall=summary, objects=per_object)
    all_rows.extend(rows)
assert len({(r['task'], r['object'], r['seed']) for r in all_rows}) == 42
all_rows.sort(key=lambda r: (r['task'], r['object'], r['seed']))
(out/'trials.json').write_text(json.dumps(all_rows, indent=2, allow_nan=False)+'\n')
(out/'statistics.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
with (out/'trials.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=['task', 'object', 'seed', 'success', 'steps',
        'completion_time_s', 'planning_wall_total_s', 'planning_wall_count',
        'planning_wall_mean_ms', 'final_pos_err', 'final_quat_err', 'solver_backend'], extrasaction='ignore')
    writer.writeheader()
    writer.writerows(all_rows)
report = '# A7: prespecified 7-object x 3-seed cohort\n\n'
report += 'Both tasks use seed IDs 0, 1, 2 for each object, with a 2500-step limit. All completed trials are included. '
report += 'Execution time is simulated time on successful trials; planning time pools all planning callbacks and cycles; final pose errors include failures. '
report += 'MPPI uses 128 sampled trajectories, horizon 10, one update per cycle, temperature 0.005, and noise standard deviation 0.003 m. '
report += 'The model, weights and settings were fixed before the cohort. CPU workers run on distinct physical cores.\n\n'
report += '| Task | Object | Success | Execution (s) | Planning (s/cycle) | Position error (m) | Quaternion error |\n|---|---|---:|---:|---:|---:|---:|\n'
for task in TASKS:
    for obj, record in [*results[task]['objects'].items(), ('ALL', results[task]['overall'])]:
        execution = record['execution_time_s']['mean']
        report += '| %s | %s | %d/%d | %s | %.6f | %.6f | %.6f |\n' % (
            task, obj, record['successes'], record['trials'], '--' if execution is None else '%.4f' % execution,
            record['planning_time_s'], record['final_pos_error_m']['mean'], record['final_quat_error']['mean'])
(out/'report.md').write_text(report)

tex = ROOT/'tro_revision/experiment.tex'
text = tex.read_text()
start = text.index('\n\\label{tab:all_data}\n')+1
end = text.index('\\end{table*}', start)
table = text[start:end]
lines = table.splitlines(keepends=True)
task, count = None, 0
for index, line in enumerate(lines):
    if '\\multirow{5}{*}{Rotation}' in line:
        task = 'rotation'
    elif '\\multirow{5}{*}{Flipping}' in line:
        task = 'flipping'
    if task is None or '&' not in line:
        continue
    cells = line.rstrip().removesuffix('\\\\').split('&')
    assert len(cells) == 21
    row_metric = cells[1].strip()
    summary = results[task]['overall']
    values = {'Success Rate': '%.2f' % summary['success_rate'],
        'Execution Time (s)': ('--' if summary['execution_time_s']['mean'] is None
                               else '%.2f' % summary['execution_time_s']['mean']),
        'Planning Time (s)': '%.3f' % summary['planning_time_s'],
        'Obj. Pos. Error (m)': '%.3f' % summary['final_pos_error_m']['mean'],
        'Obj. Quat. Error': '%.3f' % summary['final_quat_error']['mean']}
    assert cells[-1].strip() == '--', 'Do not overwrite previously reported A7 measurements'
    cells[-1] = ' '+values[row_metric]+' '
    lines[index] = '&'.join(cells)+'\\\\\n'
    count += 1
assert count == 10
text = text[:start]+''.join(lines)+text[end:]
old_caption = 'Comparison of nonprehensile manipulation tasks in the fingertip and robot experiments (200 trials, except the interim A4 results).'
new_caption = ('Comparison of nonprehensile manipulation tasks in the fingertip and robot experiments '
    '(200 trials, except interim A4 results and A7: 21 trials per task). For A7, execution time is success-only; '
    'planning time pools all control cycles, and pose errors include all trials.')
assert text.count(old_caption) == 1
text = text.replace(old_caption, new_caption)
source = '% A7 source: output/benchmarks/a7_mppi_7x3_20261015/statistics.json; 7 objects x seeds 0,1,2 per task; 2500-step limit.\n'
point = text.index('\\end{table*}', text.index('\n\\label{tab:all_data}\n'))
text = text[:point]+source+text[point:]
tex.write_text(text)
manifest.update(status='complete', completed_trials=42, recovered_infrastructure_failures=['mug', 'rubber_duck'])
(out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps({task: results[task]['overall'] for task in TASKS}, indent=2))
