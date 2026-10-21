"""Fill Table IV A6 Flipping after the configured frozen-cohort trials complete."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from tro_revision.update_table_iv_rotation import stats, latex, METRICS, METHODS

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = {'foam_brick', 'stanford_bunny2', 'elephant', 'piggy_bank', 'teapot', 'mug', 'rubber_duck'}


def collect(batch):
    manifest = json.loads((batch/'manifest.json').read_text())
    assert manifest['task'] == 'flipping' and manifest['status'] == 'complete'
    assert manifest['expected_trials'] in (15, 35) and manifest['methods'] == ['a6']
    objects = set(manifest['objects'])
    assert objects <= OBJECTS and len(objects)*5 == manifest['expected_trials']
    rows, hashes = [], {}
    for obj in sorted(objects):
        folder = batch/'a6'/obj
        raw_path = folder/'trials.jsonl'
        raw = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
        assert len(raw) == 5 and sorted(r['trial'] for r in raw) == list(range(5))
        assert all(r['object'] == obj and r['ablation_mode'] == 'a6'
                   and not r.get('aborted') for r in raw)
        path = folder/'normalized_trials.json'
        cell = json.loads(path.read_text())
        assert {(r['seed'], r['success']) for r in cell} == {(r['trial'], bool(r['success'])) for r in raw}
        for record in cell:
            assert record['method'] == 'a6' and record['object'] == obj
            path_metrics = batch/'records'/'a6'/obj/('%03d' % record['seed'])/'metrics.json'
            metrics = json.loads(path_metrics.read_text())
            assert metrics['success'] == record['success'] and not metrics['aborted']
            assert metrics['steps'] == record['steps']
            hashes[str(path_metrics.relative_to(ROOT))] = hashlib.sha256(path_metrics.read_bytes()).hexdigest()
        rows.extend(cell)
        for source in (raw_path, path, folder/'statistics.json'):
            hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(rows) == manifest['expected_trials']
    successful = [r for r in rows if r['success']]
    count = sum(r['planning_count'] for r in rows)
    total = math.fsum(r['planning_total_s'] for r in rows)
    squares = math.fsum(r['planning_sum_squares_s2'] for r in rows)
    result = dict(trials=len(rows), successes=len(successful), success_rate=len(successful)/len(rows),
                  execution_time_s=stats([r['execution_time_s'] for r in successful]),
                  planning_time_ms=dict(n=count, mean=1000*total/count,
                      std=1000*math.sqrt(max(0., (squares-total*total/count)/(count-1)))),
                  final_pos_error_m=stats([r['final_pos_err'] for r in rows]),
                  final_quat_error=stats([r['final_quat_err'] for r in rows]))
    return result, rows, hashes


def update(text, result):
    start = text.index(r'\multirow{5}{*}{Flipping}')
    end = text.index(r'\bottomrule', start)
    lines, count = [], 0
    for line in text[start:end].splitlines(keepends=True):
        if ' & ' in line:
            cells = line.rstrip('\n').rsplit(r' \\', 1)[0].split(' & ')
            if len(cells) == 15 and cells[1] in METRICS:
                cells[METHODS.index('A6')+2] = latex(result[METRICS[cells[1]]])
                line = ' & '.join(cells) + r' \\' + '\n'
                count += 1
        lines.append(line)
    assert count == 5
    updated = text[:start] + ''.join(lines) + text[end:]
    marker = '% A6 Flipping cohort:'
    updated = '\n'.join(line for line in updated.split('\n') if not line.startswith(marker))
    note = '%% A6 Flipping cohort: %d observed trials; %d successes.\n' % (result['trials'], result['successes'])
    location = updated.index(r'\multirow{5}{*}{Flipping}')
    updated = updated[:location] + note + updated[location:]
    note_prefix = r'\parbox{\linewidth}{\scriptsize A6 Flipping:'
    updated = '\n'.join(line for line in updated.split('\n') if not line.startswith(note_prefix))
    cohort_note = note_prefix + ' %d trials across %d objects (5 seeds each).}\n' % (result['trials'], result['trials']//5)
    table_start = updated.index(r'\label{tab:all_data}')
    note_start = updated.index(r'\end{table*}', table_start)
    updated = updated[:note_start] + cohort_note + updated[note_start:]
    return updated


def compile_pdf():
    commands = [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'root.tex'],
                ['bibtex', 'root'],
                ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'root.tex'],
                ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'root.tex']]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT/'tro_revision', stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        if result.returncode:
            raise RuntimeError(result.stdout[-6000:])
    log = (ROOT/'tro_revision/root.log').read_text(errors='replace')
    assert not any(s in log for s in ('There were undefined references', 'There were undefined citations',
                                     'Rerun to get', 'Label(s) may have changed'))
    for suffix in ('.aux', '.log', '.out', '.blg', '.fls', '.fdb_latexmk', '.synctex.gz'):
        (ROOT/'tro_revision'/('root'+suffix)).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    result, rows, hashes = collect(args.batch.resolve())
    snapshot = dict(task='flipping', method='a6', batch=str(args.batch.resolve()),
                    statistics=result, trajectories=rows, source_sha256=hashes,
                    definitions=dict(execution='Successful trajectories only; simulated seconds.',
                        planning='Pooled recorded solver durations over all cycles, including failures; milliseconds.',
                        errors='Terminal errors over all completed cohort trials, including failures.',
                        std='Sample SD, ddof=1; no timing exclusions.'))
    if args.apply:
        paper = ROOT/'tro_revision/experiment.tex'
        text = paper.read_text()
        (args.batch/'experiment.before_a6_flipping.tex').write_text(text)
        paper.write_text(update(text, result))
        (ROOT/'tro_revision/table_iv_a6_flipping_sources.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    if args.compile:
        assert args.apply, '--compile requires --apply'
        compile_pdf()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
