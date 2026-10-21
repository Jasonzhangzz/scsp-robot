"""Refine an acceptance grid around a center, reusing audited matched trials."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from examples.mpc.franka.experiments.mpc_ablation.config import selection_parser
from examples.mpc.franka.experiments.mpc_ablation import sensitivity


def refinement_grid(center, radius, step):
    center, radius, step = map(lambda x: Decimal(str(x)), (center, radius, step))
    if not all(v.is_finite() for v in (center, radius, step)) or radius <= 0 or step <= 0:
        raise ValueError('Center, radius and step must be finite; radius/step positive')
    if radius % step or center-radius < 0 or center+radius > 1:
        raise ValueError('Radius must be divisible by step and grid must be in [0,1]')
    values = [float(center)]
    for i in range(1, int(radius/step)+1):
        values.extend((float(center-i*step), float(center+i*step)))
    return values


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def saved_cells(sources, cases, beta):
    """Sources are newest first. Validate raw outcomes, not only saved rates."""
    found = {}
    for source in sources:
        source = Path(source).resolve()
        manifest = json.loads((source/'grid.json').read_text())
        if manifest.get('mode') != 'acceptance' or manifest.get('verify_beta') != beta:
            raise ValueError('Reuse requires matching acceptance mode and EMA beta')
        for cell in manifest['cells']:
            value = cell['value']
            if value in found:
                continue
            raw = Path(cell['directory'])/'trials.jsonl'
            if not raw.exists():
                continue
            rows = read_rows(raw)
            sensitivity.summarize_mixed(rows, cases)
            sensitivity.validate_acceptance_trials(rows, value)
            if any(row.get('verify_beta') != beta for row in rows):
                raise ValueError('Saved per-trial EMA beta does not match')
            found[value] = dict(source=str(raw), rows=rows)
    return found


def plot_local(records, output, center):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    rows = sorted(records, key=lambda row: row['value'])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([r['value'] for r in rows], [r['success_rate'] for r in rows],
            '-o', color='#C83232', linewidth=1.8, markersize=4)
    selected = next((r for r in rows if r['value'] == center), None)
    if selected:
        ax.scatter([center], [selected['success_rate']], marker='*', s=180,
                   c='#F4A340', edgecolors='#AB6500', linewidths=.6, zorder=5,
                   label='Center: %.2f' % center)
    ax.set(xlabel='Verify acceptance threshold', ylabel='Success rate', ylim=(-.04, 1.04))
    ax.set_xticks([r['value'] for r in rows])
    ax.set_xticklabels(['%.2f'%r['value'] for r in rows], rotation=45, ha='right')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output/('local_success_rate.'+suffix), dpi=300)
    plt.close(fig)


def run(args):
    if args.workers < 1:
        raise ValueError('workers must be positive')
    values = refinement_grid(args.center, args.radius, args.step)
    sources = [Path(p).resolve() for p in args.reuse]
    origin = json.loads((sources[0]/'grid.json').read_text())
    cases, beta = origin['cases'], origin['verify_beta']
    if len(cases) != 5:
        raise ValueError('Require the same five-case EMA cohort')
    saved = saved_cells(sources, cases, beta)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty')
    selection = selection_parser().parse_args(['--acceptance'])
    cells = []
    for i, value in enumerate(values):
        directory = output/('cell_%03d'%i)
        reused = saved.get(value)
        runs = [] if reused else [dict(case_id=c['id'], command=sensitivity._trial_command(
            selection, [], directory/c['id'], c['object'], c['trial'],
            '--verify-quality-threshold', value, beta)) for c in cases]
        cells.append(dict(index=i, value=value, directory=str(directory), runs=runs,
                          reused_source=None if reused is None else reused['source']))
    manifest = dict(mode='acceptance', center=args.center, baseline_value=args.center,
                    radius=args.radius, step=args.step, verify_beta=beta,
                    acceptance_gate_version='quality_gate_v1', trial_num=5,
                    cases=cases, cells=cells, output=str(output),
                    reuse_sources=[str(p) for p in sources],
                    new_rollouts=sum(len(c['runs']) for c in cells),
                    reused_rollouts=sum(5 for c in cells if c['reused_source']))
    if args.dry_run:
        print(json.dumps(manifest, indent=2)); return
    output.mkdir(parents=True, exist_ok=True)
    (output/'grid.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'cohort.json').write_text(json.dumps(cases, indent=2)+'\n')
    records = []
    # Save the audited existing points immediately; never simulate them again.
    order = [c for c in cells if c['reused_source']] + [c for c in cells if not c['reused_source']]
    for cell in order:
        directory = Path(cell['directory']); directory.mkdir()
        value = cell['value']
        if cell['reused_source']:
            rows = saved[value]['rows']
        else:
            print('[refine] running %.2f: 5 matched trials' % value, flush=True)
            def evaluate(entry):
                case, entry_run = entry
                return sensitivity._mixed_trial(entry_run['command'], directory/case['id'],
                    case['object'], case['trial'], expected_beta=beta, expected_acceptance=value)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                rows = list(pool.map(evaluate, zip(cases, cell['runs'])))
        (directory/'trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        record = dict(mode='acceptance', parameter='verify_quality_threshold', value=value,
                      baseline_value=args.center, result_role='sweep', baseline_comparable=True,
                      source_role='reused' if cell['reused_source'] else 'new',
                      source=cell['reused_source'] or str(directory/'trials.jsonl'),
                      **sensitivity.summarize_mixed(rows, cases))
        records.append(record)
        records.sort(key=lambda r:r['value'])
        sensitivity._write_results(output, records, args.center)
        plot_local(records, output, args.center)
        print('[refine] %.2f: %d/5 = %.0f%% (%s)' % (
            value, record['successes'], 100*record['success_rate'], record['source_role']), flush=True)
    (output/'complete.json').write_text(json.dumps(dict(complete=True, grid_points=len(values),
        new_rollouts=manifest['new_rollouts'], reused_rollouts=manifest['reused_rollouts']))+'\n')
    print('[refine] complete: %s' % output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--center', type=float, default=.65)
    parser.add_argument('--radius', type=float, default=.05)
    parser.add_argument('--step', type=float, default=.01)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--reuse', action='append', required=True, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    run(parser.parse_args())
