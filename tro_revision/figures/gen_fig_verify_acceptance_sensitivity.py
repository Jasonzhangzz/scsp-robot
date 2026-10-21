"""Merge saved acceptance trials and render a sparse red curve around selected .75.

No simulations are imported. New outcomes replace old outcomes at matching values;
the five trials from two runs are never pooled into a ten-trial estimate.
"""
import argparse
import csv
import json
from pathlib import Path

from gen_selected import register_times_new_roman

DISPLAY_GRID = (0., .30, .50, .65, .70, .75, .80, .85, 1.)
RERUN_GRID = (.65, .70, .75, .80, .85)
SELECTED_THRESHOLD = .75


def merge_results(previous, current):
    merged = {}
    for path, origin in ((Path(previous), 'reused'), (Path(current), 'rerun')):
        rows = json.loads(path.read_text())
        for row in rows:
            value = round(float(row['value']), 12)
            if row['trials'] != 5 or row.get('result_role', 'sweep') != 'sweep':
                raise ValueError('Expected five completed sweep trials at %s' % value)
            successes = int(row['successes'])
            if not 0 <= successes <= 5 or abs(row['success_rate']-successes/5.) > 1e-12:
                raise ValueError('Invalid success rate at %s' % value)
            if origin == 'rerun' and value not in RERUN_GRID:
                raise ValueError('Only the five requested values may be recomputed')
            merged[value] = dict(value=value, trials=5, successes=successes,
                                 success_rate=successes/5., origin=origin, source=str(path.resolve()))
    if SELECTED_THRESHOLD not in merged:
        raise ValueError('Wait for the five .75 trials before drawing the selected star')
    return merged


def generate(previous, current, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    register_times_new_roman()

    merged = merge_results(previous, current)
    plotted = [merged[v] for v in DISPLAY_GRID if v in merged]
    complete = all(v in merged and merged[v]['origin'] == 'rerun' for v in RERUN_GRID)
    prefix = Path(output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 16,
                         'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
                         'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
                         'mathtext.it': 'Times New Roman:italic',
                         'mathtext.bf': 'Times New Roman:bold',
                         'mathtext.cal': 'Times New Roman',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    red, orange = '#C83232', '#F4A340'
    ax.plot([r['value'] for r in plotted], [r['success_rate'] for r in plotted],
            color=red, linewidth=1.8, zorder=2)
    dots = [r for r in plotted if r['value'] != SELECTED_THRESHOLD]
    ax.scatter([r['value'] for r in dots], [r['success_rate'] for r in dots],
               s=28, color=red, zorder=3)
    ax.scatter([SELECTED_THRESHOLD], [merged[SELECTED_THRESHOLD]['success_rate']],
               marker='*', s=210, color=orange, edgecolor='#AB6500', linewidth=.6,
               label='Selected: 0.75', zorder=5)
    ax.set(xlabel=r'Acceptance Threshold $\bar{\rho}$', ylabel='Success Rate',
           xlim=(-.025, 1.025), ylim=(-.045, 1.045))
    ticks = [r['value'] for r in plotted]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{v:.2f}' for v in ticks], fontsize=16,
                       rotation=45, ha='right', rotation_mode='anchor')
    for value, label in zip(ticks, ax.get_xticklabels()):
        if value == SELECTED_THRESHOLD:
            label.set_color('#AB6500')
            label.set_fontweight('bold')
    ax.set_yticks([i/5 for i in range(6)])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', color='#E1E1E1', linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False, fontsize=14)
    if not complete:
        ax.set_title('Interim results: requested reruns in progress', fontsize=14)
    fig.tight_layout()
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(prefix.with_suffix('.'+suffix), dpi=300, bbox_inches='tight')
    plt.close(fig)
    records = [merged[v] for v in sorted(merged)]
    with prefix.with_suffix('.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = dict(selected_threshold=SELECTED_THRESHOLD,
                    rerun_values=list(RERUN_GRID), rerun_complete=complete,
                    points=plotted, all_results=records,
                    pending_values=[v for v in RERUN_GRID
                                    if v not in merged or merged[v]['origin'] != 'rerun'],
                    trials_per_point=5, new_rollouts=25,
                    note='New results replace prior estimates; runs are not pooled.')
    prefix.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--current', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('fig_verify_acceptance_sensitivity'))
    args = parser.parse_args()
    result = generate(args.previous, args.current, args.output)
    print(json.dumps({key: result[key] for key in
                      ('selected_threshold', 'rerun_complete', 'pending_values')}, indent=2))
