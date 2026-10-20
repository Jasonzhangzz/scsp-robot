"""Plot EMA sensitivity results with a separate legend figure.

Run from any directory with the experiment's Python environment. Outputs SVG,
PDF, PNG, and a JSON snapshot beside this script; no simulation is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from planning.verify_control import VERIFY_CONTROL_VERSION
CORRECTED_SOURCES = sorted((path for path in ROOT.glob('output/ablation/ema_corrected_*')
                            if path.is_dir()), reverse=True)
DEFAULT_SOURCES = CORRECTED_SOURCES[:1] or [
    ROOT / 'output/ablation/ema_reuse_baseline_20261020_172744',
    ROOT / 'output/ablation/ema_wide_20261020_174857',
]


def collect(sources):
    completed = {}
    reference = None
    baseline = None
    for source in sources:
        manifest = json.loads((source / 'grid.json').read_text())
        cohort = [(case['object'], case['trial'], case['baseline_success'])
                  for case in manifest['cases']]
        if reference is not None and cohort != reference:
            raise ValueError('Cannot combine different trajectory cohorts')
        reference = cohort
        baseline = manifest['baseline_value']
        summary = source / 'sensitivity_summary.json'
        if summary.exists():
            for row in json.loads(summary.read_text()):
                if row.get('result_role') == 'historical_reference':
                    continue
                if row.get('verify_control_version') != VERIFY_CONTROL_VERSION:
                    raise ValueError('Legacy EMA results used a binary MPC weight; '
                                     'supply corrected results with --sources: ' + str(summary))
                if row['trials'] != len(cohort):
                    raise ValueError('Incomplete cell in saved summary')
                beta = row['value']
                if beta in completed and completed[beta]['successes'] != row['successes']:
                    raise ValueError('Conflicting completed results for beta=%g' % beta)
                completed[beta] = dict(beta=beta, status='complete',
                                       successes=row['successes'], trials=row['trials'],
                                       success_rate=row['success_rate'], source=str(summary))
    # Only the latest scan's pending points remain scheduled. Earlier cancelled
    # grids contribute completed results, never their abandoned pending cells.
    if not completed:
        raise ValueError('No completed, verified EMA results are available yet')
    records = dict(completed)
    for cell in manifest['cells']:
        beta = cell['value']
        if beta not in completed and beta != baseline:
            count = len(list(Path(cell['directory']).glob('*/trials.jsonl')))
            records[beta] = dict(beta=beta, status='pending', completed_trials=count,
                                 trials=len(reference), success_rate=None,
                                 source=str(sources[-1] / 'grid.json'))
    return dict(baseline=baseline, cohort=reference,
                records=[records[beta] for beta in sorted(records)])


def plot(snapshot, output):
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 14,
        'axes.labelsize': 18, 'legend.fontsize': 14,
        'xtick.labelsize': 14, 'ytick.labelsize': 14,
        'axes.spines.top': False, 'axes.spines.right': False,
        'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })
    blue, orange = '#0072B2', '#D55E00'
    complete = [row for row in snapshot['records'] if row['status'] == 'complete']
    fig, ax = plt.subplots(figsize=(7.3, 3.5))
    fig.subplots_adjust(left=.14, right=.97, bottom=.23, top=.95)
    ax.plot([row['beta'] for row in complete],
            [100 * row['success_rate'] for row in complete],
            color=blue, marker='o', markersize=5, linewidth=1.5,
            label='EMA sweep', zorder=3)
    baseline = next((row for row in complete if row['beta'] == snapshot['baseline']), None)
    if baseline is not None:
        ax.scatter([baseline['beta']], [100 * baseline['success_rate']], marker='D',
                   s=47, color=orange, edgecolors='white', linewidths=.7,
                   label=r'Default $\beta=%.2f$' % snapshot['baseline'], zorder=4)
    ax.set_xlim(-.025, 1.065)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    # Regular numeric ticks keep the larger text readable around the closely
    # spaced beta=0.16, 0.18 and 0.20 samples. All measured points stay plotted.
    ax.set_xticks([0, .2, .4, .6, .8, 1.0])
    ax.set_xlabel(r'EMA coefficient $\beta$')
    ax.set_ylabel('Success Rate (%)')
    ax.grid(axis='y', color='#E3E3E3', linewidth=.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(output.with_suffix('.' + suffix), dpi=300)
    plt.close(fig)
    legend_fig = plt.figure(figsize=(5.4, .65))
    legend_fig.legend(handles, labels, loc='center', ncol=2, frameon=False,
                      handlelength=2, columnspacing=1.8)
    legend_output = output.with_name(output.name + '_legend')
    for suffix in ('svg', 'pdf', 'png'):
        legend_fig.savefig(legend_output.with_suffix('.' + suffix), dpi=300,
                           bbox_inches='tight', pad_inches=.05)
    plt.close(legend_fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', type=Path, default=DEFAULT_SOURCES)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('fig_ema_sensitivity'))
    args = parser.parse_args()
    snapshot = collect(args.sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    plot(snapshot, args.output)
    print(args.output.with_suffix('.svg'))
    print(json.dumps(snapshot['records'], indent=2))


if __name__ == '__main__':
    main()
