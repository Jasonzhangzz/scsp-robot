"""Draw a user-specified illustrative curve; this is not experimental evidence."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from gen_selected import generate_selected, register_times_new_roman

# User-specified anchors and illustrative intermediate values. No simulator data.
POINTS = [
    (0., .01), (.20, .12), (.40, .32), (.50, .46), (.55, .53),
    (.60, .60), (.61, .70), (.62, .76), (.63, .785), (.64, .792),
    (.65, .798), (.66, .803), (.67, .807), (.68, .809), (.69, .8098),
    (.70, .81), (.71, .809), (.72, .807), (.73, .804), (.74, .800),
    (.75, .795), (.80, .75), (.85, .67), (.90, .57), (1., .36),
]


def generate():
    register_times_new_roman()
    output = Path(__file__).with_name('fig_verify_acceptance_illustrative')
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
    ax.plot([x for x,y in POINTS], [y for x,y in POINTS], color=red, linewidth=1.8, zorder=2)
    dots = [(x,y) for x,y in POINTS if x != .70]
    ax.scatter([x for x,y in dots], [y for x,y in dots], s=14, color=red, zorder=3)
    ax.scatter([.70], [.81], marker='*', s=210, color=orange, edgecolor='#AB6500',
               linewidth=.6, zorder=5)
    ax.annotate('(0.70, 81%)', (.70, .81), xytext=(-7, 13), textcoords='offset points',
                ha='center', fontsize=13, color='#AB6500')
    ax.set(xlabel=r'Acceptance Threshold $\bar{\rho}$', ylabel='Success Rate',
           xlim=(-.025, 1.025), ylim=(-.035, 1.025))
    ax.set_xticks([0., .2, .4, .6, .7, .8, 1.])
    ax.set_xticklabels(['0.00', '0.20', '0.40', '0.60', '0.70', '0.80', '1.00'])
    ax.get_xticklabels()[4].set_color('#AB6500')
    ax.get_xticklabels()[4].set_fontweight('bold')
    ax.set_yticks([0., .2, .4, .6, .8, 1.])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', color='#E1E1E1', linewidth=.7)
    ax.set_axisbelow(True)
    fig.text(.99, .005, 'Illustrative curve', ha='right', fontsize=10, color='#666666')
    fig.tight_layout(rect=(0, .035, 1, 1))
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output.with_suffix('.'+suffix), dpi=300, bbox_inches='tight')
    plt.close(fig)
    generate_selected()
    rows = [dict(threshold=x, illustrative_rate=y, data_type='illustrative') for x,y in POINTS]
    with output.with_suffix('.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    output.with_suffix('.json').write_text(json.dumps(dict(
        data_type='illustrative', experimental_measurements=False,
        description='User-specified target shape, separate from all measured trial statistics.',
        peak=dict(threshold=.70, rate=.81), points=rows), indent=2)+'\n')
    print(output.with_suffix('.png'))


if __name__ == '__main__':
    generate()
