"""Render a user-specified illustrative EMA curve, separate from measured results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from gen_selected import generate_selected, register_times_new_roman

PEAK_X, PEAK_Y = .20, .81
SAMPLED_BETAS = np.array([0., .05, .10, .14, .16, .18, .20, .22, .24, .26,
                         .30, .40, .60, .80, 1.])


def illustrative_rate(beta):
    """Two quadratic branches meet with zero slope at the specified peak."""
    beta = np.asarray(beta, dtype=float)
    return np.where(beta <= PEAK_X,
                    PEAK_Y-.15*((PEAK_X-beta)/PEAK_X)**2,
                    PEAK_Y-.33*((beta-PEAK_X)/(1-PEAK_X))**2)


def generate():
    register_times_new_roman()
    prefix = Path(__file__).with_name('fig_ema_illustrative')
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 16,
                         'axes.labelsize': 18, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
                         'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
                         'mathtext.it': 'Times New Roman:italic',
                         'mathtext.bf': 'Times New Roman:bold',
                         'mathtext.cal': 'Times New Roman',
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    blue, orange = '#0072B2', '#F4A340'
    curve_x = np.linspace(0., 1., 501)
    curve_y = illustrative_rate(curve_x)
    ax.plot(curve_x, curve_y, color=blue, linewidth=1.8, zorder=2)
    dots = SAMPLED_BETAS[SAMPLED_BETAS != PEAK_X]
    ax.scatter(dots, illustrative_rate(dots), s=15, color=blue, zorder=3)
    ax.scatter([PEAK_X], [PEAK_Y], marker='*', s=210, color=orange,
               edgecolor='#AB6500', linewidth=.6, zorder=5)
    ax.annotate('(0.20, 81%)', (PEAK_X, PEAK_Y), xytext=(0, 14),
                textcoords='offset points', ha='center', fontsize=13, color='#AB6500')
    # Keep the label in one text run: mixed MathText emits positioned SVG
    # letters that can overlap in viewers with different font shaping.
    ax.set(xlabel='EMA coefficient β', ylabel='Success Rate',
           xlim=(-.025, 1.025), ylim=(-.035, 1.025))
    ax.set_xticks([0., .2, .4, .6, .8, 1.])
    ax.set_xticklabels(['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'])
    ax.get_xticklabels()[1].set_color('#AB6500')
    ax.get_xticklabels()[1].set_fontweight('bold')
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', color='#E1E1E1', linewidth=.7)
    ax.set_axisbelow(True)
    fig.text(.99, .005, 'Illustrative curve', ha='right', fontsize=10, color='#666666')
    fig.tight_layout(rect=(0, .035, 1, 1))
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(prefix.with_suffix('.'+suffix), dpi=300, bbox_inches='tight')
    plt.close(fig)
    generate_selected()
    rows = [dict(beta=float(x), illustrative_rate=float(y), data_type='illustrative')
            for x,y in zip(SAMPLED_BETAS, illustrative_rate(SAMPLED_BETAS))]
    with prefix.with_suffix('.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    prefix.with_suffix('.json').write_text(json.dumps(dict(
        data_type='illustrative', experimental_measurements=False,
        description='User-specified EMA target curve; all measured sweep results remain separate.',
        peak=dict(beta=PEAK_X, rate=PEAK_Y), endpoints={'0': .66, '1': .48},
        line_color=blue, star_color=orange,
        interpolation='Piecewise quadratic, zero derivative at beta=.20.', points=rows), indent=2)+'\n')
    assert np.allclose(illustrative_rate([0., .2, 1.]), [.66, .81, .48])
    assert np.all(np.diff(curve_y[curve_x <= PEAK_X]) >= 0)
    assert np.all(np.diff(curve_y[curve_x >= PEAK_X]) <= 0)
    print(prefix.with_suffix('.png'))


if __name__ == '__main__':
    generate()
