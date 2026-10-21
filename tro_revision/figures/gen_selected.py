"""Render the shared star legend for the two illustrative curves."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager


def register_times_new_roman():
    # Register installed fonts explicitly when Matplotlib has an older font cache.
    for path in font_manager.findSystemFonts(fontext='ttf'):
        if 'times' in Path(path).name.lower():
            font_manager.fontManager.addfont(path)
    font_manager.findfont('Times New Roman', fallback_to_default=False)


def generate_selected():
    register_times_new_roman()
    output = Path(__file__).with_name('selected')
    with plt.rc_context({'font.family': 'Times New Roman', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none'}):
        fig = plt.figure(figsize=(1.8, .45))
        star = Line2D([], [], linestyle='none', marker='*', markersize=14.5,
                      markerfacecolor='#F4A340', markeredgecolor='#AB6500',
                      markeredgewidth=.6)
        fig.legend([star], ['Selected'], loc='center', frameon=False,
                   fontsize=16, handlelength=1, handletextpad=.5)
        for suffix in ('png', 'pdf', 'svg'):
            fig.savefig(output.with_suffix('.' + suffix), dpi=300,
                        bbox_inches='tight', pad_inches=.05)
        plt.close(fig)


if __name__ == '__main__':
    generate_selected()
