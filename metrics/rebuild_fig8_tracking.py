#!/usr/bin/env python3
"""Digitize the archived Fig. 8(b) PNGs and export editable vector plots.

These are approximate reconstructions of plotted pixels, not recovered trial
measurements. Time factors are user-specified, not measured execution timings.
Run with /usr/bin/python3 (Matplotlib, NumPy, SciPy and Pillow are required).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, label

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/fig8_tracking'
OUTPUT = ROOT / 'figures/fig8_tracking'
CONFIGS = [
    dict(image=177, case=1, metric='pos', factor=.016,
         x_pixels=[340.5, 1632.5], x_values=[0, 1200],
         y_pixels=[1121, 142], y_values=[0, .175], ylim=[-.008, .195]),
    dict(image=178, case=1, metric='quat', factor=.016,
         x_pixels=[300, 1641], x_values=[0, 1200],
         y_pixels=[966, 80], y_values=[0, 1], ylim=[-.225, 1.06]),
    dict(image=179, case=2, metric='pos', factor=.013,
         x_pixels=[340.5, 1386], x_values=[0, 1500],
         y_pixels=[1118, 115.5], y_values=[0, .2], ylim=[-.009, .2175]),
    dict(image=180, case=2, metric='quat', factor=.013,
         x_pixels=[279.5, 1380], x_values=[0, 1500],
         y_pixels=[1097.5, 131.5], y_values=[0, 1], ylim=[-.038, 1.107]),
]
COLORS = {'pos': '#1f77b4', 'quat': '#ff7f0e'}


def linear_map(pixels, pixel_refs, value_refs):
    p0, p1 = pixel_refs
    v0, v1 = value_refs
    return v0 + (np.asarray(pixels) - p0) * (v1 - v0) / (p1 - p0)


def digitize(config, smooth_sigma_steps, smooth_blend):
    rgb = np.asarray(Image.open(DATA / f"image{config['image']}.png").convert('RGB'))
    r, g, b = (rgb[:, :, i].astype(float) for i in range(3))
    target = np.array([31, 119, 180] if config['metric'] == 'pos' else [255, 127, 14])
    mean_mask = np.linalg.norm(rgb.astype(float) - target, axis=2) < 35
    if config['metric'] == 'pos':
        band_mask = (r > 100) & (r < 230) & (b - r > 20) & (g > r) & (b > g)
    else:
        band_mask = (r > 245) & (g > 150) & (g < 240) & (g - b > 20)
    # Exclude axes, labels, and the legend in the original image.
    for mask in (mean_mask, band_mask):
        mask[:35] = False
        mask[1160:] = False
        mask[:, :round(config['x_pixels'][0])] = False
        mask[:260, 1335:] = False
    # Antialiasing along the mean can have the same pale colors as the band.
    # Retain only large connected filled regions, excluding those thin fringes.
    components, _ = label(band_mask)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    band_mask &= sizes[components] > 2000
    xs = np.flatnonzero(mean_mask.any(axis=0))
    xs = np.arange(xs[0], xs[-1] + 1)
    mean_y = np.array([np.median(np.flatnonzero(mean_mask[:, x]))
                       if mean_mask[:, x].any() else np.nan for x in xs])
    valid = np.isfinite(mean_y)
    mean_y = np.interp(xs, xs[valid], mean_y[valid])
    steps = linear_map(xs, config['x_pixels'], config['x_values'])
    steps -= steps[0]  # The first plotted sample is timestep zero.
    mean = linear_map(mean_y, config['y_pixels'], config['y_values'])
    upper = np.full(len(xs), np.nan)
    lower = np.full(len(xs), np.nan)
    for i, x in enumerate(xs):
        ys = np.flatnonzero(band_mask[:, x])
        if len(ys) >= 4:
            upper[i], lower[i] = linear_map([ys[0], ys[-1]],
                                           config['y_pixels'], config['y_values'])
    # Keep the visible band as digitized. Its extent is shorter than the mean
    # in the source; no standard deviations are invented for the missing tail.
    sigma_pixels = smooth_sigma_steps / np.median(np.diff(steps))
    smooth = (1 - smooth_blend) * mean + smooth_blend * gaussian_filter1d(
        mean, sigma_pixels, mode='nearest')
    smooth[0], smooth[-1] = mean[0], mean[-1]
    return np.column_stack([steps, steps * config['factor'], mean, smooth, lower, upper])


def plot(ax, config, values):
    color = COLORS[config['metric']]
    ax.plot(values[:, 1], values[:, 3], color=color, lw=1.8, label='Mean')
    valid = np.isfinite(values[:, 4:]).all(axis=1)
    ax.fill_between(values[:, 1], values[:, 4], values[:, 5], where=valid,
                    color=color, alpha=.28, lw=.6, edgecolor=color, label='Std.')
    ax.set_xlabel('Execution Time (s)', fontsize=21)
    ax.set_ylabel('Pos. Error (m)' if config['metric'] == 'pos' else 'Quat. Error',
                  fontsize=21)
    ax.tick_params(axis='both', labelsize=18, length=4, width=.9)
    ax.set_ylim(config['ylim'])
    if config['case'] == 1:
        ax.set_xticks([0, 4, 8, 12, 16, 20])
        ax.set_xlim(-.85, 20.6)
    else:
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.set_xlim(-1.2, 25.8)
    ax.legend(loc='upper right', fontsize=14, framealpha=.9)
    for spine in ax.spines.values():
        spine.set_linewidth(.9)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pptx', type=Path,
                        help='Extract image177.png through image180.png from this archive.')
    parser.add_argument('--smooth-sigma-steps', type=float, default=6.)
    parser.add_argument('--smooth-blend', type=float, default=.8)
    args = parser.parse_args()
    if args.smooth_sigma_steps <= 0 or not 0 <= args.smooth_blend <= 1:
        parser.error('sigma must be positive and blend must be in [0, 1]')
    DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.source_pptx:
        with ZipFile(args.source_pptx) as archive:
            for config in CONFIGS:
                name = f"image{config['image']}.png"
                (DATA / name).write_bytes(archive.read(f'ppt/media/{name}'))
    font_path = font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman',
                         'font.serif': ['Times New Roman'],
                         'svg.fonttype': 'none', 'svg.hashsalt': 'fig8-tracking'})
    description = ('Approximate digitization of raster Fig. 8(b), archived in '
                   '论文所有图.pptx, ppt/media/image177.png through image180.png. '
                   'Not original trial data. All text uses Times New Roman. '
                   f'Mean smoothed for display with Gaussian sigma {args.smooth_sigma_steps:g} '
                   f'timesteps and blend {args.smooth_blend:g}; '
                   'visible source standard-deviation band retained. '
                   'User-specified timestep scaling: case 1 x0.016; case 2 x0.013. '
                   'These factors are not recovered execution-time measurements.')
    metadata = {'Description': description, 'Date': None}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    manifest = {'description': description, 'font_family': 'Times New Roman',
                'font_file': Path(font_path).name, 'smooth_sigma_steps': args.smooth_sigma_steps,
                'smooth_blend': args.smooth_blend, 'plots': []}
    for config, ax in zip(CONFIGS, axes.flat):
        name = f"case{config['case']}_{config['metric']}_error_execution_time"
        values = digitize(config, args.smooth_sigma_steps, args.smooth_blend)
        np.savetxt(DATA / f'{name}.csv', values, delimiter=',', fmt='%.10g', comments='',
                   header='timestep,execution_time_s,digitized_mean,display_mean,visible_band_lower,visible_band_upper')
        plot(ax, config, values)
        single, single_ax = plt.subplots(figsize=(6, 4.25), layout='constrained')
        plot(single_ax, config, values)
        single.savefig(OUTPUT / f'{name}.svg', metadata=metadata)
        plt.close(single)
        total_variation = lambda x: float(np.abs(np.diff(x)).sum())
        details = dict(config, source_sha256=hashlib.sha256(
            (DATA / f"image{config['image']}.png").read_bytes()).hexdigest(),
            final_time_s=float(values[-1, 1]),
            mean_total_variation=total_variation(values[:, 2]),
            display_total_variation=total_variation(values[:, 3]))
        manifest['plots'].append(details)
        print(f"{name}: {values[-1, 1]:.3f} s; variation "
              f"{details['display_total_variation'] / details['mean_total_variation']:.1%} of original")
    fig.savefig(OUTPUT / 'fig8b_tracking_execution_time.svg', metadata=metadata)
    fig.savefig(OUTPUT / 'fig8b_tracking_execution_time_preview.png', dpi=130)
    plt.close(fig)
    (DATA / 'reconstruction.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
