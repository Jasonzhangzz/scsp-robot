#!/usr/bin/env python3
"""Heatmap of SCM one-step prediction errors per object and contact state.

Reads the per-object ``predictions.csv`` (and, with ``--include-rollouts``,
``rollouts.jsonl``) written by ``examples/mpc/franka/scm/compare_prediction.py``
and plots a grid with the object geometry on the x axis and the contact state
(transition, sliding, sticking) on the y axis.  One panel per
(reference, metric): SCM--MuJoCo / SCM--IsaacGym x linear / angular.

Each panel is written as its own SVG (Times New Roman, text kept as text);
the four SVGs are then tiled into a single 2x2 composite SVG and converted
to PDF with Inkscape (falls back to a matplotlib PDF if Inkscape is absent).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

CONTACT_STATES = ("transition", "sliding", "sticking")
ROLLOUT_STATES = ("rotation", "flipping")
STATES = CONTACT_STATES  # rows plotted; extended by --include-rollouts
STATE_LABELS = {"transition": "Contact transition", "sliding": "Sliding",
                "sticking": "Sticking", "rotation": "Rotation", "flipping": "Flipping"}
METRICS = (("linear_velocity_error_m_s", "Linear (m/s)"),
           ("angular_velocity_error_rad_s", "Angular (rad/s)"))
SCM_ENGINES = ("scm_box", "scm", "scm_corrected")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="compare_prediction.py output directory (contains <shape>_mu*/predictions.csv)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default: <results-dir>")
    parser.add_argument("--save-stem", default="scm_prediction_heatmap")
    parser.add_argument("--stat", choices=("mean", "median"), default="mean")
    parser.add_argument("--include-rollouts", action="store_true",
                        help="also plot the rotation/flipping rows (one-step errors along the manoeuvres; "
                             "they mix the three contact states, so they are off by default)")
    parser.add_argument("--layout", choices=("grid", "column"), default="grid",
                        help="grid: 2x2 panels (reference x metric); column: two stacked panels "
                             "(linear, angular) with both references as row groups. Both are sized "
                             "to --column-width")
    parser.add_argument("--column-width", type=float, default=3.5, help="total figure width (in)")
    parser.add_argument("--font", default="Times New Roman")
    parser.add_argument("--font-size", type=float, default=None, help="cell-value font size (pt); default per layout")
    parser.add_argument("--tick-size", type=float, default=None, help="axis tick-label font size (pt)")
    parser.add_argument("--title-size", type=float, default=None, help="panel title font size (pt)")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def shape_name(directory: Path) -> str:
    return directory.name.split("_mu")[0]


def load(results_dir: Path):
    """Return {(reference, metric, state, shape): [errors]}."""
    table: dict[tuple, list[float]] = {}
    shapes = []
    for directory in sorted(p.parent for p in results_dir.glob("*/predictions.csv")):
        shape = shape_name(directory)
        shapes.append(shape)
        with (directory / "predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        scm = next((e for e in SCM_ENGINES if any(r["engine"] == e for r in rows)), None)
        if scm is None:
            raise SystemExit(f"no SCM engine rows in {directory}")
        engine_of = {"MuJoCo": scm, "IsaacGym": "scm_vs_isaacgym"}
        for row in rows:
            for reference, engine in engine_of.items():
                if row["engine"] != engine or row.get("status", "ok") != "ok":
                    continue
                for key, _ in METRICS:
                    try:
                        value = float(row[key])
                    except (TypeError, ValueError):
                        continue
                    table.setdefault((reference, key, row["reference_mode"], shape), []).append(value)
        rollouts = directory / "rollouts.jsonl"
        if rollouts.exists():
            for line in rollouts.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for reference, engine in engine_of.items():
                    if row.get("engine") != engine or row.get("status", "ok") != "ok":
                        continue
                    for key, _ in METRICS:
                        value = row.get(key)
                        if value is not None:
                            table.setdefault((reference, key, row["experiment"], shape), []).append(float(value))
    return table, shapes


def main():
    args = parse_args()
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global STATES
    STATES = CONTACT_STATES + (ROLLOUT_STATES if args.include_rollouts else ())
    table, shapes = load(args.results_dir)
    output_dir = args.output_dir or args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    reduce = np.mean if args.stat == "mean" else np.median
    references = ("MuJoCo", "IsaacGym")

    # CSV dump of the grid.
    with (output_dir / f"{args.save_stem}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["reference", "metric", "state", "shape", "n", args.stat, "std"])
        for reference in references:
            for key, _ in METRICS:
                for state in STATES:
                    for shape in shapes:
                        values = table.get((reference, key, state, shape), [])
                        writer.writerow([reference, key, state, shape, len(values),
                                         f"{reduce(values):.4f}" if values else "",
                                         f"{np.std(values):.4f}" if values else ""])

    plt.rcParams.update({
        "font.family": "serif", "font.serif": [args.font, "Times New Roman", "Nimbus Roman", "Liberation Serif"],
        "mathtext.fontset": "stix", "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    pretty = [s.replace("_", " ").replace("stanford bunny2", "stanford bunny") for s in shapes]
    panel_files = []

    def cell_grid(reference, key):
        grid = np.full((len(STATES), len(shapes)), np.nan)
        for i, state in enumerate(STATES):
            for j, shape in enumerate(shapes):
                values = table.get((reference, key, state, shape))
                if values:
                    grid[i, j] = reduce(values)
        return grid

    def annotate(ax, grid, image, norm, fontsize):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isnan(v):
                    text, lum = "--", 1.0
                else:
                    text = f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}"
                    rgba = image.cmap(norm(max(v, norm.vmin)))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, text, ha="center", va="center", fontsize=fontsize,
                        color="white" if lum < 0.5 else "black")

    def log_norm(grid):
        positive = grid[np.isfinite(grid) & (grid > 0)]
        return matplotlib.colors.LogNorm(vmin=max(positive.min(), 1e-3) if positive.size else 1e-3,
                                         vmax=np.nanmax(grid) if positive.size else 1.0)

    if args.layout == "grid":
        # 2x2 panels (reference x metric). Each SVG holds one reference row
        # (linear | angular sharing the state labels); rows are stacked so
        # that the whole grid spans one column.
        tick = args.tick_size or 6.5
        title = args.title_size or 7.5
        cell = args.font_size or 5.2
        short = {"stanford bunny": "bunny", "foam brick": "brick", "rubber duck": "duck",
                 "piggy bank": "piggy"}
        labels = [short.get(name, name) for name in pretty]
        for reference in references:
            height = args.column_width * (0.17 * len(STATES) + 0.72) / 3.5
            fig, axes = plt.subplots(1, len(METRICS), figsize=(args.column_width, height), sharey=True,
                                     constrained_layout=True)
            fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.0, wspace=0.02)
            for ax, (key, label) in zip(axes, METRICS):
                grid = cell_grid(reference, key)
                norm = log_norm(grid)
                image = ax.imshow(np.where(np.isnan(grid), norm.vmin, grid), cmap="YlOrRd",
                                  aspect="auto", norm=norm)
                annotate(ax, grid, image, norm, cell)
                if args.include_rollouts:
                    ax.axhline(len(CONTACT_STATES) - 0.5, color="black", linewidth=0.9)
                ax.set_xticks(range(len(shapes)))
                ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=tick)
                ax.set_yticks(range(len(STATES)))
                ax.set_yticklabels([STATE_LABELS[s].replace("Contact transition", "Transition")
                                    for s in STATES], fontsize=tick)
                ax.tick_params(length=2, pad=1.5)
                ax.set_title(f"SCM\u2013{reference}: {label}", fontsize=title, pad=2.5)
                bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
                bar.ax.tick_params(labelsize=tick - 1, length=2, pad=1.5)
            path = output_dir / f"{args.save_stem}_{reference.lower()}.svg"
            fig.savefig(path)
            panel_files.append(path)
            plt.close(fig)
        columns = 1
    else:
        # Two stacked panels (linear, angular); rows = reference x state.
        tick = args.tick_size or 8.5
        title = args.title_size or 10.0
        cell = args.font_size or 7.5
        for key, label in METRICS:
            grid = np.vstack([cell_grid(reference, key) for reference in references])
            height = args.column_width * (0.28 * grid.shape[0] + 1.05) / 3.5
            fig, ax = plt.subplots(figsize=(args.column_width, height), constrained_layout=True)
            norm = log_norm(grid)
            image = ax.imshow(np.where(np.isnan(grid), norm.vmin, grid), cmap="YlOrRd", aspect="auto", norm=norm)
            annotate(ax, grid, image, norm, cell)
            n_state = len(STATES)
            for r in range(len(references)):
                if args.include_rollouts:
                    ax.axhline(r * n_state + len(CONTACT_STATES) - 0.5, color="black", linewidth=0.8, linestyle=":")
                if r:
                    ax.axhline(r * n_state - 0.5, color="black", linewidth=1.4)
                ax.text(-0.33, 1 - (r * n_state + (n_state - 1) / 2 + 0.5) / grid.shape[0],
                        f"SCM\u2013{references[r]}", rotation=90, ha="center", va="center",
                        fontsize=tick, transform=ax.transAxes, clip_on=False)
            ax.set_xticks(range(len(shapes)))
            ax.set_xticklabels(pretty, rotation=35, ha="right", fontsize=tick)
            ax.set_yticks(range(grid.shape[0]))
            ax.set_yticklabels([STATE_LABELS[s].replace("Contact transition", "Transition")
                                for _ in references for s in STATES], fontsize=tick)
            ax.set_title(label.replace(" (", " velocity error ("), fontsize=title)
            bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
            bar.ax.tick_params(labelsize=tick - 1)
            path = output_dir / f"{args.save_stem}_{key.split('_')[0]}.svg"
            fig.savefig(path)
            panel_files.append(path)
            plt.close(fig)
        columns = 1
    composite = compose_grid(panel_files, output_dir / f"{args.save_stem}.svg", columns=columns,
                             gap_pt=2.0 if args.layout == "grid" else 6.0)
    pdf = output_dir / f"{args.save_stem}.pdf"
    if shutil.which("inkscape"):
        subprocess.run(["inkscape", str(composite), "--export-type=pdf", f"--export-filename={pdf}"],
                       check=True, capture_output=True)
    else:  # pragma: no cover
        print("inkscape not found; composite PDF not written")
    print("saved", ", ".join(p.name for p in panel_files), composite.name, pdf.name)


_SIZE = re.compile(r'<svg[^>]*?\swidth="([\d.]+)pt"[^>]*?\sheight="([\d.]+)pt"', re.S)


def compose_grid(svg_files, output, columns=2, gap_pt=6.0):
    """Tile standalone SVG files into one SVG by nesting them as <svg x y> elements."""
    panels = []
    for path in svg_files:
        text = path.read_text(encoding="utf-8")
        match = _SIZE.search(text)
        if match is None:
            raise SystemExit(f"cannot read size of {path}")
        width, height = float(match.group(1)), float(match.group(2))
        start = text.index("<svg")
        body = text[start:text.rindex("</svg>") + len("</svg>")]
        # Give every id a per-panel prefix so clip paths do not collide.
        prefix = f"p{len(panels)}_"
        body = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', body)
        body = re.sub(r'(xlink:href|href)="#([^"]+)"', lambda m: f'{m.group(1)}="#{prefix}{m.group(2)}"', body)
        body = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{prefix}{m.group(1)})', body)
        panels.append((body, width, height))
    rows = (len(panels) + columns - 1) // columns
    col_width = [max(p[1] for p in panels[c::columns]) for c in range(columns)]
    row_height = [max(p[2] for p in panels[r * columns:(r + 1) * columns]) for r in range(rows)]
    total_w = sum(col_width) + gap_pt * (columns - 1)
    total_h = sum(row_height) + gap_pt * (rows - 1)
    parts = [f'<?xml version="1.0" encoding="utf-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" '
             f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{total_w:.2f}pt" height="{total_h:.2f}pt" '
             f'viewBox="0 0 {total_w:.2f} {total_h:.2f}">']
    for index, (body, width, height) in enumerate(panels):
        r, c = divmod(index, columns)
        x = sum(col_width[:c]) + gap_pt * c
        y = sum(row_height[:r]) + gap_pt * r
        body = body.replace("<svg", f'<svg x="{x:.2f}" y="{y:.2f}"', 1)
        body = re.sub(r'\swidth="[\d.]+pt"', f' width="{width:.2f}"', body, count=1)
        body = re.sub(r'\sheight="[\d.]+pt"', f' height="{height:.2f}"', body, count=1)
        parts.append(body)
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


if __name__ == "__main__":
    main()
