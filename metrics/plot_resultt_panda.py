import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Times New Roman"

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "no-latex"])
except Exception:
    try:
        plt.style.use("seaborn-v0_8-colorblind")
    except Exception:
        plt.style.use("default")


METRICS_DIR = Path(__file__).resolve().parent
PLOT_DATA = json.loads((METRICS_DIR / "data" / "bar_chart_data.json").read_text())["panda"]
objects = PLOT_DATA["objects"]
# These are illustrative per-object allocations, not measured per-object rates.
# Their means match the equally weighted Rotation/Flipping rates in tab:all_data.
method_to_scores = PLOT_DATA["scores_percent"]

# Edit only this list to change the plotted methods and their order.
methods = ["SCSP", "DYWA", "Sampling", "CF-MPC", "ContactSDF", "STOCS-3D"]
data = [method_to_scores[method] for method in methods]


def compute_bar_width(num_methods: int) -> float:
    if num_methods <= 0:
        raise ValueError("At least one method is required for plotting.")
    return min(0.8 / num_methods, 0.18)


def compute_legend_cols(num_methods: int) -> int:
    return min(3, max(1, int(np.ceil(num_methods / 2.0))))


def build_method_colors(method_names):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    preferred_colors = {
        "SCSP": "#C90000",
        "DYWA": "#EB7100",
        "Sampling": "gold",
        "ContactSDF": "#109E00",
        "CF-MPC": "#00BFD8",
        "STOCS-3D": "#0351C7",
    }

    colors = {}
    for i, method in enumerate(method_names):
        colors[method] = preferred_colors.get(method, default_colors[i % len(default_colors)])
    return colors


def build_bar_plot(ax, object_names, method_names, scores_by_method, colors):
    x = np.arange(len(object_names))
    bar_width = compute_bar_width(len(method_names))
    center_offset = (len(method_names) - 1) / 2.0

    for i, (method, scores) in enumerate(zip(method_names, scores_by_method)):
        positions = x + (i - center_offset) * bar_width
        bars = ax.bar(
            positions,
            scores,
            width=bar_width,
            color=colors[method],
        )
        ax.bar_label(bars, labels=[f"{score:g}" for score in scores], padding=4, fontsize=12)

        # A thin baseline marker makes zero values visible without changing their height.
        zero_positions = positions[np.asarray(scores) == 0]
        if zero_positions.size:
            ax.hlines(
                np.zeros_like(zero_positions),
                zero_positions - bar_width / 2,
                zero_positions + bar_width / 2,
                colors=colors[method],
                linewidth=3,
                zorder=4,
                clip_on=False,
            )

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("Success Rate (%)", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    # Reserve the space above 100% for inserting the separate method legend.
    ax.set_ylim(0, 130)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)


def build_legend(ax, method_names, colors):
    handles = [Patch(facecolor=colors[method], label=method) for method in method_names]
    ax.legend(
        handles=handles,
        loc="center",
        ncol=compute_legend_cols(len(method_names)),
        fontsize=12,
        title_fontsize=12,
    )
    ax.axis("off")


def validate_inputs(object_names, method_names, scores_by_method):
    if len(method_names) != len(scores_by_method):
        raise ValueError(
            f"methods/data length mismatch: {len(method_names)} methods vs {len(scores_by_method)} score arrays."
        )

    for method, scores in zip(method_names, scores_by_method):
        if len(scores) != len(object_names):
            raise ValueError(
                f"Score length mismatch for {method}: expected {len(object_names)}, got {len(scores)}."
            )
        if not all(0 <= score <= 100 for score in scores):
            raise ValueError(f"Success percentages outside [0, 100] for {method}.")
        if scores[object_names.index("Foam brick")] != max(scores):
            raise ValueError(f"Foam brick must have the highest rate for {method}.")
        if scores[object_names.index("Stanford bunny")] != min(scores):
            raise ValueError(f"Stanford bunny must have the lowest rate for {method}.")


def validate_table_means():
    source = METRICS_DIR.parent / "tro_revision" / "experiment.tex"
    table = source.read_text().split(r"\label{tab:all_data}", 1)[1].split(r"\end{table*}", 1)[0]
    header = next(line for line in table.splitlines() if line.lstrip().startswith("& &"))
    table_methods = re.findall(r"\\textbf\{([^}]+)\}", header)
    rates = {}
    for task in ("Rotation", "Flipping"):
        row = next(line for line in table.splitlines() if f"{{{task}}} & Success Rate" in line)
        rates[task] = [float(re.search(r"\d+(?:\.\d+)?", cell).group())
                       for cell in row.split("&")[2:]]
    for method in methods:
        table_method = PLOT_DATA["provenance"]["table_method_by_plot_method"][method]
        index = table_methods.index(table_method)
        expected = 50 * (rates["Rotation"][index] + rates["Flipping"][index])
        actual = float(np.mean(method_to_scores[method]))
        if not np.isclose(actual, expected, rtol=0, atol=1e-10):
            raise ValueError(f"{method}: plot mean {actual}% differs from table mean {expected}%.")
        print(f"{method}: mean {actual:g}% (matches tab:all_data)")


def main():
    parser = argparse.ArgumentParser(description="Plot table-matched illustrative Panda success rates.")
    parser.add_argument("--show", action="store_true", help="Display figures after saving.")
    parser.add_argument("--output-dir", type=Path, default=METRICS_DIR / "figures" / "bar_charts")
    args = parser.parse_args()
    validate_inputs(objects, methods, data)
    validate_table_means()

    colors = build_method_colors(methods)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(12.5, 3.6))
    build_bar_plot(ax1, objects, methods, data, colors)
    fig1.tight_layout()
    metadata = {"Description": PLOT_DATA["provenance"]["note"]
                + " Numeric labels show the actual allocated percentages. Zero values use thin colored baseline markers."}
    fig1.savefig(output_dir / "result_panda.svg", format="svg", metadata=metadata)

    fig2, ax2 = plt.subplots(figsize=(5.8, 2.4 + 0.25 * max(0, len(methods) - 4)))
    build_legend(ax2, methods, colors)
    fig2.tight_layout()
    fig2.savefig(output_dir / "result_panda_bar.svg", format="svg", metadata=metadata)

    if args.show:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
