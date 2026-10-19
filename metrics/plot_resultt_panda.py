import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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


objects = ["Soap", "Teapot", "Donut", "Mug", "Piggy bank", "Bunny", "Elephant"]

sampling = [62, 40, 39, 25, 25, 19, 28]
ampc = [12, 8, 12, 10, 1, 2, 0]
cfmpc = [5, 2, 1, 0, 0, 0, 0]
impc = [2, 1, 0, 0, 0, 0, 0]
wrs = [25, 20, 18, 25, 16, 21, 15]
dywa = [92, 89, 85, 86, 76, 70, 81]
scsp = [100, 98, 83, 90, 83, 85, 91]

method_to_scores = {
    "SCSP*": scsp,
    "DyWA": dywa,
    "Sampling": sampling,
    "A-MPC": ampc,
    "CF-MPC": cfmpc,
    "I-MPC": impc,
    "w/o RS": wrs,
}

# Edit only this list to change the plotted methods and their order.
methods = ["SCSP*", "DyWA", "Sampling", "A-MPC", "CF-MPC", "I-MPC", "w/o RS"]
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
        "SCSP*": "#C90000",
        "DyWA": "#EB7100",
        "Sampling": "gold",
        "A-MPC": "#109E00",
        "CF-MPC": "#00BFD8",
        "I-MPC": "#0351C7",
        "w/o RS": "#AA00A2",
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
        ax.bar(
            x + (i - center_offset) * bar_width,
            scores,
            width=bar_width,
            color=colors[method],
        )

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylabel("Success Rate (%)", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, 110)
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


def main():
    validate_inputs(objects, methods, data)

    colors = build_method_colors(methods)
    output_dir = Path(__file__).resolve().parent / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    build_bar_plot(ax1, objects, methods, data, colors)
    fig1.tight_layout()
    fig1.savefig(output_dir / "result_panda.svg", format="svg")

    fig2, ax2 = plt.subplots(figsize=(5.8, 2.4 + 0.25 * max(0, len(methods) - 4)))
    build_legend(ax2, methods, colors)
    fig2.tight_layout()
    fig2.savefig(output_dir / "result_panda_bar.svg", format="svg")

    plt.show()
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
