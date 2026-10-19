import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Patch


def configure_matplotlib(show: bool) -> None:
    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "no-latex"])
    except Exception:
        try:
            plt.style.use("seaborn-v0_8-colorblind")
        except Exception:
            plt.style.use("default")


def compute_bar_width(num_methods: int) -> float:
    if num_methods <= 0:
        raise ValueError("At least one method is required for plotting.")
    return min(0.8 / num_methods, 0.18)


def build_bar_plot(ax, objects, methods, data, colors, bar_width):
    x = np.arange(len(objects))

    for i, (method, scores) in enumerate(zip(methods, data)):
        ax.bar(
            x + (i - (len(methods) - 1) / 2.0) * bar_width,
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


def build_legend(ax, methods, colors):
    handles = [Patch(facecolor=colors[method], label=method) for method in methods]
    ax.legend(handles=handles, loc="center", ncol=3, fontsize=12, title_fontsize=12)
    ax.axis("off")


def validate_inputs(objects, methods, data):
    if len(methods) != len(data):
        raise ValueError(
            f"methods/data length mismatch: {len(methods)} methods vs {len(data)} score arrays."
        )

    for method, scores in zip(methods, data):
        if len(scores) != len(objects):
            raise ValueError(
                f"Score length mismatch for {method}: expected {len(objects)}, got {len(scores)}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fingertip manipulation results.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures interactively after saving them.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "figs"),
        help="Directory where the generated SVG files will be saved.",
    )
    args = parser.parse_args()

    configure_matplotlib(show=args.show)

    import matplotlib.pyplot as plt

    objects = ["Soap", "Teapot", "Donut", "Mug", "Piggy bank", "Bunny", "Elephant"]
    impc = [6, 2, 0, 0, 0, 0, 0]
    cf_mpc = [8, 2, 1, 2, 10, 3, 2]
    ampc = [84, 8, 12, 10, 6, 5, 6]
    sampling = [60, 45, 42, 40, 46, 52, 42]
    scsp = [100, 100, 79, 92, 83, 87, 92]
    wrs=[33, 25, 40, 34, 25, 29, 36]

    methods = ["SCSP*", "Sampling", "A-MPC", "CF-MPC", "I-MPC", "w/o RS"]
    data = [scsp, sampling, ampc, cf_mpc, impc, wrs]
    validate_inputs(objects, methods, data)
    bar_width = compute_bar_width(len(methods))

    colors = {
        "Sampling": "gold",
        "A-MPC": "#109E00",
        "CF-MPC": "#00BFD8",
        "I-MPC": "#0351C7",
        "SCSP*": "#C90000",
        "w/o RS": "#AA00A2",
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    build_bar_plot(ax1, objects, methods, data, colors, bar_width)
    fig1.tight_layout()
    fig1.savefig(output_dir / "result_fingertip.svg", format="svg")

    fig2, ax2 = plt.subplots(figsize=(6.4, 2.8))
    build_legend(ax2, methods, colors)
    fig2.tight_layout()
    fig2.savefig(output_dir / "result_fingertip_bar.svg", format="svg")

    if args.show:
        plt.show()

    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
