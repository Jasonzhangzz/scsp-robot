#!/usr/bin/env python3
"""Plot the cached A1--A7 ablation curves against physical execution time."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

try:
    from trajectory_io import cached_rows, execution_time_s
except ImportError:  # pragma: no cover
    from metrics.trajectory_io import cached_rows, execution_time_s


# Correct the historical summary labels at the point where curves are loaded.
METHODS = ["SCSP*", "A1", "ContactSDF", "A3", "Sampling", "DyWA", "CF-MPC", "STOCS"]
SOURCE_MODES = {
    "SCSP*": "isaac", "A1": "a1", "ContactSDF": "a2", "A3": "a3",
    "Sampling": "a4", "DyWA": "a5", "CF-MPC": "a6", "STOCS": "a7",
}
COLORS = {
    "SCSP*": "#C90000", "A1": "#7F7F7F", "ContactSDF": "#0351C7",
    "A3": "#AA00A2", "Sampling": "gold", "DyWA": "#EB7100",
    "CF-MPC": "#00BFD8", "STOCS": "#109E00",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--trial-id", required=True, type=int)
    parser.add_argument("--execution-time-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-stem", required=True)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def load_curves(root: Path, object_name: str, trial_id: int):
    curves = {}
    times = {}
    for method in METHODS:
        loaded = cached_rows(root, method, object_name, trial_id)
        if loaded is None:
            continue
        _, rows = loaded
        values = []
        for row in rows:
            try:
                values.append(float(row["error"]))
            except (KeyError, TypeError, ValueError):
                values.append(np.nan)
        curves[method] = np.asarray(values, dtype=float)
        times[method] = execution_time_s(rows)
    if not curves:
        raise FileNotFoundError(f"No cached curves for {object_name} trial {trial_id}")
    return curves, times


def main():
    args = parse_args()
    curves, times = load_curves(args.execution_time_root, args.object_name, args.trial_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.save_stem}.csv"
    max_length = max(len(values) for values in curves.values())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([item for method in METHODS for item in
                         (f"{method}_execution_time_s", f"{method}_error")])
        for index in range(max_length):
            row = []
            for method in METHODS:
                values = curves.get(method)
                if values is None or index >= len(values) or np.isnan(values[index]):
                    row.extend(["", ""])
                else:
                    row.extend([f"{times[method][index]:.8f}", f"{values[index]:.8f}"])
            writer.writerow(row)

    matplotlib.use("Agg" if args.no_show else matplotlib.get_backend())
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for method in METHODS:
        if method not in curves:
            continue
        ax.plot(times[method], curves[method], color=COLORS[method], linewidth=1.8, label=method)
    ax.set_xlabel("Execution Time (s)", fontsize=20)
    ax.set_ylabel("Error", fontsize=20)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / f"{args.save_stem}.svg", format="svg")
    if not args.no_show:
        plt.show()
    plt.close(fig)
    print(csv_path)


if __name__ == "__main__":
    main()
