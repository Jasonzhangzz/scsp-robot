"""Section IV reporting conventions, also usable on saved data without simulation.

IV-A: mean +/- population std of ALL individual detected contact depths.
IV-B/C: individual rigid-body trajectories, not time-averaged bars.
IV-D: per-step contact count versus full-step wall time (one world here).
The contact-mode prediction-error table is a new experiment, not a paper metric.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def mean_pm_std(mean, std, digits=None):
    if mean is None or std is None:
        return "undefined"
    return f"{mean:.{digits}f} ± {std:.{digits}f}" if digits is not None else f"{mean:.6g} ± {std:.6g}"


def write_table(path, rows, columns, markdown_columns=None):
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    display = markdown_columns or columns
    lines = ["| " + " | ".join(display) + " |", "| " + " | ".join("---" for _ in display) + " |"]
    lines.extend("| " + " | ".join(str(row.get(c, "")) for c in display) + " |" for row in rows)
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


PREDICTION_METRICS = ("linear_velocity_error_m_s", "angular_velocity_error_rad_s",
                      "position_error_m", "orientation_error_rad", "increment_angle_deg")


def write_prediction_report(reports, output):
    """Keep numerical mean/std and formatted mean +/- std, including undefineds."""
    rows = []
    for configuration, report in reports.items():
        for engine, modes in report["by_reference_mode"].items():
            for mode, group in modes.items():
                row = {"configuration": configuration, "engine": engine, "reference_mode": mode,
                       "attempted": group["attempted"], "valid": group["valid"], "failed": group["failed"]}
                for metric in PREDICTION_METRICS:
                    stats = group["metrics"][metric]
                    row.update({metric + "_count": stats["count"], metric + "_mean": stats["mean"],
                                metric + "_std": stats["std"],
                                metric + "_mean_pm_std": mean_pm_std(stats["mean"], stats["std"])})
                rows.append(row)
    columns = ["configuration", "engine", "reference_mode", "attempted", "valid", "failed"] + [
        metric + suffix for metric in PREDICTION_METRICS for suffix in ("_count", "_mean", "_std", "_mean_pm_std")]
    write_table(Path(output) / "prediction_mean_std", rows, columns,
                ["configuration", "engine", "reference_mode", "valid", "failed"] +
                [metric + "_mean_pm_std" for metric in PREDICTION_METRICS] + ["increment_angle_deg_count"])


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    import matplotlib.pyplot as plt
    plt.close(fig)


def physics_series(scene, trajectory):
    """Exactly the isolated body quantity in Fig. 5/6; no max/mean over bodies."""
    if scene == "ball_rotation":
        if trajectory["qvel"].shape[1] != 1:
            raise ValueError("Torsion reporting requires the isolated z-hinge sphere")
        # Signed z angular velocity preserves reversal; a norm would hide it.
        return {"angular_velocity_z_rad_s": trajectory["qvel"][:, 0]}
    if scene in ("cylinder_rolling", "sliding"):
        if trajectory["body_position_m"].shape[1] != 1:
            raise ValueError("Fig. 5/6 reporting requires one dynamic body")
        if scene == "cylinder_rolling":
            return {"com_speed_m_s": trajectory["body_linear_speed_m_s"][:, 0]}
        return {"horizontal_speed_m_s": trajectory["body_horizontal_speed_m_s"][:, 0],
                "z_position_m": trajectory["body_position_m"][:, 0, 2]}
    return {}


def write_physics_report(output, trajectories, summaries, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    args = metadata["arguments"]
    scene = args["scene"]
    depth_rows = []
    for engine, summary in summaries.items():
        stats = summary.get("penetration_all_contacts_mm")
        if stats is None:
            # Older runs recorded frame maxima only. Never substitute them for
            # paper all-contact mean/std or imply an exact statistical match.
            continue
        depth_rows.append({"engine": engine, "k_user": args["contact_stiffness"] if engine == "comfree" else "",
                           "d_user": args["contact_damping"] if engine == "comfree" else "",
                           "contact_samples": stats["count"], "mean_mm": stats["mean"], "std_mm": stats["std"],
                           "depth_mm_mean_pm_std": mean_pm_std(stats["mean"], stats["std"], digits=1),
                           "completed_steps": summary["completed_steps"], "requested_steps": summary["requested_steps"],
                           "status": summary.get("status", "legacy")})
    write_table(output / "penetration_mean_std", depth_rows,
                ["engine", "k_user", "d_user", "contact_samples", "mean_mm", "std_mm",
                 "depth_mm_mean_pm_std", "completed_steps", "requested_steps", "status"],
                ["engine", "k_user", "d_user", "depth_mm_mean_pm_std", "contact_samples",
                 "completed_steps", "requested_steps", "status"])
    artifacts = {"penetration": "penetration_mean_std.csv/.md"}
    if scene in ("ball_rotation", "cylinder_rolling", "sliding"):
        series = {engine: physics_series(scene, trajectory) for engine, trajectory in trajectories.items()}
        names = list(next(iter(series.values())))
        basename = {"ball_rotation": "torsional_velocity", "cylinder_rolling": "rolling_velocity",
                    "sliding": "stability_trajectory"}[scene]
        rows = []
        for engine, quantities in series.items():
            trajectory = trajectories[engine]
            for step, seconds in enumerate(trajectory["time"]):
                rows.append({"engine": engine, "step": step, "time_s": seconds,
                             **{name: values[step] for name, values in quantities.items()}})
        with (output / f"{basename}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["engine", "step", "time_s", *names])
            writer.writeheader()
            writer.writerows(rows)
        labels = {"angular_velocity_z_rad_s": "Angular velocity about z [rad/s]",
                  "com_speed_m_s": "Rolling COM speed [m/s]", "horizontal_speed_m_s": "Horizontal speed [m/s]",
                  "z_position_m": "z position [m]"}
        # Fig. 5 and Fig. 6-left use steps; Fig. 6-right uses seconds for dt sweeps.
        axes_choices = ("step", "time") if scene == "sliding" else ("step",)
        for x_axis in axes_choices:
            fig, axes = plt.subplots(len(names), 1, figsize=(6.4, 3 * len(names)), sharex=True, squeeze=False)
            for engine, quantities in series.items():
                trajectory = trajectories[engine]
                x = np.arange(len(trajectory["time"])) if x_axis == "step" else trajectory["time"]
                label = engine + (" (partial)" if summaries[engine]["completed_steps"] != summaries[engine]["requested_steps"] else "")
                for axis, name in zip(axes[:, 0], names):
                    axis.plot(x, quantities[name], label=label)
                    axis.set_ylabel(labels[name])
                    axis.grid(alpha=.25)
            axes[0, 0].legend()
            axes[-1, 0].set_xlabel("Simulation step" if x_axis == "step" else "Simulation time [s]")
            save_figure(fig, output / (basename + ("_" + x_axis if scene == "sliding" else "")))
        artifacts["trajectory"] = basename + ".csv"

    runtime_rows = []
    fig, axis = plt.subplots(figsize=(6.4, 4))
    for engine, trajectory in trajectories.items():
        if "step_seconds" not in trajectory or "solve_ncontact" not in trajectory:
            continue
        valid = np.isfinite(trajectory["step_seconds"]) & (trajectory["step_seconds"] > 0)
        indices = np.flatnonzero(valid)
        counts = trajectory["solve_ncontact"][indices]
        milliseconds = trajectory["step_seconds"][indices] * 1000
        axis.scatter(counts, milliseconds, s=9, alpha=.3, label=engine)
        runtime_rows.extend({"engine": engine, "step": int(step), "time_s": trajectory["time"][step],
                             "contact_count": int(count), "full_step_wall_time_ms": elapsed,
                             "nworld": metadata.get("nworld", 1)}
                            for step, count, elapsed in zip(indices, counts, milliseconds))
    if runtime_rows:
        with (output / "runtime_vs_contacts.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(runtime_rows[0]))
            writer.writeheader()
            writer.writerows(runtime_rows)
        axis.set_xlabel("Total detected contacts")
        axis.set_ylabel("Full-step wall time [ms]")
        axis.set_title(f"Contact count vs full-step time ({metadata.get('nworld', 1)} world)")
        axis.legend()
        axis.grid(alpha=.25)
        save_figure(fig, output / "runtime_vs_contacts")
        artifacts["runtime"] = "runtime_vs_contacts.csv/.png/.pdf"
    else:
        plt.close(fig)
    (output / "paper_metrics.json").write_text(json.dumps({
        "scene": scene, "artifacts": artifacts,
        "penetration": {"paper": "Section IV-A / Fig. 3", "unit": "mm", "presentation": "mean ± std",
                        "samples": "all individual detected solve-time contacts; not frame maxima",
                        "std_definition": "population std (ddof=0); paper does not specify ddof",
                        "rows": depth_rows},
        "trajectories": {"paper": "Section IV-B / Fig. 5 or IV-C / Fig. 6", "presentation": "individual body trajectory",
                         "no_time_averaging": True},
        "runtime": {"paper": "Section IV-D / Fig. 7", "presentation": "contact-count vs per-full-step wall-time scatter",
                    "nworld": metadata.get("nworld", 1), "paper_nworld": 512,
                    "scope": "single-world diagnostic; not a reproduction of the paper's 512-world workload"},
        "throughput": {"paper": "Section IV-E / Fig. 8", "presentation": "mean ± std throughput vs nworld",
                       "status": "not measured by this single-world experiment"}
    }, indent=2, allow_nan=False) + "\n")


def read_prediction_rows(path):
    def value(text):
        if not text:
            return None
        if text in ("True", "False"):
            return text == "True"
        try:
            return float(text)
        except ValueError:
            return text
    with path.open() as stream:
        return [{key: value(v) for key, v in row.items()} for row in csv.DictReader(stream)]


def regenerate(directory):
    report_path = directory / "metrics.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if "by_reference_mode" in report:
            write_prediction_report({directory.name: report}, directory)
            if __package__:
                from .compare_prediction import plot_results
            else:
                from compare_prediction import plot_results
            plot_results(read_prediction_rows(directory / "predictions.csv"), directory)
        elif report["metadata"].get("arguments", {}).get("scene"):
            trajectories = {engine: dict(np.load(directory / f"{engine}.npz")) for engine in report["engines"]}
            write_physics_report(directory, trajectories, report["engines"], report["metadata"])
        else:
            raise ValueError(f"{directory}: not a contact physics/prediction report")
    else:
        reports = {}
        for path in sorted(directory.glob("*/metrics.json")):
            report = json.loads(path.read_text())
            if "by_reference_mode" in report:
                regenerate(path.parent)
                reports[path.parent.name] = report
        if not reports:
            raise ValueError(f"{directory}: no supported saved reports")
        write_prediction_report(reports, directory)
    print(f"Updated paper-aligned reports: {directory.resolve()}")


if __name__ == "__main__":
    if __package__:
        from .runtime_env import activate_dependencies
    else:
        from runtime_env import activate_dependencies
    activate_dependencies()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", nargs="+", type=Path, required=True,
                        help="Regenerate tables/plots from saved data; does not rerun physics")
    for directory in parser.parse_args().input_dir:
        regenerate(directory)
