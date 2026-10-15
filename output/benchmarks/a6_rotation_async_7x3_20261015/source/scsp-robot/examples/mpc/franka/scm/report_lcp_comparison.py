#!/usr/bin/env python3
"""Offline paired soft-SCM differences against MJWarp and LCP, two decimals.

The table pools samplewise absolute differences in maximum POST-STATE
penetration and velocity-vector difference norms. Raw values are saved in SI
units; display units expose small nonzero values without rounding them away.
"""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

ENGINES = ("mjwarp", "lcp", "scm_soft")
MODES = ("separation", "sliding", "sticking")
METRICS = ("penetration_difference_m", "linear_velocity_error_m_s", "angular_velocity_error_rad_s")
ENGINE_METRICS = ("poststate_max_penetration_m", "linear_velocity_error_m_s", "angular_velocity_error_rad_s")


def display_units(mode, metric):
    if metric == "penetration_difference_m":
        return 1.e6, "um", r"$e_d$ ($\mu$m)"
    if metric == "linear_velocity_error_m_s":
        return (1.e9, "nm/s", r"$e_v$ (nm/s)") if mode == "separation" else (1.e3, "mm/s", r"$e_v$ (mm/s)")
    if metric == "angular_velocity_error_rad_s":
        return (1.e9, "nrad/s", r"$e_\omega$ (nrad/s)") if mode == "separation" else (1., "rad/s", r"$e_\omega$ (rad/s)")
    raise ValueError(f"Unknown paired metric: {metric}")


def paired_penetration(rows, baseline):
    grouped = {engine: {r["sample_id"]: r for r in rows if r["engine"] == engine}
               for engine in ("scm_soft", baseline)}
    if grouped["scm_soft"].keys() != grouped[baseline].keys():
        raise ValueError("Unmatched penetration sample IDs")
    values = {mode: [] for mode in MODES}
    for sample_id, scm in grouped["scm_soft"].items():
        ref = grouped[baseline][sample_id]
        if scm["reference_mode"] != ref["reference_mode"]:
            raise ValueError("Unmatched penetration reference modes")
        values[scm["reference_mode"]].append(abs(float(scm["poststate_max_penetration_m"])
                                                - float(ref["poststate_max_penetration_m"])))
    return values


def statistics(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite values in comparison report")
    if not values.size:
        return {"count": 0, "mean": None, "std": None, "max": None}
    return {"count": int(values.size), "mean": float(values.mean()),
            "std": float(values.std()), "max": float(values.max())}


def formatted(stats, latex=False):
    if stats["count"] == 0:
        return "--"
    mean = max(0., stats["mean"])
    separator = r"\pm" if latex else "±"
    value = f"{mean:.2f} {separator} {stats['std']:.2f}"
    return f"${value}$" if latex else value


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(root):
    manifest = json.loads((root / "manifest.json").read_text())
    settings = manifest["arguments"]
    all_rows, checks, paired, depths = [], [], [], {}
    residuals = []
    configurations = manifest["configurations"]
    if len(configurations) != len(settings["shapes"]) * len(settings["frictions"]) * len(settings["timesteps"]):
        raise ValueError("Incomplete configuration manifest")
    for configuration in configurations:
        folder = root / configuration
        metrics = json.loads((folder / "metrics.json").read_text())
        expected = len(settings["modes"]) * settings["samples_per_mode"]
        if metrics["stopped_by_viewer"] or metrics["completed_shared_states"] != expected:
            raise ValueError(f"Incomplete configuration: {configuration}")
        with (folder / "predictions.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != expected * len(ENGINES) or any(r["status"] != "ok" for r in rows):
            raise ValueError(f"Missing or failed predictions: {configuration}")
        if {r["engine"] for r in rows} != set(ENGINES):
            raise ValueError("This report requires exactly MJWarp, LCP, soft-SCM")
        if any(r["reference_mode"] not in MODES for r in rows):
            raise ValueError("Transition/invalid inputs present; report them before pooling pure modes")
        all_rows.extend(dict(r, configuration=configuration) for r in rows)
        for baseline in ("mjwarp", "lcp"):
            for mode, values in paired_penetration(rows, baseline).items():
                paired.append({"configuration": configuration, "mode": mode,
                               "baseline": baseline, "metric": "penetration_difference_m", "values": values})
        residuals.extend(float(r["lcp_residual"]) for r in rows if r["engine"] == "lcp")
        if max(residuals) > settings["lcp_tolerance"]:
            raise ValueError("LCP certificate exceeds requested tolerance")
        with np.load(folder / "states.npz", allow_pickle=False) as states:
            indices = {e: np.flatnonzero(states["engine"] == e) for e in ENGINES}
            ref = indices["mjwarp"]
            for engine, selected in indices.items():
                for key in ("sample_id", "qpos_initial", "qvel_initial", "wrench_world_N_Nm"):
                    np.testing.assert_array_equal(states[key][ref], states[key][selected])
            for baseline in ("mjwarp", "lcp"):
                difference = states["qvel_predicted"][indices["scm_soft"]] - states["qvel_predicted"][indices[baseline]]
                labels = states["reference_mode"][ref]
                for mode in MODES:
                    mask = labels == mode
                    for name, block in (("linear_velocity_error_m_s", slice(0, 3)),
                                        ("angular_velocity_error_rad_s", slice(3, 6))):
                        paired.append({"configuration": configuration, "mode": mode,
                                       "baseline": baseline, "metric": name,
                                       "values": np.linalg.norm(difference[mask, block], axis=1).tolist()})
        checks.append({"configuration": configuration, "shared_states": expected,
                       "all_inputs_exact": True, "failed_predictions": 0})
        with gzip.open(folder / "poststate_contacts.csv.gz", "rt") as stream:
            for row in csv.DictReader(stream):
                depths.setdefault((row["engine"], row["reference_mode"]), []).append(float(row["depth_m"]) * 1000)

    summaries, by_object = [], []
    for engine in ENGINES:
        for mode in MODES:
            selected = [r for r in all_rows if r["engine"] == engine and r["reference_mode"] == mode]
            if len(selected) != len(configurations) * settings["samples_per_mode"]:
                raise ValueError("Measured pure-mode counts differ from intended sample counts")
            for key in ENGINE_METRICS:
                stats = statistics([float(r[key]) for r in selected])
                summaries.append({"engine": engine, "mode": mode, "metric": key,
                                  **stats})
                for configuration in configurations:
                    values = [float(r[key]) for r in selected if r["configuration"] == configuration]
                    by_object.append({"configuration": configuration, "engine": engine,
                                      "mode": mode, "metric": key, **statistics(values)})
    write_csv(root / "per_engine_statistics_si.csv", summaries)
    write_csv(root / "per_engine_by_object_si.csv", by_object)
    pair_stats, paired_by_object = [], []
    for baseline in ("mjwarp", "lcp"):
        for mode in MODES:
            for key in METRICS:
                values = [v for r in paired if (r["baseline"], r["mode"], r["metric"]) == (baseline, mode, key) for v in r["values"]]
                scale, unit, _ = display_units(mode, key)
                raw, shown = statistics(values), statistics(np.asarray(values) * scale)
                pair_stats.append({"model": "scm_soft", "baseline": baseline, "mode": mode,
                                   "metric": key, **raw, "display_unit": unit, "display_scale": scale,
                                   "display_mean": shown["mean"], "display_std": shown["std"],
                                   "mean_pm_std_2dp": formatted(shown)})
                for configuration in configurations:
                    obj_values = [v for r in paired if (r["configuration"], r["baseline"], r["mode"], r["metric"]) == (configuration, baseline, mode, key) for v in r["values"]]
                    paired_by_object.append({"configuration": configuration, "model": "scm_soft",
                                             "baseline": baseline, "mode": mode, "metric": key,
                                             **statistics(obj_values)})
    write_csv(root / "soft_scm_pairwise_errors.csv", pair_stats)
    write_csv(root / "lcp_comparison_pooled.csv", pair_stats)
    write_csv(root / "lcp_comparison_by_object.csv", paired_by_object)
    contact_stats = [{"engine": e, "mode": mode, **statistics(depths.get((e, mode), []))}
                     for e in ENGINES for mode in MODES]
    write_csv(root / "poststate_all_contact_depth_mm.csv", contact_stats)
    lookup = {(r["baseline"], r["mode"], r["metric"]): r for r in pair_stats}
    columns = [(mode, key, display_units(mode, key)[1], display_units(mode, key)[2])
               for mode in MODES for key in METRICS]
    lines = ["| Baseline | " + " | ".join(f"{mode.capitalize()}: {key} ({unit})" for mode, key, unit, _ in columns) + " |",
             "|---|" + "---:|" * len(columns)]
    table = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Paired differences between SCM and each baseline (mean $\pm$ std; 600 inputs per mode). Separation velocity differences use nano-units to expose roundoff.}",
             r"\label{tab:scm_lcp_accuracy}", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\renewcommand{\arraystretch}{0.95}",
             r"\begin{tabular}{@{}l*{9}{c}@{}}", r"\toprule",
             r"\multirow{2}{*}{Baseline} & \multicolumn{3}{c}{Separation} & \multicolumn{3}{c}{Sliding} & \multicolumn{3}{c}{Sticking} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             " & " + " & ".join(latex_label for _, _, _, latex_label in columns) + r" \\",
             r"\midrule"]
    for baseline, name in (("mjwarp", "MJWarp"), ("lcp", "LCP")):
        values, latex_values = [], []
        for mode, key, _, _ in columns:
            stats = lookup[(baseline, mode, key)]
            shown = {"count": stats["count"], "mean": stats["display_mean"], "std": stats["display_std"]}
            values.append(formatted(shown))
            latex_values.append(formatted(shown, latex=True))
        lines.append("| " + name + " | " + " | ".join(values) + " |")
        table.append(name + " & " + " & ".join(latex_values) + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    table[2] = table[2].replace("600", str(len(configurations) * settings["samples_per_mode"]))
    lines.extend(["", "Every entry compares soft-SCM with the baseline in its row on identical inputs.",
                  "Penetration: samplewise absolute difference of maximum poststate depths. Velocities: norms of difference vectors.",
                  "Penetration uses um. Separation velocities use nm/s and nrad/s to expose roundoff; sliding/sticking use mm/s and rad/s.",
                  "Raw mean/std/max in CSV are in SI units; display_mean/display_std have the explicitly recorded display units."])
    (root / "lcp_comparison.md").write_text("\n".join(lines) + "\n")
    (root / "lcp_comparison_table.tex").write_text("\n".join(table) + "\n")
    report_data = {"settings": settings, "shared_input_checks": checks,
                   "total_shared_inputs": sum(r["shared_states"] for r in checks),
                   "maximum_lcp_residual": max(residuals), "lcp_failure_count": 0,
                   "penetration_definition": "per-query maximum poststate CPU geometric depth, zero when no contacts",
                   "paired_penetration_definition": "abs(d_max_scm_soft - d_max_baseline), computed per shared sample before pooling",
                   "std_definition": "population std, ddof=0; full-precision CSV retained",
                   "paired_velocity_references": ["mjwarp", "lcp"], "statistics_si": summaries,
                   "soft_scm_pairwise": pair_stats, "all_contact_depth_mm": contact_stats}
    (root / "lcp_comparison_report.json").write_text(json.dumps(report_data, indent=2, allow_nan=False) + "\n")
    print("\n".join(lines))
    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    report(parser.parse_args().input_dir)
