#!/usr/bin/env python3
"""Cross-check all identical-input predictions from active-set and Lemke runs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def validate(active_root, lemke_root, output):
    active_manifest = json.loads((active_root / "manifest.json").read_text())
    lemke_manifest = json.loads((lemke_root / "manifest.json").read_text())
    configurations = active_manifest["configurations"]
    if configurations != lemke_manifest["configurations"]:
        raise ValueError("LCP runs have different configurations")
    results = []
    for configuration in configurations:
        a_folder, b_folder = active_root / configuration, lemke_root / configuration
        a_metrics = json.loads((a_folder / "metrics.json").read_text())
        b_metrics = json.loads((b_folder / "metrics.json").read_text())
        if "active-set" not in a_metrics["metadata"]["lcp"]["solver"]:
            raise ValueError("First run must use the independent active-set solver")
        if b_metrics["metadata"]["lcp"]["solver"] != "lemke":
            raise ValueError("Second run must use the dedicated Lemke solver")
        for key in ("mesh_asset", "friction", "mass_kg", "timestep_s"):
            if a_metrics["metadata"][key] != b_metrics["metadata"][key]:
                raise ValueError(f"Changed {key} in {configuration}")
        for key in ("regularization", "tolerance"):
            if a_metrics["metadata"]["lcp"][key] != b_metrics["metadata"]["lcp"][key]:
                raise ValueError(f"Changed LCP {key} in {configuration}")
        with np.load(a_folder / "states.npz", allow_pickle=False) as a, np.load(b_folder / "states.npz", allow_pickle=False) as b:
            ia, ib = a["engine"] == "lcp", b["engine"] == "lcp"
            expected = active_manifest["arguments"]["samples_per_mode"] * len(active_manifest["arguments"]["modes"])
            if ia.sum() != expected or ib.sum() != expected:
                raise ValueError("Missing LCP predictions")
            for key in ("sample_id", "qpos_initial", "qvel_initial", "wrench_world_N_Nm", "reference_mode"):
                np.testing.assert_array_equal(a[key][ia], b[key][ib])
            if np.any(a["status"][ia] != "ok") or np.any(b["status"][ib] != "ok"):
                raise ValueError("Failed LCP prediction")
            difference = a["qvel_predicted"][ia] - b["qvel_predicted"][ib]
            linear = float(np.linalg.norm(difference[:, :3], axis=1).max())
            angular = float(np.linalg.norm(difference[:, 3:], axis=1).max())
        certificates = []
        for folder in (a_folder, b_folder):
            with (folder / "predictions.csv").open() as stream:
                values = [float(r["lcp_residual"]) for r in csv.DictReader(stream) if r["engine"] == "lcp"]
            if len(values) != expected or not np.isfinite(values).all():
                raise ValueError("Missing/nonfinite LCP certificates")
            certificates.append(max(values))
        if max(certificates) > lemke_manifest["arguments"]["lcp_tolerance"]:
            raise ValueError("An LCP certificate exceeds tolerance")
        if linear > 1e-6 or angular > 1e-6:
            raise ValueError(f"LCP solvers disagree in {configuration}: {linear} m/s, {angular} rad/s")
        results.append({"configuration": configuration, "identical_inputs": True, "queries": expected,
                        "active_set_max_residual": certificates[0], "lemke_max_residual": certificates[1],
                        "max_linear_velocity_difference_m_s": linear,
                        "max_angular_velocity_difference_rad_s": angular})
    report = {"dedicated_solver": "unmodified QuantEcon.py v0.11.2 Lemke (vendored MIT modules)",
              "independent_solver": "feasible active-set nonnegative QP",
              "queries": sum(r["queries"] for r in results),
              "maximum_lemke_residual": max(r["lemke_max_residual"] for r in results),
              "maximum_linear_velocity_difference_m_s": max(r["max_linear_velocity_difference_m_s"] for r in results),
              "maximum_angular_velocity_difference_rad_s": max(r["max_angular_velocity_difference_rad_s"] for r in results),
              "velocity_agreement_threshold": 1e-6, "passed": True, "configurations": results}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-set-dir", type=Path, required=True)
    parser.add_argument("--lemke-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.active_set_dir, args.lemke_dir, args.output)
