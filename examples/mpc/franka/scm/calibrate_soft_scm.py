#!/usr/bin/env python3
"""Fit two global soft-SCM parameters on recorded TRAINING inputs only.

Caches native SCM contact rows, then evaluates a predefined scalar grid without
additional collision solves. Validation must use another seed in the actual GPU
prediction/rollout tools. Visualization is enabled unless --headless is given.
"""

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from runtime_env import activate_dependencies
activate_dependencies()
import mujoco
import numpy as np
from compare_prediction import fresh_data, make_model, backend_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true", help="Disable the default contact-query visualization")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = json.loads((args.input_dir / "manifest.json").read_text())["arguments"]
    options = SimpleNamespace(**settings)
    samples = []
    cache_max_qvel_difference = 0.
    for folder in sorted(args.input_dir.glob("*/")):
        if not (folder / "states.npz").exists():
            continue
        metadata = json.loads((folder / "metrics.json").read_text())["metadata"]
        model = make_model(metadata["shape"], metadata["friction"], metadata["timestep_s"], metadata["mass_kg"])
        backend = backend_for("scm_soft", model, fresh_data(model), options)
        display = fresh_data(model)
        if args.headless:
            context = nullcontext(None)
        else:
            from mujoco import viewer
            context = viewer.launch_passive(model, display)
        with np.load(folder / "states.npz", allow_pickle=False) as states, context as window:
            for i in np.flatnonzero(states["engine"] == "mjwarp"):
                if states["reference_mode"][i] not in ("sliding", "sticking"):
                    continue
                data = fresh_data(model, states["qpos_initial"][i], states["qvel_initial"][i], states["wrench_world_N_Nm"][i])
                M = np.zeros((6, 6))
                mujoco.mj_fullM(model, M, data.qM)
                backend.reset(data)
                backend.step(data)
                if window is not None:
                    if not window.is_running():
                        raise RuntimeError("Calibration window closed before all training inputs were collected")
                    mujoco.mj_copyData(display, model, data)
                    window.sync()
                count = int(backend.wdata.nefc.numpy()[0])
                J = backend.wdata.efc.J.numpy()[0, :count, :6].astype(float)
                gap = backend.wdata.efc.efc_dist.numpy()[0, :count].astype(float)
                free = backend.wdata.qvel_smooth_pred.numpy()[0, :6].astype(float)
                inv = np.linalg.inv(M)
                D = np.einsum("ij,jk,ik->i", J, inv, J) + options.scm_regularization
                x = float(model.opt.timestep) / backend.api.response_time
                divisor = 1 + 2*backend.api.damping_ratio*x + x*x
                a = (2*backend.api.damping_ratio*x+x*x)/divisor
                b = x*x/divisor
                projected = free
                if backend.api.motion == 'increment':
                    projected = free - np.asarray(states['qvel_initial'][i],dtype=np.float32).astype(float)
                predicted = free + inv @ J.T @ np.maximum(-(a*(J @ projected) + b*gap/model.opt.timestep) / D, 0)
                # Check the cache against the GPU before using it for tuning.
                # GPU reaction accumulation and mass solves use float32;
                # this grid uses float64. Record the discrepancy explicitly.
                np.testing.assert_allclose(predicted, data.qvel, atol=3e-6, rtol=2e-5)
                cache_max_qvel_difference = max(cache_max_qvel_difference, float(np.max(np.abs(predicted - data.qvel))))
                samples.append(dict(shape=metadata["shape"], mode=str(states["reference_mode"][i]),
                                    sample_id=int(states["sample_id"][i]), h=float(model.opt.timestep),
                                    J=J, gap=gap, free=free, M=M, inverse=inv, D=D,
                                    reference=states["qvel_predicted"][i]))
        print(f"Cached {metadata['shape']}: {len(samples)} training inputs", flush=True)
    max_rows = max(len(s["D"]) for s in samples)
    dense = dict(J=np.zeros((len(samples), max_rows, 6)), gap=np.zeros((len(samples), max_rows)),
                 D=np.ones((len(samples), max_rows)), nefc=np.array([len(s["D"]) for s in samples]))
    for i, sample in enumerate(samples):
        n = len(sample["D"])
        for key in ("J", "gap", "D"):
            dense[key][i, :n] = sample[key]
    for key in ("shape", "mode", "sample_id", "h", "free", "M", "inverse", "reference"):
        dense[key] = np.asarray([s[key] for s in samples])
    np.savez_compressed(args.output_dir / "training_contact_rows.npz", **dense)
    grid = []
    velocity = np.einsum("sij,sj->si", dense["J"], dense["free"])
    # The grid and objective are fixed before observing held-out predictions.
    for tau in (.002, .004, .008, .016, .032, .05, .08, .128, .2, .32, .5, 1.):
        for zeta in (.1, .25, .5, 1., 2., 4.):
            x = dense["h"][:, None] / tau
            denominator = 1 + 2 * zeta * x + x * x
            a, b = (2 * zeta * x + x * x) / denominator, x * x / denominator
            p = np.maximum(-(a * velocity + b * dense["gap"] / dense["h"][:, None]) / dense["D"], 0)
            reaction = np.einsum("sij,si->sj", dense["J"], p)
            predicted = dense["free"] + np.einsum("sij,sj->si", dense["inverse"], reaction)
            diff = predicted - dense["reference"]
            weighted = np.sqrt(np.maximum(np.einsum("si,sij,sj->s", diff, dense["M"], diff), 0))
            row = dict(response_time_s=tau, damping_ratio=zeta,
                       objective_mean_weighted_velocity_error=float(weighted.mean()))
            for mode in ("sliding", "sticking"):
                selected = dense["mode"] == mode
                for metric, values in (("linear", np.linalg.norm(diff[:, :3], axis=1)),
                                       ("angular", np.linalg.norm(diff[:, 3:], axis=1)), ("weighted", weighted)):
                    row[f"{mode}_{metric}_mean"] = float(values[selected].mean())
                    row[f"{mode}_{metric}_std"] = float(values[selected].std())
            grid.append(row)
    grid.sort(key=lambda r: r["objective_mean_weighted_velocity_error"])
    with (args.output_dir / "training_grid.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    selected = {"training_source": str(args.input_dir.resolve()), "training_seed": settings["seed"],
                "training_count": len(samples), "objective": "mean inertia-weighted velocity error on equally sized sliding/sticking training sets",
                "selected": grid[0], "held_out_predictions_used": False,
                "cache_float64_vs_gpu_float32_max_qvel_difference": cache_max_qvel_difference,
                "warning": "parameters fitted to a numerical reference; accuracy on new states/parameters and long-rollout stability require validation"}
    (args.output_dir / "selected_parameters.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == "__main__":
    main()
