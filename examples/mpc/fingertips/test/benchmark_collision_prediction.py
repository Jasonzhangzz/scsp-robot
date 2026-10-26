#!/usr/bin/env python3
"""SCM/ComFree 20-ms collision predictions against ten 2-ms MJWarp steps.

Run with the pi0 interpreter; pinned dependencies are activated before MuJoCo
is imported. ``report`` is offline; ``replay`` reruns a saved complete query.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()

import mujoco
import numpy as np

from examples.mpc.franka.scm.benchmark_utils import UPSTREAM_COMMIT
from examples.mpc.fingertips.test.collision_prediction_assets import (
    DEFAULT_CACHE, MESH_OBJECTS, OBJECTS, digest, package_scene, prepare_asset, scene_xml)
from examples.mpc.fingertips.test.collision_prediction_core import (
    ALL_MODES, FORCES, GROUPS, METHODS, METRICS, MODES, SPEEDS, RELATIVE_METRICS, RELATIVE_FLOORS,
    Experiment, Settings, classify_contacts, native_diagnostics)

DEFAULT_OUTPUT = ROOT / "outputs/collision_prediction"
ROW_FIELDS = ("sample_id", "object", "seed", "attempt", "requested_mode", "mode", "method",
              "reference_execution",
              "selected", "geometry", "force_N", "speed_level_m_s", "initial_speed_m_s", "initial_angular_speed_rad_s",
              "ball_initial_speed_m_s", "contact_group", "ncontact_start", "ncontact_floor_start",
              "ncontact_ball_start", "ncontact_max", "ncontact_end", "established", "separated",
              "free_flight", "max_slip_m_s", "status", "failure_stage", "failure_reason") + METRICS + RELATIVE_METRICS + tuple(
                  key + "_denominator" for key in RELATIVE_METRICS)
REPORT_METRICS = METRICS + RELATIVE_METRICS


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def settings_from_args(args):
    return Settings(device=args.device, nconmax=args.nconmax, njmax=args.njmax)


def environment():
    versions = {"mujoco": mujoco.__version__, "numpy": np.__version__}
    for name in ("coacd", "trimesh", "warp-lang", "comfree_warp"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import warp as wp
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    return dict(python=sys.executable, packages=versions, repository_commit=revision,
                upstream_commit=UPSTREAM_COMMIT, gpu=str(wp.get_device("cuda:0")),
                reference_api="comfree_warp.mujoco_warp", scm_api="scm_box CPU float64")


def prepare(args):
    import coacd
    coacd.set_log_level("warn")
    for name in args.objects:
        if name in MESH_OBJECTS:
            path = prepare_asset(name, args.asset_cache, args.coacd_threshold)
            print(f"Prepared {name}: {path}", flush=True)


def save_query(folder, sample, metadata, predictions, reference_q, reference_v):
    arrays = {key: sample[key] for key in ("qpos", "qvel", "wrench")}
    arrays.update(reference_qpos=reference_q, reference_qvel=reference_v,
                  reference_times=np.arange(11)*.002,
                  metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
    for method in METHODS:
        prediction = predictions.get(method, {})
        arrays[method+"_qpos"] = prediction.get("qpos", np.full(14, np.nan))
        arrays[method+"_qvel"] = prediction.get("qvel", np.full(12, np.nan))
    np.savez_compressed(folder / "queries" / (metadata["sample_id"] + ".npz"), **arrays)


def run_seed(experiment, name, seed, args, folder, writer, contacts, samples):
    rng = np.random.default_rng(np.random.SeedSequence([seed, OBJECTS.index(name)]))
    counts, requested_counts, outcomes = Counter(), Counter(), Counter()
    case_selected, case_attempts = Counter(), Counter()
    limit = args.samples_per_mode * args.max_attempt_multiplier
    batch_size = getattr(args, "reference_batch_size", 1)
    serial = 0
    while True:
        candidates = []
        for _ in range(batch_size):
            pending = [mode for mode in MODES if counts[mode] < args.samples_per_mode
                       and requested_counts[mode] < limit]
            if not pending:
                break
            mode = min(pending, key=lambda key: (requested_counts[key], MODES.index(key)))
            attempt = requested_counts[mode]
            requested_counts[mode] += 1
            metadata = dict(sample_id=f"{name}_s{seed}_{serial:06d}", object=name, seed=seed,
                            attempt=attempt, requested_mode=mode, selected=False)
            serial += 1
            sample, error = None, None
            try:
                with native_diagnostics() as warnings:
                    sample = experiment.sample(name, mode, rng, attempt)
                metadata.update({key: value for key, value in sample.items()
                                 if key not in ("qpos", "qvel", "wrench")})
                metadata["preparation_warnings"] = warnings
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            candidates.append(dict(sample=sample, metadata=metadata, sampling_error=error))
        if not candidates:
            break
        valid = [candidate for candidate in candidates if candidate["sampling_error"] is None]
        if batch_size > 1:
            reference_results = experiment.reference_batch([
                dict(candidate["sample"], _sample_id=candidate["metadata"]["sample_id"]) for candidate in valid])
        else:
            reference_results = []
            for candidate in valid:
                reference, error = None, None
                try:
                    with native_diagnostics() as warnings:
                        reference = experiment.reference(candidate["sample"])
                    if warnings:
                        raise RuntimeError("MJWarp native collision warning: " + "; ".join(warnings))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                reference_results.append(dict(reference=reference, error=error, execution="serial", fallback_reason=""))
        for candidate, reference in zip(valid, reference_results):
            candidate["reference_result"] = reference
        for candidate in candidates:
            sample, metadata = candidate["sample"], candidate["metadata"]
            snapshots = []
            reference_q, reference_v = np.empty((0, 14)), np.empty((0, 12))
            stage = "sampling"
            try:
                if candidate["sampling_error"] is not None:
                    raise RuntimeError(candidate["sampling_error"])
                stage = "reference"
                result = candidate["reference_result"]
                metadata.update(reference_execution=result["execution"], reference_fallback_reason=result["fallback_reason"])
                metadata["reference_batch_sample_ids"] = result.get("batch_sample_ids", [])
                if result["reference"] is not None:
                    reference, snapshots, reference_q, reference_v = result["reference"]
                if result["error"] is not None:
                    raise RuntimeError(result["error"])
                label = classify_contacts(snapshots, experiment.settings.force_threshold,
                                          experiment.settings.slip_threshold)
                metadata.update(label)
                actual = label["mode"]
                outcomes[actual] += 1
                selected = actual == "mixed_transition" or counts[actual] < args.samples_per_mode
                metadata["selected"] = selected
                case = (actual, label["contact_group"], metadata["geometry"].removesuffix("_airborne"))
                case_attempts[case] += 1
                case_selected[case] += int(selected)
                if actual in MODES and selected:
                    counts[actual] += 1
                predictions = experiment.predict(sample, reference)
            except Exception as exc:
                failure_status = "sampling_failure" if stage == "sampling" else "invalid_reference"
                metadata.update(mode=failure_status, contact_group="unknown", failure_stage=stage,
                                failure_reason=f"{type(exc).__name__}: {exc}")
                outcomes[failure_status] += 1
                predictions = {method: dict(status=failure_status, failure_reason=metadata["failure_reason"],
                                           **{key: None for key in METRICS}) for method in METHODS}
            if sample is not None:
                save_query(folder, sample, metadata, predictions, reference_q, reference_v)
            samples.write(json.dumps(metadata, allow_nan=False) + "\n")
            contacts.write(json.dumps(dict(sample_id=metadata["sample_id"], snapshots=snapshots), allow_nan=False) + "\n")
            for method, prediction in predictions.items():
                row = {**metadata, "method": method,
                       **{key: value for key, value in prediction.items() if key not in ("qpos", "qvel")}}
                writer.writerow({key: row.get(key) for key in ROW_FIELDS})
            # A machine-readable .log contains actual errors, not just progress.
            with (folder.parent / "results.log").open("a") as log:
                for method, prediction in predictions.items():
                    log.write(json.dumps(dict(kind="prediction", sample_id=metadata["sample_id"], object=name,
                        seed=seed, mode=metadata["mode"], contact_group=metadata.get("contact_group"),
                        geometry=metadata.get("geometry"), selected=metadata["selected"], method=method,
                        status=prediction["status"], failure_reason=prediction.get("failure_reason", ""),
                        absolute_errors={key: prediction.get(key) for key in METRICS},
                        relative_errors={key: prediction.get(key) for key in RELATIVE_METRICS},
                        relative_denominators={key: prediction.get(key+"_denominator") for key in RELATIVE_METRICS}),
                        allow_nan=False) + "\n")
        samples.flush()
        json_write(folder / f"case_counts_seed_{seed}.json", dict(seed=seed,
            measured_mode_selected=dict(counts), target_per_mode=args.samples_per_mode,
            cases=[dict(mode=key[0], contact_group=key[1], geometry=key[2],
                        n_selected=case_selected[key], n_valid_all_attempts=case_attempts[key])
                   for key in sorted(case_attempts)]))
        if batch_size > 1 or serial % 25 == 0:
            print(f"{name} seed={seed} attempts={serial} accepted={dict(counts)} outcomes={dict(outcomes)}", flush=True)
    return dict(seed=seed, accepted={mode: counts[mode] for mode in MODES},
                shortfall={mode: max(0, args.samples_per_mode-counts[mode]) for mode in MODES},
                requested_attempts=dict(requested_counts), measured_outcomes=dict(outcomes))


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    settings = settings_from_args(args)
    manifest = dict(schema_version=1, status="running", settings=settings.to_dict(),
                    objects=args.objects, seeds=args.seeds, samples_per_mode_per_seed=args.samples_per_mode,
                    max_attempt_multiplier=args.max_attempt_multiplier, force_levels_N=FORCES,
                    reference_batch_size=getattr(args, "reference_batch_size", 1),
                    object_speed_levels_m_s=SPEEDS, coacd_threshold_m=args.coacd_threshold,
                    reference="MJWarp .002 s x 10; not physical measurements", contact_count_time="start",
                    contact_count_definition="native active geometric point contacts involving object; gap<=0, efc_address>=0; no ball-floor",
                    mode_definition="loaded body-pair appearance/disappearance; incoming solve-pose slip; mixed events retained",
                    contact_force_decoding="all per-edge efc_address indices, world-local force array; never assume contiguous rows",
                    native_warning_policy="batch warnings trigger smaller-batch and serial retries; individual reference warnings invalidate reference; prediction warnings fail that method; preparation warnings recorded separately",
                    selection="first quota per measured mode and seed; mixed/oversubscribed samples retained",
                    pose_error_definition="sqrt((position_error_m / body_AABB_diagonal_m)^2 + orientation_error_rad^2)",
                    relative_error_definition="pose relative to reference 20ms displacement/rotation; velocities relative to reference terminal magnitudes; below floor is null/NA",
                    relative_denominator_floors=RELATIVE_FLOORS,
                    implementation_sha256={path.name: digest(path) for path in
                                           (Path(__file__), Path(__file__).with_name("collision_prediction_core.py"),
                                            Path(__file__).with_name("collision_prediction_assets.py"),
                                            Path(__file__).with_name("collision_prediction_batch.py"),
                                            Path(__file__).with_name("collision_prediction_smoke.py"))},
                    **environment(), results={})
    json_write(output / "manifest.json", manifest)
    implementation = output / "implementation"
    implementation.mkdir()
    for filename in manifest["implementation_sha256"]:
        shutil.copy2(Path(__file__).with_name(filename), implementation / filename)
    try:
        for name in args.objects:
            try:
                asset = prepare_asset(name, args.asset_cache, args.coacd_threshold) if name in MESH_OBJECTS else None
                xml, info = scene_xml(name, asset, friction=settings.friction,
                                      primitive_mass=settings.primitive_mass, ball_mass=settings.ball_mass,
                                      ball_radius=settings.ball_radius)
                folder = output / name
                model_path = package_scene(xml, folder)
                (folder / "queries").mkdir()
                info["model_sha256"] = digest(model_path)
                info["packaged_asset_sha256"] = {str(path.relative_to(folder)): digest(path)
                                                  for path in sorted((folder / "assets").glob("*"))}
                json_write(folder / "object.json", info)
                experiment = Experiment(model_path, info, settings)
                results = []
                with (folder / "predictions.csv").open("w", newline="") as stream, \
                        gzip.open(folder / "reference_contacts.jsonl.gz", "wt") as contacts, \
                        (folder / "samples.jsonl").open("w") as samples:
                    writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
                    writer.writeheader()
                    for seed in args.seeds:
                        results.append(run_seed(experiment, name, seed, args, folder, writer, contacts, samples))
                        stream.flush()
                        manifest["results"][name] = dict(status="running", seeds=results)
                        json_write(output / "manifest.json", manifest)
                        print(f"COMPLETED {name} seed={seed}: {results[-1]}", flush=True)
                manifest["results"][name] = dict(status="complete", seeds=results)
                if hasattr(experiment, "parallel_reference"):
                    json_write(folder / "reference_batches.json", experiment.parallel_reference.batches)
            except Exception as exc:
                manifest["results"][name] = dict(status="failed", error=f"{type(exc).__name__}: {exc}",
                                                 traceback=traceback.format_exc())
                print(f"FAILED {name}: {exc}", file=sys.stderr, flush=True)
            json_write(output / "manifest.json", manifest)
        failed = any(item["status"] == "failed" for item in manifest["results"].values())
        shortage = any(any(seed["shortfall"].values()) for result in manifest["results"].values()
                       for seed in result.get("seeds", []))
        manifest["status"] = "failed" if failed else "complete_with_shortfalls" if shortage else "complete"
    finally:
        if manifest["status"] == "running":
            manifest["status"] = "interrupted"
        json_write(output / "manifest.json", manifest)
    report_directory(output, plots=not args.no_plots)
    print(f"Results: {output} ({manifest['status']})", flush=True)
    return int(manifest["status"] == "failed")


def statistics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("mean", "std", "rmse", "median", "p95")}
    return dict(mean=float(values.mean()), std=float(values.std()),
                rmse=float(np.sqrt(np.mean(values**2))), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)))


def summarize(rows, objects):
    summaries = []
    # Explicit empty cells distinguish absent geometry/mode combinations from
    # measured zero error. Only selected paired queries enter the main table.
    for name in [*objects, "all"]:
        for mode in ALL_MODES:
            for group in [*GROUPS, "all"]:
                subset = [row for row in rows if (row["object"] == name or name == "all")
                          and row["mode"] == mode and (row["contact_group"] == group or group == "all")
                          and (row["selected"] == "True" or mode in ("invalid_reference", "sampling_failure"))]
                for method in METHODS:
                    selected = [row for row in subset if row["method"] == method]
                    valid = [row for row in selected if row["status"] == "ok"]
                    for metric in REPORT_METRICS:
                        defined = [float(row[metric]) for row in valid if row.get(metric) not in (None, "")]
                        summaries.append(dict(object=name, mode=mode, contact_group=group,
                                              method=method, metric=metric, n_total=len(selected),
                                              n_valid=len(valid), n_failed=len(selected)-len(valid),
                                              n_defined=len(defined), n_undefined=len(valid)-len(defined),
                                              **statistics(defined)))
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strata(rows):
    result = []
    for dimension in ("force_N", "speed_level_m_s", "seed"):
        for value in sorted({row[dimension] for row in rows if row[dimension] != ""}):
            for method in METHODS:
                subset = [row for row in rows if row[dimension] == value and row["method"] == method
                          and row["selected"] == "True"]
                valid = [row for row in subset if row["status"] == "ok"]
                for metric in REPORT_METRICS:
                    defined = [float(row[metric]) for row in valid if row.get(metric) not in (None, "")]
                    result.append(dict(dimension=dimension, value=value, method=method, metric=metric,
                                       n_total=len(subset), n_valid=len(valid), n_failed=len(subset)-len(valid),
                                       n_defined=len(defined), n_undefined=len(valid)-len(defined), **statistics(defined)))
    return result


def case_counts(rows, objects, mode_target):
    """Count queries once; marginal and joint tables include empty cells.

    Only object x measured mode has a controlled quota. Other cells describe
    observed coverage, not a claim that their Cartesian product is feasible.
    """
    references = [row for row in rows if row["method"] == "mjwarp"]
    predictions = {(row["sample_id"], row["method"]): row for row in rows}
    result = []
    for name in objects:
        obj = [row for row in references if row["object"] == name]
        geometries = sorted({row.get("geometry", "").removesuffix("_airborne") for row in obj
                             if row.get("geometry")})
        for mode in (*MODES, "mixed_transition"):
            for grouping, groups, shapes in (
                    ("object_mode", ["all"], ["all"]),
                    ("object_mode_count", GROUPS, ["all"]),
                    ("object_mode_geometry", ["all"], geometries),
                    ("object_mode_count_geometry", GROUPS, geometries)):
                for group in groups:
                    for geometry in shapes:
                        subset = [row for row in obj if row["mode"] == mode
                                  and (group == "all" or row["contact_group"] == group)
                                  and (geometry == "all" or row["geometry"].removesuffix("_airborne") == geometry)]
                        selected = [row for row in subset if row["selected"] == "True"]
                        controlled = grouping == "object_mode" and mode in MODES
                        result.append(dict(grouping=grouping, object=name, mode=mode, contact_group=group,
                            geometry=geometry, n_selected=len(selected), n_valid_all_attempts=len(subset),
                            n_failed_scm=sum(predictions[(row["sample_id"], "scm")]["status"] != "ok" for row in selected),
                            n_failed_comfree=sum(predictions[(row["sample_id"], "comfree")]["status"] != "ok" for row in selected),
                            n_invalid_reference=0,
                            target=mode_target if controlled else None,
                            shortfall=max(0, mode_target-len(selected)) if controlled else None,
                            below_100=len(selected) < 100, below_200=len(selected) < 200,
                            status=("no_samples_unverified_feasibility" if not selected else
                                    "quota_shortfall" if controlled and len(selected) < mode_target else
                                    "observed")))
        invalid = sum(row["status"] != "ok" for row in obj)
        result.append(dict(grouping="reference_failures", object=name, mode="invalid_reference",
                           contact_group="unknown", geometry="all", n_selected=0, n_valid_all_attempts=0,
                           n_failed_scm=0, n_failed_comfree=0, n_invalid_reference=invalid, target=None, shortfall=None,
                           below_100=True, below_200=True, status=f"{invalid} invalid references; no actual-mode assignment"))
    return result


def plot_report(rows, summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, METRICS[:3]):
        groups = [[float(row[metric]) for row in rows if row["method"] == method
                   and row["status"] == "ok" and row["selected"] == "True"]
                  for method in ("scm", "comfree")]
        ax.boxplot([values if values else [np.nan] for values in groups], tick_labels=["SCM", "ComFree"],
                   showfliers=False)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "error_distributions.png", dpi=160)
    fig.savefig(output / "error_distributions.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = list(MODES) + ["mixed_transition"]
    maximum = max((row["mean"] for row in summaries if row["object"] == "all"
                   and row["method"] in ("scm", "comfree") and row["mode"] in labels
                   and row["contact_group"] in GROUPS and row["metric"] == "pose_error"
                   and row["mean"] is not None), default=0.)
    for ax, method in zip(axes, ("scm", "comfree")):
        matrix = np.full((len(labels), len(GROUPS)), np.nan)
        for i, mode in enumerate(labels):
            for j, group in enumerate(GROUPS):
                item = next(row for row in summaries if row["object"] == "all" and row["method"] == method
                            and row["mode"] == mode and row["contact_group"] == group and row["metric"] == "pose_error")
                if item["mean"] is not None:
                    matrix[i, j] = item["mean"]
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=max(maximum, 1e-12))
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.3g}" if np.isfinite(value) else "no samples", ha="center", va="center",
                    fontsize=8, color="white" if np.isfinite(value) and value < .4*maximum else "black")
        ax.set_xticks(range(len(GROUPS)), GROUPS)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_title(method)
        ax.set_xlabel("Initial object contact points")
        fig.colorbar(im, ax=ax, label="Mean normalized pose error")
    fig.tight_layout()
    fig.savefig(output / "contact_count_heatmap.png", dpi=160)
    fig.savefig(output / "contact_count_heatmap.pdf")
    plt.close(fig)


def report_directory(output, plots=True):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    rows = []
    for name in manifest["objects"]:
        path = output / name / "predictions.csv"
        if path.exists():
            with path.open() as stream:
                rows.extend(csv.DictReader(stream))
    summaries = summarize(rows, manifest["objects"])
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "stratified_summary.csv", strata(rows))
    counts = case_counts(rows, manifest["objects"], len(manifest["seeds"])*manifest["samples_per_mode_per_seed"])
    write_csv(output / "case_counts.csv", counts)
    count_lines = ["# Measured sample counts", "",
                   "Only object × mode has a sampling quota. Empty joint cells have unverified feasibility.", "",
                   "| Grouping | Object | Mode | Initial contacts | Geometry | Selected | All valid attempts | Target |",
                   "|---|---|---|---|---|---|---|---|"]
    for item in counts:
        count_lines.append(f"| {item['grouping']} | {item['object']} | {item['mode']} | {item['contact_group']} | "
                           f"{item['geometry']} | {item['n_selected']} | {item['n_valid_all_attempts']} | {item['target'] or '—'} |")
    (output / "case_counts.md").write_text("\n".join(count_lines) + "\n")
    # Reporting can be rerun offline; keep summaries separate from immutable
    # per-query errors so repeated reports do not duplicate query records.
    with (output / "summary.log").open("w") as log:
        for item in counts:
            log.write(json.dumps(dict(kind="case_count", **item), allow_nan=False) + "\n")
        for item in summaries:
            log.write(json.dumps(dict(kind="error_summary", **item), allow_nan=False) + "\n")
    lines = ["# 20 ms collision prediction error", "", f"Run status: **{manifest['status']}**.", "",
             "MJWarp (.002 s x 10) is the numerical reference. SCM/ComFree take one .02 s step.",
             "Contact count is measured at the query start. Empty cells mean no samples.",
             "Errors exclude failed predictions; n_failed remains in the total denominator.", "",
             "| Object | Measured mode | Initial contacts | Method | Position mm, mean ± std | Rotation deg, mean ± std | Pose error, mean ± std | Valid / total |",
             "|---|---|---|---|---|---|---|---|"]
    for item in summaries:
        if item["metric"] != "pose_error" or item["n_total"] == 0 or item["contact_group"] == "all":
            continue
        def cell(metric):
            match = next(row for row in summaries if all(row[key] == item[key] for key in
                         ("object", "mode", "contact_group", "method")) and row["metric"] == metric)
            return "—" if match["mean"] is None else f"{match['mean']:.5g} ± {match['std']:.3g}"
        lines.append(f"| {item['object']} | {item['mode']} | {item['contact_group']} | {item['method']} | "
                     f"{cell('position_error_mm')} | {cell('orientation_error_deg')} | {cell('pose_error')} | "
                     f"{item['n_valid']} / {item['n_total']} |")
    lines.extend(["", "## Coverage", "", "```json", json.dumps(manifest["results"], indent=2), "```", ""])
    (output / "summary.md").write_text("\n".join(lines))
    json_write(output / "coverage.json", manifest["results"])
    if plots and rows:
        plot_report(rows, summaries, output)
    manifest["report_implementation_sha256"] = digest(__file__)
    implementation = output / "implementation"
    implementation.mkdir(exist_ok=True)
    shutil.copy2(__file__, implementation / "report_benchmark_collision_prediction.py")
    json_write(output / "manifest.json", manifest)
    return summaries


def replay(args):
    path = args.query.resolve()
    folder = path.parent.parent
    root = folder.parent
    manifest = json.loads((root / "manifest.json").read_text())
    info = json.loads((folder / "object.json").read_text())
    if digest(folder / "model.xml") != info["model_sha256"]:
        raise ValueError("Saved model.xml hash mismatch")
    for name, checksum in info["packaged_asset_sha256"].items():
        if digest(folder / name) != checksum:
            raise ValueError(f"Saved asset hash mismatch: {name}")
    settings = Settings(**manifest["settings"])
    experiment = Experiment(folder / "model.xml", info, settings)
    with np.load(path, allow_pickle=False) as archive:
        sample = {key: archive[key].copy() for key in ("qpos", "qvel", "wrench")}
        metadata = json.loads(str(archive["metadata_json"]))
        peers = metadata.get("reference_batch_sample_ids", [])
        replay_execution = "serial"
        if metadata.get("reference_execution") == "gpu_batch" and len(peers) > 1:
            batch = []
            for peer in peers:
                with np.load(folder / "queries" / (peer + ".npz"), allow_pickle=False) as saved:
                    batch.append({key: saved[key].copy() for key in ("qpos", "qvel", "wrench")})
            result = experiment.reference_batch(batch)[peers.index(metadata["sample_id"])]
            warnings = [result["error"]] if result["error"] else []
            replay_execution = result["execution"]
            if result["reference"] is None:
                raise RuntimeError(f"Replay reference failed: {result['error']}")
            reference, snapshots, qpos, qvel = result["reference"]
        else:
            with native_diagnostics() as warnings:
                reference, snapshots, qpos, qvel = experiment.reference(sample)
        predictions = (experiment.predict(sample, reference) if not warnings else
                       {method: dict(status="invalid_reference", **{key: None for key in METRICS})
                        for method in METHODS})
        comparison = {}
        for method, result in predictions.items():
            previous = archive[method + "_qpos"]
            comparison[method] = dict(status=result["status"],
                                      saved_qpos_max_abs_difference=float(np.max(np.abs(previous-result["qpos"])))
                                      if result["status"] == "ok" and np.isfinite(previous).all() else None,
                                      **{key: result.get(key) for key in REPORT_METRICS})
        report = dict(sample_id=metadata["sample_id"], original_mode=metadata["mode"],
                      original_reference_execution=metadata.get("reference_execution", "serial"),
                      replay_reference_execution=replay_execution, replay_batch_worlds=len(peers) if peers else 1,
                      reference_warnings=warnings,
                      label=classify_contacts(snapshots) if not warnings else {"mode": "invalid_reference"},
                      predictions=comparison,
                      reference_max_abs_difference=float(np.max(np.abs(qpos-archive["reference_qpos"])))
                      if qpos.shape == archive["reference_qpos"].shape else None)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    if args.output:
        json_write(args.output, report)
    if args.viewer:
        import mujoco.viewer
        import time
        model = experiment.models["mjwarp"]
        data = mujoco.MjData(model)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                for q, v in zip(qpos, qvel):
                    if not viewer.is_running():
                        break
                    data.qpos[:], data.qvel[:] = q, v
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(.1)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for command in ("prepare-assets", "run", "smoke-test"):
        p = commands.add_parser(command)
        p.add_argument("--objects", nargs="+", choices=OBJECTS, default=list(OBJECTS))
        p.add_argument("--asset-cache", type=Path, default=DEFAULT_CACHE)
        p.add_argument("--coacd-threshold", type=positive_float, default=.001, help="CoACD tolerance in meters")
        if command in ("run", "smoke-test"):
            smoke = command == "smoke-test"
            p.add_argument("--seeds", nargs="+", type=int, default=[20261026] if smoke else list(range(42, 47)))
            p.add_argument("--samples-per-mode", type=positive_int, default=6 if smoke else 20, help="Measured-mode quota per object and seed")
            p.add_argument("--max-attempt-multiplier", type=positive_int, default=100)
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--reference-batch-size", type=positive_int, default=16 if smoke else 1,
                           help="Independent states of the same object per native MJWarp GPU batch; 1 preserves serial sampling")
            p.add_argument("--nconmax", type=positive_int, default=2048)
            p.add_argument("--njmax", type=positive_int, default=16384)
            p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT.parent / "collision_prediction_smoke" if smoke else DEFAULT_OUTPUT)
            p.add_argument("--headless", action="store_true", help="Explicit batch mode (run is always headless)")
            p.add_argument("--no-plots", action="store_true")
    p = commands.add_parser("report")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--no-plots", action="store_true")
    p = commands.add_parser("smoke-check")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--verify-parallel", action="store_true")
    p.add_argument("--batch-size", type=positive_int, default=16)
    p = commands.add_parser("replay")
    p.add_argument("--query", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--viewer", action="store_true")
    return result


def main():
    args = parser().parse_args()
    if hasattr(args, "objects") and len(set(args.objects)) != len(args.objects):
        raise ValueError("--objects must not contain duplicates")
    if hasattr(args, "seeds") and (min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)):
        raise ValueError("--seeds must contain distinct nonnegative integers")
    if args.command == "prepare-assets":
        prepare(args)
    elif args.command in ("run", "smoke-test"):
        code = run(args)
        if code or args.command == "run":
            return code
        from examples.mpc.fingertips.test.collision_prediction_smoke import write_smoke_report
        report = write_smoke_report(args.output_dir, parallel=True, batch_size=args.reference_batch_size)
        print(f"Smoke test: {report['status']} -> {args.output_dir / 'smoke_test.md'}", flush=True)
        return int(report["status"] != "passed")
    elif args.command == "report":
        report_directory(args.input_dir, not args.no_plots)
    elif args.command == "smoke-check":
        from examples.mpc.fingertips.test.collision_prediction_smoke import write_smoke_report
        report = write_smoke_report(args.input_dir, args.verify_parallel, args.batch_size)
        print(f"Smoke test: {report['status']}", flush=True)
        return int(report["status"] != "passed")
    else:
        replay(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
