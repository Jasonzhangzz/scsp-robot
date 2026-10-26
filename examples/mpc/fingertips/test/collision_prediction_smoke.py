"""Measured coverage audit and serial/native multi-world GPU smoke checks."""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from examples.mpc.fingertips.test.collision_prediction_assets import MESH_OBJECTS
from examples.mpc.fingertips.test.collision_prediction_core import (
    GROUPS, MODES, Experiment, Settings, classify_contacts, native_diagnostics, pose_errors)


def floor_geometry(snapshot, tolerance=1e-4):
    """Affine dimension of actual floor contact positions, in meters.

    Geom-pair duplicates do not create extra dimensions. This measurement is
    independent of the sampler's point/edge/face name and the raw point count.
    """
    points = np.asarray(snapshot["position"], float).reshape(-1, 3)
    points = points[np.asarray(snapshot["pair"], int) == 0]
    if not len(points):
        return "none"
    singular = np.linalg.svd(points-points.mean(0), compute_uv=False)
    rank = int(np.count_nonzero(singular > tolerance))
    return ("point", "line", "face", "volume")[rank]


def audit_coverage(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    report = dict(status="passed", contact_geometry_tolerance_m=1e-4, objects={}, missing=[])
    mode_totals, count_totals, geometry_totals, all_modes = Counter(), Counter(), Counter(), Counter()
    for name in manifest["objects"]:
        folder = directory / name
        info = json.loads((folder / "object.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
        obj = model.body("object").id
        rows = {row["sample_id"]: row for row in csv.DictReader((folder / "predictions.csv").open())
                if row["method"] == "mjwarp" and row["status"] == "ok"}
        primary = [row for row in rows.values() if row["selected"] == "True" and row["mode"] in MODES]
        all_modes.update(row["mode"] for row in rows.values())
        modes = Counter(row["mode"] for row in primary)
        counts = Counter(row["contact_group"] for row in primary)
        mode_totals.update(modes)
        count_totals.update(counts)
        example_ids, observed = {}, Counter()
        max_parts = 0
        mode_examples, count_examples = {}, {}
        for row in primary:
            mode_examples.setdefault(row["mode"], row["sample_id"])
            count_examples.setdefault(row["contact_group"], row["sample_id"])
        with gzip.open(folder / "reference_contacts.jsonl.gz", "rt") as stream:
            for line in stream:
                record = json.loads(line)
                row = rows.get(record["sample_id"])
                if row is None or not record["snapshots"]:
                    continue
                snapshots = record["snapshots"]
                measured = classify_contacts(snapshots, manifest["settings"]["force_threshold"],
                                             manifest["settings"]["slip_threshold"])
                if measured["mode"] != row["mode"] or measured["contact_group"] != row["contact_group"]:
                    raise AssertionError(f"Saved/native label mismatch: {record['sample_id']}")
                geometry = floor_geometry(snapshots[0])
                observed[geometry] += 1
                nominal = row["geometry"].removesuffix("_airborne")
                key = nominal + ":" + geometry
                if geometry != "none":
                    example_ids.setdefault(key, row["sample_id"])
                    geometry_totals[geometry] += 1
                for snapshot in snapshots:
                    geom = np.asarray(snapshot["geom"], int).reshape(-1, 2)
                    if len(geom):
                        object_geoms = geom[model.geom_bodyid[geom] == obj]
                        max_parts = max(max_parts, len(np.unique(object_geoms)))
        expected = {"sphere": ("point:point",), "cube": ("vertex:point", "edge:line", "face:face"),
                    "cylinder": ("line:line", "face:face")}.get(name, ())
        for mode in MODES:
            if not modes[mode]:
                report["missing"].append(f"{name}: mode={mode}")
        for kind in expected:
            if kind not in example_ids:
                report["missing"].append(f"{name}: measured geometry={kind}")
        if name in MESH_OBJECTS and (info.get("part_count", 0) <= 1 or max_parts <= 1):
            report["missing"].append(f"{name}: multi-component contact geometry")
        report["objects"][name] = dict(primary_modes=dict(modes), primary_contact_counts=dict(counts),
                                       measured_floor_geometry=dict(observed), geometry_examples=example_ids,
                                       mode_examples=mode_examples, count_examples=count_examples,
                                       convex_parts=info.get("part_count", 0), max_contacting_object_geoms=max_parts,
                                       mode_contact_count_cells=[dict(mode=mode, contacts=group,
                                           n=sum(row["mode"] == mode and row["contact_group"] == group for row in primary))
                                           for mode in MODES for group in GROUPS])
    for group in GROUPS:
        if not count_totals[group]:
            report["missing"].append(f"overall: contacts={group}")
    report.update(primary_modes=dict(mode_totals), primary_contact_counts=dict(count_totals),
                  measured_floor_geometry=dict(geometry_totals), all_reference_modes=dict(all_modes))
    if not all_modes["mixed_transition"]:
        report["missing"].append("overall: mixed_transition")
    if report["missing"]:
        report["status"] = "failed"
    return report


def physics_timing(engine, samples, repeats=3):
    """Identical 10-step + endpoint-forward work; exclude allocation and I/O.

    This is a physics-API throughput comparison, NOT the full benchmark's
    speedup: CPU contact diagnostics and SCM predictions remain separate.
    """
    wp, backend = engine.wp, engine.backend
    timings = {"serial": [], "batch": []}
    with wp.ScopedDevice(backend.device):
        for repeat in range(repeats + 1):
            for variant in ("serial", "batch"):
                groups = [[sample] for sample in samples] if variant == "serial" else [samples]
                elapsed = 0.
                with native_diagnostics() as warnings:
                    for group in groups:
                        data = engine.new_data(group)
                        wp.synchronize_device(backend.device)
                        started = time.perf_counter()
                        for _ in range(engine.settings.reference_steps):
                            engine.api.step(backend.wmodel, data)
                        engine.api.forward(backend.wmodel, data)
                        wp.synchronize_device(backend.device)
                        elapsed += time.perf_counter() - started
                        engine.native_contacts(data)  # validate bounds outside timing
                if warnings:
                    return dict(status="native_warning", warnings=warnings)
                if repeat:  # first repetition warms all branches in both paths
                    timings[variant].append(elapsed)
    serial, batch = [float(np.median(timings[key])) for key in ("serial", "batch")]
    return dict(status="ok", samples=len(samples), repeats=repeats, serial_seconds=serial,
                batch_seconds=batch, physics_speedup=serial/batch,
                batch_queries_per_second=len(samples)/batch)


def verify_parallel(directory, coverage, batch_size=16):
    if batch_size < 2:
        raise ValueError("Parallel verification requires --batch-size >= 2")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    settings = Settings(**manifest["settings"])
    # Hardware float32 reductions/contact ordering are not bitwise equivalent.
    # Require exact discrete labels/counts, and a small stated numerical budget
    # for both rigid bodies over ALL eleven frames. Keep tighter differences
    # visible as well; do not conceal the raw discrepancies by rounding.
    result = dict(status="passed", batch_size=batch_size, objects={}, tolerances=dict(
        position_m=2e-5, rotation_rad=1e-3, linear_velocity_m_s=2e-3, angular_velocity_rad_s=5e-2),
        strict_tolerances=dict(position_m=5e-6, rotation_rad=1e-4,
                               linear_velocity_m_s=5e-4, angular_velocity_rad_s=5e-3))
    for name in manifest["objects"]:
        folder = directory / name
        info = json.loads((folder / "object.json").read_text())
        details = coverage["objects"][name]
        ids = list(dict.fromkeys([*details["mode_examples"].values(), *details["count_examples"].values(),
                                 *details["geometry_examples"].values()]))
        for row in csv.DictReader((folder / "predictions.csv").open()):
            if len(ids) >= batch_size:
                break
            if row["method"] == "mjwarp" and row["status"] == "ok" and row["selected"] == "True" and row["sample_id"] not in ids:
                ids.append(row["sample_id"])
        ids = ids[:batch_size]
        samples = []
        for sample_id in ids:
            with np.load(folder / "queries" / (sample_id + ".npz"), allow_pickle=False) as archive:
                samples.append({key: archive[key].copy() for key in ("qpos", "qvel", "wrench")})
        experiment = Experiment(folder / "model.xml", info, settings)
        batched = experiment.reference_batch(samples)
        maxima = dict(position_m=0., rotation_rad=0., linear_velocity_m_s=0., angular_velocity_rad_s=0.)
        mismatches, reference_failures = [], []
        compared, native_compared = 0, 0
        native_worlds = sum(item["execution"] == "gpu_batch" for item in batched)
        for sample_id, sample, parallel in zip(ids, samples, batched):
            try:
                with native_diagnostics() as warnings:
                    serial = experiment.reference(sample)
            except Exception as exc:
                reference_failures.append(dict(sample_id=sample_id, reason=f"{type(exc).__name__}: {exc}"))
                continue
            if warnings or parallel["error"]:
                reference_failures.append(dict(sample_id=sample_id, reason="reference warning", serial_warnings=warnings,
                                              batch_error=parallel["error"]))
                continue
            compared += 1
            native_compared += parallel["execution"] == "gpu_batch"
            sq, sv = serial[2:]
            bq, bv = parallel["reference"][2:]
            for qadr in (0, 7):
                maxima["position_m"] = max(maxima["position_m"], float(np.max(np.linalg.norm(
                    sq[:, qadr:qadr+3]-bq[:, qadr:qadr+3], axis=1))))
                for qs, qb in zip(sq[:, qadr:qadr+7], bq[:, qadr:qadr+7]):
                    error = pose_errors(qs, qb, 1.)
                    maxima["rotation_rad"] = max(maxima["rotation_rad"], float(np.radians(error["orientation_error_deg"])))
            maxima["linear_velocity_m_s"] = max(maxima["linear_velocity_m_s"], float(np.max(np.abs(
                sv[:, [0, 1, 2, 6, 7, 8]]-bv[:, [0, 1, 2, 6, 7, 8]]))))
            maxima["angular_velocity_rad_s"] = max(maxima["angular_velocity_rad_s"], float(np.max(np.abs(
                sv[:, [3, 4, 5, 9, 10, 11]]-bv[:, [3, 4, 5, 9, 10, 11]]))))
            slabel, blabel = [classify_contacts(item[1], settings.force_threshold, settings.slip_threshold)
                              for item in (serial, parallel["reference"])]
            if slabel["mode"] != blabel["mode"] or slabel["ncontact_start"] != blabel["ncontact_start"]:
                mismatches.append(dict(sample_id=sample_id, reason="label mismatch", serial=slabel, batch=blabel))
        passed = (native_compared > 1 and compared >= max(2, len(samples)//2) and not mismatches
                  and all(maxima[key] <= value for key, value in result["tolerances"].items()))
        timing = physics_timing(experiment.parallel_reference, samples) if native_worlds == len(samples) else dict(status="serial_fallback")
        result["objects"][name] = dict(status="passed" if passed else "failed", tested_samples=len(samples),
                                       native_gpu_worlds=native_worlds, max_differences=maxima, mismatches=mismatches,
                                       compared_samples=compared, native_gpu_compared_samples=native_compared,
                                       reference_failures=reference_failures,
                                       meets_strict_tolerances=all(maxima[key] <= value for key, value in result["strict_tolerances"].items()),
                                       batch_diagnostics=experiment.parallel_reference.batches, physics_timing=timing)
        if not passed:
            result["status"] = "failed"
        print(f"PARALLEL {name}: {result['objects'][name]}", flush=True)
    return result


def write_smoke_report(directory, parallel=False, batch_size=16):
    directory = Path(directory)
    coverage = audit_coverage(directory)
    report = dict(coverage=coverage)
    if parallel:
        report["parallel"] = verify_parallel(directory, coverage, batch_size)
    report["status"] = "passed" if all(part["status"] == "passed" for part in report.values()) else "failed"
    report["check_implementation_sha256"] = {}
    implementation = directory / "implementation"
    implementation.mkdir(exist_ok=True)
    for filename in ("collision_prediction_core.py", "collision_prediction_batch.py", "collision_prediction_smoke.py"):
        source = Path(__file__).with_name(filename)
        report["check_implementation_sha256"][filename] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, implementation / ("check_" + filename))
    (directory / "smoke_test.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Collision prediction smoke test", "", f"Status: **{report['status']}**", "",
             "Geometry is measured from the initial native floor-contact positions (SVD tolerance 0.1 mm).", "",
             "| Object | Modes | Initial counts | Measured floor geometry | Contacting components, max |",
             "|---|---|---|---|---|"]
    for name, details in coverage["objects"].items():
        lines.append(f"| {name} | {details['primary_modes']} | {details['primary_contact_counts']} | "
                     f"{details['measured_floor_geometry']} | {details['max_contacting_object_geoms']} |")
    if parallel:
        lines.extend(["", "Native nworld batch vs independent serial 20 ms rollouts:", "",
                      "| Object | GPU worlds | Status | Max position difference (m) | Physics speedup |",
                      "|---|---|---|---|---|"])
        for name, details in report["parallel"]["objects"].items():
            speedup = details["physics_timing"].get("physics_speedup")
            lines.append(f"| {name} | {details['native_gpu_worlds']} | {details['status']} | "
                         f"{details['max_differences']['position_m']:.3g} | {speedup if speedup else 'not measured'} |")
        lines.extend(["", "Physics timings exclude allocations, transfers, contact diagnostics and SCM/ComFree predictions.",
                      "These ratios are not full-pipeline speedups.", "",
                      "Acceptance tolerances over all 11 frames and both bodies: 20 µm position, 1 mrad rotation,",
                      "0.002 m/s linear and 0.05 rad/s angular velocity. Labels and initial counts must agree exactly.",
                      "The JSON also retains results against the earlier stricter tolerances (5 µm, 0.1 mrad, 0.0005 m/s, 0.005 rad/s).",
                      "CCD warnings cause smaller GPU batches and/or serial retries; diagnostics retain every retry.",
                      "Invalid reference repetitions are excluded with reasons, never counted as matches; at least half the samples",
                      "and at least two genuine GPU worlds must remain valid for each object to pass."])
    lines.extend(["", "Missing coverage:", json.dumps(coverage["missing"]), ""])
    (directory / "smoke_test.md").write_text("\n".join(lines))
    return report
