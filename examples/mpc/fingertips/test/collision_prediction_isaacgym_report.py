"""Offline reports for the PhysX-native collision prediction benchmark.

This module intentionally imports neither IsaacGym nor MuJoCo. Reports must be
regenerable on a CPU machine from the saved CSVs and native contact records.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("scm", "comfree", "isaacgym")
MODES = ("separation", "contact_establishment", "sticking", "sliding")
ALL_MODES = MODES + ("mixed_transition", "invalid_reference")
GROUPS = ("0", "1", "2", ">=3")
METRICS = ("position_error_mm", "orientation_error_deg", "pose_error", "quaternion_error",
           "linear_velocity_error_m_s", "angular_velocity_error_rad_s")
RELATIVE_METRICS = ("relative_position_error", "relative_orientation_error", "relative_pose_error",
                    "relative_linear_velocity_error", "relative_angular_velocity_error")
REPORT_METRICS = METRICS + RELATIVE_METRICS
MODE_LABELS = dict(contact_establishment="Onset", separation="Separation", sticking="Sticking",
                   sliding="Sliding", mixed_transition="Mixed transition", invalid_reference="Invalid reference")


def _mode(value):
    return "contact_establishment" if value in ("onset", "Onset") else value


def _selected(row):
    return str(row.get("selected", "True")).lower() in ("true", "1", "yes")


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def statistics(values):
    values = np.asarray([number for item in values if (number := _finite(item)) is not None], float)
    if not len(values):
        return dict.fromkeys(("mean", "std", "rmse", "median", "p95"))
    return dict(mean=float(values.mean()), std=float(values.std()), rmse=float(np.sqrt(np.mean(values**2))),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)))


def _metric_rows(rows, identity):
    valid = [row for row in rows if row.get("status") == "ok"]
    for metric in REPORT_METRICS:
        values = [value for row in valid if (value := _finite(row.get(metric))) is not None]
        yield dict(**identity, metric=metric, n_total=len(rows), n_valid=len(valid),
                   n_failed=len(rows)-len(valid), n_defined=len(values), n_undefined=len(valid)-len(values),
                   **statistics(values))


def _modes(rows):
    return tuple(dict.fromkeys((*ALL_MODES, *(sorted({_mode(row.get("mode", "invalid_reference")) for row in rows})))))


def summarize(rows, objects, methods=METHODS):
    """All object × regime × start-count cells, including empty and mixed cells."""
    bins = defaultdict(list)
    for row in rows:
        if not _selected(row):
            continue
        mode = _mode(row.get("mode", "invalid_reference"))
        for obj in (row["object"], "all"):
            for regime in (mode, "all"):
                for group in (row.get("contact_group", "unknown"), "all"):
                    bins[obj, regime, group, row["method"]].append(row)
    return [item for obj in (*objects, "all") for mode in (*_modes(rows), "all") for group in (*GROUPS, "all")
            for method in methods for item in _metric_rows(bins[obj, mode, group, method],
                dict(object=obj, mode=mode, contact_group=group, method=method))]


def stratified_summary(rows, methods=METHODS):
    result = []
    for dimension in ("force_N", "speed_level_m_s", "seed", "geometry"):
        values = {str(row.get(dimension, "")) for row in rows if _selected(row)} - {"", "None"}
        for value in sorted(values, key=lambda item: (0, float(item)) if _finite(item) is not None else (1, item)):
            for method in methods:
                subset = [row for row in rows if _selected(row) and str(row.get(dimension, "")) == value
                          and row["method"] == method]
                result.extend(_metric_rows(subset, dict(dimension=dimension, value=value, method=method)))
    return result


def geometry_summary(rows, objects, methods=METHODS):
    """Nominal configurations are distinct from measured manifold rank."""
    result = []
    for obj in (*objects, "all"):
        obj_rows = [row for row in rows if _selected(row) and (obj == "all" or row["object"] == obj)]
        shapes = sorted({_geometry(row) for row in obj_rows})
        for geometry in shapes:
            for mode in (*_modes(rows), "all"):
                for method in methods:
                    subset = [row for row in obj_rows if _geometry(row) == geometry and row["method"] == method
                              and (mode == "all" or _mode(row.get("mode")) == mode)]
                    result.extend(_metric_rows(subset, dict(object=obj, geometry=geometry, mode=mode, method=method)))
    return result


def _geometry(row):
    return str(row.get("geometry", "unknown")).removesuffix("_airborne")


def case_counts(rows, objects, reference_method="isaacgym"):
    """Count input identities once, without imposing new quotas after relabeling."""
    references = [row for row in rows if row["method"] == reference_method and _selected(row)]
    predictions = {(row["object"], row["sample_id"], row["method"]): row for row in rows}
    result = []
    for obj in (*objects, "all"):
        obj_rows = [row for row in references if obj == "all" or row["object"] == obj]
        shapes = sorted({_geometry(row) for row in obj_rows}) or ["unknown"]
        for mode in (*_modes(rows), "all"):
            for grouping, groups, geometries in (
                    ("object_mode", ["all"], ["all"]),
                    ("object_mode_count", GROUPS, ["all"]),
                    ("object_mode_geometry", ["all"], shapes),
                    ("object_mode_count_geometry", GROUPS, shapes)):
                for group in groups:
                    for geometry in geometries:
                        selected = [row for row in obj_rows
                                    if (mode == "all" or _mode(row.get("mode")) == mode)
                                    and (group == "all" or row.get("contact_group") == group)
                                    and (geometry == "all" or _geometry(row) == geometry)]
                        valid = [row for row in selected if row["status"] == "ok"]
                        failures = {}
                        for method in ("scm", "comfree"):
                            predictions_here = [predictions.get((row["object"], row["sample_id"], method)) for row in selected]
                            failures["n_failed_" + method] = sum(item is not None and item["status"] != "ok"
                                                                  for item in predictions_here)
                            failures["n_missing_" + method] = sum(item is None for item in predictions_here)
                        result.append(dict(grouping=grouping, object=obj, mode=mode, contact_group=group,
                            geometry=geometry, n_selected=len(selected), n_valid_reference=len(valid),
                            n_invalid_reference=len(selected)-len(valid), **failures,
                            target=None, shortfall=None, below_100=len(valid) < 100, below_200=len(valid) < 200,
                            status="observed" if valid else "no_valid_samples_unverified_feasibility"))
    return result


def floor_geometry(snapshot, tolerance=1e-4):
    """Affine rank of actual target–floor geometric contact positions."""
    pairs = np.asarray(snapshot.get("pair", []), int)
    points = np.asarray(snapshot.get("position", []), float).reshape(-1, 3)
    if len(points) != len(pairs):
        raise ValueError("Contact position/pair length mismatch")
    keep = pairs == 0
    if "gap" in snapshot:
        gaps = np.asarray(snapshot["gap"], float)
        if len(gaps) != len(points):
            raise ValueError("Contact gap/pair length mismatch")
        keep &= gaps <= 0.
    points = points[keep]
    if not len(points):
        return "none"
    singular = np.linalg.svd(points-points.mean(0), compute_uv=False)
    return ("point", "line", "face", "volume")[int(np.count_nonzero(singular > tolerance))]


def _contact_records(folder):
    for path in sorted(folder.glob("*contacts*.jsonl*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def coverage_from_saved_contacts(output):
    """Coverage gaps are descriptive, never failure of fixed-source sampling.

    Missing/corrupt records remain separately visible as integrity errors.
    Shape identities unavailable in IsaacGym records cannot establish how many
    convex pieces are contacting, and are deliberately not inferred here.
    """
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    result = dict(status="complete", sampling_policy="fixed_source_no_new_quotas",
                  contact_geometry_tolerance_m=1e-4, objects={}, missing_coverage=[], integrity_errors=[])
    totals = {key: Counter() for key in ("modes", "contact_counts", "measured_floor_geometry")}
    for obj in manifest["objects"]:
        folder = output / obj
        rows = _read_csv(folder / "predictions.csv")
        refs = {row["sample_id"]: row for row in rows if row["method"] == "isaacgym" and _selected(row)}
        valid = {key: row for key, row in refs.items() if row["status"] == "ok"}
        modes = Counter(_mode(row.get("mode")) for row in valid.values())
        counts = Counter(row.get("contact_group", "unknown") for row in valid.values())
        geometries, examples, seen = Counter(), {}, set()
        try:
            for record in _contact_records(folder):
                sample_id = record.get("sample_id")
                if sample_id not in valid or not record.get("snapshots"):
                    continue
                if sample_id in seen:
                    result["integrity_errors"].append(f"{obj}/{sample_id}: duplicate contact record")
                    continue
                seen.add(sample_id)
                geometry = floor_geometry(record["snapshots"][0])
                geometries[geometry] += 1
                examples.setdefault(geometry, sample_id)
        except (ValueError, OSError, TypeError) as exc:
            result["integrity_errors"].append(f"{obj}: {type(exc).__name__}: {exc}")
        missing = sorted(set(valid)-seen)
        if missing:
            result["integrity_errors"].append(f"{obj}: {len(missing)} valid queries lack readable start contacts")
        for mode in (*MODES, "mixed_transition"):
            if not modes[mode]:
                result["missing_coverage"].append(f"{obj}: regime={mode}")
        for key, values in (("modes", modes), ("contact_counts", counts), ("measured_floor_geometry", geometries)):
            totals[key].update(values)
        result["objects"][obj] = dict(n_input=len(refs), n_valid_reference=len(valid), n_failed_reference=len(refs)-len(valid),
            modes=dict(modes), contact_counts=dict(counts), measured_floor_geometry=dict(geometries),
            geometry_examples=examples, n_contact_records=len(seen), missing_contact_records=missing,
            convex_contacting_shape_count=None, convex_shape_identity_available=False,
            mode_contact_count_cells=[dict(mode=mode, contact_group=group,
                n=sum(_mode(row.get("mode")) == mode and row.get("contact_group") == group for row in valid.values()))
                for mode in (*MODES, "mixed_transition") for group in GROUPS])
    for group in GROUPS:
        if not totals["contact_counts"][group]:
            result["missing_coverage"].append(f"overall: contact_count={group}")
    for geometry in ("point", "line", "face"):
        if not totals["measured_floor_geometry"][geometry]:
            result["missing_coverage"].append(f"overall: measured_floor_geometry={geometry}")
    result.update({key: dict(value) for key, value in totals.items()})
    result["coverage_complete"] = not result["missing_coverage"]
    if result["integrity_errors"]:
        result["status"] = "integrity_error"
    return result


def _read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def plot_report(rows, summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    main_metrics = (("position_error_mm", "Position error (mm)"), ("orientation_error_deg", "Rotation error (deg)"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (metric, label) in zip(axes, main_metrics):
        values = [[number for row in rows if _selected(row) and row["method"] == method and row["status"] == "ok"
                   and (number := _finite(row.get(metric))) is not None] for method in METHODS[:2]]
        ax.boxplot([value or [np.nan] for value in values], tick_labels=["SCM", "ComFree"], showfliers=False)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"error_distributions.{extension}", dpi=180)
    plt.close(fig)
    labels = (*MODES, "mixed_transition")
    indexed = {(r["method"], r["mode"], r["contact_group"], r["metric"]): r for r in summaries if r["object"] == "all"}
    for metric, label in main_metrics:
        matrices = [np.array([[indexed[method, mode, group, metric]["mean"]
                             if indexed[method, mode, group, metric]["mean"] is not None else np.nan
                             for group in GROUPS] for mode in labels], float) for method in METHODS[:2]]
        finite = np.concatenate(matrices)
        maximum = max(float(np.nanmax(finite)) if np.isfinite(finite).any() else 0., 1e-12)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, method, matrix in zip(axes, METHODS[:2], matrices):
            im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=maximum)
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:.3g}" if np.isfinite(value) else "no samples", ha="center", va="center",
                        fontsize=8, color="white" if np.isfinite(value) and value < .4*maximum else "black")
            ax.set_xticks(range(len(GROUPS)), GROUPS)
            ax.set_yticks(range(len(labels)), [MODE_LABELS[mode] for mode in labels])
            ax.set_xlabel("Contact count at query start")
            ax.set_title(method.upper() if method == "scm" else "ComFree")
            fig.colorbar(im, ax=ax, label="Mean " + label.lower())
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(output / f"contact_count_heatmap_{metric}.{extension}", dpi=180)
        plt.close(fig)


def report_directory(output, plots=True):
    """Regenerate derived reports only; query results and source remain immutable."""
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    objects = manifest["objects"]
    rows = [row for obj in objects for row in _read_csv(output / obj / "predictions.csv")]
    identities = [(row["object"], row["sample_id"], row["method"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate object/sample_id/method prediction rows")
    methods = tuple(dict.fromkeys((*METHODS, *(row["method"] for row in rows))))
    summaries = summarize(rows, objects, methods)
    strata = stratified_summary(rows, methods)
    geometries = geometry_summary(rows, objects, methods)
    counts = case_counts(rows, objects)
    for name, records in (("summary", summaries), ("stratified_summary", strata),
                          ("geometry_summary", geometries), ("case_counts", counts)):
        write_csv(output / f"{name}.csv", records)
    with (output / "summary.log").open("w") as stream:
        for kind, records in (("case_count", counts), ("error_summary", summaries),
                              ("stratified_error_summary", strata), ("geometry_error_summary", geometries)):
            for item in records:
                stream.write(json.dumps(dict(kind=kind, **item), allow_nan=False)+"\n")
    count_lines = ["# PhysX measured sample counts", "",
        "Frozen input queries are relabeled by PhysX; there are no new per-mode quotas.",
        "Empty cells have unverified feasibility. Flags use valid reference counts; invalid references are separate.", "",
        "| Grouping | Object | Regime | Contact count | Geometry | Inputs | Valid reference | Invalid reference | <100 | <200 |",
        "|---|---|---|---|---|---:|---:|---:|---|---|"]
    for item in counts:
        count_lines.append("| " + " | ".join(str(item[key]) for key in ("grouping", "object", "mode", "contact_group",
            "geometry", "n_selected", "n_valid_reference", "n_invalid_reference", "below_100", "below_200")) + " |")
    (output / "case_counts.md").write_text("\n".join(count_lines)+"\n")
    lines = ["# 20 ms collision prediction against PhysX", "", f"Run status: **{manifest.get('status', 'unknown')}**.", "",
        "IsaacGym GPU PhysX advances 0.002 s × 10. SCM/ComFree each advance 0.02 s × 1 with PhysX-native initial contacts.",
        "The extra terminal contact diagnostic solves 20–22 ms and does not change the 20 ms reference endpoint.",
        "Absolute position/rotation errors use the reference terminal pose. Relative ratios use reference motion; near-zero denominators remain undefined.",
        "Mixed events and invalid references are retained. Failed predictions never become zero errors; std is the population standard deviation.", "",
        "| Object | Regime | Contact count | Method | Position mm, mean ± std | Rotation deg, mean ± std | Valid / total |",
        "|---|---|---|---|---|---|---|"]
    lookup = {(row["object"], row["mode"], row["contact_group"], row["method"], row["metric"]): row for row in summaries}
    for row in summaries:
        if row["metric"] != "position_error_mm" or row["n_total"] == 0:
            continue
        def cell(item):
            return "—" if item["mean"] is None else f"{item['mean']:.5g} ± {item['std']:.3g}"
        angle = lookup[row["object"], row["mode"], row["contact_group"], row["method"], "orientation_error_deg"]
        lines.append(f"| {row['object']} | {MODE_LABELS.get(row['mode'], row['mode'])} | {row['contact_group']} | "
                     f"{row['method']} | {cell(row)} | {cell(angle)} | {row['n_valid']} / {row['n_total']} |")
    (output / "summary.md").write_text("\n".join(lines)+"\n")
    coverage = coverage_from_saved_contacts(output)
    _json_write(output / "coverage.json", coverage)
    _json_write(output / "report_provenance.json", dict(report_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        predictions_sha256={obj: hashlib.sha256((output / obj / "predictions.csv").read_bytes()).hexdigest()
                            for obj in objects if (output / obj / "predictions.csv").exists()},
        n_rows=len(rows), methods=list(methods), source_manifest_unchanged=True))
    if plots and rows:
        plot_report(rows, summaries, output)
    return summaries


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report_comfree_batches(output):
    """Frozen-input batch-size comparison, including full and physics timings.

    The completed main experiment is the baseline. We verify its saved models
    and query identities without recalculating reference dynamics or contact
    geometry. Missing original shape IDs remain absent rather than inferred.
    """
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    source = Path(manifest["source"])
    source_manifest = json.loads((source / "manifest.json").read_text())
    checks = dict(status="passed", source_manifest_sha256=_sha(source / "manifest.json"), errors=[],
                  n_source_queries=0, batch_sizes=list(manifest["batch_sizes"]))
    if source_manifest.get("status") != "complete":
        checks["errors"].append("source experiment is incomplete")
    if manifest.get("source_manifest_sha256") not in (None, checks["source_manifest_sha256"]):
        checks["errors"].append("source manifest hash mismatch")
    source_queries, originals, seals = {}, {}, []
    for obj in source_manifest["objects"]:
        folder = source / obj
        info = json.loads((folder / "object.json").read_text())
        for filename, expected in {"model.xml": info["model_sha256"], **info.get("packaged_asset_sha256", {})}.items():
            if _sha(folder / filename) != expected:
                checks["errors"].append(f"source asset hash mismatch: {obj}/{filename}")
        for path in sorted((folder / "queries").glob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"]))
                identity = (obj, metadata["sample_id"])
                if identity in source_queries:
                    checks["errors"].append(f"duplicate source query: {identity}")
                    continue
                source_queries[identity] = metadata
                originals[identity] = (archive["comfree_qpos"].copy(), archive["comfree_qvel"].copy())
                manifold = json.loads(str(archive["initial_manifold_json"]))
                canonical = json.dumps(manifold, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                seals.append(dict(object=obj, sample_id=metadata["sample_id"], query=str(path),
                    query_sha256=_sha(path), initial_manifold_sha256=hashlib.sha256(canonical).hexdigest(),
                    **{key+"_sha256": hashlib.sha256(np.ascontiguousarray(archive[key]).tobytes()).hexdigest()
                       for key in ("qpos", "qvel", "wrench")}))
    checks["n_source_queries"] = len(source_queries)
    write_csv(output / "source_queries.csv", seals)
    summaries, strata, throughputs, parity = [], [], [], []
    for size in manifest["batch_sizes"]:
        directory = output / f"batch_{size}"
        rows = [row for obj in source_manifest["objects"] for row in _read_csv(directory / obj / "predictions.csv")]
        identities = [(row["object"], row["sample_id"]) for row in rows]
        if len(identities) != len(set(identities)):
            checks["errors"].append(f"batch {size}: duplicate prediction identities")
        if set(identities) != set(source_queries):
            checks["errors"].append(f"batch {size}: frozen query set mismatch")
        for row in rows:
            old = source_queries.get((row["object"], row["sample_id"]))
            if old and any(str(row.get(key)) != str(old.get(key)) for key in ("mode", "contact_group", "source_query_sha256")):
                checks["errors"].append(f"batch {size}: saved query metadata changed: {row['sample_id']}")
        summaries.extend(dict(batch_size=size, **row) for row in summarize(rows, source_manifest["objects"], methods=("comfree",)))
        strata.extend(dict(batch_size=size, **row) for row in stratified_summary(rows, methods=("comfree",)))
        diagnostics = json.loads((directory / "comfree_batches.json").read_text())
        for obj in (*source_manifest["objects"], "all"):
            batches = [item for item in diagnostics if obj == "all" or item["object"] == obj]
            physics = [_finite(item.get("physics_seconds")) for item in batches]
            total = [_finite(item.get("end_to_end_seconds")) for item in batches]
            n_worlds = sum(int(item.get("nworld", 0)) for item in batches)
            n_failed = sum(int(item.get("n_failed", 0)) for item in batches)
            physics_total = sum(value for value in physics if value is not None)
            wall_total = sum(value for value in total if value is not None)
            clean = bool(batches) and not n_failed and all(value is not None and value > 0 for value in physics)
            timing = dict(batch_size=size, object=obj, n_batches=len(batches), n_physics_queries=n_worlds,
                n_native_multiworld_batches=sum(int(item.get("nworld", 0)) > 1 for item in batches),
                n_failed=n_failed, n_unmeasured_physics_batches=sum(value is None for value in physics),
                clean_physics_timing=clean, physics_seconds=physics_total if clean else None,
                end_to_end_seconds=wall_total if batches and all(value is not None for value in total) else None,
                physics_queries_per_second=n_worlds/physics_total if clean and physics_total > 0 else None,
                end_to_end_queries_per_second=n_worlds/wall_total if wall_total > 0 else None,
                physics_speedup_vs_batch_1=None,
                **{"physics_batch_seconds_"+key: value for key, value in statistics(physics).items()},
                **{"end_to_end_batch_seconds_"+key: value for key, value in statistics(total).items()})
            throughputs.append(timing)
        seen = set()
        status = {(row["object"], row["sample_id"]): row["status"] for row in rows}
        for obj in source_manifest["objects"]:
            for path in sorted((directory / obj).glob("batch_*.npz")):
                with np.load(path, allow_pickle=False) as archive:
                    for sample_id, qpos, qvel in zip(archive["sample_ids"], archive["qpos"], archive["qvel"]):
                        identity = (obj, str(sample_id))
                        seen.add(identity)
                        expected = originals.get(identity)
                        finite = (expected is not None and np.isfinite(np.r_[qpos, qvel, *expected]).all())
                        distance = angle = qpos_abs = qvel_abs = None
                        if finite:
                            oldq, oldv = expected
                            qpos_abs, qvel_abs = float(np.max(np.abs(qpos-oldq))), float(np.max(np.abs(qvel-oldv)))
                            distance = float(np.linalg.norm(qpos[:3]-oldq[:3]))
                            norms = np.linalg.norm(qpos[3:7])*np.linalg.norm(oldq[3:7])
                            if norms > 0:
                                cosine = float(np.clip(abs(np.dot(qpos[3:7], oldq[3:7]))/norms, 0., 1.))
                                angle = float(np.degrees(2*np.arccos(1. if 1-cosine < 1e-15 else cosine)))
                        parity.append(dict(batch_size=size, object=obj, sample_id=str(sample_id),
                            status=status.get(identity, "missing"), baseline_finite=bool(finite),
                            qpos_max_abs_difference=qpos_abs, qvel_max_abs_difference=qvel_abs,
                            target_position_difference_m=distance, target_rotation_difference_deg=angle))
        if seen != set(source_queries):
            checks["errors"].append(f"batch {size}: saved prediction NPZ query set mismatch")
    baseline = {row["object"]: row for row in throughputs if row["batch_size"] == 1 and row["clean_physics_timing"]}
    for row in throughputs:
        old = baseline.get(row["object"])
        if old and row["clean_physics_timing"] and row["n_physics_queries"] == old["n_physics_queries"]:
            row["physics_speedup_vs_batch_1"] = row["physics_queries_per_second"]/old["physics_queries_per_second"]
    for name, records in (("summary", summaries), ("stratified_summary", strata), ("throughput", throughputs),
                          ("batch_parity", parity)):
        write_csv(output / f"{name}.csv", records)
    with (output / "summary.log").open("w") as stream:
        for kind, records in (("error_summary", summaries), ("stratified_error_summary", strata), ("throughput", throughputs)):
            for row in records:
                stream.write(json.dumps(dict(kind=kind, **row), allow_nan=False)+"\n")
    checks["status"] = "failed" if checks["errors"] else "passed"
    checks["throughput_scope"] = "ComFree physics API excludes allocation/upload/warmup; end-to-end includes them"
    _json_write(output / "batch_verification.json", checks)
    return checks
