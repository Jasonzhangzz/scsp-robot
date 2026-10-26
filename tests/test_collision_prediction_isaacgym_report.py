"""Offline checks: no IsaacGym, MuJoCo or GPU imports are required."""
import csv
import gzip
import json

import numpy as np
import pytest

from examples.mpc.fingertips.test.collision_prediction_isaacgym_report import (
    METRICS, RELATIVE_METRICS, case_counts, coverage_from_saved_contacts, floor_geometry,
    report_comfree_batches, report_directory, statistics, summarize)


def row(sample="q", method="scm", mode="sticking", count="1", status="ok", **kwargs):
    return dict(sample_id=sample, object="sphere", method=method, mode=mode, contact_group=count,
                status=status, selected="True", geometry="point", seed="42", force_N="0.1",
                speed_level_m_s="0.2", **{metric: "2" for metric in METRICS},
                **{metric: "" for metric in RELATIVE_METRICS}, **kwargs)


def cell(summary, **match):
    return next(item for item in summary if all(item[key] == value for key, value in match.items()))


def test_failure_empty_relative_and_nonfinite_are_not_zero():
    rows = [row(), row("failed", status="failed"), row("nonfinite")]
    rows[-1]["orientation_error_deg"] = "nan"
    summary = summarize(rows, ["sphere"])
    item = cell(summary, object="sphere", mode="sticking", contact_group="1", method="scm", metric="position_error_mm")
    assert (item["n_total"], item["n_valid"], item["n_failed"], item["n_defined"]) == (3, 2, 1, 2)
    assert item["mean"] == 2.
    undefined = cell(summary, object="sphere", mode="sticking", contact_group="1", method="scm", metric="relative_position_error")
    assert undefined["mean"] is None and undefined["n_undefined"] == 2
    finite = cell(summary, object="sphere", mode="sticking", contact_group="1", method="scm", metric="orientation_error_deg")
    assert finite["n_defined"] == 1 and finite["n_undefined"] == 1
    empty = cell(summary, object="sphere", mode="sliding", contact_group="0", method="comfree", metric="position_error_mm")
    assert empty["n_total"] == 0 and empty["mean"] is None
    assert statistics([1., 3., "nan"])["std"] == pytest.approx(1.)


def test_onset_alias_mixed_and_invalid_are_retained():
    rows = [row(mode="onset"), row("mixed", mode="mixed_transition"),
            row("invalid", mode="invalid_reference", count="unknown", status="invalid_reference")]
    summary = summarize(rows, ["sphere"])
    assert cell(summary, object="sphere", mode="contact_establishment", contact_group="1", method="scm",
                metric="position_error_mm")["n_valid"] == 1
    assert cell(summary, object="all", mode="mixed_transition", contact_group="all", method="scm",
                metric="position_error_mm")["n_valid"] == 1
    assert cell(summary, object="sphere", mode="invalid_reference", contact_group="all", method="scm",
                metric="position_error_mm")["n_failed"] == 1


def test_query_counts_not_three_methods_and_no_new_quota():
    rows = [row(method=method) for method in ("scm", "comfree", "isaacgym")]
    rows[1]["status"] = "failed"
    rows.extend([row("bad", method=method, mode="invalid_reference", count="unknown", status="invalid_reference")
                 for method in ("scm", "comfree", "isaacgym")])
    counts = case_counts(rows, ["sphere"])
    marginal = cell(counts, grouping="object_mode", object="sphere", mode="sticking")
    assert marginal["n_selected"] == marginal["n_valid_reference"] == 1
    assert marginal["n_failed_comfree"] == 1 and marginal["n_missing_scm"] == 0
    assert marginal["target"] is None and marginal["shortfall"] is None
    assert marginal["below_100"] and marginal["below_200"]
    total = cell(counts, grouping="object_mode", object="all", mode="all")
    assert total["n_selected"] == 2 and total["n_invalid_reference"] == 1
    assert cell(counts, grouping="object_mode_count", object="sphere", mode="sliding", contact_group="2")["n_selected"] == 0


def test_floor_geometry_excludes_ball_and_positive_gap_candidates():
    snapshot = dict(pair=[0, 0, 0, 1], gap=[-1e-5, -1e-5, .0005, -1e-5],
                    position=[[0, 0, 0], [.01, 0, 0], [0, .01, 0], [0, 1, 1]])
    assert floor_geometry(snapshot) == "line"
    snapshot["gap"][2] = 0
    assert floor_geometry(snapshot) == "face"
    assert floor_geometry(dict(pair=[1], position=[[0, 0, 0]])) == "none"
    with pytest.raises(ValueError, match="length mismatch"):
        floor_geometry(dict(pair=[0], position=[]))


def fixture_directory(tmp_path, include_contacts=True):
    folder = tmp_path / "sphere"
    folder.mkdir(parents=True)
    manifest = dict(objects=["sphere"], methods=["scm", "comfree", "isaacgym"], status="complete")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows = [row(method=method) for method in ("scm", "comfree", "isaacgym")]
    with (folder / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if include_contacts:
        with gzip.open(folder / "reference_contacts.jsonl.gz", "wt") as stream:
            stream.write(json.dumps(dict(sample_id="q", snapshots=[dict(pair=[0], gap=[-1e-5], position=[[0, 0, 0]])])) + "\n")
    return tmp_path


def test_fixed_source_missing_coverage_is_nonfatal_but_missing_records_are_integrity_error(tmp_path):
    folder = fixture_directory(tmp_path)
    coverage = coverage_from_saved_contacts(folder)
    assert coverage["status"] == "complete"
    assert coverage["coverage_complete"] is False
    assert coverage["missing_coverage"] and not coverage["integrity_errors"]
    assert coverage["objects"]["sphere"]["measured_floor_geometry"] == {"point": 1}
    assert coverage["objects"]["sphere"]["convex_contacting_shape_count"] is None
    (folder / "sphere" / "reference_contacts.jsonl.gz").unlink()
    assert coverage_from_saved_contacts(folder)["status"] == "integrity_error"


def test_report_is_offline_repeatable_and_leaves_raw_data_unchanged(tmp_path):
    folder = fixture_directory(tmp_path)
    manifest = (folder / "manifest.json").read_bytes()
    predictions = (folder / "sphere" / "predictions.csv").read_bytes()
    report_directory(folder, plots=False)
    first = (folder / "summary.log").read_bytes()
    report_directory(folder, plots=False)
    assert (folder / "summary.log").read_bytes() == first
    assert (folder / "manifest.json").read_bytes() == manifest
    assert (folder / "sphere" / "predictions.csv").read_bytes() == predictions
    assert (folder / "geometry_summary.csv").exists()
    assert (folder / "stratified_summary.csv").exists()
    assert json.loads((folder / "report_provenance.json").read_text())["n_rows"] == 3
    records = [json.loads(line) for line in first.decode().splitlines()]
    assert {record["kind"] for record in records} == {"case_count", "error_summary", "stratified_error_summary", "geometry_error_summary"}
    assert any(record.get("metric") == "relative_position_error" for record in records)


def test_duplicate_rows_rejected(tmp_path):
    folder = fixture_directory(tmp_path)
    path = folder / "sphere" / "predictions.csv"
    lines = path.read_text().splitlines()
    path.write_text("\n".join([*lines, lines[1]]) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        report_directory(folder, plots=False)


def test_comfree_batch_report_freezes_inputs_measures_clean_speedup_and_checks_parity(tmp_path):
    source = fixture_directory(tmp_path / "source")
    folder = source / "sphere"
    (folder / "queries").mkdir()
    (folder / "model.xml").write_text("model")
    import hashlib
    (folder / "object.json").write_text(json.dumps(dict(model_sha256=hashlib.sha256(b"model").hexdigest(),
                                                      packaged_asset_sha256={})))
    qpos = np.zeros(14)
    qpos[[3, 10]] = 1.
    metadata = dict(sample_id="q", mode="sticking", contact_group="1", source_query_sha256="source-seal")
    np.savez_compressed(folder / "queries/q.npz", qpos=qpos, qvel=np.zeros(12), wrench=np.zeros((3, 6)),
        metadata_json=np.array(json.dumps(metadata)), initial_manifold_json=np.array('{"gap": [0]}'),
        comfree_qpos=qpos, comfree_qvel=np.zeros(12))
    output = tmp_path / "batches"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps(dict(source=str(source), batch_sizes=[1, 4, 16])))
    for size, seconds, failed in ((1, 2., 0), (4, 1., 0), (16, 1., 1)):
        destination = output / f"batch_{size}"
        (destination / "sphere").mkdir(parents=True)
        item = row(method="comfree", source_query_sha256="source-seal")
        with (destination / "sphere/predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(item))
            writer.writeheader()
            writer.writerow(item)
        np.savez_compressed(destination / "sphere/batch_000000.npz", sample_ids=np.array(["q"]),
                            qpos=[qpos], qvel=[np.zeros(12)])
        (destination / "comfree_batches.json").write_text(json.dumps([dict(object="sphere", nworld=1,
            requested_nworld=1, physics_seconds=seconds, end_to_end_seconds=seconds+1., n_failed=failed)]))
    checks = report_comfree_batches(output)
    assert checks["status"] == "passed" and checks["n_source_queries"] == 1
    with (output / "throughput.csv").open() as stream:
        throughput = list(csv.DictReader(stream))
    assert float(cell(throughput, object="all", batch_size="4")["physics_speedup_vs_batch_1"]) == 2.
    assert cell(throughput, object="all", batch_size="16")["physics_speedup_vs_batch_1"] == ""
    assert cell(throughput, object="all", batch_size="16")["physics_seconds"] == ""
    with (output / "source_queries.csv").open() as stream:
        seals = list(csv.DictReader(stream))
    assert all(len(seals[0][key]) == 64 for key in ("query_sha256", "initial_manifold_sha256", "qpos_sha256", "qvel_sha256", "wrench_sha256"))
    with (output / "batch_parity.csv").open() as stream:
        parity = list(csv.DictReader(stream))
    assert len(parity) == 3 and all(float(item["target_position_difference_m"]) == 0. for item in parity)
