"""Replay, native GPU batching and impulse audits for the PhysX dataset.

The audit uses real Isaac Gym and ComFree backends. A missing SDK or failed
native comparison is recorded as a failure, never replaced by another engine.
Native force collection at the endpoint remains the separately documented
[20,22] ms diagnostic; only the saved 20-ms state is compared here.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()
import mujoco
import numpy as np

from examples.mpc.fingertips.test.collision_prediction_core import Settings, classify_contacts, pose_errors
from examples.mpc.fingertips.test.collision_prediction_external_contacts import ExternalContactPredictors
from examples.mpc.fingertips.test.collision_prediction_isaacgym_native import (
    IsaacGymNative, normalize_manifold, quaternion_matrix)

TOLERANCES = dict(position_m=2e-5, orientation_rad=1e-3,
                  linear_velocity_m_s=.002, angular_velocity_rad_s=.05)


def _world_velocities(model, qpos, qvel):
    """Both COM velocities in world axes; no collision or dynamics call."""
    result = []
    for local, body in enumerate((model.body("object").id, model.body("ball").id)):
        q, v = local*7, local*6
        rotation = quaternion_matrix(qpos[q+3:q+7])
        omega = rotation @ qvel[v+3:v+6]
        linear = qvel[v:v+3] + np.cross(omega, rotation @ model.body_ipos[body])
        result.append(np.r_[linear, omega])
    return np.asarray(result)


def compare_states(model, expected_q, expected_v, actual_q, actual_v):
    expected_q, expected_v, actual_q, actual_v = map(
        lambda x: np.asarray(x, float), (expected_q, expected_v, actual_q, actual_v))
    if (expected_q.shape != (14,) or actual_q.shape != (14,) or expected_v.shape != (12,)
            or actual_v.shape != (12,) or not np.isfinite(np.r_[expected_q, expected_v, actual_q, actual_v]).all()):
        return dict(passed=False, failure_reason="Malformed or nonfinite endpoint state")
    positions, angles = [], []
    for start in (0, 7):
        metric = pose_errors(actual_q[start:start+7], expected_q[start:start+7], 1.)
        positions.append(metric["position_error_mm"]/1000)
        angles.append(float(np.radians(metric["orientation_error_deg"])))
    velocity = _world_velocities(model, actual_q, actual_v)-_world_velocities(model, expected_q, expected_v)
    differences = dict(position_m=max(positions), orientation_rad=max(angles),
                       linear_velocity_m_s=float(np.linalg.norm(velocity[:, :3], axis=1).max()),
                       angular_velocity_rad_s=float(np.linalg.norm(velocity[:, 3:], axis=1).max()))
    exceeded = [key for key, value in differences.items() if value > TOLERANCES[key]]
    return dict(passed=not exceeded, differences=differences, exceeded=exceeded)


def compare_labels(expected, actual):
    before, after = classify_contacts(expected), classify_contacts(actual)
    first_equal = all(before[key] == after[key] for key in (
        "ncontact_start", "ncontact_floor_start", "ncontact_ball_start"))
    sequence_before = [len(snapshot["pair"]) for snapshot in expected]
    sequence_after = [len(snapshot["pair"]) for snapshot in actual]
    sequence_equal = sequence_before == sequence_after
    mode_equal = all(before[key] == after[key] for key in ("mode", "established", "separated", "free_flight"))
    gaps = [abs(float(gap)) for snapshots in (expected, actual)
            for snapshot in snapshots for gap in snapshot.get("gap", [])]
    loads = [abs(float(force[0])-1e-4) for snapshots in (expected, actual)
             for snapshot in snapshots for force in snapshot.get("force", [])]
    return dict(passed=first_equal and mode_equal, strict_passed=first_equal and mode_equal and sequence_equal,
                mode_equal=mode_equal,
                initial_contact_count_equal=first_equal, contact_count_sequence_equal=sequence_equal,
                expected={key: before[key] for key in ("mode", "contact_group", "ncontact_start")},
                actual={key: after[key] for key in ("mode", "contact_group", "ncontact_start")},
                expected_contact_counts=sequence_before, actual_contact_counts=sequence_after,
                boundary_sensitivity=dict(minimum_absolute_gap_m=min(gaps) if gaps else None,
                    minimum_distance_from_load_threshold_N=min(loads) if loads else None,
                    note="Initial count/mode differences fail acceptance; full contact-sequence differences fail strict diagnostics"))


def compare_manifolds(expected, actual):
    """Compare native geometry independently of contact force and row ordering."""
    from scipy.optimize import linear_sum_assignment
    expected, actual = normalize_manifold(expected), normalize_manifold(actual)
    pairs = sorted(set(map(tuple, expected["body"])) | set(map(tuple, actual["body"])))
    count_differences, position, gap, angle, friction = {}, 0., 0., 0., 0.
    for pair in pairs:
        before = np.flatnonzero(np.all(expected["body"] == pair, axis=1))
        after = np.flatnonzero(np.all(actual["body"] == pair, axis=1))
        if len(before) != len(after):
            count_differences[str(pair)] = dict(expected=len(before), actual=len(after))
            continue
        distances = np.linalg.norm(expected["position"][before, None]-actual["position"][None, after], axis=2)
        first, second = linear_sum_assignment(distances)
        before, after = before[first], after[second]
        position = max(position, float(distances[first, second].max()))
        gap = max(gap, float(np.abs(expected["gap"][before]-actual["gap"][after]).max()))
        cosine = np.clip(np.sum(expected["frame"][before, 0]*actual["frame"][after, 0], axis=1), -1., 1.)
        angle = max(angle, float(np.arccos(cosine).max()))
        friction = max(friction, float(np.abs(expected["friction"][before]-actual["friction"][after]).max()))
    passed = not count_differences and position <= 2e-5 and gap <= 2e-5 and angle <= 1e-3 and friction <= 1e-6
    return dict(passed=bool(passed), strict_passed=bool(passed), body_pair_count_differences=count_differences,
                max_position_difference_m=position, max_gap_difference_m=gap,
                max_normal_angle_rad=angle, max_friction_difference=friction,
                native_contact_count_expected=len(expected["gap"]), native_contact_count_actual=len(actual["gap"]))


def _read_queries(folder, limit):
    available = []
    for path in sorted((folder/"queries").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            if metadata.get("mode") == "invalid_reference":
                continue
            arrays = {key: archive[key].copy() for key in archive.files}
        available.append(dict(path=path, sample={key: arrays[key] for key in ("qpos", "qvel", "wrench")},
                              metadata=metadata, arrays=arrays))
    # Prefer different measured regimes/counts/configurations; use every valid
    # query when the smoke set is already small.
    chosen, remaining, strata = [], [], set()
    for item in available:
        metadata = item["metadata"]
        key = tuple(metadata.get(key, "") for key in ("mode", "contact_group", "geometry"))
        if key not in strata and len(chosen) < limit:
            chosen.append(item)
            strata.add(key)
        else:
            remaining.append(item)
    return (chosen+remaining)[:limit], available


def _saved_snapshots(folder):
    import gzip
    path = folder/"reference_contacts.jsonl.gz"
    if not path.exists():
        return {}
    records = {}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            record = json.loads(line)
            records[record["sample_id"]] = record["snapshots"]
    return records


def _timings(native, reference, geometry=None):
    return dict(metadata=native.metadata, reference=reference.get("timings", {}),
                geometry=geometry.get("timings", {}) if geometry else {},
                reference_diagnostic=reference.get("diagnostic", {}),
                geometry_diagnostic=geometry.get("diagnostic", {}) if geometry else {},
                elapsed_seconds=reference.get("elapsed_seconds"),
                asset_validation=reference.get("asset_validation", []))


def _record_comparison(checks, model, name, expected_q, expected_v, actual_q, actual_v,
                       expected_snapshots=None, actual_snapshots=None, sample_id=None):
    result = compare_states(model, expected_q, expected_v, actual_q, actual_v)
    if expected_snapshots is not None and actual_snapshots is not None:
        result["labels"] = compare_labels(expected_snapshots, actual_snapshots)
        result["strict_passed"] = result["passed"] and result["labels"]["strict_passed"]
        result["passed"] &= result["labels"]["passed"]
    else:
        result["strict_passed"] = result["passed"]
    checks.append(dict(check=name, sample_id=sample_id, **result))


def _audit_object(folder, settings, python, batch_size):
    model = mujoco.MjModel.from_xml_path(str(folder/"model.xml"))
    spec = json.loads((folder/"physx_spec.json").read_text())
    selected, available = _read_queries(folder, min(8, batch_size))
    if len(selected) < 2:
        raise RuntimeError("Native GPU parallel audit needs at least two valid saved queries")
    samples = [item["sample"] for item in selected]
    ids = [item["metadata"]["sample_id"] for item in selected]
    previous_snapshots = _saved_snapshots(folder)
    checks, timing = [], {}
    with IsaacGymNative(spec, batch_size=batch_size, use_gpu=True, python=python,
                        log_path=folder/"audit_native.log") as native:
        if not native.metadata.get("physx_gpu") or native.metadata.get("device") != "cuda:0":
            raise RuntimeError("Audit requires real GPU PhysX; CPU or fallback metadata was returned")
        geometry = native.geometry_samples(samples)
        baseline = native.reference_samples(samples)
        timing["batch"] = _timings(native, baseline, geometry)
        serial_physics = 0.
        for index, (item, sample_id) in enumerate(zip(selected, ids)):
            serial_geometry = native.geometry_samples([item["sample"]])
            checks.append(dict(check="physx_geometry_batch_vs_serial", sample_id=sample_id,
                               **compare_manifolds(geometry["contacts"][index], serial_geometry["contacts"][0])))
            serial = native.reference_samples([item["sample"]])
            serial_physics += serial.get("timings", {}).get("physics_seconds", 0.)
            _record_comparison(checks, model, "physx_batch_vs_serial", baseline["qpos"][index],
                baseline["qvel"][index], serial["qpos"][0], serial["qvel"][0],
                baseline["snapshots"][index], serial["snapshots"][0], sample_id)
            arrays = item["arrays"]
            _record_comparison(checks, model, "physx_saved_vs_replayed", arrays["reference_qpos"][-1],
                arrays["reference_qvel"][-1], baseline["qpos"][index], baseline["qvel"][index],
                previous_snapshots.get(sample_id), baseline["snapshots"][index], sample_id)
        reverse = native.reference_samples(samples[::-1])
        reverse_geometry = native.geometry_samples(samples[::-1])
        for index, sample_id in enumerate(ids):
            other = len(samples)-1-index
            checks.append(dict(check="physx_geometry_batch_reordering", sample_id=sample_id,
                               **compare_manifolds(geometry["contacts"][index], reverse_geometry["contacts"][other])))
            _record_comparison(checks, model, "physx_batch_reordering", baseline["qpos"][index],
                baseline["qvel"][index], reverse["qpos"][other], reverse["qvel"][other],
                baseline["snapshots"][index], reverse["snapshots"][other], sample_id)
        # Baseline A, reversed B, then A again exercise process persistence and
        # reset isolation. Each reference call must create fresh PhysX scenes.
        repeated = native.reference_samples(samples)
        repeated_geometry = native.geometry_samples(samples)
        for index, sample_id in enumerate(ids):
            checks.append(dict(check="physx_geometry_A_B_A_reset", sample_id=sample_id,
                               **compare_manifolds(geometry["contacts"][index], repeated_geometry["contacts"][index])))
            _record_comparison(checks, model, "physx_A_B_A_reset", baseline["qpos"][index],
                baseline["qvel"][index], repeated["qpos"][index], repeated["qvel"][index],
                baseline["snapshots"][index], repeated["snapshots"][index], sample_id)
        physics = baseline.get("timings", {}).get("physics_seconds", 0.)
        timing["sum_serial_reference_physics_seconds"] = serial_physics
        timing["native_physics_speedup_vs_serial_sum"] = serial_physics/physics if physics else None
        timing["speedup_scope"] = "Synchronized native physics only; excludes startup, IPC and endpoint probe"
    predictor = ExternalContactPredictors(folder/"model.xml", settings)
    manifolds = [normalize_manifold(rows) for rows in geometry["contacts"]]
    batch = predictor.comfree_batch(samples, manifolds)
    timing["comfree_batch"] = dict(predictor.last_comfree_timing)
    serial_physics = 0.
    for index, sample_id in enumerate(ids):
        single = predictor.comfree_batch([samples[index]], [manifolds[index]])[0]
        serial_physics += predictor.last_comfree_timing.get("physics_seconds") or 0.
        if single["status"] != "ok" or batch[index]["status"] != "ok":
            checks.append(dict(check="comfree_batch_vs_serial", sample_id=sample_id, passed=False,
                               failure_reason=single.get("failure_reason") or batch[index].get("failure_reason")))
        else:
            _record_comparison(checks, model, "comfree_batch_vs_serial", single["qpos"], single["qvel"],
                               batch[index]["qpos"], batch[index]["qvel"], sample_id=sample_id)
    reverse = predictor.comfree_batch(samples[::-1], manifolds[::-1])
    repeat = predictor.comfree_batch(samples, manifolds)
    for index, sample_id in enumerate(ids):
        for title, outcome in (("comfree_batch_reordering", reverse[len(samples)-1-index]),
                               ("comfree_A_B_A_reset", repeat[index])):
            if outcome["status"] != "ok" or batch[index]["status"] != "ok":
                checks.append(dict(check=title, sample_id=sample_id, passed=False,
                                   failure_reason=outcome.get("failure_reason") or batch[index].get("failure_reason")))
            else:
                _record_comparison(checks, model, title, batch[index]["qpos"], batch[index]["qvel"],
                                   outcome["qpos"], outcome["qvel"], sample_id=sample_id)
    physics = timing["comfree_batch"].get("physics_seconds")
    timing["comfree_sum_serial_physics_seconds"] = serial_physics
    timing["comfree_physics_speedup_vs_serial_sum"] = serial_physics/physics if physics else None
    return dict(status="passed" if all(check["passed"] for check in checks) else "failed",
                strict_status="passed" if all(check.get("strict_passed", check["passed"]) for check in checks) else "failed",
                n_queries=len(samples), sample_ids=ids, checks=checks, timings=timing,
                observed_modes=dict(Counter(item["metadata"]["mode"] for item in available)),
                observed_contact_counts=dict(Counter(item["metadata"]["contact_group"] for item in available)),
                observed_geometries=dict(Counter(item["metadata"]["geometry"] for item in available)))


def _freeflight(folder, settings, python, batch_size):
    model = mujoco.MjModel.from_xml_path(str(folder/"model.xml"))
    spec = json.loads((folder/"physx_spec.json").read_text())
    ball_mass = float(model.body_mass[model.body("ball").id])
    forces, samples = [0., .05, .1, .2, .5], []
    for force in forces:
        q = model.qpos0.copy()
        q[:3], q[7:10] = [0., 0., 1.], [.3, 0., 1.]
        q[3:7] = q[10:14] = [1., 0., 0., 0.]
        wrench = np.zeros((3, 6)); wrench[2, 0] = force
        samples.append(dict(qpos=q, qvel=np.zeros(12), wrench=wrench))
    references, native_timing, native_meta = [], [], None
    with IsaacGymNative(spec, batch_size=batch_size, use_gpu=True, python=python,
                        log_path=folder/"audit_native.log") as native:
        native_meta = native.metadata
        if not native_meta.get("physx_gpu"):
            raise RuntimeError("Free-flight audit requires native GPU PhysX")
        for first in range(0, len(samples), batch_size):
            reference = native.reference_samples(samples[first:first+batch_size])
            native_timing.append(reference.get("timings", {}))
            references.extend(dict(qpos=q, qvel=v, snapshots=snapshots,
                                   time=reference["elapsed_seconds"],
                                   raw_contacts=(reference.get("raw_contacts") or reference.get("contacts_snapshots"))[i])
                              for i, (q, v, snapshots) in enumerate(zip(
                                  reference["qpos"], reference["qvel"], reference["snapshots"])))
    predictor = ExternalContactPredictors(folder/"model.xml", settings)
    empty = normalize_manifold([])
    comfree = predictor.comfree_batch(samples, [empty]*len(samples))
    records, checks = [], []
    g = np.asarray(spec["gravity"], float)
    for force, sample, reference, gpu in zip(forces, samples, references, comfree):
        acceleration = g + np.array([force/ball_mass, 0., 0.])
        expected_v = acceleration*.02
        native_expected_p = sample["qpos"][7:10] + acceleration*(.002**2*10*11/2)
        coarse_expected_p = sample["qpos"][7:10] + acceleration*.02**2
        method_records = {}
        for method, prediction, expected_p in (("isaacgym", reference, native_expected_p),
                ("scm", predictor.scm(sample, empty), coarse_expected_p), ("comfree", gpu, coarse_expected_p)):
            if prediction.get("status", "ok") != "ok":
                passed = False
                detail = dict(failure_reason=prediction.get("failure_reason"))
            else:
                v, q = np.asarray(prediction["qvel"]), np.asarray(prediction["qpos"])
                velocity_error = float(np.linalg.norm(v[6:9]-expected_v))
                position_error = float(np.linalg.norm(q[7:10]-expected_p))
                impulse = ball_mass*v[6]
                impulse_error = abs(impulse-force*.02)
                time_value = prediction.get("time", .02)
                passed = velocity_error <= 3e-6 and position_error <= 3e-6 and impulse_error <= 5e-8 and abs(time_value-.02) <= 1e-7
                detail = dict(velocity_error_m_s=velocity_error, position_error_m=position_error,
                              measured_horizontal_impulse_Ns=float(impulse), impulse_error_Ns=float(impulse_error),
                              time_s=float(time_value))
            method_records[method] = dict(passed=bool(passed), **detail)
            checks.append(dict(check="freeflight_force_and_horizon", method=method, force_N=force, passed=bool(passed)))
        no_contact = all(not rows for rows in reference["raw_contacts"])
        checks.append(dict(check="freeflight_native_contacts_empty", force_N=force, passed=no_contact))
        records.append(dict(force_N=force, expected_external_impulse_Ns=force*.02,
                            expected_delta_vx_m_s=force*.02/ball_mass, native_contacts_empty=no_contact,
                            methods=method_records))
    return dict(status="passed" if all(check["passed"] for check in checks) else "failed",
                checks=checks, records=records, native_metadata=native_meta, native_timings=native_timing,
                comfree_timings=predictor.last_comfree_timing, ball_mass_kg=ball_mass,
                note="Fine and coarse semi-implicit position formulas differ; total imposed impulse is identical")


def audit_dataset(output, isaacgym_python=None, batch_size=16):
    """Write audit.json and return it; failures stay explicit and nonzero-counted."""
    output = Path(output).resolve()
    began = time.perf_counter()
    report = dict(status="running", tolerances=TOLERANCES, objects={}, failures=[],
                  batch_size=int(batch_size), native_backend_required=True,
                  initial_contact_count_and_mode_equality_required=True,
                  full_contact_sequence="Strict supplementary diagnostic; all differences are retained")
    try:
        if batch_size < 2:
            raise ValueError("Parallel native audit requires batch_size >= 2")
        manifest = json.loads((output/"manifest.json").read_text())
        settings = Settings(**manifest["settings"])
        for name in manifest["objects"]:
            try:
                result = _audit_object(output/name, settings, isaacgym_python, batch_size)
                report["objects"][name] = result
                if result["status"] != "passed":
                    report["failures"].append(f"{name}: replay/batching/discrete-label comparisons failed")
            except Exception as exc:
                report["objects"][name] = dict(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
                report["failures"].append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"Native audit {name}: {report['objects'][name]['status']}", flush=True)
        names = manifest["objects"]
        if not names:
            raise ValueError("No objects in the saved dataset")
        name = "sphere" if "sphere" in names else names[0]
        try:
            report["freeflight"] = _freeflight(output/name, settings, isaacgym_python, batch_size)
            if report["freeflight"]["status"] != "passed":
                report["failures"].append("Free-flight force/horizon/native-contact audit failed")
        except Exception as exc:
            report["freeflight"] = dict(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
            report["failures"].append("Free-flight: " + report["freeflight"]["failure_reason"])
        modes, counts, geometries = Counter(), Counter(), Counter()
        for result in report["objects"].values():
            modes.update(result.get("observed_modes", {}))
            counts.update(result.get("observed_contact_counts", {}))
            geometries.update(result.get("observed_geometries", {}))
        report["coverage"] = dict(modes=dict(modes), contact_counts=dict(counts), geometries=dict(geometries),
            missing_primary_modes=sorted(set(("separation", "contact_establishment", "sticking", "sliding"))-set(modes)),
            missing_contact_counts=sorted(set(("0", "1", "2", ">=3"))-set(counts)),
            note="Coverage gaps in a frozen smoke subset are retained; no sample is relabeled to fill a cell")
    except Exception as exc:
        report["failures"].append(f"{type(exc).__name__}: {exc}")
    report["status"] = "failed" if report["failures"] else "passed"
    report["strict_status"] = ("failed" if report["failures"] or any(
        result.get("strict_status", result["status"]) != "passed" for result in report["objects"].values())
        else "passed")
    report["elapsed_seconds"] = time.perf_counter()-began
    report["n_failed"] = sum(not check["passed"] for result in report["objects"].values()
                              for check in result.get("checks", []))
    report["n_failed"] += sum(not check["passed"] for check in report.get("freeflight", {}).get("checks", []))
    report["n_exception_failures"] = sum("failure_reason" in result for result in report["objects"].values())
    report["n_strict_failed"] = sum(not check.get("strict_passed", check["passed"])
        for result in report["objects"].values() for check in result.get("checks", []))
    report["n_contact_sequence_differences"] = sum(not check.get("labels", {}).get("contact_count_sequence_equal", True)
        for result in report["objects"].values() for check in result.get("checks", []))
    (output/"audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report
