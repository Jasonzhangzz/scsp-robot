"""Run in a fresh process with the benchmark's pinned MuJoCo dependencies."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()
import mujoco

from examples.mpc.fingertips.test import collision_prediction_assets as assets
from examples.mpc.fingertips.test.benchmark_collision_prediction import case_counts, statistics, summarize
from examples.mpc.fingertips.test.collision_prediction_core import (
    Experiment, METHODS, METRICS, Settings, classify_contacts, native_diagnostics, pose_errors, relative_errors)
from examples.mpc.fingertips.test.collision_prediction_batch import BatchedMjwarpReference
from examples.mpc.fingertips.test.collision_prediction_smoke import floor_geometry


def snapshot(pairs=(), slip=0., load=1.):
    return dict(pair=list(pairs), force=[[load, 0, 0, 0, 0, 0] for _ in pairs],
                relative_velocity=[[0, slip, 0, 0, 0, 0] for _ in pairs])


@pytest.mark.parametrize("snapshots,expected", [
    ([snapshot(), snapshot()], "separation"),
    ([snapshot([0]), snapshot([0, 1])], "contact_establishment"),
    ([snapshot([0, 1]), snapshot([0])], "separation"),
    ([snapshot([0, 1]), snapshot([0, 1])], "sticking"),
    ([snapshot([0]), snapshot([0], slip=.1)], "sliding"),
    ([snapshot([0]), snapshot([0, 1]), snapshot([0])], "mixed_transition"),
])
def test_measured_modes(snapshots, expected):
    assert classify_contacts(snapshots)["mode"] == expected


def test_contact_count_is_initial_geometric_points_not_load_or_constraint_rows():
    result = classify_contacts([snapshot([0, 0], load=0.), snapshot([0, 0, 1, 1])])
    assert result["ncontact_start"] == 2
    assert result["contact_group"] == "2"
    assert result["ncontact_floor_start"] == 2
    assert result["ncontact_ball_start"] == 0
    assert result["ncontact_max"] == 4
    assert classify_contacts([snapshot(), snapshot([1])])["contact_group"] == "0"
    assert classify_contacts([snapshot([0, 0, 0])])["contact_group"] == ">=3"


def test_pose_metrics_known_translation_rotation_and_quaternion_sign():
    reference = np.array([0., 0, 0, 1, 0, 0, 0])
    predicted = np.array([.003, .004, 0, np.sqrt(.5), 0, 0, np.sqrt(.5)])
    result = pose_errors(predicted, reference, .1)
    assert result["position_error_mm"] == pytest.approx(5.)
    assert result["orientation_error_deg"] == pytest.approx(90.)
    assert result["quaternion_error"] == pytest.approx(.5)
    assert result["pose_error"] == pytest.approx(np.hypot(.05, np.pi/2))
    predicted[3:] *= -4.
    assert pose_errors(predicted, reference, .1) == pytest.approx(result)
    with pytest.raises(ValueError):
        pose_errors(np.zeros(7), reference, .1)


def test_failed_predictions_remain_in_denominator_and_empty_cells_are_null():
    base = dict(object="sphere", mode="sticking", contact_group="1", selected="True", method="scm")
    rows = [{**base, "status": "ok", **{metric: "2" for metric in METRICS}},
            {**base, "status": "failed", **{metric: "" for metric in METRICS}}]
    result = summarize(rows, ["sphere"])
    cell = next(r for r in result if r["object"] == "sphere" and r["mode"] == "sticking"
                and r["contact_group"] == "1" and r["method"] == "scm" and r["metric"] == "pose_error")
    assert (cell["n_valid"], cell["n_total"], cell["n_failed"]) == (1, 2, 1)
    assert cell["mean"] == 2.
    empty = next(r for r in result if r["object"] == "sphere" and r["method"] == "comfree")
    assert empty["mean"] is None and empty["n_total"] == 0
    assert statistics([])["mean"] is None


def test_relative_errors_use_reference_motion_and_mark_zero_denominators_undefined():
    absolute = dict(position_error_mm=5., orientation_error_deg=5., pose_error=.1,
                    linear_velocity_error_m_s=.2, angular_velocity_error_rad_s=.5)
    movement = dict(position_error_mm=10., orientation_error_deg=10., pose_error=.2)
    result = relative_errors(absolute, movement, np.array([2., 0, 0, 0, 0, 5.]))
    assert result["relative_position_error"] == pytest.approx(.5)
    assert result["relative_orientation_error"] == pytest.approx(.5)
    assert result["relative_pose_error"] == pytest.approx(.5)
    assert result["relative_linear_velocity_error"] == pytest.approx(.1)
    assert result["relative_angular_velocity_error"] == pytest.approx(.1)
    zero = relative_errors(absolute, dict(position_error_mm=0., orientation_error_deg=0., pose_error=0.), np.zeros(6))
    assert zero["relative_position_error"] is None
    assert zero["relative_orientation_error"] is None
    assert zero["relative_linear_velocity_error"] is None
    assert zero["relative_position_error_denominator"] == 0.


def test_joint_count_table_counts_queries_once_and_retains_empty_cases():
    base = dict(sample_id="query", object="cube", mode="sticking", contact_group="2", geometry="edge",
                selected="True", status="ok")
    counts = case_counts([dict(base, method=method) for method in METHODS], ["cube"], 200)
    marginal = next(row for row in counts if row["mode"] == "sticking" and row["grouping"] == "object_mode")
    assert marginal["n_selected"] == 1 and marginal["shortfall"] == 199
    empty = next(row for row in counts if row["mode"] == "sticking" and row["grouping"] == "object_mode_count"
                 and row["contact_group"] == "1")
    assert empty["n_selected"] == 0 and empty["status"] == "no_samples_unverified_feasibility"


def test_coacd_cache_and_multigeom_do_not_change_inertia(tmp_path, monkeypatch):
    import coacd
    import trimesh
    calls = []

    def fake_decomposition(mesh, **parameters):
        calls.append(parameters)
        hull = trimesh.Trimesh(mesh.vertices, mesh.indices).convex_hull
        # Duplicate overlapping pieces would double mass without explicit inertia.
        return [(hull.vertices, hull.faces), (hull.vertices, hull.faces)]

    monkeypatch.setattr(coacd, "run_coacd", fake_decomposition)
    path = assets.prepare_asset("elephant", tmp_path / "cache")
    assert assets.prepare_asset("elephant", tmp_path / "cache") == path
    assert len(calls) == 1
    assert calls[0]["real_metric"] and calls[0]["threshold"] == .001
    xml, info = assets.scene_xml("elephant", path)
    packaged = assets.package_scene(xml, tmp_path / "scene")
    model = mujoco.MjModel.from_xml_path(str(packaged))
    body = model.body("object").id
    assert model.body_mass[body] == pytest.approx(info["mass"])
    np.testing.assert_allclose(model.body_ipos[body], info["ipos"])
    np.testing.assert_allclose(model.body_inertia[body], info["inertia"])
    np.testing.assert_allclose(model.body_iquat[body], info["iquat"])
    assert model.geom_contype[model.geom("object_visual").id] == 0
    assert np.sum((model.geom_bodyid == body) & (model.geom_contype != 0)) == 2
    vertices = assets.object_vertices(model)
    np.testing.assert_allclose([vertices.min(0), vertices.max(0)], info["bounds"], atol=1e-6)
    manifest = json.loads((path / "asset.json").read_text())
    (path / next(iter(manifest["part_sha256"]))).write_text("corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        assets.prepare_asset("elephant", tmp_path / "cache")


def test_decomposition_failure_does_not_fall_back(tmp_path, monkeypatch):
    import coacd
    monkeypatch.setattr(coacd, "run_coacd", lambda *a, **k: [])
    with pytest.raises(RuntimeError, match="no parts"):
        assets.prepare_asset("elephant", tmp_path)


def test_native_warnings_are_captured_and_stdout_is_restored_on_failure(capfd):
    with pytest.raises(RuntimeError):
        with native_diagnostics() as warnings:
            os.write(1, b"Warning: opt.ccd_iterations reached limit\n")
            raise RuntimeError("backend failed")
    assert warnings == ["Warning: opt.ccd_iterations reached limit"]
    print("restored")
    assert "restored" in capfd.readouterr().out


def test_top_ray_reaches_concave_bottom_below_body_origin():
    xml = '''<mujoco><worldbody>
      <body name="object"><freejoint/>
        <geom type="box" size=".03 .03 .001" pos="0 0 -.03" mass=".01"/>
        <geom type="box" size=".003 .03 .03" pos=".03 0 0" mass=".001"/>
        <geom type="box" size=".003 .03 .03" pos="-.03 0 0" mass=".001"/>
      </body><body name="ball" pos=".3 0 .1"><freejoint/>
        <geom name="ball_geom" type="sphere" size=".01" mass=".01"/>
      </body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    experiment = Experiment.__new__(Experiment)
    experiment.models = {"mjwarp": model}
    experiment.settings = Settings()
    experiment.info = {"length_scale_m": .12}
    experiment.vertices = assets.object_vertices(model)
    sample = experiment.sample("cube", "contact_establishment", np.random.default_rng(42), 0)
    assert sample["qpos"][9] < sample["qpos"][2]
    assert sample["speed_level_m_s"] == 0.


@pytest.mark.parametrize("points,expected", [
    ([], "none"), ([[0, 0, 0]], "point"), ([[0, 0, 0], [0, 0, 0]], "point"),
    ([[0, 0, 0], [.03, 0, 0]], "line"),
    ([[0, 0, 0], [.03, 0, 0], [0, .03, 0]], "face"),
])
def test_geometry_uses_actual_floor_positions_not_point_count(points, expected):
    # A far-away ball-contact point must not affect the floor's affine rank.
    frame = dict(position=points + [[1, 2, 3]], pair=[0]*len(points) + [1])
    assert floor_geometry(frame) == expected


def test_batch_contacts_decode_world_local_force_addresses():
    class Array:
        def __init__(self, values):
            self.values = np.asarray(values)

        def numpy(self):
            return self.values

    # Pyramidal edges have separate atomic addresses; they are NOT contiguous.
    contact = SimpleNamespace(worldid=Array([1, 0, 1, 0]),
                              efc_address=Array([[0, 2, 4, 6], [3, 1, 0, 2], [1, 3, 5, 7], [-1]*4]),
                              dim=Array([3]*4), friction=Array([[.5]*5]*4),
                              dist=Array([-1e-5]*4), pos=Array(np.zeros((4, 3))),
                              frame=Array(np.tile(np.eye(3), (4, 1, 1))), geom=Array([[0, 1]]*4))
    data = SimpleNamespace(nacon=Array([4]), ncollision=Array([4]), nefc=Array([4, 8]),
                           naconmax=8, njmax=8, contact=contact,
                           efc=SimpleNamespace(force=Array([[1, 2, 3, 4, 5, 6, 7, 8],
                                                            [10, 20, 30, 40, 50, 60, 70, 80]])))
    engine = BatchedMjwarpReference.__new__(BatchedMjwarpReference)
    native, worlds, _ = engine.native_contacts(data)
    np.testing.assert_array_equal(worlds, [1, 0, 1, 0])
    np.testing.assert_allclose(native["force"][:, 0], [160, 10, 200, 0])
    np.testing.assert_allclose(native["force"][:, 1], [-10, 1, -10, 0])
    np.testing.assert_allclose(native["force"][:, 2], [-10, -1, -10, 0])
    data.nefc = Array([4, 9])
    with pytest.raises(RuntimeError, match="overflow"):
        engine.native_contacts(data)


def test_batch_warning_retries_worlds_without_invalidating_healthy_queries():
    engine = BatchedMjwarpReference.__new__(BatchedMjwarpReference)
    engine.batches = []

    def failed_batch(samples):
        raise RuntimeError("unattributed GPU batch warning")

    def serial(sample):
        if sample["bad"]:
            os.write(1, b"Warning: opt.ccd_iterations reached limit\n")
        return (None, [], np.zeros((11, 14)), np.zeros((11, 12)))

    engine.rollout = failed_batch
    engine.experiment = SimpleNamespace(reference=serial)
    outcomes = engine.evaluate([dict(bad=False), dict(bad=True)])
    assert outcomes[0]["error"] is None
    assert "native collision warning" in outcomes[1]["error"]
    assert engine.batches[0]["serial_retries"] == 2


def test_batch_warning_splits_and_keeps_healthy_gpu_worlds_in_order():
    engine = BatchedMjwarpReference.__new__(BatchedMjwarpReference)
    engine.batches = []

    def rollout(samples):
        if len(samples) > 2:
            raise RuntimeError("batch collision warning")
        return ([dict(execution="gpu_batch", error=None, query=sample) for sample in samples], .01, .02)

    engine.rollout = rollout
    queries = list(range(8))
    results = engine.evaluate(queries)
    assert [result["query"] for result in results] == queries
    assert engine.batches[0]["execution"] == "split_retry"
    assert all(record["serial_retries"] == 0 for record in engine.batches)
    assert sum(record["nworld"] for record in engine.batches if record["execution"] == "gpu_batch") == 8


@pytest.mark.skipif(not os.getenv("SCSP_COLLISION_GPU_TEST"), reason="opt-in CUDA integration test")
def test_gpu_horizon_identical_inputs_reset_and_contact_filter(tmp_path):
    xml, info = assets.scene_xml("sphere")
    model_path = assets.package_scene(xml, tmp_path)
    experiment = Experiment(model_path, info, Settings())
    rng = np.random.default_rng(42)
    sample = experiment.sample("sphere", "contact_establishment", rng, 0)
    original = {key: sample[key].copy() for key in ("qpos", "qvel", "wrench")}
    reference, snapshots, qpos, qvel = experiment.reference(sample)
    assert reference.time == pytest.approx(.02, abs=1e-7)
    assert qpos.shape == (11, 14) and qvel.shape == (11, 12)
    assert len(snapshots) == 11
    for frame in snapshots:
        for geom in frame["geom"]:
            assert experiment.models["mjwarp"].body("object").id in experiment.models["mjwarp"].geom_bodyid[geom]
    first = experiment.predict(sample, reference)
    second = experiment.predict(sample, experiment.reference(sample)[0])
    for method in METHODS:
        assert first[method]["status"] == "ok"
        np.testing.assert_array_equal(first[method]["qpos"], second[method]["qpos"])
        np.testing.assert_array_equal(experiment.models[method].body_mass,
                                      experiment.models["mjwarp"].body_mass)
        np.testing.assert_array_equal(experiment.models[method].body_inertia,
                                      experiment.models["mjwarp"].body_inertia)
    assert all(first["mjwarp"][metric] == 0 for metric in METRICS)
    for key in original:
        np.testing.assert_array_equal(sample[key], original[key])
    # An actual loaded ball-floor contact must not count as an object contact.
    floor_sample = {key: value.copy() for key, value in original.items()}
    floor_sample["qpos"][2] = .3
    floor_sample["qpos"][9] = .00998
    airborne, frames, _, _ = experiment.reference(floor_sample)
    assert classify_contacts(frames)["ncontact_start"] == 0
    assert all(not frame["pair"] for frame in frames)
    native = experiment.backends["mjwarp"].solver_contacts(airborne)
    assert np.any(np.asarray(native["force"])[:, 0] > experiment.settings.force_threshold)
    free_predictions = experiment.predict(floor_sample, airborne)
    # Independent semi-implicit Euler result: g*H^2*(1-1/N)/2.
    expected_mm = 1000 * 9.81 * .02**2 * (1-1/10) / 2
    assert free_predictions["scm"]["position_error_mm"] == pytest.approx(expected_mm, abs=2e-5)


@pytest.mark.skipif(not os.getenv("SCSP_COLLISION_GPU_TEST"), reason="opt-in CUDA integration test")
def test_gpu_native_world_batch_matches_serial_and_isolates_forces(tmp_path):
    xml, info = assets.scene_xml("sphere")
    experiment = Experiment(assets.package_scene(xml, tmp_path), info, Settings())
    rng = np.random.default_rng(87)
    samples = [experiment.sample("sphere", mode, rng, attempt) for mode, attempt in
               [("separation", 0), ("contact_establishment", 3), ("sticking", 2), ("sliding", 4)]]
    # Both upload paths must preserve the same requested initial quaternion.
    samples[0]["qpos"][3:7] *= 1.01
    original = [sample["qpos"].copy() for sample in samples]
    outcomes = experiment.reference_batch(samples)
    assert len(outcomes) == 4
    for sample, result in zip(samples, outcomes):
        assert result["execution"] == "gpu_batch" and result["error"] is None
        serial = experiment.reference(sample)
        np.testing.assert_allclose(result["reference"][2], serial[2], atol=5e-6, rtol=0.)
        np.testing.assert_allclose(result["reference"][3], serial[3], atol=5e-3, rtol=0.)
        assert classify_contacts(result["reference"][1])["mode"] == classify_contacts(serial[1])["mode"]
        np.testing.assert_array_equal(result["reference"][2][0], serial[2][0])
    for sample, previous in zip(samples, original):
        np.testing.assert_array_equal(sample["qpos"], previous)
    changed = [{**sample, "wrench": sample["wrench"].copy()} for sample in samples]
    changed[1]["wrench"] *= 2.
    repeated = experiment.reference_batch(changed)
    for world in (0, 2, 3):
        assert repeated[world]["execution"] == "gpu_batch"
        np.testing.assert_allclose(repeated[world]["reference"][2], outcomes[world]["reference"][2], atol=5e-6, rtol=0.)
    assert np.max(np.abs(repeated[1]["reference"][3]-outcomes[1]["reference"][3])) > 1e-4


@pytest.mark.skipif(not os.getenv("SCSP_COLLISION_GPU_TEST"), reason="opt-in CUDA integration test")
def test_gpu_decoded_contact_forces_obey_target_linear_momentum(tmp_path):
    # Check the decoder against the actual integrated momentum, independently
    # of the serial decoder. Includes floor, lateral ball and friction forces.
    xml, info = assets.scene_xml("cube")
    experiment = Experiment(assets.package_scene(xml, tmp_path), info, Settings())
    model = experiment.models["mjwarp"]
    obj, ball = model.body("object").id, model.body("ball").id
    samples = []
    for i in range(8):
        qpos = model.qpos0.copy()
        qpos[:3], qpos[7:10] = [0, 0, .02999], [.03999, 0, .03]
        qvel, wrench = np.zeros(model.nv), np.zeros((model.nbody, 6))
        qvel[6:9] = [-.2, .02*i, 0]
        wrench[ball, :3] = [-.2, .05, 0]
        samples.append(dict(qpos=qpos, qvel=qvel, wrench=wrench))
    for outcome in experiment.reference_batch(samples):
        assert outcome["execution"] == "gpu_batch" and outcome["error"] is None
        _, frames, _, velocities = outcome["reference"]
        frame = frames[0]
        geom = np.asarray(frame["geom"], int)
        assert 0 in frame["pair"] and 1 in frame["pair"]
        force_world = np.einsum("nji,nj->ni", np.asarray(frame["frame"]), np.asarray(frame["force"])[:, :3])
        sign = np.where(model.geom_bodyid[geom[:, 1]] == obj, 1., -1.)
        net_force = (sign[:, None]*force_world).sum(0) + model.body_mass[obj]*model.opt.gravity
        momentum_rate = model.body_mass[obj]*(velocities[1, :3]-velocities[0, :3])/.002
        np.testing.assert_allclose(net_force, momentum_rate, atol=2e-6, rtol=1e-5)


@pytest.mark.skipif(not os.getenv("SCSP_COLLISION_GPU_TEST"), reason="opt-in CUDA integration test")
def test_gpu_twenty_ms_horizon_and_equal_external_force_impulse(tmp_path):
    from examples.mpc.fingertips.test.audit_collision_horizon_impulse import runtime_audit
    report = runtime_audit(tmp_path)
    assert report["actual_mjwarp_batch_step_calls"] == 10
    assert report["actual_mjwarp_serial_step_calls_per_query"] == 10
    assert report["horizon_s"] == .02
    assert max(row["velocity_error"] for row in report["free_flight_velocity_checks"]) < 2e-6
