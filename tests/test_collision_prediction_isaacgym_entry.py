"""CPU checks of frozen source identities, serialization and error semantics."""
import csv
import hashlib
import json

import numpy as np
import pytest

from examples.mpc.fingertips.test import benchmark_collision_prediction_isaacgym as entry
from examples.mpc.fingertips.test.collision_prediction_core import Settings, make_data, world_velocity


def state():
    qpos = np.zeros(14)
    qpos[[3, 10]] = 1.
    qpos[2] = .03
    qpos[7:10] = [.1, 0, .03]
    return dict(qpos=qpos, qvel=np.arange(12, dtype=float)*.01, wrench=np.arange(18, dtype=float).reshape(3, 6)*.01)


def fixture_source(tmp_path):
    folder = tmp_path / "sphere"
    folder.mkdir()
    (folder / "queries").mkdir()
    (folder / "assets").mkdir()
    (folder / "assets" / "dummy.obj").write_text("asset")
    (folder / "model.xml").write_text("model")
    info = dict(model_sha256=entry.sha(folder / "model.xml"),
                packaged_asset_sha256={"assets/dummy.obj": entry.sha(folder / "assets" / "dummy.obj")})
    (folder / "object.json").write_text(json.dumps(info))
    (tmp_path / "manifest.json").write_text(json.dumps(dict(status="complete", settings=Settings().to_dict(), objects=["sphere"])))
    rows = []
    for i, mode in enumerate(entry.MODES):
        metadata = dict(sample_id=f"q{i}", object="sphere", mode=mode, requested_mode=mode,
                        contact_group=str(i) if i < 3 else ">=3", geometry="point", selected=True)
        np.savez_compressed(folder / "queries" / f"q{i}.npz", **state(), metadata_json=np.array(json.dumps(metadata)))
        for method in ("mjwarp", "scm", "comfree"):
            rows.append(dict(metadata, method=method, selected="True", status="ok"))
    rows.append(dict(rows[0], sample_id="not_selected", selected="False"))
    rows.append(dict(rows[0], sample_id="mixed", mode="mixed_transition", selected="False"))
    with (folder / "predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_source_selection_deduplicates_methods_preserves_source_labels_and_hash(tmp_path):
    source = fixture_source(tmp_path)
    path, manifest, settings, records = entry.load_source(source, objects=["sphere"])
    assert path == source.resolve()
    assert len(records["sphere"]) == 4
    assert [item["mode"] for item in records["sphere"]] == list(entry.MODES)
    sample, metadata = entry.read_source_query(path, "sphere", records["sphere"][0])
    for key, value in state().items():
        np.testing.assert_array_equal(sample[key], value)
    assert metadata["source_mode"] == "separation"
    assert metadata["source_contact_group"] == "0"
    assert metadata["selected"] is True
    assert metadata["source_query_sha256"] == hashlib.sha256((path / "sphere/queries/q0.npz").read_bytes()).hexdigest()
    # New PhysX classifications are independent of the source labels.
    metadata.update(mode="mixed_transition", contact_group=">=3")
    assert metadata["source_mode"] == "separation" and metadata["source_contact_group"] == "0"


def test_source_rejects_corrupt_assets_or_model_and_wrong_horizon(tmp_path):
    source = fixture_source(tmp_path)
    asset = source / "sphere/assets/dummy.obj"
    asset.write_text("changed")
    with pytest.raises(ValueError, match="Corrupt source asset"):
        entry.load_source(source, objects=["sphere"])
    asset.write_text("asset")
    (source / "sphere/model.xml").write_text("changed")
    with pytest.raises(ValueError, match="Corrupt source model"):
        entry.load_source(source, objects=["sphere"])
    (source / "sphere/model.xml").write_text("model")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["settings"]["reference_steps"] = 1
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Source horizon"):
        entry.load_source(source, objects=["sphere"])


def test_source_full_dataset_cardinality_objects_and_csv_npz_agreement(tmp_path):
    source = fixture_source(tmp_path)
    with pytest.raises(ValueError, match="exactly 8000"):
        entry.load_source(source)
    with pytest.raises(ValueError, match="distinct names"):
        entry.load_source(source, objects=["sphere", "sphere"])
    with pytest.raises(ValueError, match="distinct names"):
        entry.load_source(source, objects=["cube"])
    path, _, _, records = entry.load_source(source, objects=["sphere"], limit=2)
    assert len(records["sphere"]) == 2
    wrong = dict(records["sphere"][0], mode="not_the_saved_mode")
    with pytest.raises(ValueError, match="CSV/NPZ disagreement"):
        entry.read_source_query(path, "sphere", wrong)


def test_queries_save_all_inputs_and_failures_without_pickle_and_atomic_temp(tmp_path):
    sample = state()
    metadata = dict(sample_id="q", object="sphere", mode="mixed_transition", contact_group="2",
                    source_mode="sliding", source_contact_group="1", selected=True)
    manifold = dict(pair=[0], position=[[0., 0., 0.]], gap=[-.0001], shape_id=None)
    predictions = {"scm": dict(qpos=sample["qpos"]+.01, qvel=sample["qvel"]+.02, status="ok"),
                   "comfree": entry.failure("nonfinite prediction")}
    reference_qpos = np.tile(sample["qpos"], (11, 1))
    reference_qvel = np.tile(sample["qvel"], (11, 1))
    path = tmp_path / "q.npz"
    entry.save_query(path, sample, metadata, predictions, reference_qpos, reference_qvel, manifold)
    arrays, loaded, saved_metadata, saved_manifold = entry.load_saved_query(path)
    assert saved_metadata == metadata and saved_manifold == manifold
    for key in sample:
        np.testing.assert_array_equal(sample[key], loaded[key])
    np.testing.assert_array_equal(arrays["reference_qpos"], reference_qpos)
    np.testing.assert_array_equal(arrays["reference_qvel"], reference_qvel)
    np.testing.assert_allclose(arrays["reference_times"], np.arange(11)*.002, atol=1e-15)
    assert arrays["reference_times"][-1] == .02
    assert np.isnan(arrays["comfree_qpos"]).all() and np.isnan(arrays["isaacgym_qvel"]).all()
    assert not path.with_suffix(".tmp").exists()
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            assert archive[key].dtype.kind != "O"


def model():
    return entry.mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="object"><freejoint/><inertial pos=".003 .004 0" mass=".01" diaginertia=".0001 .0002 .00025"/>
        <geom type="box" size=".03 .03 .03"/></body>
      <body name="ball"><freejoint/><geom type="sphere" size=".01" mass=".01"/></body>
    </worldbody></mujoco>''')


def test_world_velocity_matches_existing_metric_convention_and_pose_known_errors():
    m = model()
    sample = state()
    sample["qpos"][3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    original = make_data(m, **sample)
    np.testing.assert_allclose(entry.state_velocity(m, sample["qpos"], sample["qvel"]),
                               world_velocity(m, original), atol=1e-13)
    reference_q = sample["qpos"].copy()
    prediction = dict(qpos=reference_q.copy(), qvel=sample["qvel"].copy())
    prediction["qpos"][:3] += [.003, .004, 0]
    prediction["qpos"][3:7] = [1., 0, 0, 0]
    values = entry.metrics(m, sample, prediction, reference_q, sample["qvel"], .1)
    assert values["position_error_mm"] == pytest.approx(5.)
    assert values["orientation_error_deg"] == pytest.approx(90.)
    assert values["relative_position_error"] is None
    assert values["relative_orientation_error"] is None
    assert values["quaternion_error"] == pytest.approx(.5)
    prediction["qpos"][3:7] *= -1
    assert entry.metrics(m, sample, prediction, reference_q, sample["qvel"], .1) == pytest.approx(values)


def test_appended_rows_keep_native_and_source_labels_failure_reason_and_relative_denominators(tmp_path):
    folder = tmp_path / "sphere"
    folder.mkdir()
    metadata = dict(sample_id="q", object="sphere", mode="contact_establishment", contact_group="2",
                    geometry="point", source_mode="sticking", source_contact_group="1", selected=True)
    ok = dict(status="ok", qpos=state()["qpos"], qvel=state()["qvel"],
              position_error_mm=1.5, relative_position_error=.3, relative_position_error_denominator=.005,
              seconds=.01, failure_reason="")
    entry.append_predictions(folder, metadata, {"scm": ok, "comfree": entry.failure("GPU failure")})
    with (folder / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and tuple(rows[0]) == entry.ROW_FIELDS
    assert rows[0]["mode"] == "contact_establishment" and rows[0]["source_mode"] == "sticking"
    assert rows[0]["contact_group"] == "2" and rows[0]["source_contact_group"] == "1"
    assert float(rows[0]["physics_seconds"]) == .01
    assert rows[1]["position_error_mm"] == "" and rows[1]["failure_reason"] == "GPU failure"
    records = [json.loads(line) for line in (tmp_path / "results.log").read_text().splitlines()]
    assert records[0]["absolute_errors"]["position_error_mm"] == 1.5
    assert records[0]["relative_errors"]["relative_position_error"] == .3
    assert records[0]["relative_denominators"]["relative_position_error"] == .005
    assert records[1]["absolute_errors"]["position_error_mm"] is None
