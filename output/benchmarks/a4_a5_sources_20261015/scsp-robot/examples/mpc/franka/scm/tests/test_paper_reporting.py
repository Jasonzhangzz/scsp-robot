"""Scientific quantity selection for Section IV tables/trajectories."""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_env import activate_dependencies
activate_dependencies()

import numpy as np
import pytest
from fidelity_metrics import Moments
from paper_reporting import physics_series, write_physics_report


def test_torsion_trajectory_preserves_velocity_reversal():
    trajectory = {"qvel": np.array([[3.], [1.], [-.2]])}
    series = physics_series("ball_rotation", trajectory)
    np.testing.assert_array_equal(series["angular_velocity_z_rad_s"], [3, 1, -.2])
    with pytest.raises(ValueError, match="isolated"):
        physics_series("ball_rotation", {"qvel": np.ones((3, 2))})


def test_depth_table_uses_individual_contact_mean_std_not_frame_maxima(tmp_path):
    moments = Moments()
    moments.add([1])
    moments.add([3, 5, 7])
    summary = {"scm": {"penetration_all_contacts_mm": moments.report(), "max_penetration_m": 999,
                       "completed_steps": 2, "requested_steps": 2, "status": "completed"}}
    metadata = {"arguments": {"scene": "primitives", "contact_stiffness": .1, "contact_damping": .001}, "nworld": 1}
    write_physics_report(tmp_path, {"scm": {"time": np.array([0, .002, .004])}}, summary, metadata)
    with (tmp_path / "penetration_mean_std.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert int(rows[0]["contact_samples"]) == 4
    assert float(rows[0]["mean_mm"]) == 4
    assert float(rows[0]["std_mm"]) == pytest.approx(np.sqrt(5))
    assert "±" in rows[0]["depth_mm_mean_pm_std"]


def test_legacy_frame_maxima_cannot_be_used_as_paper_contact_statistics(tmp_path):
    summary = {"mjwarp": {"max_penetration_m": 999, "completed_steps": 2, "requested_steps": 2}}
    metadata = {"arguments": {"scene": "primitives", "contact_stiffness": .1, "contact_damping": .001}}
    write_physics_report(tmp_path, {"mjwarp": {"time": np.array([0, .002, .004])}}, summary, metadata)
    with (tmp_path / "penetration_mean_std.csv").open() as f:
        assert list(csv.DictReader(f)) == []
    report = json.loads((tmp_path / "paper_metrics.json").read_text())
    assert report["throughput"]["status"].startswith("not measured")
