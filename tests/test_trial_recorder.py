"""Shared trial records used by the Isaac baseline and MPC ablations."""
import json
from pathlib import Path

import numpy as np

from utils.metrics import TrialRecorder, comparison_pose_error, open_trial_recorder


def test_comparison_error_matches_plot_weights():
    goal = np.array([0.1, 0.2, 0.3])
    pose = np.array([0.1, 0.25, 0.3])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    goal_quat = np.array([0.0, 1.0, 0.0, 0.0])
    position_cost = float(np.sum((pose - goal) ** 2))
    assert comparison_pose_error(pose, quat, goal, goal_quat) == 500.0 * position_cost + 5.0


def test_trial_recorder_writes_plot_inputs_and_scalars(tmp_path):
    recorder = TrialRecorder(tmp_path, "a1", "elephant", 2,
                             [0.4, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    recorder.observe(0, [0.1, 0.0, 0.2, 0.0, 1.0, 0.0, 0.0], [0.0, -0.7, 0.0, -2.3, 0.0, 1.5, 0.8], sim_time=0.0)
    recorder.note_planning(0.25)
    recorder.observe(1, [0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], [0.1, -0.7, 0.0, -2.3, 0.0, 1.5, 0.8], sim_time=0.02)
    record = recorder.finish(success=True, execution_steps=1, policy_interval_s=0.02)

    trial_dir = Path(tmp_path) / "a1" / "elephant" / "002"
    traj = json.loads((trial_dir / "traj.json").read_text())
    trails = json.loads((trial_dir.parent / "trails.json").read_text())
    assert traj["object_pose_wxyz"][0]["xyz"] == [0.1, 0.0, 0.2]
    assert trails[0]["goal_xyz"] == [0.4, 0.0, 0.2]
    assert record["success"] is True
    assert record["planning_time_total_s"] == 0.25
    assert record["execution_time_s"] == 0.02
    assert record["samples"][0]["robot_q"][1] == -0.7
    assert record["position_error"][1] == 0.0
    assert record["error"][0] == comparison_pose_error(
        [0.1, 0.0, 0.2], [0.0, 1.0, 0.0, 0.0], [0.4, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])

    again = TrialRecorder(tmp_path, "a1", "elephant", 2,
                          [0.4, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0])
    again.observe(0, [0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], None, sim_time=0.0)
    again.finish(success=False, execution_steps=0)
    trails = json.loads((trial_dir.parent / "trails.json").read_text())
    assert len(trails) == 1
    assert trails[0]["init_xyz"] == [0.2, 0.0, 0.2]
    lines = (trial_dir.parent / "trials.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_open_trial_recorder_is_opt_in():
    class Args:
        trial_output = ""
        obj = "mug"
        ablation_mode = "a3"

    assert open_trial_recorder(Args(), 0, [0, 0, 0], [1, 0, 0, 0]) is None
    Args.trial_output = "/tmp/unused-trial-output"
    recorder = open_trial_recorder(Args(), 4, [0, 0, 0], [1, 0, 0, 0])
    assert recorder.method == "a3"
    assert recorder.trial == 4
