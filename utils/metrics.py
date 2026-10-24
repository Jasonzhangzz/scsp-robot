import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path
import pickle

from utils import rotations


def comp_pos_error(curr_pos, targ_pos):
    return np.linalg.norm(curr_pos - targ_pos)


def comp_quat_error(curr_quat, targ_quat):
    return 1 - np.dot(curr_quat, targ_quat) ** 2


# Same weights as metrics/plot_comparision.py: 500 * ||p - p*||^2 + 5 * (1 - (q·q*)^2).
COMPARISON_POSITION_WEIGHT = 500.0
COMPARISON_QUATERNION_WEIGHT = 5.0


def _as_float_list(value, size=None):
    array = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and array.size != size:
        raise ValueError(f"expected {size} values, got {array.size}")
    return [float(item) for item in array]


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def comparison_pose_error(xyz, quat_wxyz, goal_xyz, goal_quat_wxyz):
    """Pose error consumed by metrics/plot_comparision.py.

    Position cost is the squared Euclidean error. Orientation cost is
    ``1 - (q · q*)^2`` after both quaternions are normalized.
    """
    position = np.asarray(xyz, dtype=float).reshape(3)
    goal_position = np.asarray(goal_xyz, dtype=float).reshape(3)
    orientation = _normalize_quaternion(quat_wxyz)
    goal_orientation = _normalize_quaternion(goal_quat_wxyz)
    if orientation is None or goal_orientation is None:
        return float("nan")
    position_cost = float(np.sum((position - goal_position) ** 2))
    dot = float(np.clip(np.dot(orientation, goal_orientation), -1.0, 1.0))
    quaternion_cost = 1.0 - dot ** 2
    return COMPARISON_POSITION_WEIGHT * position_cost + COMPARISON_QUATERNION_WEIGHT * quaternion_cost


def _normalize_quaternion(quat_wxyz):
    quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return quat / norm


def robot_q_from_observation(obs):
    """Franka joint positions stored after the 7-D object pose in ``full_q``."""
    if not isinstance(obs, dict) or obs.get("full_q") is None:
        return None
    full_q = np.asarray(obs["full_q"], dtype=float).reshape(-1)
    if full_q.size <= 7:
        return None
    return _as_float_list(full_q[7:])


def open_trial_recorder(args, trial, target_position, target_quaternion, method=None):
    """Return the shared trial writer, or None when statistics are not requested."""
    root = str(getattr(args, "trial_output", "") or "").strip()
    if not root:
        return None
    if method is None:
        method = getattr(args, "ablation_mode", None) or "isaac"
    return TrialRecorder(
        root,
        method,
        getattr(args, "obj", "object"),
        trial,
        target_position,
        target_quaternion,
    )


class TrialRecorder:
    """One trial record shared by the Isaac baseline and the MPC ablations.

    ``--trial-output`` selects this branch. Each trial writes the goal entry
    ``plot_comparision.py`` reads from ``trails.json``, the object trajectory
    it reads from ``traj.json``, and ``metrics.json`` with robot joints,
    comparison / position / orientation errors, planning time, execution time,
    and success.
    """

    def __init__(self, output_dir, method, object_name, trial, goal_xyz, goal_quat_wxyz):
        self.output_dir = path.abspath(output_dir)
        self.method = str(method)
        self.object_name = str(object_name)
        self.trial = int(trial)
        self.goal_xyz = _as_float_list(goal_xyz, 3)
        self.goal_quat_wxyz = _as_float_list(goal_quat_wxyz, 4)
        self.samples = []

    def observe(self, step, object_pose, robot_q=None, sim_time=None):
        pose = np.asarray(object_pose, dtype=float).reshape(-1)
        if pose.size < 7:
            raise ValueError("object pose must be xyz plus quat_wxyz")
        xyz = _as_float_list(pose[:3], 3)
        quat = _as_float_list(pose[3:7], 4)
        position_error = _finite_or_none(comp_pos_error(pose[:3], self.goal_xyz))
        orientation_error = _finite_or_none(comp_quat_error(pose[3:7], self.goal_quat_wxyz))
        error = _finite_or_none(comparison_pose_error(xyz, quat, self.goal_xyz, self.goal_quat_wxyz))
        self.samples.append({
            "step": int(step),
            "xyz": xyz,
            "quat_wxyz": quat,
            "robot_q": None if robot_q is None else _as_float_list(robot_q),
            "error": error,
            "position_error": position_error,
            "orientation_error": orientation_error,
            "planning_time_s": None,
            "sim_time_s": None if sim_time is None else _finite_or_none(sim_time),
        })

    def note_planning(self, planning_time_s):
        if not self.samples:
            return
        self.samples[-1]["planning_time_s"] = _finite_or_none(planning_time_s)

    def finish(self, success, aborted=False, execution_steps=None, policy_interval_s=0.02):
        planning_times = [sample["planning_time_s"] for sample in self.samples
                          if sample["planning_time_s"] is not None]
        sim_times = [sample["sim_time_s"] for sample in self.samples
                     if sample["sim_time_s"] is not None]
        if len(sim_times) >= 2:
            execution_time_s = float(sim_times[-1] - sim_times[0])
        elif execution_steps is not None:
            execution_time_s = float(execution_steps) * float(policy_interval_s)
        else:
            execution_time_s = None
        record = {
            "trial": self.trial,
            "method": self.method,
            "object": self.object_name,
            "success": bool(success),
            "aborted": bool(aborted),
            "goal_xyz": self.goal_xyz,
            "goal_quat_wxyz": self.goal_quat_wxyz,
            "init_xyz": None if not self.samples else self.samples[0]["xyz"],
            "init_quat_wxyz": None if not self.samples else self.samples[0]["quat_wxyz"],
            "error": [sample["error"] for sample in self.samples],
            "position_error": [sample["position_error"] for sample in self.samples],
            "orientation_error": [sample["orientation_error"] for sample in self.samples],
            "planning_time_s": planning_times,
            "planning_time_total_s": float(sum(planning_times)),
            "execution_time_s": execution_time_s,
            "steps": 0 if execution_steps is None else int(execution_steps),
            "samples": self.samples,
            "definitions": {
                "error": "500 * ||p - p_goal||^2 + 5 * (1 - (q · q_goal)^2), matching metrics/plot_comparision.py",
                "position_error": "Euclidean position error in metres",
                "orientation_error": "1 - (q · q_goal)^2",
                "planning_time_s": "Wall-clock seconds of each planner call",
                "execution_time_s": "Simulated seconds from the first sample to the last",
            },
        }
        trial_dir = self._trial_dir()
        os.makedirs(trial_dir, exist_ok=True)
        _write_json(path.join(trial_dir, "metrics.json"), record)
        _write_json(path.join(trial_dir, "traj.json"), {
            "trial_id": self.trial,
            "success": bool(success),
            "interrupted": bool(aborted),
            "num_samples": len(self.samples),
            "object_pose_wxyz": [
                {
                    "step": sample["step"],
                    "sim_time_s": sample["sim_time_s"],
                    "xyz": sample["xyz"],
                    "quat_wxyz": sample["quat_wxyz"],
                }
                for sample in self.samples
            ],
            "target_pose_wxyz": {"xyz": self.goal_xyz, "quat_wxyz": self.goal_quat_wxyz},
        })
        if self.samples:
            self._update_trails(record)
        summary = {key: record[key] for key in (
            "trial", "method", "object", "success", "aborted", "steps",
            "planning_time_total_s", "execution_time_s",
        )}
        summary.update({
            "final_error": None if not record["error"] else record["error"][-1],
            "final_position_error": None if not record["position_error"] else record["position_error"][-1],
            "final_orientation_error": None if not record["orientation_error"] else record["orientation_error"][-1],
            "metrics": path.join(trial_dir, "metrics.json"),
        })
        self._replace_summary(summary)
        return record

    def _object_dir(self):
        return path.join(self.output_dir, self.method, self.object_name)

    def _trial_dir(self):
        return path.join(self._object_dir(), f"{self.trial:03d}")

    def _replace_summary(self, summary):
        summary_path = path.join(self._object_dir(), "trials.jsonl")
        rows = []
        if path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json_load_text(line)
                    if int(item.get("trial", -1)) != self.trial:
                        rows.append(item)
        rows.append(summary)
        rows.sort(key=lambda item: int(item["trial"]))
        temporary = summary_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            for item in rows:
                handle.write(json_dumps(item) + "\n")
        os.replace(temporary, summary_path)

    def _update_trails(self, record):
        trails_path = path.join(self._object_dir(), "trails.json")
        entries = []
        if path.exists(trails_path):
            with open(trails_path, "r", encoding="utf-8") as handle:
                loaded = json_load(handle)
            if isinstance(loaded, list):
                entries = [item for item in loaded if int(item.get("trial", -1)) != self.trial]
        entries.append({
            "trial": self.trial,
            "init_xyz": record["init_xyz"],
            "init_quat_wxyz": record["init_quat_wxyz"],
            "goal_xyz": record["goal_xyz"],
            "goal_quat_wxyz": record["goal_quat_wxyz"],
        })
        entries.sort(key=lambda item: int(item["trial"]))
        _write_json(trails_path, entries)


def json_dumps(payload):
    import json
    return json.dumps(payload, allow_nan=False)


def json_load(handle):
    import json
    return json.load(handle)


def json_load_text(text):
    import json
    return json.loads(text)


def _write_json(output_path, payload):
    import json
    directory = path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = output_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, output_path)


def comp_pos_error_traj(pos_traj, targ_pos):
    return np.linalg.norm(pos_traj - targ_pos, axis=1)


def comp_quat_error_traj(quat_traj, targ_quat):
    quat_error_traj = []
    for t in range(quat_traj.shape[0]):
        quat_error_traj.append(comp_quat_error(quat_traj[t], targ_quat))
    return np.array(quat_error_traj)


def comp_angle_error_traj(quat_traj, targ_quat):
    quat_error_traj = []
    for t in range(quat_traj.shape[0]):
        quat_error_traj.append(np.abs(rotations.quat_to_rpy(quat_traj[t])[2] - rotations.quat_to_rpy(targ_quat)[2]))
    return np.array(quat_error_traj)


def save_data(data, data_name, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()
    else:
        save_dir = path.join(os.getcwd(), save_dir)

    if not path.exists(save_dir):
        os.makedirs(save_dir)

    saved_path = path.join(save_dir, data_name)

    with open(saved_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_data(data_name, save_dir=None):
    if save_dir is None:
        save_dir = os.getcwd()

    saved_path = path.join(save_dir, data_name)

    try:
        with open(saved_path, 'rb') as f:
            data = pickle.load(f)
    except:
        assert False

    return data
