"""Readers for the real rollout caches used by the metrics plots.

The recorder stores physical execution time as ``sim_time_s`` in each sample.
This module keeps the plotting scripts independent of the particular cache
format (the current CSV cache and the JSON cache produced by ``TrialRecorder``)
and normalizes time to seconds elapsed since the first sample.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


# The archived experiment labels were shifted while the summary tables were
# assembled.  Keep the correction in one place so every figure uses the same
# source directory.
METHOD_TO_OUTPUT_MODE = {
    "SCSP*": "isaac",
    "DyWA": "a5",
    "Sampling": "a4",
    "ContactSDF": "a2",
    "STOCS": "a7",
    "STOCS-3D": "a7",
    "CF-MPC": "a6",
    "A-MPC": "a1",
    "w/o RS": "a1",
}


def _float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _success(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        return list(data["samples"])
    if isinstance(data, dict) and isinstance(data.get("trajectory"), list):
        return list(data["trajectory"])
    if isinstance(data, list):
        return data
    raise ValueError(f"No sample rows found in {path}")


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        return read_csv_rows(path)
    return read_json_rows(path)


def _time_value(row: dict):
    for key in ("sim_time_s", "execution_time_s", "time_s", "time"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def execution_time_s(rows: Iterable[dict], default_dt: float = 0.02) -> np.ndarray:
    """Return elapsed physical execution time for each row, in seconds."""
    rows = list(rows)
    if not rows:
        return np.empty(0, dtype=float)
    values = [_time_value(row) for row in rows]
    if all(value is not None for value in values):
        result = np.asarray(values, dtype=float)
        return result - result[0]
    steps = []
    for index, row in enumerate(rows):
        value = _float(row.get("step"))
        steps.append(index if value is None else value)
    result = np.asarray(steps, dtype=float) * float(default_dt)
    return result - result[0]


def cached_path(root: Path, mode: str, object_name: str, trial_id: int) -> Path | None:
    """Find a cached trajectory for one output mode and trial."""
    root = Path(root)
    if root.name == "trials":
        root = root / "trajectories"
    stem = f"{int(trial_id):03d}"
    candidates = [
        root / mode / object_name / f"{stem}.csv",
        root / mode / object_name / f"{stem}.json",
        root / mode / object_name / stem / "metrics.json",
        root / mode / object_name / stem / "traj.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def cached_rows(root: Path, method: str, object_name: str, trial_id: int) -> tuple[Path, list[dict]] | None:
    mode = METHOD_TO_OUTPUT_MODE.get(method, method)
    path = cached_path(root, mode, object_name, trial_id)
    if path is None:
        return None
    rows = read_rows(path)
    # metrics.json stores success at the record level, while CSV stores it per
    # sample.  Expose the record status on every row for uniform filtering.
    if path.name == "metrics.json":
        record = json.loads(path.read_text(encoding="utf-8"))
        status = record.get("success")
        if status is not None:
            rows = [dict(row, success=status) for row in rows]
    return path, rows


def successful(rows: list[dict]) -> bool:
    return bool(rows) and _success(rows[-1].get("success"))


def find_common_success_trial(root: Path, methods: Iterable[str], object_name: str,
                              trial_ids: Iterable[int] = range(5)) -> int:
    methods = list(methods)
    for trial_id in trial_ids:
        loaded = [cached_rows(root, method, object_name, trial_id) for method in methods]
        if loaded and all(item is not None and successful(item[1]) for item in loaded):
            return int(trial_id)
    modes = ", ".join(f"{method}={METHOD_TO_OUTPUT_MODE.get(method, method)}" for method in methods)
    raise FileNotFoundError(
        f"No trial is successful for every requested method ({modes}) in {Path(root)} "
        f"for {object_name}. Use the real cached trajectories or choose another trial."
    )


def row_error(rows: list[dict], index: int, goal_xyz=None, goal_quat_wxyz=None):
    value = _float(rows[index].get("error"))
    if value is not None:
        return value
    if goal_xyz is None or goal_quat_wxyz is None:
        raise ValueError("Trajectory has no error column and no goal pose was supplied")
    pose = rows[index]
    xyz = np.asarray([pose["x"], pose["y"], pose["z"]], dtype=float)
    quat = np.asarray([pose["qw"], pose["qx"], pose["qy"], pose["qz"]], dtype=float)
    goal_xyz = np.asarray(goal_xyz, dtype=float)
    goal_quat = np.asarray(goal_quat_wxyz, dtype=float)
    quat /= np.linalg.norm(quat)
    goal_quat /= np.linalg.norm(goal_quat)
    return float(500.0 * np.sum((xyz - goal_xyz) ** 2) + 5.0 * (1.0 - np.dot(quat, goal_quat) ** 2))

