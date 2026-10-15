"""Compare the same startup seed with three and ten execution NoSlip sweeps."""
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def metrics(report):
    hold = [row for row in report["steps"] if row["phase"] == "hold"]
    dt = report["prediction_intervals"]["mpc_dt"]
    tail = hold[-round(5 / dt):]
    positions = np.array([row["object_pos"] for row in tail])
    target = np.array(report["object_target_pose"])
    errors = np.linalg.norm(positions - target[:3], axis=1) * 1000
    quaternions = np.array([row["object_quat"] for row in tail])
    angles = np.degrees(2 * np.arccos(np.clip(np.abs(quaternions @ target[3:]), 0, 1)))
    return dict(
        success=report["success"],
        hold_seconds=len(hold) * dt,
        hold_window_seconds=len(tail) * dt,
        first_lost_grasp_step=next((row['step'] for row in report['steps']
                                   if row.get('replan_reason') == 'lost_grasp'), None),
        replan_count=report["replan_count"],
        minimum_hold_contacts=min(row["contacts"] for row in hold),
        hold_force_closure_fraction=float(np.mean([
            row["contact_audit"]["actual_force_closure"]["force_closure"] for row in hold])),
        hold_load_feasible_fraction=float(np.mean([
            row["contact_audit"]["actual_force_closure"]["load_feasible"] for row in hold])),
        hold_requested_wrench_available_fraction=float(np.mean([
            row["allocated_wrench_available"] for row in hold])),
        final_xyz_error_mm=float(np.linalg.norm(np.array(report["final_object_pos"]) - target[:3]) * 1000),
        hold_window_xyz_error_median_mm=float(np.median(errors)),
        hold_window_xyz_error_p95_mm=float(np.percentile(errors, 95)),
        hold_window_xyz_error_max_mm=float(np.max(errors)),
        hold_window_vertical_span_mm=float(np.ptp(positions[:, 2]) * 1000),
        hold_window_position_step_p95_mm=float(np.percentile(np.linalg.norm(np.diff(positions, axis=0), axis=1) * 1000, 95)),
        hold_window_orientation_error_median_deg=float(np.median(angles)),
        hold_window_orientation_error_p95_deg=float(np.percentile(angles, 95)),
    )


if __name__ == "__main__":
    baseline = json.loads((ROOT / "ik_startup_isolated_hold20.json").read_text())
    candidate = json.loads((ROOT / "noslip10_hold20.json").read_text())
    result = dict(
        scope="Default rubber_duck, seed 0, 20 cm lift, 20 s hold; two repetitions per setting (10-sweep repeat capped at 400 steps after first failure)",
        impratio=10,
        max_initial_hand_q_difference=float(np.max(np.abs(
            np.array(baseline["initial_ik_hand_q"]) - candidate["initial_ik_hand_q"]))),
        max_initial_object_goal_difference=float(np.max(np.abs(
            np.array(baseline["object_target_pose"]) - candidate["object_target_pose"]))),
        noslip3=metrics(baseline),
        noslip10=metrics(candidate),
        noslip3_repeat=metrics(json.loads((ROOT / 'noslip3_repeat_hold20.json').read_text())),
        noslip10_repeat=metrics(json.loads((ROOT / 'noslip10_repeat.json').read_text())),
    )
    (ROOT / "noslip_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
