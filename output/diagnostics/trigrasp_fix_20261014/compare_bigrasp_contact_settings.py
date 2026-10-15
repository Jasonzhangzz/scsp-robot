"""Summarize paired BigRasp solver/friction settings; preserve failed runs."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = ('mpc_ns3_mu5', 'mpc_ns10_mu5', 'mpc_ns3_mu8', 'mppi_ns3_mu5', 'mppi_ns10_mu5')


def metrics(report):
    initial = np.asarray(report['experiment']['initial_pose'])
    rows = report.get('telemetry', [])
    if not rows:
        return dict(success=False, settings=report['contact_solver_settings'],
            runtime_error=report['runtime_error'], simulation_seconds=report['simulation_seconds'],
            final_height_gain_mm=float((report['final_object_pos'][2]-initial[2])*1000))
    positions = np.asarray([row['object_pose'][0] for row in rows])
    normals = np.asarray([row['measured_normal_force'] for row in rows])
    live = [row['lambda_prediction']['live_grasp'] for row in rows]
    return dict(success=report['success'], settings=report['contact_solver_settings'], cycles=len(rows),
        bilateral_loaded_cycles=int(np.sum(np.all(normals > .034335, axis=1))),
        maximum_height_gain_mm=float((positions[:, 2].max()-initial[2])*1000),
        final_height_gain_mm=float((positions[-1, 2]-initial[2])*1000),
        final_horizontal_drift_mm=float(np.linalg.norm(positions[-1, :2]-initial[:2])*1000),
        final_position_error_mm=float(rows[-1]['object_position_error']*1000),
        local_contact_updates=sum(len(row['local_contact_updates']) for row in live),
        force_allocation_feasible_cycles=sum(row['force_allocation_feasible'] for row in live),
        current_wrench_available_cycles=sum(bool(row['allocated_wrench_available']) for row in live),
        global_replans=sum(row['global_contact_replan'] for row in live))


if __name__ == '__main__':
    reports = {case: json.loads((ROOT/(case+'.json')).read_text()) for case in CASES}
    baseline = np.asarray(reports[CASES[0]]['experiment']['initial_qpos'])
    summary = dict(scope='Elephant seed 0: MPC 1000 cycles; FA/MPPI horizon 8, samples 64, 160-cycle budget. One completed run per setting; failed FA/MPPI retries documented separately.',
        initialization='NoSlip=3 during startup, requested setting enabled before control',
        maximum_initial_qpos_difference=max(float(np.max(np.abs(
            np.asarray(report['experiment']['initial_qpos'])-baseline))) for report in reports.values()),
        material_friction='Only fingertip/object pair changed from 5 to 8; table/object stayed 5',
        cases={case: metrics(report) for case, report in reports.items()})
    (ROOT/'bigrasp_contact_settings_comparison.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
