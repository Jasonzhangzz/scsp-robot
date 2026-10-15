#!/usr/bin/env python3
"""Replay cached inputs to audit finite-response SCM motion-error bounds.

Reconstruct the common GPU contact rows, not contacts from CPU mj_forward.
The cached trajectories are unchanged. All norms/bounds use impulse units
and the mass-weighted velocity metric; SI linear/angular errors are separate.
"""

import argparse
import csv
import json
from pathlib import Path

from runtime_env import activate_dependencies
activate_dependencies()

import mujoco
import numpy as np
from benchmark_utils import Backend
from compare_prediction import fresh_data
from lcp_backend import solve_lcp


def weighted_norm(value, matrix):
    return float(np.sqrt(max(0., value @ matrix @ value)))


def inverse_norm(value, matrix):
    if not value.size:
        return 0.
    factor = np.linalg.cholesky(matrix)
    return float(np.linalg.norm(np.linalg.solve(factor, value)))


def bound_diagnostics(J, inverse_mass, gap, free, h, tau, zeta, regularization):
    """Exact float64 law and independent coupled references on shared rows."""
    n = len(gap)
    x = h / tau
    a, b = (2*zeta*x+x*x)/(1+2*zeta*x+x*x), x*x/(1+2*zeta*x+x*x)
    mobility = J @ inverse_mass @ J.T
    W = mobility + regularization*np.eye(n)
    diagonal = np.diag(W)
    D = np.diag(diagonal)
    E = W-D
    g = J @ free + gap/h
    s = a*(J @ free) + b*gap/h
    p = np.maximum(-s/diagonal, 0.)
    hard, hard_certificate = solve_lcp(W, g)
    # Coupled implicit spring--damper reference with the SAME response.
    W_response = W+(1/a-1)*D
    g_response = J @ free+(b/a)*gap/h
    response, response_certificate = solve_lcp(W_response, g_response)
    delta = p-hard
    mass = np.linalg.inv(inverse_mass)
    velocity_error = inverse_mass @ J.T @ delta
    response_error = inverse_mass @ J.T @ (p-response)
    reference_mismatch = inverse_mass @ J.T @ (response-hard)
    residual_coupling, residual_response = E @ p, g-s
    eta = inverse_norm(residual_coupling+residual_response, W)
    eta_response = inverse_norm(residual_coupling, W_response)
    actual = weighted_norm(velocity_error, mass)
    actual_response = weighted_norm(response_error, mass)
    if actual > eta+1e-7 or actual_response > eta_response+1e-7:
        raise AssertionError("Derived mass-weighted motion bound violated")
    if n:
        scaled = E/np.sqrt(diagonal[:, None]*diagonal[None, :])
        normalized_coupling = float(np.linalg.norm(scaled, 2))
        # Amplification without orientation: compare with the actual projected
        # impulse difference. Small ratio indicates cancellation or a benign direction.
        factor = np.linalg.cholesky(inverse_mass).T @ J.T
        naive = np.linalg.norm(factor, 2)*np.linalg.norm(delta)
        projection_ratio = actual/naive if naive > 1e-15 else None
        active_agreement = float(np.mean((p > 1e-12) == (hard > 1e-12)))
        eig, vectors = np.linalg.eigh(W)
        condition_number = float(eig[-1]/eig[0])
        residual = residual_coupling+residual_response
        components = vectors.T @ residual
        null = eig < 1.e-7*max(1., eig[-1])
        energies = components**2/eig
        null_fraction = float(energies[null].sum()/energies.sum()) if energies.sum() > 0 else None
    else:
        normalized_coupling, projection_ratio, active_agreement = 0., None, 1.
        condition_number, null_fraction = None, None
    result = {
        "constraint_rows": n, "active_scm_rows": int(np.count_nonzero(p > 1e-12)),
        "normalized_coupling_operator_norm": normalized_coupling,
        "scm_impulse_norm_N_s": float(np.linalg.norm(p)),
        "impulse_difference_norm_N_s": float(np.linalg.norm(delta)),
        "coupling_bound_hard_metric": inverse_norm(residual_coupling, W),
        "response_bound_hard_metric": inverse_norm(residual_response, W),
        "combined_bound_hard": eta, "motion_error_hard": actual,
        "matched_response_bound": eta_response, "motion_error_matched_response": actual_response,
        "hard_vs_matched_response_motion": weighted_norm(reference_mismatch, mass),
        "projected_impulse_ratio": projection_ratio, "active_row_agreement": active_agreement,
        "hard_matrix_condition_number": condition_number,
        "hard_bound_near_null_fraction": null_fraction,
        "lcp_residual": hard_certificate["residual"],
        "matched_response_lcp_residual": response_certificate["residual"],
    }
    return result, free+inverse_mass @ J.T @ p, free+inverse_mass @ J.T @ hard


def summarize(rows):
    result = {}
    for key in rows[0]:
        values = np.array([r[key] for r in rows if isinstance(r[key], (int, float))], dtype=float)
        if len(values):
            result[key] = {"n": len(values), "mean": float(values.mean()), "std": float(values.std()),
                           "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
                           "max": float(values.max())}
    return result


def analyze(root, output):
    manifest = json.loads((root/'manifest.json').read_text())
    args = manifest['arguments']
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for configuration in manifest['configurations']:
        folder = root/configuration
        model = mujoco.MjModel.from_xml_path(str((folder/'model.xml').resolve()))
        backend = Backend('lcp', model, fresh_data(model), device=args['device'],
                          nconmax=args['nconmax'], njmax=args['njmax'],
                          lcp_regularization=args['lcp_regularization'])
        with (folder/'predictions.csv').open() as stream:
            cached_metrics = {(r['engine'], int(r['sample_id'])): r for r in csv.DictReader(stream)}
        with np.load(folder/'states.npz', allow_pickle=False) as states:
            indices = {e: np.flatnonzero(states['engine'] == e) for e in ('scm_soft', 'lcp', 'mjwarp')}
            for engine in ('lcp', 'mjwarp'):
                for key in ('sample_id', 'qpos_initial', 'qvel_initial', 'wrench_world_N_Nm'):
                    np.testing.assert_array_equal(states[key][indices[engine]], states[key][indices['scm_soft']])
            for k, index in enumerate(indices['scm_soft']):
                data = fresh_data(model, states['qpos_initial'][index], states['qvel_initial'][index],
                                  states['wrench_world_N_Nm'][index])
                backend.reset(data)
                with backend.wp.ScopedDevice(backend.device):
                    backend.api.update_mass(data)
                    backend.api.prepare_step(backend.wmodel, backend.wdata)
                    backend.wp.synchronize_device(backend.device)
                    n = int(backend.wdata.nefc.numpy()[0])
                    J = backend.wdata.efc.J.numpy()[0, :n, :model.nv].astype(float)
                    gap = backend.wdata.efc.efc_dist.numpy()[0, :n].astype(float)
                    free = backend.wdata.qvel_smooth_pred.numpy()[0].astype(float)
                    inverse = (np.diag(backend.api.inverse_diagonal.numpy().astype(float))
                               if backend.api.use_diagonal else backend.api.inverse_dense.numpy().astype(float))
                h = float(backend.wmodel.opt.timestep.numpy()[0])
                diagnostics, v_scm, v_lcp = bound_diagnostics(
                    J, inverse, gap, free, h, args['scm_response_time'], args['scm_damping_ratio'],
                    args['scm_regularization'])
                if args['scm_regularization'] != args['lcp_regularization'] or args['scm_motion'] != 'dynamic':
                    raise ValueError('This audit requires matched regularization and dynamic mode')
                M = np.zeros((6, 6))
                mujoco.mj_fullM(model, M, data.qM)
                cached_scm = states['qvel_predicted'][index]
                cached_lcp = states['qvel_predicted'][indices['lcp'][k]]
                cached_mj = states['qvel_predicted'][indices['mjwarp'][k]]
                replay_scm = float(np.linalg.norm(v_scm-cached_scm))
                replay_lcp = float(np.linalg.norm(v_lcp-cached_lcp))
                if replay_scm > 2e-5 or replay_lcp > 2e-5:
                    raise AssertionError(f'Cached replay mismatch: {configuration}, {k}: {replay_scm}, {replay_lcp}')
                sid = int(states['sample_id'][index])
                metric = cached_metrics['scm_soft', sid]
                row = dict(configuration=configuration, sample_id=sid, mode=str(states['reference_mode'][index]), **diagnostics)
                row.update(reconstructed_scm_velocity_difference=replay_scm,
                           reconstructed_lcp_velocity_difference=replay_lcp,
                           cached_mass_weighted_error_mjwarp=weighted_norm(cached_scm-cached_mj, M),
                           cached_mass_weighted_error_lcp=weighted_norm(cached_scm-cached_lcp, M),
                           linear_error_mjwarp_m_s=float(metric['linear_velocity_error_m_s']),
                           angular_error_mjwarp_rad_s=float(metric['angular_velocity_error_rad_s']),
                           pose_position_error_m=float(metric['position_error_m']),
                           pose_orientation_error_rad=float(metric['orientation_error_rad']),
                           velocity_angle_deg=float(metric['velocity_angle_deg']) if metric['velocity_angle_deg'] else None,
                           positive_velocity_alignment=1 if metric['velocity_same_direction'] == 'True' else (0 if metric['velocity_same_direction'] == 'False' else None),
                           relative_velocity_error=float(metric['velocity_relative_error']) if metric['velocity_relative_error'] else None,
                           h_times_linear_error_m=h*float(metric['linear_velocity_error_m_s']))
                rows.append(row)
                if (k+1) % 100 == 0:
                    print(f'{configuration}: audited {k+1}/300 inputs', flush=True)
    with (output/'motion_bound_queries.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = {'input_directory': str(root.resolve()), 'queries': len(rows),
              'coefficients': {'h': args['timesteps'], 'tau': args['scm_response_time'], 'zeta': args['scm_damping_ratio']},
              'definitions': {'bounds': 'mass-weighted next-velocity difference, impulse units, sqrt(kg)*m/s equivalent',
                              'hard_reference': 'coupled W=J Minv J.T+rI, g=J vfree+gap/h',
                              'matched_response_reference': 'Wc=W+(1/a-1)D, gc=J vfree+(b/a)gap/h',
                              'positive_alignment': 'mass-weighted dot product > 0; undefined near-zero velocities excluded; not an accuracy threshold',
                              'scope': 'one-step full-dynamic probes, not accumulated rollouts or a CSO task-descent measurement'},
              'by_mode': {mode: summarize([r for r in rows if r['mode'] == mode])
                          for mode in ('separation', 'sliding', 'sticking')}}
    (output/'motion_bound_report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(f'Saved {len(rows)} audits to {output}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input_dir, args.output_dir)
