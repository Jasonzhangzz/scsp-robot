"""Audit saved horizons and runtime force integration without changing the experiment."""
from pathlib import Path
import argparse
import csv
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()
import numpy as np
from examples.mpc.fingertips.test.collision_prediction_core import (
    Experiment, Settings, FORCES, MODES, METHODS, native_diagnostics)
from examples.mpc.fingertips.test.collision_prediction_assets import scene_xml, package_scene


def runtime_audit(folder):
    xml, info = scene_xml('sphere')
    experiment = Experiment(package_scene(xml, folder), info, Settings())
    model = experiment.models['mjwarp']
    ball = model.body('ball').id
    samples = []
    for force in FORCES:
        q = model.qpos0.copy();q[:3] = [0, 0, 1.];q[7:10] = [.5, 0, 1.]
        wrench = np.zeros((model.nbody, 6));wrench[ball, 0] = force
        samples.append(dict(qpos=q, qvel=np.zeros(model.nv), wrench=wrench))
    # Warm the exact multiworld size on disposable data before instrumentation.
    with native_diagnostics(): experiment.reference_batch(samples)
    backend = experiment.backends['mjwarp']
    original = backend.api.step
    trace = []

    def counted(wmodel, wdata):
        before = wdata.time.numpy().copy()
        applied = wdata.xfrc_applied.numpy().copy()
        original(wmodel, wdata)
        backend.wp.synchronize_device(backend.device)
        trace.append(dict(time_before=before.tolist(), time_after=wdata.time.numpy().tolist(),
                          ball_force_N=applied[:, ball, :3].tolist()))

    backend.api.step = counted
    try:
        with native_diagnostics() as warnings:
            batch = experiment.reference_batch(samples)
        assert not warnings
    finally:
        backend.api.step = original
    assert len(trace) == 10, len(trace)
    for index, call in enumerate(trace):
        np.testing.assert_allclose(call['time_before'], index*.002, atol=1e-7)
        np.testing.assert_allclose(call['time_after'], (index+1)*.002, atol=1e-7)
        np.testing.assert_allclose(call['ball_force_N'], [s['wrench'][ball,:3] for s in samples], atol=1e-8)
    records = []
    for force, sample, batched in zip(FORCES, samples, batch):
        assert batched['execution'] == 'gpu_batch' and batched['error'] is None
        # Independently count the legacy serial reference path as well.
        trace_serial = []
        def counted_serial(wmodel, wdata):
            trace_serial.append(wdata.time.numpy().tolist())
            original(wmodel, wdata)
        backend.api.step = counted_serial
        try:
            with native_diagnostics() as warnings:
                serial = experiment.reference(sample)
            assert not warnings
        finally:
            backend.api.step = original
        assert len(trace_serial) == 10
        predictions = experiment.predict(sample, serial[0])
        expected = sample['qvel'][6:9] + .02*(model.opt.gravity + sample['wrench'][ball,:3]/model.body_mass[ball])
        for method in METHODS:
            output = predictions[method]
            assert output['status'] == 'ok', output
            np.testing.assert_allclose(output['qvel'][6:9], expected, atol=2e-6, rtol=0)
            records.append(dict(force_N=force, method=method, external_impulse_Ns=force*.02,
                expected_ball_velocity=expected.tolist(), measured_ball_velocity=output['qvel'][6:9].tolist(),
                velocity_error=float(np.linalg.norm(output['qvel'][6:9]-expected))))
        np.testing.assert_allclose(batched['reference'][3][-1,6:9], expected, atol=2e-6, rtol=0)
        assert serial[2].shape == (11,14)
        # No-contact semi-implicit Euler has a known step-size position offset.
        difference = predictions['scm']['qpos'][7:10]-serial[2][-1,7:10]
        np.testing.assert_allclose(difference, (.02-.002)*.02/2*(model.opt.gravity + sample['wrench'][ball,:3]/model.body_mass[ball]), atol=2e-7, rtol=0)
    return dict(status='passed', actual_mjwarp_batch_step_calls=len(trace),
        actual_mjwarp_serial_step_calls_per_query=10, scm_and_comfree_steps=1,
        horizon_s=.02, batch_force_trace=trace, free_flight_velocity_checks=records)


def saved_audit(source):
    manifest = json.loads((source/'manifest.json').read_text())
    settings = manifest['settings']
    assert settings['reference_steps'] == 10 and settings['reference_dt'] == .002 and settings['horizon'] == .02
    count = 0; max_input_impulse_difference = 0.; strata = []
    for folder in sorted(source.iterdir()):
        if not (folder/'predictions.csv').exists(): continue
        for row in csv.DictReader((folder/'predictions.csv').open()):
            if row['method'] != 'mjwarp' or row['selected'] != 'True' or row['mode'] not in MODES: continue
            with np.load(folder/'queries'/(row['sample_id']+'.npz'), allow_pickle=False) as a:
                np.testing.assert_allclose(a['reference_times'], np.arange(11)*.002, rtol=0, atol=1e-15)
                assert a['reference_qpos'].shape == (11,14) and a['reference_qvel'].shape == (11,12)
                np.testing.assert_array_equal(a['reference_qpos'][-1], a['mjwarp_qpos'])
                force = a['wrench'][2,:3]
                np.testing.assert_allclose(np.linalg.norm(force), float(row['force_N']), atol=1e-12)
                delta = np.max(np.abs(force*.02-sum([force*.002 for _ in range(10)])))
                max_input_impulse_difference = max(max_input_impulse_difference, float(delta))
            count += 1
        for row in csv.DictReader((folder/'predictions.csv').open()):
            if (row['method']=='scm' and row['selected']=='True' and row['mode']=='sliding'
                    and row['relative_position_error']!=''):
                strata.append(row)
    bins = []
    for lower,upper in [(0,.01),(.01,.1),(.1,1),(1,float('inf'))]:
        subset=[r for r in strata if lower<=1000*float(r['relative_position_error_denominator'])<upper]
        bins.append(dict(reference_displacement_mm_lower=lower,reference_displacement_mm_upper=upper if np.isfinite(upper) else None,
            n=len(subset),mean_position_error_mm=float(np.mean([float(r['position_error_mm']) for r in subset])),
            mean_relative_position_error=float(np.mean([float(r['relative_position_error']) for r in subset]))))
    assert count == 8000
    return dict(status='passed', queries_checked=count, saved_reference_frames_per_query=11,
        max_external_impulse_difference_Ns=max_input_impulse_difference, sliding_reference_displacement_bins=bins)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT/'outputs/collision_prediction_200_per_mode_20261026_job/data')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/collision_horizon_impulse_audit_20261026.json')
    args=parser.parse_args()
    result=dict(saved=saved_audit(args.input_dir))
    with tempfile.TemporaryDirectory(prefix='scsp_impulse_',dir='/tmp') as folder:
        result['runtime']=runtime_audit(Path(folder))
    result['status']='passed'
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
