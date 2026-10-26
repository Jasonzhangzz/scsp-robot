#!/usr/bin/env python3
"""Frozen-query SCM/ComFree predictions using native PhysX collision manifolds.

PhysX reference: .002 s x 10. Predictors: .02 s x 1. See the adjacent README
for the separate endpoint diagnostic and the deliberately hybrid integrators.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()
import mujoco
import numpy as np

from examples.mpc.franka.scm.benchmark_utils import UPSTREAM_COMMIT
from examples.mpc.fingertips.test.collision_prediction_core import (
    Settings, MODES, METRICS, RELATIVE_METRICS, RELATIVE_FLOORS, classify_contacts,
    pose_errors, relative_errors, world_velocity)

METHODS = ('scm', 'comfree', 'isaacgym')
ERRORS = METRICS + RELATIVE_METRICS
ROW_FIELDS = ('sample_id', 'object', 'seed', 'attempt', 'requested_mode', 'source_mode',
    'source_contact_group', 'mode', 'method', 'selected', 'geometry', 'force_N',
    'speed_level_m_s', 'initial_speed_m_s', 'initial_angular_speed_rad_s',
    'ball_initial_speed_m_s', 'contact_group', 'ncontact_start', 'ncontact_floor_start',
    'ncontact_ball_start', 'ncontact_max', 'ncontact_end', 'established', 'separated',
    'free_flight', 'max_slip_m_s', 'status', 'failure_stage', 'failure_reason',
    'reference_execution', 'actual_nworld', 'requested_native_batch_worlds', 'reference_fallback_reason',
    'physics_seconds', 'source_query_sha256') + ERRORS + tuple(
        key + '_denominator' for key in RELATIVE_METRICS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def dumps(value):
    return json.dumps(value, default=plain, allow_nan=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=plain, allow_nan=False, indent=2) + '\n')


def update_manifest(output, manifest, **updates):
    manifest.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
    target = output / 'manifest.json'
    tmp = target.with_suffix('.tmp')
    write_json(tmp, manifest)
    tmp.replace(target)


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def positive_int(value):
    value = int(value)
    if value < 1: raise argparse.ArgumentTypeError('must be positive')
    return value


def load_source(source, objects=None, limit=None):
    """Validate frozen assets, then select source IDs without changing labels."""
    source = Path(source).resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Source experiment must be complete')
    settings = Settings(**manifest['settings'])
    if (settings.horizon, settings.reference_dt, settings.reference_steps) != (.02, .002, 10):
        raise ValueError('Source horizon must be .02 s, reference .002 s x 10')
    names = list(manifest['objects'] if not objects or objects==['all'] else objects)
    if len(names) != len(set(names)) or not set(names) <= set(manifest['objects']):
        raise ValueError('Objects must be distinct names in the source manifest')
    records = {}
    for name in names:
        folder = source / name
        info = json.loads((folder / 'object.json').read_text())
        if sha(folder / 'model.xml') != info['model_sha256']:
            raise ValueError(f'Corrupt source model: {name}')
        for filename, checksum in info['packaged_asset_sha256'].items():
            if sha(folder / filename) != checksum:
                raise ValueError(f'Corrupt source asset: {name}/{filename}')
        with (folder / 'predictions.csv').open() as stream:
            rows = [r for r in csv.DictReader(stream) if r['method'] == 'mjwarp'
                    and r['selected'] == 'True' and r['mode'] in MODES]
        if len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate source IDs: {name}')
        if any(r['status'] != 'ok' for r in rows):
            raise ValueError(f'Invalid selected source reference: {name}')
        if limit:
            # Smoke subsets balance original regimes/counts/configurations only;
            # their NEW PhysX labels are never inferred from these strata.
            groups = defaultdict(deque)
            for row in rows:
                groups[(row['mode'], row['contact_group'], row['geometry'])].append(row)
            selected = []
            while groups and len(selected) < limit:
                for key in sorted(list(groups)):
                    selected.append(groups[key].popleft())
                    if not groups[key]: del groups[key]
                    if len(selected) == limit: break
            rows = selected
        records[name] = rows
    if not objects and not limit and sum(map(len, records.values())) != 8000:
        raise ValueError('The full frozen source must contain exactly 8000 primary queries')
    return source, manifest, settings, records


def read_source_query(source, name, row):
    path = source / name / 'queries' / (row['sample_id'] + '.npz')
    checksum = sha(path)
    with np.load(path, allow_pickle=False) as archive:
        sample = {key: archive[key].copy() for key in ('qpos', 'qvel', 'wrench')}
        metadata = json.loads(str(archive['metadata_json']))
    if any(str(metadata[key]) != row[key] for key in ('sample_id','mode','contact_group','object','geometry')):
        raise ValueError(f'Source CSV/NPZ disagreement: {path}')
    for key, shape in (('qpos', (14,)), ('qvel', (12,)), ('wrench', (3, 6))):
        if sample[key].shape != shape or not np.isfinite(sample[key]).all():
            raise ValueError(f'Invalid source {key}: {path}')
    metadata.update(source_mode=metadata['mode'], source_contact_group=metadata['contact_group'],
                    source_query_sha256=checksum, source_query=str(path), selected=True)
    for key in ('reference_batch_sample_ids', 'reference_fallback_reason'):
        metadata.pop(key, None)
    return sample, metadata


def copy_scene(source, name, output):
    origin, folder = source / name, output / name
    folder.mkdir()
    for filename in ('model.xml', 'object.json'):
        shutil.copy2(origin / filename, folder / filename)
    if (origin / 'assets').exists(): shutil.copytree(origin / 'assets', folder / 'assets')
    (folder / 'queries').mkdir()
    return folder, json.loads((folder / 'object.json').read_text())


def state_velocity(model, qpos, qvel):
    """Kinematics only: no collision or constraint calculation."""
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    return world_velocity(model, data)


def metrics(model, sample, prediction, reference_qpos, reference_qvel, length):
    errors = pose_errors(prediction['qpos'][:7], reference_qpos[:7], length)
    ref_v = state_velocity(model, reference_qpos, reference_qvel)
    dv = state_velocity(model, prediction['qpos'], prediction['qvel']) - ref_v
    errors.update(linear_velocity_error_m_s=float(np.linalg.norm(dv[:3])),
                  angular_velocity_error_rad_s=float(np.linalg.norm(dv[3:])))
    motion = pose_errors(reference_qpos[:7], sample['qpos'][:7], length)
    errors.update(relative_errors(errors, motion, ref_v))
    return errors


def save_query(path, sample, metadata, predictions, reference_qpos, reference_qvel, manifold):
    arrays = dict(sample, metadata_json=np.array(dumps(metadata)),
        reference_qpos=reference_qpos, reference_qvel=reference_qvel,
        reference_times=np.arange(11)*.002,
        initial_manifold_json=np.array(dumps(manifold)))
    for method in METHODS:
        prediction = predictions.get(method, {})
        arrays[method+'_qpos'] = prediction['qpos'] if prediction.get('qpos') is not None else np.full(14, np.nan)
        arrays[method+'_qvel'] = prediction['qvel'] if prediction.get('qvel') is not None else np.full(12, np.nan)
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix('.tmp').replace(path)


def load_saved_query(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    sample = {key: arrays[key] for key in ('qpos', 'qvel', 'wrench')}
    metadata = json.loads(str(arrays['metadata_json']))
    manifold = json.loads(str(arrays['initial_manifold_json']))
    return arrays, sample, metadata, manifold


def verify_scene_files(folder):
    info=json.loads((folder/'object.json').read_text())
    if sha(folder/'model.xml')!=info['model_sha256']:
        raise ValueError(f'Saved model hash mismatch: {folder}')
    for filename,expected in info['packaged_asset_sha256'].items():
        if sha(folder/filename)!=expected:raise ValueError(f'Saved asset hash mismatch: {folder/filename}')
    spec=json.loads((folder/'physx_spec.json').read_text())
    if sha(folder/'model.xml')!=spec['source_model_sha256']:
        raise ValueError('PhysX export source model hash mismatch')
    for filename,expected in spec['asset_sha256'].items():
        if sha(filename)!=expected:raise ValueError(f'PhysX exported asset hash mismatch: {filename}')
    return info,spec


def append_predictions(folder, metadata, predictions):
    path = folder / 'predictions.csv'
    exists = path.exists()
    with path.open('a', newline='') as stream, (folder.parent/'results.log').open('a') as log:
        writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS, extrasaction='ignore')
        if not exists: writer.writeheader()
        for method, prediction in predictions.items():
            public = {k: v for k, v in prediction.items() if k not in ('qpos', 'qvel', 'time')}
            if 'seconds' in public: public['physics_seconds'] = public.pop('seconds')
            writer.writerow(dict(metadata, method=method, **public))
            log.write(dumps(dict(kind='prediction', sample_id=metadata['sample_id'],
                object=metadata['object'], method=method, source_mode=metadata['source_mode'],
                mode=metadata['mode'], contact_group=metadata['contact_group'], geometry=metadata['geometry'],
                status=prediction['status'], failure_reason=prediction.get('failure_reason', ''),
                absolute_errors={key: prediction.get(key) for key in METRICS},
                relative_errors={key: prediction.get(key) for key in RELATIVE_METRICS},
                relative_denominators={key: prediction.get(key+'_denominator') for key in RELATIVE_METRICS}))+'\n')


def failure(reason, status='failed'):
    return dict(status=status, failure_reason=str(reason), **{key: None for key in ERRORS})


def evaluate_native(native, samples):
    """Isolate individual errors without silently changing engines or assets."""
    from examples.mpc.fingertips.test.collision_prediction_isaacgym_native import normalize_manifold
    try:
        geometry = native.geometry_samples(samples)
        if geometry.get('native_warnings'):
            raise RuntimeError('PhysX geometry warning: '+'; '.join(geometry['native_warnings']))
        reference = native.reference_samples(samples)
        if reference.get('native_warnings'):
            raise RuntimeError('PhysX reference warning: '+'; '.join(reference['native_warnings']))
        for key, width in (('trajectory_qpos',14),('trajectory_qvel',12)):
            array=np.asarray(reference[key],float)
            if array.shape!=(len(samples),11,width) or not np.isfinite(array).all():
                raise ValueError(f'Malformed/nonfinite {key}')
        return [dict(reference=reference, index=i,
                     manifold=normalize_manifold(geometry['contacts'][i]),
                     probe=geometry, batch_worlds=len(samples),
                     batch_sample_ids=[s.get('_sample_id') for s in samples]) for i in range(len(samples))]
    except Exception as exc:
        process=getattr(native,'process',None)
        if process is not None and process.poll() is not None:
            raise RuntimeError('PhysX worker terminated; aborting instead of labeling healthy queries invalid') from exc
        if len(samples) > 1:
            k = len(samples)//2
            results=evaluate_native(native, samples[:k])+evaluate_native(native, samples[k:])
            for result in results:
                result.setdefault('native_retries',[]).insert(0,dict(requested_nworld=len(samples),
                    sample_ids=[s.get('_sample_id') for s in samples],
                    reason=f'{type(exc).__name__}: {exc}'))
            return results
        return [dict(error=f'{type(exc).__name__}: {exc}', batch_worlds=1)]


def setup_native(spec, args):
    from examples.mpc.fingertips.test.collision_prediction_isaacgym_native import IsaacGymNative
    return IsaacGymNative(spec, batch_size=args.reference_batch_size,
                          use_gpu=True, python=args.isaacgym_python,
                          log_path=Path(spec['source_model']).parent/'physx_worker.log')


def check_implementation(output, manifest):
    for filename, expected in manifest['implementation_sha256'].items():
        if sha(Path(__file__).with_name(filename)) != expected:
            raise RuntimeError(f'Implementation changed during experiment: {filename}; use a new output directory')


def run(args):
    from examples.mpc.fingertips.test.collision_prediction_isaacgym_native import export_scene_spec
    from examples.mpc.fingertips.test.collision_prediction_external_contacts import ExternalContactPredictors
    from examples.mpc.fingertips.test.collision_prediction_isaacgym_report import report_directory
    source, original, settings, records = load_source(args.input_dir, args.objects, args.limit_per_object)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    manifest = dict(schema_version=1, status='running', phase='reference_and_scm',
        source=str(source), source_manifest_sha256=sha(source/'manifest.json'),
        objects=list(records), methods=list(METHODS), settings=settings.to_dict(),
        n_input=sum(map(len, records.values())), frozen_source=True, resampling=False,
        reference='native GPU PhysX .002 s x 10; CPU pipeline, individual native contact records',
        reference_batch_size=args.reference_batch_size, comfree_batch_size=args.comfree_batch_size,
        endpoint_diagnostic='isolated .002 s probe from t=.02; load interval [.020,.022]; output state discarded',
        initial_collision='independent zero-velocity, zero-gravity probe; geometry only, forces discarded',
        smooth_dynamics='existing MuJoCo SCM/ComFree smooth dynamics and .02 integration; native PhysX collision',
        relative_denominator_floors=RELATIVE_FLOORS, upstream_commit=UPSTREAM_COMMIT,
        packages={name:importlib.metadata.version(name) for name in ('numpy','mujoco','warp-lang','comfree_warp')},
        started_at=datetime.now(timezone.utc).isoformat(), results={})
    update_manifest(output, manifest)
    implementation = output/'implementation'; implementation.mkdir()
    files = [Path(__file__), *Path(__file__).parent.glob('collision_prediction_isaacgym_*.py'),
             Path(__file__).with_name('collision_prediction_external_contacts.py'),
             Path(__file__).with_name('collision_prediction_core.py')]
    manifest['implementation_sha256'] = {}
    for path in files:
        shutil.copy2(path, implementation/path.name)
        manifest['implementation_sha256'][path.name] = sha(path)
    source_rows, timing_rows = [], []
    try:
        for name, rows in records.items():
            check_implementation(output,manifest)
            folder, info = copy_scene(source, name, output)
            spec = export_scene_spec(folder/'model.xml', folder/'physx_assets', cylinder_slices=96)
            write_json(folder/'physx_spec.json', spec)
            predictor = ExternalContactPredictors(folder/'model.xml', settings)
            model = mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
            counts = Counter(); manifest['results'][name] = dict(status='running', n_input=len(rows))
            with setup_native(spec, args) as native, gzip.open(folder/'reference_contacts.jsonl.gz','wt') as contacts, \
                    (folder/'samples.jsonl').open('w') as samples_log:
                manifest['results'][name]['worker']=native.metadata
                for first in range(0, len(rows), args.reference_batch_size):
                    source_batch = rows[first:first+args.reference_batch_size]
                    pairs = [read_source_query(source, name, row) for row in source_batch]
                    samples = [pair[0] for pair in pairs]
                    results = evaluate_native(native, [dict(s,_sample_id=m['sample_id']) for s,m in pairs])
                    peers = [pair[1]['sample_id'] for pair in pairs]
                    for (sample, metadata), result in zip(pairs, results):
                        metadata.update(reference_execution='gpu_batch' if result['batch_worlds']>1 else 'gpu_single',
                            actual_nworld=result['batch_worlds'],
                            requested_native_batch_worlds=len(samples),
                            reference_batch_sample_ids=result.get('batch_sample_ids') or peers,
                            native_retries=result.get('native_retries',[]),
                            reference_fallback_reason=dumps(result['native_retries']) if result.get('native_retries') else '')
                        snapshots, raw, manifold = [], [], {}
                        q, v = np.empty((0,14)), np.empty((0,12))
                        predictions = {}
                        if 'error' in result:
                            for key in ('ncontact_start','ncontact_floor_start','ncontact_ball_start','ncontact_max',
                                        'ncontact_end','established','separated','free_flight','max_slip_m_s'):
                                metadata[key]=None
                            metadata.update(mode='invalid_reference', contact_group='unknown',
                                failure_stage='native_reference', failure_reason=result['error'])
                            predictions = {method: failure(result['error'], 'invalid_reference')
                                           for method in ('scm','isaacgym')}
                        else:
                            ref, i = result['reference'], result['index']
                            q = np.asarray(ref['trajectory_qpos'][i], float)
                            v = np.asarray(ref['trajectory_qvel'][i], float)
                            if q.shape != (11,14) or v.shape != (11,12) or not np.isfinite(np.r_[q.ravel(),v.ravel()]).all():
                                raise RuntimeError('Malformed/nonfinite native reference trajectory')
                            snapshots = ref['snapshots'][i]
                            raw = ref.get('contacts_snapshots',ref.get('raw_contacts'))[i]
                            manifold = result['manifold']
                            metadata.update(classify_contacts(snapshots, settings.force_threshold, settings.slip_threshold))
                            predictions['isaacgym'] = dict(qpos=q[-1], qvel=v[-1], status='ok', failure_reason='', time=.02)
                            try:
                                predictions['scm'] = predictor.scm(sample, manifold)
                            except Exception as exc:
                                predictions['scm'] = failure(f'{type(exc).__name__}: {exc}')
                            for pred in predictions.values():
                                if pred['status'] == 'ok':
                                    try:
                                        if not np.isclose(pred['time'],.02,rtol=0,atol=1e-7):
                                            raise ValueError('Prediction horizon mismatch')
                                        pred.update(metrics(model,sample,pred,q[-1],v[-1],info['length_scale_m']))
                                    except Exception as exc:
                                        pred.update(failure(f'{type(exc).__name__}: {exc}'))
                            write_json(folder/'asset_validation.json',ref.get('asset_validation',[]))
                            timing_rows.append(dict(object=name, sample_id=metadata['sample_id'],
                                batch_first=first, batch_worlds=result['batch_worlds'],
                                reference_timings=ref.get('timings', {}), probe_timings=result['probe'].get('timings', {}),
                                reference_diagnostic=ref.get('diagnostic',{}),
                                native_retries=result.get('native_retries',[]),
                                initial_readback=ref.get('initial_readback',{}),
                                physics_configuration=ref.get('physics_configuration',{}),
                                native_warnings=ref.get('native_warnings',[]),physics_device_log=ref.get('physics_device_log',[])))
                        counts[metadata['mode']] += 1
                        save_query(folder/'queries'/(metadata['sample_id']+'.npz'), sample, metadata,
                                   predictions, q, v, manifold)
                        append_predictions(folder, metadata, predictions)
                        contacts.write(dumps(dict(sample_id=metadata['sample_id'], snapshots=snapshots,
                            raw_snapshots=raw, endpoint_force_interval_s=[.020,.022]))+'\n')
                        samples_log.write(dumps(metadata)+'\n')
                        source_rows.append(dict(sample_id=metadata['sample_id'], object=name,
                            path=metadata['source_query'], sha256=metadata['source_query_sha256'],
                            source_mode=metadata['source_mode'], mode=metadata['mode']))
                    contacts.flush(); samples_log.flush()
                    manifest['results'][name].update(n_completed=sum(counts.values()), measured_modes=dict(counts))
                    update_manifest(output, manifest, elapsed_seconds=time.perf_counter()-began)
                    write_csv(output/'source_queries.csv', source_rows)
                    write_json(folder/'reference_batches.json', [r for r in timing_rows if r['object']==name])
                    print(f'{name}: {sum(counts.values())}/{len(rows)} PhysX+SCM; modes={dict(counts)}', flush=True)
            manifest['results'][name]['status']='reference_and_scm_complete'
            del predictor
        update_manifest(output,manifest,phase='comfree')
        check_implementation(output,manifest)
        comfree_saved(output, settings, args.comfree_batch_size, manifest=manifest)
        update_manifest(output,manifest,status='complete',phase='report',elapsed_seconds=time.perf_counter()-began)
        report_directory(output,plots=not args.no_plots)
        verification = validate_saved(output)
        write_json(output/'verification.json',verification)
        update_manifest(output,manifest,status='complete' if verification['status']=='passed' else 'failed',
                        phase='finished',elapsed_seconds=time.perf_counter()-began)
    except BaseException as exc:
        update_manifest(output,manifest,status='failed',failure_reason=f'{type(exc).__name__}: {exc}',
                        traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-began)
        raise
    print(f"Completed {manifest['n_input']} frozen queries in {manifest['elapsed_seconds']:.1f}s: {output}",flush=True)
    return int(manifest['status']!='complete')


def comfree_saved(output, settings, batch_size, manifest=None, destination=None):
    from examples.mpc.fingertips.test.collision_prediction_external_contacts import ExternalContactPredictors
    source_manifest = json.loads((output/'manifest.json').read_text())
    diagnostics=[]
    for name in source_manifest['objects']:
        folder=output/name
        info,_=verify_scene_files(folder)
        engine=ExternalContactPredictors(folder/'model.xml', settings)
        model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
        paths=sorted((folder/'queries').glob('*.npz'))
        target=destination/name if destination else folder
        if destination: target.mkdir()
        for first in range(0,len(paths),batch_size):
            batch=[load_saved_query(p) for p in paths[first:first+batch_size]]
            valid=[i for i,(_,_,m,_) in enumerate(batch) if m['mode']!='invalid_reference']
            predictions={}
            if valid:
                try:
                    results=engine.comfree_batch([batch[i][1] for i in valid], [batch[i][3] for i in valid])
                except Exception as exc:
                    results=[failure(f'{type(exc).__name__}: {exc}') for _ in valid]
                predictions.update(zip(valid,results))
                diagnostics.append(dict(object=name,batch_size=batch_size,
                    first=first,**(getattr(engine,'last_comfree_timing',None) or dict(nworld=len(valid)))))
            saved_q,saved_v,ids=[],[],[]
            for i,(arrays,sample,metadata,manifold) in enumerate(batch):
                pred=predictions.get(i,failure(metadata.get('failure_reason','Invalid PhysX reference'),'invalid_reference'))
                if pred['status']=='ok':
                    try:
                        if not np.isclose(pred['time'],.02,rtol=0,atol=1e-7):
                            raise ValueError('ComFree horizon mismatch')
                        pred.update(metrics(model,sample,pred,arrays['reference_qpos'][-1],arrays['reference_qvel'][-1],info['length_scale_m']))
                    except Exception as exc:
                        pred.update(failure(f'{type(exc).__name__}: {exc}'))
                metadata=dict(metadata,actual_nworld=len(valid) if i in valid else 0)
                append_predictions(target,metadata,{'comfree':pred})
                if not destination:
                    arrays['comfree_qpos']=pred['qpos'] if pred.get('qpos') is not None else np.full(14,np.nan)
                    arrays['comfree_qvel']=pred['qvel'] if pred.get('qvel') is not None else np.full(12,np.nan)
                    with paths[first+i].with_suffix('.tmp').open('wb') as stream: np.savez_compressed(stream,**arrays)
                    paths[first+i].with_suffix('.tmp').replace(paths[first+i])
                else:
                    saved_q.append(pred['qpos'] if pred.get('qpos') is not None else np.full(14,np.nan))
                    saved_v.append(pred['qvel'] if pred.get('qvel') is not None else np.full(12,np.nan)); ids.append(metadata['sample_id'])
            if destination:
                np.savez_compressed(target/f'batch_{first:06d}.npz',sample_ids=np.asarray(ids),qpos=saved_q,qvel=saved_v)
            print(f'{name}: ComFree batch={batch_size} {min(first+batch_size,len(paths))}/{len(paths)}',flush=True)
        if manifest:
            manifest['results'][name]['status']='complete'
            update_manifest(output,manifest)
        del engine
    write_json((destination or output)/'comfree_batches.json',diagnostics)
    return diagnostics


def validate_saved(output):
    manifest=json.loads((output/'manifest.json').read_text())
    errors=[]; checked=0
    for name in manifest['objects']:
        folder=output/name
        verify_scene_files(folder)
        model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
        length=json.loads((folder/'object.json').read_text())['length_scale_m']
        with (folder/'predictions.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        grouped=defaultdict(dict)
        for row in rows:
            if row['method'] in grouped[row['sample_id']]: errors.append(f'duplicate row {row["sample_id"]}')
            grouped[row['sample_id']][row['method']]=row
        for path in sorted((folder/'queries').glob('*.npz')):
            arrays,sample,metadata,_=load_saved_query(path)
            records=grouped.get(metadata['sample_id'],{})
            if set(records)!=set(METHODS): errors.append(f'missing method {path.name}');continue
            if sha(metadata['source_query'])!=metadata['source_query_sha256']:
                errors.append(f'source changed {path.name}')
            with np.load(metadata['source_query'],allow_pickle=False) as original:
                for key in ('qpos','qvel','wrench'):
                    if not np.array_equal(sample[key],original[key]): errors.append(f'input {key} changed {path.name}')
            if metadata['mode']!='invalid_reference':
                np.testing.assert_allclose(arrays['reference_qpos'][0],sample['qpos'],atol=2e-6,rtol=0)
                np.testing.assert_allclose(arrays['reference_qvel'][0],sample['qvel'],atol=2e-6,rtol=0)
                np.testing.assert_allclose(arrays['reference_times'],np.arange(11)*.002,atol=1e-12,rtol=0)
                for method,row in records.items():
                    if row['status']!='ok': continue
                    result=metrics(model,sample,dict(qpos=arrays[method+'_qpos'],qvel=arrays[method+'_qvel']),
                                   arrays['reference_qpos'][-1],arrays['reference_qvel'][-1],length)
                    for key,value in result.items():
                        if value is None:
                            if row.get(key) not in ('',None): errors.append(f'{path.name}: {key} expected undefined')
                        elif row.get(key) in ('',None) or not np.isclose(float(row[key]),value,rtol=1e-9,atol=1e-10):
                            errors.append(f'{path.name}: {method} {key} mismatch')
            checked+=1
    if checked!=manifest['n_input']: errors.append(f'query count {checked} != {manifest["n_input"]}')
    return dict(status='passed' if not errors else 'failed',n_queries=checked,errors=errors)


def replay(args):
    path=args.query.resolve(); folder=path.parent.parent; root=folder.parent
    manifest=json.loads((root/'manifest.json').read_text())
    arrays,sample,metadata,manifold=load_saved_query(path)
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
    # Reuse original peer composition to reproduce native batch coordinate offsets.
    peers=metadata.get('reference_batch_sample_ids',[metadata['sample_id']])
    peer_queries=[load_saved_query(folder/'queries'/(peer+'.npz')) for peer in peers]
    index=peers.index(metadata['sample_id'])
    _,spec=verify_scene_files(folder)
    with setup_native(spec,args) as native:
        results=evaluate_native(native,[dict(item[1],_sample_id=peer) for peer,item in zip(peers,peer_queries)])
    result=results[index]
    if 'error' in result: raise RuntimeError(result['error'])
    ref,i=result['reference'],result['index']
    new_q=np.asarray(ref['trajectory_qpos'][i]);new_v=np.asarray(ref['trajectory_qvel'][i])
    from examples.mpc.fingertips.test.collision_prediction_external_contacts import ExternalContactPredictors
    engine=ExternalContactPredictors(folder/'model.xml',Settings(**manifest['settings']))
    predictions={'scm':engine.scm(sample,result['manifold']),
                 'comfree':engine.comfree_batch([sample],[result['manifold']])[0]}
    comparison={}
    for method,pred in predictions.items():
        comparison[method]=dict(status=pred['status'],qpos_max_abs_difference=float(np.max(np.abs(pred['qpos']-arrays[method+'_qpos'])))
                                if pred['status']=='ok' else None)
    write_json(output/'replay.json',dict(sample_id=metadata['sample_id'],
        mode_original=metadata['mode'],mode_replayed=classify_contacts(ref['snapshots'][i])['mode'],
        reference_qpos_max_abs_difference=float(np.max(np.abs(new_q-arrays['reference_qpos']))),
        reference_qvel_max_abs_difference=float(np.max(np.abs(new_v-arrays['reference_qvel']))),predictions=comparison))
    replay_arrays=dict(reference_qpos=new_q,reference_qvel=new_v)
    for method,pred in predictions.items():
        replay_arrays[method+'_qpos']=pred['qpos'] if pred.get('qpos') is not None else np.full(14,np.nan)
        replay_arrays[method+'_qvel']=pred['qvel'] if pred.get('qvel') is not None else np.full(12,np.nan)
    np.savez_compressed(output/'replay.npz',**replay_arrays)
    print((output/'replay.json').read_text(),flush=True)
    return 0


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    for name in ('run','smoke-test'):
        sub=commands.add_parser(name)
        sub.add_argument('--input-dir',type=Path,required=True)
        sub.add_argument('--output-dir',type=Path,required=True)
        sub.add_argument('--objects',nargs='+',default=['sphere','cube','cylinder','foam_brick'] if name=='smoke-test' else None)
        sub.add_argument('--limit-per-object',type=positive_int,default=8 if name=='smoke-test' else None)
        sub.add_argument('--reference-batch-size',type=positive_int,default=16)
        sub.add_argument('--comfree-batch-size',type=positive_int,default=16)
        sub.add_argument('--isaacgym-python',default=None)
        sub.add_argument('--headless',action='store_true',help='Always headless; accepted for explicitness')
        sub.add_argument('--no-plots',action='store_true')
    sub=commands.add_parser('report');sub.add_argument('--input-dir',type=Path,required=True)
    sub.add_argument('--no-plots',action='store_true')
    sub=commands.add_parser('replay');sub.add_argument('--query',type=Path,required=True)
    sub.add_argument('--output-dir',type=Path,required=True)
    sub.add_argument('--reference-batch-size',type=positive_int,default=16)
    sub.add_argument('--isaacgym-python',default=None)
    sub=commands.add_parser('comfree-batch');sub.add_argument('--input-dir',type=Path,required=True)
    sub.add_argument('--output-dir',type=Path,required=True)
    sub.add_argument('--batch-sizes',type=positive_int,nargs='+',default=[1,16,64])
    return p


def main():
    args=parser().parse_args()
    if args.command=='report':
        from examples.mpc.fingertips.test.collision_prediction_isaacgym_report import report_directory
        report_directory(args.input_dir,plots=not args.no_plots);return 0
    if args.command=='replay': return replay(args)
    if args.command=='comfree-batch':
        source=args.input_dir.resolve(); manifest=json.loads((source/'manifest.json').read_text())
        if manifest['status']!='complete':raise ValueError('Source Isaac experiment must be complete')
        if len(set(args.batch_sizes))!=len(args.batch_sizes):raise ValueError('Repeated batch sizes')
        output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=False)
        all_times=[]
        for size in args.batch_sizes:
            destination=output/f'batch_{size}';destination.mkdir()
            all_times.extend(comfree_saved(source,Settings(**manifest['settings']),size,destination=destination))
        write_json(output/'throughput.json',all_times)
        write_json(output/'manifest.json',dict(status='complete',source=str(source),source_manifest_sha256=sha(source/'manifest.json'),batch_sizes=args.batch_sizes))
        from examples.mpc.fingertips.test.collision_prediction_isaacgym_report import report_comfree_batches
        report_comfree_batches(output)
        return 0
    code=run(args)
    if args.command=='smoke-test' and not code:
        from examples.mpc.fingertips.test.collision_prediction_isaacgym_report import coverage_from_saved_contacts
        report=coverage_from_saved_contacts(args.output_dir)
        write_json(args.output_dir/'smoke_test.json',report)
        from examples.mpc.fingertips.test.collision_prediction_isaacgym_audit import audit_dataset
        audit=audit_dataset(args.output_dir,isaacgym_python=args.isaacgym_python,
                            batch_size=max(2,args.reference_batch_size))
        print(f"Native GPU audit: {audit['status']} (strict: {audit.get('strict_status','unknown')})",flush=True)
        return int(audit['status']!='passed' or bool(report.get('integrity_errors')))
    return code


if __name__=='__main__':
    raise SystemExit(main())
