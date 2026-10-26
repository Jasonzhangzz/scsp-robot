#!/usr/bin/env python3
"""Paired SCM(3 GS)/rigid circular-cone NCP collision predictions.

Run standalone, or reuse frozen queries with --input-dir. Predictions share
FCL initial geometry. Reports compare to BOTH the external simulator and the
NCP 2ms x 10 rollout; they do not call either one physical ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
import csv
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()
import mujoco
import numpy as np
from threadpoolctl import threadpool_limits

from examples.mpc.franka.scm.contact_dynamics import contact_snapshot
from examples.mpc.fingertips.test.collision_prediction_assets import (
    OBJECTS, MESH_OBJECTS, DEFAULT_CACHE, digest, object_vertices, prepare_asset, package_scene, scene_xml)
from examples.mpc.fingertips.test.collision_prediction_core import (
    Experiment, Settings, MODES, classify_contacts, reference_snapshot)
from examples.mpc.fingertips.test.collision_prediction_ncp import NCPPredictor
from examples.mpc.fingertips.test.collision_prediction_isaacgym_report import summarize, statistics
from examples.mpc.fingertips.test.benchmark_collision_prediction_isaacgym import (
    metrics, write_csv, positive_int)
from models.ncp_contact import NCPSettings

CONTACTNETS_COMMIT = 'c3dc21c94577ff4ed99123f0b00bfe57bb2d1d90'


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=plain, allow_nan=False)+'\n')
    temporary.replace(path)


class CPUSampler(Experiment):
    """Reuse the existing query recipes; CPU settling is outside predictions."""
    def __init__(self, model, info, settings):
        self.info, self.settings = info, settings
        self.models = {'mjwarp': model}
        self.vertices, self.equilibria, self.support_pose = object_vertices(model), {}, None

    def settle(self, sample, steps=500):
        data = mujoco.MjData(self.models['mjwarp'])
        data.qpos[:], data.qvel[:], data.xfrc_applied[:] = sample['qpos'], sample['qvel'], sample['wrench']
        for _ in range(steps): mujoco.mj_step(self.models['mjwarp'], data)
        return data.qpos.copy(), data.qvel.copy()


def cpu_reference(model, sample, settings):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:], data.xfrc_applied[:] = sample['qpos'], sample['qvel'], sample['wrench']
    qs, vs, snapshots = [data.qpos.copy()], [data.qvel.copy()], []
    began = time.perf_counter()
    for step in range(settings.reference_steps+1):
        mujoco.mj_forward(model, data)
        native, _ = contact_snapshot(model, data)
        ids = [i for i in range(data.ncon) if data.contact[i].efc_address >= 0]
        for k, contact in enumerate(ids):
            mujoco.mj_contactForce(model, data, contact, native['force'][k])
        snapshots.append(reference_snapshot(model, data, native, np.zeros(len(ids), int),
                                            step*settings.reference_dt))
        if step < settings.reference_steps:
            mujoco.mj_step(model, data)
            qs.append(data.qpos.copy()); vs.append(data.qvel.copy())
    if not np.isfinite(qs).all() or not np.isfinite(vs).all():
        raise RuntimeError('Nonfinite reference')
    return dict(status='ok', failure_reason='', qpos=qs[-1], qvel=vs[-1],
        trajectory_qpos=np.asarray(qs), trajectory_qvel=np.asarray(vs), snapshots=snapshots,
        seconds=time.perf_counter()-began)


def source_queries(source, name, limit):
    folder = source/name
    info = json.loads((folder/'object.json').read_text())
    if digest(folder/'model.xml') != info['model_sha256']:
        raise ValueError(f'Source model hash mismatch: {name}')
    for filename, checksum in info['packaged_asset_sha256'].items():
        if digest(folder/filename) != checksum: raise ValueError(f'Source asset hash mismatch: {filename}')
    rows = list(csv.DictReader((folder/'predictions.csv').open()))
    chosen, seen = [], set()
    for row in rows:
        if row['sample_id'] in seen or row['selected'] != 'True' or row['mode'] not in MODES:
            continue
        seen.add(row['sample_id']); chosen.append(row)
    # Round robin over measured regimes for meaningful small smoke subsets.
    ordered = []
    queues = {mode: [r for r in chosen if r['mode'] == mode] for mode in MODES}
    while any(queues.values()):
        for mode in MODES:
            if queues[mode]: ordered.append(queues[mode].pop(0))
    for row in ordered[:limit] if limit else ordered:
        path = folder/'queries'/(row['sample_id']+'.npz')
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        metadata = json.loads(str(arrays['metadata_json']))
        metadata.update(source_query=str(path), source_query_sha256=digest(path), source_mode=metadata['mode'])
        yield {k: arrays[k] for k in ('qpos','qvel','wrench')}, metadata, arrays


def report(output):
    rows = list(csv.DictReader((output/'predictions.csv').open()))
    manifest = json.loads((output/'manifest.json').read_text())
    summaries = []
    for reference in sorted({r['reference'] for r in rows}):
        subset = [r for r in rows if r['reference'] == reference]
        summaries.extend(dict(r, reference=reference) for r in summarize(
            subset, manifest['objects'], methods=manifest['methods']))
    write_csv(output/'summary.csv', summaries)
    timings=[]
    for name in (*manifest['objects'],'all'):
        for method in manifest['methods']:
            selected=[r for r in rows if r['reference']==manifest['methods'][-1] and r['method']==method
                      and (name=='all' or r['object']==name)]
            values=[r['seconds'] for r in selected if r['status']=='ok' and r.get('seconds') not in ('',None)]
            timings.append(dict(object=name,method=method,n_total=len(selected),
                n_failed=sum(r['status']!='ok' for r in selected),n_timed=len(values),
                **{'seconds_'+k:v for k,v in statistics(values).items()}))
    write_csv(output/'timing.csv',timings)
    lines = ['reference,method,valid,total,mean_position_mm,mean_rotation_deg,mean_pose_error']
    for reference in sorted({r['reference'] for r in rows}):
        for method in manifest['methods']:
            group = [r for r in summaries if r['reference']==reference and r['method']==method
                     and r['object']=='all' and r['mode']=='all' and r['contact_group']=='all']
            if not group: continue
            by_metric = {r['metric']:r for r in group}
            first = group[0]
            lines.append(','.join(map(str, [reference, method, first['n_valid'], first['n_total'],
                *[by_metric[k]['mean'] for k in ('position_error_mm','orientation_error_deg','pose_error')]])))
    (output/'summary.log').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


def saved_reference(arrays, original):
    method = next((m for m in ('isaacgym','mjwarp','mujoco')
                   if m in original.get('methods',[]) and m+'_qpos' in arrays),None)
    if method is None: raise ValueError('Unrecognized saved reference engine')
    if method+'_qpos' not in arrays or method+'_qvel' not in arrays:
        raise ValueError('Saved query lacks its native reference endpoint')
    q, v = arrays[method+'_qpos'], arrays[method+'_qvel']
    if not np.isfinite(np.r_[q,v]).all(): raise ValueError('Invalid saved reference')
    return method, dict(status='ok', failure_reason='', qpos=q, qvel=v, seconds=0.,
                        trajectory_qpos=arrays.get('reference_qpos', q[None]),
                        trajectory_qvel=arrays.get('reference_qvel', v[None]))


def save_prediction(path, sample, metadata, manifold, predictions):
    arrays = dict(sample, metadata_json=np.array(json.dumps(metadata, default=plain, allow_nan=False)))
    arrays.update({'initial_'+k:v for k,v in manifold.items()})
    for method, result in predictions.items():
        if result['status']=='ok':
            for key in ('qpos','qvel','trajectory_qpos','trajectory_qvel'):
                if key in result: arrays[method+'_'+key] = result[key]
        arrays[method+'_diagnostics_json'] = np.array(json.dumps(result.get('diagnostics', []), allow_nan=False))
        for i, problem in enumerate(result.get('problems', [])):
            arrays.update({f'{method}_step{i}_{k}':v for k,v in problem.items()})
    np.savez_compressed(path, **arrays)


def run(args):
    if args.reference_engine=='saved' and not args.input_dir:
        raise ValueError('--reference-engine saved requires --input-dir')
    if not np.isfinite(args.candidate_margin) or args.candidate_margin<0:
        raise ValueError('--candidate-margin must be finite and nonnegative')
    if args.coacd_max_vertices<4:
        raise ValueError('--coacd-max-vertices must be >= 4')
    if args.limit_per_object and not args.input_dir:
        raise ValueError('--limit-per-object requires --input-dir; standalone runs use --samples-per-mode')
    settings, ncp_settings = Settings(), NCPSettings(args.ncp_tolerance, args.ncp_max_evaluations)
    original = json.loads((args.input_dir/'manifest.json').read_text()) if args.input_dir else None
    if original:
        if original['status']!='complete': raise ValueError('Source must be complete')
        settings = Settings(**original['settings'])
        if (settings.horizon, settings.reference_dt, settings.reference_steps)!=(.02,.002,10):
            raise ValueError('Expected .02 horizon and .002 x 10 reference')
    names = args.objects or (original['objects'] if original else ['sphere','cube','cylinder'])
    if len(set(names))!=len(names) or not set(names)<=set(OBJECTS): raise ValueError('Invalid objects')
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    reference_method = args.reference_engine
    if reference_method=='saved':
        reference_method = next((m for m in ('isaacgym','mjwarp','mujoco') if m in original.get('methods',[])),None)
        if reference_method is None: raise ValueError('Unrecognized source reference')
    methods = ['scm','ncp','ncp_substeps',reference_method]
    manifest = dict(status='running', objects=names, methods=methods, settings=settings.to_dict(),
        ncp_settings=asdict(ncp_settings), arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        python=sys.executable, packages={p:importlib.metadata.version(p) for p in
            ('numpy','scipy','mujoco','coacd','python-fcl')},
        contact_nets_commit=CONTACTNETS_COMMIT,
        contact_nets_finding='Stewart-Trinkle polyhedral LCP + .001 I, Lemke; no exact circular-cone NCP solver',
        law='Signorini normal complementarity + circular Coulomb maximum dissipation; full coupled W; no compliance',
        geometry='FCL convex parts + analytic plane/vertex manifolds; cylinder 96-sided; no component deduplication',
        discretization='first-order velocity-level impulse time stepping; frozen geometry per substep, zero restitution',
        timings='CPU host wall time, includes smooth dynamics/solve/integration; NCP substeps include later collisions; not GPU throughput',
        initial_collision_seconds_recorded_separately=True, candidate_margin_m=args.candidate_margin,
        labels='native reference forces/geometry; requested modes are recipes, not measured labels',
        source=str(args.input_dir.resolve()) if args.input_dir else None, results={}, implementation_sha256={})
    if args.input_dir: manifest['source_manifest_sha256']=digest(args.input_dir/'manifest.json')
    implementation=output/'implementation'; implementation.mkdir()
    for filename in [__file__, str(ROOT/'models/ncp_contact.py'),str(ROOT/'models/soft_box_contact.py'),
                     *[str(ROOT/'examples/mpc/franka/scm'/f) for f in
                       ('runtime_env.py','contact_dynamics.py','benchmark_utils.py','fidelity_metrics.py')],
                     *[str(Path(__file__).with_name(f)) for f in
                       ('benchmark_collision_prediction_isaacgym.py','collision_prediction_isaacgym_native.py',
                        'collision_prediction_isaacgym_worker.py','collision_prediction_isaacgym_report.py')],
                     str(Path(__file__).with_name('collision_prediction_ncp.py')),
                     str(Path(__file__).with_name('collision_prediction_external_contacts.py')),
                     str(Path(__file__).with_name('collision_prediction_core.py')),
                     str(Path(__file__).with_name('collision_prediction_assets.py'))]:
        relative=Path(filename).resolve().relative_to(ROOT)
        target=implementation/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(filename,target); manifest['implementation_sha256'][str(relative)]=digest(filename)
    write_json(output/'manifest.json', manifest)
    all_rows, began = [], time.perf_counter()
    try:
        with (output/'results.log').open('w', buffering=1) as log:
            for name in names:
                folder=output/name; folder.mkdir(); (folder/'queries').mkdir()
                if args.input_dir:
                    info=json.loads((args.input_dir/name/'object.json').read_text())
                    shutil.copy2(args.input_dir/name/'model.xml',folder/'model.xml')
                    if (args.input_dir/name/'assets').exists():
                        shutil.copytree(args.input_dir/name/'assets',folder/'assets')
                    queries=source_queries(args.input_dir,name,args.limit_per_object)
                else:
                    asset=prepare_asset(name,args.asset_cache,max_vertices=args.coacd_max_vertices) if name in MESH_OBJECTS else None
                    xml,info=scene_xml(name,asset)
                    package_scene(xml,folder)
                    model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
                    sampler=CPUSampler(model,info,settings)
                    def generated():
                        rng=np.random.default_rng(np.random.SeedSequence([args.seed,OBJECTS.index(name)]))
                        for attempt in range(args.samples_per_mode):
                            for mode in MODES:
                                sample=sampler.sample(name,mode,rng,attempt)
                                metadata={k:v for k,v in sample.items() if k not in ('qpos','qvel','wrench')}
                                metadata.update(sample_id=f'{name}_s{args.seed}_{mode}_{attempt:06d}',
                                                object=name,seed=args.seed,selected=True)
                                yield sample,metadata,{}
                    queries=generated()
                info.update(model_sha256=digest(folder/'model.xml'), packaged_asset_sha256={
                    str(p.relative_to(folder)):digest(p) for p in (folder/'assets').glob('*')})
                write_json(folder/'object.json',info)
                model=mujoco.MjModel.from_xml_path(str(folder/'model.xml'))
                model.opt.timestep=settings.reference_dt
                predictor=NCPPredictor(folder/'model.xml',settings,ncp_settings,args.candidate_margin)
                native_context=nullcontext(None)
                if args.reference_engine=='isaacgym':
                    from examples.mpc.fingertips.test.collision_prediction_isaacgym_native import export_scene_spec, IsaacGymNative
                    spec=export_scene_spec(folder/'model.xml',folder/'physx_assets')
                    native_context=IsaacGymNative(spec,batch_size=1,python=args.isaacgym_python,
                                                  log_path=folder/'physx_worker.log')
                counts, failures, nqueries=Counter(), Counter(), 0
                with native_context as native, (folder/'solver_diagnostics.jsonl').open('w',buffering=1) as solver_log:
                    if native is not None: manifest.setdefault('native_workers',{})[name]=native.metadata
                    for sample,metadata,arrays in queries:
                        nqueries+=1
                        if args.reference_engine=='saved':
                            actual_reference,reference=saved_reference(arrays,original)
                            if actual_reference!=reference_method: raise ValueError('Mixed source reference engines')
                        elif native is None:
                            reference=cpu_reference(model,sample,settings)
                            metadata.update(classify_contacts(reference['snapshots']))
                        else:
                            ref=native.reference(sample['qpos'],sample['qvel'],sample['wrench'])
                            if ref.get('native_warnings'): raise RuntimeError(str(ref['native_warnings']))
                            q,v=np.asarray(ref['trajectory_qpos'])[0],np.asarray(ref['trajectory_qvel'])[0]
                            if not np.isfinite(q).all() or not np.isfinite(v).all(): raise RuntimeError('Nonfinite PhysX reference')
                            reference=dict(status='ok',failure_reason='',qpos=q[-1],qvel=v[-1],
                                trajectory_qpos=q,trajectory_qvel=v,seconds=ref.get('elapsed_seconds'),snapshots=ref['snapshots'][0])
                            metadata.update(classify_contacts(reference['snapshots']))
                        counts[metadata['mode']]+=1
                        clock=time.perf_counter()
                        collision_error=None
                        try: manifold=predictor.initial_manifold(sample)
                        except Exception as exc:
                            manifold={}
                            collision_error=f'{type(exc).__name__}: {exc}'
                        metadata.update(initial_collision_seconds=time.perf_counter()-clock,
                            n_fcl_candidates=len(manifold.get('gap',[])),
                            n_fcl_touching=int(np.count_nonzero(np.asarray(manifold.get('gap',[]))<=0)),
                            collision_failure=collision_error)
                        if collision_error:
                            predictions={m:dict(status='failed',failure_reason=collision_error,seconds=None)
                                         for m in ('scm','ncp','ncp_substeps')}
                            predictions[reference_method]=reference
                        else:
                            predictions={'scm':predictor.scm(sample,manifold),
                                'ncp':predictor.ncp(sample,manifold,steps=1),
                                'ncp_substeps':predictor.ncp(sample,manifold,steps=settings.reference_steps),
                                reference_method:reference}
                        for method,prediction in predictions.items():
                            if prediction['status']!='ok': failures[method]+=1
                            diagnostics=prediction.get('diagnostics',[])
                            solver_log.write(json.dumps(dict(sample_id=metadata['sample_id'],method=method,
                                status=prediction['status'],failure_reason=prediction.get('failure_reason'),
                                diagnostics=diagnostics),allow_nan=False)+'\n')
                        save_prediction(folder/'queries'/(metadata['sample_id']+'.npz'),
                            {k:sample[k] for k in ('qpos','qvel','wrench')}, metadata, manifold, predictions)
                        for refname in (reference_method,'ncp','ncp_substeps'):
                            target=predictions[refname]
                            for method,prediction in predictions.items():
                                row=dict(metadata,method=method,reference=refname,status=prediction['status'],
                                    failure_reason=prediction.get('failure_reason',''),seconds=prediction.get('seconds'))
                                if target['status']!='ok':
                                    row.update(status='invalid_reference',failure_reason=f'{refname} reference failed: '+target['failure_reason'])
                                elif prediction['status']=='ok':
                                    row.update(metrics(model,sample,prediction,target['qpos'],target['qvel'],info['length_scale_m']))
                                diagnostics=prediction.get('diagnostics',[])
                                row.update(ncp_max_residual_m_s=max((d['residual_m_s'] for d in diagnostics),default=None),
                                           ncp_evaluations=sum(d['evaluations'] for d in diagnostics) if diagnostics else None)
                                all_rows.append(row); log.write(json.dumps(row,default=plain,allow_nan=False)+'\n')
                        print(json.dumps(dict(object=name,sample_id=metadata['sample_id'],mode=metadata['mode'],
                            statuses={k:v['status'] for k,v in predictions.items()},
                            residuals={k:max((d['residual_m_s'] for d in v.get('diagnostics',[])),default=0.)
                                       for k,v in predictions.items()})),flush=True)
                        manifest['results'][name]=dict(n_queries=nqueries,measured_modes=dict(counts),failures=dict(failures),
                            fcl_independent_gjk_retries=predictor.collision.independent_gjk_retries,
                            sphere_witness_repairs=predictor.collision.sphere_witness_repairs)
                        write_json(output/'manifest.json',manifest)
                write_csv(folder/'predictions.csv',[r for r in all_rows if r['object']==name])
        write_csv(output/'predictions.csv',all_rows)
        report(output)
        manifest.update(status='complete',elapsed_seconds=time.perf_counter()-began,
                        n_prediction_failures=sum(r['status']!='ok' for r in all_rows if r['reference']==reference_method))
        write_json(output/'manifest.json',manifest)
    except BaseException as exc:
        if all_rows: write_csv(output/'predictions.csv',all_rows)
        manifest.update(status='failed',failure_reason=f'{type(exc).__name__}: {exc}')
        write_json(output/'manifest.json',manifest)
        raise


def replay(args):
    path=args.query.resolve(); root=path.parents[2]
    manifest=json.loads((root/'manifest.json').read_text())
    info=json.loads((path.parents[1]/'object.json').read_text())
    if digest(path.parents[1]/'model.xml')!=info['model_sha256']:
        raise ValueError('Replay model hash mismatch')
    for filename,checksum in info['packaged_asset_sha256'].items():
        if digest(path.parents[1]/filename)!=checksum: raise ValueError('Replay asset hash mismatch')
    with np.load(path,allow_pickle=False) as a: arrays={k:a[k].copy() for k in a.files}
    predictor=NCPPredictor(path.parents[1]/'model.xml',Settings(**manifest['settings']),
        NCPSettings(**manifest['ncp_settings']),manifest['candidate_margin_m'])
    sample={k:arrays[k] for k in ('qpos','qvel','wrench')}
    manifold={k[len('initial_'):]:v for k,v in arrays.items() if k.startswith('initial_')}
    results={}
    for method,steps in (('ncp',1),('ncp_substeps',10)):
        prediction=predictor.ncp(sample,manifold,steps)
        result=dict(status=prediction['status'],diagnostics=prediction['diagnostics'])
        if prediction['status']=='ok' and method+'_qpos' in arrays:
            result.update(max_qpos_difference=float(np.max(np.abs(prediction['qpos']-arrays[method+'_qpos']))),
                          max_qvel_difference=float(np.max(np.abs(prediction['qvel']-arrays[method+'_qvel']))))
        results[method]=result
    print(json.dumps(results,indent=2,allow_nan=False))


def parser():
    p=argparse.ArgumentParser(description=__doc__); commands=p.add_subparsers(dest='command',required=True)
    r=commands.add_parser('run')
    r.add_argument('--input-dir',type=Path)
    r.add_argument('--output-dir',type=Path,required=True)
    r.add_argument('--objects',nargs='+',choices=OBJECTS)
    r.add_argument('--samples-per-mode',type=positive_int,default=10)
    r.add_argument('--limit-per-object',type=positive_int)
    r.add_argument('--seed',type=int,default=42)
    r.add_argument('--reference-engine',choices=('mujoco','isaacgym','saved'),default='mujoco')
    r.add_argument('--isaacgym-python',type=Path)
    r.add_argument('--asset-cache',type=Path,default=DEFAULT_CACHE)
    r.add_argument('--coacd-max-vertices',type=positive_int,default=64,
                   help='Explicit per-component CoACD decimation budget; reference and predictors share these assets')
    r.add_argument('--candidate-margin',type=float,default=.01)
    r.add_argument('--ncp-tolerance',type=float,default=1e-6,
                   help='Natural-map convergence threshold in m/s; failed solves do not publish endpoints')
    r.add_argument('--ncp-max-evaluations',type=positive_int,default=500)
    r=commands.add_parser('report');r.add_argument('--input-dir',type=Path,required=True)
    r=commands.add_parser('replay');r.add_argument('--query',type=Path,required=True)
    return p


def main():
    args=parser().parse_args()
    with threadpool_limits(limits=1):
        if args.command=='run': run(args)
        elif args.command=='report': report(args.input_dir)
        else: replay(args)


if __name__=='__main__': main()
