#!/usr/bin/env python3
"""Post-hoc ComFree nworld benchmark on the exact completed SCM queries.

No sampling, SCM computation or reference rollout is performed here. All batch
sizes use the same frozen input IDs, wrenches, labels and MJWarp endpoints.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()
import mujoco
import numpy as np
from examples.mpc.franka.scm.benchmark_utils import Backend, UPSTREAM_COMMIT
from examples.mpc.fingertips.test.collision_prediction_batch import BatchedMjwarpReference
from examples.mpc.fingertips.test.collision_prediction_core import (
    MODES, METRICS, RELATIVE_METRICS, Settings, make_data, native_diagnostics,
    pose_errors, relative_errors, world_velocity)
from examples.mpc.fingertips.test.benchmark_collision_prediction import statistics, write_csv


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ComfreeBatch:
    def __init__(self, folder, settings):
        self.model = mujoco.MjModel.from_xml_path(str(folder / "model.xml"))
        self.model.opt.timestep = settings.horizon
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        self.settings = settings
        self.backend = Backend("comfree", self.model, make_data(self.model), device=settings.device,
            stiffness=settings.contact_stiffness, damping=settings.contact_damping,
            nconmax=settings.nconmax, njmax=settings.njmax, collect_contacts=False)
        self.api, self.wp = self.backend.api, self.backend.wp
        self.warmed = set()
        self.diagnostics = []

    # The shared uploader constructs independent qpos/qvel/xfrc per world.
    new_data = BatchedMjwarpReference.new_data

    def step(self, samples):
        with self.wp.ScopedDevice(self.backend.device):
            if len(samples) not in self.warmed:
                with native_diagnostics() as warm_warnings:
                    warm = self.new_data(samples)
                    self.api.step(self.backend.wmodel, warm)
                    self.wp.synchronize_device(self.backend.device)
                self.warmed.add(len(samples))
                self.diagnostics.append(dict(kind="warmup", nworld=len(samples), warnings=warm_warnings))
            began = time.perf_counter()
            data = self.new_data(samples)
            self.wp.synchronize_device(self.backend.device)
            with native_diagnostics() as warnings:
                clock = time.perf_counter()
                self.api.step(self.backend.wmodel, data)
                self.wp.synchronize_device(self.backend.device)
                physics = time.perf_counter()-clock
            q, v, times = data.qpos.numpy(), data.qvel.numpy(), data.time.numpy()
            if (int(data.nacon.numpy()[0]) > data.naconmax
                    or int(data.ncollision.numpy()[0]) > data.naconmax
                    or np.any(data.nefc.numpy() > data.njmax)):
                raise RuntimeError("ComFree GPU buffer overflow")
            if warnings:
                raise RuntimeError("Native collision warning: " + "; ".join(warnings))
            if not np.isfinite(q).all() or not np.isfinite(v).all() or not np.allclose(times, .02, atol=1e-7, rtol=0):
                raise RuntimeError("Nonfinite prediction or incorrect horizon")
            execution = "gpu_batch" if len(samples)>1 else "gpu_single"
            results = [dict(qpos=q[i].astype(float), qvel=v[i].astype(float), status="ok",
                            execution=execution, actual_nworld=len(samples), failure_reason="") for i in range(len(samples))]
            self.diagnostics.append(dict(kind="measured", nworld=len(samples), execution=execution,
                physics_seconds=physics, end_to_end_seconds=time.perf_counter()-began))
            return results

    def evaluate(self, samples):
        try:
            return self.step(samples)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self.diagnostics.append(dict(kind="retry", nworld=len(samples), reason=reason))
            if len(samples)>1:
                k=len(samples)//2
                return self.evaluate(samples[:k])+self.evaluate(samples[k:])
            return [dict(status="failed", execution="gpu_single", actual_nworld=1, failure_reason=reason)]


def metrics(model, sample, prediction, length):
    ref = make_data(model, sample["reference_qpos"], sample["reference_qvel"])
    pred = make_data(model, prediction["qpos"], prediction["qvel"])
    ref_velocity = world_velocity(model, ref)
    error = pose_errors(pred.qpos[:7], ref.qpos[:7], length)
    dv = world_velocity(model, pred)-ref_velocity
    error.update(linear_velocity_error_m_s=float(np.linalg.norm(dv[:3])),
                 angular_velocity_error_rad_s=float(np.linalg.norm(dv[3:])))
    error.update(relative_errors(error, pose_errors(ref.qpos[:7], sample["qpos"][:7], length), ref_velocity))
    error["serial_qpos_max_abs_difference"] = float(np.max(np.abs(pred.qpos-sample["serial_qpos"])))
    error["serial_qvel_max_abs_difference"] = float(np.max(np.abs(pred.qvel-sample["serial_qvel"])))
    error.update({"serial_"+key: value for key, value in pose_errors(pred.qpos[:7], sample["serial_qpos"][:7], length).items()})
    return error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1,16,64])
    p.add_argument("--objects", nargs="+")
    p.add_argument("--limit", type=int, help="Smoke-only limit per object; omitted for all 8000 primary queries")
    args=p.parse_args()
    if min(args.batch_sizes)<1 or len(set(args.batch_sizes))!=len(args.batch_sizes):
        raise ValueError("Batch sizes must be positive and unique")
    source=args.input_dir.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    original=json.loads((source/"manifest.json").read_text())
    if original["status"]!="complete":
        raise RuntimeError("Wait for the original SCM experiment to complete")
    settings=Settings(**original["settings"])
    names=args.objects or original["objects"]
    manifest=dict(status="running", source=str(source), source_manifest_sha256=sha(source/"manifest.json"),
        batch_sizes=args.batch_sizes, objects=names, settings=original["settings"], upstream_commit=UPSTREAM_COMMIT,
        selection="exact same selected four-mode primary query IDs for every batch size; no resampling",
        timing="one 20ms step, synchronized host wall time; warmup excluded; no CUDA graph; allocation/upload separate",
        source_script_sha256=sha(__file__), results={})
    shutil.copy2(__file__,out/Path(__file__).name)
    def save(): (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    save(); began=time.perf_counter(); catalog=[]; summaries=[]; timings=[]; all_rows=[]
    with (out/"results.log").open("w") as log:
        for name in names:
            folder=source/name; info=json.loads((folder/"object.json").read_text())
            if sha(folder/"model.xml")!=info["model_sha256"]: raise ValueError("Source model hash mismatch")
            for asset, checksum in info["packaged_asset_sha256"].items():
                if sha(folder/asset)!=checksum: raise ValueError("Source geometry hash mismatch")
            rows=list(csv.DictReader((folder/"predictions.csv").open()))
            scm={r["sample_id"]:r for r in rows if r["method"]=="scm"}
            selected=[r for r in rows if r["method"]=="mjwarp" and r["selected"]=="True" and r["mode"] in MODES and r["status"]=="ok"]
            if args.limit: selected=selected[:args.limit]
            samples=[]
            for row in selected:
                if row["sample_id"] not in scm: raise ValueError("SCM query has not completed")
                path=folder/"queries"/(row["sample_id"]+".npz")
                with np.load(path,allow_pickle=False) as a:
                    sample={key:a[key].copy() for key in ("qpos","qvel","wrench")}
                    sample.update(reference_qpos=a["mjwarp_qpos"].copy(),reference_qvel=a["mjwarp_qvel"].copy(),
                                  serial_qpos=a["comfree_qpos"].copy(),serial_qvel=a["comfree_qvel"].copy())
                samples.append(sample)
                catalog.append(dict(object=name,sample_id=row["sample_id"],query=str(path),sha256=sha(path),
                    mode=row["mode"],contact_group=row["contact_group"],geometry=row["geometry"],scm_status=scm[row["sample_id"]]["status"]))
            engine=ComfreeBatch(folder,settings)
            objout=out/name;objout.mkdir(); objrows=[]
            for size in args.batch_sizes:
                engine.diagnostics=[]; predicted=[]; run_start=time.perf_counter()
                for offset in range(0,len(samples),size): predicted.extend(engine.evaluate(samples[offset:offset+size]))
                rollout_wall=time.perf_counter()-run_start
                current=[]
                for row,sample,result in zip(selected,samples,predicted):
                    record={key:row[key] for key in ("sample_id","object","seed","mode","contact_group","geometry","force_N","initial_speed_m_s")}
                    record.update(batch_size=size,method="comfree",selected="True",**{k:v for k,v in result.items() if k not in ("qpos","qvel")})
                    if result["status"]=="ok": record.update(metrics(engine.model,sample,result,info["length_scale_m"]))
                    current.append(record);log.write(json.dumps(record,allow_nan=False)+"\n")
                log.flush();objrows.extend(current);all_rows.extend(current)
                np.savez_compressed(objout/f"batch_{size}.npz",sample_ids=np.array([r['sample_id'] for r in selected]),
                    qpos=np.array([r.get('qpos',np.full(14,np.nan)) for r in predicted]),
                    qvel=np.array([r.get('qvel',np.full(12,np.nan)) for r in predicted]))
                (objout/f"batch_{size}_diagnostics.json").write_text(json.dumps(engine.diagnostics,indent=2)+"\n")
                good=[d for d in engine.diagnostics if d['kind']=='measured']
                retry=sum(d['kind']=='retry' for d in engine.diagnostics)
                physics=sum(d['physics_seconds'] for d in good)
                timing=dict(object=name,batch_size=size,n_queries=len(samples),n_valid=sum(r['status']=='ok' for r in current),
                    n_native_multiworld=sum(r['status']=='ok' and r['actual_nworld']>1 for r in current),
                    retry_batches=retry,physics_seconds=physics,rollout_wall_including_warmup_seconds=rollout_wall,
                    n_measured_batch_calls=len(good),
                    **{"batch_step_"+k:v for k,v in statistics([d['physics_seconds'] for d in good]).items()},
                    clean_timing=retry==0,physics_queries_per_second=len(samples)/physics if physics and retry==0 else None)
                timings.append(timing)
                print(json.dumps(timing),flush=True)
            write_csv(objout/'predictions.csv',[{key:r.get(key) for key in sorted(set().union(*(r.keys() for r in objrows)))} for r in objrows])
            manifest['results'][name]=dict(n_primary=len(samples),status='complete');save()
    for size in args.batch_sizes:
        for name in [*names,'all']:
            for mode in [*MODES,'all']:
                for group in ['0','1','2','>=3','all']:
                    subset=[r for r in all_rows if r['batch_size']==size and (name=='all' or r['object']==name)
                            and (mode=='all' or r['mode']==mode) and (group=='all' or r['contact_group']==group)]
                    for metric in (*METRICS,*RELATIVE_METRICS,'serial_position_error_mm','serial_orientation_error_deg','serial_qpos_max_abs_difference','serial_qvel_max_abs_difference'):
                        valid=[r[metric] for r in subset if r['status']=='ok' and r.get(metric) is not None]
                        summaries.append(dict(object=name,mode=mode,contact_group=group,batch_size=size,metric=metric,
                            n_total=len(subset),n_failed=sum(r['status']!='ok' for r in subset),n_defined=len(valid),**statistics(valid)))
    for item in timings:
        baseline=next(t for t in timings if t['object']==item['object'] and t['batch_size']==1) if 1 in args.batch_sizes else None
        item['physics_speedup_vs_1']=(baseline['physics_seconds']/item['physics_seconds']
            if baseline and baseline['clean_timing'] and item['clean_timing'] and item['physics_seconds'] else None)
    write_csv(out/'source_queries.csv',catalog);write_csv(out/'summary.csv',summaries);write_csv(out/'throughput.csv',timings)
    geometry_rows=[]
    for name in names:
        geometries=sorted({r['geometry'].removesuffix('_airborne') for r in all_rows if r['object']==name})
        for size in args.batch_sizes:
            for geometry in geometries:
                subset=[r for r in all_rows if r['object']==name and r['batch_size']==size
                        and r['geometry'].removesuffix('_airborne')==geometry]
                for metric in (*METRICS,*RELATIVE_METRICS):
                    valid=[r[metric] for r in subset if r['status']=='ok' and r.get(metric) is not None]
                    geometry_rows.append(dict(object=name,geometry=geometry,batch_size=size,metric=metric,
                        n_total=len(subset),n_failed=sum(r['status']!='ok' for r in subset),n_defined=len(valid),**statistics(valid)))
    write_csv(out/'geometry_summary.csv',geometry_rows)
    with (out/'summary.log').open('w') as log:
        for row in [*timings,*summaries]:log.write(json.dumps(row,allow_nan=False)+'\n')
    lines=['# Saved-query ComFree GPU batch experiment','','Same frozen SCM queries and MJWarp references for every batch size.','',
        '| Object | Batch | Valid / total | Native multiworld queries | Retries | Clean physics speedup |',
        '|---|---|---|---|---|---|']
    for r in timings:lines.append(f"| {r['object']} | {r['batch_size']} | {r['n_valid']}/{r['n_queries']} | {r['n_native_multiworld']} | {r['retry_batches']} | {r['physics_speedup_vs_1']} |")
    lines.extend(['','Timings exclude input allocation/uploads and error computation; retries invalidate speedup estimates.','Full absolute/relative errors and serial agreement are in summary.csv and per-object predictions.csv.'])
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    manifest.update(status='complete',elapsed_seconds=time.perf_counter()-began,n_unique_queries=len(catalog),
                    n_predictions=len(all_rows),n_failed=sum(r['status']!='ok' for r in all_rows));save()


if __name__=='__main__': main()
