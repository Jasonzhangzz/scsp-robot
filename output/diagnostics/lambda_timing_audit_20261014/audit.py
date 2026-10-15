"""Replay sampled states from successful cached trials; do not alter paper data."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[key]='1'
os.environ['MUJOCO_HEADLESS']='1';os.environ['ACADOS_SOURCE_DIR']='/home/zz/zz_ws/acados'
import sys,json,time,copy,statistics
from pathlib import Path
ROOT=Path('/home/zz/scsp-robot'); OUT=ROOT/'output/diagnostics/lambda_timing_audit_20261014'
SNAP=ROOT/'output/benchmarks/scm_solvers_7x5_20261013_211740/source/scsp-robot'
sys.path.insert(0,str(SNAP))
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from examples.mpc.fingertips.test import test_scm_mujoco as mod
records=[]
for method in ['ipopt','acados','qp','lcp']:
 for obj in ['foam_brick','elephant']:
  d=ROOT/'output/diagnostics/cso_contact_guidance_7x5_20261013' if method in ['qp','lcp'] else ROOT/'output/benchmarks/scm_solvers_7x5_20261013_211740'/method
  pattern=f'trial_{method}_{obj}_*.json' if method in ['qp','lcp'] else f'trial_{obj}_*.json'
  payload=None
  for path in sorted(d.glob(pattern)):
   x=json.loads(path.read_text())
   if x['metadata']['trial_summary']['success']:payload=x;break
  if payload is None: raise RuntimeError('No successful trial '+method+' '+obj)
  rows=payload['rows']; summary=payload['metadata']['trial_summary']; seed=summary['seed']
  cfgdict=payload['metadata'].get('args',payload['metadata'].get('protocol',{}))
  cfg=mod.build_parser().parse_args([])
  for k,v in cfgdict.items():setattr(cfg,k,v)
  cfg.viewer=False;cfg.solver=method if method in ['ipopt','acados'] else 'acados'
  original=mod._solve_step
  def replay(param,args,env,contact,mode,**kwargs):
   opt=param.lambda_optimizer
   initial_qpos=env.data_.qpos.copy(); initial_qvel=env.data_.qvel.copy()
   orig_choose=opt.choose_contact_points;orig_candidate=opt._solve_optimization
   choose_times=[];candidate_times=[];candidate_args=[]
   def candidate(**kw):
    st=time.perf_counter()
    try:return orig_candidate(**kw)
    finally:candidate_times.append(1000*(time.perf_counter()-st));candidate_args.append(kw)
   def choose(*aa,**kw):
    st=time.perf_counter()
    try:return orig_choose(*aa,**kw)
    finally:choose_times.append(1000*(time.perf_counter()-st))
   opt._solve_optimization=candidate;opt.choose_contact_points=choose
   for fraction in [.25,.75]:
    index=max(1,int((len(rows)-1)*fraction));before=rows[index-1]; targetrow=rows[index]
    state=np.asarray(before['object_pose']);vel=np.asarray(before['velocity_body']);rot=Rotation.from_quat(state[[4,5,6,3]]).as_matrix()
    for repeat in range(4):
     env.data_.qpos[:]=initial_qpos;env.data_.qpos[:7]=state
     env.data_.qvel[:]=initial_qvel;env.data_.qvel[:3]=rot@vel[:3];env.data_.qvel[3:6]=vel[3:]
     env.data_.time=float(before['simulation_time_s']);mujoco.mj_forward(env.model_,env.data_)
     param.force_observation_pose_=None
     param.force_cost_stages_.stage=targetrow['object_cost_stage']
     opt._dwell_idx=opt._dwell_best_cost=None;opt._dwell_steps=0
     opt._blocked_contact_indices.clear()
     choose_times.clear();candidate_times.clear();candidate_args.clear()
     st=time.perf_counter(); solved=original(param,args,env,contact,mode,**kwargs); total=1000*(time.perf_counter()-st)
     rec=dict(method=method,object=obj,seed=seed,source=str(path.relative_to(ROOT)),sample_step=index,fraction=fraction,repeat=repeat,warmup=repeat==0,cached_round_ms=targetrow['planning_time_ms'],replay_full_round_ms=total,lambda_choose_ms=sum(choose_times),candidate_count=len(candidate_times),candidate_total_ms=sum(candidate_times),candidate_mean_ms=statistics.mean(candidate_times) if candidate_times else None,candidate_max_ms=max(candidate_times,default=None),backend=opt.last_solver_status,stage=param.force_cost_stages_.stage)
     records.append(rec);print(json.dumps(rec),flush=True)
   opt._solve_optimization=orig_candidate;opt.choose_contact_points=orig_choose
   return solved
  mod._solve_step=replay
  try:mod._run_rollout(cfg,method if method in ['qp','lcp'] else 'surrogate',obj,seed=seed,mass=.01,steps=1)
  finally:mod._solve_step=original
  (OUT/'records.json').write_text(json.dumps(records,indent=2)+'\n')
report={}
for method in ['ipopt','acados','qp','lcp']:
 rs=[r for r in records if r['method']==method and not r['warmup']]
 report[method]={k:statistics.mean(r[k] for r in rs) for k in ['cached_round_ms','replay_full_round_ms','lambda_choose_ms','candidate_count','candidate_mean_ms','candidate_total_ms']};report[method]['n_states']=len({(r['object'],r['sample_step']) for r in rs});report[method]['n_repetitions']=len(rs)
(OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('SUMMARY',json.dumps(report),flush=True)
