"""Warm state replay with CUDA events isolating the actual GPU LM graph."""
import os,sys,time,json,statistics
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_HEADLESS']='1';os.environ['ACADOS_SOURCE_DIR']='/home/zz/zz_ws/acados'
ROOT=Path('/home/zz/scsp-robot');OUT=ROOT/'output/diagnostics/gpu_batch_lm_optimized_fp32_20261014';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
import numpy as np,mujoco,warp as wp
from scipy.spatial.transform import Rotation
from gpu_lm import GPUExactBatchLM
from examples.mpc.fingertips.test import test_scm_mujoco as mod
original_config=mod.configure_direct_force_optimizer;original_step=mod._solve_step;records=[];scaling=[]
for obj in ['foam_brick','elephant','stanford_bunny2']:
 payload=json.loads((OUT/f'trial_{obj}_000.json').read_text());rows=payload['rows'];cfg=mod.build_parser().parse_args([])
 for k,v in payload['metadata']['args'].items():setattr(cfg,k,v)
 holder={}
 def configure(param,env,args):
  opt=original_config(param,env,args);lm=GPUExactBatchLM(opt,iterations=20);lm.install();holder['lm']=lm
  for count in range(1,len(opt.sample_point)+1):lm.buffers(count)
  return opt
 def replay(param,args,env,contact,mode,**kw):
  lm=holder['lm'];opt=param.lambda_optimizer;qp=env.data_.qpos.copy();qv=env.data_.qvel.copy()
  start=wp.Event(device=lm.device,enable_timing=True);end=wp.Event(device=lm.device,enable_timing=True)
  original_solve=lm.solve;original_prepare=lm.prepare;original_capture=wp.capture_launch;original_choose=opt.choose_contact_points
  timing={}
  def prepare(*a,**k):
   st=time.perf_counter();result=original_prepare(*a,**k);timing['prepare_upload_ms']=1000*(time.perf_counter()-st);return result
  def launch(graph):
   # Flush input uploads; measure GPU execution with device-side events.
   wp.synchronize_device(lm.device);st=time.perf_counter();wp.record_event(start);original_capture(graph);wp.record_event(end);wp.synchronize_device(lm.device)
   timing['graph_wall_ms']=1000*(time.perf_counter()-st);timing['gpu_graph_ms']=wp.get_event_elapsed_time(start,end,synchronize=False)
  def solve(*a,**k):
   st=time.perf_counter();result=original_solve(*a,**k);timing['lm_adapter_ms']=1000*(time.perf_counter()-st);return result
  def choose(*a,**k):
   st=time.perf_counter();result=original_choose(*a,**k);timing['choose_ms']=1000*(time.perf_counter()-st);return result
  lm.prepare=prepare;lm.solve=solve;wp.capture_launch=launch;opt.choose_contact_points=choose
  try:
   for frac in [.25,.75]:
    idx=max(1,int((len(rows)-1)*frac));before=rows[idx-1];row=rows[idx];state=np.asarray(before['object_pose']);vel=np.asarray(before['velocity_body']);rot=Rotation.from_quat(state[[4,5,6,3]]).as_matrix()
    for rep in range(21):
     env.data_.qpos[:]=qp;env.data_.qpos[:7]=state;env.data_.qvel[:]=qv;env.data_.qvel[:3]=rot@vel[:3];env.data_.qvel[3:6]=vel[3:];env.data_.time=before['simulation_time_s'];mujoco.mj_forward(env.model_,env.data_)
     param.force_observation_pose_=None;param.force_cost_stages_.stage=row['object_cost_stage'];opt._dwell_idx=opt._dwell_best_cost=None;opt._dwell_steps=0;opt._blocked_contact_indices.clear()
     timing.clear();st=time.perf_counter();solved=original_step(param,args,env,contact,mode,**kw);full=1000*(time.perf_counter()-st)
     rec=dict(object=obj,step=idx,stage=row['object_cost_stage'],repeat=rep,warmup=rep==0,cached_planning_ms=row['planning_time_ms'],full_round_ms=full,candidates=len(opt.last_candidate_ids) if hasattr(opt,'last_candidate_ids') and opt.last_candidate_ids is not None else None,**timing)
     rec['other_step_ms']=full-rec['choose_ms'];rec['ranking_ms']=rec['choose_ms']-rec['lm_adapter_ms'];rec['readback_other_ms']=rec['lm_adapter_ms']-rec['prepare_upload_ms']-rec['graph_wall_ms']
     records.append(rec)
    print('STATE',obj,idx,json.dumps({k:statistics.mean(r[k] for r in records[-20:]) for k in ['full_round_ms','gpu_graph_ms','prepare_upload_ms','choose_ms']}),flush=True)
  finally:lm.prepare=original_prepare;lm.solve=original_solve;wp.capture_launch=original_capture;opt.choose_contact_points=original_choose
  return solved
 mod.configure_direct_force_optimizer=configure;mod._solve_step=replay
 try:mod._run_rollout(cfg,'surrogate',obj,seed=0,mass=.01,steps=1)
 finally:mod.configure_direct_force_optimizer=original_config;mod._solve_step=original_step
 (OUT/'profile_records.json').write_text(json.dumps(records,indent=2)+'\n');(OUT/'profile_scaling.json').write_text(json.dumps(scaling,indent=2)+'\n')
rs=[r for r in records if not r['warmup']];summary={k:dict(mean=statistics.mean(r[k] for r in rs),std=statistics.pstdev(r[k] for r in rs)) for k in ['cached_planning_ms','full_round_ms','gpu_graph_ms','graph_wall_ms','prepare_upload_ms','lm_adapter_ms','choose_ms','other_step_ms','ranking_ms','readback_other_ms']};summary['n']=len(rs)
(OUT/'profile_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('SUMMARY',json.dumps(summary),flush=True)
