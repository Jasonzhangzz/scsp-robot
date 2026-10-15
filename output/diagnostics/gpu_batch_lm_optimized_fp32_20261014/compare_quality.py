"""Validate GPU predictions/objective at multiple states of completed LM trials."""
import os,sys,json
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['MUJOCO_HEADLESS']='1';os.environ['ACADOS_SOURCE_DIR']='/home/zz/zz_ws/acados'
ROOT=Path('/home/zz/scsp-robot');OUT=ROOT/'output/diagnostics/gpu_batch_lm_optimized_fp32_20261014';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from gpu_lm import GPUExactBatchLM
from gpu_lm_reference import GPUExactBatchLM as ReferenceLM
from examples.mpc.fingertips.test import test_scm_mujoco as mod
original_configure=mod.configure_direct_force_optimizer;original_step=mod._solve_step
results=[]
for obj in ['foam_brick','elephant','stanford_bunny2']:
 payload=json.loads((OUT/f'trial_{obj}_000.json').read_text());rows=payload['rows'];cfg=mod.build_parser().parse_args([])
 for k,v in payload['metadata']['args'].items():setattr(cfg,k,v)
 holder={}
 def configure(param,env,args):
  opt=original_configure(param,env,args);lm=GPUExactBatchLM(opt,iterations=20);lm.install();lm.validate=True;lm.collect_stats=True;holder['lm']=lm
  reference=ReferenceLM(opt,iterations=20);reference.validate=False;holder['reference']=reference
  original_solve=lm.solve
  def compare(*a,**k):
   result=original_solve(*a,**k);ref=reference.solve(*a,**k);opt.last_solver_status='batch-lm-gpu-analytic-fp32'
   best=float(np.min(result[1]));best_ref=float(np.min(ref[1]));relative=(best-best_ref)/max(abs(best_ref),1e-8)
   holder['quality']=dict(best_cost_optimized=best,best_cost_reference=best_ref,best_cost_relative_change=relative,mean_cost_optimized=float(np.mean(result[1])),mean_cost_reference=float(np.mean(ref[1])))
   assert best<=best_ref+max(1e-5,.01*abs(best_ref)),holder['quality']
   return result
  lm.solve=compare
  return opt
 def replay(param,args,env,contact,mode,**kw):
  lm=holder['lm'];qp=env.data_.qpos.copy();qv=env.data_.qvel.copy()
  for fraction in [.25,.75,.95]:
   i=max(1,int((len(rows)-1)*fraction));before=rows[i-1];row=rows[i];state=np.asarray(before['object_pose']);velocity=np.asarray(before['velocity_body']);rot=Rotation.from_quat(state[[4,5,6,3]]).as_matrix()
   env.data_.qpos[:]=qp;env.data_.qpos[:7]=state;env.data_.qvel[:]=qv;env.data_.qvel[:3]=rot@velocity[:3];env.data_.qvel[3:6]=velocity[3:];env.data_.time=before['simulation_time_s'];mujoco.mj_forward(env.model_,env.data_)
   param.force_observation_pose_=None;param.force_cost_stages_.stage=row['object_cost_stage']
   solved=original_step(param,args,env,contact,mode,**kw)
   results.append(dict(object=obj,step=i,stage=row['object_cost_stage'],quality=holder['quality'],jacobian=lm.jac_checks[-1],iteration_stats=lm.history[-1],**lm.validations[-1]));print(json.dumps({k:v for k,v in results[-1].items() if k!='iteration_stats'}),flush=True)
  return solved
 mod.configure_direct_force_optimizer=configure;mod._solve_step=replay
 try:mod._run_rollout(cfg,'surrogate',obj,seed=0,mass=.01,steps=1)
 finally:mod.configure_direct_force_optimizer=original_configure;mod._solve_step=original_step
(OUT/'quality_comparison.json').write_text(json.dumps(results,indent=2)+'\n')
