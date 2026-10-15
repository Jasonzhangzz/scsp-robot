import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['MUJOCO_HEADLESS']='1';os.environ['ACADOS_SOURCE_DIR']='/home/zz/zz_ws/acados'
import sys,json,time,traceback
from pathlib import Path
ROOT=Path('/home/zz/scsp-robot');OUT=ROOT/'output/diagnostics/gpu_batch_lm_success_rollouts_20261014';sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OUT))
from gpu_lm import GPUExactBatchLM
from examples.mpc.fingertips.test import test_scm_mujoco as mod
from planning.rollout_timing import trial_timing,aggregate_trials
original=mod.configure_direct_force_optimizer
summaries=[]
for obj in ['foam_brick','elephant','stanford_bunny2']:
 source=ROOT/'output/benchmarks/scm_solvers_7x5_20261013_211740/acados'/f'trial_{obj}_000.json';cached=json.loads(source.read_text());assert cached['metadata']['trial_summary']['success']
 args=mod.build_parser().parse_args([])
 for k,v in cached['metadata']['args'].items():setattr(args,k,v)
 args.viewer=False;args.solver='acados'
 holder={}
 def configure(param,env,cfg):
  opt=original(param,env,cfg);st=time.perf_counter();lm=GPUExactBatchLM(opt,iterations=20);lm.install();holder['lm']=lm
  for count in range(1,len(opt.sample_point)+1):lm.buffers(count)
  lm.validate=True
  # Warm the real initial state and verify every candidate against CasADi.
  import copy
  saved={}
  for key,value in opt.__dict__.items():
   if key.startswith(('last_','_dwell','_pending','_blocked','contact_switch')):
    try:saved[key]=copy.deepcopy(value)
    except Exception:pass
  mod._solve_step(param,cfg,env,mod.Contact(param),'surrogate')
  for key,value in saved.items():setattr(opt,key,value)
  from planning.object_pose_cost import ObjectCostStages,target_up_axis
  param.force_cost_stages_=ObjectCostStages(cfg,target_up_axis(param.target_q_));param.force_observation_pose_=None
  lm.validate=False
  (OUT/f'validation_{obj}.json').write_text(json.dumps(lm.validations,indent=2)+'\n')
  print('LM_SETUP',obj,time.perf_counter()-st,flush=True);return opt
 mod.configure_direct_force_optimizer=configure
 rows=[];error=None
 try:rows=mod._run_rollout(args,'surrogate',obj,seed=0,mass=.01,steps=5000)
 except Exception as e:rows=getattr(e,'rows',rows);error=str(e);traceback.print_exc()
 finally:mod.configure_direct_force_optimizer=original
 timing=trial_timing(rows) if rows else {};last=rows[-1] if rows else {}
 summary=dict(object=obj,seed=0,solver='batch-lm-gpu-exact-scm',success=bool(last.get('success',False)),settled=bool(last.get('settled',False)),steps=len(rows),error=error,iterations=20,source_success_trial=str(source.relative_to(ROOT)),reference= cached['metadata']['trial_summary'],residual_cost_checks=holder['lm'].calls if 'lm' in holder else 0,**timing)
 mod._write_rows(str(OUT/f'trial_{obj}_000.json'),rows,dict(trial_summary=summary,args=vars(args),scope='small sample selected from successful acados tasks; GPU float64 batched LM with exact SCM/objective'))
 summaries.append(summary);(OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n');print('LM_RESULT',json.dumps(summary),flush=True)
