import os,sys,subprocess,json,time
from pathlib import Path
root=Path('/home/lab423/scsp/scsp-robot');out=root/'output/diagnostics/contact_experiment_20261013';out.mkdir(parents=True,exist_ok=True)
base=Path('/tmp/scsp-contact-baseline-20261013/scsp-robot')
python='/home/lab423/miniconda3/envs/acc/bin/python'
env=dict(os.environ,PATH='/home/lab423/miniconda3/envs/acc/bin:'+os.environ['PATH'],ACADOS_SOURCE_DIR='/home/lab423/push_ws/acados',LD_LIBRARY_PATH='/home/lab423/push_ws/acados/lib:/home/lab423/miniconda3/envs/acc/lib:'+os.environ.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
cases=[('baseline',base,[]),('optimized',root,[]),('physical_cpo',root,['--contact-dynamics-variant','physical']),('shared_legacy',root,['--shared-contact-dynamics']),('shared_physical',root,['--shared-contact-dynamics','--contact-dynamics-variant','physical'])]
results=[]
for name,cwd,extra in cases:
 for obj in ['rubber_duck','stanford_bunny2']:
  metric=out/(name+'_'+obj+'.jsonl');log=out/(name+'_'+obj+'.log')
  cmd=[python,'examples/mpc/franka/ik2/test_mpc_isaac.py','--obj',obj,'--headless','--sync-planner','--trial_num','1','--max_rollout_length','400','--ranking-metrics-jsonl',str(metric),*extra]
  started=time.monotonic()
  with log.open('w') as f:
   try:
    proc=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=150)
    result=dict(case=name,object=obj,returncode=proc.returncode,wall_s=time.monotonic()-started)
    if proc.returncode==0:result['metrics']=json.loads(metric.read_text().splitlines()[-1])
   except subprocess.TimeoutExpired:result=dict(case=name,object=obj,error='150-second timeout')
  results.append(result);(out/'experiments.json').write_text(json.dumps(results,indent=2)+'\n')
  s=result.get('metrics',{});print(name,obj,'tilt',s.get('object_tilt_deg'),'cycle_ms',s.get('planning_wall_mean_ms'),'fallbacks',s.get('mpc_fallbacks'),flush=True)
