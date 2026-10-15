import os,sys,time,json,statistics
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.path.insert(0,'/home/zz/scsp-robot')
import numpy as np
from planning.lambda_batched_lm import BatchedLambdaLM
records=[]
for count in [54,70]:
 theta=np.linspace(0,2*np.pi,count,endpoint=False)
 points=np.stack([.04*np.cos(theta),.04*np.sin(theta),np.zeros(count)],axis=1)[:,None,:]
 normals=-points.copy();normals/=np.linalg.norm(normals,axis=2,keepdims=True)
 for iterations in [5,10,20]:
  solver=BatchedLambdaLM(device='cuda:0',iterations=iterations,num_contacts=1)
  kw=dict(current_x=np.array([0.,0.,0.,1.,0.,0.,0.]),x_d=np.array([.01,0.,.05,1.,0.,0.,0.]),prediction_dt=.02,object_inertia=np.diag([.2,.2,.2,.004,.004,.004]))
  for _ in range(3):solver.solve(points,normals,**kw)
  times=[]
  for _ in range(30):
   st=time.perf_counter();result=solver.solve(points,normals,**kw);times.append(1000*(time.perf_counter()-st))
  row=dict(candidates=count,iterations=iterations,mean_ms=statistics.mean(times),median_ms=statistics.median(times),p95_ms=float(np.percentile(times,95)),measurement='host inputs, GPU batched solve, synchronized selected-result readback; warmed; synthetic geometry, different dynamics from paper benchmark')
  records.append(row);print(json.dumps(row),flush=True)
open('output/diagnostics/lambda_timing_audit_20261014/lm_microbenchmark.json','w').write(json.dumps(records,indent=2)+'\n')
