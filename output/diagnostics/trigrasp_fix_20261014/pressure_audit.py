import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
 os.environ.setdefault(name,'1')
import numpy as np,mujoco,json,sys
sys.path.insert(0,'/home/zz/scsp-robot')
from planning.bimanual_lambda_prediction import BimanualLambdaPrediction
xml='''<mujoco><compiler angle="radian"/><option timestep=".01" gravity="0 0 -9.81" cone="elliptic" impratio="10" noslip_iterations="3" integrator="implicitfast"/>
<default><geom friction=".7 .01 .001" condim="3" solimp=".9 .95 .01" solref=".02 1"/></default><worldbody>
<body name="obj" pos="0 0 .3"><freejoint/><geom name="object" type="box" size=".03 .03 .03" mass=".01"/></body>
<body name="left" pos="-.04 0 .3" gravcomp="1"><joint name="lx" type="slide" axis="1 0 0"/><joint name="ly" type="slide" axis="0 1 0"/><joint name="lz" type="slide" axis="0 0 1"/><geom name="lp" type="sphere" size=".01" mass="2.5"/></body>
<body name="right" pos=".04 0 .3" gravcomp="1"><joint name="rx" type="slide" axis="1 0 0"/><joint name="ry" type="slide" axis="0 1 0"/><joint name="rz" type="slide" axis="0 0 1"/><geom name="rp" type="sphere" size=".01" mass="2.5"/></body>
</worldbody><actuator><motor joint="lx"/><motor joint="ly"/><motor joint="lz"/><motor joint="rx"/><motor joint="ry"/><motor joint="rz"/></actuator></mujoco>'''

points=np.array([[-.03,0,0],[.03,0,0]]);normals=np.array([[1.,0,0],[-1.,0,0]])
results=[]
for cone,dim,depth in [(c,k,d) for c,k in [('elliptic',3),('pyramidal',6)] for d in [0.,.0003,.003]]:
 m=mujoco.MjModel.from_xml_string(xml.replace('cone="elliptic"','cone="'+cone+'"').replace('condim="3"','condim="'+str(dim)+'"'))
 d=mujoco.MjData(m);d.qpos[7],d.qpos[10]=depth,-depth;d.ctrl[:]=1.8*normals.ravel();mujoco.mj_forward(m,d)
 mass=np.zeros((m.nv,m.nv));mujoco.mj_fullM(m,d,mass)
 s=BimanualLambdaPrediction(horizon=8)
 pose=d.qpos[:7].copy();s.set_observation(pose,pose+[0,0,.05,0,0,0,0],d.qvel[:6].copy(),mass[:6,:6],-d.qfrc_bias[:6],friction=.7,support_friction=.7,tips_world=np.array([d.xpos[m.body('left').id],d.xpos[m.body('right').id]]))
 prediction=s.evaluate(points,normals);mujoco.mj_step(m,d)
 fn=np.zeros(2)
 for ci,c in enumerate(d.contact):
  f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
  for side,name in enumerate(['lp','rp']):
   if m.geom(name).id in [c.geom1,c.geom2]:fn[side]+=f[0]
 result={'cone':cone,'condim':dim,'depth':depth,'mujoco_fn':fn.tolist(),'scm_fn':np.sum(prediction['force_vectors_local']*normals,axis=1).tolist(),'external':s.external.tolist(),'mu':.7,'command_normal':1.8}
 results.append(result)
print(json.dumps(results,indent=2))
from pathlib import Path
Path(__file__).with_suffix('.json').write_text(json.dumps(results,indent=2))
