"""Isolate contact-pair quality with physical position-servo spherical pads.

Uses the actual object collider/inertia, friction, NoSlip and 10 ms step.
Panda arm dynamics and MPC are absent. Object motion is caused solely by
pad contacts; no object wrench or pose correction is applied.
Run from any directory after generating the default BigRasp scene.
"""
import copy,json,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
root=ET.parse(ROOT/'envs/xmls/_generated_bigrasp_scene.xml').getroot()
for name in ('mpc_ns3_mu5','mpc_pair_final','mpc_matched_section_sync','mpc_stable_section'):
 report=json.loads((OUT/(name+'.json')).read_text())
 pose=np.asarray(report['experiment']['initial_pose']); rot=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
 live=report['telemetry'][0]['lambda_prediction']['live_grasp'];points=np.asarray(live['force_points_local']);normals=np.asarray(live['force_normals_local'])
 minimal=ET.Element('mujoco');ET.SubElement(minimal,'compiler',angle='radian')
 minimal.append(copy.deepcopy(root.find('option')))
 asset=ET.SubElement(minimal,'asset')
 for mesh in root.findall('asset/mesh'):
  if mesh.get('name','').startswith('object'):asset.append(copy.deepcopy(mesh))
 world=ET.SubElement(minimal,'worldbody');ET.SubElement(world,'geom',name='support',type='plane',pos='0 0 .32',size='0 0 .01',contype='8',conaffinity='8',friction='5 .08 .01',condim='6')
 obj=copy.deepcopy(root.find('.//body[@name="obj"]'));world.append(obj)
 for geom in obj.findall('geom'):geom.attrib.pop('material',None)
 for i in range(2):
  body=ET.SubElement(world,'body',name=f'pad{i}',gravcomp='1')
  for axis in range(3):
   ET.SubElement(body,'joint',name=f'j{i}_{axis}',type='slide',axis=' '.join(str(int(a==axis)) for a in range(3)),damping='1')
  ET.SubElement(body,'geom',name=f'pad{i}',type='sphere',size='.01',contype='1',conaffinity='2',friction='5 .08 .01',condim='3',mass='.05',solimp='0.9 0.95 0.01',solref='0.02 1.0')
 act=ET.SubElement(minimal,'actuator')
 for i in range(2):
  for axis in range(3):ET.SubElement(act,'position',joint=f'j{i}_{axis}',kp='1000',kv='50')
 ET.ElementTree(minimal).write(OUT/'lambda_pair_physics.xml')
 model=mujoco.MjModel.from_xml_string(ET.tostring(minimal,encoding='unicode'));data=mujoco.MjData(model);data.qpos[:7]=pose
 initial=pose[:3]+(points-.012*normals)@rot.T
 data.qpos[7:]=initial.ravel();data.ctrl[:]=initial.ravel();mujoco.mj_forward(model,data)
 rows=[]
 for step in range(800):
  squeeze=min(step/100,1)
  offset=.012-.0023*squeeze
  lift=.06*np.clip((step-150)/300,0,1)
  data.ctrl[:]=(pose[:3]+(points-offset*normals)@rot.T+[0,0,lift]).ravel()
  mujoco.mj_step(model,data)
  if step%10==0:rows.append(data.qpos[:7].copy())
 result=dict(pair=points.tolist(),final_height_gain_mm=1000*(data.qpos[2]-pose[2]),final_xy_drift_mm=1000*np.linalg.norm(data.qpos[:2]-pose[:2]),peak_height_mm=1000*(max(r[2] for r in rows)-pose[2]),final_pose=data.qpos[:7].tolist(),method='Position-servo spherical pads with physical velocities, actual collider/material/NoSlip, 6 cm lift and 3.5 s hold; no applied object wrench')
 print(name,result,flush=True)
 (OUT/(name+'_servo_pad_lift.json')).write_text(json.dumps(result,indent=2))
