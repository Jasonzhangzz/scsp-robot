import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
archives={label:{stage:np.load(root/(label+'_'+stage+'.npz'),allow_pickle=False) for stage in ['native','raw','final']} for label in ['current','historical']}
def metric(a,b):
 a=np.asarray(a);b=np.asarray(b)
 result={'current_shape':list(a.shape),'historical_shape':list(b.shape),'exact_equal':bool(np.array_equal(a,b))}
 if a.shape!=b.shape:return result
 if a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
  result['current_finite']=bool(np.isfinite(a).all());result['historical_finite']=bool(np.isfinite(b).all())
  d=np.abs(a.astype(np.float64)-b.astype(np.float64))
  result.update(max_abs_diff=float(d.max()) if d.size else 0.,rmse=float(np.sqrt(np.mean(d*d))) if d.size else 0.,different_elements=int(np.count_nonzero(a!=b)),total_elements=int(a.size))
  if d.size and d.max()>0:result['max_diff_index']=[int(i) for i in np.unravel_index(np.argmax(d),d.shape)]
 return result
report={'manifest':manifest,'comparison':'same original episode and matched model/initial target parameters','runs':{},'native':{},'raw':{},'final':{},'uploaded_vs_outputs':{}}
for label,stages in archives.items():
 raw=stages['raw'];final=stages['final']
 candidates=list((root/(label+'_published')).rglob(manifest['source_episode'].split('/')[-1]+'.json'))
 output=next(p for p in candidates if not p.is_symlink() and '/.' not in str(p.relative_to(root)))
 metadata=json.loads((output.parent/'replay_metadata.json').read_text())
 report['runs'][label]={'raw_frames':len(raw['arm_q']),'final_frames':len(final['arm_q']),'ramp_frames':int(metadata['move_from_start_ramp_frame_count']),'arm_ik_success':int(raw['arm_ik_success'].sum()),'arm_ik_total':int(raw['arm_ik_success'].size),'source_frame_indices':raw['frame_indices'].tolist(),'published_json':str(output),'final_npz':str(root/(label+'_final.npz'))}
for side in ['left','right']:
 for suffix in ['hand_q','skeleton_position','skeleton_orientation_xyzw','action_timestamp_ns','source_timestamp_ns']:
  k=side+'_'+suffix
  report['native'][k]=metric(archives['current']['native'][k],archives['historical']['native'][k])
a=archives['current']['raw'];b=archives['historical']['raw']
keys=['frame_indices','timestamps','arm_q_names','wuji_hand_q_names','arm_q','wuji_hand_q','left_hand_q','right_hand_q','arm_ik_success','left_wrist_position_base','right_wrist_position_base','left_wrist_quaternion_base_xyzw','right_wrist_quaternion_base_xyzw','world_to_base_transform','world_to_base_rotation','world_to_base_translation','left_skeleton_position_world','right_skeleton_position_world']
for k in keys:
 if k in a.files and k in b.files:report['raw'][k]=metric(a[k],b[k])
for side in ['left','right']:
 names=np.asarray(a['arm_q_names']).astype(str)
 cols=[i for i,n in enumerate(names) if n.startswith(side)]
 report['raw'][side+'_arm_q']=metric(a['arm_q'][:,cols],b['arm_q'][:,cols])
 for suffix in ['arm_q','hand_q']:
  r=report['raw'][side+'_'+suffix]
  if 'max_abs_diff' in r:r['max_abs_diff_deg']=float(np.rad2deg(r['max_abs_diff']))
 poskey=side+'_wrist_position_base';qkey=side+'_wrist_quaternion_base_xyzw'
 if a[poskey].shape==b[poskey].shape:
  report['raw'][poskey]['max_position_distance_m']=float(np.linalg.norm(a[poskey]-b[poskey],axis=1).max())
 if a[qkey].shape==b[qkey].shape:
  r1=Rotation.from_quat(a[qkey]);r2=Rotation.from_quat(b[qkey])
  report['raw'][qkey]['max_rotation_distance_deg']=float(np.rad2deg((r1.inv()*r2).magnitude()).max())
for k in keys:
 ca=archives['current']['final'];hb=archives['historical']['final']
 if k in ca.files and k in hb.files:report['final'][k]=metric(ca[k],hb[k])
report['trajectory_exact_equal']=all(report['raw'][k]['exact_equal'] for k in ['frame_indices','timestamps','arm_q','wuji_hand_q','left_wrist_position_base','right_wrist_position_base','left_wrist_quaternion_base_xyzw','right_wrist_quaternion_base_xyzw'])
report['final_trajectory_exact_equal']=all(report['final'][k]['exact_equal'] for k in ['timestamps','arm_q','wuji_hand_q','left_wrist_position_base','right_wrist_position_base','left_wrist_quaternion_base_xyzw','right_wrist_quaternion_base_xyzw'])
uploaded=json.loads((root.parent/'retarget_1.json').read_text())
control_keys=['left_arm_joint_controller_commands','right_arm_joint_controller_commands','wuji_left_joint_position','wuji_right_joint_position','follow_left_position','follow_right_position','follow_left_ee_rotation_6D','follow_right_ee_rotation_6D']
for label,r in report['runs'].items():
 doc=json.loads(Path(r['published_json']).read_text())
 report['uploaded_vs_outputs'][label]={'uploaded_frames':len(uploaded['data']),'output_frames':len(doc['data']),'fields':{}}
 for k in control_keys:
  report['uploaded_vs_outputs'][label]['fields'][k]=metric(np.array([f[k] for f in doc['data']]),np.array([f[k] for f in uploaded['data']]))
(root/'comparison_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['trajectory_exact_equal','final_trajectory_exact_equal','runs','native','raw']},ensure_ascii=False,indent=2))
