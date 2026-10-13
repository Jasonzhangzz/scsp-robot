import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parent
baseline=json.loads((root/'comparison_report.json').read_text())
name=Path(baseline['manifest']['source_episode']).name
history_path=Path(baseline['runs']['historical']['published_json'])
history_doc=json.loads(history_path.read_text());history_meta=json.loads((history_path.parent/'replay_metadata.json').read_text())
history_frames=history_doc['data'][history_meta['move_from_start_ramp_frame_count']:]
history_npz=np.load(root/'historical_raw.npz')
def r6(frames):
 v=np.array([f['follow_right_ee_rotation_6D'] for f in frames])
 return Rotation.from_matrix(np.stack([v[:,:3],v[:,3:],np.cross(v[:,:3],v[:,3:])],axis=2))
metrics={}
for label in ['current','current_capture_wrist','current_legacy_bones']:
 path=next(p for p in (root/(label+'_published')).rglob(name+'.json') if not p.is_symlink())
 doc=json.loads(path.read_text());meta=json.loads((path.parent/'replay_metadata.json').read_text())
 frames=doc['data'][meta['move_from_start_ramp_frame_count']:]
 a=np.load(root/(label+'_raw.npz'));b=history_npz
 assert np.array_equal(a['frame_indices'],b['frame_indices'])
 m={'action_frames':len(frames),'ramp_frames':meta['move_from_start_ramp_frame_count'],'published_json':str(path),'raw_arrays':{},'right_follow':{}}
 for k in ['arm_q','wuji_hand_q','right_wrist_position_base','right_wrist_quaternion_base_xyzw','world_to_base_transform','right_source_skeleton_positions_world','right_source_skeleton_quaternions_xyzw']:
  m['raw_arrays'][k]={'exact_equal':bool(np.array_equal(a[k],b[k])),'max_abs_diff':float(np.abs(a[k]-b[k]).max())}
 for key in ['follow_right_position','follow_right_ee_rotation_6D']:
  v=np.array([f[key] for f in frames]);h=np.array([f[key] for f in history_frames])
  m['right_follow'][key]={'exact_equal':bool(np.array_equal(v,h)),'max_abs_diff':float(np.abs(v-h).max())}
 cp=np.array([f['follow_right_position'] for f in frames]);hp=np.array([f['follow_right_position'] for f in history_frames])
 distance=np.linalg.norm(cp-hp,axis=1)*1000
 angle=np.rad2deg((r6(history_frames).inv()*r6(frames)).magnitude())
 m['right_follow']['position_distance_mm']={'mean':float(distance.mean()),'max':float(distance.max())}
 m['right_follow']['rotation_difference_deg']={'mean':float(angle.mean()),'max':float(angle.max())}
 oldw=Rotation.from_quat(b['right_source_skeleton_quaternions_xyzw'][:,0]);neww=Rotation.from_quat(a['right_source_skeleton_quaternions_xyzw'][:,0])
 angle=np.rad2deg((oldw.inv()*neww).magnitude())
 m['source_right_wrist_world_rotation_difference_deg']={'mean':float(angle.mean()),'max':float(angle.max())}
 wp=np.linalg.norm(a['right_source_skeleton_positions_world'][:,0]-b['right_source_skeleton_positions_world'][:,0],axis=1)*1000
 m['source_right_wrist_world_position_difference_mm']={'mean':float(wp.mean()),'max':float(wp.max())}
 metrics[label]=m
(root/'ablation_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2))
print(json.dumps(metrics,ensure_ascii=False,indent=2))
