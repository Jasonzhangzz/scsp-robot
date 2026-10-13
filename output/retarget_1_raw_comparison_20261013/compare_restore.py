import json,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
label=sys.argv[1]
base=json.loads((root/'comparison_report.json').read_text());name=Path(base['manifest']['source_episode']).name
paths={'historical':Path(base['runs']['historical']['published_json']),label:next(p for p in (root/(label+'_published')).rglob(name+'.json') if not p.is_symlink())}
data={};meta={}
for key,path in paths.items():
 document=json.loads(path.read_text());meta[key]=json.loads((path.parent/'replay_metadata.json').read_text());data[key]=document['data'][meta[key]['move_from_start_ramp_frame_count']:]
a=np.load(root/(label+'_raw.npz'));b=np.load(root/'historical_raw.npz')
assert np.array_equal(a['frame_indices'],b['frame_indices'])
result={'label':label,'compared_frames':len(data[label]),'source_frame_indices_equal':True,'current_ramp_frames':meta[label]['move_from_start_ramp_frame_count'],'historical_ramp_frames':meta['historical']['move_from_start_ramp_frame_count'],'published_json':str(paths[label]),'control_fields':{},'raw_arrays':{}}
for key in ['follow_right_position','follow_right_ee_rotation_6D','follow_left_position','follow_left_ee_rotation_6D','wuji_right_joint_position','wuji_left_joint_position','right_arm_joint_controller_commands','left_arm_joint_controller_commands']:
 va=np.array([f[key] for f in data[label]]);vb=np.array([f[key] for f in data['historical']])
 result['control_fields'][key]={'exact_equal':bool(np.array_equal(va,vb)),'max_abs_diff':float(np.abs(va-vb).max()),'different_elements':int(np.count_nonzero(va!=vb))}
for key in ['world_to_base_transform','arm_q','wuji_hand_q','left_wrist_position_base','right_wrist_position_base','left_wrist_quaternion_base_xyzw','right_wrist_quaternion_base_xyzw']:
 result['raw_arrays'][key]={'exact_equal':bool(np.array_equal(a[key],b[key])),'max_abs_diff':float(np.abs(a[key]-b[key]).max())}
result['right_follow_exact_equal']=all(result['control_fields'][key]['exact_equal'] for key in ['follow_right_position','follow_right_ee_rotation_6D'])
(root/(label+'_comparison.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
