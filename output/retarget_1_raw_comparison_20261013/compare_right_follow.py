import json,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
root=Path(__file__).resolve().parent
report=json.loads((root/'comparison_report.json').read_text())
doc={};meta={};data={}
for label in ['current','historical']:
 p=Path(report['runs'][label]['published_json'])
 doc[label]=json.loads(p.read_text())
 meta[label]=json.loads((p.parent/'replay_metadata.json').read_text())
 ramp=meta[label]['move_from_start_ramp_frame_count']
 data[label]=doc[label]['data'][ramp:]
raw={s:np.load(root/(s+'_raw.npz')) for s in ['current','historical']}
source=Path(report['manifest']['source_episode'])
source_document=json.loads((source/(source.name+'.json')).read_text())
source_stamps=np.array([int(f['timestamp']) for f in source_document['data']],dtype=np.int64)
source_times=(source_stamps-source_stamps[0])/1e9
assert np.array_equal(raw['current']['frame_indices'],raw['historical']['frame_indices'])
assert len(data['current'])==len(data['historical'])==195
pos={s:np.array([f['follow_right_position'] for f in data[s]]) for s in data}
r6={s:np.array([f['follow_right_ee_rotation_6D'] for f in data[s]]) for s in data}
rot={s:Rotation.from_matrix(np.stack([v[:,:3],v[:,3:],np.cross(v[:,:3],v[:,3:])],axis=2)) for s,v in r6.items()}
dp=pos['current']-pos['historical']; dist=np.linalg.norm(dp,axis=1)*1000
relative=rot['historical'].inv()*rot['current']
angle=np.rad2deg(relative.magnitude())
variation=np.rad2deg((relative[0].inv()*relative).magnitude())
def stats(v):
 return {'min':float(v.min()),'mean':float(v.mean()),'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'max':float(v.max()),'rms':float(np.sqrt(np.mean(v*v)))}
def sample(i):
 return {'source_frame_index':int(raw['current']['frame_indices'][i]),'source_elapsed_time_s':float(source_times[int(raw['current']['frame_indices'][i])]),'current_json_index':int(meta['current']['move_from_start_ramp_frame_count']+i),'historical_json_index':int(meta['historical']['move_from_start_ramp_frame_count']+i),'current_position_m':pos['current'][i].tolist(),'historical_position_m':pos['historical'][i].tolist(),'delta_current_minus_historical_mm':(dp[i]*1000).tolist(),'position_distance_mm':float(dist[i]),'current_rotation_6d':r6['current'][i].tolist(),'historical_rotation_6d':r6['historical'][i].tolist(),'rotation_difference_deg':float(angle[i])}
r={'alignment':'Exclude current 55-frame and historical 54-frame start ramps; compare the same 195 original source frames in identical frame_indices order. Controller frame is right_arm_neutral, commanded link is right_arm_link6.','current_final_frames':len(doc['current']['data']),'historical_final_frames':len(doc['historical']['data']),'compared_action_frames':len(dist),'follow_right_position':{'exact_equal':bool(np.array_equal(pos['current'],pos['historical'])),'different_frames':int(np.count_nonzero(np.any(pos['current']!=pos['historical'],axis=1))),'distance_mm':stats(dist),'max_abs_xyz_diff_mm':(np.max(np.abs(dp),axis=0)*1000).tolist(),'mean_xyz_delta_mm':(dp.mean(axis=0)*1000).tolist(),'worst_frame':sample(int(np.argmax(dist)))},'follow_right_ee_rotation_6D':{'exact_equal':bool(np.array_equal(r6['current'],r6['historical'])),'different_frames':int(np.count_nonzero(np.any(r6['current']!=r6['historical'],axis=1))),'max_abs_component_diff':float(np.abs(r6['current']-r6['historical']).max()),'rotation_difference_deg':stats(angle),'relative_rotation_variation_from_first_frame_deg':stats(variation),'worst_frame':sample(int(np.argmax(angle)))},'right_finger_joint_q':report['raw']['right_hand_q'],'first_frame':sample(0)}
report['right_follow']=r
(root/'comparison_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(root/'right_follow_comparison.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
with (root/'right_follow_per_frame.csv').open('w',newline='') as f:
 writer=csv.writer(f)
 writer.writerow(['source_frame_index','source_elapsed_time_s','current_json_index','historical_json_index','current_x_m','current_y_m','current_z_m','historical_x_m','historical_y_m','historical_z_m','delta_x_mm','delta_y_mm','delta_z_mm','position_distance_mm','rotation_difference_deg'])
 for i in range(len(dist)):
  v=sample(i)
  writer.writerow([v['source_frame_index'],v['source_elapsed_time_s'],v['current_json_index'],v['historical_json_index'],*pos['current'][i],*pos['historical'][i],*(dp[i]*1000),dist[i],angle[i]])
print(json.dumps(r,ensure_ascii=False,indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(11,8),layout='constrained')
times=source_times[raw['current']['frame_indices']]
ax=fig.add_subplot(2,2,1,projection='3d')
for label,color in [('current','#0072B2'),('historical','#D55E00')]:
 v=pos[label]*1000
 ax.plot(v[:,0],v[:,1],v[:,2],label=label,color=color,lw=1.7)
 ax.scatter(v[0,0],v[0,1],v[0,2],color=color,s=25)
ax.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title='Right follow position; dots = first action frame')
ax.legend(fontsize=9)
ax=fig.add_subplot(2,2,2)
ax.plot(times,dist,color='#0072B2');ax.set(xlabel='Original source time (s)',ylabel='Position distance (mm)',title='Current vs main~2: position distance',ylim=(0,50));ax.grid(alpha=.25)
ax=fig.add_subplot(2,2,3)
for i,name in enumerate(['X','Y','Z']):ax.plot(times,dp[:,i]*1000,label=name)
ax.axhline(0,color='black',lw=.7);ax.set(xlabel='Original source time (s)',ylabel='Current - historical (mm)',title='Right follow position component differences');ax.grid(alpha=.25);ax.legend()
ax=fig.add_subplot(2,2,4)
ax.plot(times,angle,color='#D55E00');ax.set(xlabel='Original source time (s)',ylabel='Rotation difference (degrees)',title='Right follow rotation: relative SO(3) angle',ylim=(0,26));ax.grid(alpha=.25)
fig.suptitle('ex001_wuji | matched parameters | 195 action frames (start ramps excluded)',fontsize=13)
fig.savefig(root/'right_follow_comparison.png',dpi=160)
plt.close(fig)
