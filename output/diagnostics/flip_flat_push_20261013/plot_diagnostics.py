import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
angle=lambda x:float(np.rad2deg(np.arctan2(x[2],np.linalg.norm(x[:2]))))
fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
for k,obj in enumerate(('elephant','stanford_bunny2')):
    rows=[json.loads(l) for l in (OUT/(obj+'_sync_forces_probe.jsonl')).read_text().splitlines()]
    times=np.array([r['sim_time'] for r in rows]);times-=times[0]
    tilt=[float(np.rad2deg(np.arccos(np.clip(Rotation.from_quat(np.asarray(r['state'])[[4,5,6,3]])
           .apply(r['config'][3:6])[2],-1,1)))) for r in rows]
    axes[k,0].plot(times,tilt,color='#214b72')
    axes[k,0].axhline(10,color='gray',ls='--')
    axes[k,0].set(title=obj+' tilt',ylabel='Degrees',xlabel='Simulation time (s)',ylim=(0,140))
    commands=np.full(len(times),np.nan);forces=np.full(len(times),np.nan)
    for index,r in enumerate(rows):
        samples=[f for f in r.get('force_samples') or [] if f['contact']]
        if samples and r['executed_action'] is not None:
            mean=np.mean([f['force'] for f in samples],axis=0)
            if np.linalg.norm(mean)>1e-4:
                commands[index]=angle(r['executed_action']);forces[index]=angle(mean)
    axes[k,1].plot(times,commands,label='Executed increment',color='#5a94c9',alpha=.8,lw=1)
    axes[k,1].plot(times,forces,label='Mean contact OSC force',color='#d3713d',lw=1)
    axes[k,1].axhspan(-15,15,color='gray',alpha=.12)
    axes[k,1].set(title='Elevation: command vs force',ylabel='Degrees',xlabel='Simulation time (s)',ylim=(-90,90),xlim=(0,6))
    axes[k,1].legend(fontsize=8)
    positions=np.array([r['state'][:3] for r in rows]);delta=positions-positions[0]
    axes[k,2].plot(times,np.linalg.norm(delta[:,:2],axis=1)*1000,label='XY displacement',color='#214b72')
    axes[k,2].plot(times,delta[:,2]*1000,label='COM Z change',color='#d3713d')
    axes[k,2].set(title='Object displacement',ylabel='mm',xlabel='Simulation time (s)')
    axes[k,2].legend(fontsize=8)
    for ax in axes[k]:ax.grid(alpha=.2)
fig.suptitle('Seed 0, synchronous diagnostic: 300 planner cycles per object',fontsize=13)
fig.savefig(OUT/'command_force_motion.png',dpi=160)
fig.savefig(OUT/'command_force_motion.pdf')
