"""Run/review raw GraspSonic DexYCB retarget cases with trajectory plots."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from examples.retarget.dexycb_data import sync_sequence,prepare_sequence


def summarize(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True);summary={}
    for folder in map(Path,results):
        geometry=np.load(folder/'wuji_q.npy')
        with np.load(folder/'semantic_contacts.npz') as contacts:
            times=contacts['times'];reference=contacts['object_pose'];source_ids=contacts['frame_ids']
        config=json.loads((folder/'config.json').read_text());modes={}
        fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
        for axis in range(3):
            axes[0,axis].plot(times,geometry[:,axis],'k--',label='geometric IK reference')
            axes[1,axis].plot(times,reference[:,axis],'k--',label='demonstrated object')
            axes[0,axis].set_title(f'Wrist {"xyz"[axis]} (m)')
            axes[1,axis].set_title(f'Object {"xyz"[axis]} (m)')
            axes[1,axis].set_xlabel('Time (s)')
        for mode in ('ik-only','full'):
            if not (folder/mode/'metrics.json').exists():continue
            metrics=json.loads((folder/mode/'metrics.json').read_text())
            with np.load(folder/mode/'rollout.npz') as rollout:
                q=rollout['hand_q'];poses=rollout['object_pose'];frames=rollout['reference_frame']
            t=(np.arange(len(q))+1)*config['command_dt']
            for axis in range(3):
                axes[0,axis].plot(t,q[:,axis],label=mode)
                axes[1,axis].plot(t,poses[:,axis],label=mode)
            metrics['commanded_source_frames']=source_ids[np.unique(frames)].tolist()
            metrics['geometric_ik_frames']=len(geometry)
            modes[mode]=metrics
        axes[0,0].legend();axes[1,0].legend()
        for ax in axes.ravel():ax.grid(alpha=.2)
        fig.suptitle(folder.name+' — measured MuJoCo trajectories')
        fig.tight_layout();fig.savefig(output/f'{folder.name}.png',dpi=160);plt.close(fig)
        summary[folder.name]=dict(result_directory=str(folder.resolve()),modes=modes,
            physics=json.loads((folder/'physics_parameters.json').read_text()))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/wuji_dexycb_benchmark')
    p.add_argument('--results',type=Path,nargs='+',help='Only summarize existing result folders.')
    p.add_argument('--dataset',type=Path,default=ROOT/'thirdparty/retarget_assets/dexycb_raw')
    p.add_argument('--subject',default='20200709-subject-01')
    p.add_argument('--sequences',nargs='+',default=['20200709_142802','20200709_142553'])
    p.add_argument('--camera');p.add_argument('--sync',action='store_true')
    p.add_argument('--scene-template',type=Path,default=ROOT/'thirdparty/retarget_assets/DexForge/data/dexycb/dexycb_004_sugar_box_20200709_142802/scene/leaphand/sugar_box_mesh.xml')
    p.add_argument('--config',type=Path,default=ROOT/'examples/retarget/default.yaml')
    p.add_argument('--max-steps',type=int,default=600);p.add_argument('--video',action='store_true')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.results:return summarize(args.results,args.output)
    results=[]
    for identifier in args.sequences:
        if args.sync:sequence,camera=sync_sequence(args.dataset,args.subject,identifier,args.camera)
        else:sequence,camera=args.dataset/args.subject/identifier,args.camera
        prepared=prepare_sequence(args.dataset,sequence,camera,args.scene_template,args.output/'prepared'/identifier)
        destination=args.output/identifier
        command=[sys.executable,str(ROOT/'examples/retarget/retarget.py'),'--config',str(args.config),
            '--sequence',str(prepared),'--output',str(destination),'--headless','--compare',
            '--max-steps',str(args.max_steps)]
        if args.video:command.append('--video')
        environment=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        subprocess.run(command,cwd=ROOT,env=environment,check=True);results.append(destination)
    summarize(results,args.output)


if __name__=='__main__':main()
