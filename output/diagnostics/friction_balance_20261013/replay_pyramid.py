"""Re-solve measured CSO contacts with isolated friction/regularization changes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/zz/scsp-robot'); OUT=ROOT/'output/diagnostics/flip_flat_push_20261013'; RESULT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from examples.mpc.franka.ik2.params import ExplicitMPCParams,build_lambda_optimizer
import examples.mpc.franka.ik2.params as param_module
from planning.mlqp_point import LambdaContactControlOptimizer
CONE = 'pyramid'
def diagnostic_optimizer(**kwargs):
    kwargs['friction_cone'] = CONE
    return LambdaContactControlOptimizer(**kwargs)
param_module.LambdaContactControlOptimizer = diagnostic_optimizer
from utils import rotations
from examples.mpc.franka.ik2.contact_frames import gravity_wrench_object_frame


def main():
    records=[]
    variants=[('penalty_'+str(w),1.2,w,.7) for w in (.01,.03,.05)]
    for obj in ('elephant','stanford_bunny2'):
        paths=sorted((OUT/(obj+'_sync_snapshots')).glob('*.npz'))
        first=np.load(paths[0])
        args=argparse.Namespace(**json.loads(str(first['args_json'])))
        for name,arm_mu,penalty,env_mu in variants:
            args.friction_reg_coef=penalty
            p=ExplicitMPCParams(args,rand_seed=0,build_optimizer=False)
            p.mu_fingertip_object_=arm_mu
            p.mu_table_object_=env_mu
            p.mu_object_=env_mu
            opt=build_lambda_optimizer(p,args)
            for path in paths:
                data=np.load(path);row=json.loads(str(data['row_json']))
                contact=row['actual_contact'];state=data['state']
                r=Rotation.from_quat(state[[4,5,6,3]]).as_matrix()
                point=r.T@(np.asarray(contact['point_world'])-state[:3])
                inward=-r.T@np.asarray(contact['normal_world'])
                target=np.r_[r.T@(data['target'][:3]-state[:3]),
                    rotations.quaternion_multiply(rotations.quaternion_conjugate(state[3:7]),data['target'][3:])]
                current=np.array([0.,0.,0.,1.,0.,0.,0.])
                gravity=gravity_wrench_object_frame(p.gravity_[:3],p.lambda_obj_mass_,r,None)
                jac=data['env_jac'].copy()
                for start in range(0,len(jac),4):
                    block=jac[start:start+4];normal=block.mean(axis=0)
                    jac[start:start+4]=normal+(block-normal)*env_mu/.5
                opt.update_Jacobian(jac,phi=data['env_phi'])
                config=data['config'].copy();config[6:9]=r.T@config[6:9]
                opt.set_object_cost_config(config)
                idx=int(np.argmin(np.linalg.norm(opt.sample_point-point,axis=1)))
                _,_,predicted,error,info=opt.optimize_control_input(target,current,gravity,p_arm=point,
                    sample_idx=idx,contact_point_local=point,contact_normal_inward=inward)
                n=inward/np.linalg.norm(inward)
                ref=np.array([0.,0.,1.]) if abs(n[2])<.9 else np.array([0.,1.,0.])
                tangent=np.cross(n,ref);tangent/=np.linalg.norm(tangent)
                frame=np.column_stack([n,tangent,np.cross(n,tangent)])
                force=np.asarray(info['control_input']).reshape(3)
                world_force=r@frame@force
                local_pred=np.asarray(predicted).reshape(7)
                gain=opt._pose_cost(current,target)-opt._pose_cost(local_pred,target)
                before_up=np.dot(r@config[3:6],[0,0,1])
                after=r@Rotation.from_quat(local_pred[[4,5,6,3]]).as_matrix()@config[3:6]
                angle=lambda z:float(np.rad2deg(np.arccos(np.clip(z,-1,1))))
                recorded=(row.get('predictions') or {}).get('lambda_measured')
                records.append(dict(object=obj,variant=name,snapshot=path.name,friction_cone=CONE,
                    world_force_N=world_force.tolist(),force_contact_N=force.tolist(),
                    elevation_deg=float(np.rad2deg(np.arctan2(world_force[2],np.linalg.norm(world_force[:2])))),
                    tilt_gain_deg=angle(before_up)-angle(after[2]),
                    object_cost_gain=float(gain),tangent_penalty=float(penalty*np.dot(force[1:],force[1:])),
                    recorded_gain=None if recorded is None else recorded['object_cost_reduction'],
                    recorded_gain_difference=None if recorded is None else float(gain-recorded['object_cost_reduction']),
                    solver_failed=bool(info.get('solver_failed',False))))
            print(obj,name,'complete',flush=True)
    (RESULT/'lambda_replay_pyramid.json').write_text(json.dumps(records,indent=2)+'\n')
    for obj in ('elephant','stanford_bunny2'):
        for name,*_ in variants:
            rs=[r for r in records if r['object']==obj and r['variant']==name]
            print(obj,name,'n',len(rs),'median elevation',np.median([r['elevation_deg'] for r in rs]),
                'mean gain deg',np.mean([r['tilt_gain_deg'] for r in rs]),
                'failures',sum(r['solver_failed'] for r in rs),flush=True)


if __name__=='__main__':main()
