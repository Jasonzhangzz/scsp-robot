"""Evaluate command elevations at the recorded measured contact geometry."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/zz/scsp-robot')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from examples.mpc.franka.ik2.params import ExplicitMPCParams
from examples.mpc.franka.ik2.test_mpc_isaac import (
    _apply_dywa_physics_to_param,adapt_param_for_cartesian_solver,_configure_rollout_param)
from planning.mpc_explicit import MPCExplicit
from planning.object_pose_cost import object_cost_np,rotate_axis_np

# The exact dynamics and objective are used; native optimization is unnecessary.
MPCExplicit.init_MPC=lambda self: None


def main():
    results=[]
    for obj in ('elephant','stanford_bunny2'):
        for path in sorted((OUT/(obj+'_async_snapshots')).glob('*.npz')):
            data=np.load(path)
            args=argparse.Namespace(**json.loads(str(data['args_json'])))
            p=ExplicitMPCParams(args,rand_seed=0,build_optimizer=False)
            _apply_dywa_physics_to_param(p)
            adapt_param_for_cartesian_solver(p,args,build_optimizer=False)
            _configure_rollout_param(p,args)
            p.object_cost_config_=data['config'].copy()
            p.target_p_,p.target_q_=data['target'][:3],data['target'][3:]
            mpc=MPCExplicit(p)
            state=data['state']; baseline=data['u_traj']
            if baseline.shape!=(5,3):continue
            cp=mpc._cost_params(target_p=p.target_p_,target_q=p.target_q_,
                phi_vec=data['phi'],jac_mat=data['jac'],verify_cost_param=float(data['verify']),
                virtual_point=data['virtual_point'],contact_point=data['contact_point'],
                contact_reference=state[:7])
            cp_no_position=cp.copy()
            cp_no_position[-12]=0.
            heading=data['action'][:2].copy()
            if np.linalg.norm(heading)<1e-8:continue
            heading/=np.linalg.norm(heading)
            norm=np.linalg.norm(data['action'])
            commands=[('optimized',baseline)]
            for angle in (-45.,-30.,-15.,0.,15.,30.,45.,60.,75.):
                r=np.deg2rad(angle)
                u=norm*np.r_[np.cos(r)*heading,np.sin(r)]
                commands.append((str(angle),np.tile(u,(5,1))))
            evaluations=[]
            for name,controls in commands:
                x=state.copy();cost=0.;cost_no_position=0.;floor_min=float(x[9])
                for u in controls:
                    # Production evaluates the path cost at its selected stage state.
                    stage=x.copy()
                    x=np.asarray(mpc.stage_prediction_fn(x,u,data['phi'],data['jac'],
                        p.model_params,state)).reshape(-1)
                    evaluated=x if mpc.stage_cost_on_next else stage
                    cost+=float(mpc._path_cost(evaluated,u,cp))
                    cost_no_position+=float(mpc._path_cost(evaluated,u,cp_no_position))
                    floor_min=min(floor_min,float(x[9]))
                cost+=float(mpc._terminal_cost(x,cp))
                cost_no_position+=float(mpc._terminal_cost(x,cp_no_position))
                axis=p.object_up_axis_
                tilt=lambda q:float(np.rad2deg(np.arccos(np.clip(rotate_axis_np(q,axis)[2],-1,1))))
                evaluations.append(dict(command=name,cost=cost,cost_no_position=cost_no_position,
                    first_action=controls[0].tolist(),
                    first_within_slew_box=bool(np.all(controls[0]>=data['first_lower']-1e-9)
                        and np.all(controls[0]<=data['first_upper']+1e-9)),
                    horizon_min_tip_height_m=floor_min,
                    predicted_tilt_reduction_deg=tilt(state[3:7])-tilt(x[3:7]),
                    predicted_xy_displacement_mm=float(np.linalg.norm(x[:2]-state[:2])*1000),
                    predicted_object_height_change_mm=float((x[2]-state[2])*1000)))
            feasible=[e for e in evaluations if e['first_within_slew_box']]
            result=dict(object=obj,snapshot=path.name, evaluations=evaluations,
                        best=min(feasible,key=lambda e:e['cost'])['command'],
                        best_no_position=min(feasible,key=lambda e:e['cost_no_position'])['command'])
            results.append(result)
        print('Evaluated objective alternatives: '+obj,flush=True)
    (OUT/'objective_replay.json').write_text(json.dumps(results,indent=2)+'\n')
    for obj in ('elephant','stanford_bunny2'):
        cases=[r for r in results if r['object']==obj]
        changed=[r for r in cases if r['best']!=r['best_no_position']]
        print(obj,'snapshots',len(cases),'position_weight_changes_winner',len(changed),
              [(r['snapshot'],r['best'],r['best_no_position']) for r in changed],flush=True)


if __name__=='__main__':
    main()
