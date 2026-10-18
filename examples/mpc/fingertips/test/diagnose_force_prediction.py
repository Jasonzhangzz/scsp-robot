"""Single-step counterfactual diagnostics for the current four physical models."""
import argparse
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from examples.mpc.fingertips.test.test_scm_mujoco import build_parser
from planning.point_force_benchmark import (MODELS,OBJECTS,PointForcePlanner,scene,
    surface_points,physical_problem,integrate_numpy,write_json)
from planning.direct_contact_force import step_point_force
from planning.object_pose_cost import ObjectCostStages,target_up_axis
import mujoco
import numpy as np


def audit(obj,seed,model='scm_corrected',replay=None):
    args = build_parser().parse_args(['--obj',obj,'--model',model])
    m,d,target,_ = scene(args,obj,seed)
    if replay is not None:
        from scipy.spatial.transform import Rotation
        d.qpos[:] = replay['object_pose']
        r = Rotation.from_quat(d.qpos[[4,5,6,3]]).as_matrix()
        velocity = np.asarray(replay['velocity_body'])
        d.qvel[:] = np.r_[r@velocity[:3],velocity[3:]]
        mujoco.mj_forward(m,d)
    points,frames = surface_points(m,d,args.sample_num,seed)
    planner = PointForcePlanner(m,points,frames,target,args,model,seed)
    config = ObjectCostStages(args,target_up_axis(target[3:])).config()
    snapshot = d.qpos.copy(),d.qvel.copy(),float(d.time),0
    validation = planner.validate(snapshot,config)
    solved = planner.solve(snapshot,config)
    cases = []
    for name,force in [('continuous',solved['force_local']),('unforced',np.zeros(3))]:
        data = copy.copy(d)
        env = SimpleNamespace(model_=m,data_=data,param_=SimpleNamespace(frame_skip_=40),
                              _sync_viewer=lambda:None)
        info = step_point_force(env,solved['p_local'],force)
        measured = physical_problem(m,data)['velocity']
        cases.append(dict(execution=name,actual_pose=data.qpos.copy(),actual_velocity_body=measured,
            linear_velocity_prediction_error=float(np.linalg.norm(measured[:3]-solved['velocity'][:3])),
            angular_velocity_prediction_error=float(np.linalg.norm(measured[3:]-solved['velocity'][3:])),
            actual_impulse_body=info['actual_applied_impulse_body']))
    return dict(formal_trial=False,model=model,object=obj,seed=seed,validation=validation,
                initial_pose=snapshot[0],selected_point=solved['p_local'],
                force_body=solved['force_local'],prediction_pose=solved['predicted_pose'],
                prediction_velocity=solved['velocity'],alpha=solved['alpha'],cases=cases)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--objects',default=','.join(OBJECTS))
    parser.add_argument('--models',default=','.join(MODELS))
    args = parser.parse_args()
    results = []
    for obj in args.objects.split(','):
        for model in args.models.split(','):
            result = audit(obj,0,model=model)
            results.append(result)
            write_json(args.output,results)
            print(obj,model,result['cases'][0]['linear_velocity_prediction_error'],flush=True)


if __name__ == '__main__':
    main()
