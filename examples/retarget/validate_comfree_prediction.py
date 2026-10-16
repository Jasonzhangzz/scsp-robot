"""Validate production ComFree prediction against the upstream FA executor."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/comfree_default_20261016')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    forge=ROOT/'thirdparty/retarget_assets/DexForge';matrix=ROOT/'outputs/retarget_matrix_20261016'
    from models.wuji_hand import WujiHand,ObjectSurface,MaterialPoint,build_scene
    from models.wuji_comfree import ComFreeEngine,build_comfree_scene,comfree_model_factory
    from planning.wuji_retarget_mpc import WujiMPCParams
    native=matrix/'sequence/scene/wuji_right/sugar_box_mesh.xml'
    surface=ObjectSurface(matrix/'sequence/model/textured_simple.obj',samples=100,convex=True)
    hand=WujiHand(native,surface)
    prediction=build_comfree_scene(native,forge/'assets/wuji/right.xml',out/'prediction.xml')
    engine=ComFreeEngine(prediction,forge)
    physics=engine.validate_physics(hand)
    sys.path.insert(0,str(forge/'forceaware'))
    from forceaware.config import load_retarget_config
    from forceaware.executor import WarpActionExecutor
    from forceaware.targets import build_index
    cfg=load_retarget_config(matrix/'sequence',matrix/'forceaware.yaml',hand='wuji_right',contactaware_dir=matrix/'contactaware')
    reference=WarpActionExecutor(engine.model,engine.device_model,index=build_index(engine.model),cfg=cfg)
    # The archived diagnostic uses the upstream row layout independently of
    # the production slot assignment, making nonzero force checks meaningful.
    from examples.retarget.matrix_comfree import ComFreeEngine as ReferenceContactReader
    reader=object.__new__(ReferenceContactReader)
    reader.compiled=engine.compiled;reader.model=engine.model;reader.index=engine.index
    bodies=np.unique(engine.compiled.base.collision.output_body_ids.numpy())
    materials=[MaterialPoint(int(b),np.zeros(3),np.array([1.,0,0]),'diagnostic','diagnostic') for b in bodies]
    active=np.ones(len(materials));prior=np.load(matrix/'contactaware/wuji_right_qpos.npy')
    poses=np.load(matrix/'contactaware/object_pose_7.npy')[:,[4,5,6,3,0,1,2]]
    model=comfree_model_factory(engine)(WujiMPCParams(hand,len(materials)),len(materials));model.set_contact_materials(materials,active)
    records=[];nonzero_force=False
    for frame in [0,31,71]:
        q=engine.model.qpos0.copy();q[engine.qadr]=prior[frame];q[engine.oq:engine.oq+7]=poses[frame]
        v=np.zeros(32);target=np.clip(prior[frame]+np.r_[np.zeros(6),np.full(20,.01)],hand.lower,hand.upper)
        result_q,result_v,buffer=engine.step(q,v,target)
        expected=reference.execute_action_sequence(q,v,actions=np.repeat(target[None],3,axis=0))
        state_error=max(np.max(np.abs(result_q[0]-expected.next_qpos)),np.max(np.abs(result_v[0]-expected.next_qvel)))
        force=engine.contact_forces(buffer,materials,active).reshape(-1,3);expected_force=np.zeros_like(force)
        for contact in reader.contacts(buffer):
            ids=np.flatnonzero(bodies==contact['body'])
            if len(ids):expected_force[ids[0]]+=contact['force']
        force_error=np.max(np.abs(force-expected_force));nonzero_force|=np.linalg.norm(force)>.01
        assert state_error<1e-6,state_error
        assert force_error<1e-5,force_error
        x=engine.pack(q[None],v[None],prior[frame][None])[0];u=target-prior[frame]
        predicted_force=model.contact_force_fn(x,u,None,None,None).reshape(-1,3)
        assert np.allclose(predicted_force,expected_force,atol=1e-5,rtol=0)
        f,jx,ju=model.force_linearization_fn(x,u,None,None,None)
        assert np.isfinite(np.r_[f,jx.ravel(),ju.ravel()]).all()
        records.append(dict(frame=frame,maximum_state_error=float(state_error),maximum_force_error_N=float(force_error),total_hand_force_N=np.sum(force,axis=0).tolist()))
    assert nonzero_force,'Force validation must include a loaded contact'
    tuned_native=build_scene(forge/'assets/wuji/right_mesh.xml',ROOT/'outputs/retarget_shared_20261016/scene/leaphand/sugar_box_mesh.xml',
        out/'tuned_native.xml',poses[0],.002,finger_stiffness_scale=3.,finger_force_scale=2.,
        physics_reference=ROOT/'outputs/retarget_shared_20261016/dexforge_sdf_physics.xml')
    tuned_hand=WujiHand(tuned_native,surface)
    tuned_scene=build_comfree_scene(tuned_native,forge/'assets/wuji/right.xml',out/'tuned_prediction.xml')
    tuned_engine=ComFreeEngine(tuned_scene,forge)
    tuned_differences=tuned_engine.validate_physics(tuned_hand)
    audit=dict(default_physics_max_differences=physics,upstream_executor_and_contact_checks=records,
        tuned_physics_max_differences=tuned_differences,tuning=dict(finger_stiffness_scale=3.,finger_force_scale=2.),
        force_predictions='nonzero official ComFree constraint-row forces; no SCM force fallback')
    (out/'prediction_validation.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
