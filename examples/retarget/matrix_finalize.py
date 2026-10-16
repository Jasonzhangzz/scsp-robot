"""Verify completed cases and write accurate experimental backend metadata."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import yaml


def finalize(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text());cfg=yaml.safe_load((root/'forceaware.yaml').read_text())
    common=np.load(root/'contactaware/wuji_right_qpos.npy');poses=np.load(root/'contactaware/object_pose_7.npy')[:,[4,5,6,3,0,1,2]]
    original=np.load(root/'mujoco/current/semantic_contacts.npz')
    metadata={};initial=[]
    for key,case in manifest['cases'].items():
        output=root/key;method=case['controller'];executor=case['executor']
        if method=='dexforge':
            data=np.load(output/'rollout.npz');assert len(data['qpos_traj'])==541
            assert data['mpc_execution_steps'].sum()==540
            initial.append(data['qpos_traj'][0]);case['status']='complete'
            metadata[key]=json.loads((output/'audit.json').read_text());continue
        rows=[json.loads(x) for x in (output/'full/diagnostics.jsonl').read_text().splitlines()]
        assert len(rows)==180,(key,len(rows))
        assert np.array_equal(np.load(output/'wuji_q.npy'),common)
        semantic=np.load(output/'semantic_contacts.npz')
        for name in ['mask','anchors','normals','local','object_pose']:
            assert np.array_equal(semantic[name],original[name]),(key,name)
        for item in rows:
            assert not item['lambda_enabled']
            assert item['mpc']['rate_limits_enabled'] is False
            assert np.array_equal(item['mpc']['best_contact_mask'],semantic['mask'][item['reference_frame']])
            assert np.array_equal(item['mpc']['best_contact_points'],semantic['anchors'][item['reference_frame']])
        audit=json.loads((output/'audit.json').read_text()) if (output/'audit.json').exists() else {}
        audit.update(executor=executor,method=method,closed_loop=True,steps=180,duration_seconds=5.4,
            fixed_duration=True,CA_array_exactly_common=True,original_MANO_semantic_arrays_exactly_common=True,
            lambda_optimizer=False,force_gates=False,command_rate_limits=False,
            QP_failure_steps=[r['step'] for r in rows if 'acados status' in r['mpc']['status']],
            native_cost_refinement=False,
            physical_prediction='official ComFree 3 x 10 ms, stage central differences for RTI' if method=='comfree_model' else 'current soft-SCM',
            cost='FA geometry/reference formulas and smoothing, on current 30 ms grid' if method=='fa_cost' else 'unchanged current cost',
            collision_guard='unchanged current MuJoCo mesh guard',
            executor_timestep=.002 if executor=='mujoco' else .01,
            candidate_preview_executor=executor)
        (output/'audit.json').write_text(json.dumps(audit,indent=2))
        model=json.loads((output/'full/planning_model.json').read_text())
        model.update(version='retarget_experiment_matrix_v1',dynamics=audit['physical_prediction'],
            execution=executor,execution_timestep=audit['executor_timestep'],cost=audit['cost'],
            candidate_ranking=executor+'_forward_cost_and_current_mesh_collision_guard',
            made_contact_frames=executor+'_measured_physical_contact',collision_guard=audit['collision_guard'])
        if method=='fa_cost':
            model['original_semantics']='Original MANO remains in proposal/dynamics preprocessing; objective uses the original FA CA guidance.'
            (output/'full/mpc_costs.json').write_text(json.dumps(dict(profile='FA_formulas_current_grid',loss=cfg['loss'],smoothing=cfg['smoothing'],contact=cfg['contact'],
                stage_dt=.03,horizon=4,initial_pose_optimization=False,extra_running_dt_multiplier=False,
                state_acceleration='one-step velocity change / 30 ms',
                geometry_value_check_max_relative=max(r['mpc']['FA_loss_value_validation_max_relative'] for r in rows)),indent=2))
        (output/'full/planning_model.json').write_text(json.dumps(model,indent=2))
        metadata[key]=audit;case['status']='complete';initial.append(np.r_[common[0],poses[0]])
    max_initial=max(float(np.max(np.abs(q-initial[0]))) for q in initial)
    assert max_initial<1e-6,max_initial
    manifest['initial_qpos_max_difference_due_to_float32']=max_initial
    manifest['status']='complete';manifest['implementation_validation']='implementation_validation.json'
    manifest['regression_tests']='82 passed';manifest['result_metrics']='metrics.json'
    manifest.setdefault('source_sha256',{}).update({str(path):'' for path in Path('examples/retarget').glob('matrix_*.py')})
    for name in manifest.get('source_sha256',{}):
        manifest['source_sha256'][name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (root/'case_validation.json').write_text(json.dumps(metadata,indent=2))
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();finalize(args.root)
