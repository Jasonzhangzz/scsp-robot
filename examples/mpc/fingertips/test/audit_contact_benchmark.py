"""Audit a completed physical-contact cohort without running simulations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(directory):
    directory = Path(directory)
    protocol = json.loads((directory/'protocol.json').read_text())
    expected = {(rollout,model,obj,seed) for rollout in ('sync','async')
                for model in protocol['models'] for obj in protocol['objects']
                for seed in protocol['seeds']}
    found,starts,targets,surfaces = set(),{},{},{}
    verified_rows = planning_calls = reset_failures = 0
    failures = {}
    initialization_failures=[]
    for rollout,model,obj,seed in sorted(expected):
        path = directory/rollout/f'trial_{model}_{obj}_{seed:03d}.json'
        payload = json.loads(path.read_text())
        trial,rows,rounds = payload['metadata']['trial_summary'],payload['rows'],payload['planning_rounds']
        assert payload['metadata']['formal_trial'] is True,path
        key = trial['rollout'],trial['model'],trial['object'],trial['seed']
        assert key == (rollout,model,obj,seed) and key not in found,path
        found.add(key)
        initial = payload['initialization']
        if not initial:
            # An exception before constructing/validating the predictor is an
            # attempted failed trial, not a fabricated initialized backend.
            assert trial['error'] and not rows and not rounds and not trial['success'],path
            assert trial['steps']==0 and trial['completion_steps'] is None,path
            assert path.with_suffix('.log').exists(),path
            initialization_failures.append(dict(rollout=rollout,model=model,object=obj,seed=seed,error=trial['error']))
            reason=(rollout,model,'initialization_exception')
            failures[reason]=failures.get(reason,0)+1
            continue
        impulse_limits = protocol.get('force_constraint')
        if isinstance(impulse_limits,dict):
            w,x,y,z=initial['source_inertial_quaternion']
            cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            rotation=np.eye(3)+2*w*cross+2*cross@cross
            inertia=rotation@np.diag(initial['source_inertia'])@rotation.T
        pair = obj,seed
        for mapping,value in ((starts,initial['initial_pose']),(targets,initial['target_pose']),
                              (surfaces,initial['sample_points'])):
            if pair in mapping:
                np.testing.assert_allclose(mapping[pair],value,atol=1e-12,rtol=0)
            mapping[pair] = value
        assert trial['actual_backend'] == protocol['backends'][model],path
        assert trial['planning_rounds'] == len(rounds),path
        times = [r['planning_time_ms'] for r in rounds if r.get('planning_time_ms') is not None]
        assert len(times) == len(rounds),f'Untimed planning call: {path}'
        assert len(times) == trial['planning_time_ms']['n'],path
        if times:
            assert min(times)>=0 and np.isfinite(times).all(),path
            np.testing.assert_allclose(np.mean(times),trial['planning_time_ms']['mean'],rtol=1e-12)
        planning_calls += len(rounds)
        for call in rounds:
            if call.get('published') and 'planned_execution_step' in call:
                assert call['published_step']==call['planned_execution_step'],path
        for index,row in enumerate(rows):
            assert row['step']==index+1,path
            force = np.asarray(row['optimal_force_contact'])
            assert np.linalg.norm(force)<=protocol['args']['max_contact_force']+1e-8,path
            assert force[0]>=-1e-12,path
            assert abs(force[1])+abs(force[2])<=protocol['args']['arm_friction']*force[0]+1e-8,path
            if isinstance(impulse_limits,dict):
                force_body=np.asarray(initial['sample_frames'][row['selected_idx']])@force
                lever=np.asarray(row['best_contact_point'])-initial['source_com']
                assert protocol['policy_dt']*np.linalg.norm(force_body)/initial['source_mass']<=impulse_limits['force_only_COM_delta_v_m_s']+1e-8,path
                angular=protocol['policy_dt']*np.linalg.solve(inertia,np.cross(lever,force_body))
                assert np.linalg.norm(angular)<=impulse_limits['force_only_angular_delta_v_rad_s']+1e-8,path
            if row.get('async_planner') and row.get('command_valid_for_step') is False:
                np.testing.assert_array_equal(row['applied_force_body'],np.zeros(3))
            if row.get('async_planner') and row.get('command_valid_for_step') and 'command_hold_duration_s' in row:
                duration=row['command_hold_duration_s']
                assert row['command_age_at_start_s']<=initial['async_max_command_age_s']+1e-8,path
                assert duration*np.linalg.norm(force_body)/initial['source_mass']<=impulse_limits['force_only_COM_delta_v_m_s']+1e-8,path
                assert duration*np.linalg.norm(np.linalg.solve(inertia,np.cross(lever,force_body)))<=impulse_limits['force_only_angular_delta_v_rad_s']+1e-8,path
            info = row['execution_info']
            assert info['pose_overwrites']==0 and not info['force_was_clipped'],path
            np.testing.assert_allclose(np.asarray(info['actual_applied_impulse_body'])[:3],
                np.asarray(row['applied_force_body'])*info['applied_force_duration'],atol=1e-10,rtol=1e-12)
            if row['settling'] or row.get('command_valid_for_step') is False:
                assert info['active_substeps']==0,path
            elif row['execution_completed']:
                assert info['active_substeps']==protocol['physics_substeps'],path
            assert row['planning_time_ms'] is None or row['planning_updated'],path
            assert row['gpu_parallel_prediction'] or trial['actual_backend']=='cpu-adamw',path
            cycles = row['completed_execution_steps']
            remainder = 0 if row['execution_completed'] else row['execution_substeps']
            np.testing.assert_allclose(row['simulation_time_s'],
                cycles*protocol['policy_dt']+remainder*protocol['policy_dt']/protocol['physics_substeps'],atol=1e-10)
            if row.get('physics_fault',{}):
                fault = row['physics_fault']
                if fault['clock_reset']:
                    reset_failures += 1
                assert row['mujoco_unstable'] and not row['success'],path
                assert index == len(rows)-1,path
            verified_rows += 1
        if rows:
            assert trial['steps']==rows[-1]['completed_execution_steps'],path
        assert trial['attempted_steps']==len(rows),path
        if trial['success']:
            final = rows[-1]
            assert not trial['error'] and not final['mujoco_unstable'],path
            assert final['goal_pos']<.02 and final['goal_quat']<.015,path
            assert final['success_hold']>=protocol['args']['success_hold_steps'] and final['settled'],path
            assert trial['completion_steps']==trial['steps'],path
        else:
            assert trial['completion_steps'] is None,path
            reason = ('native_reset' if rows and (rows[-1].get('physics_fault') or {}).get('clock_reset')
                else ('physics_instability' if rows and rows[-1]['mujoco_unstable'] else
                ('planning_exception' if trial['error'] else 'step_limit')))
            failures[(rollout,model,reason)] = failures.get((rollout,model,reason),0)+1
    assert found==expected
    parity = json.loads((directory/'viewer_consistency'/'summary.json').read_text())
    assert parity['passed'] and len(parity['results'])==2*len(protocol['models'])
    changed = []
    repository = Path(__file__).resolve().parents[4]
    for name,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==digest,name
        if hashlib.sha256((repository/name).read_bytes()).hexdigest()!=digest:
            changed.append(name)
    assert not changed,f'Experiment sources changed after freezing: {changed}'
    report = dict(passed=True,total_trials=len(found),verified_execution_rows=verified_rows,
        verified_planning_calls=planning_calls,matched_initial_state_groups=len(starts),
        matched_targets=True,matched_contact_surfaces=True,continuous_force_verified=True,
        source_snapshot_verified=True,viewer_parity_checks=len(parity['results']),
        asynchronous_publication_steps_verified=True,shared_force_impulse_limits_verified=isinstance(protocol.get('force_constraint'),dict),
        initialization_failure_trials=initialization_failures,
        native_reset_failures=reset_failures,
        failures=[dict(rollout=r,model=m,reason=reason,count=n) for (r,m,reason),n in sorted(failures.items())])
    (directory/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(json.dumps(audit(parser.parse_args().directory),indent=2))
