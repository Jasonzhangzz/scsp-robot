#!/usr/bin/env python3
"""Certify paired inputs, full-duration actuation and cohort completeness."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from planning.point_force_benchmark import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    directory=a.directory.resolve();protocol=json.loads((directory/'protocol.json').read_text())
    initial=json.loads((directory/'initial_states.json').read_text())
    expected={(c,o,s) for c in protocol['configs'] for o in protocol['objects'] for s in protocol['seeds']}
    found=set();groups={};failed=[];steps=rounds=0;max_force=0.;max_impulse_error=0.;faults=collections.Counter();masses=set()
    changed=[s for s,h in protocol['source_sha256'].items() if hashlib.sha256((ROOT/s).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    for source,digest in protocol['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/source).read_bytes()).hexdigest()==digest,source
    for config,obj,seed in sorted(expected):
        path=directory/config/f'{obj}_{seed:03d}.json.gz'
        with gzip.open(path,'rt') as stream:payload=json.load(stream)
        summary=payload['metadata']['trial_summary'];rows=payload['rows'];init=payload['initialization']
        key=(summary['configuration'],summary['object'],summary['seed']);assert key==(config,obj,seed)
        assert key not in found;found.add(key)
        reference=initial[f'{obj}:{seed}']
        masses.add(float(init['source_mass']))
        for source,target in [('initial_pose','pose'),('target_pose','target'),('sample_points','points'),('sample_frames','frames')]:
            np.testing.assert_array_equal(init[source],reference[target],err_msg=str(path))
        assert all(v['passed'] for v in init['validation']),path
        assert not summary['success'] or summary['error'] is None,path
        if summary['success']:
            last=rows[-1]
            assert last['success'] and last['settled'] and last['success_hold']>=10,path
            assert last['goal_pos']<.02 and last['goal_quat']<.015,path
        for row in rows:
            info=row['execution_info'];steps+=1
            assert info['pose_overwrites']==0 and not info['force_was_clipped'],path
            force=np.asarray(row['applied_force_body']);max_force=max(max_force,float(np.linalg.norm(force)))
            # Native-reset/partial terminal cycles retain planned impulse metadata;
            # certify all fully completed cycles, exposing partial ones separately.
            if row['execution_completed']:
                if row['command_valid_for_step']:
                    assert info['active_substeps']==protocol['physics_substeps'],path
                    assert np.isclose(info['applied_force_duration'],protocol['policy_dt']),path
                else:
                    assert info['active_substeps']==0 and np.all(force==0),path
                error=float(np.max(np.abs(np.asarray(info['actual_applied_impulse_body'])[:3]-force*protocol['policy_dt'])))
                max_impulse_error=max(max_impulse_error,error);assert error<1e-10,path
            else:faults['partial_terminal_cycles']+=1
        rounds+=len(payload['planning_rounds'])
        if summary['success']:reason='success'
        elif any((r.get('physics_fault') or {}).get('clock_reset') for r in rows):reason='native_reset'
        elif summary['error'] and 'physical execution became unstable' in summary['error']:reason='physics_instability'
        elif summary['error'] and 'No contact point above the ground threshold' in summary['error']:reason='no_feasible_contact'
        elif summary['error']:reason='other_exception'
        else:reason='timeout'
        for object_key in (obj,'ALL'):
            group=groups.setdefault((config,object_key),dict(configuration=config,object=object_key,total=0,successes=0,outcomes=collections.Counter()))
            group['total']+=1;group['successes']+=int(summary['success']);group['outcomes'][reason]+=1
        if reason=='other_exception':failed.append(dict(configuration=config,object=obj,seed=seed,error=summary['error']))
    assert found==expected,(found^expected)
    report=dict(passed=not failed,complete=True,trials=len(found),unique_paired_states=len(initial),
        verified_control_cycles=steps,verified_planning_rounds=rounds,
        maximum_applied_force_N=max_force,maximum_impulse_error_Ns=max_impulse_error,
        object_masses_kg=sorted(masses),
        source_hashes_unchanged=True,partial_cycles=faults['partial_terminal_cycles'],
        unexpected_exceptions=failed,groups=list(groups.values()))
    write_json(directory/'audit.json',report)
    lines=['| Configuration | Success | Rate | Timeout | Instability | Native reset | No feasible contact |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for config in protocol['configs']:
        g=groups[(config,'ALL')];counts=g['outcomes']
        lines.append(f'| {config} | {g["successes"]}/{g["total"]} | {100*g["successes"]/g["total"]:.1f}% | '
            f'{counts["timeout"]} | {counts["physics_instability"]} | {counts["native_reset"]} | {counts["no_feasible_contact"]} |')
    (directory/'failure_breakdown.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2))
    print('\n'.join(lines))

if __name__=='__main__':main()
