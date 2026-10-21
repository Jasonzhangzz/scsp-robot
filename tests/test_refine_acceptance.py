"""Refinement must reuse complete matched outcomes and simulate only missing points."""
import json
from pathlib import Path

import pytest

from examples.mpc.franka.experiments.mpc_ablation import refine_acceptance as refine


def make_source(root, values):
    cases = [dict(object=obj, trial=i, id='case_%d'%i, baseline_success=success)
             for i, (obj, success) in enumerate(refine.sensitivity.MIXED_CASES)]
    cells = []
    root.mkdir()
    for value, successes in values:
        directory=root/str(value);directory.mkdir()
        rows=[dict(object=c['object'],trial=c['trial'],success=i<successes,
                   verify_beta=.18,verify_quality_threshold=value,acceptance_threshold=value,
                   acceptance_gate_version='quality_gate_v1',acceptance_gate_steps=10,
                   acceptance_endpoint_violations=0) for i,c in enumerate(cases)]
        (directory/'trials.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        cells.append(dict(value=value,directory=str(directory)))
    (root/'grid.json').write_text(json.dumps(dict(mode='acceptance',verify_beta=.18,
                                                cases=cases,cells=cells)))
    return cases


def test_refinement_reuses_exact_grid_points_without_rerunning_center(tmp_path,monkeypatch):
    latest,old=tmp_path/'latest',tmp_path/'old'
    cases=make_source(latest,[(.65,4),(.70,3)])
    make_source(old,[(.60,3),(.65,2)])
    calls=[]
    def run(command,directory,obj,trial,**kwargs):
        value=float(command[command.index('--verify-quality-threshold')+1])
        calls.append((value,obj,trial))
        return dict(object=obj,trial=trial,success=True)
    monkeypatch.setattr(refine.sensitivity,'_mixed_trial',run)
    monkeypatch.setattr(refine,'plot_local',lambda *a:None)
    from types import SimpleNamespace
    output=tmp_path/'sweep'
    refine.run(SimpleNamespace(center=.65,radius=.05,step=.01,workers=3,
                              reuse=[latest,old],output=output,dry_run=False))
    assert len(calls)==40
    assert {v for v,_,_ in calls}=={.61,.62,.63,.64,.66,.67,.68,.69}
    for value in {v for v,_,_ in calls}:
        assert {(obj,trial) for v,obj,trial in calls if v==value}=={
            (c['object'],c['trial']) for c in cases}
    rows=json.loads((output/'sensitivity_summary.json').read_text())
    center=next(r for r in rows if r['value']==.65)
    assert center['successes']==4 and center['source_role']=='reused'
    assert all(r['trials']==5 for r in rows)
    assert len(rows)==11 and (output/'complete.json').is_file()


def test_bad_reused_outcomes_stop_before_simulation(tmp_path):
    source=tmp_path/'source';cases=make_source(source,[(.65,4)])
    p=source/'0.65/trials.jsonl'
    p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n')
    with pytest.raises(ValueError,match='exactly once'):
        refine.saved_cells([source],cases,.18)
    assert refine.refinement_grid(.65,.05,.01)==[
        .65,.64,.66,.63,.67,.62,.68,.61,.69,.6,.7]
    for options in ((.65,.055,.01),(.99,.05,.01),(.65,.05,0)):
        with pytest.raises(ValueError):refine.refinement_grid(*options)
