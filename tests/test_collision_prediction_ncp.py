"""FCL independent geometry, shared-input SCM and NCP integration checks."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('fcl')
from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()
import mujoco
from examples.mpc.fingertips.test.collision_prediction_assets import scene_xml
from examples.mpc.fingertips.test.collision_prediction_core import Settings
from examples.mpc.fingertips.test.collision_prediction_ncp import NCPPredictor
from models.ncp_contact import NCPSettings


@pytest.fixture
def predictor(tmp_path):
    path=tmp_path/'model.xml';path.write_text(scene_xml('cube')[0])
    return NCPPredictor(path,Settings())


def sample(predictor):
    q=predictor.model.qpos0.copy();q[:3]=[0.,0.,.03];q[7:10]=[.03999,0.,.03]
    v=np.zeros(12);v[6]=-.1
    return dict(qpos=q,qvel=v,wrench=np.zeros((3,6)))


def test_plane_preserves_four_box_corners_and_pair_normal(predictor):
    initial=sample(predictor);m=predictor.initial_manifold(initial)
    floor=np.all(m['body']==[0,1],axis=1)
    pair=np.all(m['body']==[1,2],axis=1)
    assert floor.sum()==4 and pair.sum()==1
    np.testing.assert_allclose(m['gap'][floor],0.,atol=1e-12)
    np.testing.assert_allclose(m['frame'][pair,0],[[1.,0.,0.]],atol=1e-12)
    assert m['gap'][pair][0]==pytest.approx(-1e-5,abs=1e-10)


def test_predictors_never_call_mujoco_collision_or_constraint_solver(predictor,monkeypatch):
    initial=sample(predictor)
    def forbidden(*args,**kwargs): raise AssertionError('MuJoCo contact call in NCP')
    for name in ('mj_forward','mj_collision','mj_fwdPosition','mj_fwdConstraint'):
        monkeypatch.setattr(mujoco,name,forbidden)
    m=predictor.initial_manifold(initial)
    assert predictor.scm(initial,m)['status']=='ok'
    for steps in (1,10):
        prediction=predictor.ncp(initial,m,steps)
        assert prediction['status']=='ok',prediction
        assert len(prediction['diagnostics'])==steps
        for problem in prediction['problems']:
            assert np.isfinite(problem['impulse']).all()
    np.testing.assert_array_equal(initial['qvel'],sample(predictor)['qvel'])


def test_failed_ncp_does_not_publish_endpoint(predictor):
    predictor.ncp_settings=NCPSettings(max_evaluations=1,max_projection_iterations=0)
    initial=sample(predictor);initial['qvel'][:3]=[.2,.15,-.1]
    result=predictor.ncp(initial)
    assert result['status']=='failed' and result['qpos'] is None
    assert result['diagnostics'] and not result['diagnostics'][-1]['converged']


def test_cli_writes_complete_replayable_results(tmp_path):
    entry=Path(__file__).resolve().parents[1]/'examples/mpc/fingertips/test/benchmark_collision_prediction_ncp.py'
    output=tmp_path/'run'
    subprocess.run([sys.executable,str(entry),'run','--objects','sphere','cube',
                    '--samples-per-mode','1','--output-dir',str(output)],check=True,capture_output=True,text=True)
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['n_prediction_failures']==0
    assert (output/'summary.csv').stat().st_size>0
    query=next((output/'cube/queries').glob('*.npz'))
    replay=subprocess.run([sys.executable,str(entry),'replay','--query',str(query)],
                          check=True,capture_output=True,text=True)
    result=json.loads(replay.stdout)
    assert result['ncp']['max_qvel_difference']<1e-12
    assert result['ncp_substeps']['max_qvel_difference']<1e-12
