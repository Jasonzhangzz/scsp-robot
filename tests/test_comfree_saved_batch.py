"""Opt-in CUDA regression for one-step native ComFree world batching."""
import os
import numpy as np
import pytest
from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()
from examples.mpc.fingertips.test import collision_prediction_assets as assets
from examples.mpc.fingertips.test.collision_prediction_core import Settings
from examples.mpc.fingertips.test.benchmark_comfree_saved_batch import ComfreeBatch


@pytest.mark.skipif(not os.getenv('SCSP_COLLISION_GPU_TEST'), reason='opt-in CUDA test')
def test_comfree_batch_matches_single_world_and_keeps_inputs(tmp_path):
    xml, _ = assets.scene_xml('cube')
    assets.package_scene(xml,tmp_path)
    engine=ComfreeBatch(tmp_path,Settings())
    model=engine.model
    samples=[]
    for i in range(4):
        q=model.qpos0.copy(); q[:3]=[0,0,.02999]; q[7:10]=[.03999,0,.03]
        v=np.zeros(model.nv);v[6:9]=[-.2,.02*i,0]
        f=np.zeros((model.nbody,6));f[model.body('ball').id,:3]=[-.1*(i+1),.05,0]
        samples.append(dict(qpos=q,qvel=v,wrench=f))
    original=[{k:v.copy() for k,v in s.items()} for s in samples]
    batched=engine.evaluate(samples)
    for sample,result,previous in zip(samples,batched,original):
        serial=engine.evaluate([sample])[0]
        assert result['status']=='ok' and result['actual_nworld']==4
        np.testing.assert_allclose(result['qpos'],serial['qpos'],atol=2e-5,rtol=0)
        np.testing.assert_allclose(result['qvel'],serial['qvel'],atol=5e-3,rtol=0)
        for key in sample:np.testing.assert_array_equal(sample[key],previous[key])
