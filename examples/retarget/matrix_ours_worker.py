"""Isolated ablation worker; production defaults are left unchanged."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--executor',choices=['mujoco','comfree'],required=True)
    p.add_argument('--method',choices=['current','comfree_model','fa_cost'],required=True)
    p.add_argument('--steps',type=int,default=180)
    a=p.parse_args();experiment=a.experiment.resolve()
    from examples.retarget.experiment_matrix import environment
    import os
    os.environ.update(environment())
    forge=ROOT/'thirdparty/retarget_assets/DexForge'
    from examples.retarget import retarget
    import planning.wuji_retarget as reference_runtime
    # Fixed-duration benchmark: continue holding after first reaching tolerance.
    reference_runtime.tracking_task_complete=lambda *args,**kwargs:False
    import planning.wuji_retarget_mpc as module
    engine=None
    if a.executor=='comfree' or a.method in ('comfree_model','fa_cost'):
        from examples.retarget.matrix_comfree import ComFreeEngine
        engine=ComFreeEngine(experiment,forge)
    if a.executor=='comfree':
        from examples.retarget.matrix_comfree import attach_executor
        original_hand=retarget.WujiHand
        def hand_factory(*args,**kwargs):return attach_executor(original_hand(*args,**kwargs),engine)
        retarget.WujiHand=hand_factory
    original_mpc=module.WujiOnlineMPC
    def mpc_factory(*args,**kwargs):
        if a.method=='comfree_model':
            from examples.retarget.matrix_comfree import model_factory,NumericFunction
            kwargs['model_factory']=model_factory(engine)
        mpc=original_mpc(*args,**kwargs)
        if a.method=='comfree_model':
            mpc.stage_prediction_fn=NumericFunction(mpc.stage_prediction_fn,
                lambda x,u,phi,jac,mp,reference:mpc.physics.forward(x,u))
        if a.method=='fa_cost':
            from examples.retarget.matrix_fa_cost import attach_fa_cost
            attach_fa_cost(mpc,engine)
        return mpc
    module.WujiOnlineMPC=mpc_factory
    case=experiment/a.executor/a.method
    started=time.time()
    retarget.main(['--contact-model','soft-scm','--sequence',str(ROOT/'outputs/retarget_shared_20261016'),
        '--physics-reference',str(ROOT/'outputs/retarget_shared_20261016/dexforge_sdf_physics.xml'),
        '--ca-result',str(ROOT/'outputs/wuji_contact_fix_20261016/contactaware'),
        '--ca-scene',str(ROOT/'outputs/wuji_contact_fix_20261016/ca_sequence/scene/wuji_right/sugar_box_mesh.xml'),
        '--headless','--no-demo-overlay','--max-steps',str(a.steps),'--output',str(case)])
    q=np.load(case/'wuji_q.npy');common=np.load(experiment/'contactaware/wuji_right_qpos.npy')
    if not np.array_equal(q,common):raise RuntimeError('CA reference changed between arms')
    (case/'audit.json').write_text(json.dumps(dict(executor=a.executor,method=a.method,closed_loop=True,
        lambda_optimizer=False,force_gates=False,CA_array_exactly_common=True,
        solver='one SQP_RTI update plus unchanged seed/collision/cost ranking',
        prediction='official ComFree hard contact, 3x10ms, stage central-difference linearization' if a.method=='comfree_model' else 'current soft-SCM',
        cost='FA formulas on current 30ms/120ms grid' if a.method=='fa_cost' else 'current',
        wall_seconds=time.time()-started),indent=2))


if __name__=='__main__':main()
