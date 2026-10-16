"""Run independent one-factor FA ablations and preserve exact commands/status."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['probes','closed-loop'],default='probes')
    p.add_argument('--factors',nargs='+',default=['baseline','budget3','single-start','slsqp','acados','absolute-warm',
        'full-contact-jacobian','dense-smoothing','control-bounds','slow-templates','dense-controls'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    logpath=a.output/(a.mode+'_sweep.jsonl')
    for factor in a.factors:
        directory=a.output/('probes' if a.mode=='probes' else 'comfree')/factor
        if (directory/('probes.jsonl' if a.mode=='probes' else 'rollout.npz')).exists():continue
        if a.mode=='closed-loop' and (directory/'NO_OP.json').exists():continue
        if a.mode=='probes':command=[sys.executable,'-m','examples.retarget.probe_fa_ablation','--factor',factor,
            '--windows','0','80','120','--output',str(directory)]
        else:
            base=factor.removesuffix('-contiguous')
            command=[sys.executable,'-m','examples.retarget.matrix_fa_worker','--experiment','outputs/fa_admm_solver_20261016',
                '--forge','thirdparty/retarget_assets/DexForge','--executor','comfree','--variant','lbfgs','--duration','5.4',
                '--iterations','3' if base=='budget3' else '8' if base=='budget8' else '40','--multistarts','1' if base=='single-start' else '8',
                '--case-name',factor,'--output-base',str(a.output)]
            if base in ('slsqp','acados'):command+=['--hand-solver',base,'--hand-rti-iterations','40']
            elif base not in ('baseline','budget3','budget8','single-start'):command+=['--ablation',base]
            if factor.endswith('-contiguous'):command+=['--hand-single-block']
            command+=['--stop-after-drop']
        started=time();log=a.output/(a.mode+'_'+factor+'.log')
        with log.open('w') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
        row=dict(factor=factor,mode=a.mode,command=command,exit_code=result.returncode,seconds=time()-started,log=str(log))
        with logpath.open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if result.returncode:raise RuntimeError(f'{factor}: see {log}')


if __name__=='__main__':main()
