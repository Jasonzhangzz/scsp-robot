from pathlib import Path
import json, os, subprocess, sys
root=Path(sys.argv[1]).resolve()
obj, cpu=sys.argv[2],sys.argv[3]
env=os.environ.copy()
env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
env['LD_LIBRARY_PATH']=str(Path(sys.executable).parents[1]/'lib')+os.pathsep+env.get('LD_LIBRARY_PATH','')
for task in ('rotation','flipping'):
    cell=root/task/obj
    rows=cell/'trials.jsonl'
    assert not rows.exists() or not rows.read_text().strip(), 'Completed trials must never be overwritten'
    old=cell/'run.log'
    if old.exists(): old.rename(cell/'dependency_failure.log')
    command=json.loads((cell/'command.json').read_text())
    with (cell/'run.log').open('w') as log:
        result=subprocess.run(['taskset','-c',cpu,*command],env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise SystemExit(result.returncode)
    print('[a7 recovery] '+task+'/'+obj+' completed',flush=True)
