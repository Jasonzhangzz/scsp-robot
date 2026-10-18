#!/usr/bin/env python3
"""Detach the gated 168-trial benchmark and audit it automatically on exit."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import traceback
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[4]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write_status(directory,**updates):
    path=directory/'STATUS.json'
    value=json.loads(path.read_text()) if path.exists() else {}
    value.update(updates,updated_at=now())
    temp=path.with_suffix(f'.{os.getpid()}.json.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def worker(directory,extra):
    sys.path.insert(0,str(ROOT))
    from examples.mpc.fingertips.test.benchmark_contact_models import SOURCES
    frozen={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    (directory/'launch_source_sha256.json').write_text(json.dumps(frozen,indent=2)+'\n')
    write_status(directory,status='running',phase='preflight',supervisor_pid=os.getpid(),
        expected_trials=168,rollout_order=['sync','async'],started_at=now())
    try:
        benchmark=[sys.executable,'-u','examples/mpc/fingertips/test/benchmark_contact_models.py',
            '--output-dir',str(directory),*extra]
        # Separate phases expose meaningful status without a monitoring loop.
        for phase in ('preflight','viewer','sync','async'):
            write_status(directory,status='running',phase=phase)
            changed=[name for name,digest in frozen.items()
                if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
            if changed:
                raise RuntimeError(f'Frozen experiment sources changed: {changed}')
            with (directory/'controller.log').open('a',buffering=1) as log:
                print(f'{now()} PHASE {phase}',file=log,flush=True)
                process=subprocess.Popen([*benchmark,'--phase',phase],cwd=ROOT,stdout=log,
                    stderr=subprocess.STDOUT,env=os.environ.copy())
                write_status(directory,controller_pid=process.pid)
                code=process.wait()
            if code:
                raise RuntimeError(f'Benchmark phase {phase} exited {code}; see controller.log')
            if phase=='preflight':
                protocol=json.loads((directory/'protocol.json').read_text())
                write_status(directory,expected_trials=2*len(protocol['models'])*len(protocol['objects'])*len(protocol['seeds']))
        write_status(directory,phase='audit')
        with (directory/'audit.log').open('w') as log:
            subprocess.run([sys.executable,'examples/mpc/fingertips/test/audit_contact_benchmark.py',
                str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        report=json.loads((directory/'summary.json').read_text())
        write_status(directory,status='completed_and_audited',phase='complete',
            completed_trials=report['completed_trials'],completed_at=now())
    except BaseException as exc:
        write_status(directory,status='failed',error=f'{type(exc).__name__}: {exc}',failed_at=now())
        traceback.print_exc()
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args,extra=parser.parse_known_args()
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    directory=(args.output_dir or ROOT/'log'/f'contact_models_{stamp}').resolve()
    directory.mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker(directory,extra);return
    if (directory/'STATUS.json').exists():
        parser.error('Use a fresh output directory; existing cohorts must not be overwritten')
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        SCSP_PLANNER_ONLY='1',PYTHONUNBUFFERED='1',LIBGL_ALWAYS_SOFTWARE='1')
    (directory/'controller.log').touch()
    write_status(directory,status='launched',launched_at=now(),
        output_directory=str(directory),command_arguments=extra)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker','--output-dir',str(directory),*extra]
    if shutil.which('xvfb-run'):
        command=['xvfb-run','-a','-s','-screen 0 1280x960x24',*command]
    with (directory/'supervisor.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    write_status(directory,launcher_pid=process.pid)
    print(json.dumps(dict(pid=process.pid,output_directory=str(directory)),ensure_ascii=False))


if __name__=='__main__':
    main()
