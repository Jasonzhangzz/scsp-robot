"""Run the fixed A6 Rotation cohort and fill only its five table cells."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import traceback

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
MANIFEST = OUT / 'manifest.json'


def save(data):
    temporary = MANIFEST.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    temporary.replace(MANIFEST)


def metric(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values)>1 else None)


def main():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['status'] == 'prepared', 'Use a new cohort for another run'
    source = Path(manifest['source_root'])
    python = '/home/zz/miniconda3/envs/acc/bin/python'
    env = os.environ.copy()
    env['PATH'] = str(Path(python).parent)+os.pathsep+env.get('PATH','')
    env['LD_LIBRARY_PATH'] = '/home/zz/miniconda3/envs/acc/lib:'+env.get('LD_LIBRARY_PATH','')
    cores, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
        key = ((topology/'physical_package_id').read_text(), (topology/'core_id').read_text())
        if key not in seen:
            cores.append(cpu)
            seen.add(key)
    assert len(cores)>=14
    manifest.update(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    worker_cpus=[cores[2*i:2*i+2] for i in range(7)])
    save(manifest)
    print('A6 Rotation: 7 objects x 3 seeds; async planner; realtime Isaac physics.', flush=True)

    def worker(item):
        index, obj = item
        directory = OUT/'rotation'/obj
        directory.mkdir(parents=True)
        path = directory/'trials.jsonl'
        command = [python, '-u', str(source/'examples/mpc/franka/ik2/run_ablation.py'),
            '--a6','--rotation','--obj',obj,'--headless','--async-planner','--headless-realtime',
            '--physics-backend','isaac','--solver-backend','acados','--trial-start','0',
            '--trial_num','3','--max_rollout_length','2500','--sample_num','70','--compute-threads','1',
            '--ranking-metrics-jsonl',str(path),'--rollout-summary-json',str(directory/'rollout_summary.json')]
        affinity = ','.join(map(str, cores[2*index:2*index+2]))
        (directory/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (directory/'run.log').open('w') as log:
            subprocess.run(['taskset','-c',affinity,*command],cwd=source,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert sorted(r['seed'] for r in rows)==[0,1,2]
        for row in rows:
            assert row['object']==obj and not row['aborted']
            assert row['ablation_mode']=='a6' and row['cpo_variant']=='direct'
            assert row['async_planner'] and row['headless_realtime']
            assert row['physics_backend']=='isaac' and row['target_type']=='rotation'
            assert row['planner_solver_backend']=='acados' and row['cso_backend'] is None
            assert row['steps']==row['planning_wall_count'] and row['steps']<=2500
            if row['success']: assert row['completion_time_s'] is not None
        print('%s: %d/3 successes'%(obj,sum(r['success'] for r in rows)),flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=7) as executor:
        groups = list(executor.map(worker,enumerate(manifest['objects'])))
    rows = [row for group in groups for row in group]
    assert len(rows)==21 and len({(r['object'],r['seed']) for r in rows})==21
    for file,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((source/file).read_bytes()).hexdigest()==digest,file
    successful = [r for r in rows if r['success']]
    cycles = sum(r['planning_wall_count'] for r in rows)
    result = dict(trials=21,successes=len(successful),success_rate=len(successful)/21,
        execution_time_s=metric([r['completion_time_s'] for r in successful]),
        planning_time_s=sum(r['planning_wall_total_s'] for r in rows)/cycles,
        planning_cycles=cycles,final_pos_error_m=metric([r['final_pos_err'] for r in rows]),
        final_quat_error=metric([r['final_quat_err'] for r in rows]),
        async_planner=True,headless_realtime=True,
        stable_flip_successes=sum(r['stable_flip_success'] for r in rows),
        mpc_failures=sum(r['mpc_failures'] for r in rows),
        mpc_fallbacks=sum(r['mpc_fallbacks'] for r in rows))
    (OUT/'trials.json').write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'statistics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

    paper = ROOT/'tro_revision'
    file = paper/'experiment.tex'
    text = file.read_text()
    start = text.index('\\label{tab:all_data}',text.index('% End of backup.'))
    end = text.index('\\end{table*}',start)
    segment = text[start:end]
    keys = ('success_rate','execution_time_s','planning_time_s','final_pos_error_m','final_quat_error')
    output, active, index = [], False, 0
    for line in segment.splitlines(keepends=True):
        if r'\multirow{5}{*}{Rotation}' in line: active=True
        if active and ' & ' in line and index<5:
            cells=line.rstrip('\n').split(' & ')
            assert len(cells)==21
            value=result[keys[index]]
            value=value['mean'] if isinstance(value,dict) else value
            cells[19]='--' if value is None else format(value,'.2f' if index<2 else '.3f')
            line=' & '.join(cells)+'\n'
            index+=1
        output.append(line)
    assert index==5
    segment=''.join(output)+'% A6 Rotation: output/benchmarks/a6_rotation_async_7x3_20261015/statistics.json; 21 asynchronous trials, seeds 0,1,2, 2500 planner-cycle limit.\n'
    text=text[:start]+segment+text[end:]
    text=text.replace('A6 Flipping uses 15 interim trials','A6 uses 21 Rotation and 15 interim Flipping trials')
    file.write_text(text)
    manifest.update(status='compiling_pdf',completed_trials=21)
    save(manifest)
    pdf=paper/'root.pdf'
    if pdf.exists(): shutil.copy2(pdf,OUT/'root.before_rotation.pdf')
    for name,command in [('pdflatex_1',['pdflatex','-interaction=nonstopmode','-halt-on-error','root.tex']),
                         ('bibtex',['bibtex','root']),
                         ('pdflatex_2',['pdflatex','-interaction=nonstopmode','-halt-on-error','root.tex']),
                         ('pdflatex_3',['pdflatex','-interaction=nonstopmode','-halt-on-error','root.tex'])]:
        with (OUT/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=paper,stdout=log,stderr=subprocess.STDOUT,check=True)
    final_log=(paper/'root.log').read_text(errors='replace')
    assert 'There were undefined' not in final_log
    assert pdf.stat().st_size>0
    shutil.copy2(paper/'root.log',OUT/'latex_final.log')
    for extension in ('aux','log','bbl','blg','out','toc','lof','lot','fls','fdb_latexmk','synctex.gz'):
        (paper/('root.'+extension)).unlink(missing_ok=True)
    manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
    save(manifest)
    print('Complete: A6 Rotation table filled; historical Flipping values preserved; root.pdf compiled and intermediates removed.',flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        manifest=json.loads(MANIFEST.read_text())
        manifest.update(status='failed',error=str(error))
        save(manifest)
        traceback.print_exc()
        raise
