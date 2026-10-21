"""Fill SCSP/A1--A7 Rotation cells from complete cached trial cohorts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'tro_revision'
CACHE=ROOT/'output/benchmarks/rotation_a1_a7_20261021_001433'
METHODS=('SCSP','DyWA','ContactSDF','STOCS-3D','Sampling','CF-MPC',
         'A1','A2','A3','A4','A5','A6','A7')
METRICS={'Success Rate':'success_rate','Execution Time (s)':'execution_time_s',
         'Planning Time (ms)':'planning_time_ms','Obj. Pos. Error (m)':'final_pos_error_m',
         'Obj. Quat. Error':'final_quat_error'}
HASHES={}


def load(path,jsonl=False):
    content=path.read_bytes()
    HASHES[str(path.relative_to(ROOT))]=hashlib.sha256(content).hexdigest()
    return [json.loads(s) for s in content.decode().splitlines() if s.strip()] if jsonl else json.loads(content)


def stats(values):
    return dict(n=len(values),mean=statistics.mean(values) if values else None,
                std=statistics.stdev(values) if len(values)>1 else None)


def summarize(rows):
    assert len(rows)==35 and len({(r['object'],r['seed']) for r in rows})==35
    assert len({r['object'] for r in rows})==7
    for obj in {r['object'] for r in rows}:
        assert sorted(r['seed'] for r in rows if r['object']==obj)==list(range(5))
    successful=[r for r in rows if r['success']]
    n=sum(r['planning_count'] for r in rows)
    total=math.fsum(r['planning_total_s'] for r in rows)
    squares=math.fsum(r['planning_sum_squares_s2'] for r in rows)
    return dict(trials=35,successes=len(successful),success_rate=len(successful)/35,
        execution_time_s=stats([r['execution_time_s'] for r in successful]),
        planning_time_ms=dict(n=n,mean=1000*total/n,
            std=1000*math.sqrt(max(0.,(squares-total*total/n)/(n-1)))),
        final_pos_error_m=stats([r['final_pos_err'] for r in rows]),
        final_quat_error=stats([r['final_quat_err'] for r in rows]))


def collect():
    rows=[]
    for p in sorted(ROOT.glob('output/table4_rotation_20261020/*/trials.jsonl')):
        for r in load(p,True):
            assert not r.get('aborted') and r['target_type']=='rotation'
            rows.append(dict(object=r['object'],seed=r['trial'],success=bool(r['success']),
                execution_time_s=r['completion_time_s'],planning_count=r['planning_wall_count'],
                planning_total_s=r['planning_wall_total_s'],
                planning_sum_squares_s2=r['planning_wall_sum_squares_s2'],
                final_pos_err=r['final_pos_err'],final_quat_err=r['final_quat_err'],source=str(p.relative_to(ROOT))))
    cohorts={'SCSP':rows}
    archived=load(CACHE/'statistics.json')
    assert archived['completed_trials']==archived['expected_trials']==245
    for i in range(1,8):
        method='a%d'%i;rows=[]
        for p in sorted((CACHE/method).glob('*/normalized_trials.json')):
            original=load(p.parent/'trials.jsonl',True)
            assert len(original)==5 and not any(r.get('aborted') for r in original)
            source_rows=load(p)
            assert {(r['object'],r['seed'],bool(r['success'])) for r in source_rows}=={
                (r['object'],r.get('seed',r['trial']),bool(r['success'])) for r in original}
            rows.extend(dict(r,source=str(p.relative_to(ROOT))) for r in source_rows)
        cohorts[method.upper()]=rows
    result={m:summarize(rs) for m,rs in cohorts.items()}
    for method,value in result.items():
        if method=='SCSP':continue
        prior=archived['methods'][method.lower()]['overall']
        assert prior['successes']==value['successes']
        for key in ('execution_time_s','planning_time_ms','final_pos_error_m','final_quat_error'):
            for field in ('n','mean','std'):
                a,b=value[key][field],prior[key][field]
                assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10))
    return result,cohorts


def latex(value):
    if not isinstance(value,dict):return f'{value:.2f}'
    if value['mean'] is None:return '--'
    return '$'+f"{value['mean']:.2f}"+(f"\\pm{value['std']:.2f}" if value['std'] is not None else '')+'$'


def apply(text,result):
    begin=text.index(r'\multirow{5}{*}{Rotation}')
    end=text.index(r'\multirow{5}{*}{Flipping}',begin)
    part=text[begin:end];lines=[];count=0
    for line in part.splitlines(keepends=True):
        if ' & ' not in line:lines.append(line);continue
        cells=line.rstrip('\n').rsplit(r' \\',1)[0].split(' & ')
        if len(cells)<2 or cells[1] not in METRICS:lines.append(line);continue
        assert len(cells)==15
        metric=METRICS[cells[1]]
        had_bold=any(r'\textbf{' in c for c in cells[2:])
        for method,value in result.items():cells[METHODS.index(method)+2]=latex(value[metric])
        if had_bold:
            for i in range(2,len(cells)):
                if cells[i].startswith(r'\textbf{') and cells[i].endswith('}'):
                    cells[i]=cells[i][8:-1]
            means={}
            for i,c in enumerate(cells[2:],2):
                if c=='--':continue
                value=float(re.search(r'\d+\.\d+',c).group())
                exponent=re.search(r'\\times10\^\{(-?\d+)\}',c)
                if exponent:value*=10**int(exponent.group(1))
                means[i]=value
            best=(max if metric=='success_rate' else min)(means.values())
            for i,value in means.items():
                if value==best:cells[i]=r'\textbf{'+cells[i]+'}'
        lines.append(' & '.join(cells)+r' \\'+'\n');count+=1
    assert count==5
    return text[:begin]+''.join(lines)+text[end:]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();result,cohorts=collect()
    snapshot=dict(task='Rotation',cohorts=result,trajectories=cohorts,
        definitions=dict(execution='Successful trajectories only, simulated seconds.',
            planning='Pooled recorded callback wall times over all cycles; milliseconds.',
            errors='Terminal errors over all 35 trials, including failures.',std='Sample SD, ddof=1.'),
        source_sha256=HASHES)
    (PAPER/'table_iv_rotation_sources.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    output=[]
    for method,r in result.items():
        record=dict(method=method,trials=r['trials'],successes=r['successes'],success_rate=r['success_rate'])
        for key in ('execution_time_s','planning_time_ms','final_pos_error_m','final_quat_error'):
            for field in ('n','mean','std'):record[key+'_'+field]=r[key][field]
        output.append(record)
    with (PAPER/'table_iv_rotation_sources.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    if args.apply:
        p=PAPER/'experiment.tex';p.write_text(apply(p.read_text(),result))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
