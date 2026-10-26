"""Freeze available PhysX CSVs and generate paired absolute-error Table IV."""
from pathlib import Path
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
MJ=ROOT/'outputs/collision_prediction_200_per_mode_20261026_job/data'
IG=ROOT/'outputs/collision_prediction_isaacgym_8000_20261026_final_job/data'
MODES=('sliding','sticking','separation','contact_establishment')
METRICS=('position_error_mm','orientation_error_deg')
GROUPS=[('mode',x) for x in MODES]+[('count',x) for x in ('0','1','2','>=3')]+[('geometry',x) for x in ('Point','Line','Surface','Complex')]
HEADINGS=['Sliding','Sticking','Separation','Onset',r'$n_c=0$',r'$n_c=1$',r'$n_c=2$',r'$n_c\geq3$','Point','Line','Surface','Complex']


def geometry(r):
    return {'point':'Point','vertex':'Point','edge':'Line','line':'Line','face':'Surface','coacd':'Complex'}[r['geometry'].removesuffix('_airborne')]


def read_rows(files, mj=False):
    rows=[]
    for file in sorted(files):
        for row in csv.DictReader(io.StringIO(file.read_text())):
            if row.get('method')!='scm' or row.get('selected')!='True' or row.get('status')!='ok':continue
            if mj and row['mode'] not in MODES:continue
            try:
                if not all(np.isfinite(float(row[m])) for m in METRICS):continue
            except (ValueError,TypeError,KeyError):continue
            rows.append(row)
    assert len({r['sample_id'] for r in rows})==len(rows)
    return rows


def subgroup(rows, dimension, group):
    if dimension=='mode':return [r for r in rows if r['mode']==group]
    if dimension=='count':return [r for r in rows if r['contact_group']==group]
    if dimension=='geometry':return [r for r in rows if geometry(r)==group]
    return rows


def number(x):
    if 0<abs(x)<.005:return f'{x:.3g}'
    return f'{x:.2f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-isaac-snapshot',action='store_true')
    args=parser.parse_args()
    snapshot=OUT/'isaac_snapshot'
    if args.refresh_isaac_snapshot or not snapshot.exists():
        snapshot.mkdir(exist_ok=True)
        for p in snapshot.glob('*.csv'):p.unlink()
        seals={}
        for p in sorted(IG.glob('*/predictions.csv')):
            data=p.read_bytes();target=snapshot/(p.parent.name+'.csv');target.write_bytes(data)
            seals[target.name]=hashlib.sha256(data).hexdigest()
        manifest=json.loads((IG/'manifest.json').read_text())
        (snapshot/'capture.json').write_text(json.dumps(dict(captured_at=datetime.now(timezone.utc).isoformat(),source=str(IG),source_status=manifest['status'],source_phase=manifest['phase'],csv_sha256=seals),indent=2)+'\n')
    rows={'MJWarp':read_rows(MJ.glob('*/predictions.csv'),mj=True),'IsaacGym':read_rows(snapshot.glob('*.csv'))}
    assert len(rows['MJWarp'])==8000 and rows['IsaacGym']
    cells={};statistics=[]
    for engine,data in rows.items():
        for dimension,group in GROUPS+[('all','all')]:
            subset=subgroup(data,dimension,group)
            for metric in METRICS:
                values=np.array([float(r[metric]) for r in subset])
                item=dict(comparison='SCM--'+engine,metric=metric,dimension=dimension,group=group,n_total=len(subset),n_valid=len(values),n_undefined=0,mean=float(values.mean()) if len(values) else None,std=float(values.std()) if len(values) else None)
                cells[engine,metric,dimension,group]=item;statistics.append(item)
    with (OUT/'statistics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(statistics[0]));w.writeheader();w.writerows(statistics)
    counts={engine:[len(subgroup(data,d,g)) for d,g in GROUPS] for engine,data in rows.items()}
    same=counts['MJWarp']==counts['IsaacGym']
    headers=[r'\shortstack{'+h+r'\\{\scriptsize $n='+str(n)+r'$}}' for h,n in zip(HEADINGS,counts['MJWarp'])]
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{Absolute prediction errors of SCM (mean $\pm$ std).}',r'\label{tab:scm_velocity_error}',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.15}',r'\resizebox{\textwidth}{!}{',r'\begin{tabular}{@{}ll*{12}{c}@{}}',r'\toprule',r'& & \multicolumn{4}{c}{Contact regime} & \multicolumn{4}{c}{Contact count} & \multicolumn{4}{c}{Contact geometry} \\',r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}','Comparison & Metric & '+' & '.join(headers)+r' \\',r'\midrule']
    for i,engine in enumerate(rows):
        if i:
            lines.append(r'\midrule')
            if not same:lines.append(r' & & '+' & '.join(r'{\scriptsize $n='+str(n)+'$}' for n in counts[engine])+r' \\')
            if not same:lines.append(r'\cmidrule(lr){3-14}')
        for index,(metric,label) in enumerate(zip(METRICS,('Position (mm)',r'Orientation ($^{\circ}$)'))):
            values=[]
            for dimension,group in GROUPS:
                item=cells[engine,metric,dimension,group]
                values.append('--' if item['mean'] is None else '$'+number(item['mean'])+r'\pm'+number(item['std'])+'$')
            prefix=r'\multirow{2}{*}{SCM--'+engine+'}' if index==0 else ''
            lines.append(prefix+' & '+label+' & '+' & '.join(values)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'}',r'\end{table*}']
    (OUT/'table_iv.tex').write_text('\n'.join(lines)+'\n')
    provenance=dict(sources={'MJWarp':str(MJ),'IsaacGym':str(IG)},isaac_snapshot=json.loads((snapshot/'capture.json').read_text()),n_queries={k:len(v) for k,v in rows.items()},n_mixed_isaac=sum(r['mode']=='mixed_transition' for r in rows['IsaacGym']),counts=counts,subgroup_counts_equal=same,metrics=METRICS,grouping='engine-native regime/count; all frozen queries in count/geometry, mixed omitted only from regime columns',geometry_definition='prescribed support configuration')
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance['n_queries']), 'counts equal:',same)
    for engine in rows:
        print(engine,[(m,cells[engine,m,'all','all']['mean'],cells[engine,m,'all','all']['std']) for m in METRICS])

if __name__=='__main__':main()
