import csv,gzip,json,hashlib,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tro_revision/table_iv_data'
sources={'MJWarp':ROOT/'outputs/collision_prediction_200_per_mode_20261026_job/data', 'IsaacGym':ROOT/'outputs/collision_prediction_isaacgym_8000_20261026_final_job/data'}
records=[]
for engine,source in sources.items():
 for folder in sorted(source.iterdir()):
  if not (folder/'predictions.csv').exists():continue
  rows={r['sample_id']:r for r in csv.DictReader((folder/'predictions.csv').open()) if r['method']=='scm' and r['selected']=='True' and r['status']=='ok' and r['mode']=='contact_establishment'}
  found=set()
  with gzip.open(folder/'reference_contacts.jsonl.gz','rt') as f:
   for line in f:
    sid=line.split('"sample_id":',1)[-1].split(',',1)[0].strip().strip('"')
    # All records use sample_id as their first JSON field.
    if sid not in rows:continue
    r=json.loads(line)
    meta=rows[sid]
    snaps=r['snapshots']
    actual=[s for s in snaps if float(s['time']) < .02-1e-9]
    assert len(actual)==10
    sums=np.array([[sum(max(0.,float(v[0])) for p,v in zip(s['pair'],s['force']) if p==pair) for pair in (0,1)] for s in actual])
    active=np.array([[any(float(v[0])>1e-4 for p,v in zip(s['pair'],s['force']) if p==pair) for pair in (0,1)] for s in snaps])
    changes=np.diff(active.astype(int),axis=0)>0
    base=dict(engine=engine,object=folder.name,sample_id=sid,force_N=float(meta['force_N']),ball_initial_speed_m_s=float(meta['ball_initial_speed_m_s']),position_error_mm=float(meta['position_error_mm']),orientation_error_deg=float(meta['orientation_error_deg']))
    for pair,label in enumerate(['ground','ball']):
     base[label+'_onset']=bool(changes[:,pair].any())
     base[label+'_onset_in_0_20ms']=bool(changes[:9,pair].any())
     base[label+'_onset_endpoint_diagnostic']=bool(changes[9,pair])
     base[label+'_peak_summed_normal_force_N']=float(sums[:,pair].max())
     base[label+'_normal_impulse_Ns']=float(sums[:,pair].sum()*.002)
    records.append(base)
    found.add(sid)
  assert found==rows.keys(),(engine,folder.name,len(found),len(rows))
 print(engine,'onset records',sum(r['engine']==engine for r in records),flush=True)
with (OUT/'onset_loads.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def stat(v):
 a=np.asarray(v,float)
 return dict(n=len(a),min=float(a.min()),max=float(a.max()),median=float(np.median(a)),p95=float(np.percentile(a,95)),mean=float(a.mean())) if len(a) else dict(n=0)
summary={'definition':'Onset uses existing Table IV labels, including endpoint diagnostic. Loads integrate only ten actual 2ms reference solves in [0,20]ms. Peak is maximum over substeps of sum of nonnegative normal forces at points of the specified interaction pair. Impulse is scalar normal-force sum times dt, not net vector impulse. Endpoint diagnostic loads are excluded.','engines':{}}
for engine in sources:
 rs=[r for r in records if r['engine']==engine]
 item={'n':len(rs),'external_force_N':stat([r['force_N'] for r in rs]),'ball_initial_speed_m_s':stat([r['ball_initial_speed_m_s'] for r in rs])}
 for label in ['ground','ball']:
  onset=[r for r in rs if r[label+'_onset']]
  inside=[r for r in onset if r[label+'_onset_in_0_20ms']]
  item[label]={'onset_count':len(onset),'onset_in_0_20ms_count':len(inside),'endpoint_diagnostic_only_count':sum(r[label+'_onset_endpoint_diagnostic'] and not r[label+'_onset_in_0_20ms'] for r in onset)}
  for k in ['peak_summed_normal_force_N','normal_impulse_Ns']:
   item[label][k]=stat([r[label+'_'+k] for r in onset])
   item[label][k+'_onset_in_0_20ms']=stat([r[label+'_'+k] for r in inside])
 summary['engines'][engine]=item
(OUT/'onset_load_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

# Rank-based descriptive error strata, with actual-window onsets only.
strata=[]
for engine in sources:
 for pair in ['ground','ball']:
  group=sorted((r for r in records if r['engine']==engine and r[pair+'_onset_in_0_20ms']),key=lambda r:r[pair+'_normal_impulse_Ns'])
  for k,indices in enumerate(np.array_split(np.arange(len(group)),4)):
   subset=[group[i] for i in indices]
   if not subset:continue
   impulses=[r[pair+'_normal_impulse_Ns']*1000 for r in subset]
   item=dict(engine=engine,pair=pair,impulse_quartile=k+1,n=len(subset),impulse_min_mNs=min(impulses),impulse_max_mNs=max(impulses))
   for metric in ['position_error_mm','orientation_error_deg']:
    values=np.asarray([r[metric] for r in subset])
    item[metric+'_mean']=float(values.mean())
    item[metric+'_std']=float(values.std())
   strata.append(item)
with (OUT/'onset_load_strata.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(strata[0]))
 w.writeheader()
 w.writerows(strata)
