#!/usr/bin/env python3
"""Paired fixed/online SCM Isaac MPC trials; no environment is created here.

Run with the acc Python. GUI is enabled unless --headless is passed. A synchronous
planner is the default for a fair simulation-time comparison; --async-planner
checks the production latest-only IPC path. --analyze-only reuses existing data.
"""
import argparse
import ast
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]


def prediction_errors(data, key):
    pred,truth=data[key],data['q_plus']
    p=np.linalg.norm(pred[:,:3]-truth[:,:3],axis=1)*1000
    a=pred[:,3:7]/np.linalg.norm(pred[:,3:7],axis=1,keepdims=True)
    b=truth[:,3:7]/np.linalg.norm(truth[:,3:7],axis=1,keepdims=True)
    angle=2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=1)),0,1))
    return p,angle


def analyze(out):
    rows=[]; curves=[]; update_times=[]
    for mode in ('fixed','online'):
        log=out/(mode+'.log')
        if not log.exists(): continue
        summaries=[]
        for line in log.read_text().splitlines():
            # Native-library stderr can appear between print's prefix and dict.
            if "{'trial':" in line and "'final_pos_err':" in line and "'mode': 'mpc_planner'" in line:
                start=line.index("{'trial':");end=line.rfind('}')+1
                summaries.append(ast.literal_eval(line[start:end]))
        for summary in summaries:
            trial=int(summary['trial'])
            paths=list((out/mode).glob('*_trial_%03d/transitions.npz'%trial))
            row={**summary, "mode":mode}
            if paths:
                path=paths[0]
                with np.load(path,allow_pickle=False) as data:
                    p,a=prediction_errors(data,'prediction')
                    bp,ba=prediction_errors(data,'prediction_base')
                    adapted=np.any(np.abs(data['theta_at_prediction'])>1e-10,axis=1)
                    row.update(contact_samples=len(p),prediction_position_mm_mean=float(p.mean()),
                        prediction_position_mm_std=float(p.std()), prediction_angle_rad_mean=float(a.mean()),
                        prediction_angle_rad_std=float(a.std()), same_states_fixed_position_mm_mean=float(bp.mean()),
                        same_states_fixed_angle_rad_mean=float(ba.mean()), learned_steps=int(adapted.sum()),
                        final_scales=np.exp(data['theta']).tolist())
                    if adapted.any():
                        row.update(post_warmup_position_mm=float(p[adapted].mean()),
                            post_warmup_fixed_position_mm=float(bp[adapted].mean()),
                            post_warmup_angle_rad=float(a[adapted].mean()),
                            post_warmup_fixed_angle_rad=float(ba[adapted].mean()))
                    curves.append((mode,trial,data['id'].copy(),p,a,bp,ba))
                events=[]
                lp=path.with_name('learning.jsonl')
                if lp.exists():
                    # A repeated run may append events; retain the final event
                    # at each sample count rather than counting aborted runs.
                    by_count={}
                    for line in lp.read_text().splitlines():
                        e=json.loads(line);by_count[e['samples']]=e
                    events=list(by_count.values())
                if events:
                    times=np.array([e['update_ms'] for e in events])
                    if mode=='online': update_times.extend(times.tolist())
                    row.update(accepted_updates=sum(e['accepted'] for e in events),
                               update_ms_mean=float(times.mean()),update_ms_std=float(times.std()),
                               update_ms_max=float(times.max()))
            rows.append(row)
    aggregate={}
    for mode in ('fixed','online'):
        group=[r for r in rows if r['mode']==mode]
        if not group: continue
        metrics={}
        for name in ('final_pos_err','final_quat_err','prediction_position_mm_mean',
                     'prediction_angle_rad_mean','choose_dt_mean','update_ms_mean'):
            values=np.array([r[name] for r in group if name in r])
            if len(values): metrics[name]=dict(mean=float(values.mean()),std=float(values.std()),n=len(values))
        completed=np.array([r['steps'] for r in group if r['success']],dtype=float)
        if len(completed): metrics['completion_steps_success_only']=dict(mean=float(completed.mean()),std=float(completed.std()),n=len(completed))
        capped=np.array([r['steps'] for r in group],dtype=float)
        metrics['steps_with_budget_censoring']=dict(mean=float(capped.mean()),std=float(capped.std()),n=len(capped))
        aggregate[mode]=dict(successes=sum(r['success'] for r in group),trials=len(group),metrics=metrics)
    report=dict(trials=rows,aggregate=aggregate,
        notes=['Quaternion prediction error is 2*acos(abs(dot(q_pred,q_real))) in radians.',
               'Task final_quat_err is the existing 1-dot^2 success metric, not an angle.',
               'Prediction is scored before updating with that transition.',
               'Same-state fixed predictions use zero calibration on identical executed transitions.',
               'Reported aggregate std is population std across trial means; these are small descriptive trials.',
               'Stable contact windows are selected; these numbers do not cover contact entry/exit or all robot steps.',
               'Completion steps are scored separately on successful trials. Failed trials reaching the budget are censored, not completed.'])
    fixed_by_trial={r['trial']:r for r in rows if r['mode']=='fixed'}
    report['paired_completion_steps']=[dict(trial=r['trial'],fixed_steps=fixed_by_trial[r['trial']]['steps'],
        online_steps=r['steps'],change_steps=r['steps']-fixed_by_trial[r['trial']]['steps'])
        for r in rows if r['mode']=='online' and r['trial'] in fixed_by_trial
        and r['success'] and fixed_by_trial[r['trial']]['success']]
    same_rows=[r for r in rows if r['mode']=='online' and 'prediction_position_mm_mean' in r]
    if same_rows:
        bp=np.array([r['same_states_fixed_position_mm_mean'] for r in same_rows])
        p=np.array([r['prediction_position_mm_mean'] for r in same_rows])
        ba=np.array([r['same_states_fixed_angle_rad_mean'] for r in same_rows])
        a=np.array([r['prediction_angle_rad_mean'] for r in same_rows])
        stats=lambda v:dict(mean=float(v.mean()),std=float(v.std()))
        report['same_state_prediction']=dict(position_mm_fixed=stats(bp),position_mm_online=stats(p),
            orientation_rad_fixed=stats(ba),orientation_rad_online=stats(a),
            position_reduction_percent=float((1-p.mean()/bp.mean())*100),
            orientation_reduction_percent=float((1-a.mean()/ba.mean())*100))
        if update_times:
            report['same_state_prediction'].update(update_ms_pooled_mean=float(np.mean(update_times)),
                update_ms_pooled_std=float(np.std(update_times)),update_ms_max=float(np.max(update_times)),
                updates=len(update_times))
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    if rows:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (out/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    if curves:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
        for mode,trial,ids,p,a,bp,ba in curves:
            if mode!='online':continue
            def smooth(x):
                width=min(15,len(x));return np.convolve(x,np.ones(width)/width,mode='valid')
            width=min(15,len(p));axis_ids=ids[width-1:]
            for ax,actual,base in ((axes[0],p,bp),(axes[1],a,ba)):
                line,=ax.plot(axis_ids,smooth(actual),label='online trial %d'%trial)
                ax.plot(axis_ids,smooth(base),'--',color=line.get_color(),alpha=.6,label='fixed on same states %d'%trial)
        axes[0].set_ylabel('Position prediction error [mm]')
        axes[1].set_ylabel('Orientation prediction error [rad]');axes[1].set_xlabel('Executed physics window')
        axes[0].legend(ncol=2,fontsize=8);fig.tight_layout()
        fig.savefig(out/'prediction_errors.png',dpi=180);fig.savefig(out/'prediction_errors.pdf');plt.close(fig)
    print(json.dumps(aggregate,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--obj',default='elephant')
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--trial-start',type=int,default=0)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--async-planner',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=None)
    parser.add_argument('--analyze-only',action='store_true')
    args,extra=parser.parse_known_args()
    out=args.output_dir or ROOT/'examples/mpc/franka/scm/outputs'/('online_scm_'+args.obj+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    if not args.analyze_only:
        commands={}
        for mode in ('fixed','online'):
            cmd=[sys.executable,str(ROOT/'examples/mpc/franka/ik2/test_mpc_isaac.py'),
                 '--obj',args.obj,'--trial_num',str(args.trials),'--max_rollout_length',str(args.steps),
                 '--trial-start',str(args.trial_start),'--scm-learning-dir',str(out/mode),
                 '--async-planner' if args.async_planner else '--sync-planner',
                 '--adaptive' if mode=='online' else '--no-adaptive','--scm-record-data']
            if args.headless:cmd.append('--headless')
            cmd+=extra;commands[mode]=cmd
            print('Running',mode,'log:',out/(mode+'.log'),flush=True)
            with (out/(mode+'.log')).open('w') as log:
                code=subprocess.call(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            if code:raise RuntimeError('%s exited %d; inspect %s'%(mode,code,out/(mode+'.log')))
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    analyze(out)

if __name__=='__main__':main()
