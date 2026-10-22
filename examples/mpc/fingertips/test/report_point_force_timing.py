#!/usr/bin/env python3
"""Aggregate recorded CSO times and successful completion steps without reruns."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from planning.rollout_timing import distribution
from planning.point_force_benchmark import write_json


def format_stat(stat,digits=2):
    return '--' if not stat['n'] else f'{stat["mean"]:.{digits}f} ± {stat["std"]:.{digits}f}'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    directory=args.directory.resolve();protocol=json.loads((directory/'protocol.json').read_text())
    groups={};trials=[];intervals=set();total_calls=0
    for config in protocol['configs']:
        for obj in protocol['objects']:
            for seed in protocol['seeds']:
                path=directory/config/f'{obj}_{seed:03d}.json.gz'
                with gzip.open(path,'rt') as stream:payload=json.load(stream)
                trial=payload['metadata']['trial_summary'];rows=payload['rows']
                times=[r['planning_time_ms'] for r in payload['planning_rounds'] if r.get('planning_time_ms') is not None]
                timing=distribution(times)
                assert timing['n']==trial['planning_time_ms']['n'],path
                if times:
                    np.testing.assert_allclose([timing['mean'],timing['std']],
                        [trial['planning_time_ms']['mean'],trial['planning_time_ms']['std']],atol=1e-9)
                total_calls+=len(times)
                for row in rows:intervals.add((row['h_model'],row['h_exec']))
                if trial['success']:
                    assert trial['steps']==trial['completion_steps']==len(rows),path
                    np.testing.assert_allclose(trial['simulation_time_s'],trial['steps']*protocol['policy_dt'],atol=1e-9)
                item=dict(configuration=config,object=obj,seed=seed,success=trial['success'],
                    completion_steps=trial['completion_steps'],planning_rounds=trial['planning_rounds'],
                    total_planning_time_ms=sum(times),planning_time_ms=timing,
                    simulation_time_s=trial['simulation_time_s'],execution_time_s=trial['execution_time_s'])
                trials.append(item)
                for key in ((config,obj),(config,'ALL')):
                    group=groups.setdefault(key,dict(trials=[],times=[],success_times=[]))
                    group['trials'].append(item);group['times'].extend(times)
                    if item['success']:group['success_times'].extend(times)
    stats=[]
    for (config,obj),g in groups.items():
        good=[t for t in g['trials'] if t['success']]
        stats.append(dict(configuration=config,object=obj,total_trials=len(g['trials']),successful_trials=len(good),
            completion_steps_success_only=distribution([t['completion_steps'] for t in good]),
            actual_planning_calls_success_only=distribution([t['planning_rounds'] for t in good]),
            planning_time_ms_all_calls=distribution(g['times']),
            planning_time_ms_success_calls=distribution(g['success_times']),
            total_planning_time_s_success_only=distribution([t['total_planning_time_ms']/1000 for t in good]),
            planning_time_ms_trial_means_all_trials=distribution([t['planning_time_ms']['mean'] for t in g['trials'] if t['planning_time_ms']['n']]),
            simulation_time_s_success_only=distribution([t['simulation_time_s'] for t in good]),
            execution_wall_time_s_success_only=distribution([t['execution_time_s'] for t in good])))
    report=dict(total_trials=len(trials),recorded_planning_calls=total_calls,groups=stats,trials=trials,
        planning_timestep_s=protocol['policy_dt'],physical_timestep_s=protocol['policy_dt']/protocol['physics_substeps'],
        observed_prediction_execution_intervals_s=sorted(intervals),concurrent_jobs=protocol['args']['jobs'],
        definitions=dict(completion_steps='All completed 20 ms control cycles up to success, including unforced settling cycles.',
            actual_planning_calls='Real CSO calls; unforced settling cycles that skip planning do not count.',
            planning_time_ms='Wall time per actual CSO contact-point/force planning call; physical stepping excluded. Concurrent run, not isolated inference latency.',
            all_calls='Pooled calls over all 70 trials per configuration, including failures; longer trajectories receive more weight.',
            success_calls='Pooled calls only in successful trajectories.',
            standard_deviation='Population standard deviation, ddof=0.',
            planning_timestep='Fixed model/policy time increment, distinct from wall-clock planning duration.'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(directory/'timing_statistics.json',report)
    rows=[]
    for stat in stats:
        row={k:stat[k] for k in ('configuration','object','total_trials','successful_trials')}
        for key,value in stat.items():
            if isinstance(value,dict):
                for field,number in value.items():row[f'{key}_{field}']=number
        rows.append(row)
    with (directory/'timing_statistics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# 规划耗时和成功完成步数',
        '',f'固定 planning timestep：{report["planning_timestep_s"]} s；物理子步：{report["physical_timestep_s"]} s。',
        f'共 {len(trials)} 条轨迹、{total_calls} 次实际规划调用；下表为 mean ± 总体 std。',
        '', '| Configuration | Success n | Completion steps (success only) | Actual planning calls (success only) | Planning ms/call (all trials) | Planning ms/call (success only) |',
        '|---|---:|---:|---:|---:|---:|']
    for config in protocol['configs']:
        s=next(s for s in stats if s['configuration']==config and s['object']=='ALL')
        lines.append(f'| {config} | {s["successful_trials"]} | '+
            ' | '.join(format_stat(s[k]) for k in ('completion_steps_success_only','actual_planning_calls_success_only',
                'planning_time_ms_all_calls','planning_time_ms_success_calls'))+' |')
    lines+=['','完成步数包含无外力停稳阶段；实际规划次数仅统计确实调用 CSO 的轮次。两者不能互换，也不能用完成步数乘平均规划耗时计算总规划耗时。',
        '',(f'本轮 {report["concurrent_jobs"]} 个任务并发运行，墙钟耗时受 CPU/GPU 竞争影响，不能直接用于论文的公平推理耗时比较。'
            if report['concurrent_jobs']>1 else
            '本轮按单任务顺序执行，未安排其他本次基准任务并发；跨方法比较仍需一致的实现设备、精度和预算。'),
        '', 'JSON/CSV 同时保存逐物体统计、成功轨迹的总规划耗时、仿真完成时间、墙钟完成时间及 P95。未重新运行实验，也未改变成功/失败判定。']
    (directory/'timing_statistics.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
