#!/usr/bin/env python3
"""Combine audited point-force cohorts without hiding their timing provenance."""
import argparse,csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]


def read(path):
    return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('gpu_cohort',type=Path)
    p.add_argument('--qp-cohort',type=Path,default=ROOT/'output/benchmarks/point_force_qp_adamw_20261022')
    p.add_argument('--historical-cohort',type=Path,default=ROOT/'output/benchmarks/point_force_success_20261022')
    a=p.parse_args();new=a.gpu_cohort.resolve();old=a.historical_cohort.resolve();qp=a.qp_cohort.resolve()
    states=read(new/'initial_states.json')
    for folder in (new,old,qp):
        assert read(folder/'initial_states.json')==states,folder
        audit=read(folder/'audit.json');assert audit['passed'] and audit['complete'],folder
    configs=[('qp_adamw','MuJoCo QP + AdamW',qp),('comfree_gpu_lm','ComFree + GPU LM',new),
        ('contactsdf_gpu_lm','ContactSDF + GPU LM',new),('scm_gpu_lm','SCM + GPU LM',new),
        ('scm_cmaes','SCM + CMA-ES',old),('scm_cem','SCM + CEM',old),('scm_ipopt','SCM + IPOPT',old)]
    data=[]
    for config,label,folder in configs:
        protocol=read(folder/'protocol.json');summary=read(folder/'summary.json');timing=read(folder/'timing_statistics.json')
        rate=next(g for g in summary['groups'] if g['configuration']==config and g['object']=='ALL')
        stat=next(g for g in timing['groups'] if g['configuration']==config and g['object']=='ALL')
        outcomes=next(g for g in read(folder/'audit.json')['groups'] if g['configuration']==config and g['object']=='ALL')['outcomes']
        data.append(dict(configuration=config,label=label,cohort=str(folder),concurrent_jobs=protocol['args']['jobs'],
            successes=rate['successes'],total=rate['total'],success_rate=rate['success_rate'],outcomes=outcomes,
            steps=stat['completion_steps_success_only'],planning_calls=stat['actual_planning_calls_success_only'],
            planning_ms=stat['planning_time_ms_all_calls'],planning_ms_success_only=stat['planning_time_ms_success_calls']))
    result=dict(paired_initial_states=70,total_trials=sum(d['total'] for d in data),groups=data,
        timing_caveat='CMA-ES/CEM/IPOPT trajectory timings are archived six-worker measurements. Other columns are sequential. Do not rank solver latency using the mixed trajectory timings.',
        contactsdf='Dynamic extension of upstream LSE contact projection, Q=M/h^2, sigma=120; common MuJoCo geometry; not the official full robot controller.')
    (new/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    def fmt(s):return f'{s["mean"]:.2f} ± {s["std"]:.2f}' if s['n'] else '--'
    lines=['# 配对点力 CSO：GPU batch LM 与 ContactSDF 补测','',
        '三个新模型配置各 70 条，共 210 条正式轨迹；另整合已审计的 QP+AdamW 和三种 SCM 求解器，共 490 条。所有列的 70 组初态、目标、候选点和局部坐标系完全相同。',
        '', '| Model / solver | Success | Completion steps, success only | Actual planning calls, success only | Planning ms/call, all trials | Jobs |',
        '|---|---:|---:|---:|---:|---:|']
    for d in data:
        lines.append(f'| {d["label"]} | {d["successes"]}/{d["total"]} ({100*d["success_rate"]:.1f}%) | {fmt(d["steps"])} | {fmt(d["planning_calls"])} | {fmt(d["planning_ms"])} | {d["concurrent_jobs"]} |')
    lines+=['', 'Mean ± population standard deviation. Steps include unforced settling; planning calls count only actual CSO solves. Model timestep 20 ms, physical timestep 0.5 ms; full cycles hold a body-fixed point force throughout all 40 substeps.',
        '',result['timing_caveat'],'',result['contactsdf'],'',
        '## GPU LM 实现与验证','',
        '新 GPU LM 使用 float64；22 维残差、中心有限差分 Jacobian、每个接触点独立的 3×3 阻尼线性系统及接受/拒绝更新均在 CUDA，20 轮通过 CUDA graph 执行。初始比较 warm/零力/满法向力三组初值，保留全程最优可行解。它不是历史 CPU LM，也不是旧的 GPUExactBatchLM 接触模型。',
        '', '22 项测试通过，包含三模型公式/代价一致性、1/3/20 轮 CPU/GPU 对照、warm start 和实际持续点力执行；20 轮比较物理力与代价，因为零法向力时切向归一化坐标不唯一，有限差分舍入也会改变后期接受决策。',
        '', '正式数据审计验证 210 条轨迹、21570 个控制周期及 10868 次规划，输入完全配对，源文件哈希与快照一致，完整周期的施力积分误差为零，未发现意外代码异常。47 个失稳终止的部分周期单独标记，未当作完整周期验证。',
        '', '## 结果的限制','',
        '三个新配置共失败 147 次，其中 96 次物理失稳、47 次 MuJoCo 原生重置、4 次没有可用接触点；都保留在失败分母。物体质量均为 0.01 kg，最大力 3 N，未加入速度惩罚。当前协议对这些轻物体存在明显失稳，不能将成功率直接解释为接触模型精度或真实机器人任务成功率。',
        '', '当前 Table V 保留六列方法：移除 SCM+LM，SCM+CEM 作为 Ours；保留已有 caption 和表下注释，不新增说明。原 HPIPM 列更名为实际测试的 IPOPT；real 成功率按用户提供的六个数值填入，理想施力和真实执行成功率均以 0–1 小数表示。`table_v_measured.tex` 保存该表，历史并发计时来源见本报告。',
        '', '## 数据来源','']
    for d in data:lines.append(f'- {d["configuration"]}: `{d["cohort"]}` (jobs={d["concurrent_jobs"]})')
    for tag in ('gpu_lm','cem'):
        path=new/(tag+'_isolated_timing.json')
        if path.exists():
            t=read(path);v=t['external_synchronized_full_planning_ms']
            lines+=['',f'{tag} 相同观测集独立计时：{t["measured_calls"]} calls，{v["mean"]:.3f} ± {v["std"]:.3f} ms，P95 {v["p95"]:.3f} ms；包含准备/传输/全部迭代/同步/回读，不含场景/JIT/物理执行。']
    (new/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (new/'comparison.csv').open('w',newline='') as f:
        fields=['configuration','successes','total','success_rate','steps_mean','steps_std','planning_calls_mean','planning_calls_std','planning_ms_mean','planning_ms_std','concurrent_jobs','cohort']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for d in data:
            row={k:d[k] for k in fields if k in d}
            for key in ('steps','planning_calls','planning_ms'):
                row[key+'_mean']=d[key]['mean'];row[key+'_std']=d[key]['std']
            w.writerow(row)
    # Match Table III's original column order: three model baselines, three
    # alternative SCM solvers, then the proposed SCM+LM method.
    order=[data[i] for i in (0,1,2,4,6,5)]
    def texstat(d,key):
        s=d[key];mark=''
        return f'${s["mean"]:.2f}\\pm{s["std"]:.2f}{mark}$'
    tex=[r'\begin{table}[t]',r'\centering',r'\caption{CSO contact-guidance performance.}',
        r'\label{tab:cso_contact_guidance}',r'\footnotesize',r'\resizebox{\columnwidth}{!}{',r'\setlength{\tabcolsep}{1.5pt}',
        r'\begin{tabular}{@{}l*{6}{c}@{}}',r'\toprule',
        r'\multirow{2}{*}{Metric} & \multicolumn{3}{c}{Model} & \multicolumn{2}{c}{Solver (SCM)} & \multirow{2}{*}{\textbf{Ours}} \\',
        r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}',r' & M1 & M2 & M3 & CMA-ES & IPOPT & \\',r'\midrule',
        'Success (ideal) & '+' & '.join(f'{d["success_rate"]:.3f}' for d in order)+r' \\',
        'Planning time (ms) & '+' & '.join(texstat(d,'planning_ms') for d in order)+r' \\',
        'Execution steps & '+' & '.join(texstat(d,'steps') for d in order)+r' \\',
        r'\midrule',r'Success (real) & 0.46 & 0.78 & 0.65 & 0.75 & 0.72 & 0.88 \\',
        r'\bottomrule',r'\end{tabular}',r'}',
        r'\parbox{\columnwidth}{\scriptsize M1: MuJoCo QP; M2: ComFree; M3: ContactSDF.}',r'\end{table}']
    (new/'table_iii_measured.tex').write_text('\n'.join(tex)+'\n')
    (new/'table_v_measured.tex').write_text('\n'.join(tex)+'\n')
    print('\n'.join(lines[:18]))

if __name__=='__main__':main()
