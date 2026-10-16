"""Common simulation-time pose errors for the eight closed-loop rollouts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

LABELS={'fa_cost':'我们的方法：FA cost','comfree_model':'我们的方法：仅换 ComFree 模型','dexforge':'DexForge','current':'我们当前的方法'}


def reference(poses,times):
    frame=np.clip(times*30,0,len(poses)-1);a=np.floor(frame).astype(int);b=np.minimum(a+1,len(poses)-1);alpha=(frame-a)[:,None]
    result=(1-alpha)*poses[a]+alpha*poses[b]
    qa=poses[a,3:];qb=poses[b,3:]*np.where(np.sum(qa*poses[b,3:],axis=1,keepdims=True)<0,-1,1)
    result[:,3:]=(1-alpha)*qa+alpha*qb
    result[:,3:]/=np.linalg.norm(result[:,3:],axis=1,keepdims=True)
    return result


def stats(position,angle,mask):
    p=position[mask];a=angle[mask]
    if not len(p):return None
    return dict(samples=len(p),position_mean_mm=float(p.mean()),position_rmse_mm=float(np.sqrt(np.mean(p*p))),
        position_p95_mm=float(np.quantile(p,.95)),position_max_mm=float(p.max()),
        rotation_mean_deg=float(a.mean()),rotation_rmse_deg=float(np.sqrt(np.mean(a*a))),
        rotation_p95_deg=float(np.quantile(a,.95)),rotation_max_deg=float(a.max()),
        last_position_mm=float(p[-1]),last_rotation_deg=float(a[-1]))


def report(root,partial=False):
    root=Path(root);raw=np.load(root/'contactaware/object_pose_7.npy');poses=raw[:,[4,5,6,3,0,1,2]]
    results={};curves={};duration=(len(poses)-1)/30
    for executor in ['mujoco','comfree']:
        for method in LABELS:
            case=root/executor/method;key=f'{executor}/{method}'
            if method=='dexforge':
                path=case/'rollout.npz'
                if not path.exists():continue
                a=np.load(path);q=a['qpos_traj'];actual=q[3::3,-7:];times=np.arange(1,len(actual)+1)*.03
                audit=json.loads((case/'audit.json').read_text());audit['initial_qpos']=q[0].tolist()
            else:
                path=case/'full/diagnostics.jsonl'
                if not path.exists():continue
                records=[json.loads(x) for x in path.read_text().splitlines()]
                if not records:continue
                actual=np.array([r['object_pose'] for r in records]);times=(np.arange(len(actual))+1)*.03
                qprior=np.load(case/'wuji_q.npy');common=np.load(root/'contactaware/wuji_right_qpos.npy')
                if not np.array_equal(qprior,common):raise RuntimeError(f'{key}: CA changed')
                if any(r['lambda_enabled'] for r in records):raise RuntimeError(f'{key}: Lambda was enabled')
                audit=dict(closed_loop=True,CA_array_exactly_common=True,lambda_optimizer=False,
                    failures=sum('acados status' in r['mpc']['status'] for r in records),
                    selected={source:sum(r['mpc']['selected_control']==source for r in records) for source in sorted(set(r['mpc']['selected_control'] for r in records))},
                    force_gate=False,force_closure_ever=any(r['actual_force_closure']['force_closure'] for r in records),
                    FA_loss_value_validation_max_relative=max(r['mpc'].get('FA_loss_value_validation_max_relative',0) for r in records),
                    initial_qpos=np.r_[common[0],poses[0]].tolist())
            expected=reference(poses,times)
            position=1000*np.linalg.norm(actual[:,:3]-expected[:,:3],axis=1)
            qa=actual[:,3:]/np.linalg.norm(actual[:,3:],axis=1,keepdims=True)
            angle=np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(qa*expected[:,3:],axis=1)),0,1)))
            results[key]=dict(executor=executor,method=method,complete=len(actual)==180,
                duration_seconds=float(times[-1]),all=stats(position,angle,np.ones(len(times),bool)),
                motion=stats(position,angle,times<=duration+1e-8),hold=stats(position,angle,times>duration+1e-8),audit=audit)
            curves[key]=np.c_[times,position,angle,actual,expected]
    if not partial and (len(results)!=8 or not all(r['complete'] for r in results.values())):
        raise RuntimeError('All eight fixed-duration cases must complete before the final report')
    (root/'metrics.json').write_text(json.dumps(results,indent=2))
    rows=[]
    for key,r in results.items():
        for segment in ['all','motion','hold']:
            if r[segment]:rows.append(dict(case=key,segment=segment,**r[segment]))
    if rows:
        with (root/'metrics.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for key,curve in curves.items():
        np.savetxt(root/key/'pose_errors.csv',curve,delimiter=',',header='time_s,position_mm,rotation_deg,'+','.join('actual_'+str(k) for k in range(7))+','+','.join('reference_'+str(k) for k in range(7)),comments='')
    text=['# 八组闭环实验','',f'同一条 SugarBox 演示、同一 CA 数组；每次运行 5.4 s。运动段 0–{duration:.4f} s，其后保持末帧。',
        '按真实仿真时间插值示教物体位姿计算误差，所有方法统一以 30 ms 采样；不使用各控制器自己的 lookahead 目标。',
        '表中位置为欧氏距离平均误差，旋转为四元数最短角平均误差；单位 mm / °。','']
    for executor in ['mujoco','comfree']:
        text+=['## '+executor,'','| 方法 | 运动段 | 保持段 | 全程 | 最后一帧 |','|---|---:|---:|---:|---:|']
        for method,label in LABELS.items():
            r=results.get(executor+'/'+method)
            if not r:text.append(f'| {label} | 未完成 | — | — | — |');continue
            def val(s):return f"{r[s]['position_mean_mm']:.2f} / {r[s]['rotation_mean_deg']:.2f}" if r[s] else '—'
            text.append(f"| {label}{'' if r['complete'] else '（运行中）'} | {val('motion')} | {val('hold')} | {val('all')} | {r['all']['last_position_mm']:.2f} / {r['all']['last_rotation_deg']:.2f} |")
        text+=['']
    text+=['## 实验边界','',
        '- Lambda optimizer 关闭；实际接触力和力闭合只用于记录，不作运输或控制许可。',
        '- 第 2 组只换前向预测模型为官方 ComFree，保持当前 cost、一次 SQP_RTI、候选族与碰撞检查；其数值局部导数按 stage 重算。',
        '- FA cost 组使用 FA 全部几何代价公式、原始单位、终端参考缩放和速度/加速度平滑项；使用我们当前 30 ms 采样、120 ms 时域、一次 RTI 与候选评分。其加速度为该离散模型一步的速度变化率。',
        '- 原版 DexForge 保留 10 ms 预测、300 ms 时域、64 个起点与 40 次 Adam；统一初始 CA 姿态和零初速度，关闭初始状态偏移优化。两种环境中均重新闭环求解。',
        '- 两环境具有相同惯量、摩擦和位置伺服参数；MuJoCo 使用原生 mesh 碰撞，ComFree 使用原版 querypoint/SDF 碰撞。',
        '- 此为单条序列、单次确定性对比，不能估计跨物体成功率。','']
    if (root/'analysis.md').exists() and len(results)==8 and all(r['complete'] for r in results.values()):
        text.append((root/'analysis.md').read_text())
    (root/'report.md').write_text('\n'.join(text))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--partial',action='store_true');a=p.parse_args()
    results=report(a.root,a.partial)
    for k,r in results.items():print(k,len(r),r['all']['position_mean_mm'],r['all']['rotation_mean_deg'])
