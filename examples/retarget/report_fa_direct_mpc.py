"""Compare direct OCP and saved FA tracking on the demonstration clock."""
import json
from pathlib import Path
import numpy as np
from examples.retarget.matrix_report import reference,stats


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-suffix',default='retained_rti3')
    args=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('outputs/fa_direct_mpc_20261016');base=Path('outputs/fa_admm_solver_20261016')
    poses=np.load(base/'contactaware/object_pose_7.npy')[:,[4,5,6,3,0,1,2]]
    hand=np.load(base/'contactaware/wuji_right_qpos.npy')
    motion=(len(poses)-1)/30
    figure,axes=plt.subplots(2,2,figsize=(11,7),sharex='col',layout='constrained')
    results={};curves={}
    for col,executor in enumerate(('comfree','mujoco')):
        cases=[('Direct ComFree OCP',root/(executor+'_'+args.case_suffix)),
            ('Saved FA + L-BFGS',base/executor/'lbfgs_fixed')]
        for label,directory in cases:
            if (directory/'INVALIDATED.json').exists():raise ValueError(f'Invalidated case: {directory}')
            saved=np.load(directory/'rollout.npz');q=saved['qpos_traj'][1:]
            time=np.arange(1,len(q)+1)*.01;goal=reference(poses,time)
            position=1000*np.linalg.norm(q[:,-7:-4]-goal[:,:3],axis=1)
            quat=q[:,-4:];quat=quat/np.linalg.norm(quat,axis=1,keepdims=True)
            angle=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(quat*goal[:,3:],axis=1)),0,1)))
            frame=np.minimum(time*30,len(hand)-1);i=np.floor(frame).astype(int);j=np.minimum(i+1,len(hand)-1)
            h=(1-(frame-i)[:,None])*hand[i]+(frame-i)[:,None]*hand[j]
            error=q[:,:26]-h
            key=executor+'/'+label
            row=dict(all=stats(position,angle,np.ones(len(time),bool)),motion=stats(position,angle,time<=motion+1e-8),
                hold=stats(position,angle,time>motion+1e-8),
                hand_base_mean_mm=float(1000*np.linalg.norm(error[:,:3],axis=1).mean()),
                hand_finger_rmse_rad=float(np.sqrt(np.mean(error[:,6:]**2))))
            deficit=1000*(goal[:,2]-q[:,-5]);lost=np.flatnonzero((deficit>80)&(goal[:,2]-goal[0,2]>.05))
            row.update(final_object_height_mm=float(1000*q[-1,-5]),reference_final_height_mm=float(1000*goal[-1,2]),
                first_height_deficit_over_80mm_s=float(time[lost[0]]) if len(lost) else None)
            if label.startswith('Direct'):
                report=json.loads((directory/'report.json').read_text())
                windows=[json.loads(line) for line in (directory/'windows.jsonl').read_text().splitlines()]
                warm=windows[1:]
                row.update(warm_median_ms=1000*np.median([w['wall_s'] for w in warm]),
                    warm_p95_ms=1000*np.quantile([w['wall_s'] for w in warm],.95),
                    NLP_nonzero_windows=report['failed_windows'],QP_unconverged_windows=report['unconverged_QP_windows'],
                    template_updates=sum(w.get('template_update',{}).get('updated',False) for w in windows),
                    accepted_SQP_steps=sum(w['accepted_SQP_steps'] for w in windows),
                    native_mean_ms={k:1000*np.mean([w['native_seconds'].get(k,0.) for w in warm])
                        for k in ('dynamics_forward','dynamics_jacobian','FA_residual','FA_residual_jacobian')})
                notes=root/'timing_notes.json'
                if notes.exists():row['timing_note']=json.loads(notes.read_text()).get(directory.name,{})
                axes[1,col].plot(time,goal[:,2]*1000,'k--',label='Demonstration')
            results[key]=row;curves[key]=np.c_[time,position,angle,q[:,-5]*1000]
            axes[0,col].plot(time,position,label=label)
            axes[1,col].plot(time,q[:,-5]*1000,label=label)
        axes[0,col].set_title(executor)
        axes[0,col].set_ylabel('Object position error (mm)');axes[1,col].set_ylabel('Object height (mm)')
        axes[1,col].set_xlabel('Simulation time (s)')
        for axis in axes[:,col]:
            axis.axvline(motion,color='gray',linestyle=':',linewidth=1)
            axis.grid(alpha=.25);axis.legend(fontsize=8)
    figure.savefig(root/'tracking_comparison.png',dpi=180)
    for key,curve in curves.items():
        np.savetxt(root/(key.replace('/','_').replace(' ','_')+'_errors.csv'),curve,delimiter=',',
            header='time_s,position_error_mm,rotation_error_deg,object_height_mm',comments='')
    (root/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    lines=['# Direct ComFree MPC 闭环实验','',
        '同一 SugarBox MANO/ContactAware 序列，初始 CA 姿态与零速度，5.4 s 闭环。误差以真实仿真时间对齐示教；运动段结束于 %.4f s。'%motion,'',
        '| 执行器 / 方法 | 物体平均位置误差 mm | 平均旋转误差 ° | 手基座平均误差 mm | 手指关节 RMSE rad | 完整 MPC 中位数 ms |',
        '|---|---:|---:|---:|---:|---:|']
    for key,r in results.items():
        lines.append('| %s | %.2f | %.2f | %.2f | %.3f | %s |'%(key,r['all']['position_mean_mm'],r['all']['rotation_mean_deg'],
            r['hand_base_mean_mm'],r['hand_finger_rmse_rad'],('%.1f'%r['warm_median_ms']) if 'warm_median_ms' in r else '历史轨迹，未复测'))
    direct=[r for key,r in results.items() if key.endswith('Direct ComFree OCP')]
    if all(r['reference_final_height_mm']-r['final_object_height_mm']>80 for r in direct):
        lines+=['','**两组直接 OCP 都跟随了主要抬升过程，但保持段最终滑落，尚未成功完成整条轨迹。**',
            '最终物体高度均约 90–100 mm，示教保持高度约 262 mm；完整周期也尚未达到几十 ms。']
    lines+=['','![Tracking comparison](tracking_comparison.png)','',
        'Direct OCP 使用真实原生 ComFree 外部 dynamics 与解析 Jacobian，30 × 10 ms、300 ms 时域；每 30 ms 重规划。更新预算及求解状态见各 case 的 report.json。',
        'FA 几何/跟踪残差逐项进入 Gauss–Newton；候选更新经真实非线性 rollout 的同一代价下降检查。',
        'ADMM 每 5 个 MPC 周期（150 ms）搜索一次；目标点软偏移惩罚权重为 5 / mm²；候选接触力不注入 MPC 或执行器。',
        '实际动力学接触来自预测状态的原生 querypoint/SDF 碰撞；目标模板仅进入 FA 接近代价。原生代理碰撞几何不等于 MuJoCo 三角网格碰撞。','',
        '是否跟随示教搬运动作，以物体高度曲线及位置/旋转误差为依据；QP 的局部数值成功不等于闭环跟踪成功。',
        '旧 FA/L-BFGS 轨迹仅作为已有可完成跟踪的参照。它使用 5 个插值控制节点；本试验使用 30 个稠密控制节点，并使用低频模板更新，不能将差异全部归因于求解器。',
        '完整周期耗时包含原生动力学、解析导数、低频 ADMM、QP、真实轨迹检查及 CPU/GPU 传输；不含初始化/JIT。','',
        'MuJoCo 后半程观测到另一个 CUDA 计算进程，记录的 wall time 包含资源竞争，不能作为严格的求解器速度对照。ComFree 测试期间未观测到第二个计算任务。','',
        '原始结果：`comparison.json`、各 case 的 `report.json`、`windows.jsonl`、`rollout.npz` 和 `templates.npz`。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
