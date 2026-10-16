"""Report matched closed loops separately from common-state solver probes."""
import argparse
import json
from pathlib import Path
import numpy as np
from examples.retarget.matrix_report import reference


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    pose=np.load('outputs/fa_admm_solver_20261016/contactaware/object_pose_7.npy')[:,[4,5,6,3,0,1,2]]
    historical=np.load('outputs/fa_admm_solver_20261016/comfree/lbfgs_fixed/rollout.npz')
    import mujoco
    # The mesh scene carries the same actuator metadata and is readable by
    # stock MuJoCo; the SDF scene requires the native asset preprocessing.
    model=mujoco.MjModel.from_xml_path('outputs/fa_admm_solver_20261016/sequence/scene/wuji_right/sugar_box_mesh.xml')
    bounds_audit={}
    closed={};probes={};no_ops={}
    for case in sorted((a.root/'comfree').glob('*')):
        if (case/'NO_OP.json').exists():
            no_ops[case.name]=json.loads((case/'NO_OP.json').read_text());continue
        path=case/'rollout.npz';complete=path.exists() and not (case/'early_stop.json').exists()
        if not path.exists():path=case/'partial.npz'
        if not path.exists():continue
        saved=np.load(path);q=saved['qpos_traj'][1:];t=np.arange(1,len(q)+1)*.01;goal=reference(pose,t)
        position=1000*np.linalg.norm(q[:,-7:-4]-goal[:,:3],axis=1);deficit=1000*(goal[:,2]-q[:,-5])
        angle=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q[:,-4:]/np.linalg.norm(q[:,-4:],axis=1,keepdims=True)*goal[:,3:],axis=1)),0,1)))
        lost=np.flatnonzero((deficit>80)&(goal[:,2]-goal[0,2]>.05))
        early_drop=(case/'early_stop.json').exists()
        row=dict(complete=complete,status='failed_early' if early_drop else 'complete' if complete else 'running',
            trajectory_file=path.name,early_drop=early_drop,duration_s=float(t[-1]),position_mean_mm=float(position.mean()),
            motion_position_mean_mm=float(position[t<=2.4].mean()),rotation_mean_deg=float(angle.mean()),
            hold_position_mean_mm=float(position[t>2.4].mean()) if np.any(t>2.4) else None,
            final_position_error_mm=float(position[-1]),final_rotation_error_deg=float(angle[-1]),
            first_drop_s=float(t[lost[0]]) if len(lost) else None,final_height_error_mm=float(-deficit[-1]),
            warm_median_ms=float(np.median(saved['mpc_step_sec'][1:])*1000) if len(saved['mpc_step_sec'])>1 else None,
            warm_p95_ms=float(np.quantile(saved['mpc_step_sec'][1:],.95)*1000) if len(saved['mpc_step_sec'])>1 else None)
        if case.name=='baseline':row['reproduction_max_absolute_error']={k:float(np.max(abs(saved[k]-historical[k][:len(saved[k])]))) for k in ('qpos_traj','qvel_traj','ctrl_traj')}
        closed[case.name]=row
        controls=saved['ctrl_traj'];limits=model.actuator_ctrlrange
        violation=(np.maximum(limits[:,0]-controls,0)+np.maximum(controls-limits[:,1],0))*model.actuator_ctrllimited
        bounds_audit[case.name]=dict(raw_servo_commands_before_native_clamp=len(controls),
            violating_raw_commands=int(np.count_nonzero(np.any(violation>1e-7,axis=1))),max_raw_violation=float(violation.max()))
    for path in sorted((a.root/'probes').glob('*/probes.jsonl')):probes[path.parent.name]=[json.loads(line) for line in path.read_text().splitlines()]
    baseline={r['window']:r for r in probes.get('baseline',[])}
    comparison=[]
    for factor,rows in probes.items():
        for row in rows:
            old=baseline.get(row['window'])
            if old is None:continue
            comparison.append(dict(factor=factor,window=row['window'],time=row['time'],FA_cost=row['final_FA_cost'],
                baseline_FA_cost=old['final_FA_cost'],cost_ratio=row['final_FA_cost']/old['final_FA_cost'],
                terminal_position_error_mm=row['terminal_position_error_mm'],baseline_terminal_position_error_mm=old['terminal_position_error_mm'],
                solver_seconds_with_contention=row['solver_seconds']))
    timing={}
    for path in sorted((a.root/'timing_serial/probes').glob('*/probes.jsonl')):
        rows=[json.loads(line) for line in path.read_text().splitlines()];warm=[r for r in rows if r['window']==120]
        if not warm:continue
        r=warm[-1];s=r['solver']
        timing[path.parent.name]=dict(solver_seconds=r['solver_seconds'],cost=r['final_FA_cost'],forward_calls=s['forward_calls'],
            backward_calls=s['backward_calls'],qp_wall_seconds=sum(d.get('qp_wall_seconds',0.) for d in s.get('hand_subproblems',[])))
    inverse={}
    inverse_reference={'pre_fixed':'pre_admm','pre_admm_line8':'pre_admm','post_admm':'pre_admm','post_fixed':'pre_fixed',
        'post_fixed_LM_1e-4':'post_fixed','post_fixed_LM_1':'post_fixed',
        'post_fixed_relative_warm':'post_fixed','post_fixed_relative_RTI8':'post_fixed_relative_warm',
        'post_admm_consistent_commit':'post_admm','post_admm_consistent_relative_warm':'post_admm_consistent_commit'}
    directories={'pre_admm':Path('outputs/fa_slip_diagnosis_20261016/comfree'),
        'pre_fixed':Path('outputs/fa_slip_diagnosis_20261016/comfree_fixed_template'),
        'pre_admm_line8':Path('outputs/fa_slip_diagnosis_20261016/comfree_deeper_search'),
        'post_admm':a.root/'direct_post_tracking','post_fixed':a.root/'direct_post_fixed',
        'post_fixed_LM_1e-4':a.root/'direct_post_fixed_lm1em4',
        'post_fixed_LM_1':a.root/'direct_post_fixed_lm1',
        'post_fixed_relative_warm':a.root/'direct_post_fixed_relative_warm',
        'post_fixed_relative_RTI8':a.root/'direct_post_fixed_relative_rti8',
        'post_admm_consistent_relative_warm':a.root/'direct_post_relative_consistent',
        'post_admm_consistent_commit':a.root/'direct_post_consistent_commit'}
    for name,path in directories.items():
        if not (path/'report.json').exists() or (path/'INVALIDATED.json').exists():continue
        r=json.loads((path/'report.json').read_text());saved=np.load(path/'rollout.npz');q=saved['qpos_traj'][1:]
        t=np.arange(1,len(q)+1)*.01;goal=reference(pose,t);lost=np.flatnonzero(1000*(goal[:,2]-q[:,-5])>80)
        inverse[name]=dict(position_mean_mm=r['tracking']['object_position_mean_mm'],rotation_mean_deg=r['tracking']['object_rotation_mean_deg'],
            first_drop_s=float(t[lost[0]]) if len(lost) else None,QP_failed_windows=r['unconverged_QP_windows'],
            reference_case=inverse_reference.get(name))
    contacts={}
    if (a.root/'contact_comparison.json').exists():
        for name,samples in json.loads((a.root/'contact_comparison.json').read_text()).items():
            for sample in samples:
                if sample['time']!=3.6:continue
                contacts[name]={key:sample[key] for key in ('total_normal_N','hand_vertical_N','loaded_contacts',
                    'object_angular_velocity_norm','object_velocity_mm_s','replay_next_qpos_max_error')}
    findings=[]
    if 'baseline' in closed and closed['baseline']['complete']:
        r=closed['baseline']
        reproduced=all(v==0. for v in r['reproduction_max_absolute_error'].values())
        findings.append(f'FA 基准平均位置误差 {r["position_mean_mm"]:.2f} mm；qpos、qvel、执行控制与历史成功记录'+('逐位一致。' if reproduced else '存在差异，需检查 reproduction_max_absolute_error。'))
    if 'slsqp' in closed and closed['slsqp']['complete']:
        r=closed['slsqp'];verdict='未达到掉落阈值，没有单独复现 Direct 的严重掉落。' if r['first_drop_s'] is None else f'在 {r["first_drop_s"]:.2f} s 达到掉落阈值。'
        findings.append(f'仅换 SciPy SLSQP：平均位置误差 {r["position_mean_mm"]:.2f} mm，'+verdict)
    if 'acados' in closed and closed['acados']['complete']:
        r=closed['acados'];verdict='未达到掉落阈值。' if r['first_drop_s'] is None else f'在 {r["first_drop_s"]:.2f} s 达到掉落阈值。'
        findings.append(f'原五节点问题仅换 acados BFGS 控制 QP：平均位置误差 {r["position_mean_mm"]:.2f} mm，'+verdict+'局部 QP 已通过解析解及带边界 KKT 检验；它不等于 Direct 的 Gauss–Newton 多重射击。')
    if all(k in closed and closed[k]['complete'] for k in ('budget3','budget3-contiguous')) and closed['budget3']['first_drop_s'] is not None and closed['budget3-contiguous']['first_drop_s'] is None:
        findings.append(f'三次分块 L-BFGS 更新 [1,1,1,0,0] 在 {closed["budget3"]["first_drop_s"]:.2f} s 掉落；连续三次 [3,0,0,0,0] 未达到掉落阈值，但最终下滑 {abs(closed["budget3-contiguous"]["final_height_error_mm"]):.1f} mm。低预算下重置曲率历史是已复现的失败因素；该实验不能等同于三次 RTI。')
    if 'dense-controls' in closed and closed['dense-controls']['complete']:
        r=closed['dense-controls'];verdict='未达到掉落阈值。不能将自由度增加本身视为必然失稳原因。' if r['first_drop_s'] is None else f'在 {r["first_drop_s"]:.2f} s 达到掉落阈值。'
        findings.append(f'仅增加控制自由度、保留原平滑及 warm 规则的扩展：平均位置误差 {r["position_mean_mm"]:.2f} mm，'+verdict)
    if 'post_admm_consistent_commit' in inverse and 'post_admm' in inverse and inverse['post_admm_consistent_commit']['first_drop_s'] is None:
        findings.append(f'确认模板评分/提交不一致：旧 Direct 最大偏差 10.24 mm。保持其余 post-tracking Direct 配置不变，修复后平均位置误差 {inverse["post_admm_consistent_commit"]["position_mean_mm"]:.2f} mm（原 16.08 mm），5.4 s 内未复现原 5.31 s 掉落；仍未恢复 FA 精度。')
    if 'post_fixed_relative_warm' in inverse:
        findings.append(f'实际 Direct 的 warm 反向对照：固定模板下，绝对控制 warm 改为参考相对修正后，平均位置误差由 11.06 降至 {inverse["post_fixed_relative_warm"]["position_mean_mm"]:.2f} mm。它在旧 FA/40 次迭代上的单项影响不能替代低预算 Direct 中的实际影响。')
    if 'post_fixed_relative_RTI8' in inverse:
        r=inverse['post_fixed_relative_RTI8'];drop='未达到掉落阈值' if r['first_drop_s'] is None else f'在 {r["first_drop_s"]:.2f} s 掉落'
        findings.append(f'实际 Direct 只把 RTI 3→8：平均位置误差 {r["position_mean_mm"]:.2f} mm，{drop}。不能把 Direct 退化简单归因于三次 RTI 不足，更不能直接套用 L-BFGS 分块实验结论。')
    if 'post_admm_consistent_relative_warm' in inverse:
        r=inverse['post_admm_consistent_relative_warm'];drop='5.4 s 内未达到掉落阈值' if r['first_drop_s'] is None else f'在 {r["first_drop_s"]:.2f} s 掉落'
        findings.append(f'组合修正验证（post-tracking、模板评分/提交一致、参考相对 warm，保留低频 ADMM 和三次 RTI）：平均位置误差 {r["position_mean_mm"]:.2f} mm、平均旋转误差 {r["rotation_mean_deg"]:.2f}°，{drop}。仍明显弱于成功 FA。')
    findings.append('原 FA 已包含当前 SDF 接触法向/坐标系梯度，这不是两版的差异。另有已确认的 post/pre-step tracking 错位；在原 FA 上单独引入该错位并未导致掉落，因此不能单独归因于它。')
    findings.append('上述为已验证因素，不将单项影响相加。Direct 的 Gauss–Newton 曲率与多重射击、低预算之间的交互尚未被单独穷尽；BFGS 控制 QP 不等价于此前的 Direct Gauss–Newton。')
    result=dict(findings=findings,closed_loops=closed,no_op_factors=no_ops,common_state_probes=comparison,isolated_timing_at_3p6s=timing,direct_OCP_inverse_tests=inverse,
        physical_contacts_at_3p6s=contacts,
        timing_note='Concurrent closed loops/probes are correctness experiments; timings include GPU/CPU contention. Do not rank speed from these alone.')
    (a.root/'executed_control_bounds_audit.json').write_text(json.dumps(dict(
        native_control_clamping_enabled=not bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL),
        meaning='Raw commands are audited before native actuator clipping, not applied unconstrained actuation.',cases=bounds_audit),indent=2)+'\n')
    (a.root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(16,10),sharex=True)
    for color_index,(case,row) in enumerate(closed.items()):
        path=a.root/'comfree'/case/row['trajectory_file'];saved=np.load(path)
        q=saved['qpos_traj'][1:];t=np.arange(1,len(q)+1)*.01;goal=reference(pose,t)
        label=case+(' (drop)' if row['first_drop_s'] is not None else ' (running)' if not row['complete'] else '')
        color=plt.get_cmap('tab20')(color_index)
        for column in range(2):
            axes[0,column].plot(t,1000*np.linalg.norm(q[:,-7:-4]-goal[:,:3],axis=1),label=label,color=color)
            axes[1,column].plot(t,1000*(q[:,-5]-goal[:,2]),color=color)
    axes[0,0].set_title('Full range');axes[0,1].set_title('Detail (clipped to 60 mm)')
    axes[0,1].set_ylim(0,60);axes[1,1].set_ylim(-60,20)
    axes[0,0].set_ylabel('Object position error (mm)');axes[0,0].legend(fontsize=8,ncol=2)
    axes[1,0].set_ylabel('Height minus reference (mm)')
    for ax in axes[1]:ax.set_xlabel('Time (s)');ax.axhline(-80,color='k',ls='--')
    for ax in axes.flat:ax.grid(alpha=.25);ax.axvline(2.4,color='grey',ls=':',lw=1)
    fig.tight_layout();fig.savefig(a.root/'closed_loop_ablation.png',dpi=160)
    plt.close(fig)
    if 'post_admm_consistent_commit' in inverse:
        fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
        paths={'successful_FA':a.root/'comfree/baseline'}
        paths.update({name:directories[name] for name in ('post_admm','post_fixed','post_admm_consistent_commit',
            'post_fixed_relative_warm','post_fixed_relative_RTI8','post_admm_consistent_relative_warm') if name in inverse})
        for name,path in paths.items():
            if not (path/'rollout.npz').exists():continue
            saved=np.load(path/'rollout.npz');q=saved['qpos_traj'][1:];t=np.arange(1,len(q)+1)*.01;goal=reference(pose,t)
            angle=np.degrees(2*np.arccos(np.clip(np.abs(np.sum(q[:,-4:]/np.linalg.norm(q[:,-4:],axis=1,keepdims=True)*goal[:,3:],axis=1)),0,1)))
            axes[0].plot(t,1000*np.linalg.norm(q[:,-7:-4]-goal[:,:3],axis=1),label=name)
            axes[1].plot(t,angle);axes[2].plot(t,1000*(q[:,-5]-goal[:,2]))
        axes[0].set_ylabel('Position error (mm)');axes[0].legend(fontsize=9,ncol=2)
        axes[1].set_ylabel('Rotation error (deg)');axes[2].set_ylabel('Height minus reference (mm)')
        axes[2].set_xlabel('Time (s)');axes[2].axhline(-80,color='k',ls='--')
        for ax in axes:ax.grid(alpha=.25);ax.axvline(2.4,color='grey',ls=':',lw=1)
        fig.tight_layout();fig.savefig(a.root/'direct_commit_inverse.png',dpi=160);plt.close(fig)
    lines=['以成功的 FA/L-BFGS（8 起点、40 次迭代、5 控制节点）为基准的逐项消融。',
        '每个 closed-loop case 都从相同初始姿态、零速度开始；共同状态探测则使用成功轨迹相同 q/v、warm、跨窗口历史和种子。两类证据分别列出。','']
    lines+=['- '+finding for finding in findings]
    lines+=['',
        '| 单项变更 | 状态 | 已执行 s | 全程平均位置 mm | 运动段平均位置 mm | 最终高度差 mm | 平均旋转 ° | 首次掉落 s |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,row in closed.items():
        status={'complete':'完成','failed_early':'掉落后结束','running':'运行中'}[row['status']]
        lost='未达到阈值' if row['first_drop_s'] is None else f'{row["first_drop_s"]:.2f}'
        lines.append(f'| {name} | {status} | {row["duration_s"]:.2f} | {row["position_mean_mm"]:.2f} | {row["motion_position_mean_mm"]:.2f} | {row["final_height_error_mm"]:.2f} | {row["rotation_mean_deg"]:.2f} | {lost} |')
    lines+=['','运动段采用共同的 0–2.4 秒区间；掉落后结束的全程均值只覆盖实际执行区间，不能与完整 5.4 秒均值直接排名。',
        '掉落阈值为低于 MANO 抬升目标 80 mm；未达到该阈值不代表无滑移或跟踪成功。最终高度差为实际高度减目标高度。',
        '当前实验针对同一条 MANO 轨迹、相同初始条件与随机种子，不能据此推断其它轨迹上的成功率。',
        '这里是按固定仿真时间执行的闭环实验：求解耗时不推进物理时钟，因此轨迹结果不代表已满足实机实时性。所有单因素闭环均采用原生 ComFree 执行器，避免同时引入 MuJoCo mesh 与 SDF 模型差异。','',
        '| 共同状态单项变更 | 时刻 s | 最终 FA cost | 基线 cost | 预测终点误差 mm | 基线误差 mm |',
        '|---|---:|---:|---:|---:|---:|']
    for row in comparison:lines.append('| %s | %.2f | %.6f | %.6f | %.3f | %.3f |'%(row['factor'],row['time'],row['FA_cost'],row['baseline_FA_cost'],row['terminal_position_error_mm'],row['baseline_terminal_position_error_mm']))
    lines+=['','Cost 统一乘以 1e-6。closed_loop_ablation.png 展示闭环曲线；direct_commit_inverse.png 对比 Direct 模板修正前后的位置、旋转和高度。',result['timing_note'],'',
        'dense-controls 只增加自由度，仍在原来五个端点上计算控制平滑项；warm 采用 P*T_coarse*C，在原子空间与原策略交换，新内部方向每周期从端点嵌入初始化。dense-smoothing 保留五节点参数化，仅改用 10 ms 控制平滑及其边界历史。',
        'full-contact-jacobian 在当前 SDF 场景中是 no-op：原 FA 已包含该接触法向/坐标系反传；control-bounds 使用相同解析梯度与历史长度的投影 L-BFGS，将控制节点限制到 actuator 范围。',
        '原 FA 的原始指令允许超出 actuator 范围，但原生动力学已开启控制裁剪；节点先投影与插值后裁剪会产生不同的控制曲线。原始指令审计见 executed_control_bounds_audit.json。',
        'slow-templates 在原成功手部优化器上接入同一个 150 ms、anchor=5 的模板搜索；不施加理想接触力。',
        'acados 是原 130 维问题的 BFGS 控制 SQP，允许同样 40 次更新预算，不是此前 30 节点、多重射击/Gauss–Newton 版本。',
        '单起点采用原方法的第 0 个 warm 候选，因此也移除了其余候选携带的 velocity-prior 初始化族；解释时需要考虑候选族这一含义。',
        '解析梯度嵌入检验见 embedding_validation.json。所有求解器使用解析梯度，无差分 Jacobian。']
    lines+=['','独立单 GPU 共同状态计时（3.6 秒；先用其它快照完成图捕获预热）：','',
        '| 设置 | wall s | cost | 前向次数 | 反传次数 | QP wall s |','|---|---:|---:|---:|---:|---:|']
    for name,r in timing.items():lines.append('| %s | %.3f | %.6f | %d | %d | %.4f |'%(name,r['solver_seconds'],r['cost'],r['forward_calls'],r['backward_calls'],r['qp_wall_seconds']))
    lines+=['','该计时期间暂停了本任务其它 GPU 作业；物理状态和控制不受调度暂停影响。scheduling.jsonl 记录调度事件。',
        'acados 在此处仅替换原问题的控制 QP 后端，仍调用原 FA/Warp 动力学与梯度，并非将物理模型整体编译为纯 C dynamics。',
        '这些是同一快照的单次计时，不按几个百分点的小差异给求解器排名；数量级差异及 QP 占比才是本轮性能结论。','',
        '3 步分块的预算分配是 [1,1,1,0,0]；连续 3 步为 [3,0,0,0,0]。前者每步重置曲率，不能积累 L-BFGS 历史。',
        '同预算不同分块对照用于区分预算与历史重置，不能将分块 3 步失败直接解释为任何 3 步 SQP/Gauss–Newton 都必然失败。','',
        'Direct OCP 反向修正对照（相对“基准行”只改该项；除 RTI8 行外均为每周期最多 3 RTI。pre_* 为上轮已验证记录，post_* 为本轮试验）：','',
        '| 设置 | 基准行 | 平均位置误差 mm | 平均旋转误差 ° | 首次掉落 s | QP 失败窗口 |','|---|---|---:|---:|---:|---:|']
    for name,r in inverse.items():lines.append('| %s | %s | %.3f | %.3f | %s | %d |'%(name,r['reference_case'],r['position_mean_mm'],r['rotation_mean_deg'],r['first_drop_s'],r['QP_failed_windows']))
    lines+=['','3.6 秒实际物理接触快照（由已执行状态与控制重放，逐项校验下一状态；法向力总和不等价于完整摩擦扳手裕量）：','',
        '| 设置 | 法向力总和 N | 手部竖直合力 N | 承载接触数 | 竖直速度 mm/s | 角速度范数 rad/s |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in contacts.items():lines.append('| %s | %.3f | %.3f | %d | %.2f | %.3f |'%(name,r['total_normal_N'],
        r['hand_vertical_N'],r['loaded_contacts'],r['object_velocity_mm_s'][2],r['object_angular_velocity_norm']))
    lines+=['','每点位置、法向力/切向力见 contact_forces.csv；逐手指合力及掉落前采样见 contact_comparison.json。已经掉落的组在此时可能已落地，应结合首次掉落时刻阅读。']
    lines+=['','tracking-time：原 FA tracking 的 qpos 是 post-step，而 body/contact geometry 是 pre-step。旧 Direct OCP 将两者都放在 pre-step，形成一拍错位。',
        'objective_discretization_audit.json 对同一条成功控制序列逐位重建物理预测，再分解 cost 差异：初始差异的约 99.8% 来自 tracking 时间错位，而非控制平滑。',
        'tracking_time_validation.json 验证此单因素变换等价于逐项使用 pre-step tracking，误差小于 4e-7。',
        'full_horizon_gradient_validation.json 在两侧关闭不同的平滑项后，比较完整 horizon tracking/geometry cost 和控制梯度；相对梯度误差最大约 1.5e-5，使用两条独立解析反传路径。',
        'condensed_qp_validation.json 校验 acados 130 维局部二次模型的解析无约束解和有边界 KKT；局部 QP 正确并不保证非线性闭环抓持稳定。',
        '接触梯度策略在当前 SDF 场景原本相同；见 policy_audit/contact_gradient_policy.json，不能将其当作两版的有效差异。',
        '模板评分/提交一致性审计见 template_commit_discrepancy.json：旧实现对逐时刻点序列评分，再按材料点提交最早提案，两者可相差约 10 mm。--consistent-template-commit 按实际持有模板展开候选后再评分，并逐次断言提交一致；修复此不变量与恢复稳定跟踪是两个独立检查。',
        'template_drift_from_MANO.json 记录相对原始 MANO/CA 模板的累计位移。相对当前模板的二次位移惩罚不限制累计漂移，也不证明非凸接触组合问题存在唯一全局解。',
        'invalidated_* 目录中的早期接线检查不用于任何结论。',
        '总 cost 在改变模板、tracking 时序或平滑定义后不再是同一个目标，不能以较小 cost 直接宣称跟踪更好；闭环误差和掉落需独立检查。']
    (a.root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(closed,indent=2))


if __name__=='__main__':main()
