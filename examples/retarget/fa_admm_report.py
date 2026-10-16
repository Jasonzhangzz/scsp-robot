"""Common-clock trajectory and contact diagnostics for the FA contact split."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from examples.retarget.matrix_report import reference,stats


METHODS={'fa_fixed':'固定 MANO，原 FA Adam','lbfgs_fixed':'固定 MANO，L-BFGS','admm':'MANO 初始化，ADMM + L-BFGS'}


def read_case(root,executor,method,baseline,partial=False):
    case=(baseline if method=='dexforge' else root)/executor/method
    path=case/'rollout.npz'
    if not path.exists() and partial:path=case/'partial.npz'
    if not path.exists():return None
    return case,np.load(path)


def trajectory_report(root,baseline,partial=False):
    poses=np.load(root/'contactaware/object_pose_7.npy')[:,[4,5,6,3,0,1,2]]
    motion=(len(poses)-1)/30;results={};curves={}
    for executor in ['mujoco','comfree']:
        for method in METHODS:
            loaded=read_case(root,executor,method,baseline,partial)
            if loaded is None:continue
            case,a=loaded;actual=a['qpos_traj'][3::3,-7:];times=(np.arange(len(actual))+1)*.03
            expected=reference(poses,times)
            position=1000*np.linalg.norm(actual[:,:3]-expected[:,:3],axis=1)
            quaternion=actual[:,3:]/np.linalg.norm(actual[:,3:],axis=1,keepdims=True)
            angle=np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(quaternion*expected[:,3:],axis=1)),0,1)))
            key=executor+'/'+method
            results[key]=dict(samples=len(actual),complete=len(actual)==180,
                all=stats(position,angle,np.ones(len(actual),bool)),
                motion=stats(position,angle,times<=motion+1e-8),hold=stats(position,angle,times>motion+1e-8),
                median_solve_seconds=float(np.median(a['mpc_step_sec'])) if 'mpc_step_sec' in a else None)
            results[key]['finite_candidate_fraction']=float(np.mean(a['multistart_valid']))
            curves[key]=np.c_[times,position,angle]
            if (case/'contact_admm.npz').exists():
                s=np.load(case/'contact_admm.npz');mask=s['mask']>0
                indices=s['window_indices'] if 'window_indices' in s else np.arange(len(mask))
                # All horizon points and only the next executed interval separately.
                def metrics(selection):
                    primal=np.linalg.norm(s['c_before_surface_polish']-s['z'],axis=-1)[selection]*1000
                    shift=np.linalg.norm(s['z']-s['template'],axis=-1)[selection]*1000
                    handshift=np.linalg.norm(s['hand_local']-s['template_hand_local'],axis=-1)[selection]*1000
                    return dict(samples=len(primal),primal_mean_mm=float(primal.mean()),primal_p95_mm=float(np.quantile(primal,.95)),
                        contact_shift_mean_mm=float(shift.mean()),contact_shift_p95_mm=float(np.quantile(shift,.95)),
                        contact_shift_max_mm=float(shift.max()),hand_local_shift_mean_mm=float(handshift.mean()))
                next_interval=mask.copy();next_interval[:,3:]=False
                results[key]['contact_split']=dict(horizon=metrics(mask),next_interval=metrics(next_interval))
                results[key]['contact_split']['available_windows']=indices.tolist()
                results[key]['contact_split']['window_coverage_fraction']=len(indices)/len(actual)
    if not partial and (len(results)!=6 or not all(r['complete'] for r in results.values())):
        raise RuntimeError('Six full cases are required for a final report')
    return results,curves


def contact_report(root,baseline,forge,results):
    import mujoco
    from examples.retarget.matrix_comfree import ComFreeEngine
    from examples.retarget.compare_contact_trajectories import gws_margin
    from planning.contact_friction import native_tangents
    engine=ComFreeEngine(root,forge)
    from forceaware.targets import make_window
    native=mujoco.MjModel.from_xml_path(str(root/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    native.opt.timestep=.002
    object_body=engine.index.object_body
    mesh=native.geom_dataid[mujoco.mj_name2id(native,mujoco.mjtObj.mjOBJ_GEOM,'obj_collision')]
    start=native.mesh_vertadr[mesh];count=native.mesh_vertnum[mesh]
    verts=native.mesh_vert[start:start+count].astype(float)
    radius=float(np.max(np.linalg.norm(verts,axis=1)));center=np.zeros(3)
    for executor in ['mujoco','comfree']:
        for method in METHODS:
            loaded=read_case(root,executor,method,baseline)
            if loaded is None:continue
            case,a=loaded;q=a['qpos_traj'];v=a['qvel_traj'];ctrl=a['ctrl_traj']
            split=np.load(case/'contact_admm.npz') if (case/'contact_admm.npz').exists() else None
            split_index={}
            if split is not None:
                indices=split['window_indices'] if 'window_indices' in split else np.arange(len(split['mask']))
                split_index={int(window):row for row,window in enumerate(indices)}
                for row,window in enumerate(indices):
                    # Preserve workflow's Python-int arithmetic. NumPy int64
                    # promotes its float32 time grid, changing nearest-frame
                    # rounding at half frames under NumPy 2.
                    target=make_window(engine.sequence,executed_steps=3*int(window),horizon=5,
                        time_grid=engine.cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
                    assert np.array_equal(split['mask'][row],target.contact_mask*target.contact_weight)
                    assert np.array_equal(split['template'][row],target.contact_position_object)
                    assert np.array_equal(split['template_hand_local'][row],target.query_local)
                logs=[json.loads(line) for line in (case/'admm_windows.jsonl').read_text().splitlines()]
                relative=[]
                for window,log in enumerate(logs):
                    score=log.get('selection_merit',log.get('augmented_loss'))
                    relative.append(abs(score-log['FA_loss'])/max(abs(score),1.))
                assert max(relative)<2e-5,max(relative)
                results[executor+'/'+method]['snapshot_audit']=dict(
                    original_template_points_and_masks_unchanged=True,
                    selection_merit_max_relative_error=max(relative))
            data=mujoco.MjData(native);records=[]
            data.qpos[:]=q[0];data.qvel[:]=v[0]
            for window,i in enumerate(range(3,len(q),3)):
                time=i*.01
                # ComFree's last executed force uses pre-step geometry at
                # t-10 ms. MuJoCo replay exports post-step geometry at t.
                near_index=1 if executor=='comfree' else 2
                target=make_window(engine.sequence,executed_steps=3*window,horizon=5,
                    time_grid=engine.cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
                active=target.contact_mask[near_index]>0
                tb=target.query_body[near_index,active]
                template=target.contact_position_object[near_index,active]
                contacts=[];reconstruction=0.;mujoco_replay_error=0.
                if executor=='comfree':
                    qp,vp,b=engine.step(q[i-1],v[i-1],ctrl[i-1],dt=.01)
                    reconstruction=float(np.max(np.abs(qp[0]-q[i])))
                    items=engine.contacts(b)
                    pose=q[i-1,engine.oq:engine.oq+7]
                    for c in items:
                        if c['body']==0:continue
                        contacts.append(c)
                else:
                    # Replay saved controls with the executor's exact step /
                    # forward schedule, retaining qacc_warmstart and MjData.
                    # Never rerun optimization or reset to rounded snapshots.
                    for dense in range(i-3,i):
                        data.ctrl[engine.index.hand_ctrl]=ctrl[dense]
                        data.qfrc_applied[:]=0.
                        for _ in range(5):mujoco.mj_step(native,data)
                        mujoco.mj_forward(native,data)
                        error=max(float(np.max(np.abs(data.qpos.astype(np.float32)-q[dense+1]))),
                                  float(np.max(np.abs(data.qvel.astype(np.float32)-v[dense+1]))))
                        mujoco_replay_error=max(mujoco_replay_error,error)
                        if error!=0.:
                            raise RuntimeError(f'MuJoCo control replay differs at step {dense+1}: {error}')
                    pose=data.qpos[engine.oq:engine.oq+7].copy()
                    for k,c in enumerate(data.contact):
                        bodies=native.geom_bodyid[[c.geom1,c.geom2]]
                        if object_body not in bodies:continue
                        other=int(bodies[1] if bodies[0]==object_body else bodies[0])
                        if other==0:continue
                        sign=-1. if bodies[0]==object_body else 1.
                        f=np.zeros(6);mujoco.mj_contactForce(native,data,k,f)
                        basis=c.frame.reshape(3,3);force=sign*basis.T@f[:3]
                        if c.dist>.0001 and np.linalg.norm(force)<.001:continue
                        normal=sign*basis[0]
                        contacts.append(dict(point=c.pos.copy(),normal=normal,
                            tangents=native_tangents(normal,basis[1:]),force=force,body=other,gap=float(c.dist)))
                r=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
                points=np.array([(c['point']-pose[:3])@r for c in contacts]).reshape(-1,3)
                normals=np.array([c['normal']@r for c in contacts]).reshape(-1,3)
                tangents=np.array([c['tangents']@r for c in contacts]).reshape(-1,2,3)
                loaded_mask=np.array([np.dot(c['force'],c['normal'])>.01 for c in contacts],dtype=bool)
                epsilon=gws_margin(points,normals,tangents,1.,center,radius)
                epsilon_loaded=gws_margin(points[loaded_mask],normals[loaded_mask],tangents[loaded_mask],1.,center,radius)
                errors=[];made=0
                for body,target in zip(tb,template):
                    cp=[points[k] for k,c in enumerate(contacts) if c['body']==body]
                    if cp:
                        errors.append(float(np.linalg.norm(np.asarray(cp)-target,axis=1).min())*1000)
                        made+=1
                admm_errors=[];admm_made=0;admm_count=0
                has_goal=window in split_index
                if has_goal:
                    row=split_index[window]
                    sm=split['mask'][row,near_index]>0
                    admm_count=int(sm.sum())
                    for body,target in zip(split['body'][row,near_index,sm],split['z'][row,near_index,sm]):
                        cp=[points[k] for k,c in enumerate(contacts) if c['body']==body]
                        if cp:
                            distance=float(np.linalg.norm(np.asarray(cp)-target,axis=1).min())*1000
                            admm_errors.append(distance);admm_made+=distance<2.
                record=dict(time_s=time,contacts=len(contacts),loaded_contacts=int(loaded_mask.sum()),
                    epsilon=epsilon,loaded_epsilon=epsilon_loaded,
                    template_contacts=len(tb),template_body_coverage=made/len(tb) if len(tb) else None,
                    template_region_error_mm=float(np.mean(errors)) if errors else None,
                    admm_target_count=admm_count,admm_target_made_within_2mm=admm_made,
                    admm_region_error_mm=float(np.mean(admm_errors)) if admm_errors else None,
                    hand_vertical_N=float(sum(c['force'][2] for c in contacts)),
                    total_normal_N=float(sum(max(np.dot(c['force'],c['normal']),0.) for c in contacts)),
                    comfree_snapshot_step_max_state_error=reconstruction,
                    mujoco_control_replay_max_state_error=mujoco_replay_error,
                    contact_geometry_time_s=(i-1)*.01 if executor=='comfree' else time)
                record['actual_contacts']=[dict(body=int(c['body']),point_object=points[k].tolist(),
                    normal_object=normals[k].tolist(),force_world=np.asarray(c['force']).tolist(),
                    gap_m=float(c['gap'])) for k,c in enumerate(contacts)]
                record['template_points_object']=template.tolist()
                record['template_bodies']=tb.tolist()
                if has_goal:
                    record['admm_points_object']=split['z'][row,near_index,sm].tolist()
                    record['admm_bodies']=split['body'][row,near_index,sm].tolist()
                records.append(record)
            key=executor+'/'+method
            loaded_records=[s for s in records if s['template_contacts']>0]
            def mean(name):
                values=[s[name] for s in loaded_records if s[name] is not None]
                return float(np.mean(values)) if values else None
            results[key]['actual_contact']=dict(active_template_samples=len(loaded_records),
                mean_contacts=mean('contacts'),mean_loaded_contacts=mean('loaded_contacts'),
                mean_gws_epsilon=mean('epsilon'),mean_loaded_gws_epsilon=mean('loaded_epsilon'),
                force_closure_fraction=float(np.mean([s['epsilon']>1e-7 for s in loaded_records])),
                loaded_force_closure_fraction=float(np.mean([s['loaded_epsilon']>1e-7 for s in loaded_records])),
                template_body_coverage=mean('template_body_coverage'),template_region_error_mm=mean('template_region_error_mm'),
                admm_region_error_mm=mean('admm_region_error_mm'),
                auxiliary_target_window_coverage_fraction=len(split_index)/len(records),
                admm_realized_within_2mm_fraction=sum(s['admm_target_made_within_2mm'] for s in records)/max(sum(s['admm_target_count'] for s in records),1),
                mean_total_normal_N=mean('total_normal_N'),mean_hand_vertical_N=mean('hand_vertical_N'),
                comfree_snapshot_step_max_state_error=max(s['comfree_snapshot_step_max_state_error'] for s in records),
                mujoco_control_replay_max_state_error=max(s['mujoco_control_replay_max_state_error'] for s in records))
            # Put analysis next to the experiment, never edit the archived baseline.
            dest=root/executor/method;dest.mkdir(parents=True,exist_ok=True)
            (dest/'actual_contact_metrics.json').write_text(json.dumps(records,indent=2))
            print(key,results[key]['actual_contact'],flush=True)


def write_report(root,results,curves,partial=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for row,executor in enumerate(['mujoco','comfree']):
        for method in METHODS:
            curve=curves.get(executor+'/'+method)
            if curve is None:continue
            for col in range(2):axes[row,col].plot(curve[:,0],curve[:,col+1],label=method)
        for col in range(2):
            axes[row,col].axvline(71/30,color='grey',ls='--',lw=.8)
            axes[row,col].set_title(executor+(' position (mm)' if col==0 else ' rotation (deg)'))
            axes[row,col].set_xlabel('Simulation time (s)');axes[row,col].grid(alpha=.2)
        axes[row,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'tracking_errors.png',dpi=160);plt.close(fig)
    (root/('partial_metrics.json' if partial else 'metrics.json')).write_text(json.dumps(results,indent=2))
    lines=['# FA + 接触 ADMM 对照','',
        '同一条 72 帧 DexYCB SugarBox 序列、Wuji 右手、同一 DexForge CA 初值。每组闭环执行 5.4 s；前 2.3667 s 为运动，其后保持。',
        '位置为欧氏误差，旋转为四元数最短转角；均按仿真时间插值示范、每 30 ms 评分。表中为平均误差 mm / °。','']
    for executor in ['mujoco','comfree']:
        lines+=['## '+executor,'','| 方法 | 运动段 | 保持段 | 全程 | 末帧 |','|---|---:|---:|---:|---:|']
        for method,label in METHODS.items():
            r=results.get(executor+'/'+method)
            if r is None:continue
            def cell(segment):return f"{r[segment]['position_mean_mm']:.3f} / {r[segment]['rotation_mean_deg']:.3f}"
            lines.append(f"| {label}{'' if r['complete'] else '（部分）'} | {cell('motion')} | {cell('hold') if r['hold'] else '—'} | {cell('all')} | {r['all']['last_position_mm']:.3f} / {r['all']['last_rotation_deg']:.3f} |")
        lines+=['']
    lines+=['## 协议','',
        '- 三组全部使用同一 CA 手姿态参考、完整原始 MANO 19 槽接触模板和掩码；每组重新闭环运行，不复用此前采用 CA 后处理接触锚点的基线。',
        '- 全部使用原版 FA 代价公式和权重、ComFree 硬接触预测、300 ms 时域、8 个起点。原 FA 为 40 次 Adam；另两组为最多 40 次独立 L-BFGS hand 更新，线搜索评价量另计。',
        '- ADMM 为 min FA(u,c)+I_surface(z), c=z。MANO 仅提供初值；c 成为 FA 查询点项中的物体侧目标变量。4 轮交替更新后做表面可行化和 hand 优化，40 次 hand 预算分为 5 组各 8 次。',
        '- rho=1 只用于物体点与其表面副本之间的增广项；最终评分严格为纯 FA 代价。不增加 hand 接触点与物体点必须重合的约束。',
        '- 不使用力平衡 QP、Lambda 或力闭合筛选。手侧 body-local 查询点和接触掩码不变。',
        '- 保持固定 MANO 模板的 L-BFGS 组用于分离求解器替换和接触点自由度的影响。不同硬件及 GPU 共享负载下，墙钟时间不代表算法速度对比。',
        '- 几何 GWS 使用共同 mu=1、4 边摩擦棱锥、单位力、物体半径归一化力矩；仅诊断，不参与优化或控制。MuJoCo 按原子步顺序回放已保存控制并逐步核验状态完全一致；ComFree 由保存状态做一个实际 10 ms 步重建，报告状态偏差。接触模板按各引擎实际碰撞几何的时间对齐。',
        '- 这是无静态力平衡 QP 的动态改写，不是原版静态 GraspADMM。单序列不能代表跨任务成功率。','']
    probe_path=root/'frozen_budget_probes.json'
    if probe_path.exists():
        probes=json.loads(probe_path.read_text())
        lines+=['## 相同冻结状态下的求解预算检查','',
            '每个时间点使用原版 ComFree 基线的同一实际状态、warm start、控制历史和随机种子。以下是一个 300 ms 窗口的末端位置误差，不是闭环误差。','',
            '| 状态时间 | FA40 | ADMM40 rho20 | FA160 | ADMM160 rho20 | ADMM160 rho80 |',
            '|---:|---:|---:|---:|---:|---:|']
        for t in [1.2,1.8,3.6]:
            row=[]
            for name in ['FA40','ADMM40_rho20','FA160','ADMM160_rho20','ADMM160_rho80']:
                found=[r for r in probes if r['variant']==name and r['time_s']==t]
                row.append(f"{found[0]['terminal_position_error_mm']:.3f}" if found else '未完成')
            lines.append('| '+str(t)+' | '+' | '.join(row)+' |')
        lines+=['']
    if (root/'analysis.md').exists():lines.append((root/'analysis.md').read_text())
    (root/('partial_report.md' if partial else 'report.md')).write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--baseline',type=Path);p.add_argument('--forge',type=Path)
    p.add_argument('--partial',action='store_true');p.add_argument('--contacts',action='store_true');a=p.parse_args()
    baseline=a.baseline or a.experiment
    results,curves=trajectory_report(a.experiment,baseline,a.partial)
    if a.contacts:contact_report(a.experiment,baseline,a.forge,results)
    write_report(a.experiment,results,curves,a.partial)
    for key,r in results.items():print(key,r['samples'],r['all']['position_mean_mm'],r['all']['rotation_mean_deg'])


if __name__=='__main__':main()
