#!/usr/bin/env python3
"""Isolated, synchronized full-CSO timing on the saved paired observations."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['SCSP_PLANNER_ONLY']='1'
import argparse
from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
import warp as wp
from planning.point_force_success import SuccessPlanner
from planning.point_force_benchmark import scene,surface_points,write_json
from planning.object_pose_cost import ObjectCostStages,target_up_axis
from planning.rollout_timing import distribution


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('cohort',type=Path);p.add_argument('--repeats',type=int,default=30)
    args=p.parse_args();cohort=args.cohort.resolve()
    protocol=json.loads((cohort/'protocol.json').read_text())
    saved=json.loads((cohort/'initial_states.json').read_text())
    cfg=Namespace(**protocol['args']);cfg.solver='cem';cfg.async_planner=False;cfg.viewer=False
    records=[];all_external=[];all_internal=[];all_cold=[]
    wp.init();device=wp.get_device(cfg.device)
    for obj in protocol['objects']:
        cfg.obj=obj
        for seed in protocol['seeds']:
            m,d,target,_=scene(cfg,obj,seed);points,frames=surface_points(m,d,cfg.sample_num,seed)
            expected=saved[f'{obj}:{seed}']
            np.testing.assert_array_equal(d.qpos,expected['pose']);np.testing.assert_array_equal(points,expected['points'])
            planner=SuccessPlanner(m,points,frames,target,cfg,'scm_corrected',seed)
            stages=ObjectCostStages(cfg,target_up_axis(target[3:]))
            for condition in ('initial','penetrating_moving_probe'):
                pose=d.qpos.copy();velocity=d.qvel.copy()
                if condition!='initial':
                    pose[2]-=.001;velocity=np.array([.03,-.02,.01,.1,-.2,.15])
                snapshot=(pose,velocity,0.,0);config=stages.config()
                validation=planner.validate(snapshot,config)
                external=[];internal=[];evaluations=[];generations=[]
                wp.synchronize_device(device)
                cold_started=time.perf_counter_ns()
                planner.solve(snapshot,config)
                wp.synchronize_device(device)
                cold=(time.perf_counter_ns()-cold_started)/1e6
                all_cold.append(cold)
                for repetition in range(args.repeats):
                    wp.synchronize_device(device)
                    started=time.perf_counter_ns()
                    result=planner.solve(snapshot,config)
                    wp.synchronize_device(device)
                    duration=(time.perf_counter_ns()-started)/1e6
                    expected_count=int(np.count_nonzero(planner.valid))*cfg.population_size*cfg.population_iterations
                    assert result['evaluations']==expected_count,result
                    assert result['search_iterations']==cfg.population_iterations,result
                    assert result['solver_backend']=='cuda-cem' and result['gpu_parallel_prediction']
                    assert np.isfinite(result['cost'])
                    external.append(duration);internal.append(result['planning_time_ms'])
                    evaluations.append(result['evaluations']);generations.append(result['search_iterations'])
                all_external.extend(external);all_internal.extend(internal)
                records.append(dict(object=obj,seed=seed,condition=condition,validation=validation,
                    candidate_points=len(points),valid_points=int(np.count_nonzero(planner.valid)),
                    contact_constraint_rows=len(planner.problem['gap']),cold_full_plan_ms=cold,
                    external_synchronized_ms=external,internal_recorded_ms=internal,
                    candidate_evaluations_per_plan=evaluations[0],generations=generations[0]))
            print(f'CEM timing {obj} seed={seed} complete',flush=True)
    report=dict(configuration='scm_cem',observations=len(records),repetitions_per_observation=args.repeats,
        measured_calls=len(all_external),concurrent_benchmark_jobs=1,
        external_synchronized_full_planning_ms=distribution(all_external),
        internal_recorded_full_planning_ms=distribution(all_internal),
        first_full_plan_per_observation_ms=distribution(all_cold),
        scope='CPU observation preparation, collision/Jacobian/mass, transfers, all 12 CEM generations, synchronized CUDA execution, result download and selection; physics stepping excluded.',
        setup_excluded='Scene/planner construction, module compilation and model-parity validation; first graph capture is recorded separately.',
        sampling=dict(surface_candidates=cfg.sample_num,population=cfg.population_size,generations=cfg.population_iterations,
                      maximum_candidate_evaluations=cfg.sample_num*cfg.population_size*cfg.population_iterations),
        model='softplus_nonnegative_gap_v2, one frozen-contact six-DoF prediction step',
        records=records,verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(cohort/'cem_isolated_timing.json',report)
    ext=report['external_synchronized_full_planning_ms'];inside=report['internal_recorded_full_planning_ms']
    lines=['# CEM 独立计时核验','',f'{len(records)} 个观测状态，每个重复 {args.repeats} 次，共 {len(all_external)} 次完整规划调用。',
        '',f'- 外部 GPU 前后同步计时：{ext["mean"]:.3f} ± {ext["std"]:.3f} ms；P95 {ext["p95"]:.3f} ms。',
        f'- 原程序内部计时：{inside["mean"]:.3f} ± {inside["std"]:.3f} ms；P95 {inside["p95"]:.3f} ms。',
        '- 70 个候选点、每点 24 个样本、12 代；每次都检查实际代数和评价次数，没有减少预算。',
        '- 包含观测准备、碰撞/Jacobian/质量计算、数据传输、全部 CEM 迭代、GPU 同步及结果下载与选择；不包含物理执行。',
        '- 场景构建、模块编译与公式验证在外；首次完整规划单独保留。',
        '- CEM 使用 CUDA 图批量计算；原成功率测试的 LM 是 NumPy CPU 批量有限差分实现。两者的耗时差不能证明 CEM 优于 GPU batch LM。']
    (cohort/'cem_isolated_timing.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__=='__main__':main()
