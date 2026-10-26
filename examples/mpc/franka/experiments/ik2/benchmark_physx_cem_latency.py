"""Synchronized lambda_optimizer latency on a frozen Franka initial-state query.

Reports actual choose_contact_points + optimize_control_input wall time, as
well as backend time. No rendering, MPC, physics execution or startup in the
reported steady-state samples. Runs from the repository root with acc Python.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
os.environ['SCSP_PLANNER_ONLY'] = '1'
from planning.cpu_runtime import configure_compute_threads
configure_compute_threads()
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--benchmark-output', required=True)
    parser.add_argument('--benchmark-repeats', type=int, default=50)
    parser.add_argument('--benchmark-warmup', type=int, default=10)
    custom, remaining = parser.parse_known_args()
    if custom.benchmark_repeats < 1 or custom.benchmark_warmup < 0:
        parser.error('invalid benchmark repeat/warmup counts')
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args, _apply_dywa_physics_to_param
    from examples.mpc.franka.runtime.params import ExplicitMPCParams, build_lambda_optimizer
    from examples.mpc.franka.runtime.contact import table_jac_mat_env, gravity_wrench_object_frame
    from planning.object_pose_cost import ObjectCostStages
    from planning.rollout_timing import distribution
    args = _add_rollout_policy_args(argparse.ArgumentParser()).parse_args(remaining)
    output = Path(custom.benchmark_output)
    output.mkdir(parents=True, exist_ok=False)
    init_start = time.perf_counter()
    param = ExplicitMPCParams(args, rand_seed=args.trial_start, build_optimizer=False)
    _apply_dywa_physics_to_param(param)
    opt = build_lambda_optimizer(param, args)
    initialization_ms = 1000*(time.perf_counter()-init_start)
    native = opt.gpu_cem
    try:
        world_pose = np.asarray(param.init_obj_qpos_)
        rotation = Rotation.from_quat(world_pose[[4, 5, 6, 3]])
        matrix = rotation.as_matrix()
        pose = np.array([0., 0., 0., 1., 0., 0., 0.])
        target_q = (rotation.inv()*Rotation.from_quat(param.target_q_[[1, 2, 3, 0]])).as_quat()
        target = np.r_[matrix.T@(param.target_p_-world_pose[:3]), target_q[[3, 0, 1, 2]]]
        support = matrix.T@(np.array([0., 0., param.table_height])-world_pose[:3])
        up = matrix.T@np.array([0., 0., 1.])
        if hasattr(native, 'set_support_plane'):
            native.set_support_plane(support, up)
        gravity = gravity_wrench_object_frame(param.gravity_[:3], param.sim_obj_mass_, matrix)
        jac = table_jac_mat_env(world_pose[:3], world_pose[3:], param.table_height,
                               nv=9, mu=param.mu_object_, max_ncon=opt.max_contacts)
        opt.update_Jacobian(jac)
        stages = ObjectCostStages(args, param.object_up_axis_)
        opt.set_object_cost_config(stages.config(matrix))
        ids = opt.get_availble_point_idx(world_pose[:3], matrix, param.target_p_,
            args.ground_height_threshold, heading_filter=False, floor_z=param.table_height)
        ids = opt.filter_rankable_indices(ids)
        if not len(ids):
            raise RuntimeError('No rankable initial-state candidates')
        print(json.dumps(dict(initialization_ms=initialization_ms, samples=len(opt.sample_point),
            visible_candidates=len(ids), metadata=getattr(native, 'metadata', {}))), flush=True)
        rows = []
        for repeat in range(custom.benchmark_warmup+custom.benchmark_repeats):
            started = time.perf_counter()
            opt.choose_contact_points(target, pose, gravity, ids, force_required=True)
            ranked = time.perf_counter()
            rank_backend = dict(native.last_timing)
            idx = int(opt.last_best_idx)
            _, _, predicted, cost, info = opt.optimize_control_input(
                target, pose, gravity, sample_idx=idx)
            elapsed = 1000*(time.perf_counter()-started)
            if info['solver_failed'] or not np.isfinite(np.r_[predicted, cost]).all():
                raise RuntimeError('lambda_optimizer solve failed during benchmark')
            force = np.asarray(info['control_input'])
            if np.linalg.norm(force) > opt.max_contact_force+1e-5 or abs(force[1])+abs(force[2]) > opt.mu_arm_obj*force[0]+1e-5:
                raise RuntimeError('Optimizer returned infeasible force')
            row = dict(repeat=repeat, warmup=repeat < custom.benchmark_warmup,
                lambda_ms=elapsed, rank_ms=1000*(ranked-started),
                point_ms=elapsed-1000*(ranked-started), rank_backend=rank_backend,
                point_backend=dict(native.last_timing), selected_idx=idx,
                cost=cost, force=force.tolist(), predicted_pose=np.asarray(predicted).tolist())
            rows.append(row)
            if (repeat+1) % 10 == 0:
                print('Measured %d/%d; latest lambda %.3f ms' % (
                    repeat+1, custom.benchmark_warmup+custom.benchmark_repeats, elapsed), flush=True)
        steady = [r for r in rows if not r['warmup']]
        source_names = ['planning/contact_force_physx_cem.py', 'planning/contact_force_physx_native.py',
            'planning/contact_force_physx_worker.py', 'planning/mlqp_point.py',
            'planning/point_force_gpu.py', 'examples/mpc/franka/runtime/params.py',
            str(Path(__file__).resolve().relative_to(ROOT))]
        summary = dict(arguments=vars(args), repeats=custom.benchmark_repeats,
            warmup=custom.benchmark_warmup, initialization_ms=initialization_ms,
            sample_count=len(opt.sample_point), visible_candidates=len(ids),
            evaluations_per_rank=len(ids)*args.population_size*args.population_iterations,
            metadata=getattr(native, 'metadata', {}),
            gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader'], text=True).strip(),
            timings_ms={k: distribution([r[k] for r in steady]) for k in ('lambda_ms', 'rank_ms', 'point_ms')},
            definitions=dict(lambda_ms='choose_contact_points + optimize_control_input at selected sample; synchronized GPU execution and IPC included; frozen initial query, no full closed-loop rollout',
                initialization_ms='parameter/mesh/worker/scene creation, compilation and native warm-up; excluded from steady-state',
                gpu_ms='backend synchronized CEM + native steps + per-candidate download; includes Python simulation calls'),
            source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_names})
        query = dict(current=pose.tolist(), target=target.tolist(), world_pose=world_pose.tolist(),
            gravity=gravity.tolist(), support_point=support.tolist(), support_normal=up.tolist(),
            candidates=np.asarray(ids).tolist(), object_config=opt.object_cost_config().tolist(),
            jacobian=jac.tolist(), mass=param.sim_obj_mass_, inertia=param.sim_obj_inertia_diag_.tolist())
        (output/'query.json').write_text(json.dumps(query, indent=2)+'\n')
        (output/'cycles.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
        (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        print(json.dumps(summary['timings_ms'], indent=2))
    finally:
        if hasattr(native, 'close'):
            native.close()


if __name__ == '__main__':
    main()
