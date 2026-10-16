"""Full seven-object async soft-SCM/CUDA batched-LM benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def worker(args):
    from examples.mpc.fingertips.test import test_scm_mujoco as mod
    options = ['--solver', 'batched-lm', '--async-planner',
        '--obj', args.object, '--steps', str(args.steps), '--contact-model', args.contact_model]
    if args.model_param is not None:
        options += ['--model_param', str(args.model_param)]
    cfg = mod.build_parser().parse_args(options)
    rows, error = [], None
    try:
        rows = mod._run_rollout(cfg, 'surrogate', args.object, seed=args.seed,
                               mass=.01, steps=args.steps)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        rows = getattr(exc, 'rows', rows)
        error = '%s: %s' % (type(exc).__name__, exc)
    last = rows[-1] if rows else {}
    times = [r['planning_time_ms'] for r in rows if r.get('planning_updated')]
    result = dict(object=args.object, seed=args.seed, contact_model=args.contact_model,
        model_param=cfg.model_param, success=bool(last.get('success')),
        ticks=len(rows), planning_cycles=len(times),
        planning_mean_ms=sum(times)/len(times) if times else None,
        simulation_time_s=last.get('simulation_time_s'),
        elapsed_wall_time_s=last.get('elapsed_wall_time_s'),
        final_pos=last.get('goal_pos'), final_quat=last.get('goal_quat'),
        solver_backends=sorted({r['solver_backend'] for r in rows}), error=error)
    path = args.output / ('trial_%s_%03d.json' % (args.object, args.seed))
    mod._write_rows(str(path), rows, dict(args=vars(cfg), trial_summary=result))
    print('RESULT ' + json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--contact-model', choices=['soft-scm', 'comfree'], default='soft-scm')
    p.add_argument('--model-param', type=float, default=None,
                   help='Override legacy contact stiffness; ComFree native default is 0.2.')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--object')
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    from examples.mpc.fingertips.test.test_scm_mujoco import DEFAULT_OBJECTS
    files = ['examples/mpc/fingertips/test/test_scm_mujoco.py',
        'planning/direct_contact_force.py', 'planning/direct_force_async.py',
        'planning/scm_batched_lm.py', 'planning/mlqp_point.py',
        'models/soft_scm_contact.py', 'models/comfree_contact.py',
        'planning/object_pose_cost.py', 'planning/contact_limits.py',
        'examples/mpc/franka/ik2/params.py']
    hashes = {}
    for name in files:
        data = (ROOT/name).read_bytes()
        snapshot = args.output/'source'/name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    write(args.output/'protocol.json', dict(objects=DEFAULT_OBJECTS, seeds=list(range(args.trials)),
        total_trials=7*args.trials, max_physics_ticks=args.steps, policy_dt=.02,
        async_planner=True, realtime=True, jobs=args.jobs, contact_model=args.contact_model,
        solver='batched-lm', lm_iterations=20, mass=.01, force_apply_duration=.0005,
        contact_impratio=10., contact_noslip_iterations=3,
        object_linear_damping=0., object_angular_damping=.5,
        model_param=7. if args.model_param is None else args.model_param, source_sha256=hashes))
    def run(obj, seed):
        cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
            '--output', str(args.output), '--object', obj, '--seed', str(seed),
            '--steps', str(args.steps), '--contact-model', args.contact_model]
        if args.model_param is not None:
            cmd += ['--model-param', str(args.model_param)]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', MUJOCO_HEADLESS='1')
        with (args.output/('%s_%03d.log' % (obj, seed))).open('w') as log:
            code = subprocess.call(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        path = args.output/('trial_%s_%03d.json' % (obj, seed))
        if path.exists():
            return json.loads(path.read_text())['metadata']['trial_summary']
        return dict(object=obj, seed=seed, success=False, error='worker exit %d without results' % code)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [pool.submit(run, obj, seed) for seed in range(args.trials) for obj in DEFAULT_OBJECTS]
        for task in as_completed(tasks):
            result = task.result()
            results.append(result)
            summary = dict(completed=len(results), total=7*args.trials,
                successes=sum(r['success'] for r in results), trials=results)
            summary['success_rate'] = summary['successes']/len(results)
            write(args.output/'summary.json', summary)
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
