"""Paired Isaac evaluation against an isolated developer checkout.

The reference gets only the current acados loader/API compatibility shim,
in memory. Its controller, candidate initialization and constraints are kept.
Both policies use the reference's existing synchronous Isaac bus.
"""
import argparse
import concurrent.futures
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
OBJECTS = ('foam_brick', 'stanford_bunny2', 'elephant', 'piggy_bank',
           'teapot', 'mug', 'rubber_duck')


def worker(args):
    root = Path(args.worker_root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.pop('SCSP_PLANNER_ONLY', None)
    from isaacgym import gymapi, gymtorch  # Must precede torch imports.
    import numpy as np
    import planning.acados_env as reference_loader
    spec = importlib.util.spec_from_file_location('comparison_acados_loader', ROOT/'planning/acados_env.py')
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    reference_loader.build_cached_solver = loader.build_cached_solver
    reference_loader.quiet_acados_stderr = loader.quiet_acados_stderr
    import planning.mpc_explicit as planning
    # Reference checkouts can predate the runtime package extraction.
    if (root/'examples/mpc/franka/runtime/mpc.py').is_file():
        policy = importlib.import_module('examples.mpc.franka.runtime.mpc')
        physics = importlib.import_module('examples.mpc.franka.runtime.isaac')
    else:
        policy = importlib.import_module('examples.mpc.franka.ik2.test_mpc_isaac')
        physics = importlib.import_module('examples.mpc.franka.ik2.run')
    SynchronousMPCBus = physics.SynchronousMPCBus
    options = policy._add_rollout_policy_args(argparse.ArgumentParser()).parse_args([
        '--obj', args.object, '--headless', '--sync-planner', '--solver-backend', 'acados',
        '--trial-start', str(args.trial_start), '--trial_num', str(args.trials),
        '--max_rollout_length', str(args.steps), '--ranking-metrics-jsonl', args.metrics])
    options.rollout = True
    records = {}
    build = planning._build_mpc_planner_runtime

    def inspected_build(init, *rest, **kwargs):
        result = build(init, *rest, **kwargs)
        _, param, mpc, _ = result
        record = dict(trial=init['trial_count'], object=args.object,
                      planner_backend=mpc.planner_solver_,
                      contact_backend=param.lambda_optimizer.solver,
                      planner_init_error=str(mpc._acados_init_error or ''),
                      contact_init_error=str(param.lambda_optimizer.acados_error or ''))
        record['planner'] = {name: np.asarray(getattr(param, name)).tolist() for name in (
            'init_obj_qpos_', 'init_fingertip_pos_', 'target_p_', 'target_q_', 'Q',
            'sim_obj_mass_', 'sim_obj_inertia_diag_', 'gravity_', 'mpc_q_lb_', 'mpc_q_ub_')}
        record['samples'] = {name: np.asarray(getattr(param.lambda_optimizer, name)).tolist()
                             for name in ('sample_point', 'normal')}
        records[init['trial_count']] = record
        return result

    planning._build_mpc_planner_runtime = inspected_build
    if hasattr(policy, '_build_mpc_planner_runtime'):
        policy._build_mpc_planner_runtime = inspected_build

    class Bus(SynchronousMPCBus):
        def wait_obs(self, *a, **kw):
            observation = super().wait_obs(*a, **kw)
            trial = observation.get('trial')
            if observation.get('seq') == 0 and trial in records:
                record = records[trial]
                param = self.env.param_
                record['physics'] = {name: np.asarray(getattr(param, name)).tolist() for name in (
                    'init_obj_qpos_', 'init_fingertip_pos_', 'target_p_', 'target_q_',
                    'sim_obj_mass_', 'sim_obj_inertia_diag_', 'table_friction_', 'object_friction_')}
                record['observed_start'] = np.asarray(observation['policy_q']).tolist()
                with open(args.metrics+'.setup.jsonl', 'a') as output:
                    output.write(json.dumps(record, allow_nan=False)+'\n')
            return observation

    bus = Bus(options)
    try:
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
            policy.run_mpc_planner(bus, options)
    finally:
        bus.close()


def read_trials(path, expected):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if sorted(row['trial'] for row in rows) != expected:
        raise RuntimeError('Incomplete/duplicate trials in '+str(path))
    if any(row.get('aborted') for row in rows):
        raise RuntimeError('Aborted trials in '+str(path))
    return {row['trial']: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/developer_comparison')
    parser.add_argument('--objects', nargs='+', choices=OBJECTS, default=OBJECTS)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--trial-start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--worker-root')
    parser.add_argument('--object')
    parser.add_argument('--metrics')
    args = parser.parse_args()
    if args.worker_root:
        worker(args)
        return
    if args.reference_root is None or not (args.reference_root/'planning/mpc_explicit.py').is_file():
        parser.error('--reference-root must be an isolated developer checkout')
    if min(args.trials, args.steps, args.workers) <= 0 or args.trial_start < 0:
        parser.error('trials, steps and workers must be positive; trial-start must be nonnegative')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = dict(current=ROOT, developer=args.reference_root.resolve())
    commit = subprocess.check_output(['git', '-C', str(roots['developer']), 'rev-parse', 'HEAD'], text=True).strip()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def run(job):
        branch, obj = job
        path = args.output_dir/(branch+'_'+obj+'.jsonl')
        # Avoid merging previous runs into a denominator. Keep old outputs
        # reviewable by requiring a new output directory for a new experiment.
        if path.exists():
            raise FileExistsError('Choose a fresh output directory: '+str(path))
        command = [args.python, '-u', str(Path(__file__).resolve()), '--worker-root', str(roots[branch]),
                   '--object', obj, '--trials', str(args.trials), '--steps', str(args.steps),
                   '--trial-start', str(args.trial_start), '--metrics', str(path.resolve())]
        with path.with_suffix('.log').open('w') as output:
            process = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError('Evaluation failed; see '+str(path.with_suffix('.log')))
        print(branch+' '+obj+' complete', flush=True)

    jobs = [(branch, obj) for obj in args.objects for branch in roots]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, jobs))
    expected = list(range(args.trial_start, args.trial_start+args.trials))
    result = dict(reference_commit=commit, physics_backend='isaac', synchronous=True,
                  solver='acados', loader_compatibility_shim=True, steps=args.steps,
                  trial_ids=expected, objects={})
    for obj in args.objects:
        rows = {branch: read_trials(args.output_dir/(branch+'_'+obj+'.jsonl'), expected) for branch in roots}
        setups = {branch: read_trials(args.output_dir/(branch+'_'+obj+'.jsonl.setup.jsonl'), expected) for branch in roots}
        comparisons = {}
        for trial in expected:
            a, b = setups['current'][trial], setups['developer'][trial]
            if any(record['planner_init_error'] or record['contact_init_error'] for record in (a, b)):
                raise RuntimeError('acados initialization failed for '+obj+'; inspect setup logs')
            comparisons[trial] = {key: a[key] == b[key] for key in ('planner', 'physics', 'samples', 'observed_start')}
        result['objects'][obj] = dict(
            successes={branch: sum(row['success'] for row in values.values()) for branch, values in rows.items()},
            changed_trials=[trial for trial in expected if rows['current'][trial]['success'] != rows['developer'][trial]['success']],
            setup_matches=comparisons)
    (args.output_dir/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
