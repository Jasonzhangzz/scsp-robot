"""Measure the ordinary Isaac MPC rollout without changing its control policy.

Run with the Isaac Python environment, using the same arguments as
ik2/test_mpc_isaac.py plus --latency-output DIR and --latency-warmup N.
This benchmark supports one synchronous environment. Timers and wrappers are
installed only in this process; the production modules on disk are unchanged.
"""
import argparse
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from planning.cpu_runtime import configure_compute_threads
configure_compute_threads()


class LatencyProbe:
    def __init__(self):
        self.rows = []
        self.active = None
        self.lambda_depth = 0

    def mark(self, name):
        if self.active is None:
            return
        now = time.perf_counter_ns()
        if name == 'verify_start':
            self.active['_verify_start'] = now
        else:
            self.active['verify_ms'] = (now - self.active.pop('_verify_start')) / 1e6

    def wrap_lambda(self, original, name):
        @functools.wraps(original)
        def measured(*args, **kwargs):
            if self.active is None:
                return original(*args, **kwargs)
            outer = self.lambda_depth == 0
            self.lambda_depth += 1
            started = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                elapsed = (time.perf_counter_ns() - started) / 1e6
                self.lambda_depth -= 1
                if outer:
                    self.active['lambda_ms'] += elapsed
                if name == 'choose_contact_points':
                    self.active['lambda_rank_ms'] += elapsed
        return measured

    def install(self):
        from examples.mpc.franka.runtime import isaac
        from examples.mpc.fingertips.test import test_0902 as policy
        from planning import mpc_explicit
        from planning.contact_force_gpu_cem import GPUContactCEM
        from planning.mlqp_point import LambdaContactControlOptimizer

        # Enclose the actual verify decision block, including geometric gates,
        # model confidence, local-contact acceptance, update_values and EMA.
        # Exact anchors fail loudly if the shared policy is later reorganized.
        source = inspect.getsource(policy.compute_rollout_contact_via)
        anchors = (
            ("    p_arm_force = np.asarray(info.get('control_input', np.zeros(3)), dtype=np.float64).reshape(-1)",
             "    __latency_mark('verify_start')\n"),
            ("    exec_press = _patch_press_point(p_arm_track_world, p_arm_surface_world)",
             "    __latency_mark('verify_end')\n"),
        )
        for anchor, insertion in anchors:
            if source.count(anchor) != 1:
                raise RuntimeError('Verify timing boundary changed: ' + anchor)
            source = source.replace(anchor, insertion + anchor)
        namespace = dict(policy.__dict__, __latency_mark=self.mark)
        exec(compile(source, '<latency-instrumented-contact-policy>', 'exec'), namespace)
        policy.compute_rollout_contact_via = namespace['compute_rollout_contact_via']

        for name in ('choose_contact_points', 'resolve_executed_contact', 'optimize_control_input'):
            setattr(LambdaContactControlOptimizer, name,
                    self.wrap_lambda(getattr(LambdaContactControlOptimizer, name), name))

        def gpu_wrapper(original, kind):
            @functools.wraps(original)
            def measured(backend, *args, **kwargs):
                result = original(backend, *args, **kwargs)
                if self.active is not None:
                    timing = dict(backend.last_timing)
                    self.active['cem_calls'].append(dict(kind=kind, **timing))
                return result
            return measured

        for name in ('rank', 'solve_point'):
            setattr(GPUContactCEM, name, gpu_wrapper(getattr(GPUContactCEM, name), name))

        original_request = mpc_explicit.handle_mpc_request

        @functools.wraps(original_request)
        def request(*args, **kwargs):
            started = time.perf_counter_ns()
            result = original_request(*args, **kwargs)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if self.active is not None:
                self.active.update(
                    callback_ms=elapsed,
                    contact_policy_ms=1000 * result['rank_dt'],
                    mpc_ms=1000 * result['plan_dt'],
                    mpc_skipped=result['mpc_skipped'],
                    solver_backend=result['solver_backend'],
                    lambda_solver_status=result['lambda_solver_status'],
                    mpc_fallbacks=result['mpc_fallbacks'],
                    mpc_failures=result['mpc_failures'],
                    lambda_fallbacks=result['lambda_fallbacks'],
                    lambda_failures=result['lambda_failures'],
                    stage=result['object_cost_diagnostics'].get('object_cost_stage'),
                    sample_count=len(args[1].lambda_optimizer.sample_point),
                    action=result['action'].tolist(),
                )
            return result
        mpc_explicit.handle_mpc_request = request

        original_obs = isaac.SynchronousMPCBus.wait_obs
        original_cmd = isaac.SynchronousMPCBus.publish_cmd

        def wait_obs(bus, *args, **kwargs):
            obs = original_obs(bus, *args, **kwargs)
            started = time.perf_counter_ns()
            if obs is not None and obs.get('cmd') != 'stop':
                self.active = dict(
                    trial=obs['trial'], step=obs['seq'], _observation_ns=started,
                    lambda_ms=0., lambda_rank_ms=0., verify_ms=0., cem_calls=[])
            return obs

        def publish_cmd(bus, command):
            ended = time.perf_counter_ns()
            if command.get('kind') == 'via' and self.active is not None:
                row = self.active
                row['observation_to_action_ms'] = (ended-row.pop('_observation_ns'))/1e6
                row['cem_total_ms'] = sum(c['total_ms'] for c in row['cem_calls'])
                row['cem_gpu_and_download_ms'] = sum(c['gpu_ms'] for c in row['cem_calls'])
                row['other_ms'] = row['observation_to_action_ms'] - sum(
                    row[k] for k in ('lambda_ms', 'verify_ms', 'mpc_ms'))
                self.rows.append(row)
                self.active = None
            result = original_cmd(bus, command)
            if command.get('kind') == 'via' and len(self.rows) % 100 == 0:
                print('[latency] %d actions measured' % len(self.rows),
                      flush=True, file=sys.__stdout__)
            return result

        isaac.SynchronousMPCBus.wait_obs = wait_obs
        isaac.SynchronousMPCBus.publish_cmd = publish_cmd


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--latency-output', required=True)
    parser.add_argument('--latency-warmup', type=int, default=20)
    custom, remaining = parser.parse_known_args()
    if custom.latency_warmup < 0:
        parser.error('--latency-warmup must be nonnegative')
    sys.argv = [sys.argv[0], *remaining]
    # Preserve Isaac's required import order.
    from isaacgym import gymapi, gymtorch  # noqa: F401
    from examples.mpc.franka.runtime import rollout, mpc
    args = mpc._add_rollout_policy_args(argparse.ArgumentParser()).parse_args(remaining)
    mpc._apply_execution_mode(args)
    if args.physics_backend != 'isaac' or args.async_planner or args.num_envs != 1:
        raise ValueError('Use --headless --sync-planner --num-envs 1 with Isaac')
    if args.rollout_verbose or args.rollout_trace_jsonl or args.rollout_planner_snapshots:
        raise ValueError('Disable diagnostic printing/traces/snapshots for latency measurement')
    output = Path(custom.latency_output)
    output.mkdir(parents=True, exist_ok=True)
    probe = LatencyProbe()
    probe.install()
    def gpu_processes():
        return subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader'], text=True).strip().splitlines()
    gpu_processes_before = gpu_processes()
    started = time.perf_counter()
    rollout_summary = rollout.run_isaac_headless(args)
    from planning.rollout_timing import distribution
    steady = [r for r in probe.rows if r['step'] >= custom.latency_warmup]
    keys = ('lambda_ms', 'lambda_rank_ms', 'cem_total_ms', 'cem_gpu_and_download_ms',
            'verify_ms', 'mpc_ms', 'contact_policy_ms', 'callback_ms',
            'observation_to_action_ms', 'other_ms')

    def summarize(rows):
        return dict(n=len(rows), above_20ms=sum(r['observation_to_action_ms'] > 20 for r in rows),
                    timings_ms={k: distribution([r[k] for r in rows]) for k in keys})

    summary = dict(
        arguments=vars(args), warmup_steps_per_trial=custom.latency_warmup,
        python=platform.python_version(), cpu=next(
            (line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
             if line.startswith('model name')), platform.processor()),
        compute_threads={k: os.environ.get(k) for k in (
            'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
        gpu_compute_processes_before=gpu_processes_before,
        gpu_compute_processes_after=gpu_processes(),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
            'planning/mpc_explicit.py', 'planning/mlqp_point.py',
            'planning/contact_force_gpu_cem.py', 'planning/point_force_gpu.py',
            'examples/mpc/franka/runtime/params.py', 'examples/mpc/franka/runtime/mpc.py',
            'examples/mpc/fingertips/test/test_0902.py')},
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version',
                                     '--format=csv,noheader'], text=True).strip(),
        wall_time_s=time.perf_counter()-started,
        definitions=dict(
            lambda_ms='Union of choose_contact_points, resolve_executed_contact and optimize_control_input, including all CEM calls and result selection; nested calls counted once.',
            cem_total_ms='Sum of GPUContactCEM rank/solve_point totals, including prepare, synchronized CUDA execution and download.',
            cem_gpu_and_download_ms='Sum of _run timings, including CUDA graph execution, synchronization and result download.',
            verify_ms='Contiguous decision block from p_arm_force validation through _rollout_verify_cost: geometric gates, confidence, local-contact acceptance, update_values and verify EMA; excludes lambda solves and approach path update.',
            mpc_ms='Existing plan_dt: contact MPC preparation/solve/action extraction; approach cycles use bounded_step and skip the NLP.',
            observation_to_action_ms='After SynchronousMPCBus.wait_obs returns an assembled observation to entry of publish_cmd with the final action; excludes physics, observation acquisition/assembly, rendering, IPC and startup. Includes caller bookkeeping and output postprocessing.',
            other_ms='observation_to_action_ms minus lambda_ms, verify_ms and mpc_ms; includes path planning and other caller/policy work.',
        ),
        rollout_summary=rollout_summary,
        all_cycles=summarize(probe.rows),
        steady_all=summarize(steady),
        steady_contact=summarize([r for r in steady if not r['mpc_skipped']]),
        steady_approach=summarize([r for r in steady if r['mpc_skipped']]),
        first_cycles=[r for r in probe.rows if r['step'] == 0],
    )
    (output/'cycles.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in probe.rows))
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('latency_summary: ' + json.dumps({k: summary[k] for k in (
        'steady_all', 'steady_contact', 'steady_approach')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
