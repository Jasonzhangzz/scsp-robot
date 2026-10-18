"""Parallel Isaac evaluations with independent simulators and planners.

Async rollouts retain real-time pacing. Explicit sync rollouts consume one
20 ms physics interval per command without rendering or sleeps.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
import tempfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from planning.cpu_runtime import configure_compute_threads
configure_compute_threads()


def trial_ranges(start, count, lanes):
    """Partition all trial IDs exactly once, including an uneven final batch."""
    if start < 0 or count <= 0 or lanes <= 0:
        raise ValueError('trial-start must be nonnegative; trial_num and num-envs must be positive')
    lanes = min(lanes, count)
    size, extra = divmod(count, lanes)
    result = []
    for i in range(lanes):
        length = size + int(i < extra)
        result.append(range(start, start + length))
        start += length
    return result


def report(args, summaries, started):
    ids = [s['trial'] for s in summaries]
    expected = list(range(args.trial_start, args.trial_start + args.trial_num))
    if sorted(ids) != expected:
        raise RuntimeError('Evaluation incomplete or duplicate trials: %s (expected %s)' % (ids, expected))
    if any(s.get('aborted') for s in summaries):
        raise RuntimeError('Aborted trial cannot be included in success-rate statistics')
    successes = sum(int(s['success']) for s in summaries)
    flip_successes = sum(bool(s.get('stable_flip_success')) for s in summaries)
    duration = time.perf_counter() - started
    summary = dict(physics_backend=args.physics_backend, solver_requested=args.solver_backend,
                   headless=True, num_envs=min(args.num_envs, args.trial_num),
                   async_planner=bool(getattr(args, 'async_planner', False)),
                   trials=len(summaries), successes=successes, success_rate=successes / len(summaries),
                   stable_flip_successes=flip_successes,
                   stable_flip_rate=(flip_successes/len(summaries)
                       if all('stable_flip_success' in s for s in summaries) else None),
                   trial_start=args.trial_start, wall_time_s=duration,
                   trials_per_second=len(summaries) / max(duration, 1e-9),
                   planner_solver_backends=sorted({s.get('planner_solver_backend', 'unknown') for s in summaries}),
                   lambda_solver_backends=sorted({s.get('lambda_solver_backend', 'unknown') for s in summaries}),
                   mpc_fallbacks=sum(s.get('mpc_fallbacks', 0) for s in summaries),
                   lambda_fallbacks=sum(s.get('lambda_fallbacks', 0) for s in summaries))
    if all('planning_wall_count' in s for s in summaries):
        from examples.mpc.franka.ik2.summarize_rollouts import aggregate
        summary['timing_statistics'] = aggregate(summaries, range(args.trial_start, args.trial_start + args.trial_num))
    print('rollout_summary: ' + json.dumps(summary, allow_nan=False), flush=True)
    if args.rollout_summary_json:
        path = Path(args.rollout_summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


@contextlib.contextmanager
def rollout_output(verbose):
    if verbose:
        yield
    else:
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
            yield


def run_isaac_headless(args):
    if getattr(args, 'rollout_cache_dir', '') and (args.num_envs != 1 or args.async_planner):
        raise ValueError('Source cache requires --sync-planner --num-envs 1; parallelize independent case processes')
    if args.num_envs > 1 or args.async_planner:
        return run_isaac_parallel(args)
    from examples.mpc.franka.ik2.run import SynchronousMPCBus
    from examples.mpc.franka.ik2.test_mpc_isaac import run_mpc_planner
    trial_ranges(args.trial_start, args.trial_num, args.num_envs)
    args.headless, args.async_planner, args.rollout = True, False, True
    summaries = []
    bus = SynchronousMPCBus(args)
    bus.record_summary = summaries.append
    sampler = None
    if getattr(args, 'rollout_cache_dir', ''):
        from examples.mpc.franka.ik2.test_rs import MatchedRolloutBus
        from utils.ranking_cdr import DecisionSampler
        sampler = DecisionSampler()
        bus = MatchedRolloutBus(bus, sampler)
    started = time.perf_counter()
    try:
        with rollout_output(args.rollout_verbose):
            run_mpc_planner(bus, args)
        if sampler is not None:
            from utils.rollout_cache import save_source_cache
            save_source_cache(args.rollout_cache_dir, bus, sampler, args)
            summaries = bus.summaries
    finally:
        bus.close()
    return report(args, summaries, started)


def isaac_worker_entry(conn, values):
    """An independent Isaac simulator, retaining its normal planner protocol."""
    os.environ.pop('SCSP_PLANNER_ONLY', None)
    configure_compute_threads(['--compute-threads', str(values['compute_threads'])]
                              if values.get('compute_threads') is not None else [])
    try:
        from isaacgym import gymapi, gymtorch  # Must precede Torch.
        from examples.mpc.franka.ik2.run import SynchronousMPCBus, isaac_loop
        from examples.mpc.franka.ik2.test_mpc_isaac import run_mpc_planner
        from examples.mpc.franka.ik2.isaac_bus import start_planner, stop_peer
        from planning.runtime_compat import resolve_solver_backend
        args = argparse.Namespace(**values)
        args.solver = resolve_solver_backend(args.solver_backend)
        args.rollout = args.headless = True
        summaries = []
        with rollout_output(args.rollout_verbose):
            if args.async_planner:
                with tempfile.TemporaryDirectory(prefix='isaac_worker_') as directory:
                    path = Path(directory)/'trials.jsonl'
                    args.ranking_metrics_jsonl = str(path)
                    bus, proc = start_planner(args, 'mpc')

                    class CheckedBus:
                        def publish_obs(self, obs):
                            return bus.publish_obs(obs)

                        def take_cmd(self):
                            cmd = bus.take_cmd()
                            if cmd is None and not proc.is_alive():
                                raise RuntimeError('Isaac planner exited before ending the trial')
                            return cmd

                    try:
                        isaac_loop(CheckedBus(), args, 'mpc')
                    finally:
                        stop_peer(bus, proc)
                    if proc.exitcode != 0:
                        raise RuntimeError('Isaac planner exited with code %s' % proc.exitcode)
                    summaries = [json.loads(line) for line in path.read_text().splitlines()]
            else:
                args.ranking_metrics_jsonl = ''
                bus = SynchronousMPCBus(args)
                bus.record_summary = summaries.append
                try:
                    run_mpc_planner(bus, args)
                finally:
                    bus.close()
        if sorted(s['trial'] for s in summaries) != list(range(args.trial_start, args.trial_start+args.trial_num)):
            raise RuntimeError('Isaac worker returned missing or duplicate trials')
        for summary in summaries:
            conn.send(('summary', summary))
        conn.send(('done', None))
    except BaseException:
        conn.send(('error', traceback.format_exc()))
    finally:
        conn.close()


def run_isaac_parallel(args):
    """Parallel independent PhysX worlds; each async world has its own planner.

    This preserves the existing single-world Isaac APIs and collision model,
    rather than changing them to a different simulator or global tensor layout.
    """
    if args.floating_fingertip or args.sampling_c3:
        raise ValueError('Parallel Isaac evaluation requires Franka OSC + MPC')
    ranges = trial_ranges(args.trial_start, args.trial_num, args.num_envs)
    args.headless, args.rollout = True, True
    args.num_envs = len(ranges)
    started = time.perf_counter()
    ctx = mp.get_context('spawn')
    processes, readers, summaries = [], [], []
    metric = None
    try:
        if args.ranking_metrics_jsonl:
            path = Path(args.ranking_metrics_jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            metric = path.open('a')
        for i, r in enumerate(ranges):
            reader, writer = ctx.Pipe(duplex=False)
            values = dict(vars(args), trial_start=r.start, trial_num=len(r), num_envs=1,
                          ranking_metrics_jsonl='', rollout_summary_json='')
            if getattr(args, 'rollout_trace_jsonl', ''):
                values['rollout_trace_jsonl'] = args.rollout_trace_jsonl+'.worker%s.jsonl' % i
            process = ctx.Process(target=isaac_worker_entry, args=(writer, values))
            process.start()
            writer.close()
            processes.append(process)
            readers.append(reader)
        print('[isaac-rollout] %s independent Isaac worlds, %s planning' %
              (len(ranges), 'asynchronous' if args.async_planner else 'synchronous'), flush=True)
        running = set(range(len(ranges)))
        last_progress = started
        from multiprocessing.connection import wait
        while running:
            ready = wait([readers[i] for i in sorted(running)], timeout=1.)
            for reader in ready:
                i = readers.index(reader)
                kind, value = reader.recv()
                if kind == 'error':
                    raise RuntimeError('Isaac world %s failed:\n%s' % (i, value))
                if kind == 'summary':
                    value['evaluation_workers'] = len(ranges)
                    value['parallel_worker'] = i
                    summaries.append(value)
                    print('trial_summary: ' + json.dumps(value, allow_nan=False), flush=True)
                    if metric:
                        metric.write(json.dumps(value, allow_nan=False)+'\n')
                        metric.flush()
                elif kind == 'done':
                    running.remove(i)
            for i in running:
                if not processes[i].is_alive() and not readers[i].poll():
                    raise RuntimeError('Isaac world %s exited without a completion event' % i)
            if time.perf_counter()-last_progress >= 60.:
                print('[isaac-rollout] completed=%s/%s, active worlds=%s' %
                      (len(summaries), args.trial_num, len(running)), flush=True)
                last_progress = time.perf_counter()
    finally:
        if metric:
            metric.close()
        for process in processes:
            process.join(timeout=2.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.)
        for reader in readers:
            reader.close()
    return report(args, summaries, started)


class PipeBus:
    def __init__(self, conn):
        self.conn = conn

    def wait_obs(self, timeout=180.):
        if not self.conn.poll(timeout):
            raise TimeoutError('Headless physics observation timed out')
        return self.conn.recv()

    def publish_cmd(self, command):
        self.conn.send(('command', command))

    def record_summary(self, summary):
        self.conn.send(('summary', summary))


def planner_entry(conn, values):
    os.environ['SCSP_PLANNER_ONLY'] = '1'
    # CPU acados/CSO workers should not initialize CUDA just to probe Torch.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    configure_compute_threads(['--compute-threads', str(values['compute_threads'])]
                              if values.get('compute_threads') is not None else [])
    args = argparse.Namespace(**values)
    args.ranking_metrics_jsonl = ''  # Only the GPU coordinator writes metrics.
    try:
        from examples.mpc.franka.ik2.test_mpc_isaac import run_mpc_planner
        with rollout_output(args.rollout_verbose):
            run_mpc_planner(PipeBus(conn), args)
        conn.send(('done', None))
    except BaseException:
        conn.send(('error', traceback.format_exc()))
    finally:
        conn.close()


def run_mjwarp_headless(args):
    if args.async_planner:
        from examples.mpc.franka.ik2.async_mjwarp_rollout import run_mjwarp_async_headless
        return run_mjwarp_async_headless(args)
    os.environ['SCSP_PLANNER_ONLY'] = '1'
    from examples.mpc.franka.ik2.mjwarp_rollout import MJWarpBatch
    ranges = trial_ranges(args.trial_start, args.trial_num, args.num_envs)
    if args.floating_fingertip or args.sampling_c3:
        raise ValueError('MJWarp evaluation runs Franka OSC + MPC; floating-fingertip/SamplingC3 are unsupported')
    args.headless, args.async_planner, args.rollout = True, False, True
    args.num_envs = len(ranges)
    started = time.perf_counter()
    summaries, workers, connections = [], [], []
    ctx = mp.get_context('spawn')
    # Spawn inherits planner-only mode before importing __mp_main__.
    os.environ['SCSP_PLANNER_ONLY'] = '1'
    batch = MJWarpBatch(args, [r.start for r in ranges])
    trials = [r.start for r in ranges]
    running = set(range(len(ranges)))
    try:
        for r in ranges:
            parent, child = ctx.Pipe()
            values = dict(vars(args), trial_start=r.start, trial_num=len(r))
            proc = ctx.Process(target=planner_entry, args=(child, values))
            proc.start()
            child.close()
            connections.append(parent)
            workers.append(proc)
        for i in running:
            connections[i].send(batch.observation(i, trials[i]))
        metric = None
        if args.ranking_metrics_jsonl:
            path = Path(args.ranking_metrics_jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            metric = path.open('a')
        try:
            while running:
                advancing = []
                # Barrier at every policy interval: physics never advances a
                # world using a stale command while another planner is busy.
                for i in sorted(running):
                    while True:
                        if not connections[i].poll(180.):
                            raise TimeoutError('Planner %s timed out (alive=%s)' % (i, workers[i].is_alive()))
                        kind, value = connections[i].recv()
                        if kind == 'error':
                            raise RuntimeError('Planner %s failed:\n%s' % (i, value))
                        if kind == 'summary':
                            summaries.append(value)
                            print('trial_summary: ' + json.dumps(value, allow_nan=False), flush=True)
                            if metric:
                                metric.write(json.dumps(value, allow_nan=False) + '\n')
                                metric.flush()
                            continue
                        if kind == 'done':
                            running.remove(i)
                            break
                        command = value.get('cmd')
                        if command == 'end_trial':
                            trials[i] += 1
                            if trials[i] < ranges[i].stop:
                                batch.reset(i, trials[i])
                                connections[i].send(batch.observation(i, trials[i]))
                            # The child sends the summary next, then either
                            # finishes or plans the next trial's first action.
                            continue
                        if command == 'stop':
                            continue
                        batch.command(i, value)
                        advancing.append(i)
                        break
                if advancing:
                    batch.step(advancing)
                    for i in advancing:
                        connections[i].send(batch.observation(i, trials[i]))
        finally:
            if metric:
                metric.close()
    finally:
        for proc in workers:
            proc.join(timeout=2.)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.)
        for conn in connections:
            conn.close()
    return report(args, summaries, started)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--physics-backend', choices=('isaac', 'mjwarp'), default='isaac')
    backend, _ = pre.parse_known_args()
    if backend.physics_backend == 'mjwarp':
        os.environ['SCSP_PLANNER_ONLY'] = '1'
    else:
        os.environ.pop('SCSP_PLANNER_ONLY', None)
        from isaacgym import gymapi, gymtorch  # Must precede Torch.
    from examples.mpc.franka.ik2.test_mpc_isaac import (
        _add_rollout_policy_args,
        _apply_execution_mode,
    )
    parser = _add_rollout_policy_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    _apply_execution_mode(args)
    run = run_mjwarp_headless if args.physics_backend == 'mjwarp' else run_isaac_headless
    run(args)


if __name__ == '__main__':
    main()
