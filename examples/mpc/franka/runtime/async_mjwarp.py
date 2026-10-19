"""Latest-only asynchronous MJWarp rollout, with the developer bus protocol.

Physics polls commands every 2 ms frame and publishes observations every
20 ms of simulation. Planner latency can skip observations, as in Isaac.
Real-time pacing is retained; engine throughput still affects async timing.
"""
import argparse
import multiprocessing as mp
import os
import time
import traceback

from examples.mpc.franka.runtime.bus import IsaacBus


def planner_entry(cmd_q, obs_q, events, values):
    os.environ['SCSP_PLANNER_ONLY'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    from examples.mpc.franka.runtime.rollout import rollout_output
    from examples.mpc.franka.runtime.mpc import run_mpc_planner

    class Bus(IsaacBus):
        def record_summary(self, summary):
            events.send(('summary', summary))

    args = argparse.Namespace(**values)
    args.ranking_metrics_jsonl = ''
    try:
        with rollout_output(args.rollout_verbose):
            run_mpc_planner(Bus(cmd_q, obs_q), args)
        events.send(('done', None))
    except BaseException:
        events.send(('error', traceback.format_exc()))
    finally:
        events.close()


def run_mjwarp_async_headless(args):
    import json
    from pathlib import Path
    from examples.mpc.franka.runtime.rollout import trial_ranges, report
    from examples.mpc.franka.runtime.mjwarp import MJWarpBatch
    if args.floating_fingertip or args.sampling_c3:
        raise ValueError('MJWarp evaluation requires Franka OSC + MPC')
    args.headless, args.rollout = True, True
    ranges = trial_ranges(args.trial_start, args.trial_num, args.num_envs)
    args.num_envs = len(ranges)
    os.environ['SCSP_PLANNER_ONLY'] = '1'
    started = time.perf_counter()
    batch = MJWarpBatch(args, [r.start for r in ranges])
    trials = [r.start for r in ranges]
    frames = [0] * len(ranges)
    active, waiting = set(), set(range(len(ranges)))
    peers, event_readers, processes, queues, summaries = [], [], [], [], []
    last_command = [time.monotonic()] * len(ranges)
    ctx = mp.get_context('spawn')
    metric = None
    try:
        if args.ranking_metrics_jsonl:
            path = Path(args.ranking_metrics_jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            metric = path.open('a')
        for i, r in enumerate(ranges):
            cmd_q, obs_q = ctx.Queue(maxsize=1), ctx.Queue(maxsize=1)
            reader, writer = ctx.Pipe(duplex=False)
            values = dict(vars(args), trial_start=r.start, trial_num=len(r))
            process = ctx.Process(target=planner_entry, args=(cmd_q, obs_q, writer, values))
            process.start()
            writer.close()
            peers.append(IsaacBus(cmd_q, obs_q))
            event_readers.append(reader)
            processes.append(process)
            queues.extend((cmd_q, obs_q))
            peers[i].publish_obs(batch.observation(i, trials[i]))
        print('[mjwarp-rollout] asynchronous, latest-only, real-time pacing enabled', flush=True)
        while waiting:
            tick_start = time.monotonic()
            for i in sorted(waiting):
                reader = event_readers[i]
                while reader.poll():
                    kind, value = reader.recv()
                    if kind == 'error':
                        raise RuntimeError('Planner %s failed:\n%s' % (i, value))
                    if kind == 'summary':
                        summaries.append(value)
                        print('trial_summary: ' + json.dumps(value, allow_nan=False), flush=True)
                        if metric:
                            metric.write(json.dumps(value, allow_nan=False) + '\n')
                            metric.flush()
                    elif kind == 'done':
                        waiting.remove(i)
                        active.discard(i)
                        break
                if i not in waiting:
                    continue
                command = peers[i].take_cmd()
                if command is not None:
                    last_command[i] = tick_start
                    control = command.get('cmd')
                    if control in ('end_trial', 'stop'):
                        active.discard(i)
                        if control == 'end_trial':
                            trials[i] += 1
                            if trials[i] < ranges[i].stop:
                                batch.reset(i, trials[i])
                                frames[i] = 0
                                peers[i].publish_obs(batch.observation(i, trials[i]))
                    else:
                        batch.command(i, command, record_transition=False)
                        active.add(i)
                if not processes[i].is_alive() and processes[i].exitcode is not None:
                    # A normal exit always sends a reliable done event first;
                    # consume it on the next tick, or fail on abnormal exit.
                    if processes[i].exitcode:
                        raise RuntimeError('Planner %s exited with code %s' % (i, processes[i].exitcode))
                if tick_start - last_command[i] > 180.:
                    raise TimeoutError('Asynchronous planner %s stopped responding' % i)
            advancing = sorted(active)
            if advancing:
                batch.step(advancing, frames=1)
                for i in advancing:
                    frames[i] += 1
                    if frames[i] % batch.steps == 0:
                        env = batch.envs[i]
                        env.seq += 1
                        recorder = env._scm_recorder
                        if recorder is not None:
                            recorder.finish(env.get_policy_state(), batch.observation(i, trials[i]))
                        observation = batch.observation(i, trials[i])
                        peers[i].publish_obs(observation)
                        if recorder is not None:
                            recorder.begin(env.get_policy_state(), observation)
                delay = batch.model.opt.timestep - (time.monotonic() - tick_start)
            else:
                delay = 1. / max(1., args.viewer_hz)
            if waiting and delay > 0.:
                time.sleep(delay)
    finally:
        if metric:
            metric.close()
        for process in processes:
            process.join(timeout=2.)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.)
        for reader in event_readers:
            reader.close()
        for q in queues:
            q.cancel_join_thread()
            q.close()
    return report(args, summaries, started)
