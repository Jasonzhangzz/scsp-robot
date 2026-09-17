"""Latest-only Isaac <-> planner bus.  Does not import isaacgym or warp."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time


def viewer_draw_stride(sim_dt, target_hz=60.0):
    """Physics frames per viewer draw.  Caps graphics at target_hz."""
    dt = max(float(sim_dt), 1e-6)
    hz = max(float(target_hz), 1.0)
    return max(1, int(round((1.0 / hz) / dt)))


def joint_hold_target(q0, dq, frame_i, n):
    """Interpolate q0+dq over n frames.  frame_i is 1-based; i>=n holds the end."""
    n = max(1, int(n))
    i = min(max(int(frame_i), 0), n)
    frac = float(i) / float(n)
    return [float(a) + float(b) * frac for a, b in zip(q0, dq)]


def put_latest(q, item):
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
        return True
    except queue.Full:
        return False


def take_latest(q):
    item = None
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return item


def drain_latest(q, first):
    msg = first
    while True:
        try:
            newer = q.get_nowait()
        except queue.Empty:
            return msg
        if newer is None:
            return None
        msg = newer


def pickle_args(args):
    return {k: v for k, v in vars(args).items() if not callable(v)}


def args_from_init(init):
    args = argparse.Namespace(**init["args"])
    args.solver = "acados"
    args.rollout = True
    return args


class IsaacBus:
    def __init__(self, cmd_q, obs_q):
        self.cmd_q = cmd_q
        self.obs_q = obs_q

    def publish_obs(self, obs):
        return put_latest(self.obs_q, obs)

    def publish_cmd(self, cmd):
        return put_latest(self.cmd_q, cmd)

    def take_cmd(self):
        return take_latest(self.cmd_q)

    def take_obs(self):
        return take_latest(self.obs_q)

    def wait_obs(self, timeout=180.0):
        first = self.obs_q.get(timeout=timeout)
        if first is None:
            return None
        return drain_latest(self.obs_q, first)


def make_queues():
    ctx = mp.get_context("spawn")
    return ctx, ctx.Queue(maxsize=1), ctx.Queue(maxsize=1), ctx.Queue(maxsize=1)


def isaac_entry(cmd_q, obs_q, ready_q, init):
    os.environ.pop("SCSP_PLANNER_ONLY", None)
    from examples.mpc.franka.ik2.run import isaac_worker

    isaac_worker(cmd_q, obs_q, ready_q, init, announce=True)


def mppi_planner_entry(cmd_q, obs_q, ready_q, init):
    os.environ["SCSP_PLANNER_ONLY"] = "1"
    from examples.mpc.franka.ik2.test_mppi_isaac import planner_worker

    planner_worker(cmd_q, obs_q, ready_q, init)


def mpc_planner_entry(cmd_q, obs_q, ready_q, init):
    os.environ["SCSP_PLANNER_ONLY"] = "1"
    from examples.mpc.franka.ik2.test_mpc_isaac import planner_worker

    planner_worker(cmd_q, obs_q, ready_q, init)


def _start_process(target, args, ready_q, timeout):
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=target, args=args, daemon=True)
    proc.start()
    ready = ready_q.get(timeout=timeout)
    if not ready.get("ok", False):
        proc.join(timeout=2.0)
        raise RuntimeError(ready.get("error", "child process failed to start"))
    print(f"child pid={ready.get('pid', proc.pid)} parent pid={os.getpid()}", flush=True)
    return proc


def start_isaac(args, planner, timeout=180.0):
    ctx, cmd_q, obs_q, ready_q = make_queues()
    init = {"args": pickle_args(args), "planner": planner}
    proc = _start_process(isaac_entry, (cmd_q, obs_q, ready_q, init), ready_q, timeout)
    return IsaacBus(cmd_q, obs_q), proc


def start_planner(args, planner, timeout=180.0):
    ctx, cmd_q, obs_q, ready_q = make_queues()
    init = {"args": pickle_args(args), "planner": planner}
    entry = mppi_planner_entry if planner == "mppi" else mpc_planner_entry
    proc = _start_process(entry, (cmd_q, obs_q, ready_q, init), ready_q, timeout)
    return IsaacBus(cmd_q, obs_q), proc


def stop_peer(bus, proc, cmd=None):
    try:
        bus.publish_cmd(cmd or {"cmd": "stop"})
    except Exception:
        pass
    if proc is None:
        return
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)


def wait_trial_obs(bus, trial, timeout=180.0):
    deadline = time.monotonic() + float(timeout)
    while True:
        left = deadline - time.monotonic()
        if left <= 0.0:
            raise TimeoutError("timed out waiting for Isaac trial %s" % trial)
        obs = bus.wait_obs(timeout=left)
        if obs is None or obs.get("cmd") == "stop":
            return None
        if int(obs.get("trial", trial)) == int(trial):
            return obs
