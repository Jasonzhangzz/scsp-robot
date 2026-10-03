"""Latest-only Isaac <-> planner bus.  Does not import isaacgym or warp."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time

import numpy as np


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


class PolicyActionSchedule:
    """Fixed-rate zero-order hold for planner position increments.

    Planner messages may arrive at arbitrary wall-clock times.  A command is
    accepted only after the previous policy interval finishes, so solver
    latency cannot restart a 20 ms trajectory and change the physical speed.
    """

    def __init__(self, sim_dt=0.002, policy_dt=0.02, max_step=0.005,
                 max_speed=None, max_slew=None):
        self.sim_dt = max(float(sim_dt), 1e-6)
        self.policy_dt = max(float(policy_dt), self.sim_dt)
        self.max_step = abs(float(max_step))
        self.max_speed = abs(float(max_speed)) if max_speed is not None else self.max_step / self.policy_dt
        self.max_slew = None if max_slew is None else abs(float(max_slew))
        self.pending = None
        self.active = None
        self.elapsed = self.policy_dt
        self.last_seq = -1
        self.accepted = 0
        self.dropped = 0
        self.planner_latency = None
        self.planner_dt = None
        self.solver_backend = None
        self.actual_speed = 0.0
        self.table_gap = None

    @staticmethod
    def _clip_norm(value, limit):
        value = np.asarray(value, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(value))
        limit = abs(float(limit))
        if norm > limit and norm > 1e-12:
            value = value * (limit / norm)
        return value

    def submit(self, action, seq=None, policy_dt=None, origin=None,
               table_z=None, clearance=0.0, submitted_at=None,
               planner_dt=None, solver_backend=None):
        if seq is None:
            seq = self.last_seq + 1
        seq = int(seq)
        if seq <= self.last_seq:
            self.dropped += 1
            return False
        action = self._clip_norm(
            action, min(self.max_step, self.max_speed * self.policy_dt)
        )
        # The action is a displacement for the executor's interval.  Keep
        # that contract even when an older planner reports another period;
        # metadata must never stretch or restart the physical trajectory.
        if origin is not None and table_z is not None:
            origin = np.asarray(origin, dtype=np.float64).reshape(3)
            floor = float(table_z) + max(float(clearance), 0.0)
            if origin[2] + action[2] < floor:
                action[2] = floor - origin[2]
        if self.max_slew is not None:
            previous = self.pending["action"] if self.pending is not None else (
                self.active["action"] if self.active is not None else None
            )
            if previous is not None:
                delta = action - np.asarray(previous, dtype=np.float64)
                delta = self._clip_norm(delta, self.max_slew)
                action = np.asarray(previous, dtype=np.float64) + delta
                action = self._clip_norm(
                    action, min(self.max_step, self.max_speed * self.policy_dt)
                )
        self.last_seq = seq
        now = time.monotonic()
        self.pending = {
            "seq": seq, "action": action,
            "submitted_at": None if submitted_at is None else float(submitted_at),
            "planner_dt": None if planner_dt is None else float(planner_dt),
        }
        self.planner_latency = (
            None if submitted_at is None else max(0.0, now - float(submitted_at))
        )
        self.planner_dt = None if planner_dt is None else float(planner_dt)
        self.solver_backend = None if solver_backend is None else str(solver_backend)
        self.accepted += 1
        return True

    def _start_pending(self, origin):
        if self.pending is None:
            return False
        p = np.asarray(origin, dtype=np.float64).reshape(3)
        self.active = {
            "seq": self.pending["seq"],
            "action": self.pending["action"].copy(),
            "origin": p.copy(),
        }
        self.pending = None
        self.elapsed = 0.0
        return True

    def take_pending(self, origin=None):
        pending = self.pending
        self.pending = None
        if pending is not None and origin is not None:
            self.active = {
                "seq": pending["seq"],
                "action": np.asarray(pending["action"], dtype=np.float64).copy(),
                "origin": np.asarray(origin, dtype=np.float64).reshape(3).copy(),
            }
            self.elapsed = 0.0
        return pending

    def finish(self):
        self.active = None
        self.elapsed = self.policy_dt

    def record_execution(self, velocity=None, table_gap=None):
        if velocity is not None:
            self.actual_speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)))
        if table_gap is not None:
            self.table_gap = float(table_gap)

    def clear(self):
        self.pending = None
        self.active = None
        self.elapsed = self.policy_dt
        self.last_seq = -1
        self.planner_latency = None
        self.planner_dt = None
        self.solver_backend = None
        self.actual_speed = 0.0
        self.table_gap = None

    def advance(self, origin):
        """Advance one simulation frame and return ``(target, velocity, seq)``."""
        if self.active is None:
            self._start_pending(origin)
        if self.active is None:
            p = np.asarray(origin, dtype=np.float64).reshape(3)
            return p.astype(np.float32), np.zeros(3, dtype=np.float32), None
        self.elapsed = min(self.policy_dt, self.elapsed + self.sim_dt)
        alpha = min(1.0, self.elapsed / self.policy_dt)
        action = self.active["action"]
        target = self.active["origin"] + alpha * action
        velocity = action / self.policy_dt if alpha < 1.0 else np.zeros(3)
        seq = self.active["seq"]
        if self.elapsed >= self.policy_dt - 1e-9:
            self.active = None
        return target.astype(np.float32), velocity.astype(np.float32), seq

    def diagnostics(self):
        return {
            "active_seq": None if self.active is None else int(self.active["seq"]),
            "pending_seq": None if self.pending is None else int(self.pending["seq"]),
            "last_seq": int(self.last_seq),
            "accepted": int(self.accepted),
            "dropped": int(self.dropped),
            "elapsed": float(self.elapsed),
            "policy_dt": float(self.policy_dt),
            "sim_dt": float(self.sim_dt),
            "actual_speed": float(getattr(self, "actual_speed", 0.0)),
            "planner_latency": getattr(self, "planner_latency", None),
            "planner_dt": getattr(self, "planner_dt", None),
            "solver_backend": getattr(self, "solver_backend", None),
            "table_gap": getattr(self, "table_gap", None),
        }


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
    from planning.runtime_compat import resolve_solver_backend
    requested = getattr(args, "solver_backend", "auto")
    args.solver_backend = requested
    args.solver = resolve_solver_backend(requested)
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
