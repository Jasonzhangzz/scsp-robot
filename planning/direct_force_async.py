"""Latest-command asynchronous planner with isolated MuJoCo observation data."""
import copy
import time
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np


class AsyncForcePlanner:
    def __init__(self, env, param, args, solve, contact_type, mode, coupling, scale):
        self.env = copy.copy(env)
        self.env.data_ = mujoco.MjData(env.model_)
        self.param = copy.copy(param)
        self.param.force_cost_stages_ = copy.deepcopy(param.force_cost_stages_)
        self.env.param_ = self.param
        self.env.viewer_ = None
        self.contact = contact_type(self.param)
        self.args, self.solve = args, solve
        self.kw = dict(mode=mode, coupling_scale=coupling, lambda_scale=scale)
        self.future = None
        self.cycles = 0
        self.latest = self._plan(self.snapshot(env))  # warm-up outside rollout clock
        self.pool = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def snapshot(env):
        return (env.data_.qpos.copy(), env.data_.qvel.copy(), float(env.data_.time))

    def _plan(self, snapshot):
        qpos, qvel, stamp = snapshot
        self.env.data_.qpos[:] = qpos
        self.env.data_.qvel[:] = qvel
        self.env.data_.time = stamp
        mujoco.mj_forward(self.env.model_, self.env.data_)
        opt = self.param.lambda_optimizer
        stages = self.param.force_cost_stages_
        if hasattr(self, 'latest'):
            from planning.object_pose_cost import object_cost_np
            progress = object_cost_np(qpos[:7], np.r_[self.param.target_p_, self.param.target_q_],
                                      stages.config()) / stages.progress_scale()
            opt.note_contact_progress(self.latest['selected_idx'], progress, active=True,
                gamma=float(self.args.contact_dwell_gamma),
                min_dwell_steps=max(80, int(self.args.contact_dwell_steps)),
                improve_eps=1e-3, merge_radius=.03, block_radius=.03,
                block_cycles=80, time_decay=True)
        start = time.perf_counter()
        solved = self.solve(self.param, self.args, self.env, self.contact, **self.kw)
        solved['planning_time_ms'] = 1000 * (time.perf_counter() - start)
        solved['observation_time'] = stamp
        solved['selected_idx'] = int(opt.last_selected_idx)
        solved['solver_status'] = str(opt.last_solver_status)
        solved['stages'] = copy.deepcopy(stages)
        return solved

    def poll(self, env):
        updated = False
        if self.future is not None and self.future.done():
            self.latest = self.future.result()  # propagate failures, never silently reuse
            self.future = None
            self.cycles += 1
            updated = True
        if self.future is None:
            self.future = self.pool.submit(self._plan, self.snapshot(env))
        return self.latest, updated

    def close(self):
        if self.future is not None:
            self.future.cancel()
        self.pool.shutdown(wait=True)
