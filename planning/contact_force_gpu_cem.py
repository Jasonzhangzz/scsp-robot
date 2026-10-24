"""GPU-parallel CEM backend for CSO contact ranking (``solver='cem-gpu'``).

All visible contact candidates of ``LambdaContactControlOptimizer`` are
evaluated in one batched CUDA launch through ``planning.point_force_gpu``:
every (candidate, population sample) is a CUDA thread, the contact law is the
paper SCM (diagonal softplus_T followed by one residual correction on the
coupled Delassus operator) and the object dynamics are the physical one-step
model ``M v+ = M v + h (g + J_c^T f + J_env^T lambda_env)``.

The CPU population search in ``contact_force_sampling.py`` evaluates the
candidates serially with CasADi; this backend replaces it on CUDA while keeping
the ``choose_contact_points`` / ``_solve_optimization`` interfaces.
"""
import json
import os
import time
from types import SimpleNamespace

import numpy as np


def _physical_mass_matrix(mass, inertia_diag):
    mass_matrix = np.zeros((6, 6))
    mass_matrix[:3, :3] = float(mass) * np.eye(3)
    mass_matrix[3:, 3:] = np.diag(np.asarray(inertia_diag, dtype=float).reshape(3))
    return mass_matrix


class GPUContactCEM:
    """Batched CUDA CEM over the optimizer's surface samples.

    One extra slot (index ``count``) is reserved for ad-hoc contact points so
    ``optimize_control_input`` / ``resolve_executed_contact`` can solve a
    measured contact that is not one of the pre-sampled candidates.
    """

    def __init__(self, optimizer, mass, inertia_diag, population_size=24,
                 population_iterations=12, elite_fraction=.25, seed=0,
                 device='cuda:0', softplus_temperature=1e-3, scm_regularization=1e-8,
                 max_linear_force_delta_v=.3, max_angular_force_delta_v=3.):
        import warp as wp
        from planning.point_force_gpu import GPUForceCEM
        self.wp = wp
        self.optimizer = optimizer
        self.count = int(len(optimizer.sample_point))
        self.mass_matrix = _physical_mass_matrix(mass, inertia_diag)
        self.inverse = np.linalg.inv(self.mass_matrix)
        self.seed = int(seed)
        self.round = 0
        points = np.r_[np.asarray(optimizer.sample_point, dtype=float).reshape(-1, 3),
                       np.zeros((1, 3))]
        frames = np.stack((optimizer.normal, optimizer.t1, optimizer.t2), axis=2)
        frames = np.r_[np.asarray(frames, dtype=float).reshape(-1, 3, 3), np.eye(3)[None]]
        self.points_host = points
        self.frames_host = frames
        self.args = SimpleNamespace(
            device=device, population_size=int(population_size),
            population_iterations=int(population_iterations),
            cem_elite_fraction=float(elite_fraction),
            arm_friction=float(optimizer.mu_arm_obj),
            max_contact_force=float(optimizer.max_contact_force),
            policy_dt=float(optimizer.h),
            scm_regularization=float(scm_regularization), contactsdf_sigma=120.,
            softplus_temperature=float(softplus_temperature),
            force_reg_coef=float(optimizer.force_reg_coef),
            friction_reg_coef=float(optimizer.friction_reg_coef),
            linear_velocity_coef=0., angular_velocity_coef=0.,
            max_linear_force_delta_v=float(max_linear_force_delta_v),
            max_angular_force_delta_v=float(max_angular_force_delta_v),
            lcp_max_iterations=1000, lcp_tolerance=1e-8)
        self.gpu = GPUForceCEM(points, frames, self.args, 'scm_corrected')
        self.last_timing = {}
        self.last_evaluations = 0
        # Kernel compilation and CUDA-graph capture happen once here, so the
        # first measured planning cycle does not include them.
        self._warm_up()

    # ------------------------------------------------------------------ #
    def _warm_up(self):
        target = np.array([0., 0., 0., 1., 0., 0., 0.])
        valid = np.zeros(self.count + 1, dtype=np.int32)
        valid[:min(4, self.count)] = 1
        jac = np.zeros((1, 6)); jac[0, 2] = 1.
        self._prepare(np.zeros(6), np.r_[0., 0., -9.81*self.mass_matrix[0, 0], 0., 0., 0.],
                      target, target, self.optimizer.object_cost_config(), jac, np.zeros(1), valid,
                      self.optimizer.max_contact_force, self.optimizer.mu_arm_obj)
        self.gpu.solve()
        self.wp.synchronize_device(self.gpu.device)

    def _prepare(self, v_last, tau_o, current_x, x_d, config, jac, gap, valid,
                 force_limit, mu, points=None, frames=None):
        j = self.gpu.job
        if points is not None:
            j.point.assign(points)
        if frames is not None:
            j.frame.assign(frames)
        j.mu = float(mu)
        j.cap = float(force_limit) / np.sqrt(1. + j.mu * j.mu)
        self.gpu.args.max_contact_force = float(force_limit)
        self.gpu.args.arm_friction = float(mu)
        free = np.asarray(v_last, dtype=float).reshape(6) + self.optimizer.h * (
            self.inverse @ np.asarray(tau_o, dtype=float).reshape(6))
        self.round += 1
        self.gpu.prepare(self.inverse, free, np.asarray(current_x, dtype=float).reshape(7),
                         np.asarray(x_d, dtype=float).reshape(7), config, jac, gap, valid,
                         1000003 * self.seed + 7919 * self.round + 1,
                         observed=np.asarray(v_last, dtype=float).reshape(6))

    def _environment_rows(self, env_phi=None):
        jac = np.asarray(self.optimizer.J_tilde, dtype=float)[:, :6]
        gap = np.asarray(self.optimizer.env_phi if env_phi is None else env_phi,
                         dtype=float).reshape(-1)[:jac.shape[0]]
        active = np.any(np.abs(jac) > 0., axis=1)
        if not np.any(active):
            # A zero row has no effect on the predicted velocity.
            return np.zeros((1, 6)), np.zeros(1)
        return jac[active], gap[active]

    def _run(self):
        wp = self.wp
        started = time.perf_counter()
        if self.gpu.graph is None:
            # Capture the CUDA graph once (warm-up); afterwards launch directly.
            try:
                self.gpu.solve()
            except FloatingPointError:
                pass
        else:
            wp.capture_launch(self.gpu.graph)
        wp.synchronize_device(self.gpu.device)
        j = self.gpu.job
        # Only the per-candidate optimum is read back; the full population
        # (samples, velocities, poses) stays on the device.
        costs = j.best_cost.numpy()
        raw = j.best.numpy()
        poses = j.best_pose.numpy()
        self.last_timing['gpu_ms'] = 1000. * (time.perf_counter() - started)
        return costs, raw, poses

    @staticmethod
    def _pose_wxyz(pose):
        pose = np.asarray(pose, dtype=float).reshape(-1, 7)
        return np.column_stack((pose[:, :3], pose[:, [6, 3, 4, 5]]))

    # ------------------------------------------------------------------ #
    def rank(self, x_d, current_x, tau_o, visible_face_idx, v_last, env_phi=None,
             force_limit=None, config=None):
        """Solve every visible candidate; returns ids, costs, forces, x_plus."""
        started = time.perf_counter()
        opt = self.optimizer
        ids = np.asarray(visible_face_idx, dtype=np.int32).reshape(-1)
        valid = np.zeros(self.count + 1, dtype=np.int32)
        valid[ids] = 1
        jac, gap = self._environment_rows(env_phi)
        self._prepare(v_last, tau_o, current_x, x_d,
                      opt.object_cost_config() if config is None else config, jac, gap, valid,
                      opt.max_contact_force if force_limit is None else force_limit,
                      opt.mu_arm_obj, points=self.points_host, frames=self.frames_host)
        self.last_timing['prepare_ms'] = 1000. * (time.perf_counter() - started)
        costs, raw, poses = self._run()
        costs = np.asarray(costs[ids], dtype=np.float64)
        costs[~np.isfinite(costs) | (costs >= 1e99)] = np.inf
        forces = self.gpu.commands(raw[ids], ids).astype(np.float32)
        x_plus = self._pose_wxyz(poses[ids])
        failed = ~np.isfinite(costs)
        if np.any(failed):
            forces[failed] = 0.
            x_plus[failed] = np.asarray(current_x, dtype=float).reshape(7)
        self.last_evaluations = int(ids.size) * self.gpu.job.population * self.gpu.search_iterations
        self.last_timing['total_ms'] = 1000. * (time.perf_counter() - started)
        self.last_timing['candidates'] = int(ids.size)
        self.last_timing['env_rows'] = int(jac.shape[0])
        self._log_timing('rank')
        return ids, costs, forces, x_plus

    def _log_timing(self, kind):
        """Optional per-call CSO timing record (SCSP_GPU_CEM_TIMING_LOG=path.jsonl)."""
        path = os.environ.get('SCSP_GPU_CEM_TIMING_LOG', '')
        if not path:
            return
        with open(path, 'a') as handle:
            handle.write(json.dumps(dict(kind=kind, round=self.round, **self.last_timing)) + '\n')

    def solve_point(self, x_d, current_x, tau_o, v_last, p_arm, n_arm, t1, t2,
                    env_phi=None, force_limit=None, config=None, **_):
        """Single ad-hoc contact (used by optimize_control_input)."""
        started = time.perf_counter()
        opt = self.optimizer
        points = self.points_host.copy()
        frames = self.frames_host.copy()
        points[self.count] = np.asarray(p_arm, dtype=float).reshape(3)
        frames[self.count] = np.column_stack((np.asarray(n_arm, dtype=float).reshape(3),
                                              np.asarray(t1, dtype=float).reshape(3),
                                              np.asarray(t2, dtype=float).reshape(3)))
        valid = np.zeros(self.count + 1, dtype=np.int32)
        valid[self.count] = 1
        jac, gap = self._environment_rows(env_phi)
        self._prepare(v_last, tau_o, current_x, x_d,
                      opt.object_cost_config() if config is None else config, jac, gap, valid,
                      opt.max_contact_force if force_limit is None else force_limit,
                      opt.mu_arm_obj, points=points, frames=frames)
        self.last_timing['prepare_ms'] = 1000. * (time.perf_counter() - started)
        costs, raw, poses = self._run()
        self.last_timing['total_ms'] = 1000. * (time.perf_counter() - started)
        self.last_timing['candidates'] = 1
        self.last_timing['env_rows'] = int(jac.shape[0])
        self._log_timing('point')
        cost = float(costs[self.count])
        if not np.isfinite(cost) or cost >= 1e99:
            raise FloatingPointError('GPU CEM found no finite contact force')
        force = self.gpu.commands(raw[self.count][None], [self.count])[0]
        return dict(lam_arm_opt=np.asarray(force, dtype=np.float32),
                    x_plus_opt=self._pose_wxyz(poses[self.count])[0], cost=cost)
