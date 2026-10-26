"""Process-isolated GPU CEM with Isaac Gym PhysX rollouts.

The legacy SDK must import before torch; the planner need not obey that import
order or use Python 3.8. Only requests and per-candidate optima cross the pipe.
"""
import atexit
import json
import os
from pathlib import Path
import select
import subprocess
import time

import numpy as np


class PhysXContactCEM:
    def __init__(self, optimizer, mass, inertia_diag, *, scene, python=None, **settings):
        self.optimizer = optimizer
        self.count = len(optimizer.sample_point)
        self.last_timing = {}
        self.last_evaluations = 0
        executable = Path(python or os.environ.get('SCSP_ISAACGYM_PYTHON',
                          str(Path.home()/'miniconda3/envs/acc/bin/python'))).resolve()
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        environment['PATH'] = str(executable.parent)+os.pathsep+environment.get('PATH', '')
        environment['LD_LIBRARY_PATH'] = str(executable.parent.parent/'lib')+os.pathsep+environment.get('LD_LIBRARY_PATH', '')
        environment.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/scsp_physx_cem_extensions')
        self.process = subprocess.Popen(
            [str(executable), '-u', '-m', 'planning.contact_force_physx_worker'],
            cwd=str(Path(__file__).resolve().parents[1]), env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        atexit.register(self.close)
        try:
            self.metadata = self._request(dict(cmd='init', mass=float(mass),
                inertia_diag=np.asarray(inertia_diag).tolist(), scene=scene, settings=settings,
                optimizer={name: np.asarray(getattr(optimizer, name)).tolist() for name in (
                    'sample_point', 'normal', 't1', 't2', 'mu_arm_obj', 'max_contact_force',
                    'h', 'force_reg_coef', 'friction_reg_coef', 'friction_cone')},
                config=optimizer.object_cost_config().tolist()))
        except Exception:
            self.close()
            raise

    def _request(self, payload):
        if self.process.poll() is not None:
            raise RuntimeError('PhysX CEM worker exited: %s' % self.process.returncode)
        self.process.stdin.write(json.dumps(payload, allow_nan=False)+'\n')
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], 180.)
        if not ready:
            raise RuntimeError('PhysX CEM worker timed out')
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('PhysX CEM worker closed its output')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def set_support_plane(self, point, normal):
        """Plane in the same reference frame as current_x, not a contact Jacobian."""
        p, n = np.asarray(point, float).reshape(3), np.asarray(normal, float).reshape(3)
        if not np.isfinite(np.r_[p, n]).all() or np.linalg.norm(n) < 1e-10:
            raise ValueError('invalid PhysX support plane')
        self.support = dict(point=p.tolist(), normal=(n/np.linalg.norm(n)).tolist())

    def reset_seed(self, seed=0):
        """Restart the population RNG sequence, e.g. for matched replays."""
        self._request(dict(cmd='seed', seed=int(seed)))

    def _solve(self, kind, x_d, current_x, tau_o, v_last, **kwargs):
        started = time.perf_counter()
        opt = self.optimizer
        config = kwargs.pop('config', None)
        limit = kwargs.pop('force_limit', None)
        # env_phi is intentionally unused: native collision detection uses the
        # full geometry and support plane, including speculative contacts.
        kwargs.pop('env_phi', None)
        result = self._request(dict(cmd=kind, x_d=np.asarray(x_d).tolist(),
            current_x=np.asarray(current_x).tolist(), tau_o=np.asarray(tau_o).tolist(),
            v_last=np.asarray(v_last).tolist(), support=getattr(self, 'support', None),
            force_limit=float(opt.max_contact_force if limit is None else limit),
            mu=float(opt.mu_arm_obj), config=np.asarray(
                opt.object_cost_config() if config is None else config).tolist(), **kwargs))
        self.last_timing = result.pop('timing')
        self.last_timing['worker_total_ms'] = self.last_timing['total_ms']
        self.last_timing['total_ms'] = 1000*(time.perf_counter()-started)
        self.last_evaluations = result.pop('evaluations')
        return result

    def rank(self, x_d, current_x, tau_o, visible_face_idx, v_last, **kwargs):
        ids = np.asarray(visible_face_idx, dtype=np.int32).reshape(-1)
        if not len(ids) or np.any(ids < 0) or np.any(ids >= self.count):
            raise ValueError('PhysX CEM needs valid nonempty candidate IDs')
        result = self._solve('rank', x_d, current_x, tau_o, v_last, ids=ids.tolist(), **kwargs)
        return ids, np.asarray(result['costs']), np.asarray(result['forces']), np.asarray(result['poses'])

    def solve_point(self, x_d, current_x, tau_o, v_last, p_arm, n_arm, t1, t2, **kwargs):
        result = self._solve('point', x_d, current_x, tau_o, v_last,
            point=np.asarray(p_arm).tolist(), frame=np.column_stack((n_arm, t1, t2)).tolist(), **kwargs)
        return dict(cost=result['cost'], lam_arm_opt=np.asarray(result['force'], dtype=np.float32),
                    x_plus_opt=np.asarray(result['pose']))

    def close(self):
        process = getattr(self, 'process', None)
        if process is None:
            return
        if process.poll() is None:
            try:
                self._request(dict(cmd='close'))
                process.wait(timeout=5)
            except Exception:
                process.kill()
                process.wait()
        for stream in (process.stdin, process.stdout):
            stream.close()
        self.process = None
        atexit.unregister(self.close)
