"""Bounded population searches over the shared SCM contact-force objective."""
import numpy as np
import casadi as cs


def feasible_forces(samples, cap, mu, pyramid=True):
    """Map [0, 1]^3 samples to the native solver's normal cap/friction cone."""
    unit = np.clip(np.asarray(samples, dtype=float).reshape(-1, 3), 0., 1.)
    normal = 1e-6 + (cap - 1e-6) * unit[:, 0]
    tangents = 2. * unit[:, 1:] - 1.
    if pyramid:
        tangents /= np.maximum(1., np.abs(tangents).sum(axis=1, keepdims=True))
    return np.column_stack((normal, mu * normal[:, None] * tangents))


class ForcePopulationSearch:
    """Adapters for pycma CMA-ES and EvoTorch's library CEM implementation.

    Each contact starts a fresh distribution. A trial-local RNG advances
    across solves without affecting scene initialization or other optimizers.
    The best feasible evaluation survives every distribution update.
    """
    def __init__(self, backend, size=24, iterations=12, seed=0, elite_fraction=.25):
        if backend not in ('cma-es', 'cem'):
            raise ValueError('population backend must be cma-es or cem')
        if int(size) < 4 or int(iterations) < 1:
            raise ValueError('population size must be >= 4 and iterations >= 1')
        if not np.isfinite(elite_fraction) or not 0 < elite_fraction <= 1:
            raise ValueError('CEM elite fraction must be in (0, 1]')
        self.backend, self.size, self.iterations = backend, int(size), int(iterations)
        self.elites = max(2, int(np.ceil(self.size * elite_fraction)))
        self.rng = np.random.default_rng(int(seed))
        self.maps = {}
        self.evaluations = 0
        if backend == 'cma-es':
            try:
                import cma
            except ImportError as exc:
                raise ImportError('CMA-ES requires cma: python -m pip install "cma>=4,<5"') from exc
            self.cma = cma
        else:
            try:
                import torch
                from evotorch.algorithms.functional import cem, cem_ask, cem_tell
            except ImportError as exc:
                raise ImportError('CEM requires EvoTorch: python -m pip install "evotorch>=0.6,<0.7"') from exc
            self.torch = torch
            self.cem, self.cem_ask, self.cem_tell = cem, cem_ask, cem_tell

    def solve(self, optimizer, kwargs):
        fn = optimizer.force_objective_fn
        if self.size not in self.maps:
            self.maps[self.size] = fn.map(self.size, 'serial')
        batch_fn = self.maps[self.size]
        parameters = {key: cs.repmat(cs.DM(value), 1, self.size)
                      for key, value in kwargs.items()}
        theta = np.asarray(kwargs['scm_theta']).reshape(6)
        mu = (optimizer.base_mu_arm_obj * np.exp(theta[5])
              if optimizer.contact_model == 'soft-scm' else optimizer.mu_arm_obj)
        cap = float(kwargs['force_limit']) / np.sqrt(
            1. + optimizer._friction_force_cap_factor * mu ** 2)
        if not np.isfinite([mu, cap]).all() or mu < 0 or cap < 1e-6:
            raise ValueError('Invalid friction or force cap for population search')
        best_cost, best_force = np.inf, None
        self.evaluations = 0

        def evaluate(samples):
            nonlocal best_cost, best_force
            forces = feasible_forces(samples, cap, mu, optimizer.friction_cone == 'pyramid')
            result = batch_fn(lam_arm=forces.T, **parameters)
            costs = np.asarray(result['cost']).reshape(self.size)
            poses = np.asarray(result['x_plus_opt'])
            valid = np.isfinite(costs) & np.isfinite(poses).all(axis=0)
            costs = np.where(valid, costs, np.inf)
            self.evaluations += self.size
            idx = int(np.argmin(costs))
            if costs[idx] < best_cost:
                best_cost, best_force = float(costs[idx]), forces[idx].copy()
            return costs

        # Always include disengagement and normal-only presses so sampling
        # cannot lose a useful feasible incumbent on a unilateral branch.
        anchors = np.tile([.9, .5, .5], (self.size, 1))
        anchors[0, 0], anchors[1, 0] = 0., 1.
        evaluate(anchors)
        if self.backend == 'cma-es':
            es = self.cma.CMAEvolutionStrategy([.9, .5, .5], .25, {
                'bounds': [0., 1.], 'popsize': self.size,
                'maxiter': self.iterations, 'verbose': -9, 'verb_log': 0,
                'seed': int(self.rng.integers(1, 2**31)),
                'randn': lambda *shape: self.rng.standard_normal(shape),
            })
            for _ in range(self.iterations):
                if es.stop():
                    break
                samples = es.ask()
                costs = evaluate(samples)
                if not np.isfinite(costs).any():
                    break
                # pycma requires finite fitness values for covariance updates.
                worst = float(np.max(costs[np.isfinite(costs)]))
                es.tell(samples, np.where(np.isfinite(costs), costs,
                                          worst + abs(worst) + 1.).tolist())
        else:
            torch = self.torch
            # EvoTorch's functional ask/tell API uses the CPU Torch RNG.
            # Isolate it per contact, restoring the caller's RNG afterwards.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(self.rng.integers(1, 2**31)))
                state = self.cem(
                    center_init=torch.tensor([.9, .5, .5], dtype=torch.float64),
                    stdev_init=.3, stdev_min=.02,
                    parenthood_ratio=self.elites / self.size, objective_sense='min')
                for _ in range(self.iterations):
                    samples = self.cem_ask(state, popsize=self.size).clamp(0., 1.)
                    costs = evaluate(samples.numpy())
                    if not np.isfinite(costs).any():
                        break
                    state = self.cem_tell(state, samples, torch.as_tensor(costs, dtype=torch.float64))
        if best_force is None:
            raise FloatingPointError('Population search has no finite feasible prediction')
        result = fn(lam_arm=best_force, **kwargs)
        return dict(lam_arm_opt=best_force, x_plus_opt=result['x_plus_opt'],
                    cost=float(result['cost']))
