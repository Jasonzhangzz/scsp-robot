"""Check legacy SCM backends without native-solver library load-order effects."""
import os
from pathlib import Path
import subprocess
import sys
from itertools import product


def _check_backend_cases():
    import casadi as cs
    import numpy as np
    import torch
    from models.soft_scm_contact import (grouped_contact_velocity_cs,
        grouped_contact_velocity_numpy, grouped_contact_velocity_torch, soft_scm_coefficients)
    from models.support_geometry import support_jacobians

    torch.set_num_threads(1)
    for gaps in ((.1, .1), (-.0002, .1), (0., 0.), (-.0002, -.0003)):
        for friction,cone in product((0., .7, 5.),('coulomb','pyramidal')):
            rows = []
            for side, normal in enumerate(([0., 1., 0.], [0., -1., 0.])):
                block, _ = support_jacobians(np.zeros(3), np.eye(3),
                    [0., -.03 if side == 0 else .03, 0.], 12, friction, normal)
                block[:, 6+3*side:9+3*side] = -block[:, :3]
                rows.append(block)
            jac = np.vstack(rows)
            phi = np.repeat(gaps, 4)
            inverse = np.diag([100.]*3+[1e5]*3+[.4]*6)
            velocity = np.random.default_rng(1).normal(0., .02, 12)
            h, regularization = .01, 1e-8
            a, b = soft_scm_coefficients(h)
            v = cs.SX.sym('v', 12)
            fn = cs.Function('legacy_grouped_contact', [v], [grouped_contact_velocity_cs(
                v, cs.DM(jac), cs.DM(phi), cs.DM(inverse), h, a, b, regularization,friction_cone=cone)])
            native = np.asarray(fn(velocity)).ravel()
            numpy = grouped_contact_velocity_numpy(velocity, jac, phi, inverse, h, a, b, regularization,friction_cone=cone)
            tensor = grouped_contact_velocity_torch(*[torch.as_tensor(value, dtype=torch.float64)
                for value in (velocity, jac, phi, inverse)], h, a, b, regularization,friction_cone=cone).numpy()
            np.testing.assert_allclose(native, numpy, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(native, tensor, atol=1e-10, rtol=1e-10)
    print('24 backend cases passed')


def test_optional_bimanual_features_preserve_all_legacy_scm_backends():
    # Mixing earlier native-solver tests with the first Torch CPU matmul can
    # crash in this environment. Isolate the backend-equivalence check from
    # that library load order; native acados is checked independently.
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
        cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'), text=True, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stdout+result.stderr
    assert '24 backend cases passed' in result.stdout


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _check_backend_cases()
