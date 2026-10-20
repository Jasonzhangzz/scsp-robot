"""ContactSDF dynamics loaded from the upstream source file.

`examples/mpc/franka/scm/vendor/contactsdf/lse_dynamics.py` is
asu-iris/ContactSDF commit 3a21edf8e51788f72a7d77259af4dd584d5ec43f
`dynamics/lse_dynamics.py`, unmodified. Prediction calls `dyn_once_fn`.
Theta values are the arrays in that repository's `test_cube.py`,
`test_foambrick.py` and `test_stick.py`.
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CONTACTSDF_SIGMA = 120.0
_SOURCE = Path(__file__).resolve().parents[1] / 'examples/mpc/franka/scm/vendor/contactsdf/lse_dynamics.py'
_COMMIT = '3a21edf8e51788f72a7d77259af4dd584d5ec43f'

# test_cube.py, test_foambrick.py, test_stick.py
CUBE_THETA = np.array([120.417, 0.0913591, 1.4693, 0.602442, 1.32538, 0.513266, -0.000302907, 1.08711])
FOAM_THETA = np.array([120.223, 0.403609, 1.5038, 1.14279, 1.55258, 0.99718, -0.015059, 0.340951])
STICK_THETA = np.array([100.545, 0.199336, 1.90137, 1.94328, 1.60386, 0.769046, -0.00101175, 0.755457])


def theta_for_object(name):
    key = str(name or 'cube').lower().replace('-', '_')
    if 'foam' in key:
        return FOAM_THETA.copy(), 'foambrick'
    if 'stick' in key:
        return STICK_THETA.copy(), 'stick'
    return CUBE_THETA.copy(), 'cube'


def load_lse_dynamic():
    spec = importlib.util.spec_from_file_location('contactsdf_upstream_lse_dynamics', _SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LseDynamic


class ContactSDFModel:
    """Upstream LseDynamic sized to the Franka pose, with that repository's theta."""

    contact_model = 'contactsdf'
    model_parameter_dim = 1
    online_scm = False

    def __init__(self, param):
        theta = np.asarray(getattr(param, 'contactsdf_theta_', CUBE_THETA), dtype=float).reshape(8)
        if not np.isfinite(theta).all():
            raise ValueError('ContactSDF theta must be finite')
        self.theta = theta.copy()
        self.param_ = param
        upstream = SimpleNamespace(
            n_qvel_=int(param.n_qvel_), n_qpos_=int(param.n_qpos_), n_cmd_=int(param.n_cmd_),
            max_ncon_=int(param.max_ncon_), h_=float(param.h_),
            gravity_=np.asarray(param.gravity_, dtype=float).reshape(6))
        self.dynamics = load_lse_dynamic()(upstream)
        self.dyn_once_fn = self.dynamics.dyn_once_fn

    def prediction_fn(self, curr_q, cmd, phi_vec, jac_n, jac_f, model_params=None):
        del model_params
        import casadi as cs
        return self.dyn_once_fn(curr_q, cmd, phi_vec, jac_n, jac_f, cs.DM(self.theta))

    def runtime_parameters(self, sigma):
        return float(sigma)

    def step(self, curr_q, cmd, phi_vec, jac_n, jac_f):
        return self.prediction_fn(curr_q, cmd, phi_vec, jac_n, jac_f)


def contactsdf_ipopt_options(max_iter):
    """ball_mpc_lse.MPC_IPOPT options."""
    return {
        'ipopt.print_level': 0,
        'ipopt.sb': 'yes',
        'print_time': 0,
        'ipopt.max_iter': int(max_iter),
    }


def solver_backend_on_command_line(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    return any(item == '--solver-backend' or item.startswith('--solver-backend=') for item in argv)


def requested_on_command_line(argv):
    if '--contactsdf' in argv or '--contact-model=contactsdf' in argv:
        return True
    return any(flag == '--contact-model' and i+1 < len(argv) and argv[i+1] == 'contactsdf'
               for i, flag in enumerate(argv))


def dispatch_standalone(root, argv):
    """Route public ContactSDF entry points before loading the local policy."""
    if not requested_on_command_line(argv):
        return
    import os
    os.execv(sys.executable, [sys.executable, '-u',
        str(Path(root)/'examples/mpc/franka/contactsdf_run.py'), *argv])


def apply_contactsdf_args(args, argv=None):
    """Select upstream LSE dynamics. --contactsdf uses IPOPT unless a backend was named."""
    chosen = str(getattr(args, 'contact_model', 'soft-scm')).strip().lower().replace('_', '-')
    if chosen == 'scm-soft':
        chosen = 'soft-scm'
    requested = bool(getattr(args, 'contactsdf', False)) or chosen == 'contactsdf'
    if not requested:
        return args
    if bool(getattr(args, 'contactsdf', False)) and chosen not in ('soft-scm', 'contactsdf'):
        raise ValueError('--contactsdf cannot be combined with --contact-model ' + chosen)
    if (getattr(args, 'online_scm', False) or getattr(args, 'scm_parameters', '')
            or getattr(args, 'scm_record_data', False)):
        raise ValueError('ContactSDF does not use online soft-SCM identification')
    args.contactsdf = True
    args.contact_model = 'contactsdf'
    theta, source = theta_for_object(getattr(args, 'obj', 'cube'))
    command = sys.argv[1:] if argv is None else argv
    if any(item == '--contactsdf-sigma' or item.startswith('--contactsdf-sigma=') for item in command):
        theta = np.array(theta, copy=True)
        theta[0] = float(args.contactsdf_sigma)
        if not np.isfinite(theta[0]) or theta[0] <= 0:
            raise ValueError('--contactsdf-sigma must be finite and positive')
    args.contactsdf_theta = theta
    args.contactsdf_theta_source = source
    if not solver_backend_on_command_line(argv):
        args.solver_backend = 'ipopt'
    return args
