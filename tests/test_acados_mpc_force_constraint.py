"""Native RTI regression for the force bound on the executed first stage."""
from types import SimpleNamespace

import casadi as cs
import numpy as np
import pytest

from planning.acados_env import ensure_acados_env
from planning.mpc_explicit import MPCExplicit


def test_rti_diagonal_push_respects_force_ball_from_cold_and_warm_starts():
    ensure_acados_env()
    pytest.importorskip("acados_template")
    mpc = MPCExplicit.__new__(MPCExplicit)
    mpc.param_ = SimpleNamespace(
        mpc_horizon_=2, n_qpos_=3, n_cmd_=3, n_qvel_=3, max_ncon_=1,
        h_=.02, robot_stiff_=600*np.eye(3), model_params=1.,
        mpc_u_lb_=-.005*np.ones(3), mpc_u_ub_=.005*np.ones(3),
        mpc_q_lb_=-np.ones(3), mpc_q_ub_=np.ones(3),
        runtime_object_cost_=False,
    )
    x, u, target = [cs.SX.sym(name, 3) for name in ("x", "u", "target")]
    phi, jac = cs.SX.sym("phi", 4), cs.SX.sym("jac", 4, 3)
    parameters, reference = cs.SX.sym("parameters"), cs.SX.sym("reference", 3)
    mpc.stage_prediction_fn = cs.Function(
        "force_bound_regression_dynamics", [x, u, phi, jac, parameters, reference], [x+u])
    mpc.model = SimpleNamespace(
        model_parameter_dim=1, contact_model="force_regression",
        prediction_fn=mpc.stage_prediction_fn, runtime_parameters=lambda value: value,
    )
    mpc.path_cost_fn = cs.Function("force_bound_path", [x, u, target],
                                   [cs.sumsqr(x-target)+.01*cs.sumsqr(u)])
    mpc.final_cost_fn = cs.Function("force_bound_terminal", [x, target],
                                    [10*cs.sumsqr(x-target)])
    mpc.cost_kind, mpc.stage_cost_on_next = "param", True
    mpc.update_contact_gaps, mpc.contact_force_limit = False, 3.
    mpc.acados_solve_count = mpc.acados_failure_count = 0
    mpc.nlp_lam_g0_ = cs.DM.zeros(8)
    mpc.acados_solver_ = mpc._build_acados_solver()
    ocp = mpc.acados_solver_.acados_ocp
    assert ocp.dims.nh_0 == ocp.dims.nh == 1
    np.testing.assert_array_equal(ocp.constraints.uh_0, ocp.constraints.uh)

    # All axes want the box limit (5.196 N); the force ball permits only 3 N.
    guess = None
    for _ in range(3):
        result = mpc._plan_once_acados(
            np.zeros(3), np.ones(4), np.zeros((4, 3)), .02*np.ones(3),
            guess, mpc.param_.mpc_u_lb_, mpc.param_.mpc_u_ub_)
        assert result is not None, getattr(mpc, "last_acados_error", None)
        assert result["solver_backend"] == "acados"
        forces = np.linalg.norm(result["u_traj"] @ mpc.param_.robot_stiff_.T, axis=1)
        assert np.all(forces <= 3.+1e-6)
        assert forces[0] > 2.9  # A zero action must not hide a broken constraint.
        guess = result["sol_guess"]
    assert mpc.acados_failure_count == 0
