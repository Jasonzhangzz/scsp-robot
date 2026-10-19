"""Checks for the standalone sampling C3, MIQP, and acados MPC."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../..")))

from examples.mpc.franka.experiments.sampling.c3 import solve as solve_c3
from examples.mpc.franka.experiments.sampling.lcs import (
    pack_state,
    point_push_lcs,
    stage_costs,
)
from examples.mpc.franka.experiments.sampling.miqp import solve as solve_miqp
from examples.mpc.franka.experiments.sampling.sampling import SamplingController


def _push_problem():
    ee = np.array([-0.05, 0.0, 0.0])
    obj = np.array([0.0, 0.0, 0.0])
    goal = np.array([0.08, 0.0, 0.0])
    state = pack_state(ee, obj)
    model = point_push_lcs(ee - obj)
    desired = [pack_state(ee, goal) for _ in range(6)]
    return model, state, desired


def test_gap_matches_next_position():
    model = point_push_lcs(np.array([-1.0, 0.0, 0.0]))
    state = pack_state([-0.05, 0.02, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0])
    control = np.array([0.4, -0.2, 0.1])
    force = np.array([1.5])
    nxt = model.step_affine(state, control, force)
    normal = np.array([-1.0, 0.0, 0.0])
    gap = normal @ (nxt[0:3] - nxt[3:6]) - 0.05
    assert abs(float(np.asarray(model.gap(state, control, force)).reshape(-1)[0] - gap)) < 1e-9


def test_separated_contact_has_zero_force():
    model = point_push_lcs(np.array([-1.0, 0.0, 0.0]))
    state = pack_state([-0.2, 0.0, 0.0], [0.0, 0.0, 0.0])
    nxt, force = model.simulate(state, np.zeros(3))
    assert force[0] == 0.0
    assert nxt[3] == 0.0


def test_compiled_qp_matches_known_solutions():
    from examples.mpc.franka.experiments.sampling.c3 import _extension

    solve_qp = _extension().solve_qp
    hessian = 2.0 * np.eye(2)

    def solved(gradient, constraints, bound):
        return np.asarray(solve_qp(hessian, gradient, constraints, bound), dtype=float).reshape(-1)

    assert np.allclose(solved(np.array([-2.0, -4.0]), np.zeros((0, 2)), np.zeros(0)), [1, 2])
    assert np.allclose(solved(np.zeros(2), np.array([[1.0, 0.0]]), np.array([1.0])), [1, 0])
    assert np.allclose(
        solved(np.array([-2.0, -4.0]), np.array([[-1.0, -1.0]]), np.array([0.0])),
        [-0.5, 0.5],
    )


def test_c3_pushes_the_object():
    model, state, desired = _push_problem()
    plan = solve_c3(model, stage_costs(5), state, desired, u_lower=-50, u_upper=50)
    assert plan["x"].shape == (6, 12)
    assert plan["u"].shape == (5, 3)
    assert plan["x"][-1, 3] > state[3] + 1e-3


def test_compiled_qp_handles_infeasible_zero_start():
    from examples.mpc.franka.experiments.sampling.c3 import _extension

    # Zero violates x+y>=1. Enforcing that boundary first makes the step
    # hit x>=0 at length zero. This previously returned the infeasible [0,0].
    constraints = np.array([[1.0, 0.0], [1.0, 1.0]])
    bound = np.array([0.0, 1.0])
    for gradient, expected in (([0.0, -4.0], [0.0, 2.0]),
                               ([2.0, 0.0], [0.0, 1.0])):
        solution = np.asarray(_extension().solve_qp(
            2.0 * np.eye(2), np.asarray(gradient), constraints, bound,
        )).reshape(-1)
        assert np.all(constraints @ solution >= bound - 1e-8)
        assert np.allclose(solution, expected, atol=1e-8)


def test_c3_is_invariant_to_world_translation():
    # The robot's workspace is at positive world x/z. A projection returning
    # zero must not turn the consensus cost into attraction to the origin.
    results = []
    for offset in (np.zeros(3), np.array([0.5, -0.2, 0.4])):
        start = offset + np.array([-0.05, 0.0, 0.06])
        result = SamplingController(method="c3", n_samples=4).plan(start, offset, offset)
        results.append((result["mode"], result["ee_command"] - offset, result["sample_costs"]))
    assert results[0][0] == results[1][0]
    assert np.allclose(results[0][1], results[1][1], atol=1e-8)
    assert np.allclose(results[0][2], results[1][2], atol=1e-8)


def test_miqp_pushes_the_object_with_complementary_projection():
    model, state, desired = _push_problem()
    plan = solve_miqp(model, stage_costs(5), state, desired, u_lower=-50, u_upper=50)
    assert plan["solver"] == "miqp"
    assert plan["x"][-1, 3] > state[3] + 1e-3
    assert np.all(plan["lambda"] >= -1e-6)


def test_sampling_repositions_toward_a_better_contact():
    controller = SamplingController(
        method="c3", n_samples=6, travel_cost=0.0, use_relative_hysteresis=False, c3_to_reposition=0.0,
    )
    # The fingertip starts on the far side of the object, so a sample on the
    # pushing side has a lower predicted cost than staying put.
    start = np.array([0.2, 0.0, 0.0])
    result = controller.plan(start, [0.0, 0.0, 0.0], [0.08, 0.0, 0.0])
    assert result["mode"] == "reposition"
    assert result["ee"].shape[1] == 3
    assert np.linalg.norm(result["force_command"]) == 0.0
    # dairlib jacktoy: 0.20 m/s and planning_dt 0.1 s, so the next knot is 2 cm.
    assert np.allclose(np.linalg.norm(result["ee_command"] - start), 0.02, atol=1e-6)


def test_c3_command_moves_toward_the_object_goal():
    controller = SamplingController(method="c3", n_samples=1)
    start = np.array([-0.05, 0.0, 0.0])
    result = controller.plan(start, [0.0, 0.0, 0.0], [0.08, 0.0, 0.0])
    assert result["mode"] == "c3"
    # The next knot must push toward +x. Its magnitude depends on convergence
    # and weights; a hard centimetre minimum rewarded the broken projection.
    assert result["ee_command"][0] > start[0]


def test_acados_mpc_pushes_the_object():
    from examples.mpc.franka.experiments.sampling.mpc import ContactMPC

    mpc = ContactMPC(horizon=5)
    state = pack_state([-0.05, 0.0, 0.0], [0.0, 0.0, 0.0])
    desired = pack_state([-0.05, 0.0, 0.0], [0.08, 0.0, 0.0])
    plan = mpc.solve(state, desired)
    assert plan["solver"] == "acados"
    assert plan["x"].shape == (6, 12)
    assert np.all(plan["lambda"] >= -1e-6)
    assert plan["x"][-1, 3] > state[3] + 1e-3


if __name__ == "__main__":
    test_gap_matches_next_position()
    test_separated_contact_has_zero_force()
    test_compiled_qp_matches_known_solutions()
    test_c3_pushes_the_object()
    test_compiled_qp_handles_infeasible_zero_start()
    test_c3_is_invariant_to_world_translation()
    test_miqp_pushes_the_object_with_complementary_projection()
    test_sampling_repositions_toward_a_better_contact()
    test_c3_command_moves_toward_the_object_goal()
    test_acados_mpc_pushes_the_object()
    print("sampling tests passed")
