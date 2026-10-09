"""Performance changes preserve kinematics, constrained QP and final candidates."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from planning.dexforge_retarget import PandaContactAware, _PandaQueries, _nonredundant_box_rows
from tests.test_dexforge_retarget import _fast_step, _contact_inputs


def test_ca_kinematics_only_matches_full_forward_points_and_jacobians():
    step = _fast_step()
    hand = _PandaQueries(step, .01)
    qpos, _, _, _ = _contact_inputs(step)
    hand.base_qpos = qpos.copy()
    rng = np.random.default_rng(13)
    full = mujoco.MjData(step.cpu_model)
    for _ in range(5):
        q = rng.uniform(-.5, .5, 14)
        hand.forward(q)
        full.qpos[:] = hand.base_qpos
        full.qpos[hand.qadr] = q
        mujoco.mj_forward(step.cpu_model, full)
        for body in (hand.body[0], hand.body[1], step._ids["left_wrist"]):
            local = rng.normal(0, .03, 3)
            position, jp, jr = hand.point(body, local)
            expected = full.xpos[body] + full.xmat[body].reshape(3, 3) @ local
            full_jp, full_jr = np.zeros((3, step.nv)), np.zeros((3, step.nv))
            mujoco.mj_jac(step.cpu_model, full, full_jp, full_jr, expected, int(body))
            np.testing.assert_allclose(position, expected, atol=1e-14)
            np.testing.assert_allclose(jp, full_jp[:, hand.dadr], atol=1e-14)
            np.testing.assert_allclose(jr, full_jr[:, hand.dadr], atol=1e-14)


def test_redundant_box_constraints_preserve_original_qp_optimum():
    from planning.dexforge_imports import ensure_dexforge
    ensure_dexforge()
    from contactaware.solver.qp import QPProblem, QPSolverOptions
    from contactaware.solver.sparse_qp import make_joint_contact_qp
    rng = np.random.default_rng(7)
    rows = np.r_[np.eye(14), -np.eye(14), rng.normal(size=(600, 14))]
    row_lower = np.r_[np.full(28, -.025), np.full(600, -100.)]
    lower, upper = np.full(14, -.15), np.full(14, .15)
    keep = _nonredundant_box_rows(rows, row_lower, lower, upper)
    assert keep.sum() == 28
    # The worst point of every removed row is still feasible.
    corner = np.where(rows[~keep] >= 0, lower, upper)
    assert np.all(np.einsum("ij,ij->i", rows[~keep], corner) >= row_lower[~keep])
    solver = make_joint_contact_qp(QPSolverOptions(1e-8, 1e-8, 1e-6, 200))
    hessian, gradient = np.eye(14), rng.uniform(-.1, .1, 14)
    solutions = []
    for active in (np.ones(len(rows), bool), keep):
        problem = QPProblem(hessian, gradient, rows[active], row_lower[active],
                            np.full(active.sum(), np.inf), lower, upper)
        solutions.append(solver.solve(problem, label="redundancy equivalence").solution)
    np.testing.assert_allclose(solutions[0], solutions[1], atol=3e-6)


@pytest.mark.parametrize("equal_radius", [True, False])
def test_vectorized_ca_constraints_match_serial_spheres_and_nearest_support(equal_radius):
    step = _fast_step()
    qpos, pose, contacts, inward = _contact_inputs(step)
    solver = PandaContactAware(step, iterations=1)
    h = solver.hand
    h.base_qpos = qpos.copy()
    h.forward(qpos[h.qadr])
    # Symmetric centres exercise the nearest-support tie rule as well.
    center = h.point(step._ids["left_forearm"], np.zeros(3))[0] - pose[:3]
    sphere_radii = [.004, .004, .004] if equal_radius else [.004, .008, .006]
    step.spheres_local = np.c_[center + [[.01, 0, 0], [-.01, 0, 0], [0, .03, 0]], sphere_radii]
    solver.collision_spheres = {}
    for side in ("left", "right"):
        for part in ("forearm", "wrist", "tip"):
            body = step._ids[f"{side}_{part}"]
            offset = step._ids[f"{side}_tip_offset"] if part == "tip" else np.zeros(3)
            solver.collision_spheres[body] = np.array([np.r_[offset, .01], np.r_[offset + [.015, 0, 0], .02]])
    bodies, points, radii, sides = [], [], [], []
    for body, cloud in sorted(solver.collision_spheres.items()):
        side = mujoco.mj_id2name(h.model, mujoco.mjtObj.mjOBJ_BODY, body).split("_")[0]
        for sphere in cloud:
            bodies.append(body)
            points.append(h.point(body, sphere[:3]))
            radii.append(sphere[3])
            sides.append(side)
    distances, rows = [], []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if sides[i] != sides[j]:
                delta = points[i][0] - points[j][0]
                length = max(np.linalg.norm(delta), 1e-10)
                distances.append(length - radii[i] - radii[j] - .002)
                rows.append(delta / length @ (points[i][1] - points[j][1]))
    centers = step.spheres_local[:, :3] + pose[:3]
    for i, (position, jac, _) in enumerate(points):
        delta = position - centers
        length = np.maximum(np.linalg.norm(delta, axis=1), 1e-10)
        gap = length - step.spheres_local[:, 3] - radii[i]
        support = np.argmin(gap)
        clearance = -step.spheres_local[support, 3] if bodies[i] in h.body else .002
        distances.append(gap[support] - clearance)
        rows.append(delta[support] / length[support] @ jac)
    rows, distances = np.asarray(rows), np.asarray(distances)
    captured = []
    def solve(problem, **kwargs):
        captured.append(problem)
        return SimpleNamespace(solution=np.zeros(14))
    solver.qp.solve = solve
    result = solver.solve(qpos, object_pose_wxyz=pose, contact_points_local=contacts, normals_local=inward)
    problem = captured[0]
    row_lower = -np.maximum(distances, 0)
    keep = _nonredundant_box_rows(rows, row_lower, problem.lower, problem.upper)
    np.testing.assert_allclose(problem.rows, rows[keep], atol=1e-12)
    np.testing.assert_allclose(problem.lower_rows, row_lower[keep], atol=1e-12)
    assert result["diagnostics"]["collision_clearance"] == pytest.approx(distances.min(), abs=1e-12)


wp = pytest.importorskip("warp")


@wp.kernel(enable_backward=False)
def quadratic_final_cost(raw: wp.array2d(dtype=float), invalid: wp.array(dtype=int),
                         cost: wp.array(dtype=float), valid: wp.array(dtype=int)):
    world = wp.tid()
    value = float(0.)
    for i in range(14, raw.shape[1]):
        value += raw[world, i] * raw[world, i]
    cost[world] = value
    valid[world] = 1
    if invalid[0] != 0 and world == 0:
        valid[world] = 0


def test_cuda_final_graph_matches_uncaptured_candidates_across_replans():
    from tests.test_forceaware_mppi import _warp_optimizer
    solver, api = _warp_optimizer()
    if not wp.is_device_available("cuda:0"):
        pytest.skip("CUDA unavailable")
    def cuda(value):
        if isinstance(value, wp.array):
            return wp.array(value.numpy(), dtype=value.dtype, device="cuda:0")
        if isinstance(value, tuple):
            return tuple(cuda(v) for v in value)
        if isinstance(value, SimpleNamespace):
            return SimpleNamespace(**{k: cuda(v) for k, v in vars(value).items()})
        return value
    solver.device = "cuda:0"
    for name in ("control", "loss", "mean", "action_lower", "action_upper"):
        setattr(solver, name, cuda(getattr(solver, name)))
    invalid = wp.zeros(1, dtype=int, device=solver.device)
    solver._evaluate_batch = lambda: wp.launch(quadratic_final_cost, dim=8,
        inputs=[solver.control.raw, invalid], outputs=[solver.loss.total, solver.control.valid], device=solver.device)
    rng = np.random.default_rng(19)
    for case in range(3):
        best = rng.uniform(-.035, .035, (8, 42)).astype(np.float32)
        best[:, :14] = 0
        mean = rng.uniform(-.035, .035, 42).astype(np.float32)
        scores = rng.uniform(0, 1, 8).astype(np.float32)
        def prepare():
            solver.control.best_raw.assign(best)
            solver.control.best_loss.assign(scores)
            solver.control.best_available.fill_(1)
            solver.mean.assign(mean)
            invalid.fill_(case % 2)
        def values():
            return (solver.control.raw.numpy(), solver.loss.total.numpy(), solver.control.valid.numpy())
        prepare()
        solver._final_evaluate()
        expected = values()
        prepare()
        solver._evaluate_final_graph()
        for got, want in zip(values(), expected):
            np.testing.assert_array_equal(got, want)
