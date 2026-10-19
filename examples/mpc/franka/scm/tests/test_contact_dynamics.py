"""Independent contracts for the collision/contact-law bridge and LSE scaling."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_env import activate_dependencies
activate_dependencies()

import mujoco
import numpy as np
import pytest

from compare_prediction import make_model, fresh_data, backend_for, argument_parser
from contact_dynamics import (contactsdf_velocity, lse_project, dart_contact_impulse,
                              dart_pgs_solve, contact_snapshot, DART_CFM, DART_MAX_ERV)


def test_dart_port_matches_official_cpp_solver_reference():
    reference = json.loads((Path(__file__).resolve().parents[1] /
        "vendor/dart_contact/pgs_reference.json").read_text())
    rng = np.random.default_rng(reference["rng_seed"])
    for k, expected in enumerate(reference["cases"]):
        n = 3*(k % 4+1)
        b = rng.normal(size=(n, n))
        matrix = b@b.T+.2*np.eye(n)
        rhs = rng.normal(size=n)
        lo, hi = np.tile([0., -.5, -.5], n//3), np.tile([np.inf, .5, .5], n//3)
        ids = np.arange(n)
        parent = ids-ids % 3
        parent[::3] = -1
        actual, info = dart_pgs_solve(matrix, rhs, lo, hi, parent, return_info=True)
        np.testing.assert_allclose(actual, expected["impulse"], atol=1e-12, rtol=1e-11)
        assert info["converged"] == expected["converged"]


def test_dart_box_bounds_and_relative_cfm_match_analytic_contact():
    jac = np.eye(3)[None]
    free = np.array([-1., .4, .8])
    p, _ = dart_contact_impulse(jac, np.zeros(1), np.full((1, 5), .5),
                               np.eye(3), free, .001)
    expected_normal = 1/(1+DART_CFM)
    np.testing.assert_allclose(p, [expected_normal, -.4/(1+DART_CFM), -.5*expected_normal])
    p, _ = dart_contact_impulse(jac, np.array([-.01]), np.full((1, 5), .5),
                               3*np.eye(3), np.zeros(3), .001)
    # CFM scales the original diagonal, rather than adding an absolute epsilon.
    assert p[0] == pytest.approx(DART_MAX_ERV/(3*(1+DART_CFM)))


@pytest.mark.parametrize("h", [.001, .002, .02])
def test_contactsdf_free_motion_preserves_momentum_and_time_units(h):
    rng = np.random.default_rng(17)
    a = rng.normal(size=(6, 6))
    mass = a.T@a + np.eye(6)
    velocity, force = rng.normal(size=(2, 6))
    free = velocity + h*np.linalg.solve(mass, force)
    result, impulse = contactsdf_velocity(mass, free, np.zeros((0, 6)), np.zeros(0), h)
    np.testing.assert_allclose(mass@(result-velocity), h*force, atol=2e-14)
    assert impulse.size == 0


@pytest.mark.parametrize("h", [.001, .02])
def test_lse_numpy_matches_casadi_expression_and_gradient(h):
    cs = pytest.importorskip("casadi")
    rng = np.random.default_rng(31)
    jac = rng.normal(size=(8, 6))
    qinv = np.diag([.2, .3, .4, 1., 2., 3.])
    gap = rng.normal(size=8)*.001
    query = rng.normal(size=6)
    z = cs.SX.sym("z", 6)
    # Literal algebra of upstream LseDynamic.D_SDFInit, independent of NumPy.
    a = cs.DM(jac@qinv/h)
    normer = cs.sqrt(cs.inv(cs.diag(cs.diag(a@a.T)+1e-2)))
    a = normer@a
    b = normer@cs.DM(gap/h)
    distance = cs.fmax(0, cs.logsumexp(120*(-b-a@z))/120)
    fn = cs.Function("upstream_lse", [z], [z-distance*cs.gradient(distance, z)])
    actual, multipliers = lse_project(query, jac, gap, qinv, h)
    np.testing.assert_allclose(actual, np.array(fn(query)).ravel(), atol=1e-13)
    np.testing.assert_allclose(actual, query+qinv.T@jac.T@multipliers/h, atol=1e-13)
    assert np.min(multipliers) >= 0


@pytest.mark.parametrize("law", ["dart", "contactsdf", "diagonal_contact"])
@pytest.mark.parametrize("shape", ["sphere", "box", "elephant"])
def test_cpu_bridge_impulse_force_geometry_and_reset(law, shape):
    model = make_model(shape, .5, .002, .15)
    initial = fresh_data(model)
    initial.qpos[2] -= .0051
    initial.qvel[:] = [.3, -.1, -.2, .1, 0, .2]
    mujoco.mj_forward(model, initial)
    args = argument_parser().parse_args(["--headless"])
    backend = backend_for(law, model, initial, args)
    outputs = []
    for _ in range(2):
        data = fresh_data(model, initial.qpos, initial.qvel)
        backend.reset(data)
        backend.step(data)
        problem, native = backend.api.last_problem, backend.last_contacts
        assert len(native["gap"]) > 0
        # Contact normals come from MuJoCo, and the solver's impulses alone
        # account for the velocity correction (no MuJoCo contact forces).
        np.testing.assert_allclose(data.qvel, problem["free_velocity"] + np.linalg.solve(
            problem["mass"], problem["J"].T@problem["impulse"]), atol=2e-13)
        np.testing.assert_allclose(native["position"], initial.contact.pos[:initial.ncon], atol=0)
        _, native_jac = contact_snapshot(model, initial)
        np.testing.assert_allclose(.002*native_jac.reshape(-1, 6).T @ native["force"][:, :3].ravel(),
                                   problem["J"].T@problem["impulse"], atol=2e-13)
        assert native["force"][:, 0].sum() >= 0
        assert data.time == pytest.approx(.002)
        outputs.append(data.qvel.copy())
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("law", ["dart", "contactsdf", "diagonal_contact"])
def test_cpu_bridge_airborne_matches_mujoco_free_step(law):
    model = make_model("ellipsoid", .5, .002, .15)
    initial = fresh_data(model)
    initial.qpos[2] = .4
    initial.qvel[:] = [.2, .1, -.3, .2, -.4, .5]
    mujoco.mj_forward(model, initial)
    args = argument_parser().parse_args(["--headless"])
    predictions = []
    for engine in (law, "mujoco"):
        backend = backend_for(engine, model, initial, args)
        data = fresh_data(model, initial.qpos, initial.qvel)
        backend.step(data)
        predictions.append(np.r_[data.qpos, data.qvel])
    np.testing.assert_allclose(*predictions, atol=1e-14)


def test_velocity_benchmark_modes_and_rollout_cover_the_full_window():
    import importlib.util
    path = Path(__file__).resolve().parents[3] / "fingertips/test/benchmark_velocity_error.py"
    spec = importlib.util.spec_from_file_location("velocity_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    model = benchmark.make_model("foam_brick", .5, .002, .15)
    assert np.all(model.geom_condim == 3)
    settled = benchmark.settle(model)
    wrench = np.zeros(6)
    sliding = np.array([.5, 0, 0, 0, 0, 0])
    data = mujoco.MjData(model)
    benchmark.set_state(model, data, settled.qpos, sliding, wrench)
    assert benchmark.contact_mode(model, data) == "sliding"
    qpos = settled.qpos.copy()
    qpos[2] += .5
    final, (next_qpos, next_qvel) = benchmark.dart_window(model, qpos, sliding, wrench, .5, .001, 2)
    assert next_qvel[2] == pytest.approx(-9.81*.002)
    assert next_qpos[0]-qpos[0] == pytest.approx(.001)
    np.testing.assert_allclose(final[:3], next_qvel[:3], atol=1e-14)


def test_velocity_benchmark_uses_mjwarp_coupled_lcp_reference():
    import importlib.util
    path = Path(__file__).resolve().parents[3] / "fingertips/test/benchmark_velocity_error.py"
    spec = importlib.util.spec_from_file_location("velocity_lcp_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    model = benchmark.make_model("foam_brick", .5, .002, .15)
    settled = benchmark.settle(model)
    backends = benchmark.make_backends(model, settled)
    reference = backends["reference_lcp"]
    assert reference.engine == "lcp"
    assert reference.model.opt.timestep == pytest.approx(.001)
    assert model.opt.timestep == pytest.approx(.002)
    assert backends["scm_corrected"].api.response_time == 0
    assert backends["scm_corrected"].api.amplitude_correction
    assert backends["scm_soft_corrected"].api.response_time == 0
    assert backends["scm_soft_corrected"].api.softplus_temperature == 1e-4

    qpos = settled.qpos.copy()
    qpos[2] += .5
    velocity = np.array([.5, 0, 0, 0, 0, 0])
    final, (next_pos, next_vel), certificates = benchmark.reference_window(
        reference, qpos, velocity, np.zeros(6))
    assert next_vel[2] == pytest.approx(-9.81*.002, abs=1e-7)
    assert next_pos[0]-qpos[0] == pytest.approx(.001, abs=1e-7)
    np.testing.assert_allclose(final[:3], next_vel[:3], atol=1e-7)
    assert len(certificates) == 2
    assert all(c["residual"] <= 1e-8 for c in certificates)

    _, _, certificates = benchmark.reference_window(
        reference, settled.qpos, velocity, np.zeros(6), steps=1)
    matrix = reference.api.last_problem["W"]
    assert np.linalg.norm(matrix-np.diag(np.diag(matrix))) > 0
    assert all(c["residual"] <= 1e-8 for c in certificates)


@pytest.mark.parametrize("law", ["dart", "lcp"])
def test_multistep_reference_angular_velocity_uses_final_body_frame(law):
    import importlib.util
    path = Path(__file__).resolve().parents[3] / "fingertips/test/benchmark_velocity_error.py"
    spec = importlib.util.spec_from_file_location("velocity_frame_benchmark", path)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    from benchmark_utils import Backend
    model = benchmark.make_model("mug", .5, .001, .15)
    initial = mujoco.MjData(model)
    initial.qpos[2] = .5
    initial.qvel[:] = [.1, .2, .3, 12., 15., -8.]
    mujoco.mj_forward(model, initial)
    initial_rotation = initial.xmat[1].reshape(3, 3).copy()
    wrench = np.array([0., 0., 0., .01, -.02, .03])
    if law == "lcp":
        backend = Backend("lcp", model, initial, nconmax=128, njmax=2048, lcp_solver="active-set")
        velocity, nxt, _ = benchmark.reference_window(
            backend, initial.qpos, initial.qvel, wrench, steps=10)
    else:
        velocity, nxt = benchmark.dart_window(
            model, initial.qpos, initial.qvel, wrench, .5, .001, 10)
    final = mujoco.MjData(model)
    benchmark.set_state(model, final, *nxt, wrench)
    expected = final.xmat[1].reshape(3, 3) @ final.qvel[3:]
    np.testing.assert_allclose(velocity[3:], expected, atol=1e-12)
    assert np.linalg.norm(expected - initial_rotation @ final.qvel[3:]) > 1e-3
