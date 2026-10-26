"""External PhysX-manifold adapter tests; CUDA parity is explicitly opt-in."""
import copy
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from examples.mpc.franka.scm.runtime_env import activate_dependencies
activate_dependencies()
import mujoco

from examples.mpc.franka.scm.contact_dynamics import full_mass
from examples.mpc.fingertips.test.collision_prediction_assets import scene_xml
from examples.mpc.fingertips.test.collision_prediction_core import Settings
from examples.mpc.fingertips.test.collision_prediction_external_contacts import (
    ExternalContactPredictors, smooth_data)
from models.soft_box_contact import soft_box_contact_velocity_numpy


@pytest.fixture
def predictor(tmp_path):
    path = tmp_path / "model.xml"
    xml = ET.fromstring(scene_xml("cube")[0])
    # Exercise nonzero COM offsets and non-aligned principal axes as occur in
    # the complex mesh objects, rather than only diagonal primitive dynamics.
    body = xml.find("./worldbody/body[@name='object']")
    ET.SubElement(body, "inertial", mass=".01", pos=".003 -.004 .002",
                  quat=".9238795325 .3826834324 0 0", diaginertia=".000006 .000007 .000008")
    path.write_text(ET.tostring(xml, encoding="unicode"))
    return ExternalContactPredictors(path, Settings(nconmax=32, njmax=256))


def sample(predictor):
    qpos = predictor.model.qpos0.copy()
    qpos[:3] = [0., 0., .02999]
    qpos[7:10] = [.03999, 0., .03]
    qpos[3:7] = [.9238795325112867, 0., 0., .3826834323650898]
    qvel = np.array([.02, -.01, -.03, .7, -.3, .2, -.2, .04, .01, -.4, .5, -.1])
    wrench = np.zeros((3, 6))
    wrench[2] = [-.1, .05, 0., 0., 0., 0.]
    return dict(qpos=qpos, qvel=qvel, wrench=wrench)


def manifold():
    return dict(gap=np.array([-1e-5, -1e-5]),
                position=np.array([[.015, -.02, 0.], [.03, 0., .03]]),
                frame=np.array([[[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], np.eye(3)]),
                body=np.array([[0, 1], [1, 2]]),
                friction=np.array([[.5, .5, 0., 0., 0.]]*2), include_margin=np.zeros(2))


def empty_manifold():
    return dict(gap=[], position=[], frame=[], body=[], friction=[], include_margin=[])


def forbid_collision(*args, **kwargs):
    raise AssertionError("External-contact prediction must not call MuJoCo collision/forward")


def test_smooth_stages_match_full_smooth_dynamics_with_offset_inertial(predictor):
    initial = sample(predictor)
    data = smooth_data(predictor.scm_model, initial)
    expected = mujoco.MjData(predictor.scm_model)
    expected.qpos[:] = initial["qpos"]
    expected.qvel[:] = initial["qvel"]
    expected.xfrc_applied[:] = initial["wrench"]
    mujoco.mj_forward(predictor.scm_model, expected)
    np.testing.assert_allclose(full_mass(predictor.scm_model, data), full_mass(predictor.scm_model, expected), atol=1e-15)
    np.testing.assert_allclose(data.qacc_smooth, expected.qacc_smooth, atol=1e-12)
    np.testing.assert_allclose(data.xipos, expected.xipos, atol=1e-15)


def test_scm_preserves_existing_law_and_never_runs_collision(predictor, monkeypatch):
    initial, contacts = sample(predictor), manifold()
    previous_sample, previous_contacts = copy.deepcopy(initial), copy.deepcopy(contacts)
    for name in ("mj_forward", "mj_fwdPosition", "mj_collision"):
        monkeypatch.setattr(mujoco, name, forbid_collision)
    outcome = predictor.scm(initial, contacts)
    assert outcome["status"] == "ok", outcome
    assert outcome["time"] == .02
    problem = predictor.last_scm_problem
    expected_v, expected_impulses = soft_box_contact_velocity_numpy(
        problem["free_velocity"], problem["J"], contacts["gap"], np.linalg.inv(problem["mass"]), .02,
        contacts["friction"][:, 0], temperature=1e-4, regularization=1e-8, sweeps=3,
        return_impulses=True)
    expected_q = initial["qpos"].copy()
    mujoco.mj_integratePos(predictor.scm_model, expected_q, expected_v, .02)
    np.testing.assert_array_equal(outcome["qvel"], expected_v)
    np.testing.assert_array_equal(outcome["qpos"], expected_q)
    np.testing.assert_array_equal(problem["impulse"], expected_impulses)
    for current, previous in ((initial, previous_sample), (contacts, previous_contacts)):
        for key in current:
            np.testing.assert_array_equal(current[key], previous[key])
    predictor.scm(sample(predictor), empty_manifold())
    repeated = predictor.scm(initial, contacts)
    np.testing.assert_array_equal(repeated["qpos"], outcome["qpos"])
    np.testing.assert_array_equal(repeated["qvel"], outcome["qvel"])


def test_contact_jacobians_match_world_point_velocities(predictor):
    initial, contacts = sample(predictor), manifold()
    data = smooth_data(predictor.scm_model, initial)
    jac = predictor._jacobians(data, predictor._manifold(contacts))
    for rows, point, frame, bodies in zip(jac, contacts["position"], contacts["frame"], contacts["body"]):
        velocities = []
        for body in bodies:
            if body == 0:
                velocities.append(np.zeros(3))
                continue
            angular_linear = np.zeros(6)
            mujoco.mj_objectVelocity(predictor.scm_model, data, mujoco.mjtObj.mjOBJ_BODY, body,
                                    angular_linear, 0)
            velocities.append(angular_linear[3:] + np.cross(angular_linear[:3], point-data.xipos[body]))
        np.testing.assert_allclose(rows @ initial["qvel"], frame @ (velocities[1]-velocities[0]), atol=1e-14)


@pytest.mark.parametrize("force", [0., .05, .1, .2, .5])
def test_no_contacts_give_same_gravity_and_external_impulse(predictor, force):
    initial = sample(predictor)
    initial["qvel"][:] = 0.
    initial["wrench"][:] = 0.
    initial["wrench"][2, 0] = force
    outcome = predictor.scm(initial, empty_manifold())
    assert outcome["status"] == "ok", outcome
    assert outcome["qvel"][6] == pytest.approx(force*.02/.01)
    assert outcome["qvel"][8] == pytest.approx(-9.81*.02)
    assert outcome["qpos"][7]-initial["qpos"][7] == pytest.approx(force*.02**2/.01)


def test_invalid_manifold_is_reported_without_truncation(predictor):
    initial, contacts = sample(predictor), manifold()
    contacts["frame"][0, 0] = 0.
    result = predictor.scm(initial, contacts)
    assert result["status"] == "failed" and result["qpos"] is None
    assert "orthonormal" in result["failure_reason"]
    contacts = manifold()
    contacts["body"][0, 1] = 7
    assert "Canonical body" in predictor.scm(initial, contacts)["failure_reason"]
    assert predictor.comfree_batch([], []) == []
    # Invalid worlds never start CUDA and retain explicit failures.
    invalid = predictor.comfree_batch([initial], [contacts])
    assert invalid[0]["status"] == "failed"
    assert predictor.comfree_model is None
    with pytest.raises(ValueError, match="same batch length"):
        predictor.comfree_batch([initial], [])


def test_audit_pose_sign_and_world_com_velocity_comparison(predictor):
    from examples.mpc.fingertips.test.collision_prediction_isaacgym_audit import compare_states
    initial = sample(predictor)
    same_q = initial["qpos"].copy()
    same_q[3:7] *= -1
    same_q[10:14] *= -1
    identical = compare_states(predictor.model, initial["qpos"], initial["qvel"], same_q, initial["qvel"])
    assert identical["passed"]
    assert identical["differences"]["orientation_rad"] == 0.
    changed = initial["qvel"].copy()
    changed[0] += .01
    comparison = compare_states(predictor.model, initial["qpos"], initial["qvel"], same_q, changed)
    assert not comparison["passed"]
    assert comparison["exceeded"] == ["linear_velocity_m_s"]
    assert comparison["differences"]["linear_velocity_m_s"] == pytest.approx(.01)


def test_audit_does_not_hide_contact_count_boundary_changes():
    from examples.mpc.fingertips.test.collision_prediction_isaacgym_audit import compare_labels
    one = dict(pair=[0], gap=[-1e-10], force=[[.1, 0, 0, 0, 0, 0]],
               relative_velocity=[[0, 0, 0, 0, 0, 0]])
    two = {key: value*2 for key, value in one.items()}
    result = compare_labels([one, one], [two, two])
    assert not result["passed"]
    assert result["mode_equal"]
    assert not result["initial_contact_count_equal"]
    assert result["boundary_sensitivity"]["minimum_absolute_gap_m"] == pytest.approx(1e-10)
    intermediate_only = compare_labels([one, one, one], [one, two, one])
    assert intermediate_only["passed"] and not intermediate_only["strict_passed"]
    assert not intermediate_only["contact_count_sequence_equal"]


def test_audit_native_manifold_matching_is_order_independent_and_count_exact():
    from examples.mpc.fingertips.test.collision_prediction_isaacgym_audit import compare_manifolds
    contacts = [dict(body0=0, body1=1, normal_world=[0, 0, 1], point_world=[x, 0, 0],
                     gap=-1e-5, friction=.5, include_margin=.002) for x in (-.02, .02)]
    assert compare_manifolds(contacts, contacts[::-1])["passed"]
    comparison = compare_manifolds(contacts, contacts[:1])
    assert not comparison["passed"]
    assert comparison["body_pair_count_differences"]


@pytest.mark.skipif(not os.getenv("SCSP_COLLISION_GPU_TEST"), reason="opt-in CUDA parity test")
def test_external_comfree_injection_matches_pinned_collision_path_and_is_batched(predictor, monkeypatch):
    predictor._ensure_comfree()
    api, wp = predictor.api, predictor.wp
    initial = sample(predictor)
    # Native upstream collision is used ONLY to construct this equivalence
    # fixture. Production adapters always supply PhysX contact geometry.
    with wp.ScopedDevice(predictor.device):
        baseline_model = api.put_model(predictor.model, comfree_stiffness=.1, comfree_damping=.001)
        baseline = api.put_data(predictor.model, smooth_data(predictor.scm_model, initial), nworld=1,
                                nconmax=32, njmax=256)
        api.step(baseline_model, baseline)
        wp.synchronize_device(predictor.device)
        count = int(baseline.nacon.numpy()[0])
        assert count > 0
        raw = baseline.contact
        native = dict(gap=raw.dist.numpy()[:count], position=raw.pos.numpy()[:count],
                      frame=raw.frame.numpy()[:count], body=predictor.model.geom_bodyid[raw.geom.numpy()[:count]],
                      friction=raw.friction.numpy()[:count], include_margin=raw.includemargin.numpy()[:count])
        expected_q, expected_v = baseline.qpos.numpy()[0], baseline.qvel.numpy()[0]
    # Fail loudly if the new path silently recomputes geometric contacts.
    from comfree_warp.mujoco_warp._src import collision_driver
    monkeypatch.setattr(collision_driver, "collision", forbid_collision)
    for name in ("mj_forward", "mj_fwdPosition", "mj_collision"):
        monkeypatch.setattr(mujoco, name, forbid_collision)
    single = predictor.comfree_batch([initial], [native])[0]
    assert single["status"] == "ok", single
    np.testing.assert_allclose(single["qpos"], expected_q, rtol=0, atol=2e-6)
    np.testing.assert_allclose(single["qvel"], expected_v, rtol=0, atol=2e-5)
    other = copy.deepcopy(initial)
    other["wrench"][2, :3] = [.5, 0., 0.]
    batch = predictor.comfree_batch([initial, other], [native, empty_manifold()])
    assert all(item["status"] == "ok" and item["actual_nworld"] == 2 for item in batch), batch
    np.testing.assert_allclose(batch[0]["qpos"], single["qpos"], rtol=0, atol=2e-6)
    np.testing.assert_allclose(batch[0]["qvel"], single["qvel"], rtol=0, atol=2e-5)
    assert batch[1]["qvel"][6] == pytest.approx(other["qvel"][6]+1., abs=2e-6)
    assert batch[1]["time"] == pytest.approx(.02, abs=1e-7)
    assert predictor.comfree_model.opt.run_collision_detection is False
    assert predictor.last_comfree_timing["nworld"] == 2
    assert predictor.last_comfree_timing["physics_seconds"] > 0


@pytest.mark.skipif(not os.getenv("SCSP_COLLISION_GPU_TEST"), reason="opt-in CUDA force sweep")
def test_comfree_freeflight_five_forces_have_equal_total_imposed_impulse(predictor):
    forces = [0., .05, .1, .2, .5]
    samples = []
    for force in forces:
        initial = sample(predictor)
        initial["qpos"][:3] = [0., 0., 1.]
        initial["qpos"][7:10] = [.3, 0., 1.]
        initial["qvel"][:] = 0.
        initial["wrench"][:] = 0.
        initial["wrench"][2, 0] = force
        samples.append(initial)
    predictions = predictor.comfree_batch(samples, [empty_manifold() for _ in samples])
    for force, initial, prediction in zip(forces, samples, predictions):
        assert prediction["status"] == "ok", prediction
        assert prediction["actual_nworld"] == 5
        assert prediction["time"] == pytest.approx(.02, abs=1e-7)
        assert .01*prediction["qvel"][6] == pytest.approx(force*.02, abs=5e-8)
        assert prediction["qvel"][8] == pytest.approx(-9.81*.02, abs=3e-6)
        assert prediction["qpos"][7]-initial["qpos"][7] == pytest.approx(force*.02**2/.01, abs=3e-6)
    assert .01*predictions[-1]["qvel"][6] == pytest.approx(.01, abs=5e-8)
