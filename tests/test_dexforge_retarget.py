"""CA uses native DexForge residuals/QPs; FA targets use its native window API.

Only FA's expensive dynamics optimizer is replaced in the adapter tests.  The
MuJoCo kinematics, singleton contact mapping and trajectory conversion are real.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from planning.dexforge_imports import ensure_dexforge
from planning.dexforge_retarget import (
    DexForgeRetargetingMPC,
    PandaContactAware,
    _PandaQueries,
    pose_xyzw,
    singleton_pose,
)


def _fast_step():
    """Two spatial seven-hinge arms with separate actuator and joint limits."""
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.01", gravity="0 0 0")
    world = ET.SubElement(root, "worldbody")
    actuators = ET.SubElement(root, "actuator")
    axes = ("0 0 1", "0 1 0", "1 0 0", "0 0 1", "0 1 0", "1 0 0", "0 0 1")
    for side, base_x in (("left", -0.55), ("right", 0.30)):
        body = ET.SubElement(world, "body", name=f"{side}_base", pos=f"{base_x} 0 0.50")
        for joint, axis in enumerate(axes, 1):
            body = ET.SubElement(body, "body", name=f"{side}_link{joint}", pos="0.04 0.01 0.03")
            ET.SubElement(body, "joint", name=f"{side}_joint{joint}", axis=axis,
                          limited="true", range="-1.5 1.5")
            ET.SubElement(body, "geom", type="sphere", size="0.01", mass="0.1",
                          contype="0", conaffinity="0")
            ET.SubElement(actuators, "position", name=f"{side}_actuator{joint}",
                          joint=f"{side}_joint{joint}", kp="100", ctrllimited="true",
                          ctrlrange="-1.2 1.2")
        ET.SubElement(body, "site", name=f"{side}_tip_center", pos="0.02 0 0.03")
    obj = ET.SubElement(world, "body", name="obj", pos="0 0 0.8")
    ET.SubElement(obj, "freejoint", name="obj_freejoint")
    ET.SubElement(obj, "geom", type="sphere", size="0.04", mass="0.2",
                  contype="0", conaffinity="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def identifier(kind, name):
        return mujoco.mj_name2id(model, kind, name)

    obj_joint = identifier(mujoco.mjtObj.mjOBJ_JOINT, "obj_freejoint")
    ids = dict(obj_qpos=int(model.jnt_qposadr[obj_joint]),
               obj_body=identifier(mujoco.mjtObj.mjOBJ_BODY, "obj"))
    for side in ("left", "right"):
        joints = [identifier(mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint{i}") for i in range(1, 8)]
        ids[f"{side}_qpos"] = model.jnt_qposadr[joints].copy()
        ids[f"{side}_act"] = np.array([
            identifier(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_actuator{i}") for i in range(1, 8)])
        site = identifier(mujoco.mjtObj.mjOBJ_SITE, f"{side}_tip_center")
        ids[f"{side}_tip"] = int(model.site_bodyid[site])
        ids[f"{side}_tip_offset"] = model.site_pos[site].copy()
        ids[f"{side}_forearm"] = identifier(mujoco.mjtObj.mjOBJ_BODY, f"{side}_link4")
        ids[f"{side}_wrist"] = identifier(mujoco.mjtObj.mjOBJ_BODY, f"{side}_link6")
    return SimpleNamespace(cpu_model=model, _ids=ids, nq=model.nq, nv=model.nv, device_name="cpu")


def _contact_inputs(step, radius=0.01):
    qpos = step.cpu_model.qpos0.copy()
    hand_indices = np.r_[step._ids["left_qpos"], step._ids["right_qpos"]]
    qpos[hand_indices] = np.tile([0.12, -0.16, 0.18, -0.10, 0.15, 0.08, -0.12], 2)
    data = mujoco.MjData(step.cpu_model)
    data.qpos[:] = qpos
    mujoco.mj_forward(step.cpu_model, data)
    inward = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    centers = np.stack([
        data.xpos[step._ids[f"{side}_tip"]]
        + data.xmat[step._ids[f"{side}_tip"]].reshape(3, 3) @ step._ids[f"{side}_tip_offset"]
        for side in ("left", "right")])
    pose = np.r_[centers.mean(axis=0), 1.0, 0.0, 0.0, 0.0]
    qpos[step._ids["obj_qpos"]:step._ids["obj_qpos"] + 7] = pose
    contacts = centers + radius * inward - pose[:3]
    return qpos, pose, contacts, inward


def _world_points(pose, contacts):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, pose[3:])
    return pose[:3] + contacts @ rotation.reshape(3, 3).T


class TestPandaContactAware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_dexforge()
        except ImportError as error:
            raise unittest.SkipTest(str(error)) from error

    def test_singleton_pose_converts_wxyz_and_normalizes(self):
        pose = np.array([1, 2, 3, 2, 0, 0, 2], dtype=float)
        normalized = singleton_pose(pose)
        self.assertEqual(normalized.shape, (1, 7))
        np.testing.assert_allclose(normalized[0, 3:], [2**-0.5, 0, 0, 2**-0.5])
        np.testing.assert_allclose(pose_xyzw(pose)[0], [0, 0, 2**-0.5, 2**-0.5, 1, 2, 3])
        np.testing.assert_array_equal(pose, [1, 2, 3, 2, 0, 0, 2])
        for invalid in ([0, 0, 0, 0, 0, 0, 0], [np.nan, 0, 0, 1, 0, 0, 0]):
            with self.assertRaises(ValueError):
                singleton_pose(invalid)

    def test_mujoco_point_and_normal_jacobians(self):
        step = _fast_step()
        hand = _PandaQueries(step, 0.01)
        qpos, _, _, _ = _contact_inputs(step)
        hand.base_qpos = qpos.copy()
        hand.query_local += [[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]
        hand.local_normal[:] = [[0.7, 0.3, 0.2], [-0.4, 0.5, 0.6]]
        q = qpos[hand.qadr]
        positions, jacobian = hand.query_positions_jacobian(q, [0, 1])
        normals, normal_jacobian = hand.query_normals_jacobian(q, [0, 1])
        numeric, normal_numeric = np.zeros_like(jacobian), np.zeros_like(normal_jacobian)
        epsilon = 1.e-6
        for column in range(14):
            offset = np.eye(14)[column] * epsilon
            plus = hand.query_positions_jacobian(q + offset, [0, 1])[0]
            minus = hand.query_positions_jacobian(q - offset, [0, 1])[0]
            numeric[:, :, column] = (plus - minus) / (2 * epsilon)
            plus = hand.query_normals_jacobian(q + offset, [0, 1])[0]
            minus = hand.query_normals_jacobian(q - offset, [0, 1])[0]
            normal_numeric[:, :, column] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(jacobian, numeric, atol=1.e-8)
        np.testing.assert_allclose(normal_jacobian, normal_numeric, atol=1.e-8)
        self.assertEqual(positions.shape, (2, 3))
        self.assertEqual(normals.shape, (2, 3))
        np.testing.assert_array_equal(jacobian[0, :, 7:], np.zeros((3, 7)))
        np.testing.assert_array_equal(jacobian[1, :, :7], np.zeros((3, 7)))

    def test_native_ca_normal_sign_radius_and_joint_bounds(self):
        from contactaware.solver.contact import anchor_normal_linearization

        step = _fast_step()
        qpos, pose, contacts, inward = _contact_inputs(step)
        solver = PandaContactAware(step, tip_radius=0.01, iterations=30)
        result = solver.solve(qpos, object_pose_wxyz=pose, contact_points_local=contacts,
                              normals_local=inward * 4.0)
        np.testing.assert_allclose(result["normals_local"], inward[None])
        np.testing.assert_allclose(result["contact_runtime"].anchors.anchor_normal_obj, -inward[None])
        hand = solver.hand
        hand.forward(qpos[hand.qadr])
        for i, body in enumerate(hand.body):
            delta = result["query_local"][0, i] - hand.tip_offset[i]
            np.testing.assert_allclose(np.linalg.norm(delta), 0.01, atol=1.e-12)
            np.testing.assert_allclose(hand.data.xmat[body].reshape(3, 3) @ delta, inward[i] * 0.01,
                                       atol=1.e-12)
        residual, _, strength = anchor_normal_linearization(
            hand, qpos[hand.qadr], pose_xyzw(pose)[0], cfg=solver.cfg,
            runtime=result["contact_runtime"], frame_id=0)
        np.testing.assert_allclose(residual, np.zeros((2, 3)), atol=1.e-12)
        np.testing.assert_array_equal(strength, np.ones(2))
        np.testing.assert_allclose(result["diagnostics"]["contact_error"], 0, atol=1.e-8)
        np.testing.assert_array_equal(hand.lower, np.full(14, -1.2))
        np.testing.assert_array_equal(hand.upper, np.full(14, 1.2))

    def test_surface_gs_inflation_allows_the_physical_pad_to_reach_its_anchor(self):
        step = _fast_step()
        qpos, pose, contacts, inward = _contact_inputs(step, radius=.007)
        step.spheres_local = np.column_stack((contacts, np.full(2, .004)))
        solver = PandaContactAware(step)
        result = solver.solve(qpos, object_pose_wxyz=pose, contact_points_local=contacts,
                              normals_local=inward, approach_offset=.007)
        np.testing.assert_allclose(result["diagnostics"]["contact_error"], 0., atol=1.e-8)
        assert result["diagnostics"]["collision_clearance"] >= -1.e-8

    def test_approach_reference_uses_measured_object_until_bilateral_contact(self):
        step = _fast_step()
        qpos, measured, contacts, inward = _contact_inputs(step)
        predicted = measured.copy()
        predicted[2] += .06
        predicted[3:] = [np.cos(.3), 0., np.sin(.3), 0.]

        class FakeFA:
            def __init__(self, *args, **kwargs):
                pass

            def solve(self, qpos, qvel, **kwargs):
                return {"object_reference": kwargs["object_pose_wxyz"]}

        params = SimpleNamespace(adam_horizon=2, adam_knot_substeps=6, adam_iters=1,
            adam_lr=.0015, planner_joint_delta_limit=.15, tip_radius=.01,
            retarget_object_speed=.05, retarget_object_angular_speed=.5)
        with patch("planning.dexforge_forceaware.PandaForceAware", FakeFA):
            planner = DexForgeRetargetingMPC(params, step)
        inputs = dict(x_plus_opt=predicted, contact_points_local=contacts,
                      normals_local=inward, qpos0=qpos, qvel0=np.zeros(step.nv))
        approach = planner.plan_once(**inputs, contact_established=False)
        np.testing.assert_array_equal(approach["object_reference"], measured[None])
        assert approach["retargeting"]["phase"] == "approach"
        np.testing.assert_array_equal(approach["retargeting"]["lambda_x_plus_opt"], predicted[None])
        manipulation = planner.plan_once(**inputs, contact_established=True)
        reference = manipulation["object_reference"][0]
        assert reference[2] == measured[2] + .003
        angle = 2 * np.arccos(np.clip(reference[3:] @ measured[3:], -1., 1.))
        assert np.isclose(angle, .03)
        assert manipulation["retargeting"]["phase"] == "manipulate"

    def test_object_pose_changes_drive_native_ca_joint_reference(self):
        step = _fast_step()
        qpos, pose, contacts, inward = _contact_inputs(step)
        solver = PandaContactAware(step, iterations=40)
        for translation, angle in (([0, 0.01, -0.015], 0.0), ([0, 0, 0], 0.12)):
            target = pose.copy()
            target[:3] += translation
            target[3:] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
            solver.reset()
            result = solver.solve(qpos, object_pose_wxyz=target, contact_points_local=contacts,
                                  normals_local=inward)
            hand = solver.hand
            initial_points = hand.query_positions_jacobian(qpos[hand.qadr], [0, 1])[0]
            initial_error = np.linalg.norm(initial_points - _world_points(target, contacts), axis=1)
            final_error = result["diagnostics"]["contact_error"]
            self.assertLess(np.linalg.norm(final_error), 0.20 * np.linalg.norm(initial_error))
            self.assertGreater(np.linalg.norm(result["hand_reference"][0] - qpos[hand.qadr]), 0.01)
            self.assertTrue(np.all(result["hand_reference"] >= hand.lower))
            self.assertTrue(np.all(result["hand_reference"] <= hand.upper))
            np.testing.assert_array_equal(result["object_pose_wxyz"], target[None])

    def test_stale_ca_reference_cannot_seed_a_new_object_collision(self):
        step = _fast_step()
        qpos, pose, _, inward = _contact_inputs(step)
        hand = _PandaQueries(step, 0.01)
        hand.base_qpos = qpos.copy()
        initial = qpos[hand.qadr]
        hand.forward(initial)
        for i, body in enumerate(hand.body):
            hand.local_normal[i] = hand.data.xmat[body].reshape(3, 3).T @ inward[i]
        hand.query_local = hand.tip_offset + 0.01 * hand.local_normal
        stale = initial.copy()
        stale[0] += 0.6
        stale_queries = hand.query_positions_jacobian(stale, [0, 1])[0]
        stale_tip = hand.point(hand.body[0], hand.tip_offset[0])[0]
        # A moved object sphere now occupies the cached left-tip position.
        # The new anchors prefer the old reference, but its collision rejects it.
        step.spheres_local = np.r_[stale_tip - pose[:3], 0.01][None]
        solver = PandaContactAware(step, iterations=1)
        solver._reference = stale.copy()
        with patch.object(solver.qp, "solve", return_value=SimpleNamespace(solution=np.zeros(14))):
            result = solver.solve(qpos, object_pose_wxyz=pose,
                                  contact_points_local=stale_queries - pose[:3], normals_local=inward)
        np.testing.assert_array_equal(result["hand_reference"][0], initial)
        self.assertGreater(result["diagnostics"]["collision_clearance"], 0.0)

    def test_facade_replans_from_x_plus_and_reset_clears_stages(self):
        step = _fast_step()
        qpos, pose, contacts, inward = _contact_inputs(step)

        class FakeFA:
            def __init__(self, *args, **kwargs):
                self.calls = []
                self.resets = 0

            def solve(self, qpos, qvel, **kwargs):
                self.calls.append((qpos.copy(), qvel.copy(), kwargs))
                return {"action": kwargs["hand_reference"][0] - qpos[np.r_[step._ids["left_qpos"], step._ids["right_qpos"]]]}

            def reset(self):
                self.resets += 1

        params = SimpleNamespace(adam_horizon=2, adam_knot_substeps=3, adam_iters=1,
                                 adam_lr=0.001, planner_joint_delta_limit=0.15,
                                 tip_radius=0.01, retarget_ca_iters=30)
        with patch("planning.dexforge_forceaware.PandaForceAware", FakeFA):
            planner = DexForgeRetargetingMPC(params, step)
        qvel = np.linspace(0, 0.01, step.nv)
        for offset in (0.01, -0.015):
            target = pose.copy()
            target[1] += offset
            current = qpos.copy()
            current[0] += offset
            result = planner.plan_once(target_p=np.full(3, 9.0), target_q=[0, 1, 0, 0],
                                       curr_x=np.ones(21), x_plus_opt=target,
                                       contact_points_local=contacts, normals_local=inward,
                                       qpos0=current, qvel0=qvel)
            observed_q, observed_v, ca = planner.fa.calls[-1]
            np.testing.assert_array_equal(observed_q, current)
            np.testing.assert_array_equal(observed_v, qvel)
            np.testing.assert_array_equal(ca["object_pose_wxyz"], target[None])
            np.testing.assert_array_equal(ca["contact_points_local"], contacts[None])
            self.assertEqual(ca["hand_reference"].shape, (1, 14))
            self.assertEqual(ca["query_local"].shape, (1, 2, 3))
            self.assertEqual(result["retargeting"]["source_frames"], 1)
            np.testing.assert_array_equal(result["retargeting"]["x_plus_opt"], target[None])
        self.assertEqual(len(planner.fa.calls), 2)
        self.assertIsNotNone(planner.ca._reference)
        planner.reset()
        self.assertIsNone(planner.ca._reference)
        self.assertEqual(planner.fa.resets, 1)

    def test_facade_dispatches_mppi_options_and_keeps_default_adam_reference(self):
        step = _fast_step()
        qpos, pose, contacts, inward = _contact_inputs(step)

        class FakeFA:
            def __init__(self, model, **options):
                self.model, self.options = model, options
                self.references = []

            def solve(self, qpos, qvel, **references):
                self.references.append(references)
                return {"action": np.zeros(14)}

        params = SimpleNamespace(adam_horizon=3, adam_knot_substeps=2, adam_iters=11,
                                 adam_lr=0.002, planner_joint_delta_limit=0.12,
                                 tip_radius=0.01, retarget_ca_iters=4, mppi_samples_=17,
                                 mppi_iterations_=3, mppi_init_iterations_=6, mppi_lambda_=0.2,
                                 mppi_noise_sigma_=0.025, mppi_noise_decay_=0.7, mppi_seed_=9)
        target = pose.copy()
        target[1] -= 0.005
        references = []
        with patch("planning.dexforge_forceaware.PandaForceAware", FakeFA):
            default = DexForgeRetargetingMPC(params, step)
            explicit_adam = DexForgeRetargetingMPC(params, step, solver="adam")
            mppi = DexForgeRetargetingMPC(params, step, solver="mppi")
            with self.assertRaises(ValueError):
                DexForgeRetargetingMPC(params, step, solver="unknown")
        self.assertEqual(default.fa.options, explicit_adam.fa.options)
        self.assertEqual(default.fa.options["solver"], "adam")
        self.assertEqual(default.fa.options["iters"], 11)
        self.assertEqual(default.fa.options["lr"], 0.002)
        expected = dict(solver="mppi", num_samples=17, mppi_iterations=3,
                        mppi_init_iterations=6, temperature=0.2, noise_scale=0.025,
                        noise_decay=0.7, seed=9, horizon=3, knot_substeps=2, u_limit=0.12)
        for key, value in expected.items():
            self.assertEqual(mppi.fa.options[key], value)
        for planner in (default, explicit_adam, mppi):
            result = planner.plan_once(target_p=[10, 20, 30], target_q=[1, 0, 0, 0],
                                       x_plus_opt=target, contact_points_local=contacts,
                                       normals_local=inward, qpos0=qpos, qvel0=np.zeros(step.nv))
            observed = planner.fa.references[-1]
            np.testing.assert_array_equal(observed["object_pose_wxyz"], target[None])
            self.assertEqual(result["retargeting"]["source_frames"], 1)
            references.append(observed)
        for observed in references[1:]:
            for name in ("hand_reference", "object_pose_wxyz", "contact_points_local", "normals_local", "query_local"):
                np.testing.assert_array_equal(observed[name], references[0][name])


class _Array:
    def __init__(self, value):
        self.value = np.asarray(value).copy()

    def numpy(self):
        return self.value.copy()

    def assign(self, value):
        self.value = np.asarray(value).copy()


class _WindowOptimizer:
    """Observe real window construction without compiling Warp dynamics."""
    def __init__(self, step, index, cfg, geometry):
        self.step, self.index, self.cfg = step, index, cfg
        self.calls = []
        self.action_lower, self.action_upper = _Array([]), _Array([])
        self.objective = SimpleNamespace(regularizer=SimpleNamespace(_previous_previous_cpu=None))
        contacts = SimpleNamespace(active=_Array([[1, 0, 1, 0, 1, 1]]))
        force = SimpleNamespace(force=_Array([[1., 2., 3., 4., 5., 6.]]))
        runtime = SimpleNamespace(workspace=SimpleNamespace(forward=SimpleNamespace(
            base=SimpleNamespace(collision=SimpleNamespace(contacts=contacts)),
            contact=SimpleNamespace(output=force))))
        self.steps = [runtime] * cfg.optimizer.horizon * cfg.simulator.time_grid.knot_substeps

    def solve(self, qpos, qvel, targets, **kwargs):
        self.calls.append((qpos.copy(), qvel.copy(), targets, kwargs))
        controls = np.clip(targets.actions, self.action_lower.value, self.action_upper.value)
        raw = np.linspace(0, 0.01, 14 * (1 + len(controls)), dtype=np.float32)
        fractions = np.arange(1, kwargs["execution_steps"] + 1) / kwargs["execution_steps"]
        previous = kwargs["previous_action"]
        executed = previous + fractions[:, None] * (controls[0] - previous)
        return SimpleNamespace(action=controls[0], loss=1.25, seconds=0.001, raw=raw,
                               control_sequence=executed, qpos_sequence=np.tile(qpos, (len(executed), 1)),
                               diag={"mock": True, "selected": 0}, drift=0.0, grad_norm=0.5)


class _MppiWindowOptimizer(_WindowOptimizer):
    def __init__(self, *args, **options):
        super().__init__(*args)
        self.mppi_options = options
        self.resets = 0

    def reset(self):
        self.resets += 1


class TestPandaForceAwareTargets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            ensure_dexforge()
        except ImportError as error:
            raise unittest.SkipTest(str(error)) from error

    def _adapter(self, *, horizon=3, substeps=2, solver="adam", **solver_options):
        from forceaware import config
        from planning import dexforge_forceaware as adapter

        step = _fast_step()
        # Deliberately reorder outputs to exercise upstream semantic mapping.
        source = [step._ids["left_forearm"], step._ids["right_tip"],
                  step._ids["left_tip"], step._ids["right_forearm"],
                  step._ids["left_tip"], step._ids["right_tip"]]
        # The duplicate tip outputs are cross-arm contacts and must be filtered.
        pairs = [[step.cpu_model.body_geomadr[body], step.cpu_model.body_geomadr[step._ids["obj_body"]]]
                 for body in source[:4]]
        pairs += [[step.cpu_model.body_geomadr[step._ids["left_tip"]],
                   step.cpu_model.body_geomadr[step._ids["right_tip"]]]] * 2
        collision = SimpleNamespace(output_body_ids=_Array(source),
                                    output_contact_rows=_Array(np.arange(6)),
                                    contact_layout=SimpleNamespace(contact_geom=_Array(pairs)))
        contact_rows = SimpleNamespace(parameters=SimpleNamespace(row_contact=_Array(np.arange(6))))
        step.compiled = SimpleNamespace(base=SimpleNamespace(collision=collision, contact_rows=contact_rows))
        api = SimpleNamespace(config=config, PandaWindowOptimizer=_WindowOptimizer,
                              PandaMppiWindowOptimizer=_MppiWindowOptimizer)
        with patch.object(adapter, "_classes", return_value=api), patch.object(adapter, "_panda_geometry", return_value=None):
            fa = adapter.PandaForceAware(step, horizon=horizon, knot_substeps=substeps,
                                         iters=2, u_limit=0.04, solver=solver, **solver_options)
        return fa, step, api

    def test_native_make_window_holds_one_frame_and_preserves_ca_queries(self):
        from planning import dexforge_forceaware as adapter

        fa, step, api = self._adapter()
        qpos, pose, contacts, inward = _contact_inputs(step)
        ca = PandaContactAware(step).solve(qpos, object_pose_wxyz=pose,
                                           contact_points_local=contacts, normals_local=inward)
        target = pose.copy()
        target[:3] += [0.012, -0.008, 0.006]
        target[3:] = [2 * np.cos(0.1), 0, 0, 2 * np.sin(0.1)]
        hand = ca["hand_reference"] + 0.08
        with patch.object(adapter, "_classes", return_value=api):
            result = fa.solve(qpos, np.zeros(step.nv), hand_reference=hand,
                              object_pose_wxyz=target[None], contact_points_local=contacts[None],
                              normals_local=inward[None], query_local=ca["query_local"])
        _, _, window, options = fa.optimizer.calls[-1]
        self.assertEqual(window.actions.shape, (3, 14))
        self.assertEqual(window.dense_actions.shape, (6, 14))
        self.assertEqual(window.object_qpos.shape, (6, 7))
        np.testing.assert_allclose(window.actions, np.tile(hand, (3, 1)))
        np.testing.assert_allclose(window.object_qpos, np.tile(singleton_pose(target), (6, 1)), atol=1.e-7)
        np.testing.assert_array_equal(window.frames, np.zeros(6))
        np.testing.assert_array_equal(window.contact_output, np.tile([2, 1], (6, 1)))
        np.testing.assert_allclose(window.contact_position_object, np.tile(contacts[None], (6, 1, 1)))
        np.testing.assert_allclose(window.query_local, np.tile(ca["query_local"], (6, 1, 1)))
        np.testing.assert_array_equal(window.contact_mask, np.ones((6, 2)))
        self.assertFalse(options["allow_init"])
        self.assertEqual(options["execution_steps"], 2)
        self.assertEqual(result["solver_backend"], "dexforge_forceaware")
        self.assertEqual(result["action"].shape, (14,))
        self.assertTrue(np.all(np.abs(result["action"]) <= 0.040001))
        self.assertEqual(result["control_sequence"].shape, (2, 14))
        np.testing.assert_array_equal(result["control_sequence"][-1], result["ctrl"])
        np.testing.assert_array_equal(result["contact_mask"], [[True, False], [True, False]])
        np.testing.assert_array_equal(result["normal_force"], [[3., 2.], [3., 2.]])
        self.assertGreater(fa.cfg.loss.obj_pos, 0.0)
        self.assertGreater(fa.cfg.loss.obj_rot, 0.0)

    def test_fa_replan_uses_new_state_and_uniform_angular_limits_then_resets(self):
        from planning import dexforge_forceaware as adapter

        fa, step, api = self._adapter(horizon=2)
        qpos, pose, contacts, inward = _contact_inputs(step)
        hand_indices = np.r_[step._ids["left_qpos"], step._ids["right_qpos"]]
        qpos[hand_indices] = np.linspace(-1.19, 1.19, 14)
        arguments = dict(hand_reference=np.zeros((1, 14)), object_pose_wxyz=pose[None],
                         contact_points_local=contacts[None], normals_local=inward[None],
                         query_local=np.zeros((1, 2, 3)))
        with patch.object(adapter, "_classes", return_value=api):
            first = fa.solve(qpos, np.zeros(step.nv), **arguments)
            for knot in range(2):
                reach = 0.04 * (knot + 1)
                np.testing.assert_allclose(fa.optimizer.action_lower.value[knot],
                                           np.maximum(-1.2, qpos[hand_indices] - reach), atol=1.e-7)
                np.testing.assert_allclose(fa.optimizer.action_upper.value[knot],
                                           np.minimum(1.2, qpos[hand_indices] + reach), atol=1.e-7)
            updated = qpos.copy()
            updated[hand_indices] = first["ctrl"]
            updated_velocity = np.linspace(-0.1, 0.1, step.nv)
            fa.solve(updated, updated_velocity, **arguments)
        observed_q, observed_v, _, options = fa.optimizer.calls[-1]
        np.testing.assert_array_equal(observed_q, updated.astype(np.float32))
        np.testing.assert_array_equal(observed_v, updated_velocity.astype(np.float32))
        np.testing.assert_array_equal(options["previous_action"], first["ctrl"])
        self.assertTrue(np.any(options["warm"] != 0))
        fa.reset()
        self.assertIsNone(fa._previous_action)
        np.testing.assert_array_equal(fa._warm, np.zeros(42))

    def test_mppi_and_adam_share_native_fa_targets_loss_and_dynamics_config(self):
        from planning import dexforge_forceaware as adapter

        adam, adam_step, adam_api = self._adapter()
        mppi, mppi_step, mppi_api = self._adapter(
            solver="mppi", num_samples=97, mppi_iterations=3, mppi_init_iterations=5,
            temperature=0.2, noise_scale=0.025, noise_decay=0.7, seed=41)
        self.assertIsInstance(adam.optimizer, _WindowOptimizer)
        self.assertNotIsInstance(adam.optimizer, _MppiWindowOptimizer)
        self.assertIsInstance(mppi.optimizer, _MppiWindowOptimizer)
        from forceaware.numerics import LOSS_GRADIENT_SCALE

        self.assertEqual(mppi.optimizer.mppi_options, dict(
            iterations=3, init_iterations=5, temperature=0.2, noise_scale=0.025,
            noise_decay=0.7, seed=41, cost_scale=float(LOSS_GRADIENT_SCALE)))
        self.assertEqual(adam.cfg.optimizer.multistart_count, 2)
        self.assertEqual(mppi.cfg.optimizer.multistart_count, 97)
        self.assertEqual(adam.cfg.loss, mppi.cfg.loss)
        self.assertEqual(adam.cfg.contact, mppi.cfg.contact)
        self.assertEqual(adam.cfg.smoothing, mppi.cfg.smoothing)
        self.assertEqual(adam.cfg.simulator, mppi.cfg.simulator)
        qpos, pose, contacts, inward = _contact_inputs(adam_step)
        target = pose.copy()
        target[:3] += [0, 0.01, -0.01]
        arguments = dict(hand_reference=qpos[np.r_[adam_step._ids["left_qpos"], adam_step._ids["right_qpos"]]][None],
                         object_pose_wxyz=target[None], contact_points_local=contacts[None],
                         normals_local=inward[None], query_local=np.zeros((1, 2, 3)))
        with patch.object(adapter, "_classes", return_value=adam_api):
            adam_result = adam.solve(qpos, np.zeros(adam_step.nv), **arguments)
        with patch.object(adapter, "_classes", return_value=mppi_api):
            mppi_result = mppi.solve(qpos, np.zeros(mppi_step.nv), **arguments)
        for name in ("action", "control_sequence", "object_reference", "hand_reference", "query_local", "contact_mask", "normal_force"):
            np.testing.assert_array_equal(mppi_result[name], adam_result[name])
        self.assertEqual(adam_result["solver_backend"], "dexforge_forceaware")
        self.assertEqual(mppi_result["solver_backend"], "dexforge_forceaware_mppi")
        adam_window = adam.optimizer.calls[-1][2]
        mppi_window = mppi.optimizer.calls[-1][2]
        for name in ("actions", "dense_actions", "object_qpos", "contact_mask", "contact_weight",
                     "contact_position_object", "contact_output", "query_body", "query_local", "frames", "knot_frames"):
            np.testing.assert_array_equal(getattr(mppi_window, name), getattr(adam_window, name))
        mppi.reset()
        self.assertEqual(mppi.optimizer.resets, 1)
        self.assertIsNone(mppi._previous_action)
        np.testing.assert_array_equal(mppi._warm, np.zeros(56))

    def test_warm_start_preserves_absolute_future_knots_when_ca_reference_changes(self):
        from planning import dexforge_forceaware as adapter

        for solver in ("adam", "mppi"):
            with self.subTest(solver=solver):
                fa, step, api = self._adapter(solver=solver)
                qpos, pose, contacts, inward = _contact_inputs(step)
                hand_indices = np.r_[step._ids["left_qpos"], step._ids["right_qpos"]]
                reference = qpos[hand_indices][None] + 0.015
                arguments = dict(object_pose_wxyz=pose[None], contact_points_local=contacts[None],
                                 normals_local=inward[None], query_local=np.zeros((1, 2, 3)))
                with patch.object(adapter, "_classes", return_value=api):
                    first = fa.solve(qpos, np.zeros(step.nv), hand_reference=reference, **arguments)
                    raw = first["sol_guess"]["raw"]
                    absolute = reference + raw[14:].reshape(3, 14)
                    future = np.vstack((absolute[1:], absolute[-1:]))
                    np.testing.assert_allclose(fa._warm_absolute, future, atol=1.e-7)
                    changed = reference + np.linspace(-0.04, 0.04, 14)[None]
                    fa.solve(qpos, np.zeros(step.nv), hand_reference=changed, **arguments)
                _, _, window, options = fa.optimizer.calls[-1]
                warm = options["warm"]
                np.testing.assert_array_equal(warm[:14], np.zeros(14))
                reconstructed = window.actions + warm[14:].reshape(3, 14)
                np.testing.assert_allclose(reconstructed, future, atol=1.e-7)
                fa.reset()
                self.assertIsNone(fa._warm_absolute)

    def test_lagging_measured_joints_keep_executed_command_and_smoothing_history(self):
        from planning import dexforge_forceaware as adapter

        for solver in ("adam", "mppi"):
            with self.subTest(solver=solver):
                fa, step, api = self._adapter(solver=solver)
                qpos, pose, contacts, inward = _contact_inputs(step)
                hand_indices = np.r_[step._ids["left_qpos"], step._ids["right_qpos"]]
                arguments = dict(object_pose_wxyz=pose[None], contact_points_local=contacts[None],
                                 normals_local=inward[None], query_local=np.zeros((1, 2, 3)))
                with patch.object(adapter, "_classes", return_value=api):
                    first = fa.solve(qpos, np.zeros(step.nv),
                                     hand_reference=qpos[hand_indices][None] + 0.035, **arguments)
                    lagging = qpos.copy()
                    lagging[hand_indices] = first["ctrl"] - 0.35
                    history = first["ctrl"] - 0.02
                    fa.optimizer.objective.regularizer._previous_previous_cpu = history.copy()
                    second = fa.solve(lagging, np.zeros(step.nv),
                                      hand_reference=lagging[hand_indices][None] + 0.02, **arguments)
                observed_qpos, _, _, options = fa.optimizer.calls[-1]
                np.testing.assert_array_equal(observed_qpos, lagging.astype(np.float32))
                np.testing.assert_array_equal(options["previous_action"], first["ctrl"])
                np.testing.assert_array_equal(fa.optimizer.objective.regularizer._previous_previous_cpu, history)
                # The first dense target continues from the actual previous
                # command, even though the joint has not yet reached it.
                expected = first["ctrl"] + (second["ctrl"] - first["ctrl"]) / 2
                np.testing.assert_allclose(second["control_sequence"][0], expected, atol=1.e-7)

    def test_mppi_loss_scale_defaults_to_native_gradient_units_and_raw_units_are_optional(self):
        from forceaware.numerics import LOSS_GRADIENT_SCALE

        default, _, _ = self._adapter(solver="mppi", temperature=0.003)
        raw, _, _ = self._adapter(solver="mppi", temperature=0.003, cost_scale=1.0)
        self.assertEqual(default.optimizer.mppi_options["cost_scale"], float(LOSS_GRADIENT_SCALE))
        self.assertEqual(raw.optimizer.mppi_options["cost_scale"], 1.0)
        self.assertEqual(default.optimizer.mppi_options["temperature"], 0.003)
        self.assertEqual(raw.optimizer.mppi_options["temperature"], 0.003)
        self.assertEqual(default.cfg.loss, raw.cfg.loss)
        self.assertEqual(default.cfg.smoothing, raw.cfg.smoothing)
        for scale in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(cost_scale=scale), self.assertRaisesRegex(ValueError, "cost_scale"):
                self._adapter(solver="mppi", cost_scale=scale)


if __name__ == "__main__":
    unittest.main()
