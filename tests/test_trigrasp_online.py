"""n-contact force physics and live Allegro MPC on a physical MuJoCo scene."""

import importlib.util
from pathlib import Path

import casadi as cs
import numpy as np
import pytest
import trimesh

from planning.lambda_batched_lm import BatchedLambdaLM
from planning.lambda_contact_lm import LambdaContactBatchedLM, LambdaContactMultiLM


def test_circular_lambda_forces_obey_the_mpc_coulomb_disk():
    points = np.array([[[.04, 0., 0.], [-.04, .03, 0.], [-.04, -.03, 0.]]])
    normals = np.array([[[-1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]])
    solver = BatchedLambdaLM(num_contacts=3, friction_cone="coulomb", device="cpu",
        iterations=5, mu=.7, max_contact_force=1.5)
    result = solver.solve(points, normals, current_x=[0., 0., 0., 1., 0., 0., 0.],
        x_d=[.02, -.03, .1, 1., 0., 0., 0.], prediction_dt=.025,
        object_inertia=np.diag([.2]*3+[.004]*3),
        initial_forces=np.array([[[10., 10., 10.], [-5., 10., -10.], [3., -5., 6.]]]))
    force = result["control_input"]
    assert np.all(force[:, 0] >= 0)
    assert np.all(np.linalg.norm(force[:, 1:], axis=1) <= .7*force[:, 0]+1e-6)
    assert np.all(np.linalg.norm(force, axis=1) <= 1.5+1e-6)


@pytest.mark.parametrize("n", [1, 3, 4])
def test_n_contact_lm_predicts_sum_of_all_contact_wrenches(n):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    points = np.c_[.04*np.cos(angles), .04*np.sin(angles), np.zeros(n)]
    normals = -points/.04
    if n == 1:
        points[:] = 0.
        normals[:] = [0., 0., 1.]
    current = np.array([0., 0., 0., 1., 0., 0., 0.])
    target = current.copy()
    target[2] = .05
    inertia = np.diag([.2, .2, .2, .004, .004, .004])
    solver = BatchedLambdaLM(num_contacts=n, device="cpu", iterations=20, max_contact_force=3.)
    result = solver.solve(points[None], normals[None], current_x=current, x_d=target,
        object_inertia=inertia, prediction_dt=.1)
    assert result["control_input"].shape == (n, 3)
    assert result["baseline_reduction"] > 0
    if n > 1:
        assert result["pose_reduction"] > 0
    forces = result["force_vectors_local"]
    wrench = np.r_[forces.sum(0), np.cross(points, forces).sum(0)]
    expected = .1*np.linalg.solve(inertia, wrench + np.array([0., 0., -.2*9.81, 0., 0., 0.]))
    np.testing.assert_allclose(result["velocity_next"], expected, atol=2e-6)
    np.testing.assert_allclose(result["contact_force_world"], forces.sum(0), atol=2e-6)
    np.testing.assert_allclose(result["contact_torque_world"], np.cross(points, forces).sum(0), atol=2e-6)
    coefficients = result["control_input"]
    assert np.all(coefficients[:, 0] >= 0)
    assert np.all(np.abs(coefficients[:, 1:]) <= .9*coefficients[:, :1] + 1e-6)
    assert np.all(np.linalg.norm(forces, axis=1) <= 3. + 1e-6)


@pytest.mark.parametrize("n", [3, 4])
def test_shared_lambda_policy_keeps_fixed_fingertip_assignment(n):
    policy = LambdaContactBatchedLM(trimesh.creation.box(extents=[.08, .08, .08]),
        num_contacts=n, candidate_capacity=16, device="cpu", sample_num=40,
        max_point_curvature=1., max_mean_curvature=1., lm_iterations=5)
    assert isinstance(policy, LambdaContactMultiLM)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    tips = np.c_[.1*np.cos(angles), .1*np.sin(angles), np.full(n, .3)]
    current = np.array([0., 0., .3, 1., 0., 0., 0.])
    target = current + np.array([0., 0., .05, 0., 0., 0., 0.])
    dynamics = dict(tips_world=tips, object_inertia=np.diag([.2]*3+[.004]*3), prediction_dt=.1)
    _, _, _, _, initial = policy.initialize_offline(target, current, **dynamics)
    ids = initial["contact_indices"][::-1]
    p, normals, _, _, fixed = policy.update_fixed_contacts(ids, target, current, **dynamics)
    np.testing.assert_array_equal(fixed["contact_indices"], ids)
    np.testing.assert_array_equal(p, policy.sample_point[ids])
    np.testing.assert_array_equal(normals, policy.normal[ids])
    _, _, _, _, live = policy.update_contacts(target, current, freeze_contact_pair=True, **dynamics)
    np.testing.assert_array_equal(live["contact_indices"], ids)
    assert live["control_input"].shape == (n, 3)
    policy.reset()
    assert policy.active_ids is None


@pytest.fixture(scope="module")
def demo_module():
    path = Path(__file__).resolve().parents[1]/"examples/mpc/franka/trigrasp/trigrasp.py"
    spec = importlib.util.spec_from_file_location("trigrasp_online_test", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("n", [3, 4])
def test_physical_allegro_mpc_approaches_live_contact_goals(demo_module, tmp_path, n, monkeypatch):
    args = demo_module.build_argparser().parse_args([
        "--headless", "--no-video", "--screenshot-interval", "0", "--mpc-solver", "ipopt",
        "--scene-output", str(tmp_path/"scene.xml"), "--num-grasp-contacts", str(n),
        "--mpc-horizon", "2", "--max-point-combination-eval", "16", "--lambda-lm-iters", "5"])
    assert not args.show
    demo = demo_module.AllegroTabletopTrigraspDemo(args)
    try:
        assert demo.model.body_mass[demo.object_body_id] == pytest.approx(args.obj_mass)
        assert demo.model.opt.noslip_iterations == 3
        assert demo.model.opt.tolerance == pytest.approx(1e-8)
        palm, rotation, _ = demo.forward_kinematics(demo.default_hand_q)
        toward_object = demo.initial_object_pos - palm
        toward_object /= np.linalg.norm(toward_object)
        assert np.dot(rotation @ demo_module.PALM_NORMAL_LOCAL, toward_object) > .99
        # All three MuJoCo base rotations are active; verify a general
        # target rather than a yaw-only pose that hides the order error.
        target = demo_module.Rotation.from_euler("xyz", [.3, .4, .5]).as_matrix()
        q = demo.default_hand_q.copy()
        q[3:6] = demo.base_orientation_euler_from_target_palm_rot(target)
        _, actual, _ = demo.forward_kinematics(q)
        np.testing.assert_allclose(actual, target, atol=1e-12)
        wrapped_base = demo_module.Rotation.from_euler("XYZ", [3.16, .02, -.2]).as_matrix()
        wrapped_palm = wrapped_base @ demo.fixed_palm_rot_world
        continuous = demo.base_orientation_euler_from_target_palm_rot(wrapped_palm, [3.13, .02, -.2])
        np.testing.assert_allclose(continuous, [3.16, .02, -.2], atol=1e-12)
        local_tips = demo.local_tip_positions(demo.default_hand_q)[:n]
        top_down = demo.build_top_down_palm_rotation(local_tips, demo.initial_object_pos + local_tips)
        np.testing.assert_allclose(top_down @ demo_module.PALM_NORMAL_LOCAL, [0., 0., -1.], atol=1e-12)
        tips, j = demo.online_tip_kinematics()
        pos, quat, rot = demo.get_object_pose()
        state = demo.get_mpc_state()
        phi, jac = demo.detect_mpc_contacts()
        # Force sensitivities used by RTI must agree with the actual coupled
        # SCM map on a loaded branch, rather than silently dropping tracking.
        model = demo.mpc.model
        model.reference = state[:model.pose_size].copy()
        loaded_phi = phi.copy(); loaded_phi[:4] = -.002
        zero = np.zeros(demo.model.nu)
        mp = model.runtime_parameters(demo.mpc_param.model_params)
        force, fx, fu = model.force_linearization_fn(state, zero, loaded_phi, jac, mp)
        np.testing.assert_allclose(force, model.contact_force_fn(state, zero, loaded_phi, jac, mp))
        rng = np.random.default_rng(13)
        dx, du = rng.normal(size=state.size), rng.normal(size=zero.size)
        eps = 1e-7
        finite = (np.asarray(model.contact_force_fn(state+eps*dx, eps*du, loaded_phi, jac, mp))-
                  np.asarray(model.contact_force_fn(state-eps*dx, -eps*du, loaded_phi, jac, mp)))/(2*eps)
        np.testing.assert_allclose(finite.reshape(-1), np.asarray(fx@dx+fu@du).reshape(-1), rtol=2e-4, atol=2e-5)
        assert args.lambda_prediction_dt == demo.command_dt
        assert args.lift_contact_fingers == 3
        assert demo.optimizer.require_force_closure
        assert demo.optimizer.lm.friction_cone == "coulomb"
        # The blending function receives an already transported attractor;
        # it must not apply object translation a second time.
        from types import SimpleNamespace
        fixed_ik = SimpleNamespace(qpos=demo.default_hand_q.copy())
        opened = fixed_ik.qpos.copy(); opened[6:] -= .02
        shifted_object = demo.initial_object_pos + [.04, -.03, .1]
        ref = demo.online_ik_posture_reference(fixed_ik, closing=True, loaded=False,
            reach_pos=shifted_object, open_posture=opened)
        np.testing.assert_array_equal(ref, fixed_ik.qpos)
        lifted = demo.online_ik_posture_reference(fixed_ik, closing=True, loaded=True,
            reach_pos=shifted_object, open_posture=opened)
        np.testing.assert_allclose(lifted[:3]-ref[:3], [.04, -.03, .1])
        np.testing.assert_array_equal(lifted[3:], ref[3:])
        paused_lift = demo.online_ik_posture_reference(fixed_ik, closing=True, loaded=False,
            reach_pos=shifted_object, open_posture=opened, lift_offset=[0., 0., .02])
        np.testing.assert_allclose(paused_lift[:3]-ref[:3], [0., 0., .02])
        np.testing.assert_array_equal(paused_lift[3:], ref[3:])
        # Feed the same ideal point forces into both object operators.
        # Full coupled inertia, body angular velocity and world torque are
        # deliberately exercised rather than comparing diagonal gravity.
        q = demo_module.Rotation.from_euler("xyz", [.2, -.3, .4]).as_quat()[[3, 0, 1, 2]]
        pose = np.r_[pos, q]
        r = demo_module.quat_wxyz_to_mat(q)
        mass = np.diag([.2, .23, .26, .003, .004, .005])
        mass[0, 4] = mass[4, 0] = .006
        velocity = np.array([.1, -.03, .02, .2, -.1, .15])
        external = np.array([.3, -.5, -1.7, .02, -.03, .01])
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = np.c_[.04*np.cos(angles), .04*np.sin(angles), .01*np.ones(n)]
        normals = -np.c_[np.cos(angles), np.sin(angles), np.zeros(n)]
        ideal = demo.optimizer.lm.solve(points[None], normals[None], current_x=pose,
            x_d=np.r_[pos+[0., 0., .01], q], prediction_dt=demo.command_dt,
            object_inertia=mass, v_obj=velocity, tau_o=external)
        predicted, next_velocity = demo.mpc.model.applied_force_prediction_fn(pose, velocity,
            points.T, ideal["force_vectors_local"].T, np.linalg.inv(mass),
            np.r_[external[:3], r.T@external[3:]])
        np.testing.assert_allclose(np.asarray(predicted).reshape(-1), ideal["x_plus_opt"], atol=2e-6)
        np.testing.assert_allclose(np.asarray(next_velocity).reshape(-1), ideal["velocity_next"], atol=2e-6)
        # Sliding in a contact tangent leaves all four normal gaps equal.
        probe_jac = np.zeros_like(jac)
        probe_jac[:4, :3] = [[1., 1., 0.], [1., 0., 1.], [1., -1., 0.], [1., 0., -1.]]
        model = demo.mpc.model
        model.reference = state[:model.pose_size].copy()
        shifted = state.copy()
        shifted[1] += .02
        gaps = np.asarray(model.contact_gap_fn(shifted, np.zeros_like(phi), probe_jac,
            model.runtime_parameters(demo.mpc_param.model_params))).reshape(-1)
        np.testing.assert_allclose(gaps[:4], 0., atol=1e-12)
        shifted[0] += .01
        gaps = np.asarray(model.contact_gap_fn(shifted, np.zeros_like(phi), probe_jac,
            model.runtime_parameters(demo.mpc_param.model_params))).reshape(-1)
        np.testing.assert_allclose(gaps[:4], .01, atol=1e-12)
        # Move the contact goals down 1 cm. IK only seeds the solve;
        # live tip kinematics and object-frame anchors drive the objective.
        anchors = (tips - pos - [0., 0., .01]) @ rot
        goal = np.r_[pos, quat]
        ik_target = state[7:7+demo.model.nu].copy()
        ik_target[2] -= .01
        result = demo.mpc.plan_contacts(state, phi, jac, tips=tips, tip_jacobian=j, anchors=anchors, goal=goal,
            hand_posture=ik_target, loaded=False)
        assert np.isfinite(result["action"]).all()
        assert result["action"][2] < 0
        assert np.all(result["action"] <= demo.mpc_param.mpc_u_ub_ + 1e-10)
        derivative = cs.Function("finite_prediction_derivative", [demo.mpc.model.prediction_fn.sx_in(0),
            demo.mpc.model.prediction_fn.sx_in(1), demo.mpc.model.prediction_fn.sx_in(2),
            demo.mpc.model.prediction_fn.sx_in(3), demo.mpc.model.prediction_fn.sx_in(4)],
            [cs.jacobian(demo.mpc.model.prediction_fn(*demo.mpc.model.prediction_fn.sx_in()),
                         demo.mpc.model.prediction_fn.sx_in(0))])
        assert np.isfinite(np.asarray(derivative(state, result["action"], phi, jac,
            demo.mpc.model.runtime_parameters(demo.mpc_param.model_params)))).all()
        predicted_state = np.asarray(demo.mpc.model.prediction_fn(state, result["action"], phi, jac,
            demo.mpc.model.runtime_parameters(demo.mpc_param.model_params))).reshape(-1)
        np.testing.assert_allclose(predicted_state[-demo.model.nu:], state[-demo.model.nu:]+result["action"])
        # Check the motion in real MuJoCo. Servo inertia means J*u is a
        # target displacement, not the displacement achieved in one tick.
        demo.execute_online_action(result["action"], ik_target,
            anchors, np.zeros_like(anchors), False)
        executed_tips, _ = demo.online_tip_kinematics()
        # Check movement toward the target; the servo's finite inertia and
        # contact response time need not achieve a fixed fraction in 25 ms.
        assert np.all(np.linalg.norm(executed_tips-(tips-[0., 0., .01]), axis=1) < .0099)
        commanded = demo.online_servo_target.copy()
        np.testing.assert_allclose(commanded, state[-demo.model.nu:]+result["action"])
        demo.execute_online_action(np.zeros(demo.model.nu), ik_target, anchors, np.zeros_like(anchors), False)
        np.testing.assert_array_equal(demo.online_servo_target, commanded)
        # A solver can return success with an unhelpful stationary step.
        # The online controller must still evaluate the pad correction by
        # its complete MPC objective, respecting the near-contact bounds.
        demo.mpc.planner_solver_ = "acados"
        monkeypatch.setattr(demo.mpc, "_plan_once_acados", lambda *a, **kw: dict(
            action=np.zeros(demo.model.nu), u_traj=np.zeros((args.mpc_horizon, demo.model.nu)),
            solve_status="ACADOS_SUCCESS", solver_backend="acados", sol_guess=None))
        near_phi = np.ones_like(phi); near_phi[:4] = .005
        rescued = demo.mpc.plan_contacts(state, near_phi, np.zeros_like(jac), tips=tips, tip_jacobian=j, anchors=anchors, goal=goal,
            hand_posture=ik_target, loaded=False)
        assert rescued["selected_control"] == "contact_alignment_by_mpc_cost"
        assert rescued["action"][2] < 0
        assert np.all(np.abs(rescued["action"]) <= demo.mpc_param.contact_u_limit+1e-12)
        # At contact, even an IK target pulling in the opposite direction
        # cannot change either the rescue seed or the selected MPC command.
        contact_phi = near_phi.copy(); contact_phi[:4] = 0.
        conflicting_ik = ik_target.copy(); conflicting_ik[2] += .1
        common = dict(tips=tips, tip_jacobian=j, anchors=anchors, goal=goal, loaded=False)
        contact = demo.mpc.plan_contacts(state, contact_phi, np.zeros_like(jac),
            hand_posture=ik_target, **common)
        changed_ik = demo.mpc.plan_contacts(state, contact_phi, np.zeros_like(jac),
            hand_posture=conflicting_ik, **common)
        np.testing.assert_allclose(contact["action"], changed_ik["action"], atol=1e-12)
        assert changed_ik["action"][2] < 0
        # An online RTI failure must not block while building an IPOPT NLP.
        demo.mpc.planner_solver_ = "acados"
        monkeypatch.setattr(demo.mpc, "_plan_once_acados", lambda *a, **kw: None)
        def forbidden_ipopt(*a, **kw):
            pytest.fail("online RTI failure entered blocking IPOPT fallback")
        monkeypatch.setattr(demo.mpc, "_plan_once_ipopt", forbidden_ipopt)
        failed = demo.mpc.plan_contacts(state, phi, jac, tips=tips, tip_jacobian=j, anchors=anchors, goal=goal,
            hand_posture=ik_target, loaded=False)
        assert failed["solver_backend"] == "acados_failed"
        assert failed["fallback_control"] == "contact_alignment_correction"
        assert np.isfinite(failed["action"]).all()
        assert np.all(np.abs(failed["action"]) <= demo.mpc_param.mpc_u_ub_)
        # Failure telemetry describes the fallback actually executed.
        expected = demo.mpc.model.prediction_fn(state, failed["action"], phi, jac,
            demo.mpc.model.runtime_parameters(demo.mpc_param.model_params))
        np.testing.assert_allclose(failed["rollout_q"][0], np.asarray(expected).reshape(-1))
        drifted = state.copy(); drifted[:2] += [.02, -.03]
        demo.mpc.plan_contacts(drifted, phi, jac, tips=tips, tip_jacobian=j, anchors=anchors, goal=goal, hand_posture=ik_target,
            loaded=False, approach_origin=state[:3])
        # Unilateral contact must not expand the horizontal corridor every
        # cycle and thereby erase the accumulated displacement penalty.
        np.testing.assert_allclose(demo.mpc._horizontal_bounds,
            [state[:2]-demo.mpc_param.horizontal_tolerance,
             state[:2]+demo.mpc_param.horizontal_tolerance])
        # Recovery must translate AND rotate all pad seeds with the object,
        # while leaving the measured scene untouched.
        source_pos, source_quat, source_rot = demo.get_object_pose()
        moved_pos = source_pos+np.array([.05, -.02, 0.])
        moved_rot = demo_module.Rotation.from_euler("z", .35).as_matrix()@source_rot
        measured_before = demo.data.qpos.copy()
        palm_before, rot_before, tips_before = demo.forward_kinematics(demo.default_hand_q)
        transported = demo.transport_online_posture(demo.default_hand_q, source_pos, source_rot, moved_pos, moved_rot)
        palm_after, rot_after, tips_after = demo.forward_kinematics(transported)
        np.testing.assert_allclose(palm_after, moved_pos+moved_rot@source_rot.T@(palm_before-source_pos), atol=1e-10)
        np.testing.assert_allclose(rot_after, moved_rot@source_rot.T@rot_before, atol=1e-10)
        np.testing.assert_allclose(tips_after, moved_pos+(tips_before-source_pos)@source_rot@moved_rot.T, atol=1e-10)
        np.testing.assert_array_equal(demo.data.qpos, measured_before)
        # With the recovered corridor and a changed contact target, the real
        # MPC cost must select a command toward the displaced object.
        displaced = state.copy(); displaced[0] += .05
        chase_ik = ik_target.copy(); chase_ik[0] += .05
        chase = demo.mpc.plan_contacts(displaced, np.ones_like(phi), np.zeros_like(jac),
            tips=tips, tip_jacobian=j, anchors=anchors, goal=goal,
            hand_posture=chase_ik, loaded=False, approach_origin=displaced[:3])
        assert chase["action"][0] > 0
        projected_tips = tips+(j@chase["action"]).reshape(n, 3)
        targets = displaced[:3]+anchors@rot.T
        assert np.linalg.norm(projected_tips-targets) < np.linalg.norm(tips-targets)
    finally:
        demo.close()


def test_noslip_refinement_preserves_startup_pose_and_is_restored(demo_module, tmp_path):
    poses = []
    for iterations in (0, 3, 10):
        args = demo_module.build_argparser().parse_args([
            "--headless", "--no-video", "--screenshot-interval", "0", "--mpc-solver", "ipopt",
            "--mpc-horizon", "2", "--scene-output", str(tmp_path/f"noslip_{iterations}.xml"),
            "--contact-noslip-iterations", str(iterations),
            "--max-point-combination-eval", "16", "--lambda-lm-iters", "5"])
        demo = demo_module.AllegroTabletopTrigraspDemo(args)
        try:
            poses.append(np.r_[demo.initial_object_pos, demo.initial_object_quat])
            assert demo.model.opt.noslip_iterations == iterations
        finally:
            demo.close()
    np.testing.assert_allclose(poses[0], poses[1], atol=1e-12, rtol=0.)
    np.testing.assert_allclose(poses[0], poses[2], atol=1e-12, rtol=0.)


def test_single_ik_returns_the_feasible_iterate_even_with_a_higher_score(demo_module, monkeypatch):
    from types import SimpleNamespace
    demo = object.__new__(demo_module.AllegroTabletopTrigraspDemo)
    demo.args = demo_module.build_argparser().parse_args(['--ik-max-iters', '2'])
    demo.args.online_ik_tip_tolerance = .004
    demo.model = SimpleNamespace(nu=22, nv=22)
    demo.hand_dof_adr = np.arange(22)
    demo.hand_joint_limited = np.zeros(22, bool)
    demo.hand_joint_ranges = np.tile([-10., 10.], (22, 1))
    demo.hand_collision_geom_ids = set()
    demo.palm_body_id, demo.tip_site_ids = 0, [0]
    site = SimpleNamespace(xpos=np.zeros(3))
    demo.solve_data = SimpleNamespace(contact=[], site=lambda _: site)
    calls = 0
    def kinematics(q):
        nonlocal calls
        calls += 1
        site.xpos = np.array([.003 if calls == 4 else .005, 0., 0.])
        rotation = demo_module.Rotation.from_euler('z', .2 if calls == 4 else 0.).as_matrix()
        return np.zeros(3), rotation, site.xpos[None].copy()
    demo.forward_kinematics = kinematics
    monkeypatch.setattr(demo_module.mujoco, 'mj_jacBody', lambda *a: None)
    def site_jac(model, data, jp, jr, site_id):
        jp[:, :3] = np.eye(3)
    monkeypatch.setattr(demo_module.mujoco, 'mj_jacSite', site_jac)
    result = demo.solve_single_ik_candidate(np.zeros(22), np.zeros(3), np.eye(3), np.zeros((1, 3)))
    # The accepted 3 mm solution has a worse palm score than the preceding
    # 5 mm solution. Returning the old minimum caused false startup failures.
    assert result[0] > .005*1.35
    assert result[2] == pytest.approx(.003)
    assert result[3] == pytest.approx(.003)
    assert result[-1] == 0.


def test_ik_and_mpc_account_for_blocked_neighbouring_fingers(demo_module, tmp_path, monkeypatch):
    args = demo_module.build_argparser().parse_args([
        "--headless", "--no-video", "--screenshot-interval", "0", "--mpc-solver", "ipopt",
        "--mpc-horizon", "2", "--scene-output", str(tmp_path/"self_contact.xml")])
    demo = demo_module.AllegroTabletopTrigraspDemo(args)
    try:
        # Regression pose: the old IK placed all pads correctly while the
        # index/middle phalanges overlapped and generated >100 N in MuJoCo.
        blocked = np.array([.6083, .039, .4954, 3.1413, .0002, -2.6765,
            -.2399, .677, .5889, .486, .3224, .3175, 1.0963, 1.0328,
            .3753, 1.0975, .7004, .5227, .7754, .5281, .4961, .4635])
        palm, rotation, tips = demo.forward_kinematics(blocked)
        original_depth = max(-float(c.dist) for c in demo.solve_data.contact
            if int(c.geom1) in demo.hand_collision_geom_ids and int(c.geom2) in demo.hand_collision_geom_ids)
        assert original_depth > .003
        demo.set_hand_qpos(blocked)
        phi, jac = demo.detect_online_contacts()
        # A self-contact constraint acts only on hand DOFs, and increasing
        # separation does not apply a fictitious wrench to the object.
        robot_only = [i for i in range(4, demo.mpc_param.max_ncon_)
            if not np.any(jac[4*i:4*i+4, :6]) and np.any(jac[4*i:4*i+4, 6:])]
        assert robot_only
        assert any(phi[4*i] < -.003 for i in robot_only)
        result = demo.solve_single_ik_candidate(blocked, palm, rotation, tips)
        demo.forward_kinematics(result[1])
        measured_depth = max([0.] + [-float(c.dist) for c in demo.solve_data.contact
            if int(c.geom1) in demo.hand_collision_geom_ids and int(c.geom2) in demo.hand_collision_geom_ids])
        assert result[-1] == pytest.approx(measured_depth)
        # The exact requested tips need not admit collision-free IK. Even
        # perfect tip residuals must not make a blocked candidate executable.
        monkeypatch.setattr(demo, "solve_single_ik_candidate", lambda **kw:
            (0., blocked.copy(), 0., 0., 0., 0., tips.copy(), original_depth))
        invalid = demo.solve_grasp_ik(tips)
        assert not invalid.success
        assert invalid.max_self_penetration == pytest.approx(original_depth)
    finally:
        demo.close()


def test_pregrasp_keeps_every_pad_clear_of_the_object(demo_module, tmp_path, monkeypatch):
    from types import SimpleNamespace
    args = demo_module.build_argparser().parse_args([
        "--headless", "--no-video", "--screenshot-interval", "0", "--mpc-solver", "ipopt",
        "--mpc-horizon", "2", "--scene-output", str(tmp_path/"pregrasp.xml")])
    demo = demo_module.AllegroTabletopTrigraspDemo(args)
    try:
        grasp = np.array([.6049664, -.0070583, .4998021, 3.141299, -.0000361, -1.9780314,
            -.2491271, .2887215, .9614909, .8887658, .0273367, .7956865, .4708246, .3376268,
            .3395088, .4342971, 1.3352366, 1.1599994, .7454797, .5242496, .7758554, .6328333])
        palm, rotation, _ = demo.forward_kinematics(grasp)
        points, inward = [], []
        for name in demo.active_finger_names:
            geom = demo_module.mujoco.mj_name2id(demo.model, demo_module.mujoco.mjtObj.mjOBJ_GEOM,
                demo_module.TIP_COLLISION_GEOM_NAMES[name])
            contacts = [c for c in demo.solve_data.contact
                if {int(c.geom1), int(c.geom2)} == {geom, demo.object_geom_id}]
            assert contacts, name
            contact = min(contacts, key=lambda c:c.dist)
            normal = np.asarray(contact.frame[:3]).copy()
            if int(contact.geom1) == demo.object_geom_id:
                normal *= -1
            points.append((np.asarray(contact.pos)-demo.initial_object_pos)@demo.initial_object_rot)
            inward.append(normal@demo.initial_object_rot)
        ik = SimpleNamespace(qpos=grasp, target_palm_pos_world=palm, target_palm_rot_world=rotation)
        opened, clearance = demo.solve_online_pregrasp(ik, np.array(points), np.array(inward),
            demo.initial_object_pos, demo.initial_object_rot)
        assert clearance >= .003
        assert not np.allclose(opened, grasp)
        assert all(float(c.dist) >= -1e-4 for c in demo.solve_data.contact
            if int(c.geom1) in demo.hand_collision_geom_ids and int(c.geom2) in demo.hand_collision_geom_ids)
        monkeypatch.setattr(demo, "solve_single_ik_candidate", lambda *a, **kw:
            (0., grasp.copy(), 0., 0., 0., 0., np.zeros((4, 3)), 0.))
        with pytest.raises(RuntimeError, match="minimum pad clearance"):
            demo.solve_online_pregrasp(ik, np.array(points), np.array(inward),
                demo.initial_object_pos, demo.initial_object_rot)
    finally:
        demo.close()


@pytest.mark.parametrize("n", [3, 4])
def test_allegro_contact_cost_frees_ik_joints_and_preserves_object_pose(demo_module, n):
    param = object.__new__(demo_module.AllegroOnlineMPCParams)
    param.num_contacts, param.n_cmd_, param.n_qpos_ = n, 22, 57
    param.posture_weights = np.r_[np.ones(3), np.full(3, .5), np.full(16, 5.)]
    param.tip_weight = 5000.
    param.action_weights = np.full(22, .03)
    param.object_weight, param.quat_weight = 120., 500.
    param.lateral_weight, param.horizontal_recovery_weight = 300., 2e6
    param.terminal_weight = 2.
    cost, terminal = param.init_cost_fns()
    pose = np.array([0., 0., .3, 1., 0., 0., 0.])
    ik = np.linspace(-.1, .1, 22)
    state = np.r_[pose, ik, np.zeros(28)]
    anchors = np.c_[np.linspace(-.03, .03, n), np.zeros(n), np.zeros(n)]
    tips = pose[:3] + anchors
    j = np.zeros((3*n, 22)); j[:, :3] = np.tile(np.eye(3), (n, 1))
    context = np.r_[pose, ik, 0., [-.005, -.005], [.005, .005], ik,
        tips.reshape(-1), j.reshape(-1, order="F"), anchors.reshape(-1), 0.]
    action = np.zeros(22)
    assert not param.cost_uses_force_reference
    assert float(cost(state, action, context)) == pytest.approx(0., abs=1e-12)
    # Changing the IK solution or moving a joint in the tip Jacobian's
    # nullspace cannot penalize an established contact.
    changed_ik = context.copy(); changed_ik[7:29] += .5
    assert float(cost(state, action, changed_ik)) == pytest.approx(0., abs=1e-12)
    nullspace = state.copy(); nullspace[13:] += .1
    assert float(cost(nullspace, action, context)) == pytest.approx(0., abs=1e-12)
    moved = state.copy(); moved[7] += .01
    assert float(cost(moved, action, context)) == pytest.approx(n*5000*.01**2)
    assert float(terminal(moved, context)) == pytest.approx(2*n*5000*.01**2)
    # Weak attraction is explicitly enabled only away from contact;
    # neither loaded grasps nor terminal cost track hand q.
    approach = changed_ik.copy(); approach[-1] = 1.
    assert float(cost(state, action, approach)) > 0
    assert float(terminal(state, approach)) == pytest.approx(0., abs=1e-12)
    approach[29] = 1.
    assert float(cost(state, action, approach)) == pytest.approx(0., abs=1e-12)
    # Hold tip alignment fixed while probing object tracking/corridor costs.
    for distance in (.004, .009):
        displaced = state.copy(); displaced[0] = displaced[7] = distance
        expected = 300*distance**2 + 2e6*max(distance-.005, 0)**2
        # hand_reference[0] is nonzero; move it by the same displacement.
        displaced[7] = state[7]+distance
        assert float(cost(displaced, action, context)) == pytest.approx(expected)
    raised = state.copy(); raised[2] += .02; raised[9] += .02
    assert float(cost(raised, action, context)) == pytest.approx(120*.02**2)
    rotated = state.copy()
    rotated[3:7] = demo_module.Rotation.from_euler("z", .2).as_quat()[[3, 0, 1, 2]]
    assert float(cost(rotated, action, context)) > 0
    opposite = rotated.copy(); opposite[3:7] *= -1
    assert float(cost(opposite, action, context)) == pytest.approx(float(cost(rotated, action, context)))


def test_live_force_solve_uses_measured_contacts_without_moving_planned_patches(demo_module):
    from types import SimpleNamespace
    demo = object.__new__(demo_module.AllegroTabletopTrigraspDemo)
    points = np.array([[.04, 0., 0.], [-.04, .03, 0.], [-.04, -.03, 0.]])
    normals = np.array([[-1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    demo.args = SimpleNamespace(lambda_prediction_dt=.025, object_friction=.7, lambda_force_limit=3.)
    demo.active_finger_names = demo_module.FINGER_NAMES[:3]
    demo.args.contact_force_threshold = .01
    demo.get_fingertip_contact_summary = lambda: dict.fromkeys(demo.active_finger_names, 0.)
    demo.object_dof_adr = 0
    demo.data = SimpleNamespace(qvel=np.zeros(6))
    demo.online_contact_points_local = points + [0., .002, 0.]
    demo.online_contact_normals_local = normals.copy()
    demo.online_contact_normals_local[0] = [-.99, .1, 0.]
    demo.online_contact_normals_local[0] /= np.linalg.norm(demo.online_contact_normals_local[0])
    demo.optimizer = SimpleNamespace(lm=BatchedLambdaLM(num_contacts=3,
        friction_cone="coulomb", device="cpu", iterations=5, mu=.7, max_contact_force=3.))
    pose = np.array([0., 0., .3, 1., 0., 0., 0.])
    target = pose.copy(); target[2] += .001
    inertia = np.diag([.2]*3+[.004]*3)
    external = np.array([0., 0., -1.962, 0., 0., 0.])
    phi = np.repeat([-.001, .005, .05], 4)
    initial = dict(contact_indices=np.arange(3))
    result = demo.update_online_contact_forces(points, normals, initial, target, pose, inertia, external, phi)
    np.testing.assert_allclose(result["force_points_local"][:2], demo.online_contact_points_local[:2])
    np.testing.assert_array_equal(result["force_points_local"][2], points[2])
    np.testing.assert_allclose(result["force_normals_local"][0], demo.online_contact_normals_local[0])
    np.testing.assert_array_equal(result["measured_force_contacts"], [True, True, False])
    # The re-solved force vectors are in the measured contact friction cones.
    normal_load = np.einsum("ij,ij->i", result["force_vectors_local"], result["force_normals_local"])
    tangent = result["force_vectors_local"]-normal_load[:, None]*result["force_normals_local"]
    assert np.all(normal_load >= -1e-6)
    assert np.all(np.linalg.norm(tangent, axis=1) <= .7*normal_load+1e-6)
    np.testing.assert_array_equal(points, [[.04, 0., 0.], [-.04, .03, 0.], [-.04, -.03, 0.]])
    np.testing.assert_array_equal(result["contact_indices"], initial["contact_indices"])
    # Moving a near-contact point must trigger another force solve even
    # when the selected surface IDs remain frozen.
    demo.online_contact_points_local[0, 2] += .004
    updated = demo.update_online_contact_forces(points, normals, result, target, pose, inertia, external, phi)
    np.testing.assert_allclose(updated["force_points_local"][0], demo.online_contact_points_local[0])
    np.testing.assert_array_equal(updated["contact_indices"], initial["contact_indices"])
    force = updated["force_vectors_local"]
    wrench = np.r_[force.sum(0), np.cross(updated["force_points_local"], force).sum(0)]
    expected_velocity = .025*np.linalg.solve(inertia, external+wrench)
    np.testing.assert_allclose(updated["velocity_next"], expected_velocity, atol=2e-6)
    np.testing.assert_allclose(updated["contact_torque_world"], wrench[3:], atol=2e-6)


@pytest.mark.parametrize("force_weight", [0., 1.])
def test_ik_attraction_shuts_off_at_predicted_contact(demo_module, force_weight):
    from types import SimpleNamespace
    nq, n = 22, 3
    param = SimpleNamespace(n_cmd_=nq, num_contacts=n, n_qvel_=6+nq,
        n_qpos_=7+nq+6+nq+nq, max_ncon_=7, h_=.025,
        obj_inertia_=np.eye(6), obj_mass_=.2, gravity_=np.array([0., 0., -9.81, 0., 0., 0.]),
        robot_damping_=np.ones(nq), robot_stiff_=np.eye(nq),
        tip_weight=5000., posture_weights=np.ones(nq), action_weights=np.ones(nq),
        lateral_weight=300., object_weight=120., quat_weight=500.,
        horizontal_recovery_weight=2e6, terminal_weight=2.,
        force_tracking_weight=force_weight, pressure_sync_weight=0.)
    model = param.online_model = demo_module.AllegroOnlineContactModel(param)
    from planning.allegro_mpc_costs import build_allegro_contact_costs
    cost, terminal = build_allegro_contact_costs(param)
    state = np.zeros(param.n_qpos_); state[3] = 1.
    model.reference = state[:model.pose_size].copy()
    phi = np.ones(4*param.max_ncon_); phi[:4*n] = .03
    jac = np.zeros((phi.size, param.n_qvel_)); jac[:4, 6] = -1.
    cp = np.r_[state[:7], np.zeros(nq), 0., [-1., -1.], [1., 1.],
        np.zeros(nq), np.zeros(3*n), np.zeros(3*n*nq), np.zeros(3*n), 1.,
        np.zeros(3*n), phi, jac.reshape(-1, order="F"), model.runtime_parameters(12.5)]
    changed = cp.copy(); changed[7] = .5
    assert float(cost(state, np.zeros(nq), changed)) > float(cost(state, np.zeros(nq), cp))
    touching = state.copy(); touching[7] += .03
    # Measured hand was clear, but a predicted pad is now touching. There
    # must be no IK preference at this future stage or at the terminal node.
    assert float(cost(touching, np.zeros(nq), changed)) == pytest.approx(float(cost(touching, np.zeros(nq), cp)))
    assert float(terminal(touching, changed)) == pytest.approx(float(terminal(touching, cp)))
    x, u, p = param.residual_fn.sx_in()
    derivative = cs.Function("contact_objective_derivative", [x, u, p],
        [cs.jacobian(param.residual_fn(x, u, p), x)])
    assert np.isfinite(np.asarray(derivative(touching, np.zeros(nq), changed))).all()
    # The compact derivative shares its scalar graph across columns, but
    # must preserve the same Gauss-Newton matrix at and away from contact.
    from planning.mpc_explicit import _compact_residual_gauss_newton
    mx, mu, mp = cs.MX.sym('x', x.numel()), cs.MX.sym('u', u.numel()), cs.MX.sym('cp', p.numel())
    compact = cs.Function('compact_cost_hessian', [mx, mu, mp],
        [_compact_residual_gauss_newton(param.residual_fn, mx, mp, mu)])
    j = cs.jacobian(param.residual_fn(x, u, p), cs.vertcat(u, x))
    expanded = cs.Function('expanded_cost_hessian', [x, u, p], [2*j.T@j])
    for measured in (state, touching):
        np.testing.assert_allclose(compact(measured, np.zeros(nq), changed),
            expanded(measured, np.zeros(nq), changed), rtol=1e-10, atol=1e-10)


def test_physical_lift_gate_does_not_require_startup_ik_patch_alignment(demo_module):
    from types import SimpleNamespace
    demo = object.__new__(demo_module.AllegroTabletopTrigraspDemo)
    demo.args = SimpleNamespace(lift_contact_fingers=4)
    audit = dict(all_planned_contacts_reached=False,
        actual_force_closure=dict(force_closure=True, load_feasible=True))
    assert demo.online_lift_readiness(audit, 4) == (True, [])
    assert demo.online_lift_readiness(audit, 3) == (False, ['contacts'])
    audit['actual_force_closure']['load_feasible'] = False
    assert demo.online_lift_readiness(audit, 4) == (False, ['load_support'])
    audit['actual_force_closure']['force_closure'] = False
    assert demo.online_lift_readiness(audit, 4) == (False, ['force_closure', 'load_support'])


def test_online_scm_configuration_uses_shared_parameters(demo_module):
    from types import SimpleNamespace
    h, nq, n = .025, 22, 3
    base = dict(h_=h, n_cmd_=nq, n_qpos_=79, n_qvel_=28, num_contacts=n,
        max_ncon_=n, obj_inertia_=np.diag([.03]*3+[.0001]*3)/h**2,
        obj_mass_=.03, gravity_=np.array([0., 0., -9.81, 0., 0., 0.]),
        robot_stiff_=np.eye(nq)*200., robot_damping_=np.full(nq, 40.))
    models = [demo_module.AllegroOnlineContactModel(SimpleNamespace(**base, soft_scm_response_time=tau,
        soft_scm_damping_ratio=1., soft_scm_regularization=1e-7)) for tau in (.05, .01)]
    state = np.zeros(79); state[3] = 1.
    jac = np.zeros((4*n, 28)); jac[:4, :3] = [[1, 1, 0], [1, 0, 1], [1, -1, 0], [1, 0, -1]]
    jac[:4, 6:9] = -jac[:4, :3]
    gaps = np.ones(4*n); gaps[:4] = -.002
    normals = []
    for model, tau in zip(models, (.05, .01)):
        assert model.soft_scm_response_time == tau
        assert model.soft_scm_regularization == 1e-7
        model.reference = state[:model.pose_size].copy()
        force = np.asarray(model.contact_force_fn(state, np.zeros(nq), gaps, jac, model.runtime_parameters(12.5))).reshape(n,3)
        normals.append(force[0, 0])
    assert normals[1] > normals[0] > 0
    args = demo_module.build_argparser().parse_args(['--force-response-time', '.01', '--soft-scm-regularization', '1e-7'])
    assert args.soft_scm_response_time == .01
    assert args.soft_scm_regularization == 1e-7


def test_recovery_debounces_loss_retries_a_miss_and_rearms_after_grasp(demo_module):
    tracker = demo_module.OnlineGraspRecovery(np.zeros(3), np.eye(3), .02, cooldown=.1, stall_timeout=.06)
    def observe(**kw):
        return tracker.observe(np.zeros(3), np.eye(3), contacts=kw.pop('contacts', 0),
            was_loaded=kw.pop('was_loaded', False), loaded=kw.pop('loaded', False),
            closing=kw.pop('closing', False))
    # Hovering with no contacts is not a failed grasp attempt.
    for _ in range(10):
        assert observe() is None
    assert observe(contacts=4, was_loaded=True) == 'lost_grasp'
    assert tracker.recovering
    # A loss during cooldown is remembered rather than permanently missed.
    assert observe(was_loaded=True) is None
    for _ in range(5):
        reason = observe()
        if reason:
            break
    assert reason == 'lost_grasp'
    assert observe(contacts=4, loaded=True) is None
    assert not tracker.recovering
    for _ in range(6):
        reason = observe(closing=True)
        if reason:
            break
    assert reason == 'missed_contact'


def test_preload_hysteresis_keeps_a_physically_load_bearing_grasp(demo_module):
    from types import SimpleNamespace
    from planning.grasp_verification import verify_grasp
    demo = object.__new__(demo_module.AllegroTabletopTrigraspDemo)
    demo.args = SimpleNamespace(lift_contact_fingers=3, contact_force_threshold=.01,
        online_min_normal_force=.6)
    points = [[.04, 0, 0], [-.04, .03, 0], [-.04, -.03, 0]]
    normals = [[-1., 0, 0], [1., 0, 0], [1., 0, 0]]
    loads = [.6, .3, .3]
    certificate = verify_grasp(points, normals, .8, external_wrench=[0, 0, -.2943, 0, 0, 0],
        normal_force_limits=loads)
    assert certificate['force_closure'] and certificate['load_feasible']
    audit = dict(actual_force_closure=certificate,
        fingers=[dict(touching=True, normal_force=f) for f in loads])
    assert not demo.online_lift_readiness(audit, 3)[0]
    assert demo.online_lift_readiness(audit, 3, grasp_active=True)[0]
    audit['actual_force_closure'] = dict(certificate, load_feasible=False)
    assert not demo.online_lift_readiness(audit, 3, grasp_active=True)[0]


def test_certified_local_repair_precedes_global_opening_but_has_a_deadline(demo_module):
    tracker = demo_module.OnlineGraspRecovery(np.zeros(3), np.eye(3), .025)
    common = dict(contacts=3, loaded=False, closing=True, repair_available=True)
    assert tracker.observe(np.zeros(3), np.eye(3), was_loaded=True, **common) is None
    assert not tracker.opening
    for _ in range(25):
        reason = tracker.observe(np.zeros(3), np.eye(3), was_loaded=False, **common)
        if reason:
            break
    assert reason == 'lost_grasp' and tracker.opening
    # A physical recovery during the local repair window clears the loss.
    tracker = demo_module.OnlineGraspRecovery(np.zeros(3), np.eye(3), .025)
    tracker.observe(np.zeros(3), np.eye(3), was_loaded=True, **common)
    assert tracker.observe(np.zeros(3), np.eye(3), contacts=3, was_loaded=False,
        loaded=True, closing=True) is None
    assert not tracker.pending_loss and tracker.repair_time == 0.


def test_recovery_retries_stalled_partial_contacts(demo_module):
    tracker = demo_module.OnlineGraspRecovery(np.zeros(3), np.eye(3), .02, stall_timeout=.1)
    reason = None
    for _ in range(6):
        reason = tracker.observe(np.zeros(3), np.eye(3), contacts=2,
            was_loaded=False, loaded=False, closing=True)
        if reason:
            break
    assert reason == 'unconfirmed_grasp'
    assert tracker.opening
    # Opening is deliberately contact-free; that must not restart the search.
    for _ in range(100):
        assert tracker.observe(np.zeros(3), np.eye(3), contacts=0,
            was_loaded=False, loaded=False, closing=False) is None


def test_recovery_replans_when_partial_contact_blocks_opening(demo_module):
    tracker = demo_module.OnlineGraspRecovery(np.zeros(3), np.eye(3), .02,
        cooldown=.1, opening_timeout=.2)
    assert tracker.observe(np.zeros(3), np.eye(3), contacts=1,
        was_loaded=True, loaded=False, closing=True) == 'lost_grasp'
    reason = None
    for _ in range(12):
        reason = tracker.observe(np.zeros(3), np.eye(3), contacts=1,
            was_loaded=False, loaded=False, closing=False)
        if reason:
            break
    assert reason == 'blocked_pregrasp'
    assert tracker.replan_count == 2


@pytest.mark.parametrize('displacement,angle', [([.04, 0., 0.], 0.), ([0., 0., 0.], .4)])
def test_recovery_relocates_the_approach_region_only_on_confirmed_event(demo_module, displacement, angle):
    tracker = demo_module.OnlineGraspRecovery(np.zeros(3), np.eye(3), .02)
    pos = np.asarray(displacement)
    rot = demo_module.Rotation.from_euler('z', angle).as_matrix()
    assert tracker.observe(pos, rot, contacts=0, was_loaded=False, loaded=False, closing=False) == 'object_moved'
    np.testing.assert_array_equal(tracker.origin, pos)
    assert tracker.observe(pos, rot, contacts=0, was_loaded=False, loaded=False, closing=False) is None


@pytest.mark.parametrize('flags,enabled', [([], False), (['--video-output-path', '/tmp/explicit.mp4'], False),
    (['--save'], True), (['--save', '--no-video'], False)])
def test_video_requires_explicit_save(demo_module, flags, enabled):
    demo = object.__new__(demo_module.AllegroTabletopTrigraspDemo)
    demo.args = demo_module.build_argparser().parse_args(flags)
    demo.mesh_path = Path('elephant.stl')
    assert (demo._resolve_video_output_path() is not None) is enabled


def test_contact_transport_prices_airborne_goal_without_tracking_ik(demo_module):
    from types import SimpleNamespace
    n, nq = 3, 22
    param = SimpleNamespace(n_cmd_=nq, num_contacts=n, n_qpos_=79,
        tip_weight=5000., contact_transport_weight=1000., posture_weights=np.ones(nq),
        action_weights=np.ones(nq), lateral_weight=300., object_weight=120., quat_weight=500.,
        horizontal_recovery_weight=2e6, terminal_weight=4.)
    from planning.allegro_mpc_costs import build_allegro_contact_costs
    cost, terminal = build_allegro_contact_costs(param)
    state = np.zeros(79); state[3] = 1.
    goal = state[:7].copy(); goal[2] = .2
    j = np.zeros((3*n, nq)); j[:, :3] = np.tile(np.eye(3), (n, 1))
    cp = np.r_[goal, np.ones(nq), 0., [-1., -1.], [1., 1.], np.zeros(nq),
        np.zeros(3*n), j.reshape(-1, order='F'), np.zeros(3*n), 0.]
    unloaded = float(cost(state, np.zeros(nq), cp))
    cp[7+nq] = 1.
    assert float(cost(state, np.zeros(nq), cp))-unloaded == pytest.approx(n*1000*.2**2)
    changed_ik = cp.copy(); changed_ik[7:7+nq] += 2.
    assert float(cost(state, np.zeros(nq), changed_ik)) == pytest.approx(float(cost(state, np.zeros(nq), cp)))
    raised = state.copy(); raised[2] = raised[9] = .2
    assert float(terminal(raised, cp)) == pytest.approx(0., abs=1e-12)
