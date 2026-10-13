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
        assert args.lambda_prediction_dt == demo.command_dt
        assert args.lift_contact_fingers == n
        assert demo.optimizer.require_force_closure
        assert demo.optimizer.lm.friction_cone == "coulomb"
        # An ungrasped object's drift must not keep moving the IK target.
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
        # Move the IK palm target down 1 cm; the MPC cost does not need EE
        # targets, Jacobians or sampled contact-point coordinates.
        anchors = (tips - pos - [0., 0., .01]) @ rot
        goal = np.r_[pos, quat]
        ik_target = state[7:7+demo.model.nu].copy()
        ik_target[2] -= .01
        result = demo.mpc.plan_ik_pose(state, phi, jac, goal=goal,
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
        assert np.all(np.linalg.norm(executed_tips-(tips-[0., 0., .01]), axis=1) < .0095)
        commanded = demo.online_servo_target.copy()
        np.testing.assert_allclose(commanded, state[-demo.model.nu:]+result["action"])
        demo.execute_online_action(np.zeros(demo.model.nu), ik_target, anchors, np.zeros_like(anchors), False)
        np.testing.assert_array_equal(demo.online_servo_target, commanded)
        # A solver can return success with an unhelpful stationary step.
        # The online controller must still evaluate the IK correction by
        # its complete MPC objective, respecting the near-contact bounds.
        demo.mpc.planner_solver_ = "acados"
        monkeypatch.setattr(demo.mpc, "_plan_once_acados", lambda *a, **kw: dict(
            action=np.zeros(demo.model.nu), u_traj=np.zeros((args.mpc_horizon, demo.model.nu)),
            solve_status="ACADOS_SUCCESS", solver_backend="acados", sol_guess=None))
        near_phi = np.ones_like(phi); near_phi[:4] = .005
        rescued = demo.mpc.plan_ik_pose(state, near_phi, np.zeros_like(jac), goal=goal,
            hand_posture=ik_target, loaded=False)
        assert rescued["selected_control"] == "ik_correction_by_mpc_cost"
        assert rescued["action"][2] < 0
        assert np.all(np.abs(rescued["action"]) <= demo.mpc_param.contact_u_limit+1e-12)
        # An online RTI failure must not block while building an IPOPT NLP.
        demo.mpc.planner_solver_ = "acados"
        monkeypatch.setattr(demo.mpc, "_plan_once_acados", lambda *a, **kw: None)
        def forbidden_ipopt(*a, **kw):
            pytest.fail("online RTI failure entered blocking IPOPT fallback")
        monkeypatch.setattr(demo.mpc, "_plan_once_ipopt", forbidden_ipopt)
        failed = demo.mpc.plan_ik_pose(state, phi, jac, goal=goal,
            hand_posture=ik_target, loaded=False)
        assert failed["solver_backend"] == "acados_failed"
        assert failed["fallback_control"] == "ik_posture_correction"
        assert np.isfinite(failed["action"]).all()
        assert np.all(np.abs(failed["action"]) <= demo.mpc_param.mpc_u_ub_)
        drifted = state.copy(); drifted[:2] += [.02, -.03]
        demo.mpc.plan_ik_pose(drifted, phi, jac, goal=goal, hand_posture=ik_target,
            loaded=False, approach_origin=state[:3])
        # Unilateral contact must not expand the horizontal corridor every
        # cycle and thereby erase the accumulated displacement penalty.
        np.testing.assert_allclose(demo.mpc._horizontal_bounds,
            [state[:2]-demo.mpc_param.horizontal_tolerance,
             state[:2]+demo.mpc_param.horizontal_tolerance])
    finally:
        demo.close()


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
def test_allegro_ik_cost_tracks_all_joints_and_preserves_object_pose_terms(demo_module, n):
    param = object.__new__(demo_module.AllegroOnlineMPCParams)
    param.num_contacts, param.n_cmd_, param.n_qpos_ = n, 22, 57
    param.posture_weights = np.r_[np.full(3, 1000.), np.full(3, 500.), np.full(16, 5000.)]
    param.action_weights = np.full(22, .03)
    param.object_weight, param.quat_weight = 120., 500.
    param.lateral_weight, param.horizontal_recovery_weight = 300., 2e6
    param.terminal_weight = 2.
    cost, terminal = param.init_cost_fns()
    pose = np.array([0., 0., .3, 1., 0., 0., 0.])
    ik = np.linspace(-.1, .1, 22)
    state = np.r_[pose, ik, np.zeros(28)]
    context = np.r_[pose, ik, 0., [-.005, -.005], [.005, .005]]
    action = np.zeros(22)
    # The 34 parameters contain no Cartesian tip/anchor/Jacobian targets.
    assert cost.size1_in(2) == 34
    assert not param.cost_uses_force_reference
    assert float(cost(state, action, context)) == pytest.approx(0., abs=1e-12)
    for joint, weight in enumerate(param.posture_weights):
        error = state.copy()
        error[7+joint] += .01
        assert float(cost(error, action, context)) == pytest.approx(weight*.01**2)
        assert float(terminal(error, context)) == pytest.approx(2*weight*.01**2)
    inside = state.copy(); inside[0] = .004
    assert float(cost(inside, action, context)) == pytest.approx(300*.004**2)
    outside = state.copy(); outside[0] = .009
    assert float(cost(outside, action, context)) == pytest.approx(300*.009**2+2e6*.004**2)
    raised = state.copy(); raised[2] += .02
    assert float(cost(raised, action, context)) == pytest.approx(120*.02**2)
    rotated = state.copy()
    rotated[3:7] = demo_module.Rotation.from_euler("z", .2).as_quat()[[3, 0, 1, 2]]
    assert float(cost(rotated, action, context)) > 0
    opposite = rotated.copy(); opposite[3:7] *= -1
    assert float(cost(opposite, action, context)) == pytest.approx(float(cost(rotated, action, context)))
