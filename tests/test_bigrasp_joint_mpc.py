from types import SimpleNamespace

import mujoco
import numpy as np

from envs.panda_fkin import franka_fingertip_fk
from planning.mpc_costs import build_cost_fns, pack_cost_params
from planning.mpc_explicit import MPCExplicit


def _make_joint_param(max_ncon=1):
    return SimpleNamespace(
        n_qpos_=21,
        n_qvel_=20,
        n_cmd_=14,
        n_robot_qpos_=14,
        max_ncon_=max_ncon,
        left_base_pos_=np.zeros(3),
        left_base_rot_=np.eye(3),
        right_base_pos_=np.array([1.0, 0.0, 0.0]),
        right_base_rot_=np.diag([-1.0, -1.0, 1.0]),
        Q=np.eye(20),
        robot_stiff_=np.eye(14),
        obj_mass_=1.0,
        gravity_=np.array([0.0, 0.0, -9.8, 0.0, 0.0, 0.0]),
        model_params=1.0,
        contact_cost_param=0.0,
        attract_coef=0.5,
        reject_coef=0.001,
        contact_coef=0.7,
        reject_dis=0.005,
        object_position_cost_weight_=500.0,
        object_orientation_cost_weight_=5.0,
        joint_reference_cost_weight_=0.05,
        planner_force_tracking_weight_=1.0,
        planner_torque_tracking_weight_=1.0,
        mpc_cost_kind="bigrasp_joint",
    )


def _cost_kwargs(param):
    n_phi = param.max_ncon_ * 4
    return dict(
        target_p=np.array([0.0, 0.0, 0.3]),
        target_q=np.array([1.0, 0.0, 0.0, 0.0]),
        phi_vec=np.ones(n_phi),
        jac_mat=np.zeros((n_phi, param.n_qvel_)),
        verify_cost_param_1=0.0,
        verify_cost_param_2=0.0,
        virtual_point_1=np.zeros(3),
        virtual_point_2=np.ones(3),
        contact_point_1=np.zeros(3),
        contact_point_2=np.ones(3),
        joint_reference=np.zeros(param.n_robot_qpos_),
    )


def test_bigrasp_joint_cost_packs_14d_state_and_contact_map():
    param = _make_joint_param(max_ncon=2)
    path_fn, final_fn = build_cost_fns(param)
    packed = pack_cost_params("bigrasp_joint", param, path_fn, **_cost_kwargs(param))

    assert path_fn.size_in(0) == (21, 1)
    assert path_fn.size_in(1) == (14, 1)
    assert final_fn.size_in(0) == (21, 1)
    assert packed.shape == (path_fn.size_in(2)[0],)
    value = float(path_fn(np.zeros(21), np.zeros(14), packed))
    assert np.isfinite(value)


def test_joint_space_mpc_returns_14d_increment():
    param = _make_joint_param()
    param.h_ = 0.01
    param.mpc_horizon_ = 2
    param.mpc_model = "explicit"
    param.planner_solver_ = "ipopt"
    param.mpc_u_lb_ = np.full(14, -0.02)
    param.mpc_u_ub_ = np.full(14, 0.02)
    param.mpc_q_lb_ = np.r_[-1e7 * np.ones(7), -2.9 * np.ones(14)]
    param.mpc_q_ub_ = np.r_[1e7 * np.ones(7), 2.9 * np.ones(14)]
    param.ipopt_max_iter_ = 20
    param.smooth_contact_detour = False

    planner = MPCExplicit(param)
    result = planner.plan_once(
        np.array([0.0, 0.0, 0.3]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.r_[np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]), np.zeros(14)],
        np.ones(4),
        np.zeros((4, 20)),
        verify_cost_param_1=0.0,
        verify_cost_param_2=0.0,
        virtual_point_1=np.zeros(3),
        virtual_point_2=np.ones(3),
        contact_point_1=np.zeros(3),
        contact_point_2=np.ones(3),
        joint_reference=np.zeros(14),
    )

    action = np.asarray(result["action"])
    assert action.shape == (14,)
    assert np.all(action <= param.mpc_u_ub_ + 1e-8)
    assert np.all(action >= param.mpc_u_lb_ - 1e-8)


def test_panda_symbolic_fingertip_fk_matches_mujoco():
    model = mujoco.MjModel.from_xml_path("envs/xmls/panda_nohand.xml")
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    qpos_adr = [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip")

    for q in (
        np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
        np.array([0.2, -1.0, 0.4, -1.8, 0.3, 1.2, 0.5]),
    ):
        data.qpos[qpos_adr] = q
        mujoco.mj_forward(model, data)
        expected = np.asarray(data.geom_xpos[tip_id], dtype=np.float64)
        predicted = np.asarray(franka_fingertip_fk(q)).reshape(3)
        np.testing.assert_allclose(predicted, expected, atol=1e-8)
