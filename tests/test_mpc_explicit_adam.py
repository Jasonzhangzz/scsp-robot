"""CPU tests for comfree-GS collision and Warp/torch Adam planners."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from models.dexforge_fast_step import _as_host_numpy
from models.comfree_gs_torch import (
    ComfreeGSModel,
    GaussianCloud,
    gs_sphere_features,
)
from planning.mpc_explicit_adam import (
    PLAN_ONCE_KEYS,
    LambdaContactAdamOptimizer,
    MPCExplicitEEAdam,
    evaluate_bigrasp_gs_cost,
    BigraspGSCostWeights,
    configure_adam_layout,
)


def _box_cloud(half=0.04, n=5, radius=0.006):
    axis = np.linspace(-half, half, n, dtype=np.float32)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    pts = np.stack((xx, yy, zz), axis=-1).reshape(-1, 3)
    on_skin = np.max(np.abs(pts), axis=1) > (half * 0.85)
    return GaussianCloud(pts[on_skin], radius=radius)


def test_as_host_numpy_accepts_torch_tensor():
    cpu = torch.arange(4, dtype=torch.float32)
    host = _as_host_numpy(cpu)
    assert host.dtype == np.float32
    assert np.allclose(host, np.arange(4, dtype=np.float32))
    if torch.cuda.is_available():
        host_cuda = _as_host_numpy(cpu.to("cuda"))
        assert np.allclose(host_cuda, np.arange(4, dtype=np.float32))


def test_gs_signed_distance_signs():
    spheres = torch.tensor([[0.0, 0.0, 0.0, 0.05]], dtype=torch.float32)
    on_surface, _, inward = gs_sphere_features(
        torch.tensor([[[0.05, 0.0, 0.0]]], dtype=torch.float32),
        spheres,
        query_radius=0.0,
        tau=1.0e-4,
    )
    assert abs(float(on_surface)) < 2.0e-3
    assert float(inward[..., 0]) < 0.0
    outside, _, _ = gs_sphere_features(
        torch.tensor([[[0.12, 0.0, 0.0]]], dtype=torch.float32),
        spheres,
        query_radius=0.0,
        tau=1.0e-4,
    )
    assert float(outside) > 0.05


def test_comfree_step_closes_gap_when_tip_pushes_in():
    cloud = GaussianCloud(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), radius=0.04)
    model = ComfreeGSModel(cloud, obj_mass=0.2, table_z=None, device="cpu", timestep=0.01)
    state = torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, -0.08, 0.0, 0.0],
        dtype=torch.float32,
    )
    before = model.collide(state)["phi"].clone()
    nxt, extras = model.step(state, torch.tensor([-0.03, 0.0, 0.0, 0.03, 0.0, 0.0]))
    after = extras["phi"]
    assert torch.isfinite(nxt).all()
    assert float(after[0]) <= float(before[0]) + 1.0e-5
    assert float(after[1]) <= float(before[1]) + 1.0e-5


def test_lambda_adam_returns_two_contacts():
    cloud = _box_cloud()
    opt = LambdaContactAdamOptimizer(cloud=cloud, adam_steps=8, adam_lr=0.03, device="cpu")
    points, normals, cost, score, margin = opt.choose_contact_set()
    assert points.shape == (2, 3)
    assert normals.shape == (2, 3)
    assert np.isfinite(points).all()
    assert opt.last_grasp_result["witness_contact_forces_local"].shape == (2, 3)
    assert points.shape[0] == 2
    _ = cost, score, margin


def test_mpc_adam_plan_once_contract():
    cloud = _box_cloud()
    param = SimpleNamespace(
        obj_mass_=0.2,
        gravity_=np.array([0.0, 0.0, -9.8], dtype=np.float32),
        planner_object_inertia_pos=50.0,
        planner_object_inertia_rot=0.05,
        robot_stiff_=np.eye(6, dtype=np.float32) * 8.0,
        contact_stiffness=12.5,
        h_=0.01,
        mpc_horizon_=4,
        adam_iters=5,
        adam_lr=0.03,
        adam_restarts=1,
        planner_cmd_limit=0.04,
        mppi_device_="cpu",
        planner_object_target_weight_=0.0,
        planner_force_tracking_weight_=0.0,
    )
    planner = MPCExplicitEEAdam(param, cloud=cloud)
    state = np.array(
        [0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0, 0.07, 0.0, 0.05, -0.07, 0.0, 0.05],
        dtype=np.float64,
    )
    result = planner.plan_once(
        np.array([0.0, 0.0, 0.12]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        state,
        contact_points_local=np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]]),
        normals_local=np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert set(PLAN_ONCE_KEYS) <= set(result)
    assert result["action"].shape == (6,)
    assert result["u_traj"].shape == (4, 6)
    assert result["rollout_q"].shape == (4, 13)
    assert result["solver_backend"] == "adam"
    assert np.isfinite(result["action"]).all()


def test_zero_weight_terms_do_not_change_cost():
    states = torch.zeros(3, 13)
    states[:, 3] = 1.0
    extras = {
        "phi": torch.zeros(3, 2),
        "normal": torch.zeros(3, 2, 3),
        "contact_force": torch.zeros(3, 2),
    }
    cmd = torch.zeros(3, 6)
    ctx = {
        "contact_points_local": torch.zeros(2, 3),
        "normals_local": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        "target_object_pos": torch.ones(3),
        "target_object_quat": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    }
    off = BigraspGSCostWeights(
        contact_attract=0.0,
        contact_depth=0.0,
        penetration=0.0,
        object_position=0.0,
        object_lateral=0.0,
        object_orientation=0.0,
        action=0.0,
        smooth=0.0,
        sync=0.0,
        force=0.0,
        swap=0.0,
        inter_arm=0.0,
        tip_sep=0.0,
    )
    cost = evaluate_bigrasp_gs_cost(states, extras, cmd, ctx, off)
    assert float(cost) == 0.0


def test_assigned_contacts_beat_swapped_assignment():
    contact = torch.tensor([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
    normals = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    extras = {"phi": torch.zeros(1, 2), "normal": torch.zeros(1, 2, 3), "contact_force": torch.zeros(1, 2)}
    cmd = torch.zeros(1, 6)
    ctx = {"contact_points_local": contact, "normals_local": normals}
    weights = BigraspGSCostWeights(
        contact_attract=10.0,
        contact_depth=0.0,
        penetration=0.0,
        object_position=0.0,
        object_lateral=0.0,
        object_orientation=0.0,
        action=0.0,
        smooth=0.0,
        sync=0.0,
        force=0.0,
        swap=20.0,
        inter_arm=0.0,
        tip_sep=0.0,
    )
    assigned = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, -0.05, 0.0, 0.0]])
    swapped = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 0.05, 0.0, 0.0]])
    cost_ok = float(evaluate_bigrasp_gs_cost(assigned, extras, cmd, ctx, weights))
    cost_swap = float(evaluate_bigrasp_gs_cost(swapped, extras, cmd, ctx, weights))
    assert cost_ok < cost_swap


def test_approach_offset_targets_the_sphere_center():
    contact = torch.zeros(2, 3)
    normals = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    ctx = {"contact_points_local": contact, "normals_local": normals}
    weights = BigraspGSCostWeights(
        contact_attract=1.0,
        contact_depth=0.0,
        penetration=0.0,
        action=0.0,
        smooth=0.0,
        sync=0.0,
        swap=0.0,
        inter_arm=0.0,
        tip_sep=0.0,
        force=0.0,
        query_mm=1.0,
        terminal=0.0,
        approach_offset=0.01,
    )
    states = torch.zeros(1, 13)
    states[0, 3] = 1.0
    states[0, 9] = 0.01
    states[0, 12] = 0.01
    extras = {"phi": torch.zeros(1, 2), "normal": torch.zeros(1, 2, 3), "contact_force": torch.zeros(1, 2)}
    cmd = torch.zeros(1, 6)
    assert float(evaluate_bigrasp_gs_cost(states, extras, cmd, ctx, weights)) == 0.0


def test_zero_predicted_force_does_not_penalize_approach():
    extras = {"phi": torch.zeros(1, 2), "normal": torch.zeros(1, 2, 3), "contact_force": torch.zeros(1, 2)}
    cmd = torch.zeros(1, 6)
    ctx = {
        "contact_points_local": torch.tensor([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]]),
        "normals_local": torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "desired_force_local": torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    }
    weights = BigraspGSCostWeights(
        contact_attract=0.0,
        contact_depth=0.0,
        penetration=0.0,
        action=0.0,
        smooth=0.0,
        sync=0.0,
        swap=0.0,
        inter_arm=0.0,
        tip_sep=0.0,
        force=4.0,
        query_mm=1.0,
        terminal=0.0,
        gate_scale=0.02,
    )
    near = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, -0.05, 0.0, 0.0]])
    far = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, -0.25, 0.0, 0.0]])
    near_cost = float(evaluate_bigrasp_gs_cost(near, extras, cmd, ctx, weights))
    far_cost = float(evaluate_bigrasp_gs_cost(far, extras, cmd, ctx, weights))
    assert abs(near_cost - far_cost) < 1.0e-4


def test_object_lift_is_gated_before_assigned_contact():
    extras = {"phi": torch.ones(1, 2), "normal": torch.zeros(1, 2, 3), "contact_force": torch.zeros(1, 2)}
    cmd = torch.zeros(1, 6)
    ctx = {
        "contact_points_local": torch.tensor([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]]),
        "normals_local": torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "target_object_pos": torch.tensor([0.0, 0.0, 0.1]),
    }
    weights = BigraspGSCostWeights(
        contact_attract=0.0,
        contact_depth=0.0,
        penetration=0.0,
        object_position=300.0,
        object_lateral=0.0,
        object_orientation=0.0,
        action=0.0,
        smooth=0.0,
        sync=0.0,
        force=0.0,
        swap=0.0,
        inter_arm=0.0,
        tip_sep=0.0,
        gate_scale=0.02,
    )
    far = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.2, -0.2, 0.0, 0.2]])
    near = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, -0.05, 0.0, 0.0]])
    assert float(evaluate_bigrasp_gs_cost(far, extras, cmd, ctx, weights)) < 1.0
    assert float(evaluate_bigrasp_gs_cost(near, extras, cmd, ctx, weights)) > 2.0


def test_tip_separation_and_capsule_penalty():
    extras = {
        "phi": torch.zeros(1, 2),
        "normal": torch.zeros(1, 2, 3),
        "contact_force": torch.zeros(1, 2),
        "capsules": torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.01], [0.0, 0.0, 0.01], [0.0, 0.0, 0.01]]]
        ),
    }
    states = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01]])
    cmd = torch.zeros(1, 6)
    ctx = {
        "contact_points_local": torch.zeros(2, 3),
        "normals_local": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
    }
    weights = BigraspGSCostWeights(
        contact_attract=0.0,
        contact_depth=0.0,
        penetration=0.0,
        action=0.0,
        smooth=0.0,
        sync=0.0,
        swap=0.0,
        inter_arm=10.0,
        tip_sep=10.0,
        query_mm=1.0,
        tip_sep_min=0.05,
        capsule_radius=0.04,
        capsule_margin=0.02,
    )
    cost = float(evaluate_bigrasp_gs_cost(states, extras, cmd, ctx, weights))
    assert cost > 0.0


def test_inter_arm_counts_left_right_query_spheres():
    extras = {
        "phi": torch.zeros(1, 2),
        "normal": torch.zeros(1, 2, 3),
        "contact_force": torch.zeros(1, 2),
        "capsules": torch.tensor(
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]]
        ),
    }
    far = extras["capsules"].clone()
    far[..., 3:, 0] = 0.5
    states = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]])
    cmd = torch.zeros(1, 6)
    ctx = {
        "contact_points_local": torch.zeros(2, 3),
        "normals_local": torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
    }
    weights = BigraspGSCostWeights(
        contact_attract=0.0,
        contact_depth=0.0,
        penetration=0.0,
        action=0.0,
        smooth=0.0,
        sync=0.0,
        swap=0.0,
        inter_arm=10.0,
        tip_sep=0.0,
        query_mm=1.0,
        capsule_radius=0.045,
        wrist_radius=0.035,
        tip_sphere_radius=0.01,
        capsule_margin=0.02,
    )
    near = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0]])
    near_cost = float(evaluate_bigrasp_gs_cost(near, extras, cmd, ctx, weights))
    far_cost = float(evaluate_bigrasp_gs_cost(states, {**extras, "capsules": far}, cmd, ctx, weights))
    assert near_cost > 0.0
    assert near_cost > far_cost


def test_configure_adam_layout_is_joint_14d():
    param = SimpleNamespace()
    configure_adam_layout(param)
    assert param.n_qpos_ == 21
    assert param.n_cmd_ == 14
    assert param.n_action_ == 14
    assert param.mpc_model == "dexforge_step"


def test_planner_scene_writes_gs_and_mutual_groups(tmp_path):
    import xml.etree.ElementTree as ET

    from models.dexforge_planner_scene import write_dexforge_planner_scene

    src = tmp_path / "viewer.xml"
    src.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="obj"><geom name="obj" type="sphere" size="0.05"/></body>
            <body name="left_link4"><geom name="left_col" type="capsule" size="0.03 0.05"/></body>
            <body name="left_link6"><geom name="left_w" type="sphere" size="0.02"/></body>
            <body name="left_attachment"><geom name="left_tip" type="sphere" size="0.01"/></body>
            <body name="right_link4"><geom name="right_col" type="capsule" size="0.03 0.05"/></body>
            <body name="right_link6"><geom name="right_w" type="sphere" size="0.02"/></body>
            <body name="right_attachment"><geom name="right_tip" type="sphere" size="0.01"/></body>
            <body name="left_ghost_tip">
              <geom name="left_ghost_tip_sphere" type="sphere" size="0.01" contype="0" conaffinity="0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    cloud = GaussianCloud(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), radius=0.02)
    out = write_dexforge_planner_scene(src, cloud, output_xml=tmp_path / "planner.xml")
    root = ET.parse(out).getroot()
    option = root.find("option")
    assert option is not None
    assert option.get("integrator") == "Euler"
    assert option.get("gravity") == "0 0 0"
    geoms = {geom.get("name"): geom for geom in root.iter("geom")}
    assert geoms["obj_gs"].get("type") == "gs"
    assert geoms["obj"].get("contype") == "1"
    assert geoms["obj"].get("conaffinity") == "6"
    assert geoms["left_col"].get("contype") == "2"
    assert geoms["left_col"].get("conaffinity") == "5"
    assert geoms["right_col"].get("contype") == "4"
    assert geoms["right_col"].get("conaffinity") == "3"
    assert geoms["left_tip_gs"].get("type") == "gs"
    assert geoms["right_tip_gs"].get("type") == "gs"
    assert geoms["left_ghost_tip_sphere"].get("contype") == "0"
    assert (tmp_path / "_generated_object_gs.npz").is_file()


class _FakeDexForgeStep:
    """Kinematic stand-in: first 3 / next 3 joint cmds move the two tips."""

    def set_cloud(self, cloud):
        self.cloud = cloud

    def rollout(self, planner_x, cmd_traj, planner_xd=None, qpos0=None, qvel0=None):
        del planner_xd, qpos0, qvel0
        obj = planner_x[..., :7]
        left = planner_x.new_tensor([0.12, 0.0, 0.0])
        right = planner_x.new_tensor([-0.12, 0.0, 0.0])
        states = []
        extras = []
        for step in range(int(cmd_traj.shape[0])):
            left = left + cmd_traj[step, :3]
            right = right + cmd_traj[step, 7:10]
            planner = torch.cat((obj, left, right), dim=-1)
            states.append(planner)
            extras.append(
                {
                    "phi": planner.new_zeros(2),
                    "normal": planner.new_zeros(2, 3),
                    "contact_force": planner.new_zeros(2),
                    "capsules": torch.stack((left, left, right, right), dim=0),
                }
            )
        return torch.stack(states, dim=0), extras


def test_warp_layout_plan_once_reduces_assigned_distance():
    torch.manual_seed(0)
    param = SimpleNamespace(
        mpc_horizon_=4,
        adam_iters=8,
        adam_lr=0.25,
        adam_restarts=1,
        planner_joint_delta_limit=0.06,
        mppi_device_="cpu",
        planner_ee_position_weight_=80.0,
        planner_object_target_weight_=0.0,
        planner_object_lateral_weight_=0.0,
        planner_object_orientation_weight_=0.0,
        planner_action_weight_=0.05,
        planner_smooth_action_weight_=0.0,
        planner_synchronization_weight_=0.0,
        planner_force_tracking_weight_=0.0,
    )
    planner = MPCExplicitEEAdam(
        param,
        warp_model=_FakeDexForgeStep(),
        cost_weights=BigraspGSCostWeights(
            contact_attract=80.0,
            contact_depth=0.0,
            penetration=0.0,
            object_position=0.0,
            object_lateral=0.0,
            object_orientation=0.0,
            action=0.05,
            smooth=0.0,
            sync=0.0,
            force=0.0,
            swap=0.0,
            inter_arm=0.0,
            tip_sep=0.0,
        ),
    )
    state = np.zeros(21, dtype=np.float64)
    state[3] = 1.0
    contacts = np.array([[0.04, 0.0, 0.0], [-0.04, 0.0, 0.0]], dtype=np.float64)
    normals = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    result = planner.plan_once(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        state,
        contact_points_local=contacts,
        normals_local=normals,
    )
    assert result["action"].shape == (14,)
    assert result["u_traj"].shape == (4, 14)
    assert result["rollout_q"].shape == (4, 13)
    last = result["rollout_q"][-1]
    assert float(np.linalg.norm(last[7:10] - contacts[0])) < 0.08
    assert float(np.linalg.norm(last[10:13] - contacts[1])) < 0.08


def test_warp_plan_once_contract_optional():
    import os

    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for DexForge Warp plan_once")
    if not os.environ.get("SCSP_TEST_DEXFORGE"):
        pytest.skip("Set SCSP_TEST_DEXFORGE=1 to compile DexForge Warp in tests")
    try:
        from models.dexforge_fast_step import ensure_comfree_warp

        ensure_comfree_warp()
        import comfree_warp  # noqa: F401
        import warp  # noqa: F401
    except Exception:
        pytest.skip("DexForge comfree_warp is not importable")
    pytest.skip("Full dual-Panda compile_fast_step is exercised by examples/mpc/.../bigrasp.py --mpc")
