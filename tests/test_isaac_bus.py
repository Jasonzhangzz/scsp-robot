"""Latest-only Isaac bus does not import isaacgym."""

import argparse
import queue

import numpy as np

from examples.mpc.franka.ik2.contact_frames import table_jac_mat_env
from examples.mpc.franka.ik2.isaac_bus import (
    IsaacBus,
    joint_hold_target,
    pickle_args,
    viewer_draw_stride,
)


def test_bus_latest_only_cmd_and_obs():
    cmd_q = queue.Queue(maxsize=1)
    obs_q = queue.Queue(maxsize=1)
    bus = IsaacBus(cmd_q, obs_q)
    bus.publish_obs({"seq": 1})
    bus.publish_obs({"seq": 2})
    bus.publish_cmd({"kind": "joint", "action": [0.0] * 7})
    bus.publish_cmd({"kind": "via", "via": [0.4, 0.0, 0.36]})
    assert bus.take_obs()["seq"] == 2
    assert bus.take_cmd()["kind"] == "via"
    assert bus.take_obs() is None
    assert bus.take_cmd() is None


def test_pickle_args_drops_callables():
    args = argparse.Namespace(obj="elephant", hook=lambda: None)
    packed = pickle_args(args)
    assert packed["obj"] == "elephant"
    assert "hook" not in packed


def test_viewer_draw_stride_caps_at_60hz():
    assert viewer_draw_stride(0.002, 60.0) == 8
    assert viewer_draw_stride(0.016, 60.0) == 1
    assert viewer_draw_stride(0.002, 1.0) >= 1


def test_joint_hold_target_finishes_then_holds():
    q0 = [0.0, 1.0]
    dq = [0.2, -0.4]
    assert np.allclose(joint_hold_target(q0, dq, 0, 10), [0.0, 1.0])
    assert np.allclose(joint_hold_target(q0, dq, 5, 10), [0.1, 0.8])
    end = joint_hold_target(q0, dq, 10, 10)
    assert np.allclose(end, [0.2, 0.6])
    assert np.allclose(joint_hold_target(q0, dq, 20, 10), end)


def test_table_jac_mat_env_matches_planar_table():
    from examples.mpc.franka.ik2.contact_frames import planar_table_jacobians

    obj_pos = np.array([0.4, 0.05, 0.375], dtype=np.float64)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    jac = table_jac_mat_env(obj_pos, quat, 0.35, nv=9, mu=0.5, max_ncon=10)
    _, body, _ = planar_table_jacobians(obj_pos, np.eye(3), 0.35, 9, 0.5)
    assert jac.shape == (40, 9)
    assert np.allclose(jac[0:4, :6], body)
    assert np.allclose(jac[4:], 0.0)
