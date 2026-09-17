"""Isaac flip fixes: OSC press cap, table plane, raised-table ranking."""

import numpy as np
from scipy.spatial.transform import Rotation

from examples.mpc.franka.ik2.contact_frames import (
    contact_aware_task_force,
    planar_table_jacobians,
    planar_table_support_local,
)
from examples.mpc.franka.ik2.params import (
    _actor_quat_from_body,
    _box_inertia_diag,
    _ground_rotation_target_q,
    _source_standing_quat_wxyz,
    _xml_obj_geom_quat_wxyz,
)
from utils import rotations


def test_contact_force_keeps_lift_and_caps_downward_press():
    n_out = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    force_contact = np.array([0.3, 0.0, -0.4], dtype=np.float32)

    lift = contact_aware_task_force(
        np.array([20.0, 0.0, 80.0], dtype=np.float32), force_contact, n_out)
    assert np.allclose(lift, [20.0, 0.0, 80.0], atol=1e-5)

    press = contact_aware_task_force(
        np.array([20.0, 0.0, -80.0], dtype=np.float32), force_contact, n_out)
    assert np.allclose(press[0], 20.0, atol=1e-5)
    assert np.allclose(press[2], -0.4, atol=1e-5)


def test_table_support_uses_world_plane_not_mesh_minus_z():
    obj_pos = np.array([0.4, 0.0, 0.375], dtype=np.float64)
    table_height = 0.35
    identity = np.eye(3)
    local_z_up = planar_table_support_local(obj_pos, identity, table_height)
    assert np.allclose(local_z_up, [0.0, 0.0, -0.025], atol=1e-6)

    # Source-STL standing: Ry(-90) maps +X to +Z, so the table is -X.
    r_x_up = Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
    local_x_up = planar_table_support_local(obj_pos, r_x_up, table_height)
    assert np.allclose(local_x_up, [-0.025, 0.0, 0.0], atol=1e-6)
    assert abs(float(local_x_up[2])) < 1e-6


def test_planar_table_jacobian_is_a_single_world_up_constraint():
    obj_pos = np.array([0.4, 0.05, 0.375], dtype=np.float32)
    rot = Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
    con_jac, con_jac_body, local = planar_table_jacobians(
        obj_pos, rot, 0.35, nv=9, mu=0.5)
    assert con_jac.shape[0] == 4
    assert con_jac_body.shape == (4, 6)
    assert np.isfinite(con_jac).all()
    assert np.isfinite(con_jac_body).all()
    assert np.allclose(local, [-0.025, 0.0, 0.0], atol=1e-5)


def test_source_standing_quat_matches_xml_geom_plus_90y():
    geom = _xml_obj_geom_quat_wxyz("envs/xmls/env_fingertips_foam_brick.xml")
    standing = _source_standing_quat_wxyz("envs/xmls/env_fingertips_foam_brick.xml")
    assert np.allclose(standing, geom, atol=1e-8)
    rot = Rotation.from_quat([standing[1], standing[2], standing[3], standing[0]]).as_matrix()
    # XML +90 Y sends authored +X to world -Z (feet down).  Ry(-90) inverts it.
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-6)
    upside_down = rotations.rpy_to_quaternion(np.array([0.0, -0.5 * np.pi, 0.0]))
    assert float(np.abs(np.dot(standing, upside_down))) < 0.1


def test_ground_rotation_target_matches_fingertips_0902():
    roll = 0.25
    body_q = rotations.rpy_to_quaternion(np.hstack([0.0, -0.5 * np.pi, np.pi * roll - 0.5 * np.pi]))
    goal_geom = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
    expected = rotations.quaternion_multiply(body_q, goal_geom)
    got = _ground_rotation_target_q(roll)
    assert np.allclose(np.abs(np.dot(got, expected)), 1.0, atol=1e-8)
    # A standing-only yaw target stays near identity after the geom offset.
    # The flip target must be about 90 deg away from that family.
    standing = rotations.quaternion_multiply(
        rotations.rpy_to_quaternion(np.array([0.0, 0.0, 0.0])), goal_geom
    )
    quat_err = 1.0 - float(np.dot(got, standing)) ** 2
    assert quat_err > 0.4


def test_extracted_mesh_keeps_mujoco_body_quat():
    body_q = _ground_rotation_target_q(0.4)
    standing = _source_standing_quat_wxyz("envs/xmls/env_fingertips_foam_brick.xml")
    assert np.allclose(_actor_quat_from_body(body_q, standing, True), body_q)
    mapped = _actor_quat_from_body(body_q, standing, False)
    expected = rotations.quaternion_multiply(body_q, standing)
    assert np.allclose(np.abs(np.dot(mapped, expected)), 1.0, atol=1e-8)


def test_mujoco_hull_extracts_foam_brick_body_frame():
    from examples.mpc.fingertips.test.params import _mujoco_collision_mesh

    source = "envs/assets/objects/foam_brick.stl"
    extracted = _mujoco_collision_mesh(source, "envs/xmls/env_fingertips_foam_brick.xml")
    assert extracted != source


def test_box_inertia_scales_with_mass_not_urdf_1e4():
    inertia = _box_inertia_diag(
        0.01,
        np.array([-0.025, -0.02, -0.02]),
        np.array([0.025, 0.02, 0.02]),
    )
    assert np.all(inertia < 5e-6)
    assert np.all(inertia > 1e-7)
    double = _box_inertia_diag(
        0.02,
        np.array([-0.025, -0.02, -0.02]),
        np.array([0.025, 0.02, 0.02]),
    )
    assert np.allclose(double, 2.0 * inertia, atol=1e-12)
