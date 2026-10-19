"""Finite-tool geometry and release of a suspended mug."""
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from models.compound_tool_geometry import (
    CompoundToolGeometry, CAVITY_TOOL_ROTATION, MugUnloadingGuard,
    collision_torque_projection,
    cavity_entry_rotation,
)
from planning.compound_contact_path import CompoundApproachVia


@pytest.fixture(scope='module')
def mug_tool():
    from examples.mpc.fingertips.test.params import canonical_object_meshes
    from models.object_collision_mesh import decomposed_collision_mesh
    root = Path(__file__).resolve().parents[1]
    _,visual = canonical_object_meshes(str(root/'envs/assets/objects/mug.stl'),
        str(root/'envs/xmls/env_fingertips_mug.xml'))
    _,paths = decomposed_collision_mesh(visual,object_name='mug')
    return [trimesh.load_mesh(p) for p in paths],CompoundToolGeometry(paths)


def test_signed_distance_and_outward_normal_survive_penetration(tmp_path):
    path = tmp_path/'wall.obj'
    trimesh.creation.box(extents=[.002,.06,.1]).export(path)
    query = CompoundToolGeometry([path],wrist=False)
    contacts,gaps = query.contacts(np.zeros(3),np.eye(3),[.009,0.,0.],np.eye(3))
    assert gaps['sphere'] == pytest.approx(-.002,abs=1e-7)
    contact = next(c for c in contacts if c['kind'] == 'sphere')
    np.testing.assert_allclose(contact['normal'],[1.,0.,0.],atol=1e-5)
    surface = contact['point']-contact['gap']*contact['normal']
    assert surface[0] == pytest.approx(.001,abs=1e-7)


def test_wrist_overlap_cannot_be_inferred_from_sphere_or_shaft(tmp_path):
    path = tmp_path/'wall.obj'
    trimesh.creation.box(extents=[.002,.3,.3]).export(path)
    query = CompoundToolGeometry([path])
    _,gaps = query.contacts(np.zeros(3),np.eye(3),[.03,0.,0.],np.eye(3))
    assert gaps['sphere'] > .002 and gaps['shaft'] > .002
    assert gaps['wrist'] < -.001


def test_full_tool_rejects_deep_cavity_target_and_keeps_mouth_access(mug_tool):
    parts,query = mug_tool
    via = CompoundApproachVia(parts,tool_aware=True,table_z=0.,floor_z=.012)
    rotation = Rotation.from_euler('y',-90,degrees=True).as_matrix()
    position = np.array([0.,0.,.05])
    via.set_geometry_context(position,rotation,.0092)
    goals = np.array([[-.02,0.,.028],[.03,0.,.028]])
    sphere_only = CompoundApproachVia(parts,floor_z=.012)
    sphere_only.set_geometry_context(position,rotation,.0092)
    assert np.all(sphere_only.free(goals))
    np.testing.assert_array_equal(via.free(goals),[False,True])
    # A separate GJK query with the authored wrist/cylinder agrees with the
    # Minkowski planning obstacles at the accepted target.
    _,gaps = query.contacts(position,rotation,position+rotation@goals[1],
        rotation@CAVITY_TOOL_ROTATION)
    assert gaps['shaft'] >= .0009 and gaps['wrist'] >= .0009
    start = np.array([.09,0.,.028])
    path = via._plan(start,goals[1])
    assert path is not None
    for a,b in zip([start]+path[:-1],path):
        for point in np.linspace(a,b,30):
            _,gaps = query.contacts(position,rotation,position+rotation@point,
                rotation@CAVITY_TOOL_ROTATION)
            assert gaps['sphere'] >= -.0008
            assert gaps['shaft'] >= .0009 and gaps['wrist'] >= .0009


def test_suspended_cup_is_unloaded_until_quiet_table_support():
    guard = MugUnloadingGuard()
    contacts = [dict(kind='sphere',gap=0.,normal=np.array([1.,0.,0.]))]
    assert guard.update(.002,np.zeros(6),contacts,np.zeros(3)) is None
    force = guard.update(.01,np.zeros(6),contacts,np.zeros(3))
    assert force[0] > 0. and force[2] == 0.  # unload, do not press/lift
    for _ in range(30):
        assert guard.update(.002,np.zeros(6),contacts,np.zeros(3)) is not None
    for _ in range(24):
        assert guard.update(0.,np.zeros(6),contacts,np.zeros(3)) is not None
    assert guard.update(0.,np.zeros(6),contacts,np.zeros(3)) is None


def test_wrist_rotation_is_checked_at_the_contact_point():
    jac = np.zeros((6,7));jac[:6,:6] = np.eye(6)
    torque = np.zeros(7);torque[4] = 1.
    contacts = [dict(kind='shaft',gap=0.,normal=np.array([1.,0.,0.]),
        point=np.array([0.,0.,-.06]))]
    result,active = collision_torque_projection(torque,np.eye(7),jac,np.zeros(7),
        np.zeros(3),contacts,3.)
    row = np.array([1.,0.,0.,0.,-.06,0.,0.])
    assert active and row@torque < 0. and row@result > 0.


def test_current_geometry_replaces_stale_physx_overlap():
    from types import SimpleNamespace
    from examples.mpc.franka.runtime.mpc import ContactIsaacCartesian
    param = SimpleNamespace(n_qvel_=9,max_ncon_=10,mu_object_=.5,table_height=0.,
        physx_contact_offset_=.002)
    raw = np.zeros(1,dtype=[('initial_overlap','f4')])
    raw['initial_overlap'] = .004
    pose = np.array([0.,0.,.1,1.,0.,0.,0.,.05,0.,.1])
    current = dict(kind='sphere',gap=.001,normal=np.array([1.,0.,0.]),point=np.array([.04,0.,.1]))
    sim = SimpleNamespace(param_=param,obj_body_idx=11,table_body_idx=12,
        franka_body_name_to_index={'fingertip':7},env=None,
        gym=SimpleNamespace(get_env_rigid_contacts=lambda *_:raw),
        get_state=lambda:pose,get_end_effector_pos=lambda:(pose[7:],np.eye(3)),
        _contact_bodies=lambda c:(11,7),
        _contact_sep_normal=lambda c:(-.004,False,np.array([1.,0.,0.])),
        _contact_world_pos=lambda *a:np.array([.044,0.,.1]),
        tool_collision_geometry=lambda:([current],dict(sphere=.001,shaft=.003,wrist=.003)),
        _skew=lambda v:np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]]))
    detector = ContactIsaacCartesian(param)
    phi,_,_,_,touching = detector.detect_once(sim)
    assert not touching
    assert phi[0] == pytest.approx(.001)
    measured = detector.get_actual_fingertip_contact()
    assert measured['dist'] == pytest.approx(.001)
    np.testing.assert_allclose(measured['point_world'],[.039,0.,.1])


def test_sideways_entry_leaves_vertical_wrist_clearance():
    frame = cavity_entry_rotation([1.,0.,0.,0.])
    back = -frame[:,2]
    assert back[2] > .4 and back[0] > .9
    np.testing.assert_allclose(frame.T@frame,np.eye(3),atol=1e-12)
    upright = Rotation.from_euler('y',-90,degrees=True).as_quat()[[3,0,1,2]]
    np.testing.assert_allclose(cavity_entry_rotation(upright),CAVITY_TOOL_ROTATION,atol=1e-12)


def test_collision_guard_uses_velocity_relative_to_the_moving_cup():
    jac = np.zeros((6,7));jac[:6,:6] = np.eye(6)
    contact = dict(kind='shaft',gap=0.,normal=np.array([1.,0.,0.]),point=np.zeros(3))
    qvel = np.array([.1,0.,0.,0.,0.,0.,0.])
    moving = (np.array([.1,0.,0.]),np.zeros(3),np.zeros(3))
    torque,active = collision_torque_projection(np.zeros(7),np.eye(7),jac,qvel,
        np.zeros(3),[contact],3.,object_motion=moving)
    assert active and torque[0] > 0.  # equal absolute velocities have not separated


def test_unloading_slides_a_supporting_ball_out_along_the_table():
    guard = MugUnloadingGuard()
    contacts = [dict(kind='sphere',gap=0.,normal=np.array([0.,0.,-1.]))]
    force = guard.update(.01,np.zeros(6),contacts,np.zeros(3),floor_gap=0.,
        escape_direction=[-1.,0.,0.])
    assert force[0] < -1. and force[2] == 0.
    assert np.linalg.norm(force) <= 4.
