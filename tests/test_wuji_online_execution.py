from types import SimpleNamespace

import mujoco
import numpy as np
import trimesh

from examples.retarget.visualization import DemonstrationOverlay
from examples.retarget.offline_contacts import restore_plan
from models.wuji_hand import MaterialPoint, ObjectSurface
from planning.wuji_regrasp import OnlineRegrasp, RegraspSettings


def test_ghost_mesh_deforms_in_world_coordinates_without_changing_physics(tmp_path):
    scene=tmp_path/'scene.xml'
    scene.write_text('''<mujoco><asset/><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="obj" pos="0 0 .1"><freejoint/><geom type="box" size=".02 .03 .04" mass=".5"/></body>
      </worldbody></mujoco>''')
    mesh=trimesh.creation.icosphere(subdivisions=1,radius=.03)
    vertices=np.stack([mesh.vertices+[.1,0,.1],mesh.vertices*[1,1,2]+[.12,.01,.15]])
    poses=np.array([[0,0,.1,1,0,0,0],[.02,0,.15,1,0,0,0]])
    overlay=DemonstrationOverlay(scene,vertices,mesh.faces,poses)
    model=mujoco.MjModel.from_xml_path(str(scene));physical=mujoco.MjData(model)
    mujoco.mj_forward(model,physical)
    before=physical.qpos.copy()
    assert overlay.update(physical,1)
    np.testing.assert_array_equal(physical.qpos,before)
    compiled=overlay.model.mesh_vert[overlay.va:overlay.va+overlay.vn]
    world=compiled@overlay.rotation.T+overlay.origin
    np.testing.assert_allclose(world,vertices[1,overlay.vertex_ids],atol=1e-7)
    assert not overlay.update(physical,1)
    assert (model.nq,model.nv,model.nu)==(overlay.model.nq,overlay.model.nv,overlay.model.nu)
    extra=np.arange(model.ngeom,overlay.model.ngeom)
    assert np.all(overlay.model.geom_contype[extra]==0)
    assert np.all(overlay.model.geom_conaffinity[extra]==0)
    assert np.all(overlay.model.geom_rgba[extra,3]<1)
    for _ in range(10):
        mujoco.mj_step(model,physical);mujoco.mj_step(overlay.model,overlay.data)
    np.testing.assert_allclose(physical.qpos,overlay.data.qpos,atol=1e-12)
    np.testing.assert_allclose(physical.qvel,overlay.data.qvel,atol=1e-12)


def recovery_fixture():
    surface=ObjectSurface(trimesh.creation.box(extents=[.04,.04,.04]),samples=100)
    pose=np.array([0.,0.,.02,1,0,0,0])
    materials=[MaterialPoint(1,np.zeros(3),np.array([-1.,0,0]),'thumb','distal'),
        MaterialPoint(2,np.zeros(3),np.array([1.,0,0]),'index','distal')]
    anchors=np.array([[.02,-.01,0],[-.02,0,0]])
    original=anchors.copy();original[0,1]=.01
    normals=np.array([[-1.,0,0],[1.,0,0]])
    hand=SimpleNamespace(surface=surface,object_pose=lambda:pose.copy(),points=pose[:3]+anchors,
        friction=.5,dynamics=lambda:(np.eye(6),np.ones(1),np.array([0,0,-5.,0,0,0])))
    hand.kinematics=lambda materials:(hand.points.copy(),normals,None)
    actual=[dict(slot=k,body=k+1,point=hand.points[k].copy(),normal=normals[k],
        force=normals[k]*(.1 if k==0 else 1.),gap=0.,tangents=np.array([[0,1,0],[0,0,1]])) for k in range(2)]
    return hand,materials,anchors,original,normals,actual


def test_recovery_requires_real_separation_then_returns_to_mano_not_arbitrary_anchor():
    hand,materials,anchors,original,normals,actual=recovery_fixture()
    controller=OnlineRegrasp(RegraspSettings(failure_cycles=1,phase_cycles=20))
    certificate=dict(force_closure=False,load_feasible=False)
    def update(contacts):
        return controller.update(hand,materials,anchors,normals,np.ones(2),contacts,certificate,
            np.array([0,0,.1,1,0,0,0]),original)
    update(actual)
    selected,_,report,_=update(actual)
    assert report['phase']=='release' and report['slot']==0
    assert report['target_source']=='MANO_object_anchor'
    np.testing.assert_array_equal(report['target'],original[0])
    assert not np.allclose(selected[0],original[0])
    _,_,still,_=update(actual)
    assert still['phase']=='release'  # Time cannot fabricate a release.
    hand.points[0]=hand.object_pose()[:3]+controller.move['retreat']
    returned,_,approach,_=update(actual[1:])
    assert approach['phase']=='approach'
    np.testing.assert_array_equal(returned[0],original[0])
    hand.points[0]=hand.object_pose()[:3]+original[0]-.003*normals[0]
    _,_,approach,_=update(actual[1:])
    assert approach['phase']=='approach'
    hand.points[0]=hand.object_pose()[:3]+original[0]
    for _ in range(3):_,_,settled,_=update(actual)
    assert settled['phase']=='track' and settled['event']=='measured_recontact'
    assert settled['persistent_slots']==[0]


def test_nearby_recontact_remains_soft_even_when_mano_anchor_is_unreachable():
    hand,materials,anchors,original,normals,actual=recovery_fixture()
    controller=OnlineRegrasp(RegraspSettings(failure_cycles=1))
    kwargs=dict(hand=hand,materials=materials,anchors=anchors,normals=normals,active=np.ones(2),
        actual=actual,certificate=dict(load_feasible=False),goal=hand.object_pose(),original=original)
    controller.update(**kwargs);controller.update(**kwargs)
    hand.points[0]=hand.object_pose()[:3]+controller.move['retreat']
    kwargs['actual']=actual[1:];controller.update(**kwargs)
    # A real nearby touch need not reproduce MANO exactly. Its soft MANO
    # objective persists after the measured recovery is acknowledged.
    hand.points[0]=hand.object_pose()[:3]+anchors[0]
    kwargs['actual']=actual
    for _ in range(3):_,_,report,_=controller.update(**kwargs)
    assert report['event']=='measured_recontact'
    returned,_,_,_=controller.update(**kwargs)
    np.testing.assert_allclose(returned[0],original[0])


def test_geometric_recovery_visits_each_misplaced_finger_without_force_ranking():
    hand,materials,anchors,original,normals,actual=recovery_fixture()
    controller=OnlineRegrasp(RegraspSettings(failure_cycles=1))
    kwargs=dict(hand=hand,materials=materials,anchors=anchors,normals=normals,active=np.ones(2),
        actual=actual,certificate=dict(load_feasible=False),goal=hand.object_pose(),original=original)
    repaired=[]
    for _ in range(2):
        # Move the next desired region tangentially. A failed load certificate
        # alone must no longer trigger another release of a correct grasp.
        if repaired:original[1,1]+=.01
        controller.update(**kwargs)
        if controller.move is None:controller.update(**kwargs)
        moving=controller.move['slot'];repaired.append(controller.move['finger'])
        hand.points[moving]=hand.object_pose()[:3]+controller.move['retreat']
        kwargs['actual']=[c for c in actual if c['slot']!=moving]
        controller.update(**kwargs)
        hand.points[moving]=hand.object_pose()[:3]+original[moving]
        kwargs['actual']=actual
        for _ in range(3):_,_,report,_=controller.update(**kwargs)
        assert report['event']=='measured_recontact'
    assert repaired==['thumb','index']


def test_recovery_ignores_force_magnitude_and_load_verdict_when_regions_match():
    hand,materials,anchors,original,normals,actual=recovery_fixture()
    original=anchors.copy()
    for verdict in (False,True):
        controller=OnlineRegrasp(RegraspSettings(failure_cycles=1))
        for c in actual:c['force']=np.zeros(3)
        for _ in range(5):
            _,_,report,_=controller.update(hand,materials,anchors,normals,np.ones(2),actual,
                dict(force_closure=verdict,load_feasible=verdict),hand.object_pose(),original)
            assert report['phase']=='track' and not report['moving_slots']


def test_release_checks_native_gap_even_after_contact_force_falls_below_cutoff():
    hand,materials,anchors,original,normals,actual=recovery_fixture()
    controller=OnlineRegrasp(RegraspSettings(failure_cycles=1))
    kwargs=dict(hand=hand,materials=materials,anchors=anchors,normals=normals,active=np.ones(2),
        actual=actual,certificate=dict(load_feasible=False),goal=hand.object_pose(),original=original)
    controller.update(**kwargs);controller.update(**kwargs)
    hand.points[0]=hand.object_pose()[:3]+controller.move['retreat']
    kwargs['actual']=actual[1:]  # No loaded thumb patch reported.
    hand.finger_contact_gaps=lambda:{'thumb':-.0001}
    _,_,report,_=controller.update(**kwargs)
    assert report['phase']=='release'
    hand.finger_contact_gaps=lambda:{}
    _,_,report,_=controller.update(**kwargs)
    assert report['phase']=='approach' and report['event']=='measured_separation'


def test_recontact_on_another_link_of_same_finger_does_not_finish_recovery():
    hand,materials,anchors,original,normals,actual=recovery_fixture()
    controller=OnlineRegrasp(RegraspSettings(failure_cycles=1))
    kwargs=dict(hand=hand,materials=materials,anchors=anchors,normals=normals,active=np.ones(2),
        actual=actual,certificate=dict(load_feasible=False),goal=hand.object_pose(),original=original)
    controller.update(**kwargs);controller.update(**kwargs)
    hand.points[0]=hand.object_pose()[:3]+controller.move['retreat']
    kwargs['actual']=actual[1:];controller.update(**kwargs)
    other_link=dict(actual[0],body=99)
    kwargs['actual']=[other_link,actual[1]]
    for _ in range(4):_,_,report,_=controller.update(**kwargs)
    assert report['phase']=='approach' and report['event']!='measured_recontact'
    kwargs['actual']=actual
    for _ in range(3):_,_,report,_=controller.update(**kwargs)
    assert report['event']=='measured_recontact'


def test_recovery_timeout_does_not_claim_recontact_and_reference_release_cancels():
    hand,materials,anchors,original,normals,actual=recovery_fixture()
    controller=OnlineRegrasp(RegraspSettings(failure_cycles=1,phase_cycles=2))
    kwargs=dict(hand=hand,materials=materials,anchors=anchors,normals=normals,active=np.ones(2),
        actual=actual,certificate=dict(load_feasible=False),goal=hand.object_pose(),original=original)
    controller.update(**kwargs);controller.update(**kwargs)
    _,_,report,_=controller.update(**kwargs)
    assert report['event']=='phase_timeout' and not controller.overrides
    controller.update(**kwargs)
    kwargs['active']=np.zeros(2)
    _,_,report,_=controller.update(**kwargs)
    assert not report['moving_slots'] and controller.move is None


def test_offline_plan_restores_material_identity():
    value=dict(valid=True,slots=[2],points=[[.1,0,0]],normals=[[1,0,0]],force_vectors_local=[[3,0,0]],
        ik=dict(q=[0,1],materials=[dict(body=4,local=[.01,0,0],normal=[1,0,0],finger='thumb',region='distal',source_local=None)]))
    plan=restore_plan(value)
    assert plan['slots']==(2,) and plan['ik']['materials'][0].finger=='thumb'
    np.testing.assert_array_equal(plan['ik']['q'],[0,1])


def test_infeasible_transition_can_approach_upcoming_valid_grasp_without_skipping_hand_frames():
    from examples.retarget.offline_contacts import execution_reference
    plans=[dict(valid=True,slots=[]),dict(valid=False,slots=[0]),
        dict(valid=False,slots=[0]),dict(valid=True,slots=[0,1,2]),dict(valid=True,slots=[])]
    assert execution_reference(plans,0,3)==0  # Preserve a feasible current phase.
    assert execution_reference(plans,1,2)==1  # No lookahead beyond object progress.
    assert execution_reference(plans,1,3)==3
    assert execution_reference(plans,3,4)==3
    plans[3]['valid']=False
    assert execution_reference(plans,1,4)==1  # A later release is not a grasp.
