"""A hollow collision model also needs a reachable, sphere-clear approach."""
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from examples.mpc.fingertips.test.params import canonical_object_meshes
from models.object_collision_mesh import decomposed_collision_mesh
from planning.compound_contact_path import CompoundApproachVia


@pytest.fixture(scope='module')
def mug():
    root = Path(__file__).resolve().parents[1]
    _, visual = canonical_object_meshes(str(root/'envs/assets/objects/mug.stl'),
        str(root/'envs/xmls/env_fingertips_mug.xml'))
    _, paths = decomposed_collision_mesh(visual, object_name='mug')
    parts = [trimesh.load_mesh(path) for path in paths]
    return parts, paths


def test_near_inner_wall_requires_a_path_through_the_opening(mug):
    parts, paths = mug
    via = CompoundApproachVia(parts, floor_z=0.)
    # Canonical cup axis +X. Rotate it upright; inner and outer sphere
    # centers are only 23 mm apart, but a cup wall separates them.
    rotation = Rotation.from_euler('y', -90, degrees=True).as_matrix()
    via.set_geometry_context([0., 0., .05], rotation, .0092)
    start = np.array([0., 0., .051])
    goal = np.array([0., 0., .028])
    assert np.linalg.norm(start-goal) < .03
    assert np.all(via.free([start, goal]))
    assert not via.segment_clear(start, goal)
    path = via._plan(start, goal)
    assert path is not None
    np.testing.assert_allclose(path[-1], goal)
    assert max(point[0] for point in path) > .041  # beyond the cup rim

    # Independent collision check: MuJoCo cooks each convex part itself.
    assets = ''.join('<mesh name="m%d" file="%s"/>' % (i,p) for i,p in enumerate(paths))
    geoms = ''.join('<geom type="mesh" mesh="m%d"/>' % i for i in range(len(paths)))
    model = mujoco.MjModel.from_xml_string('<mujoco><asset>'+assets+'</asset><worldbody>'
        '<body>'+geoms+'</body><body><freejoint/><geom name="tip" type="sphere" size=".01"/>'
        '</body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    tip = model.geom('tip').id
    for a,b in zip([start]+path[:-1], path):
        for point in np.linspace(a,b,40):
            data.qpos[:3], data.qpos[3:7] = point, [1.,0.,0.,0.]
            mujoco.mj_forward(model,data)
            gaps = [c.dist for c in data.contact if tip in (c.geom1,c.geom2)]
            # The approach clearance is 9.2 mm; use the actual 10 mm sphere
            # and explicitly bound its intentional contact loading margin.
            assert not gaps or min(gaps) >= -.0008-1e-6


def test_reachable_final_leg_is_not_replaced_by_a_35mm_standoff(mug):
    via = CompoundApproachVia(mug[0])
    via.set_geometry_context(np.zeros(3), np.eye(3), .0092)
    target = np.array([0., 0., .028])
    tip = target + [0.,0.,-.002]
    surface = np.array([0., 0., .0375])
    use, point, phase = via.update(tip, np.zeros(3), surface, target,
        .08, .1, False, target)
    assert not use and phase == 'drop'
    np.testing.assert_allclose(point, target)
    assert not via.path_obstructed(tip,target)


@pytest.mark.parametrize('retarget', [True, False])
def test_accepted_outer_target_replaces_an_unfinished_inner_route(mug, retarget):
    via = CompoundApproachVia(mug[0])
    rotation = Rotation.from_euler('y', -90, degrees=True).as_matrix()
    origin = np.array([0., 0., .05])
    via.set_geometry_context(origin, rotation, .0092)
    start, inner = np.array([0., 0., .051]), np.array([0., 0., .028])
    tip, inner_world = origin + rotation @ start, origin + rotation @ inner
    use, _, _ = via.update(tip, origin, inner_world, inner_world, .1, .1,
        False, inner_world)
    assert use and via.blocked
    assert len(via.route) > 1
    # Same observation, newly accepted exterior execution point. Its short
    # clear approach must supersede the old detour through the mouth.
    outer = start + [0., .001, 0.]
    outer_world = origin + rotation @ outer
    use, point, phase = via.update(tip, origin, outer_world, outer_world,
        .1, .1, False, outer_world, retarget=retarget)
    if retarget:
        assert not use and not via.blocked and phase == 'drop'
        np.testing.assert_allclose(via.destination_local, outer, atol=1e-12)
        np.testing.assert_allclose(point, outer_world)
    else:
        # Flip transit keeps its committed route until support is made.
        assert use and via.blocked and phase == 'cavity-transit'
        np.testing.assert_allclose(via.destination_local, inner, atol=1e-12)
    via.destination_idx = 7
    via.reset()
    assert via.destination_idx is None


@pytest.mark.parametrize('stage,confidence,travelling,protected,keep_outer', [
    ('pose', .01, False, False, False),
    ('flip', .01, False, True, True),
    ('pose', 1., True, False, True),
    ('pose', .01, True, False, False),
    ('flip', 1., True, False, False),
])
def test_contact_relocation_respects_measured_pose_progress_and_flip_support(
        stage, confidence, travelling, protected, keep_outer):
    from types import SimpleNamespace
    from examples.mpc.fingertips.test.test_0902 import (
        compute_rollout_contact_via, ContactValueTracker, ModelCostConfidence,
    )
    from planning.mlqp_point import LambdaContactControlOptimizer
    from planning.object_pose_cost import cost_config
    wall = trimesh.creation.box(extents=[.002,.06,.04])
    wall.apply_translation([.02,0.,.02])
    via = CompoundApproachVia([wall],floor_z=.35)
    opt = LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    points, normals = np.array([[.021,0.,.02],[.019,0.,.02]]), np.array([[-1.,0.,0.],[1.,0.,0.]])
    current = np.array([0.,0.,0.,1.,0.,0.,0.])
    target = current.copy(); target[0] = .08
    predicted = [current.copy(), current + [.01,0.,0.,0.,0.,0.,0.]]
    force = np.array([1.,0.,0.])
    opt.__dict__.update(sample_point=points, normal=normals, top_k=2,
        last_selected_idx=1,last_global_idx=1,last_executed_idx=0,
        last_global_total_cost=0.,last_best_x_plus=predicted[1],
        last_best_force=force,last_best_cost=0.,contact_switch_radius=.03,
        contact_switch_confidence=confidence,mu_arm_obj=.5,
        sample_geodesic=np.array([[0.,.002],[.002,0.]]),
        _object_cost_config=cost_config(500,20),
        update_Jacobian=lambda *a,**kw:None,
        get_availble_point_idx=lambda *a,**kw:np.array([0,1]),
        filter_rankable_indices=lambda ids:ids,
        choose_nearby_topk_idx=lambda query:1)
    def rank(*a,**kw):
        costs=opt._rescore_candidate_costs([0,1],[3.,3.],predicted,target,current)
        opt._store_candidate_buffers([0,1],costs,predicted,[force,force])
        return points[1],normals[1],0.,1.,1
    def solve(*a,**kw):
        return points[0],-normals[0],predicted[0],3.2,dict(control_input=force,solver_failed=False)
    opt.choose_contact_points=rank
    opt.resolve_executed_contact=opt.optimize_control_input=solve
    origin=np.array([.4,.1,.4])
    contact=dict(point_world=origin+points[0],normal_world=-normals[0])
    param=SimpleNamespace(lambda_optimizer=opt,target_p_=origin+target[:3],
        target_q_=target[3:],object_cost_stage_=stage,lambda_actual_contact_=contact)
    args=SimpleNamespace(ground_height_threshold=.012,ideal_contact_surface_margin=-.0005)
    tip=origin+points[0]-.0095*normals[0]
    if travelling:
        via.set_geometry_context(origin, np.eye(3), .0092)
        inner = origin + points[1] - .0095*normals[1]
        assert via.update(tip, origin, origin+points[1], inner, .5, .1,
            False, inner)[0]
        via.destination_idx = 1
    result=compute_rollout_contact_via(param,args,np.r_[origin,current[3:],tip],np.eye(3),
        np.zeros(6),np.zeros((40,6)),.01,ContactValueTracker(),ModelCostConfidence(),via,
        True,1,floor_ground=.362,floor_z=.35)
    value=result['value_info']
    assert result['best_contact_idx']==1
    assert value['flip_contact_protected'] == protected
    assert value['switch_observed_contact'] == (not keep_outer)
    assert value['accept_p_arm'] == keep_outer
    assert not result['arrived_at_best']
    assert value['path_blocked'] == (not keep_outer)
    np.testing.assert_allclose(result['p_arm_surface_world'],origin+points[0 if keep_outer else 1])
    if travelling and keep_outer:
        assert via.destination_idx == 0
        np.testing.assert_allclose(via.destination_local,
            result['p_arm_track_world']-origin)


def test_flip_contact_release_requires_settled_table_support():
    wall = trimesh.creation.box(extents=[.002,.06,.04])
    via = CompoundApproachVia([wall])
    via.set_geometry_context([0.,0.,.02],np.eye(3),.0092)
    for _ in range(5):
        assert via.protect_flip_contact(np.zeros(6),0.)
    assert not via.protect_flip_contact(np.zeros(6),0.)
    # Rotation, translation, or loss of table support starts a fresh wait.
    assert via.protect_flip_contact([0.,0.,0.,0.,0.,1.],0.)
    assert via.protect_flip_contact([.1,0.,0.,0.,0.,0.],0.)
    via.set_geometry_context([0.,0.,.04],np.eye(3),.0092)
    assert via.protect_flip_contact(np.zeros(6),0.)
    assert via.support_gap == pytest.approx(.02)


def test_body_collision_lattice_is_reused_when_floor_moves(mug):
    via = CompoundApproachVia(mug[0])
    via.set_geometry_context(np.zeros(3), np.eye(3), .0092)
    points, shape, mask = via.prepare_lattice()
    via.set_geometry_context([0.,0.,.2], Rotation.from_euler('y',90,degrees=True).as_matrix(), .0092)
    again, _, other = via.prepare_lattice()
    assert via.grid_builds == 1
    assert points is again
    np.testing.assert_array_equal(mask,other)
    other[:] = False
    assert np.any(via.prepare_lattice()[2])  # callers cannot corrupt the cache


def test_airborne_support_recovery_is_feasible_with_floor_force_and_slew():
    from planning.execution_constraints import supported_flip_command_bounds, floor_command_bounds
    lower,upper,active = supported_flip_command_bounds(.012,-.005,.005,.003)
    assert active and upper[2] < 0.
    stale = (np.array([-.001,-.001,.001]),np.array([.001,.001,.002]))
    bounds,info = floor_command_bounds(.4,.362,lower,upper,first_bounds=stale,
        previous=[0.,0.,.0015],slew=.001,max_norm=.003)
    assert info['rebased'] and not info['emergency']
    lo,hi = bounds
    assert np.all(lo <= hi) and hi[2] < 0.
    assert np.linalg.norm(np.clip(np.zeros(3),lo,hi)) <= .003
    assert lo[0] < 0. < hi[0]  # torque authority survives the recovery


def test_contact_flicker_retries_same_body_patch_with_bounded_patience():
    via = CompoundApproachVia([trimesh.creation.box(extents=[.02,.02,.02])])
    via.remember_contact([.01,0.,0.],[.0195,0.,0.])
    via.set_geometry_context([0.,0.,.2],np.eye(3),.0092)
    for _ in range(10):
        surface,track = via.reacquire_contact([.02,0.,.2])
        np.testing.assert_allclose(surface,[.01,0.,.2])
        np.testing.assert_allclose(track,[.0195,0.,.2])
    assert via.reacquire_contact([.02,0.,.2]) is None
    via.remember_contact([.01,0.,.2],[.0195,0.,.2])
    assert via.reacquire_contact([-.02,0.,.2]) is None  # cannot jump across the body
