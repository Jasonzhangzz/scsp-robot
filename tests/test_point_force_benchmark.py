"""Physical-model parity, continuous force execution and planning statistics."""
import copy
import numpy as np
import pytest
import mujoco

from models.corrected_scm import corrected_scm_velocity
from planning.point_force_benchmark import (PointForcePlanner,TrialRows,scene,surface_points,
    physical_problem,trial_summary)
from planning.object_pose_cost import ObjectCostStages,target_up_axis
from examples.mpc.fingertips.test.test_scm_mujoco import build_parser,_run_rollout


@pytest.mark.parametrize('rows',[0,1,4,16])
def test_corrected_impulse_matches_global_line_minimum_with_full_mass(rows):
    rng = np.random.default_rng(10)
    factor = rng.normal(size=(6,6))
    inverse = np.linalg.inv(factor.T@factor+np.eye(6))
    jac = rng.normal(size=(rows,6))
    free = rng.normal(size=(7,6))
    gap = rng.uniform(-.001,.003,rows)
    velocity,info = corrected_scm_velocity(free,jac,gap,inverse,.02)
    initial,impulse,alpha = (info[k] for k in ('initial_impulse','impulse','alpha'))
    W = jac@inverse@jac.T+1e-8*np.eye(rows)
    g = free@jac.T+np.maximum(gap,0)/.02
    assert np.all((alpha>=0)&(alpha<=1))
    np.testing.assert_allclose(velocity,free+impulse@jac@inverse)
    before = .5*np.einsum('ni,ij,nj->n',initial,W,initial)+np.sum(g*initial,axis=1)
    after = .5*np.einsum('ni,ij,nj->n',impulse,W,impulse)+np.sum(g*impulse,axis=1)
    assert np.all(after<=before+1e-10)
    if not rows:
        np.testing.assert_array_equal(velocity,free)
        np.testing.assert_array_equal(alpha,1.)


@pytest.mark.parametrize('model',['scm_corrected','contactsdf','lcp','qp'])
@pytest.mark.parametrize('stage',['flip','pose'])
def test_gpu_contact_objective_and_loaded_state_match_own_cpu_model(model,stage):
    args = build_parser().parse_args(['--sample_num','12','--population-size','8',
                                    '--population-iterations','2'])
    m,d,target,_ = scene(args,'foam_brick',0)
    points,frames = surface_points(m,d,args.sample_num,0)
    planner = PointForcePlanner(m,points,frames,target,args,model,0)
    original_pose,original_velocity = d.qpos.copy(),d.qvel.copy()
    stages = ObjectCostStages(args,target_up_axis(target[3:]))
    stages.stage=stage
    qpos = d.qpos.copy();qpos[2]-=.001
    snapshot = (qpos,np.array([.03,-.02,.01,.1,-.2,.15]),0.,0)
    report = planner.validate(snapshot,stages.config())
    assert report['passed'],report
    assert planner.backend == 'cuda-cem',report
    result = planner.solve(snapshot,stages.config())
    force = result['lam']
    assert force[0]>=0
    assert np.sum(np.abs(force[1:]))<=args.arm_friction*force[0]+1e-12
    assert np.linalg.norm(force)<=args.max_contact_force+1e-12
    body=m.body('obj').id
    fbody=frames[result['selected_idx']]@force
    assert args.policy_dt*np.linalg.norm(fbody)/m.body_mass[body]<=args.max_linear_force_delta_v+1e-12
    from scipy.spatial.transform import Rotation
    r=Rotation.from_quat(m.body_iquat[body][[1,2,3,0]]).as_matrix()
    inertia=r@np.diag(m.body_inertia[body])@r.T
    angular=args.policy_dt*np.linalg.solve(inertia,np.cross(points[result['selected_idx']]-m.body_ipos[body],fbody))
    assert np.linalg.norm(angular)<=args.max_angular_force_delta_v+1e-12
    np.testing.assert_array_equal(d.qpos,original_pose)
    np.testing.assert_array_equal(d.qvel,original_velocity)
    # Candidate queries are independently reset and reproducible.
    velocity,pose,cost,_ = planner.predict_numpy([force],[result['selected_idx']])
    again = planner.predict_numpy([force],[result['selected_idx']])
    np.testing.assert_array_equal(velocity,again[0])
    np.testing.assert_array_equal(pose,again[1])


def test_continuous_force_executes_every_substep_without_pose_assignment():
    args = build_parser().parse_args(['--sample_num','12','--population-size','8',
                                    '--population-iterations','2','--steps','3'])
    rows = _run_rollout(args,'scm_corrected','foam_brick')
    assert len(rows)==3
    assert [r['step'] for r in rows]==[1,2,3]
    assert not np.allclose(rows[0]['initial_object_pose'],rows[-1]['object_pose'])
    for row in rows:
        info = row['execution_info']
        assert info['pose_overwrites']==0
        assert info['active_substeps']==40
        assert info['applied_force_duration']==.02
        np.testing.assert_allclose(info['actual_applied_impulse_body'][:3],
            .02*row['applied_force_body'],atol=1e-14)
    assert len(rows.planning_rounds)==3


def test_async_reused_actions_do_not_add_zero_planning_samples():
    from planning.rollout_timing import trial_timing
    rows = TrialRows()
    rows.extend([dict(success=False,planning_updated=False,planning_time_ms=None)]*9)
    rows.planning_rounds = [dict(planning_time_ms=2.),dict(planning_time_ms=4.)]
    timing = trial_timing(rows)
    assert timing['planning_rounds']==2
    assert timing['planning_time_ms']['mean']==3.
    assert timing['execution_time_s'] is None
    summary = trial_summary(rows,'scm_corrected','foam_brick',0,True)
    assert summary['steps']==9 and summary['completion_steps'] is None
    assert summary['planning_time_ms']['n']==2


def test_render_process_receives_an_independent_model_and_data():
    args = build_parser().parse_args([])
    model,data,_,_ = scene(args,'foam_brick',0)
    display = copy.copy(model)
    display_data = mujoco.MjData(display)
    display.body('marker').pos = [1.,2.,3.]
    display_data.qpos[:3] = 1.
    assert not np.array_equal(model.body('marker').pos,display.body('marker').pos)
    assert not np.array_equal(data.qpos,display_data.qpos)


def test_native_auto_reset_stops_at_failed_substep_and_preserves_before_state(monkeypatch):
    from planning.point_force_benchmark import RolloutFailure
    args = build_parser().parse_args(['--sample_num','12','--population-size','8',
                                    '--population-iterations','2','--steps','3'])
    actual = mujoco.mj_step
    calls = []
    def reset_on_third_substep(model,data):
        calls.append(float(data.time))
        actual(model,data)
        if len(calls)==3:
            mujoco.mj_resetData(model,data)
            data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
    monkeypatch.setattr(mujoco,'mj_step',reset_on_third_substep)
    with pytest.raises(RolloutFailure) as caught:
        _run_rollout(args,'scm_corrected','foam_brick')
    assert len(calls)==3
    rows = caught.value.rows
    assert len(rows)==1 and not rows[-1]['success']
    row = rows[-1]
    assert row['physics_fault']['clock_reset']
    assert row['physics_fault']['warning_changed']
    assert row['execution_substeps']==2 and not row['execution_completed']
    assert row['completed_execution_steps']==0
    assert row['simulation_time_s']==pytest.approx(.001)
    np.testing.assert_array_equal(row['object_pose'],row['physics_fault']['pose_before'])
    summary = trial_summary(rows,'scm_corrected','foam_brick',0,False)
    assert summary['steps']==0 and summary['attempted_steps']==1


@pytest.mark.parametrize('failure_type',[RuntimeError,TimeoutError])
def test_failed_async_solve_is_timed_once_even_during_shutdown(monkeypatch,failure_type):
    import planning.point_force_benchmark as benchmark
    original = benchmark.PointForcePlanner
    class FailSecondSolve(original):
        def solve(self,snapshot,config,**kwargs):
            if self.round:
                exc = failure_type('deliberate asynchronous solve failure')
                exc.planning_time_ms = .125
                exc.observation_step,exc.planning_round = snapshot[3],self.round+1
                raise exc
            return super().solve(snapshot,config,**kwargs)
    monkeypatch.setattr(benchmark,'PointForcePlanner',FailSecondSolve)
    args = build_parser().parse_args(['--async-planner','--sample_num','12',
        '--population-size','8','--population-iterations','2','--max-contact-force','.2','--steps','10'])
    with pytest.raises(benchmark.RolloutFailure) as caught:
        _run_rollout(args,'scm_corrected','foam_brick')
    rounds = caught.value.rows.planning_rounds
    failed = [r for r in rounds if r.get('error')]
    # A failed future is consumed before propagating its error, so shutdown
    # cannot record it a second time.
    assert len(failed)==1 and failed[0]['planning_time_ms']>=.125
    assert len(rounds)==2
    summary = trial_summary(caught.value.rows,'scm_corrected','foam_brick',0,True)
    assert summary['planning_time_ms']['n']==2
    cfg=copy.deepcopy(args)
    cfg.replay_schedule=copy.deepcopy(rounds)
    with pytest.raises(benchmark.RolloutFailure) as replayed:
        _run_rollout(cfg,'scm_corrected','foam_brick')
    from examples.mpc.fingertips.test.benchmark_contact_models import compare_trials
    assert compare_trials(caught.value.rows,replayed.value.rows)['passed']


def test_catchup_failure_records_the_completed_unused_plan_once(monkeypatch):
    import time
    import planning.point_force_benchmark as benchmark
    original=benchmark.PointForcePlanner
    class FailsDuringCatchup(original):
        def solve(self,snapshot,config,**kwargs):
            if self.round>=2:
                exc=RuntimeError('deliberate catchup failure')
                exc.planning_time_ms=.125
                exc.observation_step,exc.planning_round=snapshot[3],self.round+1
                raise exc
            if self.round==1:
                time.sleep(.025)
            return super().solve(snapshot,config,**kwargs)
    monkeypatch.setattr(benchmark,'PointForcePlanner',FailsDuringCatchup)
    args=build_parser().parse_args(['--async-planner','--sample-num','8',
        '--population-size','4','--population-iterations','1','--steps','8',
        '--async-result-wait-ms','15'])
    with pytest.raises(benchmark.RolloutFailure) as caught:
        _run_rollout(args,'scm_corrected','foam_brick')
    rounds=caught.value.rows.planning_rounds
    assert [r['planning_round'] for r in rounds]==[1,2,3]
    assert rounds[1]['discard_reason']=='later_planning_failure'
    assert not rounds[1]['published']
    assert sum(bool(r.get('error')) for r in rounds)==1


@pytest.mark.parametrize('removed',['soft-scm','scm','diagonal-contact','scm_soft_corrected'])
def test_removed_models_cannot_be_selected(removed):
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--contact-model',removed])


@pytest.mark.parametrize('name',['scm_corrected','contactsdf','lcp','qp'])
def test_full_period_prediction_matches_executed_free_motion(name):
    from planning.point_force_prediction import CPUSubstepPredictor
    args=build_parser().parse_args([])
    model,data,_,_=scene(args,'foam_brick',0)
    pose=data.qpos.copy();pose[2]+=1.
    initial_velocity=np.array([.02,-.03,.01,.1,-.2,.15])
    point=np.array([.03,.01,.02]);force=np.array([.1,.02,-.01])
    reference=CPUSubstepPredictor(model,args,'qp')
    expected_velocity,expected_pose,_=reference.predict(pose,initial_velocity,[point],[force])
    predictor=CPUSubstepPredictor(model,args,name)
    velocity,predicted,_=predictor.predict(pose,initial_velocity,[point],[force])
    np.testing.assert_allclose(predicted,expected_pose,atol=1e-10,rtol=0)
    np.testing.assert_allclose(velocity,expected_velocity,atol=1e-9,rtol=0)
    # Querying a different candidate must not contaminate the first one.
    predictor.predict(pose,initial_velocity,[point],[-force])
    again=predictor.predict(pose,initial_velocity,[point],[force])
    np.testing.assert_array_equal(predicted,again[1])


def test_async_publication_replay_preserves_actual_search_budget():
    from examples.mpc.fingertips.test.benchmark_contact_models import compare_trials
    args=build_parser().parse_args(['--async-planner','--sample-num','8',
        '--population-size','4','--population-iterations','3','--steps','8',
        '--async-planning-budget-ms','.001',
        '--max-contact-force','.2'])
    original=_run_rollout(args,'qp','foam_brick')
    cfg=copy.deepcopy(args);cfg.replay_schedule=copy.deepcopy(original.planning_rounds)
    replayed=_run_rollout(cfg,'qp','foam_brick')
    assert compare_trials(original,replayed)['passed']
    assert original.planning_rounds[0]['observation_step']==0
    assert original.planning_rounds[1]['observation_step']==1  # next fresh observation, not duplicate bootstrap
    for row in original:
        if not row['command_valid_for_step']:
            np.testing.assert_array_equal(row['applied_force_body'],np.zeros(3))
    for call in original.planning_rounds:
        assert 'lead_steps' not in call
        assert call['search_iterations']==(3 if call['planning_round']==1 else 1)
        if call['published']:
            assert call['published_step']>=call['observation_step']


@pytest.mark.parametrize('obj',['foam_brick','mug','rubber_duck','elephant',
                               'piggy_bank','stanford_bunny2','teapot'])
def test_supported_goal_clears_real_compiled_collision_mesh(obj):
    args=build_parser().parse_args([])
    model,data,target,_=scene(args,obj,0)
    initial=data.qpos.copy()
    query=mujoco.MjData(model);query.qpos[:]=target;mujoco.mj_forward(model,query)
    gid=model.geom('obj').id;mid=int(model.geom_dataid[gid])
    start,count=int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid])
    world=model.mesh_vert[start:start+count]@query.geom_xmat[gid].reshape(3,3).T+query.geom_xpos[gid]
    assert world[:,2].min()-query.geom_xpos[model.geom('table').id,2]==pytest.approx(args.target_clearance,abs=1e-10)
    np.testing.assert_array_equal(data.qpos,initial)
    np.testing.assert_allclose(model.body('goal').pos,target[:3])
    args.target_height_mode='fixed'
    _,_,legacy,_=scene(args,obj,0)
    assert legacy[2]==.03
    np.testing.assert_array_equal(legacy[3:],target[3:])


@pytest.mark.parametrize('obj',['foam_brick','mug','rubber_duck','elephant',
                               'piggy_bank','stanford_bunny2','teapot'])
def test_mesh_mounting_is_removed_from_both_sampled_orientations(obj):
    from planning.point_force_benchmark import object_asset_rotation
    from scipy.spatial.transform import Rotation
    args=build_parser().parse_args([])
    _,asset_data,asset_target,_=scene(args,obj,1)
    args.target_frame='body'
    _,legacy_data,legacy_target,_=scene(args,obj,1)
    mounting=object_asset_rotation(obj)
    for corrected,requested in ((asset_target,legacy_target),(asset_data.qpos,legacy_data.qpos)):
        actual=Rotation.from_quat(corrected[[4,5,6,3]])*mounting
        expected=Rotation.from_quat(requested[[4,5,6,3]])
        assert (actual.inv()*expected).magnitude()<1e-12
        np.testing.assert_array_equal(corrected[:2],requested[:2])


@pytest.mark.parametrize('obj',['mug','teapot','rubber_duck'])
def test_default_goal_is_stable_after_native_unforced_release(obj):
    args=build_parser().parse_args([])
    model,data,target,_=scene(args,obj,0)
    data.qpos[:]=target;data.qvel[:]=0.;mujoco.mj_forward(model,data)
    for cycle in range(100):
        for _ in range(args.force_physics_substeps):
            mujoco.mj_step(model,data)
        if cycle>=10:
            from planning.point_force_benchmark import errors,successful
            assert successful(*errors(data.qpos,target))
            assert np.linalg.norm(data.qvel[:3])<.05
            assert np.linalg.norm(data.qvel[3:])<.5


@pytest.mark.parametrize('name',['scm_corrected','contactsdf'])
def test_pose_only_nine_value_configuration_uploads_and_matches_cpu(name):
    args=build_parser().parse_args(['--no-two-stage-object-cost','--sample-num','8',
                                  '--population-size','4','--population-iterations','1'])
    model,data,target,_=scene(args,'mug',0)
    points,frames=surface_points(model,data,args.sample_num,0)
    planner=PointForcePlanner(model,points,frames,target,args,name,0)
    config=ObjectCostStages(args,target_up_axis(target[3:])).config()
    assert len(config)==9
    assert planner.validate((data.qpos.copy(),data.qvel.copy(),0.,0),config)['passed']


def test_settled_state_between_old_and_documented_threshold_is_physically_verified(monkeypatch):
    import planning.point_force_benchmark as benchmark
    original=benchmark.scene
    def near_goal(*args,**kwargs):
        model,data,target,xml=original(*args,**kwargs)
        data.qpos[:]=target;data.qvel[:]=0.
        # Let the native soft contact reach rest before testing the success
        # band; initial target clearance otherwise creates a small impact.
        for _ in range(1000):
            mujoco.mj_step(model,data)
        data.qpos[0]+=.018;data.time=0.;mujoco.mj_forward(model,data)
        return model,data,target,xml
    monkeypatch.setattr(benchmark,'scene',near_goal)
    args=build_parser().parse_args(['--sample-num','8','--population-size','4',
        '--population-iterations','1','--steps','10'])
    rows=_run_rollout(args,'qp','foam_brick')
    assert len(rows)==10 and rows[-1]['success']
    assert rows[-1]['success_hold']==10
    for row in rows:
        assert row['settling'] and row['settled']
        assert .016<row['goal_pos']<.02
        np.testing.assert_array_equal(row['applied_force_body'],np.zeros(3))
        assert row['execution_info']['pose_overwrites']==0


def test_held_force_cannot_renew_its_impulse_budget_or_use_stale_observations():
    from planning.point_force_benchmark import held_command_status
    args=build_parser().parse_args([])
    model,data,_,_=scene(args,'foam_brick',0)
    pose=data.qpos.copy();pose[2]=1.
    config=np.arange(9,dtype=float)
    command=dict(force_local=np.array([.1,0,0]),p_local=model.body_ipos[model.body('obj').id].copy(),
                 objective_config=config.copy())
    assert held_command_status(model,pose,command,config,.02,.02,args)=='valid'
    assert held_command_status(model,pose,command,config,.04,.04,args)=='impulse_budget_exhausted'
    assert held_command_status(model,pose,command,config,.2,.02,args)=='stale_observation'
    assert held_command_status(model,pose,command,config+1,.02,.02,args)=='objective_changed'
    pose[2]=-1.
    assert held_command_status(model,pose,command,config,.02,.02,args)=='contact_below_ground'


@pytest.mark.parametrize('ready_after',[1,2,'recover'])
def test_worker_publishes_in_current_cycle_or_keeps_bounded_force_while_late(monkeypatch,ready_after):
    import planning.point_force_benchmark as benchmark
    from examples.mpc.fingertips.test.benchmark_contact_models import compare_trials
    class MeasuredPlanner:
        def __init__(self,model,points,frames,target,args,name,seed):
            self.backend='cuda-cem';self.search_budget_ms=0.;self.round=0
            self.points,self.frames=points,frames
        def validate(self,*args):
            return dict(passed=True)
        def solve(self,snapshot,config,iterations=None):
            from scipy.spatial.transform import Rotation
            pose,vel,stamp,step=snapshot;self.round+=1
            rot=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
            index=int(np.argmax((self.points@rot.T+pose[:3])[:,2]))
            lam=np.array([.001,0.,0.])
            return dict(selected_idx=index,p_local=self.points[index].copy(),
                force_local=self.frames[index]@lam,frame=self.frames[index],lam=lam,
                predicted_pose=pose.copy(),velocity=vel.copy(),planning_state_pose=pose.copy(),
                objective_config=config.copy(),observation_step=step,observation_time=stamp,
                planning_round=self.round,planning_time_ms=1.,control_deadline_exceeded=False,
                solver_backend=self.backend,gpu_parallel_prediction=True,solver_fallback_reason=None,
                search_iterations=1)
    class DelayedFuture:
        def __init__(self,fn,*args):
            self.value=fn(*args);self.polls=0
            self.ready_after=(2 if (self.value or {}).get('planning_round')==2 else 1) if ready_after=='recover' else ready_after
        def done(self):
            return self.polls>=self.ready_after
        def exception(self):return None
        def result(self,timeout=None):
            if timeout is not None:
                self.polls+=1
                if self.polls<self.ready_after:
                    raise TimeoutError()
            return self.value
    class DelayedPool:
        def __init__(self,**kwargs):pass
        def submit(self,fn,*args):return DelayedFuture(fn,*args)
        def shutdown(self,**kwargs):pass
    monkeypatch.setattr(benchmark,'PointForcePlanner',MeasuredPlanner)
    monkeypatch.setattr(benchmark,'ThreadPoolExecutor',DelayedPool)
    monkeypatch.setattr(benchmark.time,'sleep',lambda _:None)
    args=build_parser().parse_args(['--async-planner','--steps','7','--async-max-command-age','.2'])
    original=_run_rollout(args,'scm_corrected','foam_brick')
    published=[call for call in original.planning_rounds if call['published']]
    publication_steps=[0,2,3,4,5,6] if ready_after!=1 else list(range(7))
    assert [r['published_step'] for r in published]==publication_steps
    assert [r['observation_step'] for r in published]==([0,1,2,3,4,5] if ready_after==2 else publication_steps)
    assert all(r['command_valid_for_step'] for r in original)
    assert np.linalg.norm(original[1]['applied_force_body'])>0
    assert original[1]['command_hold_duration_s']==pytest.approx(.04 if ready_after!=1 else .02)
    if ready_after==1:
        assert all(r['prediction_matches_applied_command'] for r in original)
    if ready_after=='recover':
        superseded=[r for r in original.planning_rounds if r.get('discard_reason')=='superseded_before_actuation']
        assert len(superseded)==1 and superseded[0]['observation_step']==1
        assert all(r['prediction_matches_applied_command'] for r in original[2:])
    cfg=copy.deepcopy(args);cfg.replay_schedule=copy.deepcopy(original.planning_rounds)
    replayed=_run_rollout(cfg,'scm_corrected','foam_brick')
    assert compare_trials(original,replayed)['passed']
    observation=published[2]['observation_step']
    np.testing.assert_allclose(published[2]['planning_state_pose'],original[observation-1]['object_pose'])


def test_fast_goal_crossing_does_not_enter_unforced_verification(monkeypatch):
    import planning.point_force_benchmark as benchmark
    original=benchmark.scene
    def moving_goal(*args,**kwargs):
        model,data,target,xml=original(*args,**kwargs)
        data.qpos[:]=target
        data.qvel[:]=[.2,0,0,0,0,2.]
        mujoco.mj_forward(model,data)
        return model,data,target,xml
    monkeypatch.setattr(benchmark,'scene',moving_goal)
    args=build_parser().parse_args(['--model','qp','--sample-num','8',
        '--population-size','4','--population-iterations','1','--steps','1'])
    rows=_run_rollout(args,'qp','foam_brick')
    assert not rows[0]['settling']
    assert rows[0]['success_hold']==0
    assert len(rows.planning_rounds)==1


def test_upright_flip_coasts_until_native_measured_motion_is_stable(monkeypatch):
    import planning.point_force_benchmark as benchmark
    from scipy.spatial.transform import Rotation
    original=benchmark.scene
    def spinning_upright(*args,**kwargs):
        model,data,target,xml=original(*args,**kwargs)
        q=Rotation.from_euler('z',90,degrees=True)*Rotation.from_quat(target[[4,5,6,3]])
        data.qpos[:]=target;data.qpos[3:]=q.as_quat()[[3,0,1,2]]
        data.qvel[:3]=0.;data.qvel[3:]=q.inv().apply([0,0,.4])
        mujoco.mj_forward(model,data)
        return model,data,target,xml
    monkeypatch.setattr(benchmark,'scene',spinning_upright)
    args=build_parser().parse_args(['--sample-num','8','--population-size','4',
        '--population-iterations','1','--steps','40'])
    rows=_run_rollout(args,'qp','mug')
    assert rows[0]['flip_settling'] and not rows[0]['settling']
    assert rows[0]['success_hold']==0
    assert np.linalg.norm(rows[0]['applied_force_body'])==0
    assert any(r['object_cost_stage']=='pose' for r in rows)
    pose_entry=next(i for i,r in enumerate(rows) if r['object_cost_stage']=='pose')
    # The transition is still preceded by the original measured-speed hold,
    # rather than being inferred from angle alone.
    assert pose_entry>=15
    for row in rows[pose_entry-14:pose_entry]:
        assert np.linalg.norm(row['velocity_body'][:3])<=.02
        assert np.linalg.norm(row['velocity_body'][3:])<=.25


def test_cpu_reference_records_lemke_numerical_fallback(monkeypatch):
    from planning.point_force_prediction import CPUSubstepPredictor
    from examples.mpc.franka.scm import lcp_backend
    def failed_tableau(*args,**kwargs):
        raise FloatingPointError('deliberate Lemke numerical failure')
    monkeypatch.setattr(lcp_backend,'solve_lcp',failed_tableau)
    args=build_parser().parse_args([])
    model,data,_,_=scene(args,'rubber_duck',2)
    pose=data.qpos.copy();pose[2]-=.001
    predictor=CPUSubstepPredictor(model,args,'lcp')
    velocity,predicted,details=predictor.predict(pose,np.zeros(6),[[0,0,.02]],[[0,0,0]])
    assert np.isfinite(np.r_[velocity.ravel(),predicted.ravel()]).all()
    assert predictor.lcp_reference_counts['active-set']==args.prediction_substeps
    assert predictor.lcp_reference_fallbacks==['deliberate Lemke numerical failure']
    assert details[0]['residual']<=args.lcp_tolerance


def test_native_cpu_prediction_rejects_silent_mujoco_reset(monkeypatch):
    from planning.point_force_prediction import CPUSubstepPredictor
    args=build_parser().parse_args([])
    model,data,_,_=scene(args,'foam_brick',0)
    actual=mujoco.mj_step
    calls=[]
    def reset(model,data):
        actual(model,data);calls.append(1)
        if len(calls)==2:
            mujoco.mj_resetData(model,data)
    monkeypatch.setattr(mujoco,'mj_step',reset)
    predictor=CPUSubstepPredictor(model,args,'qp')
    with pytest.raises(FloatingPointError,match='prediction reset'):
        predictor.predict(data.qpos.copy(),np.zeros(6),[[0,0,.02]],[[0,0,0]])
    assert len(calls)==2
