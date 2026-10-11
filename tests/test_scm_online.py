"""Calibration must improve held-out prediction and reach every contact backend."""
from types import SimpleNamespace
import casadi as cs
import numpy as np
import pytest
import torch
from models.explicit_model import ExplicitModel
from models.scm_online import ExecutionTransitionBuffer, OnlineSCMIdentifier
from models.scm_parameters import (scaled_inverse_cs, scaled_inverse_numpy,
    friction_jacobian_numpy, SCM_LOG_LOWER, SCM_LOG_UPPER)
from models.soft_scm_contact import soft_scm_coefficients
from planning.mlqp_point import LambdaContactControlOptimizer


def fixture():
    param=SimpleNamespace(n_qpos_=10,n_qvel_=9,n_robot_qpos_=3,n_cmd_=3,max_ncon_=2,
        Q=np.diag([50.]*3+[.05]*3+[100.]*3), obj_mass_=.01,
        gravity_=np.array([0,0,-9.8,0,0,0]), robot_stiff_=np.eye(3)*100,
        h_=.02,contact_model='soft-scm',contact_dynamics='quasi-dynamic',online_scm_=True,
        soft_scm_response_time=.05,soft_scm_damping_ratio=1.,soft_scm_regularization=1e-8,
        mu_object_=.5)
    model=ExplicitModel(param)
    q=np.array([0,0,.1,1,0,0,0,.04,0,.1],dtype=float)
    rng=np.random.RandomState(7)
    samples=[]
    truth=np.log([1.6,.8,1.3,.7,1.5,.8])
    for sid in range(160):
        n=np.array([-1.,rng.uniform(-.25,.25),rng.uniform(-.2,.2)]);n/=np.linalg.norm(n)
        t=np.cross(n,[0,0,1]);t/=np.linalg.norm(t);t2=np.cross(n,t)
        point=np.array([.025,rng.uniform(-.02,.02),rng.uniform(-.02,.02)])
        skew=np.array([[0,-point[2],point[1]],[point[2],0,-point[0]],[-point[1],point[0],0]])
        cart=np.column_stack([np.eye(3),-skew,-np.eye(3)])
        cone=np.stack([n+.5*t,n-.5*t,n+.5*t2,n-.5*t2])
        j=np.zeros((8,9));j[:4]=cone@cart
        phi=np.r_[np.full(4,-.0001),np.ones(4)]
        u=np.array([rng.uniform(-.003,-.001),rng.uniform(-.001,.001),rng.uniform(-.001,.001)])
        target=np.asarray(model.step_learn_fn(q,u,phi,j,truth)).ravel()
        samples.append(dict(q=q.copy(),u=u,phi=phi,jac=j,q_plus=target,jac_plus=j.copy(),
                            phi_plus=phi.copy(),contact=True,contact_plus=True,dt=.02,id=sid,trial=0))
    return model,samples


def test_held_out_prediction_improves_without_unbounded_parameters():
    model,samples=fixture()
    learner=OnlineSCMIdentifier(model,update_every=1,gradient_steps=12,learning_rate=.03)
    for sample in samples[:100]: assert learner.add(sample)
    before,_=learner._evaluate(samples[100:],np.zeros(6))
    for _ in range(25):
        learner.since_update=1
        learner.update()
    after,_=learner._evaluate(samples[100:],learner.theta)
    assert after < before*.3
    assert np.all(learner.theta>=SCM_LOG_LOWER) and np.all(learner.theta<=SCM_LOG_UPPER)
    assert learner.diagnostics()["updates"]>0
    assert learner.diagnostics()["samples"]==100
    assert model.param_.obj_mass_==.01
    np.testing.assert_array_equal(model.param_.robot_stiff_,np.eye(3)*100)


def test_derivative_matches_finite_difference_and_quaternion_sign():
    model,samples=fixture();s=samples[0]
    learner=OnlineSCMIdentifier(model)
    theta=np.array([.1,.2,-.1,.05,.1,-.1]);eps=1e-5
    args=[s[k] for k in ('q','u','phi','jac','q_plus')]
    val,grad,_=learner.loss_fn(*args,theta)
    finite=np.array([(float(learner.loss_fn(*args,theta+eps*e)[0])-float(learner.loss_fn(*args,theta-eps*e)[0]))/(2*eps) for e in np.eye(6)])
    np.testing.assert_allclose(np.asarray(grad).ravel(),finite,rtol=1e-5,atol=1e-7)
    args[-1]=args[-1].copy();args[-1][3:7]*=-1
    assert float(learner.loss_fn(*args,theta)[0])==pytest.approx(float(val))


def test_telemetry_records_executed_target_not_queued_action():
    _,samples=fixture();s=samples[0]
    fields=dict(phi_vec=s['phi'],jac_mat=s['jac'],if_contact=True)
    recorder=ExecutionTransitionBuffer(3,.002,capacity=2)
    for i in range(3):
        recorder.begin(s['q'],fields)
        for k in range(10): recorder.record_target(s['q'][7:]+(k+1)*.0001)
        recorder.finish(s['q_plus'],fields)
    assert [s['id'] for s in recorder.snapshot()]==[1,2]
    r=recorder.snapshot()[-1]
    np.testing.assert_allclose(r['u'],np.full(3,.001))
    assert r['dt']==pytest.approx(.02) and r['trial']==3


def test_gating_deduplicates_and_rejects_stale_geometry():
    model,samples=fixture();learner=OnlineSCMIdentifier(model)
    assert learner.add(samples[0])
    assert not learner.add(samples[0])
    s=dict(samples[1],dt=.04);assert not learner.add(s)
    s=dict(samples[2],contact_plus=False);assert not learner.add(s)
    s=dict(samples[3],jac_plus=-samples[3]['jac']);assert not learner.add(s)
    assert learner.rejections=={'duration':1,'contact_entry_exit':1,'normal_jump':1}
    assert len(learner.buffer)==1


def test_learned_parameters_reach_symbolic_numpy_and_torch_cso():
    model,samples=fixture();s=samples[0]
    opt=LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.contact_model='soft-scm';opt.h=.05;opt.max_contacts=2
    opt.base_Q_inv=np.linalg.inv(model.param_.Q[:6,:6]);opt.Q_inv=opt.base_Q_inv.copy()
    opt.soft_scm_response_time=.05;opt.soft_scm_damping_ratio=1.;opt.soft_scm_regularization=1e-8
    theta=np.log([1.4,.7,1.3,1.1,1.5,.6]);opt.set_scm_parameters(theta)
    j=s['jac'][:,:6];phi=s['phi'];weights=opt.compute_env_diag_inverse(j)
    smooth=np.array([-.02,.01,-.01,.1,-.3,.1])
    expected=opt._contact_velocity_numpy(smooth,j,weights,phi)
    th=cs.SX.sym('theta',6);v=cs.SX.sym('v',6);qi=scaled_inverse_cs(opt.base_Q_inv,th)
    out=opt._contact_velocity_cs(v,cs.DM(j),cs.DM(weights),cs.DM(phi),qi,th)
    fn=cs.Function('adaptive_cso',[v,th],[out])
    np.testing.assert_allclose(np.asarray(fn(smooth,theta)).ravel(),expected,atol=1e-12)
    t=lambda a:torch.tensor(a,dtype=torch.float64)
    np.testing.assert_allclose(opt._contact_velocity_torch(t(smooth),t(j),t(weights),t(phi),t(opt.Q_inv)).numpy(),expected,atol=1e-12)


def test_mpc_runtime_parameter_interface_and_zero_theta_equivalence():
    model,samples=fixture();s=samples[0]
    q,u,phi,j=(s[k] for k in ('q','u','phi','jac'))
    old=np.asarray(model.step_once_fn(q,u,phi,j,7))
    np.testing.assert_allclose(model.prediction_fn(q,u,phi,j,model.runtime_parameters(7)),old,atol=1e-14)
    identity=id(model.prediction_fn)
    model.set_scm_parameters(np.log([1.5,1.2,.8,1.1,1.5,.8]))
    learned=np.asarray(model.prediction_fn(q,u,phi,j,model.runtime_parameters(7)))
    assert id(model.prediction_fn)==identity
    assert np.linalg.norm(learned-old)>1e-5
    assert model.model_parameter_dim==7


def test_checkpoint_round_trip_rejects_configuration_mismatch(tmp_path):
    model,samples=fixture();learner=OnlineSCMIdentifier(model)
    learner.add(samples[0]);learner.theta=np.log([1.4,.7,1.3,1.1,1.5,.6])
    path=tmp_path/'transitions.npz';learner.save(path)
    clone=OnlineSCMIdentifier(model);clone.load(path)
    np.testing.assert_array_equal(clone.theta,learner.theta)
    assert (tmp_path/'parameters.json').exists()
    model.param_.h_=.05
    with pytest.raises(ValueError,match='incompatible h'): clone.load(path)


def test_angular_coordinate_conversion_preserves_physical_contact_speed():
    from examples.mpc.franka.ik2.contact_frames import contact_jacobian_angular_body_frame
    from scipy.spatial.transform import Rotation
    rotation=Rotation.from_euler('xyz',[.7,-.4,1.2]).as_matrix()
    rng=np.random.RandomState(40);jac=rng.randn(8,9)
    converted=contact_jacobian_angular_body_frame(jac,rotation)
    v=rng.randn(9);world=v.copy();world[3:6]=rotation@v[3:6]
    np.testing.assert_allclose(converted@v,jac@world,atol=1e-12)
    np.testing.assert_array_equal(converted[:,:3],jac[:,:3])
    np.testing.assert_array_equal(converted[:,6:],jac[:,6:])


def test_learning_friction_updates_cso_wrench_cone_and_normal_cap():
    opt=LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.contact_model='soft-scm';opt.h=.05;opt.max_contacts=2
    opt.base_Q_inv=opt.Q_inv=np.eye(6)
    opt.soft_scm_response_time=.05;opt.soft_scm_damping_ratio=1.;opt.soft_scm_regularization=1e-8
    opt.base_mu_arm_obj=.5;opt.max_contact_force=10.
    theta=np.zeros(6);theta[5]=np.log(1.4)
    opt.set_scm_parameters(theta)
    assert opt.mu_arm_obj==pytest.approx(.7)
    assert opt.max_normal_force==pytest.approx(10/np.sqrt(1+2*.7**2))


def test_benchmark_parses_summary_interleaved_with_native_stderr(tmp_path):
    from examples.mpc.franka.scm.benchmark_online_scm import analyze
    summary=dict(trial=0,mode='mpc_planner',success=1,steps=21,final_pos_err=.01,final_quat_err=.002)
    (tmp_path/'fixed.log').write_text('trial_summary: native warning\n'+repr(summary)+'\n')
    (tmp_path/'online.log').write_text('trial_summary: '+repr(summary)+'\n')
    report=analyze(tmp_path)
    assert report['aggregate']['fixed']['successes']==1
    assert report['aggregate']['online']['trials']==1
    assert [r['mode'] for r in report['trials']]==['fixed','online']


def test_checkpoint_rejects_changed_contact_grouping(tmp_path):
    model, samples = fixture()
    learner = OnlineSCMIdentifier(model)
    learner.add(samples[0])
    path = tmp_path/'transitions.npz'
    learner.save(path)
    model.grouped_contact_response = True
    with pytest.raises(ValueError, match='contact response'):
        learner.load(path)


def test_driven_contact_prediction_matches_backends_with_learned_parameters():
    opt=LambdaContactControlOptimizer.__new__(LambdaContactControlOptimizer)
    opt.contact_model='soft-scm'; opt.h=.02; opt.max_contacts=1
    opt.base_Q_inv=opt.Q_inv=np.diag([.02]*3+[20.]*3)
    opt.base_mu_arm_obj=.5; opt.max_contact_force=.25
    opt.robot_stiffness=600.; opt.gravity_velocity_scale=1/opt.h
    opt.soft_scm_response_time=.05; opt.soft_scm_damping_ratio=1.; opt.soft_scm_regularization=1e-8
    theta=np.log([1.4,.7,1.3,1.1,1.5,.6]); opt.set_scm_parameters(theta)
    n,t1,t2=np.eye(3)[2],np.eye(3)[0],np.eye(3)[1]
    j=np.c_[np.stack([n+.5*t1,n+.5*t2,n-.5*t1,n-.5*t2]),np.zeros((4,3))]
    velocity=np.zeros(6); tau=np.array([0.,0.,-.098,0.,0.,0.])
    force=np.array([-.1,.03,.01]); point=np.array([.03,0.,.02]); frame=np.diag([-1.,1.,-1.]); phi=np.zeros(4)
    expected=opt._arm_velocity_numpy(velocity,tau,force,point,frame,j,phi)
    th=cs.SX.sym('theta',6)
    result=opt._arm_velocity_cs(cs.DM(velocity),cs.DM(tau),cs.DM(force),cs.DM(point),cs.DM(frame),cs.DM(j),cs.DM(phi),scaled_inverse_cs(opt.base_Q_inv,th),th)
    fn=cs.Function('learned_driven_contact',[th],[result])
    np.testing.assert_allclose(np.asarray(fn(theta)).ravel(),expected,atol=1e-12)
    tensor=lambda value:torch.tensor(value,dtype=torch.float64)
    actual=opt._arm_velocity_torch(*map(tensor,[velocity,tau,force,point,frame,j,phi,opt.Q_inv]))
    np.testing.assert_allclose(actual.numpy(),expected,atol=1e-12)
