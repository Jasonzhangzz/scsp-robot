"""Internal preload must respect joint authority and physical patch directions."""
from types import SimpleNamespace
import numpy as np
import pytest
import mujoco
from planning.wuji_load_support import static_preload_seed


def linear_hand(monkeypatch,weight):
    # Two opposed, independently actuated contact patches with 1 cm moment
    # arms. The left drive allows 3 N, the right 12 N of normal force.
    model=SimpleNamespace(nv=8,body_mass=np.array([weight/10,0.,0.]),
        opt=SimpleNamespace(gravity=np.array([0.,0.,-10.])),
        actuator_gainprm=np.ones((8,10)),actuator_biasprm=np.zeros((8,10)),
        actuator_forcelimited=np.ones(8,bool),
        actuator_forcerange=np.array([[-1000.,1000.]]*6+[[-.03,.03],[-.12,.12]]))
    data=SimpleNamespace(xipos=np.zeros((3,3)),qfrc_bias=np.zeros(8),qfrc_passive=np.zeros(8),actuator_length=np.zeros(8))
    hand=SimpleNamespace(model=model,data=data,nu=8,obj=0,dadr=np.arange(8),friction=.5,
        lower=np.r_[np.full(6,-100.),-1.,-1.],upper=np.r_[np.full(6,100.),1.,1.],
        object_pose=lambda:np.array([0.,0.,0.,1.,0.,0.,0.]))
    def jac(model,data,jp,jr,point,body):
        jp[:,:3]=np.eye(3)
        jp[0,6 if body==1 else 7]=.01 if body==1 else -.01
    monkeypatch.setattr('planning.wuji_load_support.mujoco.mj_jac',jac)
    normals=np.array([[1.,0,0],[-1.,0,0]])
    actual=[dict(slot=i,body=i+1,point=np.array([-.025 if i==0 else .025,0.,0.]),
        normal=n,force=.1*n) for i,n in enumerate(normals)]
    return hand,actual,normals


def test_internal_preload_balances_gravity_with_joint_and_contact_limits(monkeypatch):
    hand,actual,normals=linear_hand(monkeypatch,2.)
    result=static_preload_seed(hand,actual,normals,10.,friction_margin=1.)
    assert result['feasible'] and result['object_wrench_residual']<1e-10
    np.testing.assert_allclose(result['patch_normal_forces'],[2.,2.],atol=1e-9)
    np.testing.assert_allclose(result['required_joint_torque'][6:],[.02,.02],atol=1e-9)
    margin=static_preload_seed(hand,actual,normals,10.)
    assert margin['feasible']
    np.testing.assert_allclose(margin['patch_normal_forces'],[2.5,2.5],atol=1e-9)
    limited=static_preload_seed(hand,actual,normals,1.)
    assert limited['feasible'] and not limited['full_load_feasible'] and 0<limited['load_fraction']<1
    hand.model.body_mass[0]=.514
    limited=static_preload_seed(hand,actual,normals,10.)
    assert limited['feasible'] and not limited['full_load_feasible']
    assert limited['required_joint_torque'][6]<=.03+1e-8
    np.testing.assert_array_equal(hand.data.qfrc_bias,0.)


def test_physical_preload_does_not_disappear_with_zero_grouped_normal(monkeypatch):
    hand,actual,normals=linear_hand(monkeypatch,2.)
    result=static_preload_seed(hand,actual,np.zeros_like(normals),10.,friction_margin=1.)
    assert result['feasible']
    np.testing.assert_allclose(result['normal_forces'],0.)
    np.testing.assert_allclose(result['patch_normal_forces'],[2.,2.],atol=1e-9)
    assert not static_preload_seed(hand,[],normals,10.)['feasible']


def test_only_an_engaged_correct_direction_joint_stop_can_support_motor_torque(monkeypatch):
    hand,actual,normals=linear_hand(monkeypatch,5.14)
    unsupported=static_preload_seed(hand,actual,normals,10.,friction_margin=1.)
    assert not unsupported['full_load_feasible']
    hand.model.jnt_dofadr=np.array([6]);hand.model.jnt_qposadr=np.array([6])
    hand.model.jnt_range=np.array([[-1.,1.]])
    hand.data.qpos=np.zeros(8);hand.data.qpos[6]=-1.
    hand.data.efc_type=np.array([int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)])
    hand.data.efc_id=np.array([0])
    supported=static_preload_seed(hand,actual,normals,10.,friction_margin=1.)
    assert supported['full_load_feasible'] and supported['object_wrench_residual']<1e-9
    assert supported['required_joint_torque'][6]<=.03+1e-8
    assert supported['joint_limit_reaction_torque'][6]>0
    assert supported['supporting_joint_limits']==[0]
    # The upper stop cannot supply a positive torque, and proximity without
    # an active MuJoCo limit row must never count as mechanical support.
    hand.data.qpos[6]=1.
    assert not static_preload_seed(hand,actual,normals,10.,friction_margin=1.)['full_load_feasible']
    hand.data.qpos[6]=-1.;hand.data.efc_type=np.empty(0,int)
    assert not static_preload_seed(hand,actual,normals,10.,friction_margin=1.)['full_load_feasible']


def test_multi_region_feedback_controls_gaps_and_preserves_tangential_dofs():
    from planning.wuji_load_support import pressure_command_map
    normals=np.array([[1.,0,0],[0,1.,0]])
    jac=np.zeros((2,3,9));jac[0,:,6:]=np.eye(3)*.03
    jac[1,:,6:]=np.eye(3)*.03
    jac[:,:,:3]=np.eye(3)
    projection,bias=pressure_command_map(jac,normals,np.array([.5,3.]),np.array([2.,1.]),grasp_established=True)
    normal_jac=np.einsum('ij,ijk->ik',normals,jac)
    np.testing.assert_allclose(normal_jac@bias,.0003*np.array([1.5,-2.]),rtol=.002)
    # Shared rigid link: increase one face's pressure and relieve the other.
    assert bias[6]>0 and bias[7]<0
    tangent=np.zeros(9);tangent[8]=.01
    np.testing.assert_allclose(projection@tangent,tangent,atol=1e-12)
    np.testing.assert_allclose(projection[:6,:6],np.eye(6))
    # A load allocator can be infeasible while an opposing contact is still
    # being acquired. MPC must retain normal authority instead of being stuck
    # forever at the fallback pressure reference.
    compression=np.zeros(9);compression[6:8]=[.02,-.01]
    np.testing.assert_allclose(normal_jac@projection@compression,normal_jac@compression,atol=1e-12)
    wrist=np.zeros(9);wrist[:2]=[.001,-.002]
    np.testing.assert_allclose(projection@wrist,wrist,atol=1e-12)


def test_pressure_feedback_accounts_for_soft_wuji_position_servos():
    from planning.wuji_load_support import pressure_command_map
    jac=np.zeros((1,3,7));jac[0,0,6]=.05
    normals=np.array([[1.,0,0]])
    gain=np.ones(7);soft=gain.copy();soft[6]=.25
    _,firm_bias=pressure_command_map(jac,normals,np.array([1.]),np.array([1.1]),
        grasp_established=True,stiffness=gain)
    _,soft_bias=pressure_command_map(jac,normals,np.array([1.]),np.array([1.1]),
        grasp_established=True,stiffness=soft)
    assert soft_bias[6]==pytest.approx(4*firm_bias[6])
    np.testing.assert_array_equal(gain,np.ones(7))
    np.testing.assert_array_equal(soft,np.r_[np.ones(6),.25])
