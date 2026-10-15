import numpy as np
import casadi as cs
from scipy.spatial.transform import Rotation
from models.spherical_tool_contact import sphere_contact_jacobian,SphericalToolContactModel
from tests.test_bunny_contact_dynamics import rigid_model


def test_sphere_angular_contact_columns_match_surface_point_velocity():
    n=np.array([-1.,0.,0.]);t=np.array([0.,0.,1.]);b=np.cross(n,t)
    rays=np.array([n+.9*t,n+.9*b,n-.9*t,n-.9*b])
    jac=np.c_[rays,np.zeros((4,3)),-rays]
    full=np.array(sphere_contact_jacobian(cs.DM(jac),.01))
    linear=np.array([.02,.01,.03]);omega=np.array([0.,-1.,0.])
    lever=.01*n
    expected=-rays@(linear+np.cross(omega,lever))
    np.testing.assert_allclose(full@np.r_[np.zeros(6),linear,omega],expected,atol=1e-12)
    np.testing.assert_allclose(full[:,9:].mean(0),0.,atol=1e-12)
    np.testing.assert_allclose(sphere_contact_jacobian(cs.DM.zeros(4,9),.01),0.,atol=1e-12)


def test_wrist_rotation_changes_tangential_contact_response_and_lifts_the_object():
    p=rigid_model().param_;p.gravity_=np.zeros(6)
    m=SphericalToolContactModel(p)
    rays=np.array([[1.,0,.9],[1.,.9,0],[1.,0,-.9],[1.,-.9,0]])
    point=np.array([-.04,0.,0.])
    skew=np.array([[0,0,0],[0,0,.04],[0,-.04,0]])
    jac=rays@np.c_[np.eye(3),-skew,-np.eye(3)]
    x=m.observation_state(np.r_[[0,0,.4],[1,0,0,0],[-.05,0,.4]])
    ordinary=np.array(m.prediction_fn(x,[.002,0,0,0,0,0],np.zeros(4),jac,7)).ravel()
    upward=np.array(m.prediction_fn(x,[.002,0,0,0,-.02,0],np.zeros(4),jac,7)).ravel()
    assert upward[12]>ordinary[12]+1e-5
    assert np.linalg.norm(upward[23:26])>0
    assert abs(np.linalg.norm(upward[19:23])-1)<1e-12


def test_pose_model_is_equivariant_under_world_yaw():
    p=rigid_model().param_;m=SphericalToolContactModel(p)
    x=m.observation_state(np.r_[[.4,0,.4],[1,0,0,0],[.35,0,.4]],
        [.01,0,0],[0,.1,0],[.02,0,0],[0,1,0,0],[0,-.1,0])
    rays=np.array([[1.,0,.9],[1.,.9,0],[1.,0,-.9],[1.,-.9,0]])
    jac=np.c_[rays,np.zeros((4,3)),-rays];u=np.array([.002,0,.0002,0,-.02,0])
    y=np.array(m.prediction_fn(x,u,np.zeros(4),jac,7)).ravel()
    yaw=Rotation.from_euler('z',.7);r=yaw.as_matrix()
    rx=x.copy();rx[:3]=r@x[:3];rx[3:7]=yaw.as_quat()[[3,0,1,2]];rx[7:13]=np.r_[r@x[7:10],r@x[10:13]]
    rx[16:19]=r@x[16:19]
    rx[19:23]=(yaw*Rotation.from_quat(x[[20,21,22,19]])).as_quat()[[3,0,1,2]]
    rx[23:26]=r@x[23:26]
    transform=np.zeros((9,9));transform[:3,:3]=r.T;transform[3:6,3:6]=np.eye(3);transform[6:9,6:9]=r.T
    ry=np.array(m.prediction_fn(rx,np.r_[r@u[:3],r@u[3:]],np.zeros(4),jac@transform,7)).ravel()
    np.testing.assert_allclose(ry[:3],r@y[:3],atol=1e-10)
    np.testing.assert_allclose(ry[23:26],r@y[23:26],atol=1e-10)
