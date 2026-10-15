import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from planning.franka_contact_motion import OSCContactCurl


@pytest.mark.parametrize('yaw',[0.,np.pi/2,np.pi,-np.pi/2])
def test_outward_preparation_and_contact_curl_rotate_with_loading_direction(yaw):
    r=Rotation.from_rotvec([0,0,yaw]).as_matrix();n=r@np.array([-1.,0,0]);f=r@np.array([1.,0,.7]);u=-.001*n
    curl=OSCContactCurl();prep=curl.update(f,n,u,touching=False,gap=.01,active=True,dt=.02)
    axis=np.cross([0.,0,1.],n)
    assert prep@axis<0
    for _ in range(25):
        rotation=curl.update(f,n,u,touching=True,gap=0.,active=True,dt=.02)
    assert rotation@axis>0
    surface=np.cross(rotation-prep,-.01*n)
    assert surface[2]>0


def test_curl_never_rewrites_xyz_and_stops_after_one_bounded_entry():
    c=OSCContactCurl();u=np.array([.001,0,0]);original=u.copy()
    outputs=[c.update([1.,0,.7],[-1,0,0],u,touching=True,gap=0.,active=True,dt=.02) for _ in range(100)]
    np.testing.assert_equal(u,original)
    assert max(np.linalg.norm(x) for x in outputs)<=np.deg2rad(30)+1e-9
    np.testing.assert_allclose(outputs[-1],0.)
    assert c.completed
    np.testing.assert_allclose(c.update([1.,0,.7],[-1,0,0],u,touching=True,gap=0.,active=False,dt=.02),0.)


def test_withdrawal_does_not_add_wrist_pressure_and_margin_gap_does_not_restart():
    c=OSCContactCurl();f=[1.,0,.7];n=[-1.,0,0]
    c.update(f,n,[.001,0,0],touching=True,gap=0.,active=True,dt=.02)
    angle=c.angle
    np.testing.assert_allclose(c.update(f,n,[-.001,0,0],touching=True,gap=0.,active=True,dt=.02),0.)
    assert c.angle==angle
    c.update(f,n,[.001,0,0],touching=False,gap=.001,active=True,dt=.02)
    assert c.angle>angle


def test_useful_placed_tilt_is_preserved_and_an_inward_wrist_prepares_without_advancing_curl():
    base=Rotation.from_euler('y',23,degrees=True).apply([0.,0.,-1.])
    c=OSCContactCurl()
    prep=c.update([1.,0,.7],[-1,0,0],[.001,0,0],touching=False,gap=.01,active=True,dt=.02,base_tool_axis=base)
    np.testing.assert_allclose(prep,0.)
    c=OSCContactCurl()
    prep=c.update([-1.,0,.7],[1,0,0],[-.001,0,0],touching=False,gap=.01,active=True,dt=.02,base_tool_axis=base)
    desired=Rotation.from_rotvec(prep).apply(base)
    assert desired[0]>0.
    rotation=c.update([-1.,0,.7],[1,0,0],[-.001,0,0],touching=True,gap=0.,active=True,dt=.02,
                      base_tool_axis=base,actual_tool_axis=base)
    np.testing.assert_allclose(rotation,prep)
    assert c.angle==0.
    c.update([-1.,0,.7],[1,0,0],[-.001,0,0],touching=True,gap=0.,active=True,dt=.02,
             base_tool_axis=base,actual_tool_axis=desired)
    assert c.angle>0.
