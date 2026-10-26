"""Force, frame, native-contact and replay checks for the PhysX CEM backend.

Native checks: SCSP_TEST_PHYSX_CEM=1 acc/bin/python -m pytest ...
They launch an Isaac-first worker; the test process may already import torch.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from planning.contact_force_physx_cem import PhysXContactCEM
from planning.object_pose_cost import object_cost_batch_np


def test_support_plane_is_validated_and_normalized():
    backend = PhysXContactCEM.__new__(PhysXContactCEM)
    backend.set_support_plane([1, 2, 3], [0, 0, 2])
    assert backend.support == dict(point=[1, 2, 3], normal=[0, 0, 1])
    for normal in ([0, 0, 0], [0, np.nan, 1]):
        with pytest.raises(ValueError):
            backend.set_support_plane([0, 0, 0], normal)


@pytest.fixture
def native(tmp_path):
    if os.environ.get('SCSP_TEST_PHYSX_CEM') != '1':
        pytest.skip('native GPU test requires SCSP_TEST_PHYSX_CEM=1')
    import trimesh
    mesh = tmp_path/'cube.stl'
    trimesh.creation.box(extents=[.06]*3).export(mesh)
    opt = SimpleNamespace(sample_point=np.array([[0., 0., 0.], [-.03, 0., 0.]]),
        normal=np.array([[1., 0., 0.]]*2), t1=np.array([[0., 1., 0.]]*2),
        t2=np.array([[0., 0., 1.]]*2), mu_arm_obj=.5, max_contact_force=3., h=.02,
        force_reg_coef=.01, friction_reg_coef=.02, friction_cone='pyramid')
    opt.object_cost_config = lambda: np.array([100., 1., 0., 0., 0., 1., 0., 0., 1.])
    backend = PhysXContactCEM(opt, .01, [6e-6]*3, scene=dict(mesh_path=str(mesh)),
                            population_size=8, population_iterations=2, seed=42)
    try:
        yield backend
    finally:
        backend.close()


def query(backend, target=(.01, 0., 0.), gravity=(0., 0., -.0981, 0., 0., 0.), ids=(0, 1)):
    return backend.rank(np.r_[target, 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0.],
                        gravity, ids, np.zeros(6))


def test_native_free_flight_impulse_cost_and_force_constraints(native):
    native.set_support_plane([0., 0., -2.], [0., 0., 1.])
    ids, costs, forces, poses = query(native)
    # Contact at the COM: exact semi-implicit constant-acceleration trajectory.
    acceleration = forces[0]/.01 + np.array([0., 0., -9.81])
    expected_position = acceleration*.002**2*10*11/2
    np.testing.assert_allclose(poses[0, :3], expected_position, atol=3e-7)
    expected_cost = object_cost_batch_np(poses, [.01, 0., 0., 1., 0., 0., 0.],
        native.optimizer.object_cost_config()) + .01*np.sum(forces**2, axis=1)+.02*np.sum(forces[:, 1:]**2, axis=1)
    np.testing.assert_allclose(costs, expected_cost, atol=2e-7)
    assert np.all(np.linalg.norm(forces, axis=1) <= 3.+1e-7)
    assert np.all(np.sum(abs(forces[:, 1:]), axis=1) <= .5*forces[:, 0]+1e-7)
    assert np.all(.02*np.linalg.norm(forces, axis=1)/.01 <= .3+1e-7)
    assert native.metadata['gpu_pipeline'] is True
    assert native.last_timing['physics_steps'] == 22


def test_native_support_contact_and_history_reset(native):
    native.set_support_plane([0., 0., -.03], [0., 0., 1.])
    native.reset_seed(7)
    first = query(native, target=(0., 0., 0.))
    # Resting support replaces the otherwise 2.1582 mm gravity fall.
    assert np.max(abs(first[3][:, 2])) < 1e-4
    native.set_support_plane([0., 0., -2.], [0., 0., 1.])
    query(native, target=(0., .02, 0.))
    native.set_support_plane([0., 0., -.03], [0., 0., 1.])
    native.reset_seed(7)
    replay = query(native, target=(0., 0., 0.), ids=(1, 0))
    for original, permuted in zip(first[1:], replay[1:]):
        np.testing.assert_allclose(original, permuted[::-1], atol=3e-6, rtol=1e-5)


def test_native_rotated_frame_and_ad_hoc_point(native):
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler('z', 90, degrees=True)
    native.set_support_plane([0., 0., -2.], [0., 0., 1.])
    q = rotation.as_quat()[[3, 0, 1, 2]]
    pose = np.r_[np.zeros(3), q]
    result = native.solve_point(np.r_[0., .01, 0., q], pose, np.zeros(6), np.zeros(6),
        p_arm=[0., 0., 0.], n_arm=[1., 0., 0.], t1=[0., 1., 0.], t2=[0., 0., 1.])
    expected = rotation.apply(result['lam_arm_opt'])/.01*.002**2*55
    np.testing.assert_allclose(result['x_plus_opt'][:3], expected, atol=3e-7)
    assert native.last_timing['candidates'] == 1
