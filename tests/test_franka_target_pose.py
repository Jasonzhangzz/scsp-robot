"""Standing targets must describe the actual collision/visual mesh frame."""
import argparse

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.mpc.franka.ik2.params import (
    ExplicitMPCParams, _supported_mesh_height, _xml_obj_geom_quat_wxyz,
)
from examples.mpc.franka.ik2.test_mpc_isaac import _add_rollout_policy_args
from planning.object_pose_cost import ObjectCostStages, rotate_axis_np
from utils import rotations
import examples.mpc.franka.ik2.params as param_module

OBJECTS = ['elephant', 'stanford_bunny2', 'piggy_bank', 'mug',
           'teapot', 'rubber_duck', 'foam_brick']
X_UP = set(OBJECTS[2:])


@pytest.fixture(autouse=True)
def skip_contact_solver_construction(monkeypatch):
    # These checks exercise real asset transforms and target generation.
    # Solver objectives are covered by test_two_stage_object_cost.py.
    monkeypatch.setattr(param_module, 'build_lambda_optimizer', lambda *_: None)


def make_params(obj, hull=True, target='ground-rotation', seed=0, tilt=True):
    args = _add_rollout_policy_args(argparse.ArgumentParser()).parse_args(['--obj', obj])
    args.collision_hull = hull
    args.target_type = target
    args.random_init_tilt = tilt
    return args, ExplicitMPCParams(args, rand_seed=seed)


@pytest.mark.parametrize('obj', OBJECTS)
@pytest.mark.parametrize('target', ['ground-rotation', 'rotation'])
@pytest.mark.parametrize('seed', [0, 1, 4])
def test_standing_target_and_start_agree_between_raw_and_baked_mesh(obj, target, seed):
    args, baked = make_params(obj, target=target, seed=seed)
    _, raw = make_params(obj, hull=False, target=target, seed=seed)
    np.testing.assert_allclose(baked.object_up_axis_, [1, 0, 0] if obj in X_UP else [0, 0, 1], atol=1e-10)
    np.testing.assert_allclose(raw.object_up_axis_, [0, 0, 1], atol=1e-10)
    geom = _xml_obj_geom_quat_wxyz(baked.model_path_)
    for actor_q, source_q in [(baked.target_q_, raw.target_q_),
                              (baked.init_obj_qpos_[3:7], raw.init_obj_qpos_[3:7])]:
        mapped = rotations.quaternion_multiply(actor_q, geom)
        assert abs(np.dot(mapped, source_q)) == pytest.approx(1., abs=1e-10)
    np.testing.assert_allclose(baked.target_p_, raw.target_p_, atol=1e-8)
    np.testing.assert_allclose(baked.init_obj_qpos_[:3], raw.init_obj_qpos_[:3], atol=1e-8)
    for param in (raw, baked):
        heights = Rotation.from_quat(param.target_q_[[1,2,3,0]]).apply(param.object_vertices_)[:, 2]
        assert param.target_p_[2] + heights.min() == pytest.approx(param.table_height, abs=1e-10)
        stages = ObjectCostStages(args, param.object_up_axis_)
        state = dict(sim_time=0., policy_q=np.r_[param.target_p_, param.target_q_],
                     object_linear_velocity=np.zeros(3), object_angular_velocity=np.zeros(3))
        assert not stages.update(state)
        state['sim_time'] = .3
        assert stages.update(state) and stages.stage == 'pose'


@pytest.mark.parametrize('obj', OBJECTS)
def test_untilted_start_stands_in_the_target_upright_family(obj):
    _, p = make_params(obj, tilt=False)
    np.testing.assert_allclose(rotate_axis_np(p.init_obj_qpos_[3:7], p.object_up_axis_), [0,0,1], atol=1e-10)


def test_supported_height_uses_vertices_instead_of_empty_aabb_corners():
    vertices = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.]])
    q = rotations.rpy_to_quaternion([0., .7, -.4])
    expected = .35 - min(rotate_axis_np(q, v)[2] for v in vertices)
    assert _supported_mesh_height(q, vertices, .35) == pytest.approx(expected)
