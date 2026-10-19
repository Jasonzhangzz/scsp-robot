"""Physical mug/tool collision masks must use rigid-body shape ranges."""
from types import SimpleNamespace
from pathlib import Path
import ast
import hashlib
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest


@pytest.mark.parametrize('mug', [False, True])
@pytest.mark.parametrize('attribute_ranges', [False, True])
def test_tool_collision_masks_cover_every_shaft_and_sphere_shape(mug, attribute_ranges):
    from examples.mpc.franka.runtime.mpc import IsaacFrankaOSCSimulator

    # Multiple shapes per link and nonmatching shape/body orders make a guessed
    # "last shape is the fingertip" fallback physically wrong.
    names = ['panda_link0', 'attachment', 'panda_link7', 'fingertip', 'empty']
    spans = [(0, 2), (4, 2), (2, 2), (6, 2), (8, 0)]
    if attribute_ranges:
        # Matches Isaac's IndexRange: attributes exist; indexing raises TypeError.
        spans = [SimpleNamespace(start=start, count=count) for start, count in spans]
    shapes = {name: [SimpleNamespace(filter=99) for _ in range(count)]
              for name, count in [('robot', 8), ('object', 37), ('table', 1)]}
    gym = SimpleNamespace(
        get_actor_rigid_body_names=lambda *_: names,
        get_actor_rigid_body_shape_indices=lambda *_: spans,
        get_actor_rigid_shape_properties=lambda _, actor: shapes[actor],
        set_actor_rigid_shape_properties=lambda *_: None)
    tool = ('panda_link7', 'attachment', 'fingertip') if mug else ('fingertip',)
    env = SimpleNamespace(gym=gym, env=None, franka_actor='robot', obj_actor='object',
        table_actor='table', param_=SimpleNamespace(tool_object_collision_bodies_=tool,
                                                  fingertip_friction_=2.5))
    IsaacFrankaOSCSimulator._apply_fingertip_only_object_collision(env)
    enabled = {2, 3, 4, 5, 6, 7} if mug else {6, 7}
    for i, shape in enumerate(shapes['robot']):
        assert shape.filter == (0 if i in enabled else 1)
        if i in enabled:
            assert shape.friction == 2.5
    assert all(shape.filter == 1 for shape in shapes['object'])
    assert shapes['table'][0].filter == 0


def test_missing_fingertip_collision_shape_is_reported():
    from examples.mpc.franka.runtime.mpc import IsaacFrankaOSCSimulator

    gym = SimpleNamespace(get_actor_rigid_body_names=lambda *_: ['fingertip'],
        get_actor_rigid_body_shape_indices=lambda *_: [SimpleNamespace(start=0, count=0)],
        get_actor_rigid_shape_properties=lambda *_: [])
    env = SimpleNamespace(gym=gym, env=None, franka_actor=None, param_=SimpleNamespace())
    with pytest.raises(RuntimeError, match='Tool collision shapes missing: fingertip'):
        IsaacFrankaOSCSimulator._apply_fingertip_only_object_collision(env)


@pytest.mark.parametrize('ablation', [False, True])
def test_mjwarp_scene_has_native_shaft_and_sphere_contacts_with_mug(ablation):
    """Compile the real scene XML on CPU without requiring the MJWarp GPU package."""
    import argparse
    import mujoco
    from examples.mpc.franka.runtime.params import ExplicitMPCParams
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args

    root = Path(__file__).resolve().parents[1]
    directory = 'experiments/mpc_ablation' if ablation else 'runtime'
    path = root/'examples/mpc/franka'/directory/'mjwarp.py'
    # These scene-building functions have no Warp dependency. Execute their
    # source directly so the test still exercises the deployed scene factory.
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('isaac_robot_dynamics', 'scene_xml')]
    scope = dict(ROOT=root, ET=ET, np=np, os=os, json=json, hashlib=hashlib, DYWA_SIM_DT=.002)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
    args = _add_rollout_policy_args(argparse.ArgumentParser()).parse_args(
        ['--obj', 'mug', '--rotation', '--physics-backend', 'mjwarp'])
    param = ExplicitMPCParams(args, build_optimizer=False)
    model = mujoco.MjModel.from_xml_string(scope['scene_xml'](param, args))
    data = mujoco.MjData(model)
    mug_body = model.body('obj').id
    mug_geoms = set(np.flatnonzero(model.geom_bodyid == mug_body))
    qadr = model.jnt_qposadr[model.joint('object_free').id]
    for name, radius in [('attachment', .005), ('fingertip', .01)]:
        geom = model.geom(name).id
        assert model.geom_contype[geom] == 2 and model.geom_conaffinity[geom] == 1
        mujoco.mj_forward(model, data)
        center = data.geom_xpos[geom].copy()
        data.qpos[qadr:qadr+3] = center - [0., .0324 + radius - .002, 0.]
        data.qpos[qadr+3:qadr+7] = [1., 0., 0., 0.]
        mujoco.mj_forward(model, data)
        assert any(geom in (c.geom1, c.geom2) and
                   (c.geom1 in mug_geoms or c.geom2 in mug_geoms) for c in data.contact)
