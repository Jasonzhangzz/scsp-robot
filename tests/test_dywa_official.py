import csv
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import sys
from types import ModuleType

import pytest

from planning.dywa.audit import load_student, policy_digest, verify_upstream
from planning.dywa.dispatch import dispatch
from planning.dywa.inputs import install_task_inputs, sphere_robot
from planning.dywa.run import save_csv
from planning.dywa.runtime import _hydra_overrides

ROOT = Path(__file__).resolve().parents[1]


def test_official_source_hashes():
    audit = verify_upstream()
    assert audit['verified_files'] == 313
    assert audit['unmodified_files'] == 312
    assert list(audit['authorized_patches']) == ['dywa/src/models/modules.py']


def test_authorized_history_buffer_shifts_and_detaches_inputs():
    torch = pytest.importorskip('torch')
    source = ROOT / 'planning/dywa/upstream/dywa/src/models/modules.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HistoryEncoder')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'put_history')
    namespace = {'th': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    state = SimpleNamespace(num_tokens=4, history=torch.zeros(2, 3, 4, 3))
    namespace['put_history'](state, torch.ones(2, 4, 3, requires_grad=True))
    result = namespace['put_history'](state, torch.full((2, 4, 3), 2.))
    assert result is state.history
    assert torch.count_nonzero(result[:, 0]) == 0
    assert torch.all(result[:, 1] == 1)
    assert torch.all(result[:, 2] == 2)
    assert not result.requires_grad


def test_provenance_rejects_any_extra_change_beyond_history_fix(tmp_path):
    upstream = ROOT / 'planning/dywa/upstream'
    manifest = json.loads((upstream / 'PROVENANCE.json').read_text())
    name = 'dywa/src/models/modules.py'
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_bytes((upstream / name).read_bytes())
    manifest['files'] = {name: manifest['files'][name]}
    (tmp_path / 'PROVENANCE.json').write_text(json.dumps(manifest))
    (tmp_path / 'PATCHES.json').write_bytes((upstream / 'PATCHES.json').read_bytes())
    assert verify_upstream(tmp_path)['verified_files'] == 1
    path.write_bytes(path.read_bytes() + b'\n# unauthorized change\n')
    with pytest.raises(RuntimeError, match='beyond the authorized patch'):
        verify_upstream(tmp_path)


def test_dispatch_precedes_other_planners_and_rejects_mixing(monkeypatch):
    import os
    calls = []
    monkeypatch.setattr(os, 'execv', lambda *args: calls.append(args))
    dispatch(ROOT, ['--dywa', '--headless'])
    assert calls[0][1][-2:] == ['--dywa', '--headless']
    assert calls[0][1][-3].endswith('planning/dywa/run.py')
    with pytest.raises(ValueError, match='incompatible'):
        dispatch(ROOT, ['--dywa', '--contactsdf'])


def test_sphere_preserves_frames_and_arm_and_aligns_with_tool(tmp_path):
    source = ROOT / 'thirdparty/DyWA/dywa/src/data/assets/franka_description/robots/franka_panda.urdf'
    if not source.exists():
        pytest.skip('External upstream robot assets not installed')
    old = ET.parse(source).getroot()
    new = ET.parse(sphere_robot(source, tmp_path / 'robot.urdf')).getroot()
    assert [j.attrib for j in old.findall('joint')] == [j.attrib for j in new.findall('joint')]
    assert [l.get('name') for l in old.findall('link')] == [l.get('name') for l in new.findall('link')]
    for joint in old.findall('joint'):
        current = new.find("joint[@name='%s']" % joint.get('name'))
        assert ET.tostring(joint) == ET.tostring(current)
    finger = new.find("link[@name='panda_leftfinger']/collision")
    assert finger.find('geometry/sphere').get('radius') == '0.01'
    tip = float(old.find("joint[@name='panda_hand_tool_joint']/origin").get('xyz').split()[2])
    base = float(old.find("joint[@name='panda_finger_joint1']/origin").get('xyz').split()[2])
    assert base + float(finger.find('origin').get('xyz').split()[2]) == pytest.approx(tip)


def test_policy_guard_detects_parameter_changes_but_allows_history_updates():
    torch = pytest.importorskip('torch')
    student = torch.nn.Module()
    student.net = torch.nn.Linear(2, 3)
    student.memory = torch.nn.Module()
    student.memory.register_buffer('history', torch.zeros(1, 5, 3))
    before = policy_digest(student)
    student.memory.history.fill_(2)
    assert policy_digest(student) == before
    with torch.no_grad():
        student.net.weight[0, 0] += 0.01
    assert policy_digest(student) != before


def test_error_csv_does_not_fabricate_failure_or_trajectory(tmp_path):
    path = tmp_path / 'results.csv'
    save_csv(path, [dict(object='mug', trial=0, status='error', error='missing runtime')])
    row = next(csv.DictReader(path.open()))
    assert row['status'] == 'error'
    assert row['success'] == row['steps'] == ''


def test_strict_loading_only_resets_batch_dependent_history(tmp_path):
    torch = pytest.importorskip('torch')
    student = torch.nn.Module()
    student.net = torch.nn.Linear(2, 3)
    student.history_tokenizer = torch.nn.Module()
    student.history_tokenizer.register_buffer('history', torch.zeros(1, 5, 3))
    state = dict(student.state_dict())
    state['net.weight'] = torch.ones_like(state['net.weight'])
    state['history_tokenizer.history'] = torch.ones(1024, 5, 3)
    path = tmp_path / 'last.ckpt'
    torch.save({'self': state}, path)
    load_student(student, path)
    assert torch.equal(student.net.weight, state['net.weight'])
    assert student.history_tokenizer.history.shape == (1, 5, 3)
    assert student.history_tokenizer.history.count_nonzero() == 0
    del state['net.bias']
    torch.save({'self': state}, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_student(student, path)


def test_native_hydra_does_not_change_controller_or_action_distribution():
    spec = dict(init_obj_qpos=[0, 0, .44, 1, 0, 0, 0], target_p=[0, 0, .44],
                target_q=[1, 0, 0, 0], device='cuda:0', headless=True, output_dir='/tmp/test',
                teacher_ckpt='teacher', env_ckpt='env', student_ckpt='student', max_steps=1000,
                seed=100, mesh_path='mesh', mass=.01, table_pos=[0, 0, .2],
                table_dims=[.4, .5, .4], table_friction=.5, object_friction=.5, icp_ckpt='icp')
    overrides = _hydra_overrides(spec)
    assert '++dagger_train_env.deterministic_action=false' in overrides
    for flag in overrides:
        assert not flag.startswith(('++env.dt=', '++env.action_period=', '++env.pos_iter=',
                                    '++env.franka.', '++env.single_object_scene.mesh.mass=',
                                    '++env.task.goal_', '++env.task.timeout='))


def test_task_hooks_keep_original_reset_and_commit_wxyz_inputs(monkeypatch):
    torch = pytest.importorskip('torch')
    calls = []

    class Scene:
        cur_ids = torch.tensor([1])

        def reset(self, gym, sim, env, env_ids=None):
            calls.append('scene')
            env.tensors['root'].fill_(5)
            return self.cur_ids

    class Task:
        goal = torch.zeros(1, 7)

        def reset(self, env, indices):
            calls.append('task')

    class Mesh:
        _MeshObjectSet__write_urdf = lambda self: None

    for name, attr, cls in (
            ('env.scene.tabletop_with_object_scene', 'TableTopWithObjectScene', Scene),
            ('env.task.push_task', 'PushTask', Task),
            ('env.scene.mesh_object_set', 'MeshObjectSet', Mesh)):
        module = ModuleType(name)
        setattr(module, attr, cls)
        monkeypatch.setitem(sys.modules, name, module)
    spec = dict(init_obj_qpos=[.01, .02, .44, 1, 0, 0, 0],
                target_p=[.03, .04, .42], target_q=[0, 1, 0, 0])
    install_task_inputs(spec)
    env = SimpleNamespace(tensors={'root': torch.zeros(2, 13)})
    scene, task = Scene(), Task()
    ids = scene.reset(None, None, env)
    task.reset(env, None)
    assert calls == ['scene', 'task']
    assert torch.equal(ids, scene.cur_ids)
    assert torch.allclose(env.tensors['root'][1, :7], torch.tensor([.01, .02, .44, 0, 0, 0, 1]))
    assert torch.count_nonzero(env.tensors['root'][1, 7:]) == 0
    assert torch.allclose(task.goal[0], torch.tensor([.03, .04, .42, 1, 0, 0, 0]))
