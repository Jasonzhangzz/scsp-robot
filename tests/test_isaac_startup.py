"""Planner startup must not initialize the viewer's native dependencies."""
import ast
import argparse
import os
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.mpc.franka.ik2 import isaac_bus


@pytest.mark.parametrize('previous', [None, 'custom'])
def test_planner_spawn_inherits_mode_before_main_import_and_restores_parent(monkeypatch, previous):
    if previous is None:
        monkeypatch.delenv('SCSP_PLANNER_ONLY', raising=False)
    else:
        monkeypatch.setenv('SCSP_PLANNER_ONLY', previous)
    inherited = []
    process = SimpleNamespace(pid=123, start=lambda: inherited.append(os.environ['SCSP_PLANNER_ONLY']))
    context = SimpleNamespace(Process=lambda **kwargs: process)
    monkeypatch.setattr(isaac_bus.mp, 'get_context', lambda *_: context)
    ready = queue.Queue()
    ready.put(dict(ok=True, pid=123))
    assert isaac_bus._start_process(lambda: None, (), ready, 1., planner_only=True) is process
    assert inherited == ['1']
    assert os.environ.get('SCSP_PLANNER_ONLY') == previous


def test_failed_spawn_restores_simulator_environment(monkeypatch):
    monkeypatch.delenv('SCSP_PLANNER_ONLY', raising=False)

    def fail():
        assert os.environ['SCSP_PLANNER_ONLY'] == '1'
        raise RuntimeError('spawn failed')

    context = SimpleNamespace(Process=lambda **kwargs: SimpleNamespace(start=fail))
    monkeypatch.setattr(isaac_bus.mp, 'get_context', lambda *_: context)
    with pytest.raises(RuntimeError, match='spawn failed'):
        isaac_bus._start_process(lambda: None, (), queue.Queue(), 1., planner_only=True)
    assert 'SCSP_PLANNER_ONLY' not in os.environ


@pytest.mark.parametrize('floating', [False, True])
def test_entry_dispatches_before_loading_numpy_or_planner(floating):
    path = Path(__file__).resolve().parents[1] / 'examples/mpc/franka/ik2/test_mpc_isaac.py'
    tree = ast.parse(path.read_text())
    numpy_index = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Import)
                       and any(alias.name == 'numpy' for alias in node.names))
    prefix = tree.body[:numpy_index]
    calls = []
    fake_os = SimpleNamespace(path=os.path, environ={'SCSP_PLANNER_ONLY': '1'},
                              execv=lambda exe, argv: calls.append((exe, argv)))
    argv = ['test_mpc_isaac.py', '--obj', 'elephant']
    if floating:
        argv.append('--floating-fingertip')
    # Run the dispatch prefix with intercepted execv. No simulator or planner
    # imports are allowed before the process is handed to its real entry point.
    declarations = [node for node in prefix if isinstance(node, (ast.FunctionDef, ast.If))]
    scope = dict(__name__='__main__', __file__=str(path), os=fake_os, argparse=argparse,
                 sys=SimpleNamespace(executable='/python', argv=argv))
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(path), 'exec'), scope)
    assert len(calls) == 1
    executable, dispatched = calls[0]
    assert executable == '/python' and dispatched[:2] == ['/python', '-u']
    assert dispatched[2].endswith('test_0902_isaac.py' if floating else 'run.py')
    assert '--floating-fingertip' not in dispatched
    if not floating:
        assert dispatched[3:] == ['--planner', 'mpc', '--obj', 'elephant']
    assert 'SCSP_PLANNER_ONLY' not in fake_os.environ
