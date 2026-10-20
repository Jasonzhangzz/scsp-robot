"""STOCS-3D stays on its own entry path and keeps the upstream solver."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

VENDOR = Path(__file__).resolve().parents[1] / 'examples/mpc/franka/stocs3d/vendor'


def test_vendored_sources_match_the_upstream_commit():
    provenance = json.loads((VENDOR / 'PROVENANCE.json').read_text())
    assert provenance['commit'] == '1bb8739f18cc63103de76e40466433c05b860e2e'
    for name, digest in provenance['sha256'].items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest
    assert 'solver = SnoptSolver()' in (VENDOR / 'MPCC.py').read_text()
    assert 'def oracle(' in (VENDOR / 'STOCS.py').read_text()


def test_ablation_flag_leaves_for_the_offline_solver(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['ablation', '--stocs3d', '--headless'])

    def execv(executable, argv):
        assert executable == sys.executable
        assert argv[2].endswith('examples/mpc/franka/stocs3d/run.py')
        assert argv[3:] == ['--stocs3d', '--headless']
        raise SystemExit(7)

    monkeypatch.setattr(os, 'execv', execv)
    from examples.mpc.franka.experiments.test_mpc_ablation import main
    with pytest.raises(SystemExit, match='7'):
        main()


def test_isaac_entry_tracks_a_solved_trajectory(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['isaac', '--stocs-trajectory', 'output/stocs3d', '--headless'])

    def execv(executable, argv):
        assert argv[2].endswith('examples/mpc/franka/stocs3d/execute.py')
        assert '--stocs-trajectory' in argv
        raise SystemExit(8)

    monkeypatch.setattr(os, 'execv', execv)
    from examples.mpc.franka.ik2.test_mpc_isaac import main
    with pytest.raises(SystemExit, match='8'):
        main()


def test_fingertip_center_is_outside_the_inward_normal():
    from examples.mpc.franka.stocs3d.trajectory import fingertip_knots, position_at
    result = dict(total_iter=1, is_success=True, iter_1=dict(
        q=[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        x=[[0.0, 0.0, 0.4], [0.1, 0.0, 0.4]],
        dt=[0.0, 0.1]))
    params = dict(task_params=dict(
        manipulation_contact_points=[[0.05, 0.0, 0.0]],
        manipulation_contact_normals=[[-1.0, 0.0, 0.0]]))
    knots = fingertip_knots(result, params, radius=0.01)
    np.testing.assert_allclose(knots['fingertip_xyz'][0], [0.06, 0.0, 0.4])
    np.testing.assert_allclose(knots['fingertip_xyz'][1], [0.16, 0.0, 0.4])
    mid = position_at(knots['times'], knots['fingertip_xyz'], 0.05)
    np.testing.assert_allclose(mid, [0.11, 0.0, 0.4])


def test_options_reject_a_different_controller():
    from types import SimpleNamespace
    from examples.mpc.franka.stocs3d.run import validate_options
    args = SimpleNamespace(contactsdf=True, contact_model='soft-scm', physics_backend='isaac',
                           num_envs=1, floating_fingertip=False, online_scm=False)
    with pytest.raises(ValueError, match='ContactSDF'):
        validate_options(args)
