"""CUDA batched LM must evaluate the actual current contact model."""
import numpy as np
import pytest

wp = pytest.importorskip('warp')
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason='CUDA required')


@pytest.mark.parametrize('model', ['soft-scm', 'comfree', 'diagonal-contact'])
def test_gpu_contact_map_matches_current_casadi_graph(model, monkeypatch):
    from examples.mpc.fingertips.test import test_scm_mujoco as m
    from planning.scm_batched_lm import GPUExactBatchLM

    monkeypatch.setenv('MUJOCO_HEADLESS', '1')
    args = m.build_parser().parse_args(['--contact-model', model])
    args.rollout, args.solver = True, 'torch-gn'
    param = m.ExplicitMPCParams(args, rand_seed=0, target_type='ground-rotation', model='explicit')
    env = m.MjSimulator(param)
    m._set_physical_mass(param, env, .01)
    opt = m.configure_direct_force_optimizer(param, env, args)
    lm = GPUExactBatchLM(opt, iterations=20)
    lm.install()
    lm.validate = True
    solved = m._solve_step(param, args, env, m.Contact(param), 'surrogate')
    assert lm.validations and lm.validations[0]['candidates'] > 1
    assert lm.validations[0]['pose_max_abs'] < 1e-7
    assert lm.validations[0]['cost_max_abs'] < 1e-7
    np.testing.assert_allclose(solved['response']['selected']['x_plus'],
                               opt.last_best_x_plus, atol=1e-7)
    assert opt.last_solver_status == 'batch-lm-gpu-exact-' + model
    if model == 'diagonal-contact':
        np.testing.assert_allclose(opt.Q_inv/(opt.h*opt.h),
                                   np.linalg.inv(opt.diagonal_contact_physical_mass), rtol=1e-12)
        assert lm.validations[0]['all_candidates_nonincreasing']
        assert lm.validations[0]['strictly_improved_candidates'] > 0
        assert lm.validations[0]['final_cost_mean'] < lm.validations[0]['initial_cost_mean']
