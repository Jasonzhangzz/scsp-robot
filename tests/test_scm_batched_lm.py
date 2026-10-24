"""CUDA batched LM must evaluate the actual current contact model."""
import numpy as np
import pytest

wp = pytest.importorskip('warp')
pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason='CUDA required')


@pytest.mark.parametrize('model', ['soft-scm', 'comfree', 'diagonal-contact'])
def test_gpu_contact_map_matches_current_casadi_graph(model, monkeypatch):
    from examples.mpc.fingertips.test.test_0902 import build_parser
    from examples.mpc.fingertips.test.params import ExplicitMPCParams
    from envs.fingertips_env import MjSimulator
    from planning.direct_contact_force import (configure_direct_force_optimizer,
        environment_contacts_body,object_frame)
    from utils import rotations
    from planning.scm_batched_lm import GPUExactBatchLM

    monkeypatch.setenv('MUJOCO_HEADLESS', '1')
    args = build_parser().parse_args(['--contact-model', model])
    args.rollout, args.solver = True, 'batch-lm'
    param = ExplicitMPCParams(args, rand_seed=0, target_type='ground-rotation', model='explicit')
    param.trial_seed_ = 0
    env = MjSimulator(param)
    opt = configure_direct_force_optimizer(param, env, args)
    lm = GPUExactBatchLM(opt, iterations=20)
    lm.install()
    lm.validate = True
    q,rot = env.get_state(),object_frame(env)
    jac,gap = environment_contacts_body(env,param.lambda_source_param_.mu_table_object_)
    opt.update_Jacobian(jac,phi=gap)
    visible = opt.get_availble_point_idx(q[:3],rot,param.target_p_,args.ground_height_threshold,
        viewpoint_local=None,heading_filter=False)
    target = np.r_[rot.T@(param.target_p_-q[:3]),rotations.quaternion_multiply(
        rotations.quaternion_conjugate(q[3:7]),param.target_q_)]
    gravity = np.r_[rot.T@param.gravity_[:3]*param.obj_mass_,np.zeros(3)]
    opt.choose_contact_points(target,np.r_[np.zeros(3),1.,np.zeros(3)],gravity,
        opt.filter_rankable_indices(visible),v_last=opt.rank_object_velocity.copy(),
        force_required=True,query_local=rot.T@(q[7:10]-q[:3]))
    assert lm.validations and lm.validations[0]['candidates'] > 1
    assert lm.validations[0]['pose_max_abs'] < 1e-7
    assert lm.validations[0]['cost_max_abs'] < 1e-7
    assert np.isfinite(opt.last_best_x_plus).all()
    assert opt.last_solver_status == 'batch-lm-gpu-exact-' + model
    if model == 'diagonal-contact':
        np.testing.assert_allclose(opt.Q_inv/(opt.h*opt.h),
                                   np.linalg.inv(opt.diagonal_contact_physical_mass), rtol=1e-12)
        assert lm.validations[0]['all_candidates_nonincreasing']
        assert lm.validations[0]['strictly_improved_candidates'] > 0
        assert lm.validations[0]['final_cost_mean'] < lm.validations[0]['initial_cost_mean']
