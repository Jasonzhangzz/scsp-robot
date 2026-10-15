"""Alternative contact-loading guesses are optimized, never sent as actions."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from planning.franka_mpc_seeds import contact_loading_seeds, rollout_guess, best_contact_plan


def test_loading_candidates_rotate_with_contact_and_preserve_zero_lateral_force():
    force, normal = np.array([.6, 0., .4]), np.array([-1.,0.,0.])
    baseline = contact_loading_seeds(force, normal, np.eye(3)*600, 5)
    assert len(baseline) == 2
    assert baseline[0][0,2] == 0. and baseline[0][-1,2] > 0.
    assert baseline[1][0,2] < 0. < baseline[1][-1,2]
    r = Rotation.from_euler('z', 1.1).as_matrix()
    for a,b in zip(baseline,contact_loading_seeds(r@force,r@normal,np.eye(3)*600,5)):
        np.testing.assert_allclose(b,a@r.T,atol=1e-12)


@pytest.mark.parametrize('force,normal', [([.7,0.,0.],[-1,0,0]), ([0,0,0],[-1,0,0]), ([1,0,1],[0,0,0]), ([float('nan'),0,1],[-1,0,0])])
def test_nonlifting_or_invalid_candidates_do_not_seed_a_hook(force, normal):
    assert contact_loading_seeds(force, normal, np.eye(3)*600, 5) == []


def stub():
    return SimpleNamespace(param_=SimpleNamespace(mpc_u_lb_=np.full(3,-.005),mpc_u_ub_=np.full(3,.005),
        robot_stiff_=np.eye(3)*600,model_params=1.),first_action_bounds=(np.array([0.,0.,0.]),np.full(3,.001)),
        contact_force_limit=.8,model=SimpleNamespace(runtime_parameters=lambda x:x),
        stage_prediction_fn=lambda state,u,*_:state+u)


def test_guess_obeys_first_action_bounds_and_rolls_the_actual_predictor():
    mpc=stub();state=np.array([1.,2.,3.])
    seed=rollout_guess(mpc,state,np.array([[-.01,.01,-.01],[.002,0,.002]]),[],[])
    u,x=seed['u_traj'],seed['x_traj']
    assert np.all(u[0]>=0.) and np.all(u[0]<=.001)
    assert np.max(np.linalg.norm(600*u,axis=1))<=.8+1e-9
    np.testing.assert_allclose(x[1:],state+np.cumsum(u,axis=0))


def result(cost, action, backend='acados'):
    u=np.asarray(action).reshape(1,3)
    return dict(cost_opt=[cost],action=u[0].copy(),u_traj=u,solver_backend=backend,sol_guess=dict(u_traj=u))


def test_only_lowest_cost_optimized_result_is_sent_not_the_loading_seed():
    mpc=stub();ordinary=result(10,[.0002,0,0]);better=result(3,[.0003,0,.0002])
    solved=iter([ordinary,better,result(12,[.0001,0,0])]);calls=[];resets=[]
    mpc.plan_once=lambda **kw:(calls.append(kw) or next(solved))
    mpc._reset_acados_solver=lambda:resets.append(1)
    guesses=[dict(u_traj=np.array([[0.,0.,-.001]])),dict(u_traj=np.array([[.001,0,0]]))]
    selected=best_contact_plan(mpc,dict(sol_guess='ordinary'),guesses)
    assert selected is better
    assert selected['contact_seed_selected']==1
    np.testing.assert_equal(selected['contact_seed_costs'],[10,3,12])
    assert len(calls)==3 and len(resets)==3
    np.testing.assert_allclose(selected['action'],[.0003,0,.0002])


@pytest.mark.parametrize('candidate', [result(-2,[.004,0,0]), result(-2,[-.0003,0,0]), result(-2,[.0003,0,0],'acados_failed'),result(float('nan'),[0,0,0])])
def test_invalid_alternative_cannot_replace_the_ordinary_mpc_action(candidate):
    mpc=stub();ordinary=result(10,[.0002,0,0]);solved=iter([ordinary,candidate])
    mpc.plan_once=lambda **kw:next(solved);mpc._reset_acados_solver=lambda:None
    assert best_contact_plan(mpc,{},[{}]) is ordinary


def test_failed_alternative_diagnostics_are_strict_json_serializable():
    import json
    mpc=stub();ordinary=result(10,[.0002,0,0]);solved=iter([ordinary,result(float('nan'),[0,0,0])])
    mpc.plan_once=lambda **kw:next(solved);mpc._reset_acados_solver=lambda:None
    selected=best_contact_plan(mpc,{},[{}])
    assert selected['contact_seed_costs']==[10.,None]
    json.dumps(selected['contact_seed_costs'],allow_nan=False)
