"""Check the independent reverse-communication SLSQP adapter against SciPy."""
import numpy as np
import pytest

pytest.importorskip('scipy.optimize._slsqp')
from scipy.optimize import minimize
from examples.retarget.fa_hand_slsqp import solve_batched


def test_independent_searches_match_public_scipy():
    centers=np.array([[1.,2.,-3.],[-2.,.5,4.],[0.,0.,0.]])
    weights=np.array([[1.,3.,10.],[5.,.1,2.],[.5,2.,4.]])
    def evaluate(x,gradient):
        delta=x-centers
        return (weights*delta*delta).sum(1),2*weights*delta if gradient else None
    actual,_,diag=solve_batched(evaluate,np.zeros_like(centers),30,1.e-10)
    for i in range(len(centers)):
        result=minimize(lambda x: np.sum(weights[i]*(x-centers[i])**2),np.zeros(3),
            jac=lambda x: 2*weights[i]*(x-centers[i]),method='SLSQP',options={'maxiter':30,'ftol':1.e-10})
        np.testing.assert_allclose(actual[i],result.x,atol=1.e-10)
    assert diag['statuses']==[0,0,0]
    assert diag['iterations'][2]<min(diag['iterations'][:2])


def test_iteration_limit_preserves_best_finite_plan():
    initial=np.array([[0.,0.],[1.,-1.]])
    def evaluate(x,gradient):
        delta=x-np.array([3.,4.])
        value=np.sum(delta**2,axis=1)
        value[np.any(np.abs(x)>5.,axis=1)]=np.inf
        return value,2*delta if gradient else None
    actual,_,diag=solve_batched(evaluate,initial,3,1.e-12)
    assert np.isfinite(evaluate(actual,False)[0]).all()
    assert np.all(evaluate(actual,False)[0]<=evaluate(initial,False)[0])


def test_asynchronous_nonlinear_searches_match_public_scipy():
    a=np.array([100.,1.,1000.,10.]);b=np.array([1.,10.,.1,50.])
    initial=np.array([[-1.2,1.,0.],[2.,-2.,1.],[-.5,2.,-1.],[.1,-.5,2.]])
    def evaluate(x,gradient):
        r=x[:,1]-x[:,0]**2;s=x[:,2]-np.sin(x[:,0])
        f=a*r*r+(1-x[:,0])**2+b*s*s
        g=np.c_[-4*a*x[:,0]*r+2*(x[:,0]-1)-2*b*s*np.cos(x[:,0]),2*a*r,2*b*s]
        return f,g if gradient else None
    actual,_,diag=solve_batched(evaluate,initial,60,1e-10)
    assert diag['asynchronous_requests']>0
    for i in range(len(initial)):
        def single(x):
            batch=initial.copy();batch[i]=x;f,g=evaluate(batch,True)
            return f[i],g[i]
        result=minimize(single,initial[i],jac=True,method='SLSQP',options={'maxiter':60,'ftol':1e-10})
        np.testing.assert_allclose(actual[i],result.x,atol=1e-8)
        assert diag['iterations'][i]==result.nit
