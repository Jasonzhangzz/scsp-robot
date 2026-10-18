"""Independent algebra and nonsmooth-derivative contracts for FA codegen."""
import casadi as cs
import numpy as np
import pytest

from examples.retarget.compiled.spd import spd_solver
from examples.retarget.compiled.model import native_clamp


@pytest.mark.parametrize('size',[8,32])
def test_spd_primal_and_vjp_against_generic_qr(tmp_path,size):
    solve,_=spd_solver(size,tmp_path)
    raw=cs.MX.sym('matrix',size,size);b=cs.MX.sym('rhs',size);seed=cs.MX.sym('seed',size)
    matrix=(raw+raw.T)/2
    custom=solve(matrix,b);reference=cs.solve(matrix,b)
    variables=cs.vertcat(cs.vec(raw),b)
    function=cs.Function('compare',[raw,b,seed],[custom,reference,
        cs.jtimes(custom,variables,seed,True),cs.jtimes(reference,variables,seed,True)])
    rng=np.random.default_rng(1024+size)
    for _ in range(3):
        factor=rng.normal(size=(size,size));a=factor@factor.T+np.eye(size)
        for _ in range(2):  # Same factor, different rhs: exercise factor reuse.
            rhs=rng.normal(size=size);s=rng.normal(size=size)
            value,truth,gradient,truth_gradient=map(np.asarray,function(a,rhs,s))
            np.testing.assert_allclose(value,np.linalg.solve(a,rhs)[:,None],rtol=1e-11,atol=1e-11)
            np.testing.assert_allclose(value,truth,rtol=1e-11,atol=1e-11)
            np.testing.assert_allclose(gradient,truth_gradient,rtol=1e-10,atol=1e-10)


def test_clamp_uses_native_warp_boundary_derivatives():
    x=cs.SX.sym('x');y=native_clamp(x,-1,1)
    function=cs.Function('clamp',[x],[y,cs.jacobian(y,x)])
    result=np.array([list(map(float,function(x))) for x in [-2,-1,0,1,2]])
    np.testing.assert_array_equal(result[:,0],[-1,-1,0,1,1])
    np.testing.assert_array_equal(result[:,1],[0,1,1,0,0])
