from types import SimpleNamespace
import numpy as np
from examples.retarget.fa_value_cache import FAValueCache


class Array:
    def __init__(self):self.value=None
    def assign(self,x):self.value=np.array(x,copy=True)


def oracle():
    control=SimpleNamespace(raw=Array(),valid=Array())
    backend=SimpleNamespace(settings=SimpleNamespace(worlds=2),index=SimpleNamespace(action_dim=1),
        control=control,loss=SimpleNamespace(total=Array()))
    solver=SimpleNamespace(backend=backend,variable_scale=np.array([.1]),_forward_variables=None,
        evaluations=0,value_reuses=0)
    history=[];bias=[0.]
    def evaluate(x,gradient):
        history.append((x.copy(),gradient));solver.evaluations+=1
        raw=(x*.1).astype(np.float32).astype(float)
        values=raw[:,0]**2+np.arange(2)*10+bias[0]
        solver._forward_variables=x.copy();solver._forward_value=values.copy()
        return values,2*raw*.1 if gradient else None
    return solver,FAValueCache(solver,evaluate),history,bias


def test_recycle_independent_worlds_and_refresh_gradient():
    s,m,h,_=oracle()
    a=np.array([[1.],[2.]]);b=np.array([[3.],[4.]]);mixed=np.array([[1.],[4.]])
    va,_=m.evaluate(a,False);vb,_=m.evaluate(b,False)
    got,_=m.evaluate(mixed,False)
    np.testing.assert_array_equal(got,[va[0],vb[1]])
    assert len(h)==2 and m.skipped_batches==1 and m.dirty
    assert s._forward_variables is None
    np.testing.assert_array_equal(s.backend.control.raw.value[:,1:],(mixed*.1).astype(np.float32))
    _,gradient=m.evaluate(mixed,True)
    np.testing.assert_array_equal(gradient,2*(mixed*.1).astype(np.float32).astype(float)*.1)
    assert len(h)==3 and not m.dirty


def test_quantized_keys_and_contact_invalidation():
    s,m,h,bias=oracle();a=np.array([[1.],[2.]])
    values,_=m.evaluate(a,False)
    # Same controls after the original FA FP32 conversion.
    x=a+1e-9;got,_=m.evaluate(x,False)
    np.testing.assert_array_equal(got,values);assert len(h)==1
    bias[0]=5.;m.invalidate();got,_=m.evaluate(a,False)
    np.testing.assert_array_equal(got,values+5);assert len(h)==2


def test_cache_has_bounded_capacity():
    s,m,h,_=oracle();m.capacity=1
    a=np.array([[1.],[2.]]);b=np.array([[3.],[4.]])
    m.evaluate(a,False);m.evaluate(b,False);m.evaluate(a,False)
    assert len(h)==3
