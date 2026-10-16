"""Reuse identical per-candidate values without changing FA's optimizer.

Backtracking repeatedly evaluates already accepted or stalled candidates.
Worlds are independent, so a new batch assembled entirely from previously
evaluated world/raw pairs needs no rollout. Gradients retain the original
evaluator, including its native adjoint. Keys use FA's actual FP32 raw controls.
"""
from collections import OrderedDict
import numpy as np


class FAValueCache:
    def __init__(self,solver,evaluate,capacity=512):
        self.solver=solver;self.evaluate_original=evaluate;self.capacity=capacity
        self.reset()

    def reset(self):
        self.cache=[OrderedDict() for _ in range(self.solver.backend.settings.worlds)]
        self.hits=0;self.skipped_batches=0;self.dirty=False

    def invalidate(self):
        for cache in self.cache:cache.clear()
        self.solver._forward_variables=None

    def evaluate(self,variables,gradient):
        s=self.solver;b=s.backend
        raw=(variables*s.variable_scale).astype(np.float32)
        keys=[row.tobytes() for row in raw]
        # Keep the original whole-batch forward/backward reuse when its native
        # buffers are current. Only substitute expensive repeated value calls.
        same=np.array_equal(variables,getattr(s,'_forward_variables',None))
        hit=not gradient and not same and all(key in cache for key,cache in zip(keys,self.cache))
        if hit:
            values=np.array([cache[key] for key,cache in zip(keys,self.cache)])
            for key,cache in zip(keys,self.cache):cache.move_to_end(key)
            full=np.zeros((len(raw),b.index.action_dim+raw.shape[1]),np.float32)
            full[:,b.index.action_dim:]=raw;b.control.raw.assign(full)
            b.control.valid.assign(np.isfinite(values).astype(np.int32))
            b.loss.total.assign((values*1e6).astype(np.float32))
            s._forward_variables=None;s._forward_value=values.copy()
            s.evaluations+=1;s.value_reuses+=1
            self.hits+=len(raw);self.skipped_batches+=1;self.dirty=True
            return values,None
        values,g=self.evaluate_original(variables,gradient)
        self.dirty=False
        for key,cache,value in zip(keys,self.cache,values):
            cache[key]=float(value);cache.move_to_end(key)
            if len(cache)>self.capacity:cache.popitem(last=False)
        return values,g
