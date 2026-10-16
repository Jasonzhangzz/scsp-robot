"""Independent SciPy SLSQP searches sharing batched native FA evaluations.

SciPy 1.15's Fortran reverse communication avoids serializing multistarts.
No numerical objective derivatives or constraint approximations are used.
"""
import numpy as np


class SLSQPSearch:
    def __init__(self,x,value,gradient,iterations,ftol):
        try:
            from scipy.optimize._slsqp import slsqp
        except ImportError as error:
            raise RuntimeError('Batched SLSQP requires the SciPy 1.15 reverse-communication interface') from error
        self.slsqp=slsqp;self.x=np.array(x,dtype=float,copy=True);n=len(x)
        n1=n+1;mineq=2*n1
        size=3*n1*(n1+1)+(n1+1)*(mineq+2)+2*mineq+(n1+mineq)*n1+n1+(n1*n)//2+3*n+3*n1+1
        self.w=np.zeros(size);self.jw=np.zeros(mineq,dtype=np.int32)
        self.lower=np.full(n,np.nan);self.upper=np.full(n,np.nan)
        self.mode=np.array(0,dtype=np.int32);self.iteration=np.array(iterations-1,dtype=np.int32)
        self.accuracy=np.array(ftol);self.c=np.zeros(1);self.a=np.zeros((1,n+1))
        self.f=float(value);self.g=np.r_[gradient,0.]
        self.floats=[np.array(0.) for _ in range(10)]
        self.integers=[np.array(0,dtype=np.int32) for _ in range(8)]
        self.calls=0

    def advance(self):
        self.slsqp(0,0,self.x,self.lower,self.upper,self.f,self.c,self.g,self.a,
                   self.accuracy,self.iteration,self.mode,self.w,self.jw,*self.floats,*self.integers)
        self.calls+=1
        return int(self.mode)

    def supply(self,value,gradient=None):
        self.f=float(value) if np.isfinite(value) else 1.e20
        if gradient is not None:self.g[:-1]=gradient


def solve_batched(evaluate,variables,iterations,ftol):
    value,gradient=evaluate(variables,True)
    searches=[SLSQPSearch(x,f,g,iterations,ftol) for x,f,g in zip(variables,value,gradient)]
    best=variables.copy();best_value=value.copy();initial=value.copy()
    done=~np.isfinite(value);modes=np.zeros(len(searches),int);improvements=0;asynchronous_requests=0
    while not done.all():
        for i,search in enumerate(searches):
            if not done[i]:
                modes[i]=search.advance()
                done[i]=abs(modes[i])!=1
        if done.all():break
        trial=np.stack([s.x if not done[i] else best[i] for i,s in enumerate(searches)])
        need_gradient=bool(np.any((modes==-1)&~done))
        asynchronous_requests+=int(need_gradient and np.any((modes==1)&~done))
        value,gradient=evaluate(trial,need_gradient)
        better=(value<best_value)&~done
        best[better]=trial[better];best_value[better]=value[better]
        improvements+=int(better.sum())
        for i,search in enumerate(searches):
            if not done[i]:search.supply(value[i],gradient[i] if modes[i]==-1 else None)
    # Reconstruct native states for the following contact/dual update.
    evaluate(best,False)
    diagnostic=dict(backend='scipy_SLSQP',statuses=modes.tolist(),
        iterations=[int(s.iteration) for s in searches],initial_values=initial.tolist(),best_values=best_value.tolist(),
        independent_multistarts=True,analytic_gradients=True,asynchronous_requests=asynchronous_requests)
    return best,improvements,diagnostic
