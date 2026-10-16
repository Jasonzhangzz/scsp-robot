"""Compiled, callback-free FA rollout/cost/control-adjoint evaluator.

The existing optimizer, knot interpolation, line search and candidate family
remain in fa_contact_admm. This module only replaces their numerical oracle.
"""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import subprocess
import shutil
import tempfile
from time import perf_counter
import casadi as cs
import numpy as np

from examples.retarget.compiled.model import CompiledFAModel
from examples.retarget.compiled.cost import StageCost


class CompiledFAEvaluator:
    def __init__(self,backend,directory,*,threads=8):
        self.backend=backend;self.threads=max(1,int(threads));self.cache=None
        started=perf_counter();self.physics=CompiledFAModel(backend.compiled,backend.cpu_model,directory)
        self.cost=StageCost(backend);m=backend.cpu_model;self.N=len(backend.steps);self.nx=m.nq+m.nv;self.nu=m.nu
        self.nfeatures=13*backend.compiled.base.collision.contact_count
        x=cs.MX.sym('x',self.nx);u=cs.MX.sym('u',self.nu);p=cs.MX.sym('p',self.cost.np)
        selected=cs.MX.sym('selected',13,backend.compiled.base.collision.contact_count)
        out=self.physics.step_features.call([x[:m.nq],x[m.nq:],u,selected],True,False);nextx=cs.vertcat(out[0],out[1])
        loss=self.cost.function.call([out[0],out[3],out[4],p],True,False)[0]
        rm=backend.objective.regularizer.model;cfg=backend.cfg
        if cfg.smoothing.enabled:
            hv=rm.hand_qvel.numpy();ratio=cs.DM(rm.state_component_ratio.numpy())
            loss+=1e-6*cfg.smoothing.hand_state_weight*cs.dot(ratio,(out[1][hv]/cs.DM(rm.state_velocity_scale.numpy()))**2
                +2*(out[2][hv]/cs.DM(rm.state_acceleration_scale.numpy()))**2)
        core=cs.Function('fa_c_stage_core',[x,u,p,selected],[nextx,loss])
        selection=self.physics.select(x[:m.nq])
        stage=cs.Function('fa_c_stage',[x,u,p],[*core.call([x,u,p,selection],True,False),selection])
        seed=cs.MX.sym('seed',self.nx+1)
        adj=cs.jtimes(cs.vertcat(nextx,loss),cs.vertcat(x,u),seed,True)
        reverse=cs.Function('fa_c_reverse',[x,u,p,selected,seed],[adj])
        poses,contact=self.physics.observe_features.call([x[:m.nq],selected],True,False)
        terminal=self.cost.function.call([x[:m.nq],poses,contact,p],True,False)[0]
        tc=cs.Function('fa_c_terminal_core',[x,p,selected],[terminal])
        terminal_f=cs.Function('fa_c_terminal',[x,p],[tc(x,p,selection),selection])
        terminal_g=cs.Function('fa_c_terminal_gradient',[x,p,selected],[cs.gradient(terminal,x)])
        functions=[stage,reverse,terminal_f,terminal_g]
        generator=cs.CodeGenerator('equations.c',{'with_header':True})
        for f in functions:generator.add(f)
        Path(directory).mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='codegen_',dir=directory) as temporary:
            generator.generate(str(Path(temporary).resolve())+'/')
            source_text=(Path(temporary)/'equations.c').read_text()
            header_text=(Path(temporary)/'equations.h').read_text()
        digest=hashlib.sha256(source_text.encode()).hexdigest()[:16]
        equations=Path(directory)/('equations_'+digest);equations.mkdir(parents=True,exist_ok=True)
        if not (equations/'equations.c').exists():(equations/'equations.c').write_text(source_text)
        if not (equations/'equations.h').exists():(equations/'equations.h').write_text(header_text)
        optimization=os.environ.get('FA_COMPILED_OPT','2')
        if optimization not in ('0','1','2','3'):raise ValueError('FA_COMPILED_OPT must be 0, 1, 2 or 3')
        obj=equations/('equations_O'+optimization+'.o')
        # The mathematical C object is independent of the mesh-search binary
        # implementation, provided its external ABI is unchanged.
        if not obj.exists():
            for old in Path(directory).glob('*/equations.c'):
                candidate=old.with_name(obj.name)
                if not candidate.exists() and optimization=='2' and old.with_name('rollout.so').exists():candidate=old.with_name('equations.o')
                if candidate.exists() and hashlib.sha256(old.read_bytes()).hexdigest()[:16]==digest:
                    shutil.copyfile(candidate,obj);shutil.copyfile(old.with_suffix('.h'),equations/'equations.h');break
        dependencies=[self.physics.geometry_library.resolve(),self.physics.spd_library.resolve()]
        driver_digest=hashlib.sha256((_DRIVER+''.join(str(p) for p in dependencies)).encode()).hexdigest()[:12]
        build=equations/(driver_digest+'_O'+optimization);build.mkdir(exist_ok=True)
        lib=build/'rollout.so'
        if not lib.exists():
            if not obj.exists():
                # CasADi has already done common-subexpression elimination.
                # GCC's PRE pass can take many minutes on these large blocks.
                subprocess.run(['cc','-O'+optimization,'-fno-tree-pre','-fno-tree-vrp','-fdisable-tree-evrp',
                    '-fPIC','-c',str(equations/'equations.c'),'-o',str(obj)],check=True)
            sizes=dict(NX=self.nx,NU=self.nu,NP=self.cost.np,NFEATURE=self.nfeatures,
                NW=max(f.sz_w() for f in functions),NIW=max(f.sz_iw() for f in functions),
                NARG=max(f.sz_arg() for f in functions),NRES=max(f.sz_res() for f in functions))
            driver=''.join(f'#define {k} {v}\n' for k,v in sizes.items())+_DRIVER
            (build/'driver.cpp').write_text(driver)
            subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC','-fopenmp','-I'+str(equations),str(build/'driver.cpp'),str(obj),
                *[str(p) for p in dependencies],'-Wl,-rpath,'+':'.join(str(p.parent) for p in dependencies),'-lm','-o',str(lib)],check=True)
        self.shared=ctypes.CDLL(str(lib.resolve()))
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.shared.fa_rollout_batch.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,array,array,array,array,array,array,array]
        self.shared.fa_rollout_batch.restype=ctypes.c_int
        self.library=lib;self.build_seconds=perf_counter()-started
        self.P=np.zeros((self.N,backend.settings.horizon));self.Pprev=np.zeros(self.N)
        for k,(left,right,fraction) in enumerate(backend.dense_knot_map):
            if left<0:self.Pprev[k]+=1-fraction
            else:self.P[k,left]+=1-fraction
            self.P[k,right]+=fraction
        self.timing={};self.last=None

    def prepare(self):
        b=self.backend
        self.initial=np.r_[b.control.base_qpos.numpy(),b.control.base_qvel.numpy()].astype(float)
        self.previous=b.control.previous_action.numpy().astype(float)
        self.previous_previous=b.objective.regularizer._previous_previous.numpy().astype(float)
        self.actions=np.stack([a.numpy() for a in b.control.target_actions]).astype(float)
        # Targets are shared by every multistart; download them once per
        # window, rather than once per world and ADMM hand block.
        self.common_parameters=np.stack([self.cost.parameters(w) for w in (*b.loss.immediate,b.loss.terminal)])
        self.cache=None

    def parameters(self):
        if self.cache is None:
            b=self.backend
            points=[p.c.numpy() for p in b.objective.points]
            self.cache=np.broadcast_to(self.common_parameters,(b.settings.worlds,*self.common_parameters.shape)).copy()
            offset=sum(self.cost.sizes[:-2]);g=self.cost.g
            self.cache[:,:,offset:offset+3*g]=np.stack(points,axis=1).reshape(b.settings.worlds,self.N+1,3*g)
        return self.cache

    def regularization(self,knots,gradient):
        b=self.backend;cfg=b.cfg;rm=b.objective.regularizer.model
        result=np.zeros(len(knots));g=np.zeros_like(knots)
        if not cfg.smoothing.enabled:return result,g
        previous=self.previous;pp=self.previous_previous
        last_dt=cfg.simulator.time_grid.action_dt;dt=cfg.simulator.time_grid.knot_dt
        vw=rm.control_velocity_weight.numpy();vs=rm.control_velocity_scale.numpy();ascale=rm.control_acceleration_scale.numpy()
        for k in range(knots.shape[1]):
            current=knots[:,k];dv=current-previous
            vel=dv/(vs*dt);acc=(2/(last_dt+dt))*((current-previous)/dt-(previous-pp)/last_dt)/ascale
            result+=1e-6*(np.sum(vw*vel**2,axis=1)+cfg.smoothing.control_acceleration_weight*np.sum(acc**2,axis=1))
            if gradient:
                direct=2e-6*vw*dv/(vs*dt)**2
                ag=2e-6*cfg.smoothing.control_acceleration_weight*acc*(2/(last_dt+dt))/ascale
                g[:,k]+=direct+ag/dt
                if k>=1:g[:,k-1]-=direct+ag*(1/dt+1/last_dt)
                if k>=2:g[:,k-2]+=ag/last_dt
            pp,previous=previous,current;last_dt=dt
        return result,g

    def evaluate_raw(self,raw,gradient):
        started=perf_counter();b=self.backend;count=len(raw)
        knots=self.actions[None]+np.asarray(raw).reshape(count,b.settings.horizon,self.nu)
        commands=np.ascontiguousarray(np.einsum('nk,wku->wnu',self.P,knots)+self.Pprev[None,:,None]*self.previous[:,None,:])
        initial=np.ascontiguousarray(np.broadcast_to(self.initial,(count,self.nx)))
        reuse=bool(gradient and self.last is not None and np.array_equal(raw,self.last['raw'])
                   and self.last.get('parameters') is self.cache and self.cache is not None)
        values=self.last['trajectory_values'].copy() if reuse else np.empty(count)
        gu=np.empty_like(commands);states=self.last['states'].copy() if reuse else np.empty((count,self.N+1,self.nx))
        features=self.last['features'].copy() if reuse else np.empty((count,self.N+1,self.nfeatures))
        status=self.shared.fa_rollout_batch(count,self.N,min(self.threads,count),2 if reuse else int(gradient),initial,commands,self.parameters(),values,gu,states,features)
        if status:raise FloatingPointError(f'Compiled FA rollout failed: {status}')
        trajectory_values=values.copy();reg,rg=self.regularization(knots,gradient);values+=reg
        g=(np.einsum('nk,wnu->wku',self.P,gu)+rg).reshape(count,-1) if gradient else None
        valid=np.isfinite(values)
        if gradient:valid&=np.isfinite(g).all(axis=1);g[~valid]=0.
        values[~valid]=np.inf
        self.last=dict(states=states,commands=commands,values=values,gradient=g,raw=np.asarray(raw).copy(),
                       trajectory_values=trajectory_values,parameters=self.cache)
        self.last['features']=features
        label='value_gradient' if gradient else 'value'
        self.timing[label]=self.timing.get(label,0.)+perf_counter()-started
        return values,g


_DRIVER=r'''
#include "equations.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
extern "C" int fa_rollout_batch(int worlds,int N,int threads,int gradient,const double*initial,
    const double*commands,const double*params,double*values,double*gu,double*states,double*features){
 omp_set_max_active_levels(2);
 int failures=0;
 #pragma omp parallel for num_threads(threads) reduction(+:failures)
 for(int world=0;world<worlds;world++){
   std::vector<double>w(NW),seed(NX+1),adj(NX+NU);std::vector<casadi_int>iw(NIW);
   std::vector<const double*>a(NARG);std::vector<double*>r(NRES);
   double*xs=states+world*(N+1)*NX;const double*us=commands+world*N*NU;const double*ps=params+world*(N+1)*NP;
   double*fs=features+world*(N+1)*NFEATURE;
   std::copy(initial+world*NX,initial+(world+1)*NX,xs);double value=0,stage_value=0;
   for(int k=0;k<N && gradient!=2;k++){
     a[0]=xs+k*NX;a[1]=us+k*NU;a[2]=ps+k*NP;r[0]=xs+(k+1)*NX;r[1]=&stage_value;r[2]=fs+k*NFEATURE;
     failures+=fa_c_stage(a.data(),r.data(),iw.data(),w.data(),0)!=0;value+=stage_value;
   }
   a[0]=xs+N*NX;a[1]=ps+N*NP;r[0]=&stage_value;r[1]=fs+N*NFEATURE;
   if(gradient!=2){failures+=fa_c_terminal(a.data(),r.data(),iw.data(),w.data(),0)!=0;value+=stage_value;values[world]=value;}
   else value=values[world];
   if(!std::isfinite(value)){values[world]=INFINITY;std::fill(gu+world*N*NU,gu+(world+1)*N*NU,0.);continue;}
   if(gradient){
     a[2]=fs+N*NFEATURE;r[0]=seed.data();failures+=fa_c_terminal_gradient(a.data(),r.data(),iw.data(),w.data(),0)!=0;seed[NX]=1;
     for(int k=N-1;k>=0;k--){
       a[0]=xs+k*NX;a[1]=us+k*NU;a[2]=ps+k*NP;a[3]=fs+k*NFEATURE;a[4]=seed.data();r[0]=adj.data();
       failures+=fa_c_reverse(a.data(),r.data(),iw.data(),w.data(),0)!=0;
       std::copy(adj.data()+NX,adj.data()+NX+NU,gu+(world*N+k)*NU);
       std::copy(adj.data(),adj.data()+NX,seed.data());
     }
   }
 }
 return failures;
}
'''
