"""Diagnostic batched projected LM using the exact paper SCM/CasADi graph.
CPU float64 mapped residual evaluation and vectorized 3x3 LM solves.
This deliberately does not reuse the different-model Warp microbenchmark.
"""
import os,subprocess,time
from pathlib import Path
from types import MethodType
import casadi as cs
import numpy as np

class ExactBatchLM:
 def __init__(self,opt,directory,iterations=20):
  self.opt=opt;self.iterations=iterations;self.maps={};self.checks=0
  opt._precompile_optimization_function(build_solver=False)
  fn=opt.force_objective_fn
  symbols=[cs.MX.sym(fn.name_in(i),*fn.size_in(i)) for i in range(fn.n_in())]
  xp,cost=fn.call(symbols);cfg=symbols[fn.index_in('object_config')];xd=symbols[fn.index_in('x_d')];force=symbols[0]
  q=xp[3:7]/cs.norm_2(xp[3:7]);qd=xd[3:7]/cs.norm_2(xd[3:7]);axis=cfg[3:6]+2*cs.cross(q[1:],cs.cross(q[1:],cfg[3:6])+q[0]*cfg[3:6]);up=cfg[6:9]
  delta=xp[:3]-xd[:3];planar=delta-up*cs.dot(delta,up);length=cs.norm_2(planar)
  position=cs.if_else(cfg[9]>0,planar*cs.fmax(length-cfg[9],0)/cs.fmax(length,1e-12),delta)
  residual=cs.vertcat(cs.sqrt(cfg[0])*position,cs.sqrt(cfg[1])*(q-qd*cs.dot(q,qd)),.5*cs.sqrt(cfg[2])*(axis-up),cs.sqrt(cfg[11])*cs.fmax(1-cs.dot(axis,up)-cfg[10],0),np.sqrt(opt.force_reg_coef)*force,np.sqrt(opt.friction_reg_coef)*force[1:])
  evaluator=cs.Function('lm_eval',symbols,[residual,xp,cost],fn.name_in(),['residual','x_plus','cost']).expand()
  directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
  cwd=os.getcwd()
  try:
   os.chdir(directory);evaluator.generate('lm_eval.c');subprocess.run(['gcc','-O2','-fPIC','-shared','lm_eval.c','-o','lm_eval.so','-lm'],check=True)
  finally:os.chdir(cwd)
  self.fn=cs.external('lm_eval',str(directory.resolve()/'lm_eval.so'))
  self.names=self.fn.name_in();self.original_candidate=opt._solve_optimization
 def project(self,f):
  f=f.copy();f[:,0]=np.clip(f[:,0],1e-6,self.opt.max_normal_force);bound=self.opt.mu_arm_obj*f[:,0]
  if self.opt.friction_cone=='pyramid':
   total=np.abs(f[:,1:]).sum(1);f[:,1:]*=np.minimum(1,bound/np.maximum(total,1e-12))[:,None]
  else:f[:,1:]=np.clip(f[:,1:],-bound[:,None],bound[:,None])
  return f
 def solve(self,x_d,current_x,tau_o,visible_face_idx,v_last,D_inv):
  opt=self.opt;ids=np.asarray(visible_face_idx,dtype=np.int32).reshape(-1);n=len(ids)
  common=dict(x_d=x_d,current_x=current_x,v_last=v_last,J_tilde=opt.J_tilde,D_inv=D_inv,tau_o_np=tau_o,curr_ori_coef=1.,env_phi=opt.env_phi,scm_theta=opt.scm_theta,object_config=opt.object_cost_config(),force_limit=opt.max_contact_force)
  if 'tip_velocity' in self.names:common['tip_velocity']=opt.rank_tip_velocity
  matrices={name:np.tile(np.asarray(value).reshape(self.fn.size_in(name),order='F'),(1,n)) for name,value in common.items()}
  for name,values in [('n_arm',opt.normal),('t1',opt.t1),('t2',opt.t2),('p_arm',opt.sample_point)]:matrices[name]=np.asarray(values[ids]).T
  if n not in self.maps:self.maps[n]=self.fn.map(n,'serial')
  mapped=self.maps[n]
  def evaluate(f):
   values=dict(matrices,lam_arm=f.T);r,x,c=mapped.call([values[name] for name in self.names]);return np.asarray(r).T,np.asarray(x).T,np.asarray(c).reshape(-1)
  force=np.tile([.9*opt.max_normal_force,0.,0.],(n,1));r,x,c=evaluate(force)
  err=np.max(np.abs((r*r).sum(1)-c));assert err<1e-6*max(1,np.max(np.abs(c))),err
  self.checks+=1;damping=np.full(n,1e-3);eps=1e-4
  for _ in range(self.iterations):
   jac=np.stack([(evaluate(force+np.eye(3)[k]*eps)[0]-r)/eps for k in range(3)],axis=2)
   gram=np.einsum('nri,nrj->nij',jac,jac);gradient=np.einsum('nri,nr->ni',jac,r)
   diag=np.maximum(np.diagonal(gram,axis1=1,axis2=2),1e-6)
   matrix=gram+np.eye(3)[None]*((damping[:,None]*diag)+1e-10)[:,None,:]
   step=np.linalg.solve(matrix,-gradient[...,None])[...,0];step=np.clip(step,-2.,2.)
   trial=self.project(force+step);rr,xx,cc=evaluate(trial);accept=cc<c
   force=np.where(accept[:,None],trial,force);r=np.where(accept[:,None],rr,r);x=np.where(accept[:,None],xx,x);c=np.where(accept,cc,c)
   damping=np.clip(damping*np.where(accept,.5,4.),1e-8,1e8)
  assert np.isfinite(c).all() and np.isfinite(force).all()
  opt.last_solver_status='batch-lm-cpu-exact-scm'
  return ids,c,force,x
 def install(self):
  opt=self.opt;opt.solver='torch-gn'
  def batch(instance,*args,**kwargs):return self.solve(*args,**kwargs)
  opt._choose_contact_points_torch_batch=MethodType(batch,opt)
  def no_fallback(instance,**kwargs):raise RuntimeError('Batch LM failed: scalar fallback disabled')
  opt._solve_optimization=MethodType(no_fallback,opt)
