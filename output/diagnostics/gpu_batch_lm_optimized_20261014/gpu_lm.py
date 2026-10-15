"""Fused GPU projected LM with exact forward Jacobian and early stopping.
The reference SCM still performs eight coupled contact sweeps. Geometry is
frozen per round. Rejected steps reuse the current residual/Jacobian/Gram.
"""
import numpy as np
import warp as wp
from gpu_lm_base import Job,V3,V4,V7,V9,R16,Imp,M3,J16,project,residual,GPUExactBatchLM as Base
D9=wp.types.matrix(shape=(9,3),dtype=wp.float64)
D4=wp.types.matrix(shape=(4,3),dtype=wp.float64)
DI=wp.types.matrix(shape=(33,3),dtype=wp.float64)
@wp.func
def prediction_jacobian(job:Job,f:V3,c:int):
 velocity=V9(wp.float64(0.0));dv=D9(wp.float64(0.0))
 for k in range(6):velocity[k]=job.free[k]
 wrench=job.frames[c,0]*f[0]+job.frames[c,1]*f[1]+job.frames[c,2]*f[2]
 for k in range(3):
  velocity[6+k]=wrench[k]/(job.dt*job.stiffness)
  for col in range(3):dv[6+k,col]=job.frames[c,col][k]/(job.dt*job.stiffness)
 impulse=Imp(wp.float64(0.0));di=DI(wp.float64(0.0))
 for sweep in range(8):
  for block in range(job.active_blocks[0]):
   n=job.n[c,block];t1=job.t1[c,block];t2=job.t2[c,block];mn=job.mn[c,block];mt1=job.mt1[c,block];mt2=job.mt2[c,block]
   dnold=V3(di[3*block,0],di[3*block,1],di[3*block,2]);daold=V3(di[3*block+1,0],di[3*block+1,1],di[3*block+1,2]);dbold=V3(di[3*block+2,0],di[3*block+2,1],di[3*block+2,2])
   free=velocity-mn*impulse[3*block]-mt1*impulse[3*block+1]-mt2*impulse[3*block+2]
   df=dv-wp.outer(mn,dnold)-wp.outer(mt1,daold)-wp.outer(mt2,dbold)
   raw=-(wp.dot(n,free)+job.coeff[c,block,4])/job.coeff[c,block,1]
   normal=wp.max(raw,wp.float64(0.0));dn=V3(wp.float64(0.0))
   if raw>wp.float64(0.0):dn=-(wp.transpose(df)*n)/job.coeff[c,block,1]
   supported=free+mn*normal;ds=df+wp.outer(mn,dn)
   a=-wp.dot(t1,supported)/job.coeff[c,block,2];b=-wp.dot(t2,supported)/job.coeff[c,block,3]
   da=-(wp.transpose(ds)*t1)/job.coeff[c,block,2];db=-(wp.transpose(ds)*t2)/job.coeff[c,block,3]
   cap=job.coeff[c,block,0]*normal;dcap=job.coeff[c,block,0]*dn
   root=wp.sqrt(cap*cap+a*a+b*b+wp.float64(1e-12));scale=cap/root
   dscale=dcap/root-cap/(root*root*root)*(cap*dcap+a*da+b*db)
   da=da*scale+a*dscale;db=db*scale+b*dscale;a=a*scale;b=b*scale
   velocity=supported+mt1*a+mt2*b;dv=ds+wp.outer(mt1,da)+wp.outer(mt2,db)
   impulse[3*block]=normal;impulse[3*block+1]=a;impulse[3*block+2]=b
   for col in range(3):di[3*block,col]=dn[col];di[3*block+1,col]=da[col];di[3*block+2,col]=db[col]
 pose=V7(wp.float64(0.0));dp=M3(wp.float64(0.0))
 for k in range(3):
  pose[k]=job.pose[k]+job.dt*velocity[k]
  for col in range(3):dp[k,col]=job.dt*dv[k,col]
 qw=job.pose[3];qx=job.pose[4];qy=job.pose[5];qz=job.pose[6]
 wx=velocity[3];wy=velocity[4];wz=velocity[5]
 rawq=V4(qw-job.dt*wp.float64(.5)*(qx*wx+qy*wy+qz*wz),qx+job.dt*wp.float64(.5)*(qw*wx-qz*wy+qy*wz),qy+job.dt*wp.float64(.5)*(qz*wx+qw*wy-qx*wz),qz+job.dt*wp.float64(.5)*(-qy*wx+qx*wy+qw*wz))
 draw=D4(wp.float64(0.0))
 for col in range(3):
  dx=dv[3,col];dy=dv[4,col];dz=dv[5,col]
  draw[0,col]=-job.dt*wp.float64(.5)*(qx*dx+qy*dy+qz*dz)
  draw[1,col]=job.dt*wp.float64(.5)*(qw*dx-qz*dy+qy*dz)
  draw[2,col]=job.dt*wp.float64(.5)*(qz*dx+qw*dy-qx*dz)
  draw[3,col]=job.dt*wp.float64(.5)*(-qy*dx+qx*dy+qw*dz)
 norm=wp.length(rawq);q=rawq/norm;dq=(draw-wp.outer(q,wp.transpose(draw)*q))/norm
 for k in range(4):pose[3+k]=q[k]
 return pose,dp,dq
@wp.func
def residual_jacobian(job:Job,f:V3,c:int):
 pose,dp,dq=prediction_jacobian(job,f,c)
 q=V4(pose[3],pose[4],pose[5],pose[6]);qd=V4(job.target[3],job.target[4],job.target[5],job.target[6]);qd=qd/wp.length(qd)
 axis0=V3(job.config[3],job.config[4],job.config[5]);up=V3(job.config[6],job.config[7],job.config[8]);qv=V3(q[1],q[2],q[3])
 helper=wp.cross(qv,axis0)+q[0]*axis0;axis=axis0+wp.float64(2.0)*wp.cross(qv,helper);daxis=M3(wp.float64(0.0))
 for col in range(3):
  dvec=V3(dq[1,col],dq[2,col],dq[3,col])
  da=wp.float64(2.0)*(wp.cross(dvec,helper)+wp.cross(qv,wp.cross(dvec,axis0)+dq[0,col]*axis0))
  for k in range(3):daxis[k,col]=da[k]
 pos=V3(pose[0]-job.target[0],pose[1]-job.target[1],pose[2]-job.target[2])
 if job.config[9]>wp.float64(0.0):
  planar=pos-up*wp.dot(pos,up);dplanar=dp-wp.outer(up,wp.transpose(dp)*up);length=wp.length(planar)
  if length>job.config[9]:
   factor=wp.float64(1.0)-job.config[9]/length
   dlength=(wp.transpose(dplanar)*planar)/length
   dp=dplanar*factor+wp.outer(planar,(job.config[9]/(length*length))*dlength);pos=planar*factor
  else:pos=V3(wp.float64(0.0));dp=M3(wp.float64(0.0))
 r=R16(wp.float64(0.0));jac=J16(wp.float64(0.0))
 for k in range(3):
  r[k]=wp.sqrt(job.config[0])*pos[k]
  for col in range(3):jac[k,col]=wp.sqrt(job.config[0])*dp[k,col]
 orientation=q-qd*wp.dot(q,qd);dori=dq-wp.outer(qd,wp.transpose(dq)*qd)
 for k in range(4):
  r[3+k]=wp.sqrt(job.config[1])*orientation[k]
  for col in range(3):jac[3+k,col]=wp.sqrt(job.config[1])*dori[k,col]
 upright=(axis-up)*(wp.float64(.5)*wp.sqrt(job.config[2]))
 for k in range(3):
  r[7+k]=upright[k]
  for col in range(3):jac[7+k,col]=wp.float64(.5)*wp.sqrt(job.config[2])*daxis[k,col]
 barrier=wp.float64(1.0)-wp.dot(axis,up)-job.config[10];r[10]=wp.sqrt(job.config[11])*wp.max(barrier,wp.float64(0.0))
 if barrier>wp.float64(0.0):
  db=-(wp.transpose(daxis)*up)*wp.sqrt(job.config[11])
  for col in range(3):jac[10,col]=db[col]
 for k in range(3):r[11+k]=wp.sqrt(job.fw)*f[k];jac[11+k,k]=wp.sqrt(job.fw)
 r[14]=wp.sqrt(job.tw)*f[1];r[15]=wp.sqrt(job.tw)*f[2];jac[14,1]=wp.sqrt(job.tw);jac[15,2]=wp.sqrt(job.tw)
 return r,pose,jac
@wp.kernel(enable_backward=False)
def analytic_probe(job:Job,jacobians:wp.array(dtype=J16)):
 c=wp.tid();r,pose,jac=residual_jacobian(job,job.force[c],c);job.residual[c,0]=r;job.poses[c]=pose;jacobians[c]=jac
@wp.kernel(enable_backward=False)
def fused_lm(job:Job,iterations:int,counts:wp.array(dtype=int),evaluations:wp.array(dtype=int),stops:wp.array(dtype=int)):
 c=wp.tid();force=V3(wp.float64(.9)*job.cap,wp.float64(0.0),wp.float64(0.0));damping=wp.float64(1e-3)
 r,pose,jac=residual_jacobian(job,force,c);gram=wp.transpose(jac)*jac;gradient=wp.transpose(jac)*r;cost=wp.dot(r,r)
 updates=int(0);evals=int(1);stop=int(0);refresh=bool(False)
 for iteration in range(iterations):
  if refresh:
   r,pose,jac=residual_jacobian(job,force,c);gram=wp.transpose(jac)*jac;gradient=wp.transpose(jac)*r;cost=wp.dot(r,r);evals=evals+1
  # Projected-gradient criterion includes active force/friction bounds.
  pg=force-project(job,force-gradient)
  if wp.length(pg)<wp.float64(1e-6)*(wp.float64(1.0)+wp.length(force)):
   stop=1;break
  matrix=gram
  for k in range(3):matrix[k,k]=matrix[k,k]+damping*wp.max(gram[k,k],wp.float64(1e-6))+wp.float64(1e-10)
  step=-(wp.inverse(matrix)*gradient)
  for k in range(3):step[k]=wp.clamp(step[k],wp.float64(-2.0),wp.float64(2.0))
  trial=project(job,force+step);rr,pp=residual(job,trial,c);newcost=wp.dot(rr,rr);evals=evals+1;updates=updates+1;refresh=bool(False)
  if newcost<cost:
   difference=cost-newcost;change=wp.length(trial-force);force=trial;r=rr;pose=pp;refresh=True
   damping=wp.max(damping*wp.float64(.5),wp.float64(1e-8))
   if difference<wp.float64(1e-10)*(wp.float64(1.0)+cost) and change<wp.float64(1e-6)*(wp.float64(1.0)+wp.length(force)):
    stop=2;break
   cost=newcost
  else:damping=wp.min(damping*wp.float64(4.0),wp.float64(1e8))
 job.force[c]=force;job.residual[c,0]=r;job.poses[c]=pose;job.damping[c]=damping;counts[c]=updates;evaluations[c]=evals;stops[c]=stop

class GPUExactBatchLM(Base):
 def __init__(self,opt,iterations=20):
  super().__init__(opt,iterations);self.stats_cache={};self.history=[];self.jac_checks=[];self.derivative_fn=None
 def buffers(self,count):
  if count in self.cache:return self.cache[count]
  j,old_graph=super().buffers(count)
  counts=wp.zeros(count,dtype=int,device=self.device);evals=wp.zeros(count,dtype=int,device=self.device);stops=wp.zeros(count,dtype=int,device=self.device)
  wp.launch(fused_lm,dim=count,inputs=[j,self.iterations,counts,evals,stops],device=self.device,block_dim=32);wp.synchronize_device(self.device)
  with wp.ScopedCapture(device=self.device) as capture:wp.launch(fused_lm,dim=count,inputs=[j,self.iterations,counts,evals,stops],device=self.device,block_dim=32)
  self.cache[count]=(j,capture.graph);self.stats_cache[count]=(counts,evals,stops)
  return j,capture.graph
 def solve(self,*args,**kwargs):
  result=super().solve(*args,**kwargs);ids,cost,forces,poses=result
  self.opt.last_solver_status='batch-lm-gpu-analytic'
  if self.validate:
   self.check_jacobian(args,kwargs,ids,forces)
  # Iteration diagnostics are read only during validation, not planning timings.
  if getattr(self,'collect_stats',False):
   counts,evals,stops=self.stats_cache[len(ids)]
   self.history.append(dict(candidates=len(ids),iterations=counts.numpy().tolist(),evaluations=evals.numpy().tolist(),stop_reason=stops.numpy().tolist()))
  return result
 def check_jacobian(self,args,kwargs,ids,forces):
  import casadi as cs
  opt=self.opt;fn=opt.force_objective_fn
  if self.derivative_fn is None:
   # Generate an independent analytic residual Jacobian from the original graph.
   symbols=[cs.MX.sym(fn.name_in(i),*fn.size_in(i)) for i in range(fn.n_in())];xp,cost=fn.call(symbols);cfg=symbols[fn.index_in('object_config')];xd=symbols[fn.index_in('x_d')];force=symbols[0]
   q=xp[3:7]/cs.norm_2(xp[3:7]);qd=xd[3:7]/cs.norm_2(xd[3:7]);axis=cfg[3:6]+2*cs.cross(q[1:],cs.cross(q[1:],cfg[3:6])+q[0]*cfg[3:6]);up=cfg[6:9];delta=xp[:3]-xd[:3];planar=delta-up*cs.dot(delta,up);length=cs.norm_2(planar)
   position=cs.if_else(cfg[9]>0,planar*cs.fmax(length-cfg[9],0)/cs.fmax(length,1e-12),delta)
   res=cs.vertcat(cs.sqrt(cfg[0])*position,cs.sqrt(cfg[1])*(q-qd*cs.dot(q,qd)),.5*cs.sqrt(cfg[2])*(axis-up),cs.sqrt(cfg[11])*cs.fmax(1-cs.dot(axis,up)-cfg[10],0),np.sqrt(opt.force_reg_coef)*force,np.sqrt(opt.friction_reg_coef)*force[1:])
   self.derivative_fn=cs.Function('reference_jac',symbols,[cs.jacobian(res,force)],fn.name_in(),['jac']).expand()
  j,_=self.buffers(len(ids));jacbuf=wp.zeros(len(ids),dtype=J16,device=self.device);wp.launch(analytic_probe,dim=len(ids),inputs=[j,jacbuf],device=self.device,block_dim=32);wp.synchronize_device(self.device);actual=jacbuf.numpy()
  xd,xc,tau,visible,vl,D=args
  common=dict(lam_arm=forces.T,x_d=xd,current_x=xc,v_last=vl,J_tilde=opt.J_tilde,D_inv=D,tau_o_np=tau,n_arm=opt.normal[ids].T,t1=opt.t1[ids].T,t2=opt.t2[ids].T,p_arm=opt.sample_point[ids].T,curr_ori_coef=1.,env_phi=opt.env_phi,scm_theta=opt.scm_theta,object_config=opt.object_cost_config(),force_limit=opt.max_contact_force)
  inputs=[]
  for name in fn.name_in():
   value=np.asarray(common[name]).reshape(fn.size_in(name)[0],-1,order='F');inputs.append(np.tile(value,(1,len(ids))) if value.shape[1]==fn.size_in(name)[1] else value)
  ref=np.asarray(self.derivative_fn.map(len(ids)).call(inputs)[0]);ref=np.stack([ref[:,3*i:3*(i+1)] for i in range(len(ids))])
  err=float(np.max(np.abs(actual-ref)));record=dict(candidates=len(ids),jac_max_abs=err,ref_max_abs=float(np.max(np.abs(ref))))
  assert np.allclose(actual,ref,atol=2e-7,rtol=2e-6),record
  self.jac_checks.append(record)
