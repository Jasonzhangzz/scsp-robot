"""GPU float64 batched projected LM for the exact nine-DoF coupled SCM.
All candidate predictions, finite-difference Jacobians and LM solves execute
in Warp CUDA kernels. Host work only prepares frozen geometry and input data.
"""
import time
from types import SimpleNamespace,MethodType
import numpy as np
import warp as wp
wp.init()
V3=wp.types.vector(length=3,dtype=wp.float64)
V4=wp.types.vector(length=4,dtype=wp.float64)
V7=wp.types.vector(length=7,dtype=wp.float64)
V9=wp.types.vector(length=9,dtype=wp.float64)
R16=wp.types.vector(length=16,dtype=wp.float64)
Imp=wp.types.vector(length=33,dtype=wp.float64)
M3=wp.types.matrix(shape=(3,3),dtype=wp.float64)
J16=wp.types.matrix(shape=(16,3),dtype=wp.float64)
@wp.struct
class Job:
 n:wp.array2d(dtype=V9)
 t1:wp.array2d(dtype=V9)
 t2:wp.array2d(dtype=V9)
 mn:wp.array2d(dtype=V9)
 mt1:wp.array2d(dtype=V9)
 mt2:wp.array2d(dtype=V9)
 coeff:wp.array3d(dtype=wp.float64)
 frames:wp.array2d(dtype=V3)
 free:wp.array(dtype=wp.float64)
 pose:wp.array(dtype=wp.float64)
 target:wp.array(dtype=wp.float64)
 config:wp.array(dtype=wp.float64)
 force:wp.array(dtype=V3)
 trial:wp.array(dtype=V3)
 residual:wp.array2d(dtype=R16)
 proposed:wp.array(dtype=R16)
 poses:wp.array(dtype=V7)
 proposed_pose:wp.array(dtype=V7)
 damping:wp.array(dtype=wp.float64)
 blocks:int
 dt:wp.float64
 stiffness:wp.float64
 mu:wp.float64
 cap:wp.float64
 fw:wp.float64
 tw:wp.float64
 eps:wp.float64
 pyramid:int
 comfree:int
 diagonal:int
 inverse:wp.array2d(dtype=wp.float64)
 points:wp.array(dtype=V3)
 contact_stiffness:wp.float64
 contact_damping:wp.float64
 wrench_scale:wp.float64
@wp.func
def project(job:Job,f:V3):
 fn=wp.clamp(f[0],wp.float64(1e-6),job.cap)
 bound=job.mu*fn
 a=f[1];b=f[2]
 if job.pyramid!=0:
  ratio=wp.min(wp.float64(1.0),bound/wp.max(wp.abs(a)+wp.abs(b),wp.float64(1e-12)))
  a=a*ratio;b=b*ratio
 else:
  a=wp.clamp(a,-bound,bound);b=wp.clamp(b,-bound,bound)
 return V3(fn,a,b)
@wp.func
def integrate_pose(job:Job,velocity:V9):
 pose=V7(wp.float64(0.0))
 for k in range(3):pose[k]=job.pose[k]+job.dt*velocity[k]
 qw=job.pose[3];qx=job.pose[4];qy=job.pose[5];qz=job.pose[6]
 wx=velocity[3];wy=velocity[4];wz=velocity[5]
 q=V4(qw-job.dt*wp.float64(.5)*(qx*wx+qy*wy+qz*wz),qx+job.dt*wp.float64(.5)*(qw*wx-qz*wy+qy*wz),qy+job.dt*wp.float64(.5)*(qz*wx+qw*wy-qx*wz),qz+job.dt*wp.float64(.5)*(-qy*wx+qx*wy+qw*wz))
 q=q/wp.length(q)
 for k in range(4):pose[3+k]=q[k]
 return pose
@wp.func
def predict_comfree(job:Job,f:V3,c:int):
 force=job.frames[c,0]*f[0]+job.frames[c,1]*f[1]+job.frames[c,2]*f[2]
 torque=wp.cross(job.points[c],force)
 wrench=V9(wp.float64(0.0))
 for k in range(3):
  wrench[k]=force[k];wrench[k+3]=torque[k]
 smooth=V9(wp.float64(0.0))
 for k in range(6):
  smooth[k]=job.free[k]
  for j in range(6):smooth[k]=smooth[k]+job.wrench_scale*job.inverse[k,j]*wrench[j]
 velocity=smooth
 for block in range(job.blocks-1):
  n=job.n[c,block];t1=job.t1[c,block];t2=job.t2[c,block]
  mn=job.mn[c,block];mt1=job.mt1[c,block];mt2=job.mt2[c,block]
  mu=job.coeff[c,block,0]
  for edge in range(4):
   row=n+mu*t1;massrow=mn+mu*mt1
   if edge==1:row=n+mu*t2;massrow=mn+mu*mt2
   if edge==2:row=n-mu*t1;massrow=mn-mu*mt1
   if edge==3:row=n-mu*t2;massrow=mn-mu*mt2
   speed=wp.dot(row,smooth)
   acceleration=-(job.contact_damping/job.dt)*speed-(job.contact_stiffness/job.dt)*(job.dt*speed+job.coeff[c,block,2])
   reaction=wp.max(job.coeff[c,block,1]*acceleration,wp.float64(0.0))
   velocity=velocity+job.dt*massrow*reaction
 return integrate_pose(job,velocity)
@wp.func
def predict_diagonal_contact(job:Job,f:V3,c:int):
 # Single ordered pass; coefficients/geometry are frozen for all LM trials.
 # Each division is by the corresponding physical diagonal mobility + r.
 velocity=V9(wp.float64(0.0))
 for k in range(6):velocity[k]=job.free[k]
 wrench=job.frames[c,0]*f[0]+job.frames[c,1]*f[1]+job.frames[c,2]*f[2]
 for k in range(3):velocity[6+k]=wrench[k]/(job.dt*job.stiffness)
 for block in range(job.blocks):
  n=job.n[c,block];mn=job.mn[c,block]
  normal=wp.max(-(wp.dot(n,velocity)+job.coeff[c,block,4])/job.coeff[c,block,1],wp.float64(0.0))
  velocity=velocity+mn*normal
  cap=job.coeff[c,block,0]*normal
  a=wp.clamp(-wp.dot(job.t1[c,block],velocity)/job.coeff[c,block,2],-cap,cap)
  velocity=velocity+job.mt1[c,block]*a
  b=wp.clamp(-wp.dot(job.t2[c,block],velocity)/job.coeff[c,block,3],-cap,cap)
  velocity=velocity+job.mt2[c,block]*b
 return integrate_pose(job,velocity)
@wp.func
def predict(job:Job,f:V3,c:int):
 if job.comfree==1:return predict_comfree(job,f,c)
 if job.diagonal==1:return predict_diagonal_contact(job,f,c)
 velocity=V9(wp.float64(0.0))
 for k in range(6):velocity[k]=job.free[k]
 wrench=job.frames[c,0]*f[0]+job.frames[c,1]*f[1]+job.frames[c,2]*f[2]
 for k in range(3):velocity[6+k]=wrench[k]/(job.dt*job.stiffness)
 impulse=Imp(wp.float64(0.0))
 for sweep in range(8):
  for block in range(job.blocks):
   n=job.n[c,block];t1=job.t1[c,block];t2=job.t2[c,block]
   mn=job.mn[c,block];mt1=job.mt1[c,block];mt2=job.mt2[c,block]
   free=velocity-mn*impulse[3*block]-mt1*impulse[3*block+1]-mt2*impulse[3*block+2]
   normal=wp.max(-(wp.dot(n,free)+job.coeff[c,block,4])/job.coeff[c,block,1],wp.float64(0.0))
   supported=free+mn*normal
   a=-wp.dot(t1,supported)/job.coeff[c,block,2]
   b=-wp.dot(t2,supported)/job.coeff[c,block,3]
   cap=job.coeff[c,block,0]*normal
   scale=cap/wp.sqrt(cap*cap+a*a+b*b+wp.float64(1e-12))
   a=a*scale;b=b*scale
   velocity=supported+mt1*a+mt2*b
   impulse[3*block]=normal;impulse[3*block+1]=a;impulse[3*block+2]=b
 pose=V7(wp.float64(0.0))
 for k in range(3):pose[k]=job.pose[k]+job.dt*velocity[k]
 qw=job.pose[3];qx=job.pose[4];qy=job.pose[5];qz=job.pose[6]
 wx=velocity[3];wy=velocity[4];wz=velocity[5]
 q=V4(qw-job.dt*wp.float64(.5)*(qx*wx+qy*wy+qz*wz),qx+job.dt*wp.float64(.5)*(qw*wx-qz*wy+qy*wz),qy+job.dt*wp.float64(.5)*(qz*wx+qw*wy-qx*wz),qz+job.dt*wp.float64(.5)*(-qy*wx+qx*wy+qw*wz))
 q=q/wp.length(q)
 for k in range(4):pose[3+k]=q[k]
 return pose
@wp.func
def residual(job:Job,f:V3,c:int):
 pose=predict(job,f,c)
 q=V4(pose[3],pose[4],pose[5],pose[6]);qd=V4(job.target[3],job.target[4],job.target[5],job.target[6]);qd=qd/wp.length(qd)
 axis0=V3(job.config[3],job.config[4],job.config[5]);up=V3(job.config[6],job.config[7],job.config[8]);qv=V3(q[1],q[2],q[3])
 axis=axis0+wp.float64(2.0)*wp.cross(qv,wp.cross(qv,axis0)+q[0]*axis0)
 pos=V3(pose[0]-job.target[0],pose[1]-job.target[1],pose[2]-job.target[2])
 if job.config[9]>wp.float64(0.0):
  planar=pos-up*wp.dot(pos,up);length=wp.length(planar)
  pos=planar*wp.max(length-job.config[9],wp.float64(0.0))/wp.max(length,wp.float64(1e-12))
 r=R16(wp.float64(0.0))
 for k in range(3):r[k]=wp.sqrt(job.config[0])*pos[k]
 orientation=q-qd*wp.dot(q,qd)
 for k in range(4):r[3+k]=wp.sqrt(job.config[1])*orientation[k]
 upright=(axis-up)*(wp.float64(.5)*wp.sqrt(job.config[2]))
 for k in range(3):r[7+k]=upright[k]
 r[10]=wp.sqrt(job.config[11])*wp.max(wp.float64(1.0)-wp.dot(axis,up)-job.config[10],wp.float64(0.0))
 for k in range(3):r[11+k]=wp.sqrt(job.fw)*f[k]
 r[14]=wp.sqrt(job.tw)*f[1];r[15]=wp.sqrt(job.tw)*f[2]
 return r,pose
@wp.kernel(enable_backward=False)
def initialize(job:Job):
 c=wp.tid();job.force[c]=V3(wp.float64(.9)*job.cap,wp.float64(0.0),wp.float64(0.0));job.damping[c]=wp.float64(1e-3)
@wp.kernel(enable_backward=False)
def derivatives(job:Job):
 c,pert=wp.tid();f=job.force[c]
 if pert>0:f[pert-1]=f[pert-1]+job.eps
 r,pose=residual(job,f,c);job.residual[c,pert]=r
 if pert==0:job.poses[c]=pose
@wp.kernel(enable_backward=False)
def update(job:Job):
 c=wp.tid();r=job.residual[c,0];j=J16(wp.float64(0.0))
 for k in range(3):
  d=(job.residual[c,k+1]-r)/job.eps
  for row in range(16):j[row,k]=d[row]
 gram=wp.transpose(j)*j;gradient=wp.transpose(j)*r
 for k in range(3):gram[k,k]=gram[k,k]+job.damping[c]*wp.max(gram[k,k],wp.float64(1e-6))+wp.float64(1e-10)
 step=-(wp.inverse(gram)*gradient)
 for k in range(3):step[k]=wp.clamp(step[k],wp.float64(-2.0),wp.float64(2.0))
 job.trial[c]=project(job,job.force[c]+step)
@wp.kernel(enable_backward=False)
def evaluate_trial(job:Job):
 c=wp.tid();r,pose=residual(job,job.trial[c],c);job.proposed[c]=r;job.proposed_pose[c]=pose
@wp.kernel(enable_backward=False)
def accept(job:Job):
 c=wp.tid();r=job.residual[c,0];rr=job.proposed[c];scale=wp.float64(4.0)
 if wp.dot(rr,rr)<wp.dot(r,r):
  job.force[c]=job.trial[c];job.residual[c,0]=rr;job.poses[c]=job.proposed_pose[c];scale=wp.float64(.5)
 job.damping[c]=wp.clamp(job.damping[c]*scale,wp.float64(1e-8),wp.float64(1e8))

class GPUExactBatchLM:
 def __init__(self,opt,iterations=20):
  if opt.max_contacts>10:raise ValueError('At most ten environment contact blocks')
  self.opt=opt;self.iterations=iterations;self.cache={};self.validations=[];self.calls=0
  self.device=wp.get_device('cuda:0')
  opt._precompile_optimization_function(build_solver=False)
 def buffers(self,count):
  if count in self.cache:return self.cache[count]
  blocks=self.opt.max_contacts+1;j=Job();dev=self.device
  for name in ['n','t1','t2','mn','mt1','mt2']:setattr(j,name,wp.zeros((count,blocks),dtype=V9,device=dev))
  j.coeff=wp.zeros((count,blocks,5),dtype=wp.float64,device=dev);j.frames=wp.zeros((count,3),dtype=V3,device=dev)
  for name,length in [('free',6),('pose',7),('target',7),('config',12)]:setattr(j,name,wp.zeros(length,dtype=wp.float64,device=dev))
  j.force=wp.zeros(count,dtype=V3,device=dev);j.trial=wp.zeros(count,dtype=V3,device=dev);j.residual=wp.zeros((count,4),dtype=R16,device=dev);j.proposed=wp.zeros(count,dtype=R16,device=dev)
  j.poses=wp.zeros(count,dtype=V7,device=dev);j.proposed_pose=wp.zeros(count,dtype=V7,device=dev);j.damping=wp.zeros(count,dtype=wp.float64,device=dev)
  j.blocks=blocks;j.dt=self.opt.h;j.stiffness=self.opt.robot_stiffness or 1.;j.mu=self.opt.mu_arm_obj;j.cap=self.opt.max_normal_force;j.fw=self.opt.force_reg_coef;j.tw=self.opt.friction_reg_coef;j.eps=1e-4;j.pyramid=int(self.opt.friction_cone=='pyramid')
  j.comfree=int(self.opt.contact_model=='comfree')
  j.diagonal=int(self.opt.contact_model=='diagonal-contact')
  j.inverse=wp.array(np.ascontiguousarray(self.opt.Q_inv),dtype=wp.float64,device=dev)
  j.points=wp.zeros(count,dtype=V3,device=dev)
  j.contact_stiffness=self.opt.K_contact;j.contact_damping=self.opt.comfree_damping;j.wrench_scale=self.opt.wrench_velocity_scale
  def run():
   wp.launch(initialize,dim=count,inputs=[j],device=dev)
   for _ in range(self.iterations):
    wp.launch(derivatives,dim=(count,4),inputs=[j],device=dev)
    wp.launch(update,dim=count,inputs=[j],device=dev)
    wp.launch(evaluate_trial,dim=count,inputs=[j],device=dev)
    wp.launch(accept,dim=count,inputs=[j],device=dev)
  run();wp.synchronize_device(dev)
  with wp.ScopedCapture(device=dev) as capture:run()
  self.cache[count]=(j,capture.graph)
  return j,capture.graph
 def prepare(self,j,ids,xd,xc,tau,vl):
  opt=self.opt;count=len(ids);blocks=opt.max_contacts+1
  rows=opt.J_tilde.reshape(opt.max_contacts,4,6);normal=rows.mean(1,keepdims=True)
  if opt.contact_model!='diagonal-contact':rows=normal+np.exp(opt.scm_theta[5])*(rows-normal)
  env=np.zeros((opt.max_contacts,4,9));env[:,:,:6]=rows
  geometry=np.zeros((count,blocks,4,9));geometry[:,:-1]=env
  points=opt.sample_point[ids];frames=np.stack([opt.normal[ids],opt.t1[ids],opt.t2[ids]],axis=1)
  rel=np.zeros((count,3,9));rel[:,:,:3]=np.eye(3);rel[:,:,6:]=-np.eye(3)
  rel[:,0,4]=points[:,2];rel[:,0,5]=-points[:,1];rel[:,1,3]=-points[:,2];rel[:,1,5]=points[:,0];rel[:,2,3]=points[:,1];rel[:,2,4]=-points[:,0]
  n,t1,t2=frames[:,0],frames[:,1],frames[:,2];mu=opt.mu_arm_obj
  pyramid=np.stack([n+mu*t1,n+mu*t2,n-mu*t1,n-mu*t2],axis=1)
  geometry[:,-1]=pyramid@rel
  n=geometry.mean(2);t1=(geometry[:,:,0]-geometry[:,:,2])/2;t2=(geometry[:,:,1]-geometry[:,:,3])/2
  mu=np.sqrt((t1[:,:,:3]**2).sum(2)+1e-20);den=np.maximum(mu,1e-10)[:,:,None];t1=t1/den;t2=t2/den
  inverse=np.zeros((9,9));inverse[:6,:6]=opt.Q_inv;inverse[6:,6:]=np.eye(3)/(opt.robot_stiffness or 1.)
  mn=n@inverse.T;mt1=t1@inverse.T;mt2=t2@inverse.T
  a,b=opt.soft_scm_a,opt.soft_scm_b;reg=opt.soft_scm_regularization
  if opt.contact_model=='diagonal-contact':a,b=1.,0.
  gaps=np.r_[opt.env_phi.reshape(opt.max_contacts,4).mean(1),0.];bias=np.where(gaps>0,gaps,(b/a)*gaps)/opt.h
  coefficients=np.stack([mu,((n*mn).sum(2)+reg)/a,((t1*mt1).sum(2)+reg)/a,((t2*mt2).sum(2)+reg)/a,np.broadcast_to(bias,(count,blocks))],axis=2)
  if opt.contact_model=='comfree':
   from models.comfree_contact import reduced_row_masses
   masses=reduced_row_masses(opt.J_tilde,opt.Q_inv,opt.env_phi).reshape(opt.max_contacts,4)
   coefficients[:,:-1,1]=masses[:,0]
   coefficients[:,:-1,2]=opt.env_phi.reshape(opt.max_contacts,4).mean(1)
  j.inverse.assign(np.ascontiguousarray(opt.Q_inv,dtype=np.float64));j.points.assign(np.ascontiguousarray(points,dtype=np.float64))
  for name,value in [('n',n),('t1',t1),('t2',t2),('mn',mn),('mt1',mt1),('mt2',mt2),('coeff',coefficients),('frames',frames),('free',np.asarray(vl)+opt.gravity_velocity_scale*(opt.Q_inv@tau)),('pose',xc),('target',xd),('config',opt.object_cost_config())]:getattr(j,name).assign(np.ascontiguousarray(value,dtype=np.float64))
 def solve(self,x_d,current_x,tau_o,visible_face_idx,v_last,D_inv):
  ids=np.asarray(visible_face_idx,dtype=np.int32).reshape(-1);opt=self.opt;j,graph=self.buffers(len(ids));self.prepare(j,ids,x_d,current_x,tau_o,v_last)
  wp.capture_launch(graph);wp.synchronize_device(self.device)
  forces=j.force.numpy();poses=j.poses.numpy();r=j.residual.numpy()[:,0];cost=(r*r).sum(1)
  # Model checks only in the explicit validation phase, never hidden in GPU timings.
  if getattr(self,'validate',False):
   count=len(ids);fn=opt.force_objective_fn;common=dict(lam_arm=forces.T,x_d=x_d,current_x=current_x,v_last=v_last,J_tilde=opt.J_tilde,D_inv=D_inv,tau_o_np=tau_o,n_arm=opt.normal[ids].T,t1=opt.t1[ids].T,t2=opt.t2[ids].T,p_arm=opt.sample_point[ids].T,curr_ori_coef=1.,env_phi=opt.env_phi,scm_theta=opt.scm_theta,object_config=opt.object_cost_config(),force_limit=opt.max_contact_force)
   inputs=[]
   for name in fn.name_in():
    value=np.asarray(common[name]).reshape(fn.size_in(name)[0],-1,order='F');inputs.append(np.tile(value,(1,count)) if value.shape[1]==fn.size_in(name)[1] else value)
   refpose,refcost=fn.map(count).call(inputs);refpose=np.asarray(refpose).T;refcost=np.asarray(refcost).reshape(-1)
   record=dict(candidates=count,pose_max_abs=float(np.max(np.abs(poses-refpose))),cost_max_abs=float(np.max(np.abs(cost-refcost))),force_norm_max=float(np.max(np.linalg.norm(forces,axis=1))))
   if opt.contact_model=='diagonal-contact':
    initial_inputs=list(inputs)
    initial_inputs[fn.name_in().index('lam_arm')]=np.tile([.9*opt.max_normal_force,0.,0.],(count,1)).T
    _,initial_cost=fn.map(count).call(initial_inputs);initial_cost=np.asarray(initial_cost).reshape(-1)
    record.update(initial_cost_mean=float(initial_cost.mean()),final_cost_mean=float(cost.mean()),
        strictly_improved_candidates=int(np.sum(cost<initial_cost-1e-9)),
        all_candidates_nonincreasing=bool(np.all(cost<=initial_cost+1e-8)))
   assert record['pose_max_abs']<1e-7,record
   assert np.allclose(cost,refcost,rtol=1e-7,atol=1e-8),record
   self.validations.append(record)
  assert np.isfinite(cost).all() and np.isfinite(poses).all()
  self.calls+=1;opt.last_solver_status='batch-lm-gpu-exact-'+opt.contact_model
  return ids,cost,forces,poses
 def install(self):
  opt=self.opt;opt.solver='batch-lm'
  def batch(instance,*args,**kwargs):return self.solve(*args,**kwargs)
  opt._choose_contact_points_batch=MethodType(batch,opt)
  def no_fallback(instance,**kwargs):raise RuntimeError('GPU batch LM failed: scalar fallback disabled')
  opt._solve_optimization=MethodType(no_fallback,opt)
