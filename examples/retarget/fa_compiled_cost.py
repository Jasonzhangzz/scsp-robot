"""Symbolic copy of the successful FA objective, including detached weights."""
import casadi as cs
import numpy as np
from examples.retarget.fa_compiled_model import native_min,native_max,native_clamp


def point_segment(p,a,b):
    d=b-a;t=native_clamp(cs.dot(p-a,d)/native_max(cs.dot(d,d),1e-12),0,1)
    return cs.sumsqr(p-a-t*d)


def segment_distance(a,b,c,d):
    left=b-a;right=d-c;off=a-c
    aa=cs.dot(left,left);bb=cs.dot(left,right);cc=cs.dot(right,right)
    dd=cs.dot(left,off);ee=cs.dot(right,off);den=aa*cc-bb*bb
    result=native_min(point_segment(a,c,d),point_segment(b,c,d))
    result=native_min(result,point_segment(c,a,b));result=native_min(result,point_segment(d,a,b))
    safe=cs.if_else(den>1e-8*aa*cc,den,1)
    s=(bb*ee-cc*dd)/safe;t=(aa*ee-bb*dd)/safe
    eligible=cs.logic_and(den>1e-8*aa*cc,cs.logic_and(cs.logic_and(s>=0,s<=1),cs.logic_and(t>=0,t<=1)))
    return cs.if_else(eligible,native_min(result,cs.sumsqr(off+s*left-t*right)),result)


class StageCost:
    def __init__(self,backend):
        job=backend.loss.terminal.job;lm=job.model;w=job.weights;m=backend.cpu_model if hasattr(backend,'cpu_model') else backend.objective.cpu_model
        self.g=lm.guidance_count;self.nb=m.nbody;self.no=lm.output_count;self.model=lm
        g=self.g;nb=self.nb;no=self.no
        # Query body/output selectors are parameters, so MANO mappings may
        # change between windows without compiling a different objective.
        self.sizes=[26,7,g,g,3*g,nb*g,no*g,3*g,3]
        self.np=sum(self.sizes)
        q=cs.MX.sym('q',m.nq);poses=cs.MX.sym('poses',12*nb);contact=cs.MX.sym('contact',14*backend.compiled.base.collision.contact_count)
        par=cs.MX.sym('target',self.np);i=0;items=[]
        for n in self.sizes:items.append(par[i:i+n]);i+=n
        action,obj,cmask,qmask,local,selector,output_selector,c,flags=items
        P=cs.reshape(poses,12,nb);select=cs.reshape(selector,nb,g);Q=P@select
        local=cs.reshape(local,3,g);c=cs.reshape(c,3,g)
        bp=lambda b:P[:3,b]
        br=lambda b:cs.reshape(P[3:,b],3,3)
        ob=lm.object_body;oq=lm.object_qpos
        thresholds=lm.contact_thresholds.numpy();rows=lm.output_contact_rows.numpy()
        outputs=[]
        for k in range(no):
            row=int(rows[k])
            outputs.append(cs.if_else(contact[14*row+13],contact[14*row],thresholds[k]) if row>=0 else float(thresholds[k]))
        outputs=cs.vertcat(*outputs);guide=cs.reshape(output_selector,no,g).T@outputs
        aw=np.r_[np.full(3,w.action_base_position),np.full(3,w.action_base_rotation),np.full(20,w.action_finger)]
        value=cs.sumsqr(cs.DM(aw)*(q[lm.hand_qpos.numpy()]-action))
        quat=q[oq+3:oq+7];target=obj[3:]
        quat=quat/(cs.norm_2(quat)+1e-8);target=target/(cs.norm_2(target)+1e-8)
        target*=cs.if_else(cs.dot(quat,target)<0,-1,1)
        rv=target[0]*quat[1:]-quat[0]*target[1:]-cs.cross(target[1:],quat[1:])
        angle=2*cs.atan2(cs.sqrt(cs.fmax(cs.sumsqr(rv),1e-12)),cs.dot(quat,target))
        value+=w.reference_pose_scale*(w.object_position**2*cs.sumsqr(q[oq:oq+3]-obj[:3])+w.object_rotation**2*angle**2)
        value*=flags[0]
        # Everything is expressed in the normalized units returned by _evaluate.
        for k in range(g):
            deficit=cs.fmax(guide[k]-w.target_phi,0)
            value+=flags[1]*w.contact_robustness*cmask[k]*deficit**2
            hand=Q[:3,k]+cs.reshape(Q[3:,k],3,3)@local[:,k]
            target=bp(ob)+br(ob)@c[:,k];error=hand-target
            distance=cs.sqrt(1e6*cs.sumsqr(error)+1e-12)
            sigmoid=1/(1+cs.exp(-(distance-w.marked_distance_mm)/max(w.marked_temperature_mm,1e-6)))
            # Upstream _prepare_auxiliary has enable_backward=False. Allowing
            # this derivative would silently change the successful FA gradient.
            marked=cs.stop_diff(sigmoid,1)
            value+=flags[2]*w.marked_query*qmask[k]*marked*cs.sumsqr(error)
        value+=w.soft_penetration*cs.sumsqr(cs.fmax(-outputs-w.soft_penetration_limit,0))
        value+=w.penetration*cs.sumsqr(cs.fmax(-outputs-w.penetration_limit,0))
        if w.finger_joint_collision>0:
            body=lm.finger_joint_body.numpy();start=lm.finger_joint_start.numpy();end=lm.finger_joint_end.numpy();radius=lm.finger_joint_radius.numpy()
            point=lambda j,a:bp(int(body[j]))+br(int(body[j]))@cs.DM(a[j])
            # Share one compact primitive across all capsule pairs. Unrolling
            # every segment-distance branch into the stage adjoint creates
            # megabytes of duplicate C and very slow compiler optimization.
            endpoints=cs.SX.sym('endpoints',12);rs=cs.SX.sym('radius_sum')
            squared=segment_distance(endpoints[:3],endpoints[3:6],endpoints[6:9],endpoints[9:12])
            gap=cs.sqrt(squared+1e-12)-rs
            capsule=cs.Function('fa_capsule_penalty',[endpoints,rs],
                [w.finger_joint_collision*cs.fmax(w.finger_joint_surface_margin_mm*.001-gap,0)**2],{'never_inline':True})
            for left,right in zip(lm.finger_joint_pair_left.numpy(),lm.finger_joint_pair_right.numpy()):
                endpoints=cs.vertcat(point(left,start),point(left,end),point(right,start),point(right,end))
                value+=capsule(endpoints,float(radius[left]+radius[right]))
        self.function=cs.Function('fa_stage_cost',[q,poses,contact,par],[value])

    def parameters(self,workspace,points=None,world=0):
        t=workspace.target;g=self.g
        selector=np.zeros((self.nb,g));selector[t.query_body.numpy(),np.arange(g)]=1
        outputs=np.zeros((self.no,g));outputs[t.contact_output.numpy(),np.arange(g)]=1
        c=t.query_position_object.numpy() if points is None else points[world]
        return np.r_[t.action.numpy(),t.object_qpos.numpy(),t.contact_mask.numpy()*t.contact_weight.numpy(),
                     t.query_mask.numpy()*t.query_weight.numpy(),t.query_local.numpy().T.ravel(order='F'),
                     selector.ravel(order='F'),outputs.ravel(order='F'),c.T.ravel(order='F'),
                     workspace.job.reference_scale,workspace.job.guidance_enabled,workspace.job.query_enabled]
