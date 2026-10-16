"""Code-generated FA ComFree equations, with analytic CasADi derivatives.

Retains articulated inertia, bias/servo forces, real mesh contacts, joint
limits, the configured coupled contact corrections and Euler damping. The
discrete mesh-feature search is native C++, not a Python/Warp callback.
"""
from pathlib import Path
import hashlib
import subprocess
import os
import casadi as cs
import numpy as np


def unit(x):
    return x/cs.sqrt(cs.dot(x,x))


def native_min(a,b):
    """Warp routes a tie to its second operand, whereas CasADi fmin splits it."""
    return cs.if_else(a<b,a,b)


def native_max(a,b):
    return cs.if_else(a>b,a,b)


def native_clamp(x,lower,upper):
    return native_min(native_max(lower,x),upper)


def quatmul(a,b):
    return cs.vertcat(a[0]*b[0]-cs.dot(a[1:],b[1:]),a[0]*b[1:]+b[0]*a[1:]+cs.cross(a[1:],b[1:]))


def rotation(q):
    w,x,y,z=cs.vertsplit(q)
    return cs.vertcat(cs.horzcat(1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)),
                     cs.horzcat(2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)),
                     cs.horzcat(2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)))


def skew(v):
    return cs.vertcat(cs.horzcat(0,-v[2],v[1]),cs.horzcat(v[2],0,-v[0]),cs.horzcat(-v[1],v[0],0))


def motion_cross(a,b):
    return cs.vertcat(cs.cross(a[:3],b[:3]),cs.cross(a[3:],b[:3])+cs.cross(a[:3],b[3:]))


def force_cross(a,b):
    return cs.vertcat(cs.cross(a[:3],b[:3])+cs.cross(a[3:],b[3:]),cs.cross(a[:3],b[3:]))


def arr(struct,name):
    return getattr(struct,name).numpy()


def constant_sum(values,shape):
    return sum(values,cs.DM.zeros(*shape))


def subtree_sums(model,values):
    result=list(values)
    for b in range(model.nbody-1,0,-1):
        parent=int(model.body_parentid[b]);result[parent]=result[parent]+result[b]
    return result


def kinematics(model,q):
    p=[];r=[];quats=[];anchors=[None]*model.njnt;axes=[None]*model.njnt
    for b in range(model.nbody):
        parent=int(model.body_parentid[b]);pos=cs.DM(model.body_pos[b]);quat=cs.DM(model.body_quat[b])
        if b:
            pos=p[parent]+r[parent]@pos;quat=quatmul(quats[parent],quat)
        for j in range(int(model.body_jntadr[b]),int(model.body_jntadr[b]+model.body_jntnum[b])):
            address=int(model.jnt_qposadr[j]);kind=int(model.jnt_type[j]);rot=rotation(quat)
            local=cs.DM(model.jnt_pos[j]);axis=cs.DM(model.jnt_axis[j]);anchor=pos+rot@local;worldaxis=rot@axis
            if kind==0:
                pos=q[address:address+3];quat=unit(q[address+3:address+7]);anchor=pos;worldaxis=axis
            elif kind==2:
                pos+=worldaxis*(q[address]-model.qpos0[address])
            elif kind==3:
                half=(q[address]-model.qpos0[address])/2
                quat=quatmul(quat,cs.vertcat(cs.cos(half),axis*cs.sin(half)))
                pos=anchor-rotation(quat)@local
            else:
                quat=quatmul(quat,unit(q[address:address+4]));pos=anchor-rotation(quat)@local
            anchors[j]=anchor;axes[j]=worldaxis
        quat=unit(quat);p.append(pos);quats.append(quat);r.append(rotation(quat))
    return p,r,anchors,axes


def smooth_function(compiled,model,kinematic_fn):
    # Fixed-size scalar code avoids the many small vector-copy loops emitted
    # by MX. The MX variant is retained for numerical/performance comparison.
    Kind=cs.SX if os.environ.get('FA_COMPILED_SCALAR_MATH','1')=='1' else cs.MX
    q=Kind.sym('q',model.nq);v=Kind.sym('v',model.nv);u=Kind.sym('u',model.nu)
    pose,anchor_matrix,axis_matrix=kinematic_fn(q)
    p=[pose[12*b:12*b+3] for b in range(model.nbody)]
    r=[cs.reshape(pose[12*b+3:12*b+12],3,3) for b in range(model.nbody)]
    anchors=[anchor_matrix[:,j] for j in range(model.njnt)];axes=[axis_matrix[:,j] for j in range(model.njnt)]
    dm=compiled.base.dynamics.model
    descendants=arr(dm,'body_descendant');bodydof=arr(dm,'body_dof');pred=arr(dm,'dof_predecessor')
    roots=arr(dm,'body_rootid');dofbody=arr(dm,'dof_bodyid');dofjoint=arr(dm,'dof_jntid')
    mass=arr(dm,'body_mass')[0];ipos=arr(dm,'body_ipos')[0];iquat=arr(dm,'body_iquat')[0];inertia=arr(dm,'body_inertia')[0]
    inertial=[p[b]+r[b]@cs.DM(ipos[b]) for b in range(model.nbody)]
    com=[]
    for b in range(model.nbody):
        ids=np.flatnonzero(descendants[b]);total=float(mass[ids].sum())
        com.append(sum((mass[c]*inertial[c] for c in ids),cs.DM.zeros(3))/total if total>1e-12 else inertial[b])
    spatial=[]
    for b in range(model.nbody):
        rot=r[b]@rotation(cs.DM(iquat[b]));off=inertial[b]-com[roots[b]]
        I=rot@cs.diag(cs.DM(inertia[b]))@rot.T+mass[b]*(cs.dot(off,off)*cs.DM.eye(3)-off@off.T)
        h=skew(mass[b]*off)
        spatial.append(cs.blockcat([[I,h],[-h,mass[b]*cs.DM.eye(3)]]))
    cdof=[]
    for d in range(model.nv):
        j=int(dofjoint[d]);b=int(dofbody[d]);kind=int(model.jnt_type[j]);offset=d-int(model.jnt_dofadr[j])
        angular=cs.DM.zeros(3);linear=cs.DM.zeros(3)
        if kind==2:linear=axes[j]
        elif kind==0 and offset<3:linear=cs.DM.eye(3)[:,offset]
        else:
            angular=axes[j] if kind==3 else r[b][:,offset-(3 if kind==0 else 0)]
            linear=cs.cross(angular,com[roots[b]]-anchors[j])
        cdof.append(cs.vertcat(angular,linear))
    recursive=os.environ.get('FA_COMPILED_RECURSIVE_SUMS','1')=='1'
    composite=(subtree_sums(model,spatial) if recursive else
        [constant_sum([spatial[c] for c in np.flatnonzero(descendants[b])],(6,6)) for b in range(model.nbody)])
    crb=[composite[dofbody[d]]@cdof[d] for d in range(model.nv)]
    mask=arr(dm,'mass_mask');armature=arr(dm,'dof_armature')[0]
    rows=[];cols=[];entries=[]
    # Construct CSC once. Repeated MX submatrix assignment otherwise creates
    # hundreds of matrix copies in both the forward and reverse generated C.
    for b in range(model.nv):
        for a in range(model.nv):
            if mask[a,b] or a==b:
                value=cs.dot(crb[a],cdof[b]) if mask[a,b]==1 else cs.dot(crb[b],cdof[a]) if mask[a,b]==2 else 0
                if a==b:value+=armature[a]
                rows.append(a);cols.append(b);entries.append(value)
    M=Kind(cs.Sparsity.triplet(model.nv,model.nv,rows,cols),cs.vertcat(*entries))
    def body_accumulate(terms):
        result=[]
        for b in range(model.nbody):
            start=int(model.body_dofadr[b]);count=int(model.body_dofnum[b])
            local=constant_sum([terms[d]*v[d] for d in range(start,start+count)],(6,1))
            result.append(local+(result[int(model.body_parentid[b])] if b else cs.DM.zeros(6)))
        return result
    cv=body_accumulate(cdof) if recursive else [constant_sum([cdof[d]*v[d] for d in np.flatnonzero(bodydof[b])],(6,1)) for b in range(model.nbody)]
    dv=[motion_cross(constant_sum([cdof[k]*v[k] for k in np.flatnonzero(pred[d])],(6,1)),cdof[d]) for d in range(model.nv)]
    gravity=cs.DM(arr(dm,'gravity')[0])*float(dm.gravity_enabled)
    body_bias=body_accumulate(dv) if recursive else [constant_sum([dv[d]*v[d] for d in np.flatnonzero(bodydof[b])],(6,1)) for b in range(model.nbody)]
    forces=[];gravforces=[];gravcomp=arr(dm,'body_gravcomp')[0]
    for b in range(model.nbody):
        acc=body_bias[b]+cs.vertcat(cs.DM.zeros(3),-gravity)
        forces.append(spatial[b]@acc+force_cross(cv[b],spatial[b]@cv[b]))
        linear=-gravity*mass[b]*gravcomp[b]
        gravforces.append(cs.vertcat(cs.cross(inertial[b]-com[roots[b]],linear),linear))
    force=[]
    combined=subtree_sums(model,[g-f for g,f in zip(gravforces,forces)]) if recursive else None
    damping=arr(dm,'dof_damping')[0]
    for d in range(model.nv):
        ids=np.flatnonzero(descendants[dofbody[d]])
        subtree=combined[dofbody[d]] if recursive else constant_sum([gravforces[b]-forces[b] for b in ids],(6,1))
        force.append(cs.dot(cdof[d],subtree)-float(dm.damper_enabled)*damping[d]*v[d])
    gain=arr(dm,'actuator_gainprm')[0];bias=arr(dm,'actuator_biasprm')[0];gear=arr(dm,'actuator_gear')[0]
    da=arr(dm,'actuator_dofid');qa=arr(dm,'actuator_qposadr');cl=arr(dm,'actuator_ctrllimited');fl=arr(dm,'actuator_forcelimited')
    urange=arr(dm,'actuator_ctrlrange')[0];frange=arr(dm,'actuator_forcerange')[0]
    if dm.actuation_enabled:
        for a in range(model.nu):
            command=u[a]
            if dm.clamp_control and cl[a]:command=native_clamp(command,urange[a,0],urange[a,1])
            f=gain[a,0]*command+bias[a,0]+bias[a,1]*q[qa[a]]*gear[a,0]+bias[a,2]*v[da[a]]*gear[a,0]
            if fl[a]:f=native_clamp(f,frange[a,0],frange[a,1])
            force[da[a]]+=gear[a,0]*f
    poses=cs.vertcat(*[cs.vertcat(p[b],cs.reshape(r[b],9,1)) for b in range(model.nbody)])
    return cs.Function('fa_smooth',[q,v,u],[M,cs.vertcat(*force),poses,cs.horzcat(*anchors),cs.horzcat(*axes)])


def impedance(solimp,position):
    minimum=np.clip(solimp[0],.0001,.9999);maximum=np.clip(solimp[1],.0001,.9999)
    middle=np.clip(solimp[3],.0001,.9999);power=max(float(solimp[4]),1.);width=max(float(solimp[2]),1e-15)
    x=cs.fabs(position)/width
    # Clamp the unselected base before noninteger powers, keeping it finite.
    curve=cs.if_else(x<middle,x**power/middle**(power-1),1-cs.fmax(1-x,0)**power/(1-middle)**(power-1))
    return cs.if_else(x>1,maximum,cs.fmin(cs.fmax(minimum+curve*(maximum-minimum),minimum),maximum))


def compile_function(function,directory,dependencies=()):
    digest=hashlib.sha256((function.serialize()+''.join(str(p) for p in dependencies)).encode()).hexdigest()[:16]
    directory=Path(directory)/digest;directory.mkdir(parents=True,exist_ok=True)
    source=directory/(function.name()+'.c');library=directory/(function.name()+'.so')
    if not library.exists():
        import os
        # CodeGenerator's file name is relative; its output prefix is absolute.
        generator=cs.CodeGenerator(source.name);generator.add(function);generator.generate(str(directory)+'/')
        command=['cc','-O3','-fPIC','-shared',str(source),*[str(p) for p in dependencies],'-lm','-o',str(library)]
        if dependencies:command += ['-Wl,-rpath,'+':'.join(str(Path(p).parent) for p in dependencies)]
        subprocess.run(command,check=True)
    return cs.external(function.name(),str(library),{'enable_fd':False})


class CompiledFAModel:
    def __init__(self,compiled,model,directory):
        from examples.retarget.fa_compiled_geometry import build_feature_search
        self.model=model;self.compiled=compiled;self.directory=Path(directory)
        features,slot_bodies,batch,self.geometry_library=build_feature_search(compiled,model,directory)
        Kind=cs.SX if os.environ.get('FA_COMPILED_SCALAR_MATH','1')=='1' else cs.MX
        kq=Kind.sym('kinematic_q',model.nq)
        kp,kr,ka,kaxis=kinematics(model,kq)
        kposes=cs.vertcat(*[cs.vertcat(kp[b],cs.reshape(kr[b],9,1)) for b in range(model.nbody)])
        self.kinematics=cs.Function('fa_kinematics',[kq],[kposes,cs.horzcat(*ka),cs.horzcat(*kaxis)])
        self.smooth=smooth_function(compiled,model,self.kinematics)
        from examples.retarget.fa_compiled_spd import spd_solver
        solve,self.spd_library=spd_solver(model.nv,directory)
        q=cs.MX.sym('q',model.nq);v=cs.MX.sym('v',model.nv);u=cs.MX.sym('u',model.nu)
        M,force,poses,anchors,axes=self.smooth(q,v,u)
        # Nearest feature/source identities are discrete. All continuous
        # selected-feature geometry below remains differentiable in q.
        selected=cs.MX.sym('selected_features',13,compiled.base.collision.contact_count)
        rp=compiled.base.contact_rows.parameters;ep=compiled.base.contact_solver.parameters
        geom=arr(rp,'contact_geom');bodyids=arr(rp,'geom_bodyid');margins=arr(rp,'contact_includemargin')
        signs=compiled.base.collision.contact_layout.contact_normal_sign.numpy()
        offset=float(batch.model.distance_offset.numpy()[0]);obj=batch.model.target_body_id
        positions=[poses[12*b:12*b+3] for b in range(model.nbody)]
        rotations=[cs.reshape(poses[12*b+3:12*b+12],3,3) for b in range(model.nbody)]
        phi=[];points=[];frames=[];enabled=[]
        norm=lambda x:x/cs.sqrt(cs.dot(x,x)+1e-18)
        import mujoco
        plane=batch.model.plane_geom_id;plane_R=np.empty(9);mujoco.mju_quat2Mat(plane_R,model.geom_quat[plane])
        plane_n=cs.DM(plane_R.reshape(3,3)[:,2]);plane_p=cs.DM(model.geom_pos[plane])
        for c in range(compiled.base.collision.contact_count):
            f=selected[:,c];present=f[12]
            if c<batch.model.target_contact_offset:
                center=positions[obj]+rotations[obj]@f[:3]
                distance=cs.dot(center-plane_p,plane_n)-f[3]+offset
                normal=plane_n*signs[c];point=center-plane_n*f[3]+plane_n*offset
            else:
                b=slot_bodies[c-batch.model.target_contact_offset]
                source=positions[b]+rotations[b]@f[:3];anchor=positions[obj]+rotations[obj]@f[4:7]
                axis=rotations[obj]@f[8:11];kind=cs.fabs(f[11]);side=cs.if_else(f[11]<0,-1,1)
                closest=cs.if_else(kind==2,anchor+axis*cs.dot(axis,source-anchor),anchor)
                normal=norm(closest-source)*side;distance=side*cs.norm_2(closest-source)
                height=cs.dot(axis,source-anchor)
                closest=cs.if_else(kind==3,source-axis*height,closest)
                normal=cs.if_else(kind==3,-axis,normal);distance=cs.if_else(kind==3,height,distance)
                distance=distance-f[3]-f[7]+offset;point=closest-normal*f[7]+normal*offset
                normal*=signs[c]
            n=norm(normal);tx=norm(cs.cross(n,cs.DM.ones(3)));ty=norm(cs.cross(n,tx))
            frame=cs.horzcat(n,tx,ty).T
            phi.append(cs.if_else(present,distance,0));points.append(cs.if_else(present,point,cs.MX.zeros(3)))
            frames.append(cs.if_else(present,frame,cs.MX.zeros(3,3)))
            enabled.append(cs.logic_and(present,distance<margins[c]))
        dofj=arr(rp,'dof_jntid');mask=arr(rp,'body_dof_mask');row_contact=arr(rp,'row_contact');edge=arr(rp,'row_edge')
        friction=arr(rp,'contact_friction')*float(arr(rp,'friction_scale')[0]);nr=compiled.base.contact_rows.row_count
        def point_dof(b,d,point):
            if not mask[b,d]:return cs.MX.zeros(3),cs.MX.zeros(3)
            j=int(dofj[d]);kind=int(model.jnt_type[j]);delta=d-int(model.jnt_dofadr[j])
            if kind==2:return axes[:,j],cs.MX.zeros(3)
            if kind==0 and delta<3:return cs.DM.eye(3)[:,delta],cs.MX.zeros(3)
            axis=axes[:,j] if kind==3 else rotations[model.jnt_bodyid[j]][:,delta-(3 if kind==0 else 0)]
            return cs.cross(axis,point-anchors[:,j]),axis
        contact_j=[]
        for c in range(len(phi)):
            b1,b2=bodyids[geom[c]];linear=[];angular=[]
            for d in range(model.nv):
                l1,a1=point_dof(b1,d,points[c]);l2,a2=point_dof(b2,d,points[c]);linear.append(l2-l1);angular.append(a2-a1)
            contact_j.append((cs.horzcat(*linear),cs.horzcat(*angular)))
        J=[];gap=[];active=[];solrefs=[];solimps=[];invweights=[]
        bw=arr(ep,'body_invweight')[0];ratio=float(arr(ep,'impratio_invsqrt')[0])
        for row in range(compiled.base.contact_rows.contact_row_count):
            c=int(row_contact[row]);e=int(edge[row]);axis=e//2+1;sign=1 if e%2==0 else -1
            lin,ang=contact_j[c];fr=frames[c]
            value=fr[0,:]@lin+sign*friction[c,axis-1]*(fr[axis,:]@lin if axis<3 else fr[axis-3,:]@ang)
            J.append(cs.if_else(enabled[c],value,cs.MX.zeros(1,model.nv)));gap.append(phi[c]-margins[c]);active.append(enabled[c])
            solrefs.append(arr(ep,'contact_solref')[c]);solimps.append(arr(ep,'contact_solimp')[c])
            b1,b2=bodyids[geom[c]];mu=friction[c,0]
            invweights.append((bw[b1,0]+bw[b2,0])*(1+mu*mu)*(2*mu*mu*ratio*ratio))
        for j in arr(rp,'limit_joint'):
            qa=int(model.jnt_qposadr[j]);d=int(model.jnt_dofadr[j]);low,high=arr(rp,'jnt_range')[0,j]
            g=cs.fmin(q[qa]-low,high-q[qa])-arr(rp,'jnt_margin')[0,j];en=g<0
            jr=cs.MX.zeros(1,model.nv);jr[0,d]=cs.if_else(en,cs.if_else(q[qa]-low<high-q[qa],1,-1),0)
            J.append(jr);gap.append(g);active.append(en);solrefs.append(arr(ep,'jnt_solref')[0,j]);solimps.append(arr(ep,'jnt_solimp')[0,j])
            invweights.append(arr(ep,'dof_invweight')[0,d])
        J=cs.vertcat(*J);gap=cs.vertcat(*gap);active=cs.vertcat(*active)
        h=float(arr(ep,'timestep')[0]);smooth_acc=solve(M,force);jv=J@v;ja=J@smooth_acc
        free=[];weights=[]
        for row in range(nr):
            sr=solrefs[row];si=solimps[row];dmax=np.clip(si[1],.0001,.9999);tc=float(sr[0])
            if ep.l0_mode:tc/=ep.l0_ref_gain
            elif not ep.refsafe_disabled:tc=max(tc,2*h)
            k=-float(sr[0])/dmax**2 if sr[0]<=0 else 1/(dmax**2*tc**2*float(sr[1])**2)
            b=-float(sr[1])/dmax if sr[1]<=0 else 2/(dmax*tc)
            imp=impedance(si,gap[row]);D=1/cs.fmax(invweights[row]*(1-imp)/imp,1e-15)
            aref=-k*imp*gap[row]-b*jv[row]
            free.append(cs.if_else(active[row],-D*(ja[row]-aref),0));weights.append(cs.if_else(active[row],D,0))
        free=cs.vertcat(*free);D=cs.vertcat(*weights);response=cs.MX.zeros(nr)
        iterations=int(compiled.contact.iterations)
        iteration_raw=[];iteration_slopes=[]
        if iterations:
            for _ in range(iterations):
                iteration_raw.append(free-D*response)
                slope=cs.if_else(free-D*response>0,1,0)
                iteration_slopes.append(slope)
                linear_free=slope*free;weight=slope*D
                correction=solve(M+J.T@cs.diag(weight)@J,J.T@linear_free)
                response=J@correction
            fc=cs.fmax(linear_free-weight*response,0)
        else:
            from comfree_warp.comfree_core._src.types import L0_IMPEDANCE
            fc=cs.fmax(float(L0_IMPEDANCE)*free,0)
        total=force+J.T@fc;qacc=solve(M,total)
        integration=compiled.base.integration
        accel=solve(M+h*cs.diag(cs.DM(arr(integration.model,'dof_damping')[0])),total) if integration.implicit_damping else qacc
        vn=v+h*accel;qn=cs.MX.zeros(model.nq)
        for j in range(model.njnt):
            qa=int(model.jnt_qposadr[j]);d=int(model.jnt_dofadr[j]);kind=int(model.jnt_type[j])
            if kind==0:qn[qa:qa+3]=q[qa:qa+3]+h*vn[d:d+3]
            if kind in (0,1):
                a=qa+3 if kind==0 else qa;k=d+3 if kind==0 else d;omega=vn[k:k+3]
                squared=cs.dot(omega,omega);speed=cs.sqrt(cs.fmax(squared,1e-30));half=.5*h*speed
                factor=cs.if_else(squared==0,.5*h,cs.sin(half)/speed)
                delta=cs.vertcat(cs.if_else(squared==0,1,cs.cos(half)),factor*omega)
                qn[a:a+4]=unit(quatmul(unit(q[a:a+4]),delta))
            else:qn[qa]=q[qa]+h*vn[d]
        contact=cs.vertcat(*[cs.vertcat(phi[c],points[c],cs.reshape(frames[c],9,1),selected[12,c]) for c in range(len(phi))])
        core=[qn,vn,qacc,poses,contact]
        self.step_features=cs.Function('fa_equations_features',[q,v,u,selected],core)
        # Selection is saved by the rollout and supplied to its adjoint. This
        # matches the native tape and avoids repeating BVH search in reverse.
        kinematic_poses=self.kinematics(q)[0]
        selection=features(cs.stop_diff(kinematic_poses,1))
        diagnostic=cs.Function('fa_equations_diagnostic',[q,v,u,selected],[*core,M,force,J,free,D,fc])
        self.step=cs.Function('fa_equations',[q,v,u],diagnostic.call([q,v,u,selection],True,False))
        if iterations:
            debug=cs.Function('fa_contact_debug_features',[q,v,u,selected],[cs.horzcat(*iteration_raw),cs.horzcat(*iteration_slopes),linear_free-weight*response])
            self.contact_debug=cs.Function('fa_contact_debug',[q,v,u],debug.call([q,v,u,selection],True,False))
        observed_contact=cs.graph_substitute(contact,[poses],[kinematic_poses])
        self.observe_features=cs.Function('fa_observation_features',[q,selected],[kinematic_poses,observed_contact])
        selection_q=features(cs.stop_diff(kinematic_poses,1))
        self.observe=cs.Function('fa_observation',[q],self.observe_features.call([q,selection_q],True,False))
        self.select=cs.Function('fa_selection',[q],[selection_q])
        self.selected=features;self.slot_bodies=slot_bodies
        self.step_compiled=None

    def compile_step(self):
        if self.step_compiled is None:self.step_compiled=compile_function(self.step,self.directory,[self.geometry_library,self.spd_library])
        return self.step_compiled
