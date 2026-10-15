"""Measured material contacts, per-region nearest projections and SCM rows."""
import mujoco
import numpy as np
from models.wuji_hand import rotation,normalize
from planning.contact_friction import native_tangents


def native_grasp_certificate(hand,materials,active,data=None):
    """Certify only physical hand-object patches in the supplied state."""
    from planning.grasp_verification import verify_grasp
    data=hand.data if data is None else data
    _,actual=hand.observed_contacts(materials,active,data=data)
    actual=[c for c in actual if c['normal']@c['force']>=.01]
    pose=data.qpos[hand.oq:hand.oq+7];r=rotation(pose)
    gravity=hand.model.body_mass[hand.obj]*hand.model.opt.gravity
    gravity_moment=np.cross(data.xipos[hand.obj]-pose[:3],gravity)
    return verify_grasp(np.array([(c['point']-pose[:3])@r for c in actual]).reshape(-1,3),
        np.array([c['normal']@r for c in actual]).reshape(-1,3),hand.friction,edges=4,
        tangent_directions=np.array([c.get('tangents',native_tangents(c['normal']))@r for c in actual]).reshape(-1,2,3),
        external_wrench=np.r_[r.T@gravity,r.T@gravity_moment],
        normal_force_limits=np.array([c['normal']@c['force'] for c in actual]))


def frame(normal):
    normal=normalize(normal)
    tangent,second=native_tangents(normal)
    return normal,tangent,second


def contact_geometry(hand,materials,active,capacity,*,return_friction=False):
    n=len(materials);nu=hand.nu;mu=hand.friction;pose=hand.object_pose();r=rotation(pose)
    points,normals,jp=hand.kinematics(materials)
    local=(points-pose[:3])@r
    projected,gaps,inward=hand.surface.query(local) if n else (np.empty((0,3)),np.empty(0),np.empty((0,3)))
    phi=np.ones(4*capacity);jac=np.zeros((4*capacity,6+nu));stick=np.zeros(n)
    coefficients=np.full(capacity,mu);coefficients[n]=hand.floor_friction
    _,actual=hand.observed_contacts(materials,active)
    for i,m in enumerate(materials):
        matching=[c for c in actual if c['slot']==i]
        # Physical contacts survive reference release or a failed Lambda solve.
        # Only truly absent AND unrequested slots have zero contact rows.
        if not active[i] and not matching:continue
        point=pose[:3]+r@projected[i];normal=r@inward[i];gap=gaps[i];contact_jp=jp[i]
        if matching:
            weights=np.array([max(c['normal']@c['force'],1e-9) for c in matching])
            point=np.average([c['point'] for c in matching],axis=0,weights=weights)
            # Keep the dominant physical patch's native pyramid orientation.
            # Averaging normals and inventing tangent axes changes its cone.
            dominant=matching[int(np.argmax(weights))]
            normal=dominant['normal']
            stick[i]=1.;gap=min(c['gap'] for c in matching)
            jactual,jr=np.zeros((3,hand.model.nv)),np.zeros((3,hand.model.nv))
            mujoco.mj_jac(hand.model,hand.data,jactual,jr,point,m.body)
            contact_jp=jactual[:,hand.dadr]
        normal,t1,t2=frame(normal)
        if matching:
            t1,t2=native_tangents(normal,dominant.get('tangents'))
        for k,direction in enumerate((normal+mu*t1,normal+mu*t2,normal-mu*t1,normal-mu*t2)):
            jac[4*i+k,:3]=direction
            jac[4*i+k,3:6]=np.cross(point-pose[:3],direction)@r
            jac[4*i+k,6:]=-direction@contact_jp
        phi[4*i:4*i+4]=gap
    # Plane support at the actual mesh lowest point; object inertia is identical
    # to Lambda and the executed scene. It is not a fabricated hand contact.
    vertices=hand.surface.mesh.vertices@r.T+pose[:3]
    point=vertices[np.argmin(vertices[:,2])]
    floor=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    floor_z=hand.model.geom_pos[floor,2] if floor>=0 else -100.
    normal=np.array([0.,0.,1.]);normal,t1,t2=frame(normal)
    # Prefer the actual ground patch frame (box/mesh collision can choose an
    # orientation other than the normal-only completion convention).
    for c in hand.data.contact:
        if hand.obj in hand.model.geom_bodyid[[c.geom1,c.geom2]] and floor in (c.geom1,c.geom2):
            t1,t2=native_tangents(normal,c.frame.reshape(3,3)[1:]);break
    floor_mu=hand.floor_friction
    for k,direction in enumerate((normal+floor_mu*t1,normal+floor_mu*t2,normal-floor_mu*t1,normal-floor_mu*t2)):
        jac[4*n+k,:3]=direction;jac[4*n+k,3:6]=np.cross(point-pose[:3],direction)@r
    phi[4*n:4*n+4]=point[2]-floor_z
    collision=np.zeros((8,1+nu));collision[:,0]=1.
    obstacles=[];represented=set()
    allowed={m.body for i,m in enumerate(materials) if active[i] or stick[i]}
    held={c['body'] for c in actual}
    # Nonparticipating regions still constrain motion. Their nearest projections
    # are collision rows, not fictitious semantic contacts or Lambda force slots.
    for i,m in enumerate(materials):
        if m.body in allowed or m.body in held or gaps[i]>.03:continue
        point=pose[:3]+r@projected[i];normal=r@inward[i]
        normal,t1,t2=frame(normal)
        rows=np.array([np.r_[d,np.cross(point-pose[:3],d)@r,-d@jp[i]]
            for d in (normal+mu*t1,normal+mu*t2,normal-mu*t1,normal-mu*t2)])
        key=(hand.obj,m.body)
        obstacles.append((float(gaps[i]),rows,key,mu));represented.add(key)
    for c in hand.data.contact:
        b1,b2=hand.model.geom_bodyid[[c.geom1,c.geom2]]
        if hand.structural_pair(b1,b2):continue
        if b1 not in hand.hand_bodies and b2 not in hand.hand_bodies:continue
        if hand.obj in (b1,b2):
            other=b2 if b1==hand.obj else b1
            if other in allowed or (hand.obj,other) in represented:continue
        j1,j2=np.zeros((3,hand.model.nv)),np.zeros((3,hand.model.nv));jr=np.zeros_like(j1)
        mujoco.mj_jac(hand.model,hand.data,j1,jr,c.pos,int(b1));mujoco.mj_jac(hand.model,hand.data,j2,jr,c.pos,int(b2))
        relative=(j2-j1)[:,np.r_[np.arange(hand.ov,hand.ov+6),hand.dadr]]
        normal,t1,t2=c.frame[:3],c.frame[3:6],c.frame[6:9]
        contact_mu=float(c.friction[0])
        rows=np.array([d@relative for d in (normal+contact_mu*t1,normal+contact_mu*t2,normal-contact_mu*t1,normal-contact_mu*t2)])
        obstacles.append((float(c.dist),rows,(int(b1),int(b2)),contact_mu))
    for index,(gap,rows,key,contact_mu) in enumerate(sorted(obstacles,key=lambda x:x[0])[:8]):
        maintained=hand.obj in key and any(body in held for body in key if body!=hand.obj)
        if not maintained:collision[index]=np.r_[gap,rows.mean(axis=0)[6:]]
        phi[4*(n+1+index):4*(n+2+index)]=gap
        jac[4*(n+1+index):4*(n+2+index)]=rows
        coefficients[n+1+index]=contact_mu
    result=(phi,jac,points,normals,jp,stick,collision)
    return (*result,coefficients) if return_friction else result


def support_parameters(hand):
    pose=hand.object_pose();r=rotation(pose)
    vertices=hand.surface.mesh.vertices@r.T+pose[:3]
    point=vertices[np.argmin(vertices[:,2])]
    floor=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_GEOM,'floor')
    floor_z=hand.model.geom_pos[floor,2] if floor>=0 else -100.
    row=np.r_[[0.,0.,1.],np.cross(point-pose[:3],[0.,0.,1.])@r]
    # Lambda's support is a unilateral plane impulse, not a target wrench.
    return dict(support_row=row,support_gap=float(point[2]-floor_z),support_stiffness=1.,support_damping=0.)
