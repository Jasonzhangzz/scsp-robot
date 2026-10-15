"""Internal, actuator-feasible gravity preload seeds for contact-point MPC.

This supplies a position-command candidate, not an incoming wrench reference,
force-tracking objective, or applied object force. Native MPC rollout cost must
still accept the resulting command. Only already measured patches participate.
"""
import mujoco
import numpy as np
from scipy.optimize import linprog
from models.wuji_hand import rotation
from planning.contact_friction import native_tangents
from planning.grasp_verification import contact_wrench_rays


def pressure_command_map(point_jacobian,normals,measured,reference,*,grasp_established,
                         gain=.0003,limit=.0015,stiffness=None):
    """Multi-region normal feedback; tangential motion remains an MPC variable.

    Demanding a pure 3D normal displacement at every point on one rigid link
    can be incompatible. Regulate only their scalar gaps, rather than applying
    Allegro's independent spherical-pad inverse to coupled Wuji regions.
    """
    jac=np.einsum('ij,ijk->ik',normals,point_jacobian)
    finger=jac[:,6:]
    if stiffness is None:
        inverse=finger.T@np.linalg.solve(finger@finger.T+1e-6*np.eye(len(jac)),np.eye(len(jac)))
        correction=inverse@np.clip(gain*(np.asarray(reference)-measured),-limit,limit)
    else:
        # MuJoCo's position servo is tau=kp*(ctrl-q)-kv*dq. Convert a
        # fraction of the measured force error to the corresponding position
        # target increment, using the actual Wuji gains, not Allegro's tuned
        # metres/Newton constant. This does not change any physical gain.
        weighted=finger.T/np.maximum(np.asarray(stiffness)[6:,None],1e-8)
        compliance=finger@weighted
        inverse=weighted@np.linalg.solve(compliance+1e-8*np.eye(len(jac)),np.eye(len(jac)))
        correction=.2*weighted@(np.asarray(reference)-measured)
        excursion=float(np.max(np.abs(finger@correction))) if len(jac) else 0.
        correction*=min(1.,limit/max(excursion,1e-12))
    projection=np.eye(jac.shape[1]);projection[6:,6:]-=inverse@finger
    if not grasp_established:projection[6:,:6]=-inverse@jac[:,:6]
    bias=np.r_[np.zeros(6),correction]
    return projection,bias


def static_preload_seed(hand,actual,normal_rows,force_limit,friction_margin=.8,preferred_bodies=None):
    pose=hand.object_pose();r=rotation(pose);slots=len(normal_rows)
    contacts=[c for c in actual if c['normal']@c['force']>.01]
    if not contacts:return dict(feasible=False,reason='no_measured_patches')
    points=np.array([(c['point']-pose[:3])@r for c in contacts])
    normals=np.array([c['normal']@r for c in contacts])
    tangents=np.array([c.get('tangents',native_tangents(c['normal']))@r for c in contacts])
    rays=contact_wrench_rays(points,normals,hand.friction*friction_margin,edges=4,tangent_directions=tangents)
    force=hand.model.body_mass[hand.obj]*hand.model.opt.gravity
    external=np.r_[force,np.cross(hand.data.xipos[hand.obj]-pose[:3],force)]
    bias=(hand.data.qfrc_bias-hand.data.qfrc_passive)[hand.dadr].copy()
    torque_map=np.zeros((hand.nu,rays.shape[1]))
    grouped_normals=np.zeros((slots,rays.shape[1]))
    keys=list(dict.fromkeys(('slot',c['slot']) if c['slot']>=0 else ('body',c['body']) for c in contacts))
    slot_budgets=np.zeros((len(keys),rays.shape[1]))
    for i,c in enumerate(contacts):
        jp,jr=np.zeros((3,hand.model.nv)),np.zeros((3,hand.model.nv))
        mujoco.mj_jac(hand.model,hand.data,jp,jr,c['point'],c['body'])
        world_rays=r@rays[:3,4*i:4*i+4]
        torque_map[:,4*i:4*i+4]=jp[:,hand.dadr].T@world_rays
        if c['slot']>=0:grouped_normals[c['slot'],4*i:4*i+4]=normal_rows[c['slot']]@world_rays
        key=('slot',c['slot']) if c['slot']>=0 else ('body',c['body'])
        slot_budgets[keys.index(key),4*i:4*i+4]=1.
    gain=hand.model.actuator_gainprm[:,0]
    constant=hand.model.actuator_biasprm[:,0]+hand.model.actuator_biasprm[:,1]*hand.data.actuator_length
    limited=hand.model.actuator_forcelimited.astype(bool)
    lower=np.where(limited,hand.model.actuator_forcerange[:,0],-np.inf)
    upper=np.where(limited,hand.model.actuator_forcerange[:,1],np.inf)
    endpoints=np.array([gain*hand.lower+constant,gain*hand.upper+constant])
    lower=np.clip(endpoints.min(axis=0),lower,upper)
    upper=np.clip(endpoints.max(axis=0),np.where(limited,hand.model.actuator_forcerange[:,0],-np.inf),upper)
    inequalities=np.r_[torque_map,-torque_map,slot_budgets]
    budgets=np.r_[upper-bias,bias-lower,np.full(len(keys),force_limit/np.sqrt(1+hand.friction**2))]
    finite=np.isfinite(budgets);scale=np.r_[np.ones(3),np.full(3,1/max(np.linalg.norm(points,axis=1).max(),.001))]
    local_external=np.r_[r.T@external[:3],r.T@external[3:]]
    preference=np.repeat([1. if preferred_bodies is None or c['body'] in preferred_bodies else 3. for c in contacts],4)
    result=linprog(preference,A_eq=scale[:,None]*rays,b_eq=-scale*local_external,
        A_ub=inequalities[finite],b_ub=budgets[finite],bounds=(0.,None),method='highs')
    fraction=1.
    if not result.success:
        # Before the final contact forms, plan the largest feasible fraction
        # of gravity instead of demanding an infeasible uniform full preload.
        # CURRENT measured Fn is never a cap on this planned force buildup.
        partial=linprog(np.r_[1e-6*preference,-1.],
            A_eq=scale[:,None]*np.c_[rays,local_external],b_eq=np.zeros(6),
            A_ub=np.c_[inequalities[finite],np.zeros(np.count_nonzero(finite))],b_ub=budgets[finite],
            bounds=[(0.,None)]*rays.shape[1]+[(0.,1.)],method='highs')
        if not partial.success or partial.x[-1]<1e-3:
            return dict(feasible=False,full_load_feasible=False,load_fraction=0.,reason='bounded_static_support_infeasible')
        fraction=float(partial.x[-1]);coefficients=partial.x[:-1]
    else:coefficients=result.x
    return dict(feasible=True,full_load_feasible=fraction>=1-1e-6,load_fraction=fraction,
        normal_forces=np.maximum(grouped_normals@coefficients,0.),
        patch_normal_forces=coefficients.reshape(-1,4).sum(axis=1),
        required_joint_torque=bias+torque_map@coefficients,
        object_wrench_residual=float(np.linalg.norm(rays@coefficients+fraction*local_external)),
        full_gravity_wrench_residual=float(np.linalg.norm(rays@coefficients+local_external)))
