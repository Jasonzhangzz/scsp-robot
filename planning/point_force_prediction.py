"""CPU accuracy oracle for full-period, body-fixed point-force predictions.

Each candidate starts from the same observation. MuJoCo supplies geometry,
inertia and smooth dynamics at every substep; only QP uses its contact solve.
Other contact laws retain their independent, original equations.
"""
import copy
from collections import Counter
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from models.corrected_scm import corrected_scm_velocity


class CPUSubstepPredictor:
    def __init__(self,model,args,name):
        self.model=copy.copy(model)
        self.model.opt.timestep=args.policy_dt/args.prediction_substeps
        self.data=mujoco.MjData(self.model)
        self.args,self.name=args,name
        self.lcp_reference_counts=Counter()
        self.lcp_reference_fallbacks=[]

    def predict(self,pose,velocity,points,forces,duration=None):
        from examples.mpc.franka.scm.contact_dynamics import full_mass,contact_snapshot,pyramid_rows
        contactsdf_velocity = None
        if self.name == 'contactsdf':
            from examples.mpc.franka.scm.contact_dynamics import contactsdf_velocity
        solve_lcp = solve_lcp_active_set = None
        if self.name == 'lcp':
            from examples.mpc.franka.scm.lcp_backend import solve_lcp,solve_lcp_active_set
        poses,velocities,details=[],[],[]
        m,d,a=self.model,self.data,self.args
        duration=a.policy_dt if duration is None else duration
        count=max(1,int(round(duration/(a.policy_dt/a.prediction_substeps))))
        m.opt.timestep=duration/count
        h=m.opt.timestep
        body=m.body('obj').id
        for point,force in zip(points,forces):
            mujoco.mj_resetData(m,d)
            d.qpos[:],d.qvel[:]=pose,velocity
            mujoco.mj_forward(m,d)
            rotation=d.xmat[body].reshape(3,3)
            d.qfrc_applied[:]=0.
            mujoco.mj_applyFT(m,d,rotation@force,np.zeros(3),d.xpos[body]+rotation@point,body,d.qfrc_applied)
            d.qvel[:]+=np.linalg.solve(full_mass(m,d),d.qfrc_applied*duration)
            d.qfrc_applied[:]=0.
            alpha,residual=1.,0.
            for _ in range(count):
                mujoco.mj_forward(m,d)
                if self.name=='qp':
                    stamp=float(d.time)
                    warning_numbers=np.asarray(d.warning.number).copy()
                    mujoco.mj_step(m,d)
                    if float(d.time)<stamp+.5*h or np.any(np.asarray(d.warning.number)>warning_numbers):
                        raise FloatingPointError('MuJoCo candidate prediction reset or issued an instability warning')
                else:
                    mujoco.mj_forward(m,d)
                    mass=full_mass(m,d)
                    inverse=np.linalg.inv(mass)
                    native,jac=contact_snapshot(m,d)
                    rows,gap=pyramid_rows(native,jac)
                    old=d.qvel.copy()
                    free=old+h*d.qacc_smooth
                    if self.name=='scm_corrected':
                        updated,info=corrected_scm_velocity(free,rows,gap,inverse,h,a.scm_regularization)
                        alpha=float(info['alpha'])
                    elif self.name=='contactsdf':
                        updated=contactsdf_velocity(mass,free,rows,gap,h,a.contactsdf_sigma)[0]
                    else:
                        W=rows@inverse@rows.T+a.scm_regularization*np.eye(len(gap))
                        g=rows@free+gap/h
                        try:
                            impulse,info=solve_lcp(W,g,a.lcp_tolerance,a.lcp_max_iterations)
                        except FloatingPointError as exc:
                            # Redundant pyramid edges with r=1e-8 can make the
                            # double-precision Lemke tableau ill-conditioned.
                            # Independently solve the SAME SPD LCP; certify in
                            # original units, and expose the oracle fallback.
                            impulse,info=solve_lcp_active_set(W,g,a.lcp_tolerance,a.lcp_max_iterations)
                            reason=str(exc)
                            if reason not in self.lcp_reference_fallbacks:
                                self.lcp_reference_fallbacks.append(reason)
                        self.lcp_reference_counts[info['solver']]+=1
                        residual=max(residual,info['residual'])
                        updated=free+inverse@rows.T@impulse
                    # Match MuJoCo Euler's implicit damping, after contact.
                    d.qvel[:]=old+np.linalg.solve(mass+h*np.diag(m.dof_damping),mass@(updated-old))
                    mujoco.mj_integratePos(m,d.qpos,d.qvel,h)
                    d.time+=h
                if not np.isfinite(np.r_[d.qpos,d.qvel]).all() or np.linalg.norm(d.qvel)>1e5:
                    raise FloatingPointError('Nonfinite or divergent candidate trajectory')
            mujoco.mj_forward(m,d)
            rotation=d.xmat[body].reshape(3,3)
            poses.append(d.qpos.copy())
            velocities.append(np.r_[rotation.T@d.qvel[:3],d.qvel[3:]])
            details.append(dict(alpha=alpha,residual=residual))
        return np.asarray(velocities),np.asarray(poses),details


def velocity_cost(velocities,args):
    velocities=np.asarray(velocities)
    return (args.linear_velocity_coef*np.sum(velocities[...,:3]**2,axis=-1)+
            args.angular_velocity_coef*np.sum(velocities[...,3:]**2,axis=-1))


def force_commands(raw,points,frames,model,args):
    """Parameterize the shared feasible set; the executor never clips forces."""
    raw=np.clip(np.atleast_2d(raw),0,1)
    tangents=2*raw[:,1:]-1
    tangents/=np.maximum(1.,np.abs(tangents).sum(axis=1,keepdims=True))
    direction=np.column_stack((np.ones(len(raw)),args.arm_friction*tangents))
    body_force=np.einsum('nij,nj->ni',frames,direction)
    body=model.body('obj').id
    rotation=Rotation.from_quat(model.body_iquat[body][[1,2,3,0]]).as_matrix()
    inverse_inertia=np.linalg.inv(rotation@np.diag(model.body_inertia[body])@rotation.T)
    angular=np.cross(points-model.body_ipos[body],body_force)@inverse_inertia.T
    cap=np.full(len(raw),args.max_contact_force/np.sqrt(1+args.arm_friction**2))
    cap=np.minimum(cap,model.body_mass[body]*args.max_linear_force_delta_v/
        (args.policy_dt*np.maximum(np.linalg.norm(body_force,axis=1),1e-30)))
    cap=np.minimum(cap,args.max_angular_force_delta_v/
        (args.policy_dt*np.maximum(np.linalg.norm(angular,axis=1),1e-30)))
    return direction*(raw[:,0]*cap)[:,None]
