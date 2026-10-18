"""Float64 GPU rollouts of a free rigid mesh above MuJoCo's static plane.

Mass, COM, full inertia and per-DOF damping come from the actual MuJoCo model.
Every candidate updates pose, smooth dynamics, and MuJoCo 3.6 plane-mesh
contacts at every substep. Four independent contact laws share that geometry.
The QP uses native pyramidal invweight/impedance/aref, verified against mj_step.
"""
import numpy as np
import mujoco
import warp as wp
from scipy.spatial.transform import Rotation
from planning.point_force_gpu import (GPUForceCEM,ForceJob,V6,M6,initialize,sample,update,
    feasible,cholesky_solve6,pose_cost)
from planning.point_force_collision_gpu import GPUPlaneContacts,PlaneMesh,plane_mesh_contacts

P12=wp.types.vector(12,wp.float64)
R12=wp.types.matrix((12,6),wp.float64)
V5=wp.types.vector(5,wp.float64)


@wp.struct
class Physics:
    com:wp.vec3d
    inertia:wp.mat33d
    inverse_inertia:wp.mat33d
    mass:wp.float64
    damping:V6
    gravity:wp.vec3d
    solref:wp.vec2d
    solimp:V5
    invweight:wp.float64
    friction:wp.float64
    refsafe:wp.int32
    substeps:wp.int32


@wp.func
def bounded_force(j:ForceJob,c:int,raw:wp.vec3d):
    direction=feasible(wp.vec3d(wp.float64(1.),raw[1],raw[2]),wp.float64(1.),j.mu)
    body_force=j.frame[c]*direction
    angular=j.inverse_inertia*wp.cross(j.point[c]-j.center_of_mass,body_force)
    cap=wp.min(j.cap,j.object_mass*j.max_linear_force_delta_v/(j.h*wp.max(wp.length(body_force),wp.float64(1.e-30))))
    cap=wp.min(cap,j.max_angular_force_delta_v/(j.h*wp.max(wp.length(angular),wp.float64(1.e-30))))
    return direction*(wp.clamp(raw[0],wp.float64(0.),wp.float64(1.))*cap)


@wp.func
def native_coefficients(p:Physics,gap:wp.float64,velocity:wp.float64,h:wp.float64):
    # MuJoCo 3.6 mj_makeImpedance / mj_referenceConstraint, pyramidal edges.
    timeconst=p.solref[0]
    if p.refsafe!=0:
        timeconst=wp.max(timeconst,wp.float64(2.)*h)
    ratio=p.solref[1]
    dmin=wp.clamp(p.solimp[0],wp.float64(.0001),wp.float64(.9999))
    dmax=wp.clamp(p.solimp[1],wp.float64(.0001),wp.float64(.9999))
    width=wp.max(p.solimp[2],wp.float64(1.e-15))
    mid=wp.clamp(p.solimp[3],wp.float64(.0001),wp.float64(.9999))
    power=wp.max(p.solimp[4],wp.float64(1.))
    x=wp.abs(gap)/width
    imp=dmax
    if x<wp.float64(1.):
        y=wp.float64(0.)
        if x<mid:
            y=wp.pow(x,power)/wp.pow(mid,power-wp.float64(1.))
        else:
            y=wp.float64(1.)-wp.pow(wp.float64(1.)-x,power)/wp.pow(wp.float64(1.)-mid,power-wp.float64(1.))
        imp=wp.clamp(dmin+y*(dmax-dmin),dmin,dmax)
    k,b=wp.float64(0.),wp.float64(0.)
    if p.solref[0]>wp.float64(0.):
        k=wp.float64(1.)/(dmax*dmax*timeconst*timeconst*ratio*ratio)
    else:
        k=-p.solref[0]/(dmax*dmax)
    if p.solref[1]>wp.float64(0.):
        b=wp.float64(2.)/(dmax*timeconst)
    else:
        b=-p.solref[1]/dmax
    regularizer=wp.max(p.invweight*(wp.float64(1.)-imp)/imp,wp.float64(1.e-15))
    aref=-k*imp*gap-b*velocity
    return regularizer,aref


@wp.func
def native_objective(mass:M6,v:V6,free:V6,jac:R12,R:P12,bias:P12,rows:int):
    delta=v-free
    result=wp.float64(.5)*wp.dot(delta,mass*delta)
    for r in range(rows):
        slack=bias[r]
        for k in range(6):
            slack+=jac[r,k]*v[k]
        closed=wp.min(slack,wp.float64(0.))
        result+=wp.float64(.5)*closed*closed/R[r]
    return result


@wp.func
def contact_response(j:ForceJob,mass:M6,inverse:M6,v:V6,jac:R12,gap:P12,
    native_regularization:P12,native_bias:P12,rows:int,h:wp.float64):
    directions=R12(wp.float64(0.))
    diagonal=P12(wp.float64(0.))
    for r in range(rows):
        row=V6(wp.float64(0.))
        for k in range(6):
            row[k]=jac[r,k]
        direction=inverse*row
        for k in range(6):
            directions[r,k]=direction[k]
        diagonal[r]=wp.dot(row,direction)
    impulse=P12(wp.float64(0.))
    alpha=wp.float64(1.)
    residual=wp.float64(0.)
    if j.model == 0:
        numerator = wp.float64(0.)
        normp = wp.float64(0.)
        q = V6(wp.float64(0.))
        for r in range(rows):
            row = V6(wp.float64(0.))
            for k in range(6):
                row[k] = jac[r,k]
            g = wp.dot(row,v)+gap[r]/h
            p = wp.max(-g/(diagonal[r]+j.regularization),wp.float64(0.))
            q += row*p
            numerator -= g*p
            normp += p*p
        denominator = wp.dot(q,inverse*q)+j.regularization*normp
        if denominator > wp.float64(0.):
            alpha = wp.clamp(numerator/denominator,wp.float64(0.),wp.float64(1.))
        v += alpha*(inverse*q)
    elif j.model == 1:
        # Equivalent to contactsdf_velocity's Q=M/h^2 LSE projection.
        maximum = wp.float64(-1.e100)
        for r in range(rows):
            row = V6(wp.float64(0.))
            for k in range(6):
                row[k] = jac[r,k]
            scale = wp.float64(1.)/wp.sqrt(diagonal[r]+wp.float64(1.e-2))
            impulse[r] = -j.sigma*scale*(gap[r]/h+wp.dot(row,v))
            maximum = wp.max(maximum,impulse[r])
        if rows > 0:
            total = wp.float64(0.)
            for r in range(rows):
                impulse[r] = wp.exp(impulse[r]-maximum)
                total += impulse[r]
            distance = wp.max((maximum+wp.log(total))/j.sigma,wp.float64(0.))
            q = V6(wp.float64(0.))
            for r in range(rows):
                row = V6(wp.float64(0.))
                for k in range(6):
                    row[k] = jac[r,k]
                scale = wp.float64(1.)/wp.sqrt(diagonal[r]+wp.float64(1.e-2))
                q += row*(distance*scale*impulse[r]/total)
            v += inverse*q
    elif j.model == 3:
        # Native soft-contact primal Newton in six physical coordinates.
        # Its objective is exactly MuJoCo's pyramidal QP with efc_R/aref.
        free = v
        for iteration in range(100):
            gradient = mass*(v-free)
            hessian = mass
            for r in range(rows):
                row = V6(wp.float64(0.))
                for k in range(6):
                    row[k] = jac[r,k]
                slack = wp.dot(row,v)+native_bias[r]
                if slack < wp.float64(0.):
                    weight = wp.float64(1.)/native_regularization[r]
                    gradient += weight*slack*row
                    for k in range(6):
                        for l in range(6):
                            hessian[k,l] += weight*row[k]*row[l]
            direction = cholesky_solve6(hessian,-gradient)
            length = wp.float64(0.)
            for k in range(6):
                length = wp.max(length,wp.abs(direction[k]))
            if length < wp.float64(1.e-10):
                break
            objective_before = native_objective(mass,v,free,jac,native_regularization,native_bias,rows)
            descent = wp.dot(gradient,direction)
            scale = wp.float64(1.)
            for line in range(40):
                trial = v+scale*direction
                if native_objective(mass,trial,free,jac,native_regularization,native_bias,rows) <= objective_before+wp.float64(1.e-4)*scale*descent:
                    v = trial
                    break
                scale *= wp.float64(.5)
        # Stationarity certificate on the original, unscaled native QP.
        gradient = mass*(v-free)
        for r in range(rows):
            row = V6(wp.float64(0.))
            for k in range(6):
                row[k] = jac[r,k]
            slack = wp.dot(row,v)+native_bias[r]
            gradient += row*wp.min(slack,wp.float64(0.))/native_regularization[r]
        for k in range(6):
            residual = wp.max(residual,wp.abs(gradient[k]))
        if residual > j.tolerance:
            return v,alpha,residual,False
    elif j.model == 2:
        # Coupled coordinate solve; every row sees all previous impulses.
        for iteration in range(j.iterations):
            for r in range(rows):
                row = V6(wp.float64(0.))
                for k in range(6):
                    row[k] = jac[r,k]
                direction = V6(wp.float64(0.))
                for k in range(6):
                    direction[k] = directions[r,k]
                regularizer = j.regularization
                bias = gap[r]/h
                if j.model == 3:
                    regularizer = native_regularization[r]
                    bias = native_bias[r]
                g = wp.dot(row,v)+bias+regularizer*impulse[r]
                p = wp.max(impulse[r]-g/(diagonal[r]+regularizer),wp.float64(0.))
                v += direction*(p-impulse[r])
                impulse[r] = p
            if iteration%8 == 7 or iteration == j.iterations-1:
                residual = wp.float64(0.)
                for r in range(rows):
                    regularizer = j.regularization
                    bias = gap[r]/h
                    if j.model == 3:
                        regularizer = native_regularization[r]
                        bias = native_bias[r]
                    slack = bias+regularizer*impulse[r]
                    for k in range(6):
                        slack += jac[r,k]*v[k]
                    residual = wp.max(residual,wp.max(-slack,wp.abs(slack*impulse[r])))
                    if impulse[r] > wp.float64(1.e-12):
                        residual = wp.max(residual,wp.abs(slack))
                if residual <= j.tolerance:
                    break
        if residual > j.tolerance:
            return v,alpha,residual,False
    return v,alpha,residual,True


@wp.kernel
def evaluate_trajectory(jobs:wp.array(dtype=ForceJob),physics:wp.array(dtype=Physics),contacts:wp.array(dtype=PlaneMesh)):
    j,p,cfg=jobs[0],physics[0],contacts[0]
    c,s=wp.tid()
    j.costs[c,s]=wp.float64(1.e100)
    if j.valid[c]==0:
        return
    f=bounded_force(j,c,j.samples[c,s])
    force=j.frame[c]*f
    pose=j.pose
    orientation=wp.transform_get_rotation(pose)
    # Deliver the whole-interval impulse, then roll out with no applied force.
    # The linear kick includes the same COM coupling as the substep acceleration.
    angular_kick=p.inverse_inertia*wp.cross(j.point[c]-p.com,force)*j.h
    linear=wp.vec3d(j.observed[0],j.observed[1],j.observed[2])+force*j.h/p.mass-wp.cross(angular_kick,p.com)
    omega=wp.vec3d(j.observed[3],j.observed[4],j.observed[5])+angular_kick
    world_linear=wp.quat_rotate(orientation,linear)
    force=wp.vec3d(wp.float64(0.),wp.float64(0.),wp.float64(0.))
    h=j.h/wp.float64(p.substeps)
    damped=j.mass
    for k in range(6):
        damped[k,k]+=h*p.damping[k]
    max_residual=wp.float64(0.)
    last_alpha=wp.float64(1.)
    for step in range(p.substeps):
        orientation=wp.transform_get_rotation(pose)
        linear=wp.quat_rotate_inv(orientation,world_linear)
        damp_force=wp.vec3d(-p.damping[0]*linear[0],-p.damping[1]*linear[1],-p.damping[2]*linear[2])
        damp_torque=wp.vec3d(-p.damping[3]*omega[0],-p.damping[4]*omega[1],-p.damping[5]*omega[2])
        angular_acc=p.inverse_inertia*(wp.cross(j.point[c]-p.com,force)+damp_torque-wp.cross(p.com,damp_force)-wp.cross(omega,p.inertia*omega))
        linear_acc=(force+damp_force)/p.mass+wp.quat_rotate_inv(orientation,p.gravity)-wp.cross(angular_acc,p.com)-wp.cross(omega,wp.cross(omega,p.com))
        old=V6(linear[0],linear[1],linear[2],omega[0],omega[1],omega[2])
        v=old+h*V6(linear_acc[0],linear_acc[1],linear_acc[2],angular_acc[0],angular_acc[1],angular_acc[2])
        distances,positions,count=plane_mesh_contacts(cfg,pose)
        rows=count*4
        jac=R12(wp.float64(0.))
        gap,R,bias=P12(wp.float64(0.)),P12(wp.float64(0.)),P12(wp.float64(0.))
        for contact in range(count):
            point=wp.vec3d(positions[contact,0],positions[contact,1],positions[contact,2])
            local_point=wp.quat_rotate_inv(orientation,point-wp.transform_get_translation(pose))
            for edge in range(4):
                direction=wp.vec3d(wp.float64(0.),wp.float64(0.),wp.float64(1.))
                if edge==0:
                    direction[1]=p.friction
                elif edge==1:
                    direction[1]=-p.friction
                elif edge==2:
                    direction[0]=-p.friction
                else:
                    direction[0]=p.friction
                direction=wp.quat_rotate_inv(orientation,direction)
                angular=wp.cross(local_point,direction)
                row=V6(direction[0],direction[1],direction[2],angular[0],angular[1],angular[2])
                r=contact*4+edge
                for k in range(6):
                    jac[r,k]=row[k]
                gap[r]=distances[contact]
                regularizer,aref=native_coefficients(p,gap[r],wp.dot(row,old),h)
                R[r]=regularizer
                bias[r]=-wp.dot(row,old)-h*aref
        updated,alpha,residual,valid=contact_response(j,j.mass,j.inverse,v,jac,gap,R,bias,rows,h)
        if not valid:
            j.residuals[c,s]=residual
            return
        updated=old+cholesky_solve6(damped,j.mass*(updated-old))
        for k in range(6):
            if not wp.isfinite(updated[k]) or wp.abs(updated[k])>wp.float64(1.e5):
                return
        world_linear=wp.quat_rotate(orientation,wp.vec3d(updated[0],updated[1],updated[2]))
        omega=wp.vec3d(updated[3],updated[4],updated[5])
        position=wp.transform_get_translation(pose)+h*world_linear
        speed=wp.length(omega)
        if speed>wp.float64(1.e-14):
            orientation=wp.normalize(orientation*wp.quat_from_axis_angle(omega/speed,h*speed))
        pose=wp.transformd(position,orientation)
        last_alpha=alpha
        max_residual=wp.max(max_residual,residual)
    linear=wp.quat_rotate_inv(orientation,world_linear)
    cost=pose_cost(pose,j.target,j.config)+j.force_reg*wp.dot(f,f)+j.tangent_reg*(f[1]*f[1]+f[2]*f[2])
    cost+=j.linear_velocity_reg*wp.dot(linear,linear)+j.angular_velocity_reg*wp.dot(omega,omega)
    j.costs[c,s]=cost
    j.poses[c,s]=pose
    j.velocities[c,s]=V6(linear[0],linear[1],linear[2],omega[0],omega[1],omega[2])
    j.alphas[c,s]=last_alpha
    j.residuals[c,s]=max_residual


class GPUPhysicsCEM(GPUForceCEM):
    def __init__(self,points,frames,args,model,name):
        super().__init__(points,frames,args,name)
        if np.any(model.jnt_stiffness) or model.opt.viscosity or model.opt.density:
            raise ValueError('Free-body GPU predictor requires no springs or fluid forces')
        if not np.allclose(model.dof_damping[:3],model.dof_damping[0],atol=0,rtol=0):
            raise ValueError('Linear damping must be isotropic')
        self.contacts=GPUPlaneContacts(model,self.device)
        cfg=self.contacts.config
        if not np.allclose(cfg.normal,[0,0,1],atol=1e-12):
            raise ValueError('Horizontal benchmark plane required')
        body=model.body('obj').id
        p=Physics()
        p.com=wp.vec3d(*model.body_ipos[body]);p.mass=float(model.body_mass[body])
        rotation=Rotation.from_quat(model.body_iquat[body][[1,2,3,0]]).as_matrix()
        inertia=rotation@np.diag(model.body_inertia[body])@rotation.T
        p.inertia,p.inverse_inertia=wp.mat33d(inertia),wp.mat33d(np.linalg.inv(inertia))
        j=self.job
        j.center_of_mass,j.inverse_inertia,j.object_mass=p.com,p.inverse_inertia,p.mass
        j.max_linear_force_delta_v=args.max_linear_force_delta_v
        j.max_angular_force_delta_v=args.max_angular_force_delta_v
        self.cpu_model,self.points,self.frames=model,np.asarray(points),np.asarray(frames)
        p.damping=V6(model.dof_damping);p.gravity=wp.vec3d(*model.opt.gravity)
        # Read the actual mixed native contact parameters in double precision.
        data=mujoco.MjData(model);data.qpos[2]=-.1;mujoco.mj_forward(model,data)
        native=next(c for c in data.contact if set(c.geom)=={model.geom('obj').id,model.geom('table').id})
        if native.friction[0]!=native.friction[1]:
            raise ValueError('Isotropic sliding friction required')
        p.friction=float(native.friction[0]);p.solref=wp.vec2d(*native.solref);p.solimp=V5(*native.solimp)
        invweight=float(model.body_invweight0[body,0])
        mu=p.friction
        p.invweight=invweight*(1+mu*mu)*2*mu*mu/model.opt.impratio
        p.refsafe=int(not model.opt.disableflags&int(mujoco.mjtDisableBit.mjDSBL_REFSAFE))
        p.substeps=args.prediction_substeps
        self.physics=wp.array([p],dtype=Physics,device=self.device)
        wp.load_module(module=__name__,device=self.device)

    def commands(self,raw,indices):
        from planning.point_force_prediction import force_commands
        return force_commands(raw,self.points[indices],self.frames[indices],self.cpu_model,self.args)

    def _generations(self):
        j=self.job
        wp.launch(initialize,self.count,inputs=[self.jobs],device=self.device)
        for generation in range(self.search_iterations):
            wp.launch(sample,(self.count,j.population),inputs=[self.jobs,generation],device=self.device)
            wp.launch(evaluate_trajectory,(self.count,j.population),inputs=[self.jobs,self.physics,self.contacts.settings],device=self.device)
            wp.launch(update,self.count,inputs=[self.jobs],device=self.device)
