"""CUDA CEM over contact points and forces; native contact objectives.

Every (surface point, population sample) is a CUDA thread. Distribution updates
are per point. All generations are captured in one CUDA graph, with independent
trial/round seeds. No CPU prediction is hidden behind the GPU CEM designation.
"""
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation
from planning.object_pose_cost import extended_cost_config

V6 = wp.types.vector(6, wp.float64)
M6 = wp.types.matrix((6, 6), wp.float64)
P128 = wp.types.vector(128, wp.float64)


@wp.struct
class ForceJob:
    inverse: M6
    mass: M6
    free: V6
    observed: V6
    pose: wp.transformd
    target: wp.transformd
    config: wp.array(dtype=wp.float64)
    jac: wp.array2d(dtype=wp.float64)
    directions: wp.array2d(dtype=wp.float64)
    diagonal: wp.array(dtype=wp.float64)
    gap: wp.array(dtype=wp.float64)
    native_regularization: wp.array(dtype=wp.float64)
    native_bias: wp.array(dtype=wp.float64)
    point: wp.array(dtype=wp.vec3d)
    frame: wp.array(dtype=wp.mat33d)
    valid: wp.array(dtype=wp.int32)
    mean: wp.array(dtype=wp.vec3d)
    std: wp.array(dtype=wp.vec3d)
    samples: wp.array2d(dtype=wp.vec3d)
    costs: wp.array2d(dtype=wp.float64)
    velocities: wp.array2d(dtype=V6)
    poses: wp.array2d(dtype=wp.transformd)
    alphas: wp.array2d(dtype=wp.float64)
    residuals: wp.array2d(dtype=wp.float64)
    best: wp.array(dtype=wp.vec3d)
    best_cost: wp.array(dtype=wp.float64)
    best_velocity: wp.array(dtype=V6)
    best_pose: wp.array(dtype=wp.transformd)
    best_alpha: wp.array(dtype=wp.float64)
    best_residual: wp.array(dtype=wp.float64)
    seed: wp.array(dtype=wp.int32)
    cap: wp.float64
    mu: wp.float64
    h: wp.float64
    regularization: wp.float64
    sigma: wp.float64
    temperature: wp.float64
    comfree_stiffness: wp.float64
    comfree_damping: wp.float64
    constraint_mass: wp.array(dtype=wp.float64)
    force_reg: wp.float64
    tangent_reg: wp.float64
    linear_velocity_reg: wp.float64
    angular_velocity_reg: wp.float64
    center_of_mass: wp.vec3d
    inverse_inertia: wp.mat33d
    object_mass: wp.float64
    max_linear_force_delta_v: wp.float64
    max_angular_force_delta_v: wp.float64
    model: wp.int32
    population: wp.int32
    elites: wp.int32
    iterations: wp.int32
    tolerance: wp.float64


@wp.func
def feasible(raw: wp.vec3d, cap: wp.float64, mu: wp.float64):
    n = wp.clamp(raw[0], wp.float64(0.), wp.float64(1.)) * cap
    x = wp.float64(2.) * wp.clamp(raw[1], wp.float64(0.), wp.float64(1.)) - wp.float64(1.)
    y = wp.float64(2.) * wp.clamp(raw[2], wp.float64(0.), wp.float64(1.)) - wp.float64(1.)
    scale = wp.max(wp.float64(1.), wp.abs(x) + wp.abs(y))
    return wp.vec3d(n, mu*n*x/scale, mu*n*y/scale)


@wp.func
def integrate(pose: wp.transformd, v: V6, h: wp.float64):
    q = wp.transform_get_rotation(pose)
    p = wp.transform_get_translation(pose) + wp.quat_rotate(q, h*wp.vec3d(v[0], v[1], v[2]))
    omega = wp.vec3d(v[3], v[4], v[5])
    speed = wp.length(omega)
    dq = wp.quatd(wp.float64(0.), wp.float64(0.), wp.float64(0.), wp.float64(1.))
    if speed > wp.float64(1.e-14):
        dq = wp.quat_from_axis_angle(omega/speed, h*speed)
    return wp.transformd(p, wp.normalize(q*dq))


@wp.func
def pose_cost(pose: wp.transformd, target: wp.transformd, c: wp.array(dtype=wp.float64)):
    p, q = wp.transform_get_translation(pose), wp.transform_get_rotation(pose)
    qd = wp.transform_get_rotation(target)
    delta = p - wp.transform_get_translation(target)
    axis = wp.quat_rotate(q, wp.vec3d(c[3], c[4], c[5]))
    up = wp.vec3d(c[6], c[7], c[8])
    pos = wp.dot(delta, delta)
    if c[9] > wp.float64(0.):
        region = wp.max(wp.length(delta-up*wp.dot(delta,up))-c[9], wp.float64(0.))
        pos = region*region
    tilt = wp.float64(1.)-wp.dot(axis,up)
    barrier = wp.max(tilt-c[10], wp.float64(0.))
    return c[0]*pos+c[1]*(wp.float64(1.)-wp.dot(q,qd)**wp.float64(2.))+c[2]*wp.float64(.5)*tilt+c[11]*barrier*barrier


@wp.func
def native_objective(j: ForceJob, v: V6, free: V6):
    delta = v-free
    result = wp.float64(.5)*wp.dot(delta,j.mass*delta)
    for r in range(j.gap.shape[0]):
        slack = j.native_bias[r]
        for k in range(6):
            slack += j.jac[r,k]*v[k]
        closed = wp.min(slack,wp.float64(0.))
        result += wp.float64(.5)*closed*closed/j.native_regularization[r]
    return result


@wp.func
def cholesky_solve6(matrix: M6, rhs: V6):
    lower = M6(wp.float64(0.))
    for i in range(6):
        for k in range(i+1):
            value = matrix[i,k]
            for l in range(k):
                value -= lower[i,l]*lower[k,l]
            if i == k:
                lower[i,k] = wp.sqrt(wp.max(value,wp.float64(1.e-30)))
            else:
                lower[i,k] = value/lower[k,k]
    intermediate = V6(wp.float64(0.))
    for i in range(6):
        value = rhs[i]
        for k in range(i):
            value -= lower[i,k]*intermediate[k]
        intermediate[i] = value/lower[i,i]
    result = V6(wp.float64(0.))
    for reverse in range(6):
        i = 5-reverse
        value = intermediate[i]
        for k in range(i+1,6):
            value -= lower[k,i]*result[k]
        result[i] = value/lower[i,i]
    return result


@wp.kernel
def initialize(jobs: wp.array(dtype=ForceJob)):
    j = jobs[0]
    i = wp.tid()
    j.mean[i] = wp.vec3d(wp.float64(.1), wp.float64(.5), wp.float64(.5))
    j.std[i] = wp.vec3d(wp.float64(.3))
    j.best_cost[i] = wp.float64(1.e100)
    j.best[i] = wp.vec3d(wp.float64(0.))


@wp.kernel
def sample(jobs: wp.array(dtype=ForceJob), generation: int):
    j = jobs[0]
    c, s = wp.tid()
    rng = wp.rand_init(j.seed[0]+generation*104729, c*j.population+s)
    raw = j.mean[c]+wp.cw_mul(j.std[c], wp.vec3d(wp.float64(wp.randn(rng)), wp.float64(wp.randn(rng)), wp.float64(wp.randn(rng))))
    if s == 0:
        raw = wp.vec3d(wp.float64(0.), wp.float64(.5), wp.float64(.5))
    if s == 1 and generation > 0:
        raw = j.best[c]
    if generation == 0 and s > 0 and s < 6:
        raw = wp.vec3d(wp.pow(wp.float64(10.),wp.float64(s-5)),wp.float64(.5),wp.float64(.5))
    raw = wp.vec3d(wp.clamp(raw[0],wp.float64(0.),wp.float64(1.)),wp.clamp(raw[1],wp.float64(0.),wp.float64(1.)),wp.clamp(raw[2],wp.float64(0.),wp.float64(1.)))
    j.samples[c,s] = raw


@wp.kernel
def evaluate(jobs: wp.array(dtype=ForceJob)):
    j = jobs[0]
    c, s = wp.tid()
    j.costs[c,s] = wp.float64(1.e100)
    if j.valid[c] == 0:
        return
    f = feasible(j.samples[c,s],j.cap,j.mu)
    force = j.frame[c]*f
    torque = wp.cross(j.point[c],force)
    wrench = V6(force[0],force[1],force[2],torque[0],torque[1],torque[2])
    v = j.free+j.h*(j.inverse*wrench)
    rows = j.gap.shape[0]
    impulse = P128(wp.float64(0.))
    alpha = wp.float64(1.)
    residual = wp.float64(0.)
    if j.model == 0:
        numerator = wp.float64(0.)
        normp = wp.float64(0.)
        q = V6(wp.float64(0.))
        for r in range(rows):
            row = V6(wp.float64(0.))
            for k in range(6):
                row[k] = j.jac[r,k]
            g = wp.dot(row,v)+wp.max(j.gap[r],wp.float64(0.))/j.h
            impulse_trial = -g/(j.diagonal[r]+j.regularization)
            p = wp.max(impulse_trial,wp.float64(0.))+j.temperature*wp.log(wp.float64(1.)+wp.exp(-wp.abs(impulse_trial)/j.temperature))
            q += row*p
            numerator -= g*p
            normp += p*p
        denominator = wp.dot(q,j.inverse*q)+j.regularization*normp
        if denominator > wp.float64(0.):
            alpha = wp.clamp(numerator/denominator,wp.float64(0.),wp.float64(1.))
        v += alpha*(j.inverse*q)
    elif j.model == 4:
        q = V6(wp.float64(0.))
        for r in range(rows):
            row = V6(wp.float64(0.))
            for k in range(6):
                row[k] = j.jac[r,k]
            speed = wp.dot(row,v)
            p = j.constraint_mass[r]*wp.max(-j.comfree_damping*speed-j.comfree_stiffness*(j.h*speed+j.gap[r]),wp.float64(0.))
            q += row*p
        v += j.inverse*q
    elif j.model == 1:
        # Equivalent to contactsdf_velocity's Q=M/h^2 LSE projection.
        maximum = wp.float64(-1.e100)
        for r in range(rows):
            row = V6(wp.float64(0.))
            for k in range(6):
                row[k] = j.jac[r,k]
            scale = wp.float64(1.)/wp.sqrt(j.diagonal[r]+wp.float64(1.e-2))
            impulse[r] = -j.sigma*scale*(j.gap[r]/j.h+wp.dot(row,v))
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
                    row[k] = j.jac[r,k]
                scale = wp.float64(1.)/wp.sqrt(j.diagonal[r]+wp.float64(1.e-2))
                q += row*(distance*scale*impulse[r]/total)
            v += j.inverse*q
    elif j.model == 3:
        # Native soft-contact primal Newton in six physical coordinates.
        # Its objective is exactly MuJoCo's pyramidal QP with efc_R/aref.
        free = v
        for iteration in range(100):
            gradient = j.mass*(v-free)
            hessian = j.mass
            for r in range(rows):
                row = V6(wp.float64(0.))
                for k in range(6):
                    row[k] = j.jac[r,k]
                slack = wp.dot(row,v)+j.native_bias[r]
                if slack < wp.float64(0.):
                    weight = wp.float64(1.)/j.native_regularization[r]
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
            old = native_objective(j,v,free)
            descent = wp.dot(gradient,direction)
            scale = wp.float64(1.)
            for line in range(40):
                trial = v+scale*direction
                if native_objective(j,trial,free) <= old+wp.float64(1.e-4)*scale*descent:
                    v = trial
                    break
                scale *= wp.float64(.5)
        # Stationarity certificate on the original, unscaled native QP.
        gradient = j.mass*(v-free)
        for r in range(rows):
            row = V6(wp.float64(0.))
            for k in range(6):
                row[k] = j.jac[r,k]
            slack = wp.dot(row,v)+j.native_bias[r]
            gradient += row*wp.min(slack,wp.float64(0.))/j.native_regularization[r]
        for k in range(6):
            residual = wp.max(residual,wp.abs(gradient[k]))
        if residual > j.tolerance:
            j.residuals[c,s] = residual
            return
    elif j.model == 2:
        # Coupled coordinate solve; every row sees all previous impulses.
        for iteration in range(j.iterations):
            for r in range(rows):
                row = V6(wp.float64(0.))
                for k in range(6):
                    row[k] = j.jac[r,k]
                direction = V6(wp.float64(0.))
                for k in range(6):
                    direction[k] = j.directions[r,k]
                regularizer = j.regularization
                bias = j.gap[r]/j.h
                if j.model == 3:
                    regularizer = j.native_regularization[r]
                    bias = j.native_bias[r]
                g = wp.dot(row,v)+bias+regularizer*impulse[r]
                p = wp.max(impulse[r]-g/(j.diagonal[r]+regularizer),wp.float64(0.))
                v += direction*(p-impulse[r])
                impulse[r] = p
            if iteration%8 == 7 or iteration == j.iterations-1:
                residual = wp.float64(0.)
                for r in range(rows):
                    regularizer = j.regularization
                    bias = j.gap[r]/j.h
                    if j.model == 3:
                        regularizer = j.native_regularization[r]
                        bias = j.native_bias[r]
                    slack = bias+regularizer*impulse[r]
                    for k in range(6):
                        slack += j.jac[r,k]*v[k]
                    residual = wp.max(residual,wp.max(-slack,wp.abs(slack*impulse[r])))
                    if impulse[r] > wp.float64(1.e-12):
                        residual = wp.max(residual,wp.abs(slack))
                if residual <= j.tolerance:
                    break
        if residual > j.tolerance:
            j.residuals[c,s] = residual
            return
    pose = integrate(j.pose,v,j.h)
    cost = pose_cost(pose,j.target,j.config)+j.force_reg*wp.dot(f,f)+j.tangent_reg*(f[1]*f[1]+f[2]*f[2])
    cost += j.linear_velocity_reg*(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])+j.angular_velocity_reg*(v[3]*v[3]+v[4]*v[4]+v[5]*v[5])
    j.costs[c,s] = cost
    j.velocities[c,s] = v
    j.poses[c,s] = pose
    j.alphas[c,s] = alpha
    j.residuals[c,s] = residual


@wp.kernel
def update(jobs: wp.array(dtype=ForceJob)):
    j = jobs[0]
    c = wp.tid()
    mean, second = wp.vec3d(wp.float64(0.)),wp.vec3d(wp.float64(0.))
    for s in range(j.population):
        cost = j.costs[c,s]
        if cost < j.best_cost[c]:
            j.best_cost[c] = cost
            j.best[c] = j.samples[c,s]
            j.best_velocity[c] = j.velocities[c,s]
            j.best_pose[c] = j.poses[c,s]
            j.best_alpha[c] = j.alphas[c,s]
            j.best_residual[c] = j.residuals[c,s]
        rank = int(0)
        for k in range(j.population):
            other = j.costs[c,k]
            if other < cost or (other == cost and k < s):
                rank += 1
        if rank < j.elites:
            raw = j.samples[c,s]
            mean += raw
            second += wp.cw_mul(raw,raw)
    mean /= wp.float64(j.elites)
    variance = second/wp.float64(j.elites)-wp.cw_mul(mean,mean)
    sd = wp.vec3d(wp.float64(0.))
    for k in range(3):
        sd[k] = wp.sqrt(wp.max(variance[k],wp.float64(1.e-8)))
    j.mean[c] = mean
    j.std[c] = sd


class GPUForceCEM:
    def __init__(self, points, frames, args, model):
        wp.init()
        self.device = wp.get_device(args.device)
        if not self.device.is_cuda:
            raise ValueError('GPU CEM requires CUDA')
        self.args, self.count = args, len(points)
        self.model = model
        self.graphs = {}
        self.graph = None
        self.search_iterations = args.population_iterations
        self.jobs = None
        self.job = j = ForceJob()
        n, pop = self.count,args.population_size
        j.point = wp.array(points,dtype=wp.vec3d,device=self.device)
        j.frame = wp.array(frames,dtype=wp.mat33d,device=self.device)
        j.valid = wp.ones(n,dtype=wp.int32,device=self.device)
        j.mean = wp.empty(n,dtype=wp.vec3d,device=self.device)
        j.std = wp.empty(n,dtype=wp.vec3d,device=self.device)
        j.samples = wp.empty((n,pop),dtype=wp.vec3d,device=self.device)
        j.costs = wp.empty((n,pop),dtype=wp.float64,device=self.device)
        j.velocities = wp.empty((n,pop),dtype=V6,device=self.device)
        j.poses = wp.empty((n,pop),dtype=wp.transformd,device=self.device)
        j.alphas = wp.empty((n,pop),dtype=wp.float64,device=self.device)
        j.residuals = wp.zeros((n,pop),dtype=wp.float64,device=self.device)
        j.best = wp.empty(n,dtype=wp.vec3d,device=self.device)
        j.best_cost = wp.empty(n,dtype=wp.float64,device=self.device)
        j.best_velocity = wp.empty(n,dtype=V6,device=self.device)
        j.best_pose = wp.empty(n,dtype=wp.transformd,device=self.device)
        j.best_alpha = wp.empty(n,dtype=wp.float64,device=self.device)
        j.best_residual = wp.empty(n,dtype=wp.float64,device=self.device)
        j.config = wp.zeros(12,dtype=wp.float64,device=self.device)
        j.seed = wp.zeros(1,dtype=wp.int32,device=self.device)
        j.mu = args.arm_friction
        j.cap = args.max_contact_force/np.sqrt(1+j.mu*j.mu)
        j.h = args.policy_dt
        j.regularization,j.sigma = args.scm_regularization,args.contactsdf_sigma
        j.temperature = getattr(args,'softplus_temperature',1e-4)
        j.comfree_stiffness = getattr(args,'comfree_stiffness',.1)
        j.comfree_damping = getattr(args,'comfree_damping',.001)
        j.force_reg,j.tangent_reg = args.force_reg_coef,args.friction_reg_coef
        j.linear_velocity_reg = getattr(args,'linear_velocity_coef',0.)
        j.angular_velocity_reg = getattr(args,'angular_velocity_coef',0.)
        j.model = {'scm_corrected':0,'contactsdf':1,'lcp':2,'qp':3,'comfree':4}[model]
        j.population,j.elites = pop,max(2,int(np.ceil(pop*args.cem_elite_fraction)))
        j.iterations,j.tolerance = args.lcp_max_iterations,args.lcp_tolerance
        # Compile before graph capture; no trial state is consumed.
        wp.load_module(module=__name__,device=self.device)

    def prepare(self, inverse, free, pose, target, config, jac, gap, valid, seed, observed=None,
                native_data=None,velocity_scale=1.,native_model=None):
        j = self.job
        j.linear_velocity_reg = getattr(self.args,'linear_velocity_coef',0.)*velocity_scale
        j.angular_velocity_reg = getattr(self.args,'angular_velocity_coef',0.)*velocity_scale
        j.inverse,j.free = M6(inverse),V6(free)
        j.mass = M6(np.linalg.inv(np.asarray(inverse)))
        j.observed = V6(free if observed is None else observed)
        j.pose = wp.transformd(wp.vec3d(*pose[:3]),wp.quatd(*pose[[4,5,6,3]]))
        j.target = wp.transformd(wp.vec3d(*target[:3]),wp.quatd(*target[[4,5,6,3]]))
        if len(gap) > 128:
            raise RuntimeError('Contact row capacity (128) exceeded')
        # Struct scalar changes require a fresh graph. Geometry is shared by
        # all samples; graph construction is part of measured planning work.
        j.jac = wp.array(jac,dtype=wp.float64,device=self.device)
        directions = np.asarray(jac)@np.asarray(inverse).T
        j.directions = wp.array(directions,dtype=wp.float64,device=self.device)
        j.diagonal = wp.array(np.sum(jac*directions,axis=1),dtype=wp.float64,device=self.device)
        j.gap = wp.array(gap,dtype=wp.float64,device=self.device)
        from models.comfree_contact import constraint_mass
        j.constraint_mass = wp.array(constraint_mass(native_model,native_data) if self.model=='comfree'
                                    else np.zeros(len(gap)),dtype=wp.float64,device=self.device)
        if self.model == 'qp':
            if native_data is None or native_data.nefc != len(gap):
                raise ValueError('Native QP requires exactly the common contact constraint rows')
            # MuJoCo's native pyramidal soft-contact QP in impulse units:
            # W=J Minv J.T+diag(efc_R), g=h*(J*a_smooth-efc_aref).
            # This retains its native softness; it is NOT the rigid LCP.
            native_jac = native_data.efc_J.reshape(len(gap),6)
            rotation = Rotation.from_quat(np.asarray(pose)[[4,5,6,3]]).as_matrix()
            conversion = np.eye(6);conversion[:3,:3] = rotation
            if not np.allclose(native_jac@conversion,jac,atol=1e-10,rtol=1e-10):
                raise ValueError('MuJoCo native constraint row order differs')
            j.native_regularization = wp.array(native_data.efc_R,dtype=wp.float64,device=self.device)
            bias = -np.asarray(jac)@np.asarray(observed)-self.args.policy_dt*native_data.efc_aref
            j.native_bias = wp.array(bias,dtype=wp.float64,device=self.device)
        else:
            j.native_regularization = wp.empty(0,dtype=wp.float64,device=self.device)
            j.native_bias = wp.empty(0,dtype=wp.float64,device=self.device)
        j.config.assign(extended_cost_config(config))
        j.valid.assign(np.asarray(valid,dtype=np.int32))
        j.seed.assign(np.array([seed],dtype=np.int32))
        if self.jobs is None:
            self.jobs = wp.array([j],dtype=ForceJob,device=self.device)
        else:
            self.jobs.assign([j])

    def _generations(self):
        j = self.job
        wp.launch(initialize,self.count,inputs=[self.jobs],device=self.device)
        for generation in range(self.search_iterations):
            wp.launch(sample,(self.count,j.population),inputs=[self.jobs,generation],device=self.device)
            wp.launch(evaluate,(self.count,j.population),inputs=[self.jobs],device=self.device)
            wp.launch(update,self.count,inputs=[self.jobs],device=self.device)

    def commands(self,raw,indices):
        raw=np.atleast_2d(raw)
        normal=np.clip(raw[:,0],0,1)*self.job.cap
        tangent=2*np.clip(raw[:,1:],0,1)-1
        tangent/=np.maximum(1.,np.abs(tangent).sum(axis=1,keepdims=True))
        return np.column_stack((normal,self.job.mu*normal[:,None]*tangent))

    def evaluate_raw(self,raw,indices):
        """Batch arbitrary per-contact queries through the same contact law."""
        raw,indices = np.asarray(raw).reshape(-1,3),np.asarray(indices,dtype=int)
        counts = np.bincount(indices,minlength=self.count)
        if counts.max(initial=0)>self.job.population:
            raise ValueError('Batch exceeds per-contact sample capacity')
        samples = np.tile([0.,.5,.5],(self.count,self.job.population,1))
        offsets = np.zeros(self.count,dtype=int)
        slots = np.empty(len(indices),dtype=int)
        for k,index in enumerate(indices):
            slots[k]=offsets[index];samples[index,slots[k]]=raw[k];offsets[index]+=1
        self.job.samples.assign(samples)
        wp.launch(evaluate,(self.count,self.job.population),inputs=[self.jobs],device=self.device)
        velocity = self.job.velocities.numpy()[indices,slots]
        pose = self.job.poses.numpy()[indices,slots]
        return self.commands(raw,indices),velocity,np.column_stack((pose[:,:3],pose[:,[6,3,4,5]])),self.job.costs.numpy()[indices,slots]

    def solve(self,iterations=None):
        # The graph reads a device struct updated for every observation.
        # Kernels and native solver workspaces are compiled/captured once.
        self.search_iterations = self.args.population_iterations if iterations is None else int(iterations)
        if not 1 <= self.search_iterations <= self.args.population_iterations:
            raise ValueError('CEM iterations exceed the configured search budget')
        if self.search_iterations not in self.graphs:
            with wp.ScopedCapture(device=self.device,force_module_load=False) as capture:
                self._generations()
            self.graphs[self.search_iterations] = capture.graph
        self.graph = self.graphs[self.search_iterations]
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
        j = self.job
        costs = j.best_cost.numpy()
        idx = int(np.argmin(costs))
        if costs[idx] >= 1.e99 or not np.isfinite(costs[idx]):
            raise FloatingPointError('No finite certified CEM action')
        raw = j.best.numpy()[idx]
        force = self.commands(raw,[idx])[0]
        pose = j.best_pose.numpy()[idx]
        return dict(selected_idx=idx,lam=force,cost=float(costs[idx]),
                    predicted_pose=np.r_[pose[:3],pose[[6,3,4,5]]],
                    velocity=j.best_velocity.numpy()[idx],alpha=float(j.best_alpha.numpy()[idx]),
                    residual=float(j.best_residual.numpy()[idx]),
                    evaluations=int(np.count_nonzero(j.valid.numpy()))*j.population*self.search_iterations,
                    search_iterations=self.search_iterations,
                    solver_backend='cuda-cem',gpu_parallel_prediction=True)


@wp.kernel
def native_reset(jobs: wp.array(dtype=ForceJob), qpos: wp.array2d(dtype=wp.float32),
                 qvel: wp.array2d(dtype=wp.float32)):
    j = jobs[0]
    world = wp.tid()
    p,q = wp.transform_get_translation(j.pose),wp.transform_get_rotation(j.pose)
    world_v = wp.quat_rotate(q,wp.vec3d(j.observed[0],j.observed[1],j.observed[2]))
    for k in range(3):
        qpos[world,k] = wp.float32(p[k])
        qvel[world,k] = wp.float32(world_v[k])
        qvel[world,k+3] = wp.float32(j.observed[k+3])
    qpos[world,3] = wp.float32(q[3])
    for k in range(3):
        qpos[world,k+4] = wp.float32(q[k])


@wp.kernel
def native_force(jobs: wp.array(dtype=ForceJob), applied: wp.array2d(dtype=wp.float32),
                 warmstart: wp.array2d(dtype=wp.float32)):
    j = jobs[0]
    c,s = wp.tid()
    world = c*j.population+s
    f = j.frame[c]*feasible(j.samples[c,s],j.cap,j.mu)
    torque = wp.cross(j.point[c],f)
    f = wp.quat_rotate(wp.transform_get_rotation(j.pose),f)
    for k in range(3):
        applied[world,k] = wp.float32(f[k])
        applied[world,k+3] = wp.float32(torque[k])
    for k in range(6):
        warmstart[world,k] = wp.float32(0.)


@wp.kernel
def native_score(jobs: wp.array(dtype=ForceJob), qacc: wp.array2d(dtype=wp.float32)):
    j = jobs[0]
    c,s = wp.tid()
    j.costs[c,s] = wp.float64(1.e100)
    if j.valid[c] == 0:
        return
    world = c*j.population+s
    a = wp.quat_rotate_inv(wp.transform_get_rotation(j.pose),wp.vec3d(wp.float64(qacc[world,0]),wp.float64(qacc[world,1]),wp.float64(qacc[world,2])))
    v = j.observed+j.h*V6(a[0],a[1],a[2],wp.float64(qacc[world,3]),wp.float64(qacc[world,4]),wp.float64(qacc[world,5]))
    f = feasible(j.samples[c,s],j.cap,j.mu)
    j.costs[c,s] = pose_cost(integrate(j.pose,v,j.h),j.target,j.config)+j.force_reg*wp.dot(f,f)+j.tangent_reg*(f[1]*f[1]+f[2]*f[2])
    j.velocities[c,s] = v
    j.poses[c,s] = integrate(j.pose,v,j.h)
    j.alphas[c,s] = wp.float64(1.)
    j.residuals[c,s] = wp.float64(0.)


class GPUQPForceCEM(GPUForceCEM):
    def __init__(self, points, frames, args, model):
        import copy
        import mujoco
        from comfree_warp import mujoco_warp
        from comfree_warp.mujoco_warp._src import solver
        super().__init__(points,frames,args,'qp')
        self.api,self.native_solver = mujoco_warp,solver
        self.native_model = copy.copy(model)
        self.native_model.opt.timestep = args.policy_dt
        self.native_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        self.native_model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        with wp.ScopedDevice(self.device):
            self.wmodel = self.api.put_model(self.native_model)
            # The workstation's CUDA 12.2 driver has no conditional graphs.
            self.wmodel.opt.graph_conditional = False
            self.wdata = self.api.put_data(self.native_model,mujoco.MjData(self.native_model),
                nworld=self.count*args.population_size,nconmax=16,njmax=128)

    def prepare(self,*args,**kwargs):
        native_data = kwargs.get('native_data')
        super().prepare(*args,**kwargs)
        import mujoco
        with wp.ScopedDevice(self.device):
            # Use the SAME CPU collision/constraint snapshot as every other
            # model. Re-detecting mesh contacts on GPU changes the QP itself.
            # Only the native Newton force solve is batched over GPU worlds.
            if native_data is None:
                raise ValueError('Native QP needs the common measured-state snapshot')
            d,m = native_data,self.native_model
            def broadcast(target,value):
                target.assign(np.ascontiguousarray(np.broadcast_to(value,target.shape),dtype=np.float32
                    if target.dtype == wp.float32 else np.int32))
            mass = np.zeros((m.nv,m.nv))
            mujoco.mj_fullM(m,mass,d.qM)
            padded = np.zeros(self.wdata.qM.shape[1:])
            padded[:m.nv,:m.nv] = mass
            broadcast(self.wdata.qM,padded)
            broadcast(self.wdata.qLD,np.linalg.cholesky(mass))
            for name in ('qpos','qvel','qfrc_bias','qfrc_passive'):
                broadcast(getattr(self.wdata,name),getattr(d,name))
            self.wdata.nefc.fill_(int(d.nefc))
            self.wdata.ne.fill_(int(d.ne))
            self.wdata.nf.fill_(int(d.nf))
            self.wdata.nacon.fill_(int(d.ncon)*self.wdata.nworld)
            for name in ('D','aref','frictionloss','type','id'):
                target = getattr(self.wdata.efc,name)
                values = np.zeros(target.shape[1:])
                values[:d.nefc] = getattr(d,'efc_'+name)
                broadcast(target,values)
            jac = np.zeros(self.wdata.efc.J.shape[1:])
            jac[:d.nefc,:m.nv] = d.efc_J.reshape(d.nefc,m.nv)
            broadcast(self.wdata.efc.J,jac)
            self.wdata.qfrc_applied.zero_()
            self.wdata.xfrc_applied.zero_()
            self.wdata.qacc_warmstart.zero_()
            self.api.fwd_acceleration(self.wmodel,self.wdata)
            # Compile the native kernels before CEM graph capture.
            if self.graph is None:
                self.native_solver.solve(self.wmodel,self.wdata)
            # The solve-only dense tile differs from factorize+solve; load it
            # before capture as CUDA 12.2 cannot load modules while capturing.
            if self.graph is None:
                self.api.fwd_acceleration(self.wmodel,self.wdata)
        wp.synchronize_device(self.device)
        if int(native_data.nefc)>self.wdata.njmax:
            raise RuntimeError('MuJoCo Warp contact/constraint buffer overflow')

    def _generations(self):
        j = self.job
        wp.launch(initialize,self.count,inputs=[self.jobs],device=self.device)
        for generation in range(self.search_iterations):
            wp.launch(sample,(self.count,j.population),inputs=[self.jobs,generation],device=self.device)
            wp.launch(native_force,(self.count,j.population),inputs=[self.jobs,self.wdata.qfrc_applied,self.wdata.qacc_warmstart],device=self.device)
            self.api.fwd_acceleration(self.wmodel,self.wdata)
            self.native_solver.solve(self.wmodel,self.wdata)
            wp.launch(native_score,(self.count,j.population),inputs=[self.jobs,self.wdata.qacc],device=self.device)
            wp.launch(update,self.count,inputs=[self.jobs],device=self.device)
