"""CUDA batched finite-difference LM for the paired point-force experiment.

Each contact has an independent three-variable bounded least-squares problem.
All residuals, Jacobians, damped systems and accept/reject updates stay on CUDA.
The measured contact geometry is shared with the CPU reference and CUDA CEM.
"""
import numpy as np
import warp as wp
from planning.point_force_gpu import ForceJob, GPUForceCEM, evaluate, bounded_force

R22 = wp.types.vector(22, wp.float64)


@wp.func
def residual(j: ForceJob, c: int, s: int):
    pose, v = j.poses[c,s], j.velocities[c,s]
    p, q = wp.transform_get_translation(pose), wp.transform_get_rotation(pose)
    qd = wp.transform_get_rotation(j.target)
    delta = p-wp.transform_get_translation(j.target)
    up = wp.vec3d(j.config[6],j.config[7],j.config[8])
    if j.config[9] > wp.float64(0.):
        planar = delta-up*wp.dot(delta,up)
        norm = wp.length(planar)
        delta = planar*(wp.max(norm-j.config[9],wp.float64(0.))/wp.max(norm,wp.float64(1.e-30)))
    axis = wp.quat_rotate(q,wp.vec3d(j.config[3],j.config[4],j.config[5]))
    qr = q-wp.dot(q,qd)*qd
    upright = axis-up
    f = bounded_force(j,c,j.samples[c,s])
    r = R22(wp.float64(0.))
    for k in range(3):
        r[k] = wp.sqrt(j.config[0])*delta[k]
        r[7+k] = wp.sqrt(j.config[2]/wp.float64(4.))*upright[k]
        r[11+k] = wp.sqrt(j.linear_velocity_reg)*v[k]
        r[14+k] = wp.sqrt(j.angular_velocity_reg)*v[k+3]
        r[17+k] = wp.sqrt(j.force_reg)*f[k]
    for k in range(4):
        r[3+k] = wp.sqrt(j.config[1])*qr[k]
    r[10] = wp.sqrt(j.config[11])*wp.max(wp.float64(1.)-wp.dot(axis,up)-j.config[10],wp.float64(0.))
    r[20] = wp.sqrt(j.tangent_reg)*f[1]
    r[21] = wp.sqrt(j.tangent_reg)*f[2]
    return r


@wp.func
def retain(j: ForceJob, c: int, s: int):
    if j.costs[c,s] < j.best_cost[c]:
        j.best_cost[c] = j.costs[c,s]
        j.best[c] = j.samples[c,s]
        j.best_velocity[c] = j.velocities[c,s]
        j.best_pose[c] = j.poses[c,s]
        j.best_alpha[c] = j.alphas[c,s]
        j.best_residual[c] = j.residuals[c,s]


@wp.kernel
def starts(jobs: wp.array(dtype=ForceJob), warm: wp.array(dtype=wp.vec3d), damping: wp.array(dtype=wp.float64)):
    j = jobs[0]
    c = wp.tid()
    j.samples[c,0] = warm[c]
    j.samples[c,1] = wp.vec3d(wp.float64(0.),wp.float64(.5),wp.float64(.5))
    j.samples[c,2] = wp.vec3d(wp.float64(1.),wp.float64(.5),wp.float64(.5))
    j.best_cost[c] = wp.float64(1.e100)
    damping[c] = wp.float64(1.e-3)


@wp.kernel
def choose_start(jobs: wp.array(dtype=ForceJob)):
    j = jobs[0]
    c = wp.tid()
    winner = int(0)
    for s in range(3):
        if j.costs[c,s] < j.costs[c,winner]:
            winner = s
    j.mean[c] = j.samples[c,winner]
    retain(j,c,winner)


@wp.kernel
def differences(jobs: wp.array(dtype=ForceJob)):
    j = jobs[0]
    c,s = wp.tid()
    raw = j.mean[c]
    if s > 0:
        axis = (s-1)//2
        if s%2 == 1:
            raw[axis] += wp.float64(1.e-4)
        else:
            raw[axis] -= wp.float64(1.e-4)
    j.samples[c,s] = raw


@wp.func
def solve3(matrix: wp.mat33d, rhs: wp.vec3d):
    lower = wp.mat33d(wp.float64(0.))
    for i in range(3):
        for k in range(i+1):
            value = matrix[i,k]
            for l in range(k):
                value -= lower[i,l]*lower[k,l]
            if i == k:
                lower[i,k] = wp.sqrt(wp.max(value,wp.float64(1.e-30)))
            else:
                lower[i,k] = value/lower[k,k]
    mid = wp.vec3d(wp.float64(0.))
    for i in range(3):
        value = rhs[i]
        for k in range(i):
            value -= lower[i,k]*mid[k]
        mid[i] = value/lower[i,i]
    result = wp.vec3d(wp.float64(0.))
    for reverse in range(3):
        i = 2-reverse
        value = mid[i]
        for k in range(i+1,3):
            value -= lower[k,i]*result[k]
        result[i] = value/lower[i,i]
    return result


@wp.kernel
def proposal(jobs: wp.array(dtype=ForceJob), damping: wp.array(dtype=wp.float64), base_cost: wp.array(dtype=wp.float64)):
    j = jobs[0]
    c = wp.tid()
    if j.valid[c] == 0:
        return
    base_cost[c] = j.costs[c,0]
    retain(j,c,0)
    base = residual(j,c,0)
    dx = (residual(j,c,1)-residual(j,c,2))/wp.float64(2.e-4)
    dy = (residual(j,c,3)-residual(j,c,4))/wp.float64(2.e-4)
    dz = (residual(j,c,5)-residual(j,c,6))/wp.float64(2.e-4)
    a = wp.dot(dx,dx)
    b = wp.dot(dy,dy)
    d = wp.dot(dz,dz)
    xy,xz,yz = wp.dot(dx,dy),wp.dot(dx,dz),wp.dot(dy,dz)
    system = wp.mat33d(a+damping[c]*wp.max(a,wp.float64(1.)),xy,xz,
                      xy,b+damping[c]*wp.max(b,wp.float64(1.)),yz,
                      xz,yz,d+damping[c]*wp.max(d,wp.float64(1.)))
    step = solve3(system,-wp.vec3d(wp.dot(dx,base),wp.dot(dy,base),wp.dot(dz,base)))
    raw = j.mean[c]+step
    for k in range(3):
        raw[k] = wp.clamp(raw[k],wp.float64(0.),wp.float64(1.))
    j.samples[c,0] = raw


@wp.kernel
def accept(jobs: wp.array(dtype=ForceJob), damping: wp.array(dtype=wp.float64), base_cost: wp.array(dtype=wp.float64), warm: wp.array(dtype=wp.vec3d)):
    j = jobs[0]
    c = wp.tid()
    if j.valid[c] == 0:
        return
    if j.costs[c,0] < base_cost[c]:
        j.mean[c] = j.samples[c,0]
        damping[c] *= wp.float64(.4)
    else:
        damping[c] *= wp.float64(4.)
    damping[c] = wp.clamp(damping[c],wp.float64(1.e-9),wp.float64(1.e9))
    warm[c] = j.mean[c]
    retain(j,c,0)


class GPUForceLM(GPUForceCEM):
    """Fixed-budget batch LM; prediction and optimization run in float64 CUDA."""
    def __init__(self, points, frames, args, model):
        super().__init__(points,frames,args,model)
        if self.job.population < 7:
            raise ValueError('GPU LM needs seven finite-difference slots per contact')
        self.warm = wp.array(np.tile([.1,.5,.5],(self.count,1)),dtype=wp.vec3d,device=self.device)
        self.damping = wp.empty(self.count,dtype=wp.float64,device=self.device)
        self.base_cost = wp.empty(self.count,dtype=wp.float64,device=self.device)
        wp.load_module(module=__name__,device=self.device)

    def _iterations(self,iterations):
        wp.launch(starts,self.count,inputs=[self.jobs,self.warm,self.damping],device=self.device)
        wp.launch(evaluate,(self.count,3),inputs=[self.jobs],device=self.device)
        wp.launch(choose_start,self.count,inputs=[self.jobs],device=self.device)
        for _ in range(iterations):
            wp.launch(differences,(self.count,7),inputs=[self.jobs],device=self.device)
            wp.launch(evaluate,(self.count,7),inputs=[self.jobs],device=self.device)
            wp.launch(proposal,self.count,inputs=[self.jobs,self.damping,self.base_cost],device=self.device)
            wp.launch(evaluate,(self.count,1),inputs=[self.jobs],device=self.device)
            wp.launch(accept,self.count,inputs=[self.jobs,self.damping,self.base_cost,self.warm],device=self.device)

    def solve(self,iterations=None):
        iterations = self.args.lm_iterations if iterations is None else int(iterations)
        if iterations < 1:
            raise ValueError('LM requires a positive iteration budget')
        if iterations not in self.graphs:
            with wp.ScopedCapture(device=self.device,force_module_load=False) as capture:
                self._iterations(iterations)
            self.graphs[iterations] = capture.graph
        self.graph = self.graphs[iterations]
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
        j = self.job
        costs = j.best_cost.numpy()
        idx = int(np.argmin(costs))
        if costs[idx] >= 1.e99 or not np.isfinite(costs[idx]):
            raise FloatingPointError('No finite GPU LM action')
        raw = j.best.numpy()[idx]
        pose = j.best_pose.numpy()[idx]
        return dict(selected_idx=idx,lam=self.commands(raw,[idx])[0],cost=float(costs[idx]),
                    predicted_pose=np.r_[pose[:3],pose[[6,3,4,5]]],
                    velocity=j.best_velocity.numpy()[idx],alpha=float(j.best_alpha.numpy()[idx]),
                    residual=float(j.best_residual.numpy()[idx]),
                    evaluations=int(np.count_nonzero(j.valid.numpy()))*(3+8*iterations),
                    search_iterations=iterations,solver_backend='cuda-batched-lm-finite-difference',
                    gpu_parallel_prediction=True)
