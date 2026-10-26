"""Rigid Signorini--Coulomb NCP with the exact circular friction disk.

The geometry/Jacobians are frozen over a time step (standard first-order
impulse time stepping). There is no diagonal Delassus approximation, friction
pyramid, softplus, compliance, or fixed number of Gauss--Seidel sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class NCPSettings:
    tolerance: float = 1e-7  # natural-map residual, m/s
    max_evaluations: int = 500
    max_projection_iterations: int = 50000

    def __post_init__(self):
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("NCP tolerance must be finite and positive")
        if self.max_evaluations < 1:
            raise ValueError("NCP max_evaluations must be positive")
        if self.max_projection_iterations < 0:
            raise ValueError("NCP max_projection_iterations cannot be negative")


def natural_map(impulse, W, b, mu, rho, *, jacobian=False):
    """Projection residual / rho; b normal includes signed gap / timestep.

    p_n = max(0, p_n-rho*w_n)
    p_t = proj_disk(mu*p_n)(p_t-rho*u_t).

    At a zero these express 0 <= p_n perpendicular w_n >= 0 and
    p_t in argmin_{||z|| <= mu*p_n} u_t.dot(z). In particular the normal
    equation has NO mu*||u_t|| term (which would introduce dilatancy).
    """
    p = np.asarray(impulse).reshape(-1, 3)
    u = (W @ p.ravel() + b).reshape(-1, 3)
    rho = np.asarray(rho).reshape(-1)
    z = p-rho[:, None]*u
    residual = p.copy()
    residual[:, 0] -= np.maximum(z[:, 0], 0.)
    if not jacobian:
        radius = mu*np.maximum(p[:, 0], 0.)
        length = np.linalg.norm(z[:, 1:], axis=1)
        factor = np.ones_like(length)
        np.divide(radius,length,out=factor,where=length>radius)
        residual[:, 1:] -= z[:, 1:]*factor[:, None]
        return (residual/rho[:, None]).ravel()
    eye = np.eye(p.size) if jacobian else None
    derivative = eye.copy() if jacobian else None
    for i in range(len(p)):
        j = 3*i
        radius = mu[i]*max(p[i, 0], 0.)
        length = np.linalg.norm(z[i, 1:])
        if jacobian:
            dz = eye[j:j+3]-rho[i]*W[j:j+3]
            if z[i, 0] > 0:
                derivative[j] -= dz[0]
        if length <= radius:
            residual[i, 1:] -= z[i, 1:]
            if jacobian:
                derivative[j+1:j+3] -= dz[1:]
        elif length > 0:
            direction = z[i, 1:]/length
            residual[i, 1:] -= radius*direction
            if jacobian:
                tangent = radius/length*(np.eye(2)-np.outer(direction, direction))
                derivative[j+1:j+3] -= tangent @ dz[1:]
                if p[i, 0] > 0:
                    derivative[j+1:j+3, j] -= mu[i]*direction
    residual /= rho[:, None]
    if jacobian:
        derivative /= np.repeat(rho, 3)[:, None]
        return residual.ravel(), derivative
    return residual.ravel()


def solve_ncp(free_velocity, J, gap, inverse_mass, timestep, friction,
              settings=NCPSettings(), *, _active_set=True):
    """Return velocity, impulses, diagnostics; a finite iterate is not success.

    Trust-region least squares with an analytic generalized Jacobian solves
    the *unsmoothed* natural map. Redundant contacts are retained; the TRF
    linear algebra can handle singular Jacobians without regularizing W.
    """
    free = np.asarray(free_velocity, float)
    gap = np.asarray(gap, float).reshape(-1)
    J = np.asarray(J, float)
    inverse = np.asarray(inverse_mass, float)
    mu = np.broadcast_to(np.asarray(friction, float), gap.shape).copy()
    if (free.ndim != 1 or J.shape != (len(gap), 3, free.size)
            or inverse.shape != (free.size, free.size) or not np.isfinite(timestep)
            or timestep <= 0 or np.any(mu < 0)
            or not all(np.isfinite(x).all() for x in (free, J, gap, inverse, mu))):
        raise ValueError("Invalid finite NCP problem dimensions/parameters")
    if not np.allclose(inverse, inverse.T, atol=1e-12) or np.linalg.eigvalsh(inverse).min() <= 0:
        raise ValueError("inverse_mass must be symmetric positive definite")
    if not len(gap):
        return free.copy(), np.empty((0, 3)), dict(converged=True, n_contacts=0,
            evaluations=0, residual_m_s=0., normal_violation_m_s=0.,
            cone_violation_Ns=0., complementarity_Ns_m_s=0.,
            friction_dissipation_defect_Ns_m_s=0.)
    jac = J.reshape(-1, free.size)
    W = jac @ inverse @ jac.T
    b = jac @ free
    b[::3] += gap/timestep
    diagonal = np.diag(W).reshape(-1, 3)
    if np.any(diagonal[:, 0] <= 0):
        raise ValueError("Every contact must have positive normal mobility")
    if _active_set and len(gap)>1:
        # Constraint generation is exact: solve a subset, then check EVERY
        # original normal inequality and circular friction residual. Omitted
        # contacts have p=0; if their normal velocity is nonnegative they
        # satisfy the original NCP exactly, including the tangential law.
        # This avoids singular impulse distributions over a flat patch without
        # merging points, modifying W, or discarding an unchecked constraint.
        active=[]
        candidate=int(np.argmin(b[::3]))
        evaluations=projection_iterations=0
        for outer in range(min(len(gap),32)):
            active.append(candidate)
            selected=np.asarray(active)
            next_v, next_p, info=solve_ncp(free,J[selected],gap[selected],inverse,timestep,
                                          mu[selected],settings,_active_set=False)
            evaluations+=info['evaluations']
            projection_iterations+=info.get('projection_iterations',0)
            if not info['converged']: break
            full_p=np.zeros((len(gap),3));full_p[selected]=next_p
            full_rho=1/np.max(diagonal,axis=1)
            full_residual=float(np.max(np.abs(natural_map(full_p,W,b,mu,full_rho))))
            if full_residual<=settings.tolerance:
                return next_v,full_p,dict(info,**_quality(next_v,full_p,jac,gap,timestep,mu,
                    full_residual,settings),n_contacts=len(gap),active_contacts=len(active),
                    active_set_iterations=outer+1,evaluations=evaluations,
                    projection_iterations=projection_iterations,
                    full_candidate_residual_checked=True)
            normal=(jac@next_v).reshape(-1,3)[:,0]+gap/timestep
            normal[selected]=np.inf
            candidate=int(np.argmin(normal))
            if normal[candidate]>=-settings.tolerance: break
    velocity_scale = max(float(np.max(np.abs(b))), .1)
    impulse_scale = velocity_scale/float(np.max(diagonal))
    scaled_W, scaled_b = W*impulse_scale/velocity_scale, b/velocity_scale
    rho = 1/np.max(np.diag(scaled_W).reshape(-1, 3), axis=1)
    initial = np.zeros_like(b)
    initial[::3] = np.maximum(-b[::3]/diagonal[:, 0], 0.)/impulse_scale
    # These are numerical termination controls, not changes to the contact law.
    def refine(start):
        return least_squares(
            lambda x: natural_map(x, scaled_W, scaled_b, mu, rho), start,
            jac=lambda x: natural_map(x, scaled_W, scaled_b, mu, rho, jacobian=True)[1],
            method="trf", ftol=1e-13, xtol=1e-13, gtol=1e-13,
            max_nfev=settings.max_evaluations)
    result = refine(initial)
    evaluations, projection_iterations = int(result.nfev), 0
    solution, residual_vector = result.x, result.fun
    if np.max(np.abs(residual_vector))*velocity_scale > settings.tolerance:
        # Redundant patches can trap a local least-squares minimizer at nonzero
        # residual. A globally scaled projected iteration supplies a feasible
        # alternative start, without changing W or the circular friction law.
        # This is a convergence-controlled solver, never a fixed-sweep model.
        step = .8/np.linalg.eigvalsh(scaled_W)[-1]
        projected = np.zeros_like(initial)
        for iteration in range(settings.max_projection_iterations):
            u = (jac @ (inverse @ (jac.T @ projected))*(impulse_scale/velocity_scale)+scaled_b).reshape(-1,3)
            trial = projected.reshape(-1,3)-step*u
            normal = np.maximum(trial[:,0],0.)
            tangent = trial[:,1:]
            length=np.linalg.norm(tangent,axis=1);radius=mu*normal
            factor=np.ones_like(length)
            np.divide(radius,length,out=factor,where=length>radius)
            tangent *= factor[:,None]
            projected = np.column_stack((normal,tangent)).ravel()
            projection_iterations = iteration+1
            if projection_iterations % 100 == 0:
                current = natural_map(projected,scaled_W,scaled_b,mu,rho)
                if np.max(np.abs(current))*velocity_scale <= settings.tolerance*.1:
                    solution,residual_vector = projected,current
                    break
        else:
            current = natural_map(projected,scaled_W,scaled_b,mu,rho)
            if np.linalg.norm(current) < np.linalg.norm(residual_vector):
                solution,residual_vector = projected,current
        if np.max(np.abs(residual_vector))*velocity_scale > settings.tolerance:
            refined = refine(projected)
            evaluations += int(refined.nfev)
            if np.linalg.norm(refined.fun) < np.linalg.norm(residual_vector):
                solution,residual_vector,result = refined.x,refined.fun,refined
    impulse = solution.reshape(-1, 3)*impulse_scale
    velocity = free + inverse @ jac.T @ impulse.ravel()
    residual = float(np.max(np.abs(residual_vector))*velocity_scale)
    diagnostics = dict(**_quality(velocity,impulse,jac,gap,timestep,mu,residual,settings), n_contacts=len(gap),
        evaluations=evaluations, projection_iterations=projection_iterations,
        optimizer_success=bool(result.success), optimizer_message=str(result.message))
    return velocity, impulse, diagnostics


def _quality(velocity,impulse,jac,gap,timestep,mu,residual,settings):
    u=(jac@velocity).reshape(-1,3)
    normal_velocity=u[:,0]+gap/timestep
    return dict(converged=bool(residual<=settings.tolerance),residual_m_s=residual,
        normal_violation_m_s=float(max(0.,-normal_velocity.min())),
        cone_violation_Ns=float(max(0.,np.max(np.linalg.norm(impulse[:,1:],axis=1)-mu*impulse[:,0]),
                                    -impulse[:,0].min())),
        complementarity_Ns_m_s=float(np.max(np.abs(impulse[:,0]*normal_velocity))),
        friction_dissipation_defect_Ns_m_s=float(np.max(np.abs(np.sum(impulse[:,1:]*u[:,1:],axis=1)
            +mu*impulse[:,0]*np.linalg.norm(u[:,1:],axis=1)))))
