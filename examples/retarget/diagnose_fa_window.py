"""Replay one recorded OCP and audit its analytic reduced gradient and SQP step.

Scalar cost probes validate descent; they do not construct a finite-difference
Jacobian or supply derivatives to the optimizer.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--window',type=int,default=120)
    p.add_argument('--output',type=Path);a=p.parse_args()
    from examples.retarget.matrix_comfree import ComFreeEngine
    from examples.retarget.fa_direct_mpc import DirectComFreeMPC
    from examples.retarget.fa_mpc_diagnostics import split_cost
    e=ComFreeEngine(Path('outputs/fa_admm_solver_20261016'),Path('thirdparty/retarget_assets/DexForge'))
    c=DirectComFreeMPC(e)
    from forceaware.targets import make_window
    saved=np.load(a.run/f'diagnostics/window_{a.window:04d}.npz')
    target=make_window(e.sequence,executed_steps=a.window*3,horizon=5,time_grid=e.cfg.simulator.time_grid,start_frame=0.,end_frame=len(e.sequence.hand)-1.)
    target=replace(target,contact_position_object=saved['template'])
    c.native.prepare(target);states=saved['states'];commands=saved['controls'];parameters=saved['parameters']
    physical=states[0,:c.nphysical]*c.physical_scale
    rebuilt=c.rollout(physical,commands,saved['previous'],saved['previous_previous'])
    error=float(np.max(np.abs(rebuilt-states)));assert error<2e-4,error
    baseline=c.trajectory_cost(states,commands,parameters)
    previous_path=a.run/f'diagnostics/window_{a.window-1:04d}.npz'
    warm_audit=None
    if previous_path.exists():
        old=np.load(previous_path)['controls'];warm=np.concatenate([old[3:],np.repeat(old[-1:],3,axis=0)])
        warm=np.clip(warm,c.lower,c.upper)
        warm_states=c.rollout(physical,warm,saved['previous'],saved['previous_previous'])
        parts=[]
        for k in range(c.N):parts.append(split_cost(np.asarray(c.residual_function(warm_states[k],warm[k]/c.hand_scale,parameters[k])),c.native))
        parts.append(split_cost(np.asarray(c.geometry_function(warm_states[-1],parameters[-1])),c.native))
        warm_audit=dict(cost=sum(r['total'] for r in parts),components={name:sum(r[name] for r in parts) for name in parts[0]},
            terminal_object_position=(warm_states[-1,26:29]*c.physical_scale[26:29]).tolist(),
            accepted_terminal_object_position=(states[-1,26:29]*c.physical_scale[26:29]).tolist())
    s=c.solver
    for k in range(c.N+1):
        s.set(k,'x',states[k]);s.set(k,'p',parameters[k])
        if k<c.N:
            u=commands[k]/c.hand_scale;s.set(k,'u',u)
            s.constraints_set(k,'lbu',np.maximum(c.lower/c.hand_scale,u-c.trust))
            s.constraints_set(k,'ubu',np.minimum(c.upper/c.hand_scale,u+c.trust))
    s.set(0,'lbx',states[0]);s.set(0,'ubx',states[0]);status=s.solve()
    trial=np.array([s.get(k,'u') for k in range(c.N)])*c.hand_scale;direction=trial-commands
    stage_jac=c.residual_function.jacobian();terminal_jac=c.geometry_function.jacobian()
    terminal=np.asarray(c.geometry_function(states[-1],parameters[-1])).ravel()
    multiplier=2*terminal@np.asarray(terminal_jac(states[-1],parameters[-1],terminal)[0])
    gradient=np.zeros_like(commands)
    for k in reversed(range(c.N)):
        u=commands[k]/c.hand_scale;r=np.asarray(c.residual_function(states[k],u,parameters[k])).ravel()
        jac=stage_jac(states[k],u,parameters[k],r);lx=2*r@np.asarray(jac[0]);lu=2*r@np.asarray(jac[1])
        _,J=c.native.physics(states[k,:c.nphysical]*c.physical_scale,commands[k],True)
        A=np.zeros((c.nx,c.nx));B=np.zeros((c.nx,c.nu))
        A[:c.nphysical,:c.nphysical]=J[:c.nphysical,:c.nphysical]*c.physical_scale[None,:]/c.physical_scale[:,None]
        B[:c.nphysical]=J[:c.nphysical,c.nphysical:]*c.hand_scale[None,:]/c.physical_scale[:,None]
        A[c.nphysical+c.nu:,c.nphysical:c.nphysical+c.nu]=np.eye(c.nu);B[c.nphysical:c.nphysical+c.nu]=np.eye(c.nu)
        gradient[k]=(lu+multiplier@B)/c.hand_scale
        multiplier=lx+multiplier@A
    dot=float(np.sum(gradient*direction));probes=[]
    for alpha in (1.,.5,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625,0.):
        u=commands+alpha*direction
        x=c.rollout(physical,u,saved['previous'],saved['previous_previous']);cost=c.trajectory_cost(x,u,parameters)
        components=[]
        for k in range(c.N):components.append(split_cost(np.asarray(c.residual_function(x[k],u[k]/c.hand_scale,parameters[k])),c.native))
        components.append(split_cost(np.asarray(c.geometry_function(x[-1],parameters[-1])),c.native))
        probes.append(dict(alpha=alpha,cost=cost,change=cost-baseline,first_order_change=alpha*dot,
            components={name:sum(r[name] for r in components) for name in components[0]}))
    result=dict(run=str(a.run),window=a.window,time=a.window*.03,status=int(status),replay_error=error,
        baseline_cost=baseline,analytic_gradient_norm=float(np.linalg.norm(gradient)),
        analytic_SQP_directional_derivative=dot,probes=probes,warm_start_comparison=warm_audit,finite_difference_Jacobians=False)
    out=a.output or a.run/f'window_{a.window:04d}_descent.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
