"""Match complete FA tracking/geometry gradients to the post-tracking OCP.

Both sides use analytic differentiation. Smoothing weights are zeroed on both
sides so the separately audited smoothing grids do not contaminate the check.
"""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np


def main():
    from examples.retarget.matrix_comfree import ComFreeEngine
    e=ComFreeEngine(Path('outputs/fa_admm_solver_20261016'),Path('thirdparty/retarget_assets/DexForge'))
    from examples.retarget.fa_contact_admm import ADMMSettings,optimizer_class
    from examples.retarget.fa_direct_mpc import DirectComFreeMPC
    from examples.retarget.fa_matched_ablation import expand
    from forceaware.targets import make_window
    import mujoco
    e.cfg=replace(e.cfg,optimizer=replace(e.cfg.optimizer,multistart_count=1,n_iter=40),smoothing=replace(e.cfg.smoothing,
        control_velocity_weight=0.,control_acceleration_weight=0.,hand_state_weight=0.))
    root=Path('outputs/fa_matched_ablation_20261016')
    c=DirectComFreeMPC(e,post_step_tracking=True,cache=root/'validation_geometry_ocp')
    native=mujoco.MjModel.from_xml_path('outputs/fa_admm_solver_20261016/sequence/scene/wuji_right/sugar_box_mesh.xml')
    b=optimizer_class(native,ADMMSettings(optimize_contacts=False))(e.model,e.device_model,e.index,
        cfg=e.cfg,guidance_count=e.sequence.contact.mask.shape[1])
    history=np.load('outputs/fa_admm_solver_20261016/comfree/lbfgs_fixed/rollout.npz');results=[]
    P=expand(np.eye(5).T[...,None],np.zeros((5,1)))[:,:,0].T
    stage_jac=c.residual_function.jacobian();terminal_jac=c.geometry_function.jacobian()
    for window in (0,80,120):
        dense=window*3;saved=np.load(root/f'probes/baseline/window_{window:04d}.npz');q=history['qpos_traj'][dense];v=history['qvel_traj'][dense]
        previous=history['ctrl_traj'][dense-1] if dense else q[:26]
        target=make_window(e.sequence,executed_steps=dense,horizon=5,time_grid=e.cfg.simulator.time_grid,start_frame=0.,end_frame=71.)
        b.backend._prepare_window(q,v,target,warm=history['raw_warm_history'][dense],previous_action=previous,seed=window,start_time=dense*.01)
        b.evaluations=0;b._forward_variables=None
        old_value,old_gradient=b._evaluate(saved['raw'][26:][None].astype(float)/b.variable_scale,True)
        commands=np.stack([u.numpy()[0] for u in b.control.controls]);c.native.prepare(target)
        states=c.rollout(np.r_[q,v],commands,previous,previous)
        parameters=[np.r_[k,target.dense_actions[min(k,29)],target.object_qpos[min(k,29)],2. if k==30 else 0. if k==29 else 1.] for k in range(31)]
        value=c.trajectory_cost(states,commands,parameters)
        terminal=np.asarray(c.geometry_function(states[-1],parameters[-1])).ravel()
        multiplier=2*terminal@np.asarray(terminal_jac(states[-1],parameters[-1],terminal)[0]);gradient=np.zeros_like(commands)
        for k in reversed(range(30)):
            u=commands[k]/c.hand_scale;r=np.asarray(c.residual_function(states[k],u,parameters[k])).ravel()
            jac=stage_jac(states[k],u,parameters[k],r);lx=2*r@np.asarray(jac[0]);lu=2*r@np.asarray(jac[1])
            _,J=c.native.physics(states[k,:c.nphysical]*c.physical_scale,commands[k],True)
            A=np.zeros((c.nx,c.nx));B=np.zeros((c.nx,c.nu))
            A[:c.nphysical,:c.nphysical]=J[:c.nphysical,:c.nphysical]*c.physical_scale[None,:]/c.physical_scale[:,None]
            B[:c.nphysical]=J[:c.nphysical,c.nphysical:]*c.hand_scale[None,:]/c.physical_scale[:,None]
            A[c.nphysical+c.nu:,c.nphysical:c.nphysical+c.nu]=np.eye(c.nu);B[c.nphysical:c.nphysical+c.nu]=np.eye(c.nu)
            gradient[k]=(lu+multiplier@B)/c.hand_scale;multiplier=lx+multiplier@A
        pulled=(P.T@gradient).ravel()*b.variable_scale
        error=float(np.linalg.norm(pulled-old_gradient[0])/max(np.linalg.norm(old_gradient[0]),1e-12))
        row=dict(window=window,cost_FA=float(old_value[0]),cost_OCP=value,cost_relative_error=float(abs(value-old_value[0])/max(abs(old_value[0]),1e-8)),
            analytic_gradient_relative_error=error,analytic_FA_gradient_norm=float(np.linalg.norm(old_gradient[0])),
            analytic_OCP_gradient_norm=float(np.linalg.norm(pulled)),finite_difference_Jacobians=False)
        np.savez_compressed(root/f'gradient_validation_{window:04d}.npz',FA=old_gradient[0],OCP=pulled,commands=commands)
        results.append(row);print(json.dumps(row),flush=True)
        assert row['cost_relative_error']<2e-5,row
        assert error<.02,row
    (root/'full_horizon_gradient_validation.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
