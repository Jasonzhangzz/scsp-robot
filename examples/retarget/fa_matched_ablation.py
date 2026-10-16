"""Instance-local, one-factor changes to the successful FA single-shooting path.

The unchanged worker is the baseline. No finite-difference derivatives are
used. Dense controls keep the old five-endpoint smoothing functional so that
control dimension and smoothing can be tested independently.
"""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import warp as wp
from forceaware import mpc_loss as fa


def expand(values,previous):
    """The exact five-knot linear interpolation used by the baseline."""
    result=[]
    for k in range(5):
        left=previous if k==0 else values[...,k-1,:]
        for sub in range(6):result.append(left+(sub+1)/6*(values[...,k,:]-left))
    return np.stack(result,axis=-2)


def shift_dense_matched(raw,action_dim,*,execution_steps,warm_scale):
    """Extend the original five-knot warm policy: P * shift * endpoint_select.

    On an embedded coarse trajectory this commutes with the original warm
    resampling, avoiding a second factor in the control-dimension ablation.
    Newly optimized interior directions restart from the endpoint embedding.
    """
    from forceaware.control import resample_warm_corrections
    endpoints=raw[action_dim:].reshape(30,action_dim)[5::6]
    coarse=np.r_[raw[:action_dim],endpoints.ravel()]
    shifted=resample_warm_corrections(coarse,action_dim=action_dim,horizon=5,
        executed_dense_steps=execution_steps,physics_steps_per_knot=6,warm_scale=warm_scale)
    return np.r_[shifted[:action_dim],expand(shifted[action_dim:].reshape(5,action_dim),np.zeros(action_dim)).ravel()].astype(raw.dtype)


@wp.kernel(enable_backward=False)
def reduce_dense_gradient(large:wp.array2d(dtype=float),small:wp.array2d(dtype=float),P:wp.array2d(dtype=float)):
    world,knot,axis=wp.tid();value=float(0.)
    for dense in range(30):value+=P[dense,knot]*large[world,26*(1+dense)+axis]
    small[world,26*(1+knot)+axis]+=value


@wp.kernel(enable_backward=False)
def constant_tracking(job:fa.LossJob,out:wp.array(dtype=float)):
    world=wp.tid();out[world]=fa._reference_pose_loss(job,world)*fa.LOSS_GRADIENT_SCALE


def control_smoothing(regularizer,*,dense=False):
    """Change only the control smoothing grid; state regularization is kept."""
    r=regularizer
    if dense:
        expanded=wp.zeros((r.worlds,26*31),device=r.device)
        P=expand(np.eye(5).T[...,None],np.zeros((5,1)))[:,:,0].T.copy()
        interpolation=wp.array(P.astype(np.float32),device=r.device)
    def iterator(control):
        if dense:return list(enumerate(control.controls))
        return [(i,control.actions[i]) for i in (5,11,17,23,29)]
    def common(control,gradient):
        previous=control.previous_action
        previous_previous=r._dense_history if dense else r._previous_previous
        previous_dt=.01 if dense else .03;current_dt=.01 if dense else .06
        previous_step=-1;previous_previous_step=-1
        if gradient and dense:expanded.zero_()
        for index,current in iterator(control):
            job=fa.ControlSmoothnessJob();job.model=r.model;job.previous_previous=previous_previous
            job.previous=previous;job.current=current;job.components=r.runtime.components
            job.previous_dt=previous_dt;job.current_dt=current_dt
            job.acceleration_weight=r.cfg.smoothing.control_acceleration_weight
            if gradient:
                job.raw_gradient=expanded if dense else control.gradient
                job.previous_previous_step=previous_previous_step;job.previous_step=previous_step;job.current_step=index
                job.gradient_scale=fa.LOSS_GRADIENT_SCALE
            wp.launch(fa._control_smoothness_gradient if gradient else fa._control_smoothness_loss,
                (r.worlds,r.model.action_dim) if gradient else r.worlds,inputs=[job],device=r.device)
            previous_previous,previous=previous,current;previous_previous_step,previous_step=previous_step,index;previous_dt=current_dt
        if gradient and dense:wp.launch(reduce_dense_gradient,(r.worlds,5,26),inputs=[expanded,control.gradient,interpolation],device=r.device)
    r._evaluate_control_smoothness=lambda control:common(control,False)
    r._add_smooth_control_gradient=lambda control:common(control,True)
    if dense:r._ablation_owners=(expanded,interpolation)


def full_contact_gradient(backend):
    from comfree_warp.native_adjoint.geometry_vjp import GeometryTape
    from comfree_warp.native_adjoint import gaussian_collision
    from comfree_warp.native_adjoint.runtime import collision_contacts
    from examples.retarget.fa_adjoint_fusion import fuse_tape_zero,fuse_tape_seeds
    class FullContactTape(GeometryTape):
        def record_launch(self,kernel,dim,max_blocks,inputs,outputs,device,block_dim=0,metadata=None):
            if kernel in (gaussian_collision._narrow_pairs,gaussian_collision._narrow_plane):inputs[1].stop_frame_vjp=0
            super().record_launch(kernel,dim,max_blocks,inputs,outputs,device,block_dim=block_dim,metadata=metadata)
    replaced=[]
    for runtime in backend.steps:
        tape=FullContactTape()
        with tape:collision_contacts(backend.compiled.base,runtime.workspace.forward.base,freeze_frame_vjp=False)
        fuse_tape_zero(tape);fuse_tape_seeds(tape)
        replaced.append(replace(runtime,recorded=replace(runtime.recorded,contact_tape=tape)))
    backend.steps=tuple(replaced)


def install(solver,variant,native_model,output,*,sequence=None):
    b=solver.backend;prepare=b._prepare_window;plan=b._plan;state={}
    if variant=='full-contact-jacobian':
        from comfree_warp.collision_config import stop_normal_gradient,FREEZE_TANGENT_GAUGE_VJP
        import json
        batches=b.compiled.base.collision.batches
        policy=[dict(target_kind=int(batch.target.target.kind),topk=int(batch.model.contact_topk),
            stop_normal_gradient=bool(stop_normal_gradient(batch.target.target.kind,batch.model.contact_topk))) for batch in batches]
        noop=not FREEZE_TANGENT_GAUGE_VJP and not any(p['stop_normal_gradient'] for p in policy)
        solver._ablation_noop=noop
        (output/'contact_gradient_policy.json').write_text(json.dumps(dict(batches=policy,already_full_contact_gradient=noop),indent=2)+'\n')
        if not noop:full_contact_gradient(b)
        return
    if variant=='dense-smoothing':
        r=b.objective.regularizer;r._dense_history=wp.zeros((b.settings.worlds,26),device=b.device)
        control_smoothing(r,dense=True)
    if variant=='tracking-time':
        # Reindex sum_{k=0..28} tracking(x_k, ref_k) onto the original
        # post-step loss plumbing: constant(x_0,ref_0) plus losses at
        # (x_{k+1},ref_{k+1}), k=0..27. Geometry is left unchanged.
        job=fa.LossJob();old=b.loss.immediate[0].job
        job.model=old.model;job.weights=old.weights;job.state=old.state
        qstate=fa.LossState();qstate.qpos=b.steps[0].inputs.dynamics.qpos;job.state=qstate
        job.target=fa.allocate_target(26,old.model.guidance_count,b.device)
        constant=wp.zeros(b.settings.worlds,device=b.device)
        from examples.retarget.fa_contact_admm import variable_FA_loss
        from examples.retarget.fa_adjoint_fusion import fuse_tape_seeds
        workspaces=[]
        for k,workspace in enumerate(b.loss.immediate):
            workspace.job.reference_scale=1. if k<28 else 0.
            tape=wp.Tape();tape.record_launch(variable_FA_loss,b.settings.worlds,0,[workspace.job,b.objective.points[k]],
                [workspace.loss],wp.get_device(b.device),block_dim=256)
            fuse_tape_seeds(tape);workspaces.append(replace(workspace,tape=tape))
        b.objective.training=replace(b.objective.training,immediate=tuple(workspaces));b.loss=b.objective.training
        state['tracking_job']=job;state['constant']=constant
    if variant=='dense-controls':
        control_smoothing(b.objective.regularizer,dense=False)
        original_multistart=b._multistart
        def multi(warm,seed):
            noise=original_multistart(np.zeros(156,np.float32),seed)
            return np.c_[noise[:,:26],expand(noise[:,26:].reshape(-1,5,26),np.zeros((len(noise),26))).reshape(len(noise),-1)+warm[26:]].astype(np.float32)
        b._multistart=multi
    if variant=='slow-templates':
        from examples.retarget.comfree_ocp_native import NativeStages
        from examples.retarget.fa_contact_templates import ContactTemplates
        if sequence is None:
            from forceaware.targets import load_sequence
            from forceaware.workflow import _collision_output_layout,_object_surface_spheres
            bodies,enabled=_collision_output_layout(b.compiled.base.model)
            sequence=load_sequence(b.cfg,output_body_ids=bodies,enabled_output_indices=enabled,
                object_surface_spheres=_object_surface_spheres(b.compiled.base.model,b.index.object_body))
        engine=SimpleNamespace(model=b.cpu_model,index=b.index,compiled=b.compiled,device=b.device,cfg=b.cfg,sequence=sequence,
            oq=b.index.object_qpos)
        stages=NativeStages(engine,30);adapter=SimpleNamespace(native=stages,xscale=np.ones(b.cpu_model.nq+b.cpu_model.nv))
        templates=ContactTemplates(adapter,native_model,output,period=5,anchor_weight=5.)
        state['templates']=templates
    def prepared(qpos,qvel,targets,**kw):
        original_targets=targets
        if variant=='absolute-warm' and 'absolute' in state:
            shifted=np.concatenate([state['absolute'][3:],np.repeat(state['absolute'][-1:],3,axis=0)])
            warm=kw['warm'].copy();warm[26:]=(shifted[5::6]-targets.actions).ravel();kw['warm']=warm
        if variant=='dense-smoothing':
            previous=kw['previous_action']
            b.objective.regularizer._dense_history.assign(np.broadcast_to(state.get('previous_previous',previous),(b.settings.worlds,26)).astype(np.float32).copy())
        if variant=='dense-controls':
            original_knots=targets.dense_actions[5::6]
            actions=expand(original_knots,kw['previous_action'])
            targets=replace(targets,actions=actions.astype(np.float32))
        if variant=='slow-templates':targets=templates.apply(targets)
        prepare(qpos,qvel,targets,**kw)
        if variant=='tracking-time':
            for k,workspace in enumerate(b.loss.immediate[:28]):
                workspace.target.action.assign(targets.dense_actions[k+1])
                workspace.target.object_qpos.assign(targets.object_qpos[k+1])
            fa.assign_target(job.target,targets,0,grasp=False)
        if variant=='dense-controls':
            from forceaware.servo_prior import velocity_prior
            # Restore the same two multistart families as the successful
            # five-knot baseline, interpolated into the larger variable space.
            raw=b._multistart(kw['warm'],kw['seed'])
            old=replace(original_targets,actions=original_knots)
            prior=velocity_prior(old,b.servo_lag,.06,.01)
            dense_prior=expand(prior,np.zeros(26)).ravel();keep=max(1,b.settings.worlds//2)
            raw[keep:,26:]+=dense_prior-kw['warm'][26:]
            b.control.raw.assign(raw)
        if variant=='control-bounds':
            actions=targets.actions;lower=b.cpu_model.actuator_ctrlrange[:,0];upper=b.cpu_model.actuator_ctrlrange[:,1]
            state['lower']=((lower-actions)/solver.variable_scale.reshape(-1,26)).ravel()
            state['upper']=((upper-actions)/solver.variable_scale.reshape(-1,26)).ravel()
            raw=b.control.raw.numpy();x=raw[:,26:]/solver.variable_scale
            raw[:,26:]=np.clip(x,state['lower'],state['upper'])*solver.variable_scale;b.control.raw.assign(raw)
        if variant=='slow-templates':
            stages.prepare(targets)
            b._clear_contact_history();b._set_hard_contact();b._forward(allow_init=False)
            physical=[np.r_[qpos,qvel]]
            for step in b.steps:physical.append(np.r_[step.recorded.result.qpos.numpy()[0],step.recorded.result.qvel.numpy()[0]])
            updated=templates.update(qpos,qvel,targets,np.asarray(physical))
            prepare(qpos,qvel,updated,**kw)
    def planned(*args,**kw):
        result=plan(*args,**kw)
        selected=args[0]
        state['absolute']=np.stack([u.numpy()[selected,b.index.hand_ctrl] for u in b.control.controls])
        state['previous_previous']=result.control_sequence[-2].copy()
        if variant=='slow-templates':
            import json
            with (output/'template_updates.jsonl').open('a') as f:f.write(json.dumps(templates.last)+'\n')
        return result
    b._prepare_window=prepared;b._plan=planned
    if variant=='control-bounds':
        from examples.retarget.fa_contact_admm import _lbfgs_direction
        def bounded_hand_block(x,iterations):
            if not iterations:return x,0
            solver._forward_variables=None;value,g=solver._evaluate(x,True);history=[];accepted_count=0
            for _ in range(iterations):
                direction=_lbfgs_direction(g,history)
                direction=np.clip(x+direction,state['lower'],state['upper'])-x
                slope=np.sum(g*direction,axis=1);bad=slope>=0
                fallback=-g/np.maximum(np.max(abs(g),axis=1),1.)[:,None]
                direction[bad]=(np.clip(x+fallback,state['lower'],state['upper'])-x)[bad]
                slope=np.sum(g*direction,axis=1);pending=np.isfinite(value)&(slope<0)
                accepted=np.zeros(len(value),bool);alpha=np.ones(len(value));trial=x.copy();new=x.copy()
                for _ in range(settings_line_search):
                    if not pending.any():break
                    trial[pending]=x[pending]+alpha[pending,None]*direction[pending]
                    candidate,_=solver._evaluate(trial,False)
                    good=pending&(candidate<=value+1e-4*alpha*slope)
                    new[good]=trial[good];accepted|=good;pending&=~good;alpha[pending]*=.5
                next_value,next_g=solver._evaluate(new,True)
                s=new-x;y=next_g-g;keep=(np.sum(s*y,axis=1)>1e-12)&accepted
                s[~keep]=0.;y[~keep]=0.;history.append((s,y));history=history[-5:]
                x,value,g=new,next_value,next_g;accepted_count+=int(accepted.sum())
                if not accepted.any():break
            return x,accepted_count
        settings_line_search=8
        solver._hand_block=bounded_hand_block
    solver._ablation_state=state
    if variant=='tracking-time':
        evaluate=solver._evaluate
        def evaluate_tracking(x,gradient):
            f,g=evaluate(x,gradient)
            wp.launch(constant_tracking,b.settings.worlds,inputs=[job,constant],device=b.device)
            return f+constant.numpy(),g
        solver._evaluate=evaluate_tracking
