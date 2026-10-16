"""Slow ADMM template planner, separate from actual ComFree collision geometry."""
from dataclasses import replace
from types import SimpleNamespace
from time import perf_counter
import numpy as np
import warp as wp


class ContactTemplates:
    def __init__(self,controller,native_model,output,*,period=5,anchor_weight=5.,outer=2,commit_consistent=False):
        stages=controller.native;self.controller=controller
        if period<1 or anchor_weight<=0 or outer<1:raise ValueError('Positive template period, anchor and outer count required')
        from examples.retarget.fa_contact_admm import ADMMSettings,PointState,object_mesh
        from examples.retarget.comfree_contact_search import ObjectContactSearch,write_object_scene
        self.stages=stages;self.period=period;self.outer=outer;self.memory={};self.updates=[]
        self.snapshots=[]
        self.last=dict(updated=False);self.window=0;self.frozen=False;self.last_search_window=None
        e=stages.engine;cfg=e.cfg
        self.settings=ADMMSettings(contact_mode='comfree-force',contact_solver='batch-lm',contact_samples=7,
            contact_force_iterations=2,contact_template_weight=anchor_weight,
            object_scene=str(write_object_scene(cfg.sequence.xml,output/'template_object_scene.xml')))
        self.mesh=object_mesh(native_model,e.index,e.device)
        points=[]
        for stage in range(stages.horizon+1):
            p=PointState();shape=(1,stages.loss_model.guidance_count)
            for name in ('c','z','dual','old_z','last_c'):setattr(p,name,wp.zeros(shape,dtype=wp.vec3,device=e.device))
            p.mesh=self.mesh.id;p.rho=self.settings.rho;points.append(p)
        work=[stages._cost_buffer(k)['loss'] for k in range(stages.horizon+1)]
        self.objective=SimpleNamespace(index=e.index,worlds=1,device=e.device,mesh=self.mesh,points=points,
            training=SimpleNamespace(immediate=work[:-1],terminal=work[-1]),force_memory=None)
        # This planner uses the measured MuJoCo mesh contacts as extra
        # candidates. MPC dynamics always use the native collision pipeline.
        cfg=replace(cfg,simulator=replace(cfg.simulator,exec_backend='mujoco'))
        self.search=ObjectContactSearch(self.objective,native_model,cfg,self.settings)
        self.commit_consistent=commit_consistent
        if commit_consistent:self.search.candidate_expander=self._consistent_candidates

    @staticmethod
    def key(body,local):return int(body),np.asarray(local,dtype=np.float32).tobytes()

    def apply(self,targets):
        points=targets.contact_position_object.copy()
        for stage in range(len(points)):
            for slot in np.flatnonzero(targets.contact_mask[stage]>0):
                key=self.key(targets.query_body[stage,slot],targets.query_local[stage,slot])
                if key in self.memory:points[stage,slot]=self.memory[key]
        return replace(targets,contact_position_object=points)

    def _consistent_candidates(self,candidates):
        """Score the held material-point map that update() will actually commit."""
        targets=self.stages.targets;sub=self.stages.engine.cfg.simulator.time_grid.knot_substeps
        expanded=np.repeat(candidates,sub,axis=2)
        for world in range(len(expanded)):
            for sample in range(expanded.shape[1]):
                if sample==0:
                    # The acceptance reference is the actual held template,
                    # rather than the previous ADMM phase's moving incumbent.
                    expanded[world,sample]=targets.contact_position_object
                    continue
                held={}
                for k in range(len(expanded[world,sample])):
                    for slot in np.flatnonzero(targets.contact_mask[k]>0):
                        key=self.key(targets.query_body[k,slot],targets.query_local[k,slot])
                        if key not in held:held[key]=expanded[world,sample,k,slot].copy()
                        expanded[world,sample,k,slot]=held[key]
        return expanded

    def update(self,qpos,qvel,targets,states):
        started=perf_counter();e=self.stages.engine
        if self.frozen or self.window%self.period:
            self.window+=1;self.last=dict(updated=False,seconds=0.,held_template=True,frozen=self.frozen)
            return targets
        from examples.retarget.fa_contact_admm import point_subproblem,initialize_points
        from comfree_warp.native_adjoint import observation_adjoint as obs
        native=self.stages;obj=self.objective
        for k,p in enumerate(obj.points):
            b=native._cost_buffer(k);b['q'].assign((states[k,:native.nq]*self.controller.xscale[:native.nq])[None].astype(np.float32))
            obs.forward(e.compiled.base,b['q'],b['work'])
            wp.launch(initialize_points,p.c.shape,inputs=[b['loss'].job,p,1],device=e.device)
        self.search.begin(qpos,qvel,targets,self.window,3)
        self.last_search_window=self.window
        improvement=0.;changed=False
        for _ in range(self.outer):
            for k,p in enumerate(obj.points):
                wp.launch(point_subproblem,p.c.shape,inputs=[native._cost_buffer(k)['loss'].job,p],device=e.device)
            self.search.update()
            phase=self.search.last;pick=int(phase['chosen'][0])
            improvement+=float(phase['sample_merit'][0,0]-phase['sample_merit'][0,pick])
            changed|=pick!=0
        last=self.search.last;choice=int(last['chosen'][0])
        updated=changed and improvement>1e-6
        shifts=[]
        proposals={}
        if updated:
            chosen=last['dense_points'][0] if self.commit_consistent else np.repeat(last['points'][0],e.cfg.simulator.time_grid.knot_substeps,axis=0)
            # One coherent selected combination; use its earliest active
            # proposal per material query and hold it until the next update.
            seen=set()
            for k in range(len(chosen)):
                for slot in np.flatnonzero(targets.contact_mask[k]>0):
                    key=self.key(targets.query_body[k,slot],targets.query_local[k,slot])
                    if key in seen:continue
                    seen.add(key);point=chosen[k,slot]
                    old=self.memory.get(key,targets.contact_position_object[k,slot])
                    shifts.append(float(np.linalg.norm(point-old)*1000))
                    proposals[key]=point.astype(np.float32).copy()
            updated=max(shifts,default=0.)>.01
            if updated:self.memory.update(proposals)
        self.last=dict(updated=updated,seconds=perf_counter()-started,held_template=False,choice=choice,
            merit_improvement=improvement,anchor_cost=float(last['template_shift_cost'][0]),
            max_update_shift_mm=max(shifts,default=0.),mean_update_shift_mm=float(np.mean(shifts)) if shifts else 0.,
            window=self.window,period_windows=self.period,anchor_weight=self.settings.contact_template_weight)
        result=self.apply(targets)
        if self.commit_consistent and updated:
            mismatch=np.linalg.norm(result.contact_position_object-last['dense_points'][0],axis=-1)[targets.contact_mask>0]
            error=float(np.max(mismatch,initial=0.))*1000
            if error>1e-4:raise RuntimeError(f'Scored/committed template mismatch: {error} mm')
            self.last['scored_committed_max_error_mm']=error
        self.last['commit_consistent']=self.commit_consistent
        self.snapshots.append(dict(window=self.window,updated=updated,
            previous_template=targets.contact_position_object.copy(),template=result.contact_position_object.copy(),
            mask=targets.contact_mask.copy(),body=targets.query_body.copy(),hand_local=targets.query_local.copy(),
            proposed=last['points'][0].copy(),ideal_forces=last['forces'][0].copy()))
        if self.commit_consistent:self.snapshots[-1]['proposed_dense']=last['dense_points'][0].copy()
        self.updates.append(self.last.copy());self.window+=1
        return result
