"""Prediction/execution contact and FA-cost traces, outside optimization.

All forces are forces ON the object in world coordinates. Native contacts are
reported at pre-step geometry; MuJoCo records each force actually solved during
the five 2 ms substeps. Templates and ideal ADMM forces are separate fields.
"""
from pathlib import Path
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


COST_NAMES=('hand_base','hand_rotation','hand_fingers','object_position','object_rotation',
    'physical_contact','marked_query','soft_penetration','penetration','finger_collision',
    'control_velocity','control_acceleration','state_velocity','state_acceleration')


def force_summary(contacts,mass,gravity):
    hand=[c for c in contacts if c['body']!=0];floor=[c for c in contacts if c['body']==0]
    total=lambda cs:np.sum([c['force'] for c in cs],axis=0) if cs else np.zeros(3)
    h=total(hand);f=total(floor)
    return dict(hand_force=h.tolist(),floor_force=f.tolist(),total_force=(h+f).tolist(),
        hand_vertical_N=float(h[2]),floor_vertical_N=float(f[2]),weight_N=float(-mass*gravity[2]),
        loaded_contacts=sum(c['normal_force']>.01 for c in hand),
        total_normal_N=float(sum(c['normal_force'] for c in hand)),
        loaded_bodies=sorted(set(c['body'] for c in hand if c['normal_force']>.01)))


def native_contacts(engine,buffer,q,world=0):
    params=engine.compiled.base.contact_rows.parameters;workspace=buffer['work'][-1]
    collision=workspace.base.collision.contacts
    pairs=params.contact_geom.numpy();bodies=engine.model.geom_bodyid[pairs]
    rows=params.row_contact.numpy()[:params.contact_row_count];edges=params.row_edge.numpy()[:params.contact_row_count]
    mu=params.contact_friction.numpy();scale=float(params.friction_scale.numpy()[0])
    distance=collision.distance.numpy()[world];active=collision.active.numpy()[world]
    position=collision.position.numpy()[world];frame=collision.frame.numpy()[world]
    coefficient=workspace.contact.output.force.numpy()[world][:params.contact_row_count]
    oq=engine.oq;R=Rotation.from_quat(q[oq+3:oq+7],scalar_first=True).as_matrix();origin=q[oq:oq+3]
    result=[]
    for c in np.flatnonzero(np.any(bodies==engine.index.object_body,axis=1)&active.astype(bool)):
        sign=-1. if bodies[c,0]==engine.index.object_body else 1.
        other=int(bodies[c,1] if sign<0 else bodies[c,0]);force=np.zeros(3)
        for row in np.flatnonzero(rows==c):
            axis=int(edges[row])//2;side=1. if int(edges[row])%2==0 else -1.
            direction=frame[c,0].copy()
            if axis<2:direction+=side*mu[c,axis]*scale*frame[c,axis+1]
            force+=sign*coefficient[row]*direction
        normal=sign*frame[c,0].astype(float);normal/=max(np.linalg.norm(normal),1e-12)
        fn=float(normal@force);ft=float(np.linalg.norm(force-fn*normal))
        result.append(dict(contact_id=int(c),geom1=int(pairs[c,0]),geom2=int(pairs[c,1]),body=other,
            body_name=mujoco.mj_id2name(engine.model,mujoco.mjtObj.mjOBJ_BODY,other),
            point=position[c].astype(float).tolist(),point_object=((position[c]-origin)@R).tolist(),
            normal=normal.tolist(),normal_object=(normal@R).tolist(),force=force.tolist(),
            force_object=(force@R).tolist(),normal_force=fn,tangent_force=ft,
            friction=float(mu[c,0]*scale),friction_utilization=float(ft/max(fn*float(mu[c,0])*scale,1e-12)),gap=float(distance[c])))
    return result


def mujoco_contacts(model,data,object_body,object_pose):
    result=[];R=Rotation.from_quat(object_pose[3:],scalar_first=True).as_matrix()
    for i,c in enumerate(data.contact):
        bodies=model.geom_bodyid[[c.geom1,c.geom2]]
        if object_body not in bodies:continue
        sign=-1. if bodies[0]==object_body else 1.;other=int(bodies[1] if sign<0 else bodies[0])
        local=np.zeros(6);mujoco.mj_contactForce(model,data,i,local)
        force=sign*c.frame.reshape(3,3).T@local[:3];normal=sign*c.frame[:3]
        fn=float(normal@force);ft=float(np.linalg.norm(force-fn*normal))
        result.append(dict(contact_id=i,geom1=int(c.geom1),geom2=int(c.geom2),body=other,
            body_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,other),
            point=c.pos.tolist(),point_object=((c.pos-object_pose[:3])@R).tolist(),normal=normal.tolist(),
            normal_object=(normal@R).tolist(),force=force.tolist(),force_object=(force@R).tolist(),
            normal_force=fn,tangent_force=ft,friction=float(c.friction[0]),
            friction_utilization=float(ft/max(fn*c.friction[0],1e-12)),gap=float(c.dist)))
    return result


def split_cost(residual,native):
    r=np.asarray(residual).ravel();squared=r*r;n=native.loss_model.guidance_count;o=native.loss_model.output_count
    base=60
    physical=squared[base:base+4*n:4].sum();marked=squared[base:base+4*n].sum()-physical
    penetration=base+4*n;capsule=penetration+2*o;regular=30+native.nr
    values=[squared[:3].sum(),squared[3:6].sum(),squared[6:26].sum(),squared[26:29].sum(),squared[29],
        physical,marked,squared[penetration:penetration+2*o:2].sum(),squared[penetration+1:penetration+2*o:2].sum(),
        squared[capsule:regular].sum()]
    values.extend([squared[regular+i*26:regular+(i+1)*26].sum() for i in range(4)])
    result=dict(zip(COST_NAMES,map(float,values)));result['total']=float(squared.sum())
    if abs(sum(values)-result['total'])>1e-8*max(1.,result['total']):raise AssertionError('Cost decomposition mismatch')
    return result


class MPCTrace:
    def __init__(self,controller,output,executor,*,start=0.,end=10.):
        self.controller=controller;self.engine=controller.engine;self.output=Path(output)
        self.executor=executor;self.start=start;self.end=end;self.snapshot=None
        self.output.mkdir(parents=True,exist_ok=True)
        self.mass=float(self.engine.model.body_mass[self.engine.index.object_body]);self.gravity=self.engine.model.opt.gravity.copy()
        self.file=(self.output/'trace.jsonl').open('w')
        self.window=0

    def row(self,kind,time,**values):
        self.file.write(json.dumps(dict(kind=kind,window=self.window,time=float(time),**values),
            default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')

    def cost(self,state,servo,stage):
        c=self.controller;p=c.last_prediction['parameters'][stage]
        if stage==c.N:
            residual=np.asarray(c.geometry_function(state,p)).ravel()
        else:residual=np.asarray(c.residual_function(state,servo/c.hand_scale,p)).ravel()
        return split_cost(residual,c.native)

    def begin(self,dense):
        self.window=dense//3;time=dense*.01;self.active=self.start-1e-9<=time<=self.end+1e-9
        if not self.active:return
        c=self.controller;s=c.last_prediction;states=s['states'];commands=s['commands'];predicted=[];costs=[]
        physical=states[:,:c.nphysical]*c.physical_scale
        for k in range(c.N):
            q=physical[k,:c.native.nq];v=physical[k,c.native.nq:]
            q1,v1,buffer=self.engine.step(q,v,commands[k],dt=.01)
            contacts=native_contacts(self.engine,buffer,q)
            prediction_error=float(np.max(np.abs(np.r_[q1[0],v1[0]]-physical[k+1])))
            if prediction_error>2e-5:raise AssertionError(f'Trace physics differs from accepted MPC rollout: {prediction_error}')
            cost=self.cost(states[k],commands[k],k);costs.append(cost);predicted.append(contacts)
            self.row('prediction',time+k*.01,stage=k,qpos=q.tolist(),qvel=v.tolist(),
                next_qpos=q1[0].tolist(),control=commands[k].tolist(),cost=cost,contacts=contacts,
                force=force_summary(contacts,self.mass,self.gravity),native_reconstruction_error=prediction_error)
        terminal=self.cost(states[-1],commands[-1],c.N);costs.append(terminal)
        total=sum(r['total'] for r in costs)
        if abs(total-c.last['true_FA_costs'][-1])>2e-5*max(1.,abs(total)):
            raise AssertionError(('Trace cost differs from accepted MPC cost',total,c.last['true_FA_costs'][-1]))
        templates=c.templates
        target=s['targets'];np.savez_compressed(self.output/f'window_{self.window:04d}.npz',
            states=states,controls=commands,parameters=s['parameters'],previous=s['previous'],previous_previous=s['previous_previous'],
            template=target.contact_position_object,mask=target.contact_mask,weight=target.contact_weight,
            query_body=target.query_body,query_local=target.query_local,
            original_template=s['original_targets'].contact_position_object)
        self.row('plan',time,cost_components={name:float(sum(r.get(name,0.) for r in costs)) for name in (*COST_NAMES,'total')},
            accepted_steps=c.last['accepted_SQP_steps'],trial_details=c.last.get('trial_details',[]),
            template_update=templates.last if templates is not None else None,
            ideal_force_plan_time=templates.last_search_window*.03 if templates is not None and templates.last_search_window is not None else None,
            ideal_force_plan=templates.search.last['forces'][0].tolist() if templates is not None and templates.search.last is not None else None,
            ideal_point_plan=templates.search.last['points'][0].tolist() if templates is not None and templates.search.last is not None else None)
        self.snapshot=s;self.file.flush()

    def execution(self,dense,stage,q,v,servo,next_q,next_v,contacts,*,substep=None):
        if not self.active:return
        c=self.controller;s=self.snapshot
        state=np.r_[q,v,s['previous'] if stage==0 else s['commands'][stage-1],
            s['previous_previous'] if stage==0 else s['previous'] if stage==1 else s['commands'][stage-2]]/c.xscale
        cost=self.cost(state,servo,stage)
        summaries=[force_summary(row['contacts'],self.mass,self.gravity) for row in substep] if substep else [force_summary(contacts,self.mass,self.gravity)]
        average={name:np.mean([s[name] for s in summaries],axis=0).tolist() for name in
            ('hand_force','floor_force','total_force','hand_vertical_N','floor_vertical_N','total_normal_N','loaded_contacts')}
        self.row('execution',dense*.01,stage=stage,qpos=np.asarray(q).tolist(),qvel=np.asarray(v).tolist(),
            next_qpos=np.asarray(next_q).tolist(),next_qvel=np.asarray(next_v).tolist(),control=servo.tolist(),
            cost=cost,contacts=contacts,force=force_summary(contacts,self.mass,self.gravity),force_average=average,substeps=substep)
        self.file.flush()

    def close(self):self.file.close()
