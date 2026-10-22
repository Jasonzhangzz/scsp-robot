"""Measured-state contact-model planning and isolated physical execution."""
import copy
import csv
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor,TimeoutError as FutureTimeout
from types import SimpleNamespace

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from models.corrected_scm import corrected_scm_velocity
from planning.object_pose_cost import ObjectCostStages, target_up_axis, object_cost_batch_np
from planning.rollout_timing import distribution

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('scm_corrected','contactsdf','lcp','qp','comfree')
OBJECTS = ('foam_brick','mug','rubber_duck','elephant','piggy_bank','stanford_bunny2','teapot')


def json_value(value):
    if isinstance(value,dict):
        return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value,(np.integer,np.floating,np.bool_)):
        return json_value(value.item())
    if isinstance(value,Path):
        return str(value)
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def write_json(path,payload):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(json_value(payload),indent=2,allow_nan=False)+'\n')
    temp.replace(path)


def write_rows(path,rows,metadata):
    rounds = getattr(rows,'planning_rounds',[])
    write_json(path,dict(metadata=metadata,rows=list(rows),planning_rounds=rounds,
                         initialization=getattr(rows,'initialization',{})))
    for destination,records in ((Path(path).with_suffix('.csv'),rows),
                                (Path(path).with_suffix('.planning.csv'),rounds)):
        fields = list(dict.fromkeys(k for r in records for k in r))
        with destination.open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for record in records:
                writer.writerow({k:json.dumps(json_value(v)) if isinstance(v,(dict,list,tuple,np.ndarray))
                                 else json_value(v) for k,v in record.items()})


class TrialRows(list):
    def __init__(self):
        super().__init__()
        self.planning_rounds = []
        self.initialization = {}


class RolloutFailure(RuntimeError):
    def __init__(self,cause,rows):
        super().__init__(f'{type(cause).__name__}: {cause}')
        self.rows = rows


def errors(pose,target):
    return float(np.linalg.norm(pose[:3]-target[:3])),float(max(0.,1.-np.dot(pose[3:7],target[3:7])**2))


def successful(position,quaternion):
    return position < .02 and quaternion < .015


def supported_target(model,target,clearance=0.):
    """Place the compiled collision mesh above the actual table at goal rotation."""
    query = mujoco.MjData(model)
    query.qpos[:] = target
    mujoco.mj_forward(model,query)
    gid,table = model.geom('obj').id,model.geom('table').id
    mid = int(model.geom_dataid[gid])
    start,count = int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid])
    vertices = model.mesh_vert[start:start+count]
    lowest = np.min(vertices@query.geom_xmat[gid].reshape(3,3)[2]+query.geom_xpos[gid,2])
    top = query.geom_xpos[table,2]
    if model.geom_type[table] != mujoco.mjtGeom.mjGEOM_PLANE:
        top += model.geom_size[table,2]
    result = np.array(target,copy=True)
    result[2] += top-lowest+clearance
    return result


def object_asset_rotation(obj):
    """Authored mesh-to-body rotation, before MuJoCo principal-axis compilation."""
    source=ET.parse(ROOT/'envs/xmls'/f'env_fingertips_{obj}.xml')
    geom=source.find("./worldbody/body[@name='obj']/geom[@name='obj']")
    quat=np.fromstring(geom.get('quat','1 0 0 0'),sep=' ')
    return Rotation.from_quat(quat[[1,2,3,0]])


def held_command_status(model,pose,result,config,age,duration,args):
    """Validate a held command against observation age and its TOTAL impulse.

    Reusing a force never replenishes its impulse allowance. Only a newly
    optimized command starts a new allowance; expired commands are withdrawn.
    """
    if age > args.async_max_command_age+1e-12:
        return 'stale_observation'
    if not np.array_equal(result['objective_config'],config):
        return 'objective_changed'
    rotation = Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
    if (pose[:3]+rotation@result['p_local'])[2] <= args.ground_height_threshold:
        return 'contact_below_ground'
    body = model.body('obj').id
    force = result['force_local']
    if duration*np.linalg.norm(force)/model.body_mass[body] > args.max_linear_force_delta_v+1e-12:
        return 'impulse_budget_exhausted'
    inertia_rotation = Rotation.from_quat(model.body_iquat[body][[1,2,3,0]]).as_matrix()
    inertia = inertia_rotation@np.diag(model.body_inertia[body])@inertia_rotation.T
    angular = np.linalg.solve(inertia,np.cross(result['p_local']-model.body_ipos[body],force))
    if duration*np.linalg.norm(angular) > args.max_angular_force_delta_v+1e-12:
        return 'impulse_budget_exhausted'
    return 'valid'


def scene(args,obj,seed):
    """Reuse the established initialization without constructing any optimizer."""
    from examples.mpc.fingertips.test.params import ExplicitMPCParams
    cfg = copy.copy(args)
    cfg.obj,cfg.rollout,cfg.solver = obj,True,'cem'
    # Initialization needs these legacy scalar fields, never a contact graph.
    for key,value in dict(contact_cost_param=1.,attract_coef=.5,reject_coef=.001,
                          contact_coef=.7,reject_dis=.02,model_param=7.).items():
        if not hasattr(cfg,key):
            setattr(cfg,key,value)
    param = ExplicitMPCParams(cfg,rand_seed=seed,build_optimizer=False)
    source = ROOT/'envs/xmls'/f'env_fingertips_{obj}.xml'
    tree = ET.parse(source).getroot()
    for element in tree.iter():
        if 'file' in element.attrib:
            element.set('file',str((source.parent/element.get('file')).resolve()))
    actuators = tree.find('actuator')
    if actuators is not None:
        tree.remove(actuators)
    world = tree.find('worldbody')
    for body in list(world.findall('body')):
        if body.get('name') not in ('obj','goal','marker','marker_best'):
            world.remove(body)
    xml = ET.tostring(tree,encoding='unicode')
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.timestep = args.policy_dt/args.force_physics_substeps
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    model.opt.iterations = 100
    model.opt.tolerance = 1e-10
    model.opt.impratio = args.contact_impratio
    model.opt.noslip_iterations = args.contact_noslip_iterations
    for name in ('obj','table'):
        geom = model.geom(name).id
        model.geom_condim[geom] = 3
        model.geom_friction[geom,1:] = 0.
        # Isaac/bigrasp object contact: solref 0.02 1, solimp 0.9 0.95 0.01.
        model.geom_solref[geom] = (0.02, 1.0)
        model.geom_solimp[geom] = (0.9, 0.95, 0.01, 0.5, 2.0)
    body = model.body('obj').id
    mass = model.body_mass[body]
    model.dof_damping[:3] = mass*args.object_linear_damping
    model.dof_damping[3:6] = model.body_inertia[body]*args.object_angular_damping
    target = np.r_[param.target_p_,param.target_q_]
    initial=param.init_obj_qpos_.copy()
    if getattr(args,'target_frame','asset')=='asset':
        # The sampled poses specify the source mesh orientation. Most XMLs
        # mount that mesh with +90deg Y; leaving this in both sampled poses
        # makes a nominal upright mug balance on its handle, or a teapot on
        # its spout. Apply the inverse mounting rotation to BOTH states so
        # the same random tilt/yaw task is preserved in physical mesh space.
        correction=object_asset_rotation(obj).inv()
        for pose in (target,initial):
            pose[3:]=(Rotation.from_quat(pose[[4,5,6,3]])*correction).as_quat()[[3,0,1,2]]
        initial=supported_target(model,initial,.0005)
    target[2] = getattr(args,'target_height',target[2])
    if getattr(args,'target_height_mode','supported') == 'supported':
        target = supported_target(model,target,getattr(args,'target_clearance',.0005))
    goal = model.body('goal').id
    model.body_pos[goal],model.body_quat[goal] = target[:3],target[3:]
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == goal:
            model.geom_quat[gid] = model.geom_quat[model.geom('obj').id]
    data = mujoco.MjData(model)
    data.qpos[:] = initial
    mujoco.mj_forward(model,data)
    # XML mass/inertial origin are retained. Every model sees this exact plant.
    return model,data,target,xml


def surface_points(model,data,count,seed):
    gid = model.geom('obj').id
    mid = int(model.geom_dataid[gid])
    start,n = int(model.mesh_vertadr[mid]),int(model.mesh_vertnum[mid])
    body = model.body('obj').id
    rotation = data.xmat[body].reshape(3,3)
    vertices = (model.mesh_vert[start:start+n]@data.geom_xmat[gid].reshape(3,3).T+
                data.geom_xpos[gid]-data.xpos[body])@rotation
    hull = trimesh.convex.convex_hull(vertices)
    rng = np.random.default_rng(3000+int(seed))
    # Stratified area samples are shared exactly across all contact models.
    areas = hull.area_faces/hull.area
    positions = (np.arange(count)+rng.random(count))/count
    faces = np.searchsorted(np.cumsum(areas),positions).clip(max=len(areas)-1)
    bary = rng.random((count,2))
    fold = bary.sum(axis=1)>1
    bary[fold] = 1-bary[fold]
    triangles = hull.triangles[faces]
    points = triangles[:,0]+bary[:,:1]*(triangles[:,1]-triangles[:,0])+bary[:,1:]*(triangles[:,2]-triangles[:,0])
    normals = -hull.face_normals[faces]
    reference = np.tile([0.,0.,1.],(count,1))
    reference[np.abs(normals[:,2])>.9] = [0.,1.,0.]
    tangent = np.cross(normals,reference)
    tangent /= np.linalg.norm(tangent,axis=1,keepdims=True)
    frames = np.stack((normals,tangent,np.cross(normals,tangent)),axis=2)
    return points,frames


def physical_problem(model,data):
    """Body-frame free motion and native MuJoCo-order pyramid rows."""
    from examples.mpc.franka.scm.contact_dynamics import contact_snapshot,pyramid_rows
    mujoco.mj_forward(model,data)
    rotation = data.xmat[model.body('obj').id].reshape(3,3).copy()
    frame = np.eye(6)
    frame[:3,:3] = rotation
    mass = np.zeros((6,6))
    try:
        mujoco.mj_fullM(model,data,mass)
    except TypeError:
        mujoco.mj_fullM(model,mass,data.qM)
    mass = frame.T@mass@frame
    native,jac = contact_snapshot(model,data)
    rows,gap = pyramid_rows(native,jac)
    return dict(mass=mass,inverse=np.linalg.inv(mass),rotation=rotation,
                velocity=frame.T@data.qvel,smooth_acceleration=frame.T@data.qacc_smooth,
                jac=rows@frame,gap=gap,contact_mu=native['friction'][:,:2].copy())


def integrate_numpy(pose,velocity,h):
    velocities = np.asarray(velocity).reshape(-1,6)
    rotation = Rotation.from_quat(pose[[4,5,6,3]])
    positions = pose[:3]+rotation.apply(h*velocities[:,:3])
    q = (rotation*Rotation.from_rotvec(h*velocities[:,3:])).as_quat()[:,[3,0,1,2]]
    return np.column_stack((positions,q))


class PointForcePlanner:
    def __init__(self,model,points,frames,target,args,model_name,seed):
        self.model,self.data = copy.copy(model),mujoco.MjData(model)
        self.model.opt.timestep = args.policy_dt
        from planning.point_force_prediction import CPUSubstepPredictor
        self.cpu_predictor = CPUSubstepPredictor(model,args,model_name)
        self.points,self.frames,self.target = points,frames,target
        self.args,self.name,self.seed = args,model_name,seed
        self.round = 0
        self.search_iterations = args.population_iterations
        budget = getattr(args,'async_planning_budget_ms',None)
        self.search_budget_ms = (800*args.policy_dt if budget is None else budget) if args.async_planner else 0.
        self.gpu = None
        self.fallback_reason = None
        self.validation = None
        self.solver = getattr(args,'solver','cem')
        self.backend = args.locked_backends.get(model_name,'auto') if hasattr(args,'locked_backends') else 'auto'
        if self.solver == 'adamw':
            self.backend = 'cpu-adamw'
        elif self.solver == 'cma-es':
            self.backend = 'cpu-cma-es'
        elif self.solver == 'ipopt':
            self.backend = 'cpu-ipopt'
        self._ipopt_solvers = {}
        if self.solver == 'cem' and self.backend != 'cpu-adamw':
            from planning.point_force_gpu import GPUForceCEM,GPUQPForceCEM
            try:
                if args.prediction_substeps>1:
                    from planning.point_force_rollout_gpu import GPUPhysicsCEM
                    self.gpu = GPUPhysicsCEM(points,frames,args,model,model_name)
                else:
                    self.gpu = (GPUQPForceCEM(points,frames,args,self.model) if model_name == 'qp' and getattr(args,'qp_warp_probe',False)
                                else GPUForceCEM(points,frames,args,model_name))
                self.backend = 'cuda-cem'
            except Exception as exc:
                if model_name not in ('qp','lcp') or self.backend == 'cuda-cem':
                    raise
                self.fallback_reason = f'{type(exc).__name__}: {exc}'
                self.backend = 'cpu-adamw'
        self.problem = None

    def prepare(self,snapshot,config):
        qpos,qvel,stamp,step = snapshot
        self.data.qpos[:],self.data.qvel[:],self.data.time = qpos,qvel,stamp
        self.data.qfrc_applied[:] = 0.
        self.data.xfrc_applied[:] = 0.
        self.data.qacc_warmstart[:] = 0.
        p = self.problem = physical_problem(self.model,self.data)
        self.pose,self.config = qpos.copy(),np.asarray(config)
        self.cost_args = copy.copy(self.args)
        scale = getattr(self.args,'pose_velocity_scale',.1) if self.config[1]>0 else 1.
        self.cost_args.linear_velocity_coef *= scale
        self.cost_args.angular_velocity_coef *= scale
        self.observed_qvel = qvel.copy()
        self.free = p['velocity']+self.args.policy_dt*p['smooth_acceleration']
        world = self.points@p['rotation'].T+qpos[:3]
        self.valid = world[:,2]>self.args.ground_height_threshold
        if not self.valid.any():
            raise RuntimeError('No contact point above the ground threshold')
        if self.gpu is not None:
            self.gpu.prepare(p['inverse'],self.free,self.pose,self.target,self.config,
                             p['jac'],p['gap'],self.valid,
                             1000003*self.seed+self.args.solver_seed+7919*self.round+1,
                             observed=p['velocity'],native_data=self.data,velocity_scale=scale,native_model=self.model)

    def predict_numpy(self,commands,indices):
        commands,indices = np.asarray(commands).reshape(-1,3),np.asarray(indices,dtype=int)
        p,a = self.problem,self.args
        forces = np.einsum('nij,nj->ni',self.frames[indices],commands)
        if a.prediction_substeps>1:
            from planning.point_force_prediction import velocity_cost
            velocity,poses,details = self.cpu_predictor.predict(self.pose,self.observed_qvel,
                self.points[indices],forces)
            costs = (object_cost_batch_np(poses,self.target,self.config)+velocity_cost(velocity,self.cost_args)+
                a.force_reg_coef*np.sum(commands**2,axis=1)+
                a.friction_reg_coef*np.sum(commands[:,1:]**2,axis=1))
            return velocity,poses,costs,dict(substeps=details)
        wrenches = np.column_stack((forces,np.cross(self.points[indices],forces)))
        free = self.free+a.policy_dt*wrenches@p['inverse'].T
        metadata = {}
        if self.name == 'scm_corrected':
            velocity,metadata = corrected_scm_velocity(free,p['jac'],p['gap'],p['inverse'],a.policy_dt,a.scm_regularization)
        elif self.name == 'comfree':
            # Pinned upstream forward.py: impulse = efc_mass *
            # max(-damping*Jv - stiffness*(h*Jv + signed_gap), 0).
            row_velocity = free @ p['jac'].T
            impulse = np.maximum(-a.comfree_damping*row_velocity -
                a.comfree_stiffness*(a.policy_dt*row_velocity+p['gap']), 0.)
            from models.comfree_contact import constraint_mass
            impulse *= constraint_mass(self.model,self.data)
            velocity = free + impulse @ p['jac'] @ p['inverse'].T
        elif self.name == 'contactsdf':
            from examples.mpc.franka.scm.contact_dynamics import contactsdf_velocity
            velocity = np.array([contactsdf_velocity(p['mass'],v,p['jac'],p['gap'],a.policy_dt,a.contactsdf_sigma)[0] for v in free])
        elif self.name == 'lcp':
            from examples.mpc.franka.scm.lcp_backend import solve_lcp
            W = p['jac']@p['inverse']@p['jac'].T+a.scm_regularization*np.eye(len(p['gap']))
            velocity,certificates = [],[]
            for v in free:
                impulse,certificate = solve_lcp(W,p['jac']@v+p['gap']/a.policy_dt,
                    a.lcp_tolerance,a.lcp_max_iterations)
                velocity.append(v+p['inverse']@p['jac'].T@impulse)
                certificates.append(certificate)
            velocity = np.asarray(velocity)
            metadata['certificates'] = certificates
        else:
            velocity = []
            for force,index in zip(forces,indices):
                # No candidate advances the pose or inherits a solver warm start.
                self.data.qfrc_applied[:] = 0.
                self.data.xfrc_applied[:] = 0.
                self.data.qacc_warmstart[:] = 0.
                mujoco.mj_applyFT(self.model,self.data,p['rotation']@force,np.zeros(3),
                    self.pose[:3]+p['rotation']@self.points[index],self.model.body('obj').id,
                    self.data.qfrc_applied)
                mujoco.mj_forwardSkip(self.model,self.data,mujoco.mjtStage.mjSTAGE_VEL,1)
                v = self.data.qvel+a.policy_dt*self.data.qacc
                velocity.append(np.r_[p['rotation'].T@v[:3],v[3:]])
            velocity = np.asarray(velocity)
            self.data.qfrc_applied[:] = 0.
        poses = integrate_numpy(self.pose,velocity,a.policy_dt)
        from planning.point_force_prediction import velocity_cost
        costs = object_cost_batch_np(poses,self.target,self.config)+velocity_cost(velocity,self.cost_args)+a.force_reg_coef*np.sum(commands**2,axis=1)+a.friction_reg_coef*np.sum(commands[:,1:]**2,axis=1)
        return velocity,poses,costs,metadata

    def validate(self,snapshot,config):
        """Certify GPU outputs on measured states, including nonzero velocity."""
        try:
            self.prepare(snapshot,config)
        except Exception as exc:
            if self.name not in ('lcp','qp') or getattr(self.args,'locked_backends',{}).get(self.name) == 'cuda-cem':
                raise
            self.fallback_reason = f'GPU setup/validation unavailable: {type(exc).__name__}: {exc}'
            self.gpu,self.backend = None,'cpu-adamw'
            self.prepare(snapshot,config)
        if self.gpu is None:
            self.validation = dict(passed=True,backend=self.backend,fallback_reason=self.fallback_reason)
            return self.validation
        try:
            result = self.gpu.solve()
        except Exception as exc:
            if self.name not in ('lcp','qp') or getattr(self.args,'locked_backends',{}).get(self.name) == 'cuda-cem':
                raise
            self.fallback_reason = f'GPU execution validation failed: {type(exc).__name__}: {exc}'
            self.gpu,self.backend = None,'cpu-adamw'
            self.validation = dict(passed=False,backend=self.backend,fallback_reason=self.fallback_reason)
            return self.validation
        indices = np.flatnonzero(self.valid)
        raw = self.gpu.job.best.numpy()[indices]
        commands = self.gpu.commands(raw,indices)
        expected,poses,costs,_ = self.predict_numpy(commands,indices)
        actual = self.gpu.job.best_velocity.numpy()[indices]
        actual_cost = self.gpu.job.best_cost.numpy()[indices]
        velocity_error = float(np.max(np.abs(actual-expected)))
        cost_error = float(np.max(np.abs(actual_cost-costs)))
        tolerance = 2e-3 if self.name == 'qp' else (2e-4 if self.name == 'lcp' else 2e-9)
        if self.args.prediction_substeps>1:
            actual_pose = self.gpu.job.best_pose.numpy()[indices]
            actual_pose = np.column_stack((actual_pose[:,:3],actual_pose[:,[6,3,4,5]]))
            position_error = float(np.linalg.norm(actual_pose[:,:3]-poses[:,:3],axis=1).max(initial=0))
            orientation_error = float((Rotation.from_quat(actual_pose[:,[4,5,6,3]]).inv()*
                Rotation.from_quat(poses[:,[4,5,6,3]])).magnitude().max(initial=0))
            linear_error = float(np.linalg.norm(actual[:,:3]-expected[:,:3],axis=1).max(initial=0))
            angular_error = float(np.linalg.norm(actual[:,3:]-expected[:,3:],axis=1).max(initial=0))
            tolerance = 2e-3 if self.name in ('lcp','qp') else 1e-4
            passed = bool(np.isfinite(actual).all() and position_error<1e-6 and
                orientation_error<1e-4 and linear_error<tolerance and angular_error<tolerance)
            trajectory_validation = dict(max_position_error_m=position_error,
                max_orientation_error_rad=orientation_error,max_linear_velocity_error=linear_error,
                max_angular_velocity_error=angular_error,geometry_precision='float64 GPU rigid-body/plane-mesh vs CPU MuJoCo',
                position_tolerance_m=1e-6,orientation_tolerance_rad=1e-4,
                linear_velocity_tolerance=tolerance,angular_velocity_tolerance=tolerance)
        else:
            passed = bool(np.isfinite(actual).all() and velocity_error <= tolerance
                          and cost_error <= (2e-3 if self.name in ('qp','lcp') else 2e-8))
            trajectory_validation = {}
        validation = dict(passed=passed,model=self.name,backend='cuda-cem',
                          compared_points=len(indices),max_velocity_error=velocity_error,
                          max_cost_error=cost_error,velocity_tolerance=tolerance,
                          max_residual=float(self.gpu.job.best_residual.numpy()[indices].max(initial=0)),
                          **trajectory_validation)
        if self.name=='lcp':
            validation.update(cpu_reference_solver_counts=dict(self.cpu_predictor.lcp_reference_counts),
                cpu_reference_fallback_reasons=self.cpu_predictor.lcp_reference_fallbacks.copy())
        if not passed:
            if self.name not in ('lcp','qp'):
                raise AssertionError(f'GPU equation mismatch: {validation}')
            if self.backend == 'cuda-cem' and getattr(self.args,'locked_backends',{}).get(self.name) == 'cuda-cem':
                raise AssertionError(f'Locked GPU backend failed validation: {validation}')
            self.fallback_reason = f'GPU numerical validation failed: {validation}'
            self.backend,self.gpu = 'cpu-adamw',None
            validation.update(backend=self.backend,fallback_reason=self.fallback_reason)
        self.validation = validation
        return validation

    def _evaluate(self,raw,indices):
        from planning.point_force_prediction import force_commands
        raw = np.clip(np.asarray(raw,dtype=np.float64).reshape(-1,3),0.,1.)
        indices = np.asarray(indices,dtype=int)
        command = force_commands(raw,self.points[indices],self.frames[indices],self.model,self.args)
        velocity,poses,cost,_ = self.predict_numpy(command,indices)
        return command,velocity,poses,np.asarray(cost,dtype=np.float64)

    def _best_record(self,indices,command,velocity,poses,cost):
        finite = np.isfinite(cost)
        if not finite.any():
            return None
        k = int(np.argmin(np.where(finite,cost,np.inf)))
        return dict(selected_idx=int(indices[k]),lam=command[k].copy(),cost=float(cost[k]),
                    velocity=velocity[k].copy(),predicted_pose=poses[k].copy())

    def _adamw(self,iterations):
        import torch
        indices = np.flatnonzero(self.valid)
        unit = torch.tensor(np.tile([.01,.5,.5],(len(indices),1)),dtype=torch.float64,requires_grad=True)
        optimizer = torch.optim.AdamW([unit],lr=.025,weight_decay=0.)
        best = None
        def evaluate(values):
            return self._evaluate(values,indices)
        evaluations = 0
        for _ in range(iterations):
            raw = unit.detach().numpy().copy()
            command,velocity,poses,cost = evaluate(raw)
            evaluations += len(indices)
            k = int(np.argmin(cost))
            if best is None or cost[k] < best['cost']:
                best = dict(selected_idx=int(indices[k]),lam=command[k].copy(),cost=float(cost[k]),
                    velocity=velocity[k].copy(),predicted_pose=poses[k].copy())
            gradient = np.zeros_like(raw)
            for axis in range(3):
                delta = np.zeros_like(raw);delta[:,axis] = 1e-5
                gradient[:,axis] = (evaluate(raw+delta)[3]-evaluate(raw-delta)[3])/2e-5
                evaluations += 2*len(indices)
            optimizer.zero_grad()
            unit.grad = torch.from_numpy(gradient)
            optimizer.step()
            with torch.no_grad():
                unit.clamp_(0,1)
        best.update(alpha=None,residual=None,evaluations=evaluations,
                    solver_backend='cpu-adamw',gpu_parallel_prediction=False,search_iterations=iterations)
        return best

    def _cmaes(self,iterations):
        import cma
        indices = np.flatnonzero(self.valid)
        pop = int(self.args.population_size)
        rng = np.random.default_rng(1000003*self.seed+self.args.solver_seed+7919*self.round+1)
        strategies = [cma.CMAEvolutionStrategy([.01,.5,.5],.25,{
            'bounds':[0.,1.],'popsize':pop,'maxiter':int(iterations),'verbose':-9,'verb_log':0,
            'seed':int(rng.integers(1,2**31))}) for _ in indices]
        best,evaluations = None,0
        for generation in range(int(iterations)):
            asked = []
            for es in strategies:
                sample = np.asarray(es.ask(),dtype=np.float64)
                if generation == 0 and len(sample):
                    sample[0] = (.0,.5,.5)
                    if len(sample) > 1:
                        sample[1] = (1.,.5,.5)
                asked.append(np.clip(sample,0.,1.))
            raw = np.vstack(asked)
            command,velocity,poses,cost = self._evaluate(raw,np.repeat(indices,pop))
            cost = np.where(np.isfinite(cost),cost,np.inf).reshape(len(indices),pop)
            evaluations += len(raw)
            for row,es,sample in zip(cost,strategies,asked):
                finite = np.isfinite(row)
                if not finite.any():
                    continue
                worst = float(np.max(row[finite]))
                es.tell(sample.tolist(),np.where(finite,row,worst+abs(worst)+1.).tolist())
            candidate = self._best_record(np.repeat(indices,pop),command,velocity,poses,cost.reshape(-1))
            if candidate is not None and (best is None or candidate['cost'] < best['cost']):
                best = candidate
        if best is None:
            raise FloatingPointError('CMA-ES has no finite feasible prediction')
        best.update(alpha=None,residual=None,evaluations=evaluations,
                    solver_backend='cpu-cma-es',gpu_parallel_prediction=False,search_iterations=iterations)
        return best

    def _ipopt_solver(self,iterations):
        import casadi as cs
        iterations = int(iterations)
        cached = self._ipopt_solvers.get(iterations)
        if cached is not None:
            return cached[0]
        owner = self
        def point_cost(raw):
            _,_,_,cost = owner._evaluate(np.asarray(raw,dtype=np.float64).reshape(1,3),
                                          np.asarray([owner._ipopt_index]))
            value = float(cost[0])
            return 1e6 if not np.isfinite(value) else value
        class Cost(cs.Callback):
            def __init__(self):
                cs.Callback.__init__(self)
                self.jacobians = []
                self.construct(f'point_force_cost_{iterations}',{})
            def get_n_in(self):
                return 1
            def get_n_out(self):
                return 1
            def get_sparsity_in(self,i):
                return cs.Sparsity.dense(3,1)
            def get_sparsity_out(self,i):
                return cs.Sparsity.scalar()
            def has_jacobian(self):
                return True
            def get_jacobian(self,name,inames,onames,opts):
                class Jacobian(cs.Callback):
                    def __init__(self):
                        cs.Callback.__init__(self)
                        self.construct(name,opts)
                    def get_n_in(self):
                        return 2
                    def get_n_out(self):
                        return 1
                    def get_sparsity_in(self,i):
                        return cs.Sparsity.dense(3,1) if i==0 else cs.Sparsity.scalar()
                    def get_sparsity_out(self,i):
                        return cs.Sparsity.dense(1,3)
                    def eval(self,arg):
                        base = np.asarray(arg[0],dtype=np.float64).reshape(3)
                        step = 1e-4
                        gradient = np.zeros(3)
                        origin = point_cost(base)
                        for axis in range(3):
                            moved = base.copy();moved[axis] += step
                            gradient[axis] = (point_cost(moved)-origin)/step
                        return [gradient.reshape(1,3)]
                jacobian = Jacobian()
                self.jacobians.append(jacobian)
                return jacobian
            def eval(self,arg):
                return [point_cost(arg[0])]
        self._ipopt_cost = Cost()
        x = cs.MX.sym('u',3)
        solver = cs.nlpsol(f'point_force_ipopt_{iterations}','ipopt',{'x':x,'f':self._ipopt_cost(x)},{
            'ipopt.print_level':0,'ipopt.sb':'yes','print_time':0,
            'ipopt.hessian_approximation':'limited-memory','ipopt.max_iter':iterations,
            'ipopt.tol':1e-4,'ipopt.acceptable_tol':1e-3,'ipopt.acceptable_iter':1})
        self._ipopt_solvers[iterations] = (solver,self._ipopt_cost)
        return solver

    def _ipopt(self,iterations):
        solver = self._ipopt_solver(iterations)
        indices = np.flatnonzero(self.valid)
        best,evaluations = None,0
        for index in indices:
            self._ipopt_index = int(index)
            solved = solver(x0=[.01,.5,.5],lbx=[0.,0.,0.],ubx=[1.,1.,1.])
            stats = solver.stats()
            evaluations += int(stats.get('n_call_nlp_f',0))+1
            raw = np.asarray(solved['x'],dtype=np.float64).reshape(1,3)
            command,velocity,poses,cost = self._evaluate(raw,np.asarray([index]))
            candidate = self._best_record(np.asarray([index]),command,velocity,poses,cost)
            if candidate is not None and (best is None or candidate['cost'] < best['cost']):
                best = candidate
        if best is None:
            raise FloatingPointError('CasADi IPOPT has no finite feasible prediction')
        best.update(alpha=None,residual=None,evaluations=evaluations,
                    solver_backend='casadi-ipopt',gpu_parallel_prediction=False,search_iterations=iterations)
        return best

    def solve(self,snapshot,config,iterations=None):
        started = time.perf_counter()
        iterations = self.search_iterations if iterations is None else int(iterations)
        try:
            self.prepare(snapshot,config)
            if self.solver == 'cma-es':
                result = self._cmaes(iterations)
            elif self.solver == 'ipopt':
                result = self._ipopt(iterations)
            else:
                result = self.gpu.solve(iterations=iterations) if self.gpu is not None else self._adamw(iterations)
        except Exception as exc:
            exc.planning_time_ms = 1000*(time.perf_counter()-started)
            exc.observation_step = int(snapshot[3])
            exc.planning_round = self.round+1
            exc.search_iterations = iterations
            raise
        index = result['selected_idx']
        result['prediction_substep_evaluations']=result['evaluations']*self.args.prediction_substeps
        result.update(p_local=self.points[index].copy(),frame=self.frames[index].copy(),
                      force_local=self.frames[index]@result['lam'],
                      predicted_pose=result['predicted_pose'] if 'predicted_pose' in result else integrate_numpy(self.pose,result['velocity'],self.args.policy_dt)[0],
                      observation_time=snapshot[2],observation_step=snapshot[3],
                      planning_state_pose=self.pose.copy(),objective_config=np.asarray(config).copy(),
                      velocity_cost_coefficients=[self.cost_args.linear_velocity_coef,self.cost_args.angular_velocity_coef],
                      model=self.name,planning_round=self.round+1,
                      planning_time_ms=1000*(time.perf_counter()-started),
                      solver_fallback_reason=self.fallback_reason)
        result['search_budget_ms'] = self.search_budget_ms
        result['control_deadline_exceeded'] = result['planning_time_ms'] > 1000*self.args.policy_dt
        if self.search_budget_ms:
            # Full search is an upper bound. Adapt actual generations without
            # changing collision fidelity or omitting any candidate point.
            self.search_iterations = max(1,min(self.args.population_iterations,
                int(iterations*self.search_budget_ms/max(result['planning_time_ms'],1e-6))))
        result['contact_solver'] = ('native-soft-qp-primal-cuda-newton' if self.name=='qp' and self.gpu is not None
            else ('coupled-lcp-cuda-pgs' if self.name=='lcp' and self.gpu is not None else self.name))
        self.round += 1
        return result


def _viewer_process(model,initial_pose,initial_velocity,updates,ready,status):
    """Render an independent model/data; never share live physics buffers."""
    try:
        import mujoco.viewer
        data = mujoco.MjData(model)
        data.qpos[:],data.qvel[:] = initial_pose,initial_velocity
        mujoco.mj_forward(model,data)
        with mujoco.viewer.launch_passive(model,data) as viewer:
            viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation = .65,135.,-30.
            ready.set()
            while viewer.is_running():
                try:
                    state = updates.get(timeout=.02)
                except queue.Empty:
                    viewer.sync()
                    continue
                if state is None:
                    status.send(dict(rendered=True))
                    return
                pose,velocity,point,force = state
                data.qpos[:],data.qvel[:] = pose,velocity
                if point is not None:
                    model.body('marker_best').pos = point
                    model.body('marker').pos = point+.04*force/max(np.linalg.norm(force),1e-12)
                mujoco.mj_forward(model,data)
                viewer.sync()
            status.send(dict(error='Viewer closed before trial completed'))
    except BaseException as exc:
        status.send(dict(error=f'{type(exc).__name__}: {exc}'))
        ready.set()


class SnapshotViewer:
    def __init__(self,model,data):
        context = mp.get_context('spawn')
        self.updates,self.ready = context.Queue(2),context.Event()
        self.status,status_writer = context.Pipe(duplex=False)
        self.process = context.Process(target=_viewer_process,args=(copy.copy(model),data.qpos.copy(),
            data.qvel.copy(),self.updates,self.ready,status_writer))
        self.process.start()
        if not self.ready.wait(45) or not self.process.is_alive():
            raise RuntimeError('Visualizer did not start')
        if self.status.poll():
            raise RuntimeError(str(self.status.recv()))

    def update(self,data,point,force):
        value = (data.qpos.copy(),data.qvel.copy(),point,force)
        try:
            self.updates.put_nowait(value)
        except queue.Full:
            pass

    def close(self):
        self.updates.put(None,timeout=10)
        if not self.status.poll(20):
            self.process.terminate();self.process.join()
            raise RuntimeError('Visualizer did not acknowledge final snapshot')
        result=self.status.recv()
        self.process.join(5)
        if self.process.is_alive():
            self.process.terminate();self.process.join()
            # Native GLFW teardown can hang on software GL after rendering
            # has finished. The synchronous acknowledgement precedes teardown;
            # terminating this isolated renderer cannot affect physics.
            result['renderer_shutdown_terminated']=True
        if result.get('error'):
            raise RuntimeError(result['error'])
        return result


def run_rollout(args,name,obj,seed=0,steps=None,planner_class=PointForcePlanner):
    rows = TrialRows()
    initialized = time.perf_counter()
    model,data,target,_ = scene(args,obj,seed)
    points,frames = surface_points(model,data,args.sample_num,seed)
    stages = ObjectCostStages(args,target_up_axis(target[3:]))
    planner = planner_class(model,points,frames,target,args,name,seed)
    def snapshot(step):
        return data.qpos.copy(),data.qvel.copy(),float(data.time),int(step)
    start_state = data.qpos.copy()
    # Compile and validate disposable observations; restore all query state.
    validation = planner.validate(snapshot(0),stages.config())
    probe = list(snapshot(0))
    probe[0][2] -= .001  # exercise loaded contacts, not only separation
    probe[1] = np.array([.03,-.02,.01,.1,-.2,.15])
    moving_validation = planner.validate(tuple(probe),stages.config())
    planner.round = 0
    viewer = SnapshotViewer(model,data) if args.viewer else None
    requested_target=target.copy();requested_target[2]=args.target_height
    if args.target_frame=='asset':
        requested_target[3:]=(Rotation.from_quat(target[[4,5,6,3]])*object_asset_rotation(obj)).as_quat()[[3,0,1,2]]
    rows.initialization = dict(seconds=time.perf_counter()-initialized,
        validation=[validation,moving_validation],actual_backend=planner.backend,
        source_mass=float(model.body_mass[model.body('obj').id]),target_pose=target,
        requested_target_pose=requested_target,target_frame=args.target_frame,
        asset_to_body_quaternion=object_asset_rotation(obj).as_quat()[[3,0,1,2]],
        target_height_mode=args.target_height_mode,target_clearance=args.target_clearance,
        source_com=model.body_ipos[model.body('obj').id].copy(),
        source_inertia=model.body_inertia[model.body('obj').id].copy(),
        source_inertial_quaternion=model.body_iquat[model.body('obj').id].copy(),
        initial_pose=start_state,sample_points=points,sample_frames=frames,
        force_duration_s=args.policy_dt,policy_dt=args.policy_dt,physics_dt=model.opt.timestep,
        discrete_prediction=f'{args.prediction_substeps} candidate-dependent substeps per policy interval',
        prediction_substeps=args.prediction_substeps,
        velocity_cost=dict(linear=args.linear_velocity_coef,angular=args.angular_velocity_coef,
                           pose_scale=args.pose_velocity_scale),
        asynchronous_policy='measured-state latest-result publication; bounded hold with age/contact/total-impulse checks',
        async_max_command_age_s=args.async_max_command_age,
        async_search_budget_ms=planner.search_budget_ms,
        viewer_enabled=args.viewer)
    async_mode = bool(args.async_planner)
    result_wait_ms=800*args.policy_dt if args.async_result_wait_ms is None else args.async_result_wait_ms
    rows.initialization['async_result_wait_ms']=result_wait_ms
    rows.initialization['async_wait_semantics']='wall-clock bounded wait before physics; fast results apply to the same measured simulation state'
    replay = getattr(args,'replay_schedule',None)
    try:
        bootstrap = planner.solve(snapshot(0),stages.config()) if async_mode and replay is None else None
    except Exception as exc:
        rows.planning_rounds.append(dict(planning_time_ms=getattr(exc,'planning_time_ms',None),
            observation_step=0,planning_round=1,error=str(exc),published=False,published_step=None))
        if viewer:
            rows.initialization['viewer_status']=viewer.close()
        raise RolloutFailure(exc,rows) from exc
    rows.initialization['seconds'] = time.perf_counter()-initialized
    pool = ThreadPoolExecutor(max_workers=1) if async_mode and replay is None else None
    if pool:
        def warm_worker():
            # CUDA thread setup and the first adaptive graph capture belong
            # to initialization, just like main-thread equation validation.
            if getattr(planner,'gpu',None) is not None:
                planner.prepare(snapshot(0),stages.config())
                planner.gpu.solve(iterations=planner.search_iterations)
        try:
            warm_started=time.perf_counter()
            pool.submit(warm_worker).result()
            rows.initialization['worker_warmup_ms']=1000*(time.perf_counter()-warm_started)
        except Exception as exc:
            pool.shutdown(wait=True)
            if viewer:
                rows.initialization['viewer_status']=viewer.close()
            raise RolloutFailure(exc,rows) from exc
    rows.initialization['seconds'] = time.perf_counter()-initialized
    future,latest = None,None
    hold = 0
    flip_settling = False
    initial_time = float(data.time)
    completed_substeps_total = 0
    bad_warning_ids = [int(mujoco.mjtWarning.mjWARN_BADQPOS),
                       int(mujoco.mjtWarning.mjWARN_BADQVEL),
                       int(mujoco.mjtWarning.mjWARN_BADQACC),
                       int(mujoco.mjtWarning.mjWARN_BADCTRL)]
    started = time.perf_counter()
    schedule = {}
    for r in (replay or []):
        schedule.setdefault(int(r['observation_step']),[]).append(r)
    for records in schedule.values():
        records.sort(key=lambda r:r['planning_round'])
    pending_replay = {}
    pending_replay_errors = {}
    latest_publication_step = None
    command_duration = 0.
    deadline_misses = 0
    max_steps = args.steps if steps is None else steps

    def record(result,published_step=None,discard_reason=None):
        item = {k:v for k,v in result.items() if k not in ('frame','force_local','predicted_pose','velocity')}
        item['published_step'] = published_step
        item['published'] = published_step is not None
        item['discard_reason'] = discard_reason
        rows.planning_rounds.append(item)

    try:
        for step in range(int(max_steps)):
            cycle_started=time.perf_counter()
            rotation = data.xmat[model.body('obj').id].reshape(3,3).copy()
            measured_velocity = np.r_[rotation.T@data.qvel[:3],data.qvel[3:]]
            stages.update(dict(sim_time=data.time,policy_q=data.qpos.copy(),
                object_linear_velocity=measured_velocity[:3],object_angular_velocity=measured_velocity[3:]))
            pos,quat = errors(data.qpos,target)
            measured_settled = np.linalg.norm(measured_velocity[:3])<.05 and np.linalg.norm(measured_velocity[3:])<.5
            settling = measured_settled and successful(pos,quat)
            # Start quiet upright verification using the general settling
            # speed limits; native friction can then reach the stricter flip
            # limits without being continually disturbed by new forces.
            if not stages.enabled or stages.stage!='flip' or stages.angle>stages.exit_angle:
                flip_settling=False
            elif stages.angle<=stages.enter_angle and measured_settled:
                flip_settling=True
            updated = False
            if replay is not None:
                for scheduled in schedule.get(step,[]):
                    try:
                        if 'lead_steps' in scheduled:
                            raise ValueError('Legacy projected-state schedules require their frozen experiment source')
                        planned = planner.solve(snapshot(step),stages.config(),
                                                iterations=scheduled.get('search_iterations'))
                    except Exception as exc:
                        if not scheduled.get('error') or str(exc)!=scheduled['error']:
                            raise
                        reported=scheduled.get('failure_reported_step')
                        if reported is None:
                            rows.planning_rounds.append(dict(scheduled,planning_time_ms=exc.planning_time_ms))
                        else:
                            pending_replay_errors[int(reported)]=exc
                        continue
                    if scheduled.get('error'):
                        raise AssertionError('Recorded asynchronous planning failure did not reproduce')
                    publication = scheduled['published_step']
                    if publication is not None:
                        pending_replay[int(publication)] = planned
                    else:
                        record(planned,discard_reason=scheduled.get('discard_reason'))
                if step in pending_replay:
                    latest = pending_replay.pop(step);updated = True;record(latest,step)
                    latest_publication_step=step
                    command_duration=0.
                if step in pending_replay_errors:
                    raise pending_replay_errors.pop(step)
            elif async_mode:
                if latest is None:
                    latest = bootstrap;updated=True;record(latest,step)
                    latest_publication_step=step
                else:
                    candidate=None
                    wait_until=min(cycle_started+result_wait_ms/1000,
                                   started+(step+1)*args.policy_dt)
                    # Consume a previous overrun, then request the CURRENT
                    # observation. If both finish in this period, only the
                    # newer command is published. With zero waiting, this
                    # still keeps the worker busy instead of skipping a cycle.
                    for _ in range(2):
                        if future is None:
                            if settling or flip_settling:
                                break
                            future=pool.submit(planner.solve,snapshot(step),stages.config().copy())
                        try:
                            completed=future.result(timeout=max(0.,wait_until-time.perf_counter()))
                        except FutureTimeout:
                            if future.done() and future.exception() is not None:
                                future=None
                                if candidate is not None:
                                    record(candidate,discard_reason='later_planning_failure')
                                raise
                            break
                        except Exception:
                            future=None
                            if candidate is not None:
                                record(candidate,discard_reason='later_planning_failure')
                            raise
                        future=None
                        completed['completed_step']=step
                        if completed['control_deadline_exceeded']:
                            deadline_misses+=1
                        age=max(0.,float(data.time-completed['observation_time']))
                        if age>args.async_max_command_age+1e-12:
                            record(completed,discard_reason='stale_observation')
                        elif not np.array_equal(completed['objective_config'],stages.config()):
                            record(completed,discard_reason='objective_changed')
                        elif settling or flip_settling:
                            record(completed,discard_reason='settling' if settling else 'flip_settling')
                        else:
                            if candidate is not None:
                                record(candidate,discard_reason='superseded_before_actuation')
                            candidate=completed
                        if completed['observation_step']==step:
                            break
                    if candidate is not None:
                        latest=candidate;updated=True;record(latest,step)
                        latest_publication_step=step;command_duration=0.
            elif not (settling or flip_settling) or latest is None:
                latest = planner.solve(snapshot(step),stages.config());updated=True;record(latest,step)
                latest_publication_step=step;command_duration=0.
            if latest is None:
                raise RuntimeError('Replay has no initial command')
            command_age = max(0.,float(data.time-latest['observation_time']))
            if settling:
                command_status='settling'
            elif flip_settling:
                command_status='flip_settling'
            elif async_mode:
                command_status=held_command_status(model,data.qpos,latest,stages.config(),command_age,
                                                   command_duration+args.policy_dt,args)
            else:
                command_status='valid'
            command_valid = command_status == 'valid'
            command = latest['force_local'].copy() if command_valid else np.zeros(3)
            point = latest['p_local']
            actuation_wall_delay_ms=1000*(time.perf_counter()-cycle_started)
            mujoco.mj_forward(model,data)
            body = model.body('obj').id
            # Keep the optimized force attached to the selected body-fixed
            # point for the whole control interval.  Applying one equivalent
            # impulse to qvel and then stepping without force changes the
            # contact/friction response at every substep and no longer matches
            # the candidate predictor, which integrates the point force during
            # each substep.
            data.qfrc_applied[:] = 0.
            applied_impulse = args.policy_dt*np.r_[command,np.cross(point-model.body_ipos[body],command)]
            physics_fault = None
            completed_substeps = 0
            for _ in range(args.force_physics_substeps):
                # mj_step integrates qpos after refreshing xmat/xpos. Refresh
                # kinematics before applying the next body-attached point force.
                mujoco.mj_kinematics(model,data)
                mujoco.mj_comPos(model,data)
                rot = data.xmat[body].reshape(3,3)
                # mj_applyFT adds into qfrc_applied; clear the previous
                # substep so a constant command is not accumulated 1+2+3+...
                # times over the interval.
                data.qfrc_applied[:] = 0.
                mujoco.mj_applyFT(model,data,rot@command,np.zeros(3),
                                  data.xpos[body]+rot@point,body,data.qfrc_applied)
                time_before = float(data.time)
                pose_before,velocity_before = data.qpos.copy(),data.qvel.copy()
                warnings_before = np.asarray(data.warning.number)[bad_warning_ids].copy()
                mujoco.mj_step(model,data)
                warning_changed = np.any(np.asarray(data.warning.number)[bad_warning_ids]>warnings_before)
                time_reset = float(data.time)<time_before+.5*float(model.opt.timestep)
                nonfinite = not np.isfinite(np.r_[data.qpos,data.qvel,data.qacc]).all()
                if warning_changed or time_reset or nonfinite:
                    physics_fault = dict(warning_changed=bool(warning_changed),clock_reset=bool(time_reset),
                        nonfinite=bool(nonfinite),substep=completed_substeps+1,
                        time_before=time_before,time_after=float(data.time),
                        pose_before=pose_before,velocity_before=velocity_before,
                        native_pose_after=data.qpos.copy(),native_velocity_after=data.qvel.copy(),
                        warning_numbers=np.asarray(data.warning.number).copy())
                    break
                completed_substeps += 1
                completed_substeps_total += 1
            data.qfrc_applied[:] = 0.
            if command_valid:
                command_duration += completed_substeps*float(model.opt.timestep)
            if physics_fault is None:
                mujoco.mj_forward(model,data)
                reported_pose,reported_velocity = data.qpos.copy(),data.qvel.copy()
                rot = data.xmat[model.body('obj').id].reshape(3,3)
            else:
                # Native auto-reset is a terminal failure, never a new rollout
                # state. Preserve both sides without writing anything back.
                reported_pose,reported_velocity = pose_before,velocity_before
                rot = Rotation.from_quat(reported_pose[[4,5,6,3]]).as_matrix()
            pos,quat = errors(reported_pose,target)
            velocity = np.r_[rot.T@reported_velocity[:3],reported_velocity[3:]]
            pose_success = successful(pos,quat)
            settled = np.linalg.norm(velocity[:3])<.05 and np.linalg.norm(velocity[3:])<.5
            hold = hold+1 if pose_success and settling and settled else 0
            unstable = (physics_fault is not None or
                        not np.isfinite(np.r_[data.qpos,data.qvel,data.qacc]).all() or
                        np.max(np.abs(data.qacc))>1e6)
            success = bool(not unstable and pose_success and hold>=args.success_hold_steps and settled)
            # A stale observation's 20ms endpoint is not a prediction of this
            # interval. Do not report its lag as a contact-model error.
            prediction_applied = command_valid and latest['observation_step']==step
            prediction_pos_error = float(np.linalg.norm(latest['predicted_pose'][:3]-reported_pose[:3])) if prediction_applied else None
            prediction_rot_error = float((Rotation.from_quat(latest['predicted_pose'][[4,5,6,3]]).inv()*
                Rotation.from_quat(reported_pose[[4,5,6,3]])).magnitude()) if prediction_applied else None
            simulation_time = completed_substeps_total*float(model.opt.timestep)
            complete_cycles = completed_substeps_total//args.force_physics_substeps
            control_work_ms=1000*(time.perf_counter()-cycle_started)
            rows.append(dict(object=obj,seed=seed,step=step+1,model=name,success=success,
                completed_execution_steps=int(complete_cycles),
                execution_substeps=completed_substeps,execution_completed=completed_substeps==args.force_physics_substeps,
                completion_steps=step+1 if success else None,goal_pos=pos,goal_quat=quat,
                object_pose=reported_pose.copy(),velocity_body=velocity.copy(),physics_fault=physics_fault,
                initial_object_pose=start_state if step==0 else None,target_pose=target,
                optimal_force_contact=latest['lam'].copy(),applied_force_body=command.copy(),
                best_contact_point=point.copy(),selected_idx=latest['selected_idx'],
                predicted_pose=latest['predicted_pose'],prediction_alpha=latest.get('alpha'),
                predicted_velocity_body=latest.get('velocity'),
                optimizer_cost=latest.get('cost'),
                prediction_matches_applied_command=prediction_applied,
                prediction_position_error_m=prediction_pos_error,
                prediction_orientation_error_rad=prediction_rot_error,
                prediction_lcp_residual=latest.get('residual'),
                async_planner=async_mode,planning_updated=updated,
                planning_round=latest['planning_round'],
                planning_time_ms=latest['planning_time_ms'] if updated else None,
                observation_step=latest['observation_step'],
                command_age_s=max(0.,float(initial_time+simulation_time-latest['observation_time']-args.policy_dt)),
                command_age_at_start_s=command_age,command_hold_duration_s=command_duration,
                cycle_observation_to_actuation_wall_ms=actuation_wall_delay_ms,
                control_work_time_ms=control_work_ms,
                command_valid_for_step=command_valid,command_status=command_status,
                publication_step=latest_publication_step,
                search_iterations=latest['search_iterations'],
                async_deadline_misses=deadline_misses,
                solver_backend=latest['solver_backend'],gpu_parallel_prediction=latest['gpu_parallel_prediction'],
                solver_fallback_reason=latest['solver_fallback_reason'],
                optimizer_model=name,settling=settling,flip_settling=bool(flip_settling),settled=bool(settled),success_hold=hold,
                object_cost_stage=stages.stage,stable_flip_success=stages.first_flip_time is not None,
                simulation_time_s=simulation_time,native_data_time_s=float(data.time),elapsed_wall_time_s=time.perf_counter()-started,
                execution_time_s=time.perf_counter()-started if success else None,
                h_model=args.policy_dt,h_exec=args.policy_dt,mujoco_unstable=unstable,
                execution_info=dict(actual_applied_impulse_body=applied_impulse,
                    applied_force_duration=completed_substeps*float(model.opt.timestep) if command_valid else 0.,
                    active_substeps=completed_substeps if command_valid else 0,
                    force_was_clipped=False,pose_overwrites=0)))
            if viewer:
                viewer.update(SimpleNamespace(qpos=reported_pose,qvel=reported_velocity),
                              reported_pose[:3]+rot@point,rot@command)
            if step%250==0 or success:
                print(f'{"async" if async_mode else "sync"} {name} {obj} seed={seed} step={step+1} pos={pos:.6f} quat={quat:.6f} success={success}',flush=True)
            if unstable:
                raise FloatingPointError('MuJoCo physical execution became unstable')
            if success:
                break
            if async_mode and replay is None:
                delay = started+(step+1)*args.policy_dt-time.perf_counter()
                if delay>0:
                    time.sleep(delay)
    except Exception as exc:
        if hasattr(exc,'planning_time_ms'):
            rows.planning_rounds.append(dict(planning_time_ms=exc.planning_time_ms,
                error=str(exc),published=False,observation_step=getattr(exc,'observation_step',step),
                planning_round=getattr(exc,'planning_round',planner.round+1),published_step=None,
                search_iterations=getattr(exc,'search_iterations',None),
                failure_reported_step=step))
        raise RolloutFailure(exc,rows) from exc
    finally:
        if pool:
            pool.shutdown(wait=True)
            if future is not None:
                try:
                    record(future.result(),discard_reason='trial_finished_before_publication')
                except Exception as exc:
                    rows.planning_rounds.append(dict(planning_time_ms=getattr(exc,'planning_time_ms',None),
                        observation_step=getattr(exc,'observation_step',None),
                        planning_round=getattr(exc,'planning_round',None),
                        search_iterations=getattr(exc,'search_iterations',None),
                        error=str(exc),published=False,published_step=None))
        if viewer:
            try:
                rows.initialization['viewer_status'] = viewer.close()
            except Exception as exc:
                rows.initialization['viewer_status']=dict(error=str(exc))
                raise RolloutFailure(exc,rows) from exc
    return rows


def trial_summary(rows,model,obj,seed,async_mode,error=None):
    last = rows[-1] if rows else {}
    rounds = getattr(rows,'planning_rounds',[])
    times = [r['planning_time_ms'] for r in rounds if r.get('planning_time_ms') is not None]
    policy_dt=getattr(rows,'initialization',{}).get('policy_dt')
    work_times=[r['control_work_time_ms'] for r in rows if 'control_work_time_ms' in r]
    valid=[r for r in rows if r.get('command_valid_for_step')]
    failure_reason = None
    if not last.get('success',False):
        fault=last.get('physics_fault') or {}
        failure_reason=('native_reset' if fault.get('clock_reset') else
            ('physics_instability' if last.get('mujoco_unstable') else
            ('planning_or_initialization_exception' if error else 'step_limit')))
    return dict(model=model,object=obj,seed=seed,rollout='async' if async_mode else 'sync',
        success=bool(last.get('success',False)),steps=int(last.get('completed_execution_steps',len(rows))),
        attempted_steps=len(rows),
        completion_steps=len(rows) if last.get('success') else None,
        planning_rounds=len(rounds),planning_time_ms=distribution(times),
        final_pos=last.get('goal_pos'),final_quat=last.get('goal_quat'),
        simulation_time_s=last.get('simulation_time_s'),
        elapsed_wall_time_s=last.get('elapsed_wall_time_s'),
        execution_time_s=last.get('execution_time_s'),
        actual_backend=getattr(rows,'initialization',{}).get('actual_backend'),error=error,
        failure_reason=failure_reason,async_deadline_misses=last.get('async_deadline_misses',0),
        planning_within_policy_fraction=float(np.mean(np.asarray(times)<=1000*policy_dt)) if times and policy_dt else None,
        control_within_policy_fraction=float(np.mean(np.asarray(work_times)<=1000*policy_dt)) if work_times and policy_dt else None,
        same_cycle_command_fraction=float(np.mean([r['observation_step']==r['step']-1 for r in valid])) if valid else None,
        valid_command_fraction=float(np.mean([r.get('command_valid_for_step',True) for r in rows])) if rows else None,
        command_age_s=distribution([r['command_age_at_start_s'] for r in rows
            if r.get('command_valid_for_step') and 'command_age_at_start_s' in r]))
