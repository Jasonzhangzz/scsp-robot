"""Semantic MANO contacts, monotonic reference progress and local Lambda policy.

Retargeting preserves the original anchors when their force solve and geometric
IK are valid. Force-closure and pose-descent certificates are diagnostics by
default; callers may explicitly enable the legacy certificate requirement.
"""
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import json
import time

import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation

from models.wuji_hand import MaterialPoint, normalize, rotation
from planning.grasp_verification import verify_grasp
from planning.lambda_batched_lm import BatchedLambdaLM

FINGERS = ('index','middle','pinky','ring','thumb')
WUJI_FINGERS = {'thumb':1,'index':2,'middle':3,'ring':4,'pinky':5}
REGIONS = ('proximal','middle','distal')


def xyzw_xyz_to_pose(poses):
    poses=np.asarray(poses,float)
    if poses.shape[-1]!=7 or not np.isfinite(poses).all(): raise ValueError('finite (...,7) source poses required')
    out=poses[..., [4,5,6,3,0,1,2]].copy()
    lengths=np.linalg.norm(out[...,3:],axis=-1,keepdims=True)
    if np.any(lengths<1e-12): raise ValueError('nonzero quaternion required')
    out[...,3:]/=lengths
    return out


def classify_vertices(weights):
    bones=np.argmax(weights,axis=1)
    fingers=np.full(len(bones),'palm',dtype='<U8'); regions=np.full(len(bones),'palm',dtype='<U8')
    for i,finger in enumerate(FINGERS):
        for k,region in enumerate(REGIONS):
            mask=bones==1+3*i+k
            fingers[mask]=finger; regions[mask]=region
    return fingers,regions


@dataclass
class Demonstration:
    poses: np.ndarray
    vertices: np.ndarray
    joints: np.ndarray
    canonical: np.ndarray
    canonical_joints: np.ndarray
    faces: np.ndarray
    fingers: np.ndarray
    regions: np.ndarray
    frame_ids: np.ndarray
    times: np.ndarray
    side: str


def load_demonstration(sequence, dexforge_root, mano_root, fps=30., subsample=1, max_frames=0):
    """Reuse current DexForge's inference-only MANO decoder without global aliases."""
    sequence=Path(sequence); root=Path(dexforge_root)
    source=root/'contactaware/contactaware/models/numpy_mano.py'
    spec=importlib.util.spec_from_file_location('wuji_numpy_mano',source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    raw=sequence/'mano_raw'; meta=json.loads((raw/'meta.json').read_text())
    if meta.get('source')!='dexycb_mano_raw': raise ValueError('v1 supports prepared DexYCB inputs')
    side=meta.get('hand_side','right')
    if side!='right': raise ValueError('v1 Wuji asset is right hand; left demonstrations must not be mirrored silently')
    model=module.NumpyMano(mano_root,side,np.load(raw/'hand_shape_10.npy'))
    hand=np.load(raw/'hand_pose_51.npy').reshape(-1,51)
    valid=np.abs(hand).sum(1)>1e-5
    if (raw/'frame_valid.npy').exists():valid &= np.load(raw/'frame_valid.npy').astype(bool)
    ids=np.flatnonzero(valid)[::subsample]
    if max_frames: ids=ids[:max_frames]
    if not len(ids): raise ValueError('no valid MANO frames')
    vertices,joints=model.forward(hand[ids,:48],hand[ids,48:])
    ext=np.linalg.inv(np.load(raw/'extrinsics_4x4.npy')); r,t=ext[:3,:3],ext[:3,3]
    vertices=vertices@r.T+t; joints=joints@r.T+t
    objects=np.load(raw/'object_pose_camera_7.npy').reshape(len(hand),-1,7)
    oi=int(meta.get('object_index',np.argmax(np.linalg.norm(objects[-1,:,4:]-objects[0,:,4:],axis=1))))
    poses=xyzw_xyz_to_pose(objects[ids,oi])
    poses[:,:3]=poses[:,:3]@r.T+t
    q=Rotation.from_matrix(r)*Rotation.from_quat(poses[:,[4,5,6,3]])
    poses[:,3:]=q.as_quat()[:,[3,0,1,2]]
    if 'world_up' in meta:
        alignment,_=Rotation.align_vectors(np.array([[0.,0.,1.]]),normalize(np.asarray(meta['world_up']))[None])
        rc=alignment.as_matrix()
        vertices=vertices@rc.T;joints=joints@rc.T;poses[:,:3]=poses[:,:3]@rc.T
        poses[:,3:]=(alignment*Rotation.from_quat(poses[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
    for i in range(1,len(poses)):
        if poses[i,3:]@poses[i-1,3:]<0: poses[i,3:]*=-1
    fingers,regions=classify_vertices(model.weights)
    return Demonstration(poses,vertices,joints,model.canonical_vertices,model.canonical_joints,
        model.f,fingers,regions,ids,(ids-ids[0])/fps,side)


def stabilize_demonstration(demo, surface, center_mass=None):
    """One rigid correction to the nearest stable supporting face, preserving yaw.

    Prepared DexYCB poses include reconstruction tilt. A minimum-height shift
    alone can leave a tall box balanced on one corner and falling during approach.
    Hand/object relative geometry and all object-local anchors are preserved.
    """
    transforms,_=surface.mesh.convex_hull.compute_stable_poses(center_mass=np.zeros(3) if center_mass is None else center_mass,threshold=.01)
    pose=demo.poses[0].copy();r0=rotation(pose);up=np.array([0.,0.,1.])
    axes=transforms[:,2,:3]
    best=int(np.argmax(axes@(r0.T@up)))
    correction,_=Rotation.align_vectors(up[None],(r0@axes[best])[None])
    rc=correction.as_matrix();origin=pose[:3].copy()
    demo.poses[:,:3]=(demo.poses[:,:3]-origin)@rc.T+origin
    demo.vertices[:]=(demo.vertices-origin)@rc.T+origin
    demo.joints[:]=(demo.joints-origin)@rc.T+origin
    demo.poses[:,3:]=(correction*Rotation.from_quat(demo.poses[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]]
    lowest=float((surface.mesh.vertices@rotation(demo.poses[0]).T+demo.poses[0,:3])[:,2].min())
    demo.poses[:,2]-=lowest;demo.vertices[:,:,2]-=lowest;demo.joints[:,:,2]-=lowest
    return dict(rotation=rc,rotation_angle_rad=float(correction.magnitude()),vertical_shift=-lowest,
        method='nearest_stable_support_face_common_rigid_transform')


@dataclass
class ContactEpisode:
    finger: str
    region: str
    vertex: int
    start: int
    end: int
    material: MaterialPoint


@dataclass
class SemanticContacts:
    episodes: list
    mask: np.ndarray
    anchors: np.ndarray
    normals: np.ndarray
    hand_targets: np.ndarray
    hand_normals: np.ndarray

    @property
    def materials(self): return [e.material for e in self.episodes]


def anatomical_coordinates(points,long_axis,palmar_axis,thumb_direction):
    """Normalized regional coordinates invariant to global frame rotation."""
    axis=normalize(long_axis)
    palmar=normalize(np.asarray(palmar_axis)-axis*(np.asarray(palmar_axis)@axis))
    transverse=normalize(np.cross(axis,palmar))
    if transverse@thumb_direction<0:transverse=-transverse
    coordinates=np.asarray(points)@np.column_stack((transverse,palmar,axis))
    return (coordinates-coordinates.min(0))/np.maximum(np.ptp(coordinates,axis=0),.001)


def semantic_material(hand, asset_dir, demo, vertex, finger, region):
    """Same anatomical region, normalized longitudinal and palmar coordinates."""
    if finger=='palm': name='right_palm_link'
    else: name=f'right_finger{WUJI_FINGERS[finger]}_link{REGIONS.index(region)+2}'
    body=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,name)
    if body<0: raise ValueError(f'missing semantic body {name}')
    query=np.load(Path(asset_dir)/'qp_right'/f'{name}.npz')
    points=query['local_pos']; normals=normalize(query['local_normal'])
    ids=np.flatnonzero((demo.fingers==finger)&(demo.regions==region))
    canonical_mesh=trimesh.Trimesh(demo.canonical,demo.faces,process=False)
    source_normals=canonical_mesh.vertex_normals
    source=demo.canonical[ids]
    # Axial location and palmar side are anatomical descriptors. The
    # proximal Wuji joint2 is abduction, so its local +x is NOT its pad side.
    q=np.clip(np.zeros(hand.nu),hand.lower,hand.upper)
    reference=hand.set_state(q,hand.object_pose())
    body_rotation=reference.xmat[body].reshape(3,3)
    if finger=='palm':
        robot_palmar=np.array([1.,0.,0.])
        axis=normalize(demo.canonical_joints[4]-demo.canonical_joints[0])
    else:
        closing_joint=(2 if finger=='thumb' else 1) if region=='proximal' else REGIONS.index(region)+2
        joint=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_JOINT,f'right_finger{WUJI_FINGERS[finger]}_joint{closing_joint}')
        robot_palmar=normalize(body_rotation.T@np.cross(reference.xaxis[joint],body_rotation[:,2]))
        bone=1+3*FINGERS.index(finger)+REGIONS.index(region)
        root=demo.canonical_joints[bone]
        end=(demo.canonical_joints[bone+1] if region!='distal' else source[np.argmax(np.linalg.norm(source-root,axis=1))])
        axis=normalize(end-root)
    palm_normal=normalize(np.cross(demo.canonical_joints[4]-demo.canonical_joints[0],
                                  demo.canonical_joints[1]-demo.canonical_joints[0]))
    pad_vertex={'index':328,'middle':438,'ring':566,'thumb':760,'pinky':687}
    source_palmar=-palm_normal if finger=='palm' else source_normals[pad_vertex[finger]]
    if finger=='palm' and source_palmar@source_normals[list(pad_vertex.values())].mean(0)<0:
        source_palmar=-source_palmar
    source_thumb=demo.canonical_joints[1]-demo.canonical_joints[7]
    descriptor=anatomical_coordinates(source,axis,source_palmar,source_thumb)[np.flatnonzero(ids==vertex)[0]]
    if finger!='palm':
        descriptor[2]=np.clip((demo.canonical[vertex]-root)@axis/max(np.linalg.norm(end-root),.001),0,1)
    index=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger2_link1')
    pinky=mujoco.mj_name2id(hand.model,mujoco.mjtObj.mjOBJ_BODY,'right_finger5_link1')
    robot_thumb=body_rotation.T@(reference.xpos[index]-reference.xpos[pinky])
    descriptors=anatomical_coordinates(points,np.array([0.,0.,1.]),robot_palmar,robot_thumb)
    palmar=float(source_normals[vertex]@source_palmar)
    target_palmar=np.clip(palmar,-1.,1.)
    score=np.sum((descriptors-descriptor)**2*np.array([.15,.15,1.]),axis=1)
    score+=.4*(normals@robot_palmar-target_palmar)**2
    selected=int(np.argmin(score))
    return hand.material_on_collider(body,points[selected],normals[selected],finger,region)


def extract_contacts(demo, hand, assets, enter=.004, leave=.008, min_frames=2):
    frames=len(demo.poses); vertices=len(demo.canonical)
    all_anchors=np.empty((frames,vertices,3)); gaps=np.empty((frames,vertices)); normals=np.empty_like(all_anchors)
    posed_normals=np.empty_like(demo.vertices)
    facing=np.empty_like(gaps)
    for i,pose in enumerate(demo.poses):
        posed_normals[i]=trimesh.Trimesh(demo.vertices[i],demo.faces,process=False).vertex_normals
        local=(demo.vertices[i]-pose[:3])@rotation(pose)
        all_anchors[i],gaps[i],normals[i]=hand.surface.query(local)
        facing[i]=np.einsum('ij,ij->i',posed_normals[i],normals[i]@rotation(pose).T)
    episodes=[]
    for finger,region in sorted(set(zip(demo.fingers,demo.regions))):
        group=np.flatnonzero((demo.fingers==finger)&(demo.regions==region))
        start=None; chosen=None
        for i in range(frames+1):
            # Hysteresis and fixed source material vertex within an episode.
            eligible=group[facing[i,group]>=.2] if i<frames else []
            nearest=int(eligible[np.argmin(np.abs(gaps[i,eligible]))]) if len(eligible) else None
            if start is None:
                if nearest is not None and abs(gaps[i,nearest])<=enter:
                    start=i; chosen=nearest
            elif i==frames or abs(gaps[i,chosen])>leave or facing[i,chosen]<-.1:
                if i-start>=min_frames:
                    material=semantic_material(hand,assets,demo,chosen,finger,region)
                    episodes.append(ContactEpisode(finger,region,chosen,start,i,material))
                start=None; chosen=None
    count=len(episodes)
    mask=np.zeros((frames,count),bool); anchors=np.zeros((frames,count,3)); inward=np.zeros_like(anchors)
    targets=np.zeros_like(anchors); hand_normals=np.zeros_like(anchors)
    for k,e in enumerate(episodes):
        mask[e.start:e.end,k]=True
        anchors[:,k]=all_anchors[:,e.vertex]; inward[:,k]=normals[:,e.vertex]
        targets[:,k]=demo.vertices[:,e.vertex]
        hand_normals[:,k]=posed_normals[:,e.vertex]
    return SemanticContacts(episodes,mask,anchors,inward,targets,hand_normals)


class ReferenceProgress:
    def __init__(self, poses, times, window=.5, lookahead=.06, equivalent_position=.005, equivalent_angle=np.deg2rad(5.)):
        self.poses=np.asarray(poses); self.times=np.asarray(times); self.window=window; self.lookahead=lookahead
        self.index=0; self.target_index=0; self.clock=0.
        self.equivalent_position=equivalent_position;self.equivalent_angle=equivalent_angle

    def update(self, current, dt, ready=True, minimum_index=None):
        if minimum_index is not None:
            self.index=max(self.index,min(int(minimum_index),len(self.times)-1))
            self.target_index=max(self.target_index,self.index)
        if ready: self.clock+=dt
        else: return self.index,self.target_index
        candidates=np.flatnonzero((self.times>=self.times[self.index]) & (self.times<=self.times[self.index]+self.window))
        position=np.linalg.norm(self.poses[candidates,:3]-current[:3],axis=1)
        angle=2*np.arccos(np.clip(np.abs(self.poses[candidates,3:]@normalize(current[3:])),0,1))
        score=(position/.02)**2+(angle/np.deg2rad(10))**2
        best_id=candidates[int(np.argmin(score))]
        near_position=np.linalg.norm(self.poses[candidates,:3]-self.poses[best_id,:3],axis=1)<=self.equivalent_position
        near_angle=2*np.arccos(np.clip(np.abs(self.poses[candidates,3:]@self.poses[best_id,3:]),0,1))<=self.equivalent_angle
        tied=candidates[near_position&near_angle]
        match=int(tied[np.argmin(np.abs(self.times[tied]-self.clock))])
        self.index=max(self.index,match)
        target=min(int(np.searchsorted(self.times,self.times[self.index]+self.lookahead)),len(self.times)-1)
        self.target_index=max(self.target_index,target)
        return self.index,self.target_index


def tracking_task_complete(reference,total_frames,pose,final_pose,position_tolerance=.01,angle_tolerance=np.deg2rad(5.)):
    """Finishing the hand clock alone does not finish object retargeting."""
    pose=np.asarray(pose);final_pose=np.asarray(final_pose)
    angle=2*np.arccos(np.clip(abs(normalize(pose[3:])@normalize(final_pose[3:])),0,1))
    return bool(reference==total_frames-1 and np.linalg.norm(pose[:3]-final_pose[:3])<=position_tolerance and angle<=angle_tolerance)


def hand_reference_rate(measured,reference):
    """Continuously retime hand playback to measured tracking lag.

    No pose-error feasibility threshold or frame skipping: a delayed wrist
    gets more simulation time before the next geometric pose arrives.
    """
    error=np.asarray(measured)-np.asarray(reference)
    lag=np.linalg.norm(error[:3])/.02+np.linalg.norm(error[3:6])/.3
    if len(error)>6:lag+=np.sqrt(np.mean(error[6:]**2))/.3
    return float(1/(1+lag))


class HandReferenceProgress:
    """Visit every decoded hand frame, independently of object pose matching.

    Even a long control period cannot skip a hand frame. A tracking/contact
    pause stops this demonstration clock rather than dropping queued poses.
    """
    def __init__(self,times):
        self.times=np.asarray(times,float)
        if self.times.ndim!=1 or not len(self.times) or np.any(np.diff(self.times)<0):
            raise ValueError('nonempty monotonic hand times required')
        self.index=0;self.clock=float(self.times[0])

    def update(self,dt,ready=True):
        if ready:
            self.clock+=dt
            if self.index+1<len(self.times) and self.clock>=self.times[self.index+1]-1e-12:
                self.index+=1
        return self.index


def execution_contact_subsets(contacts,frame,geometric_gaps,observed_force):
    """Bounded semantic relaxation when simultaneous full contact IK fails.

    These are candidate sets, not an independent-fingertip hand model. The full
    multi-region set is always tried first. Distal, current-contact and nearest
    region proposals retain finger/palm identity and leave all original regions
    as soft reproduction targets in MPC.
    """
    active=tuple(int(k) for k in np.flatnonzero(contacts.mask[frame]));groups={}
    for k in active:groups.setdefault(contacts.episodes[k].finger,[]).append(k)
    result=[]
    for strategy in ('distal','observed','nearest'):
        chosen=[]
        for ids in groups.values():
            if strategy=='distal':
                key=lambda k:(contacts.episodes[k].region!='distal',abs(geometric_gaps[k]),k)
            elif strategy=='observed':
                key=lambda k:(-observed_force[k],abs(geometric_gaps[k]),k)
            else:key=lambda k:(abs(geometric_gaps[k]),k)
            chosen.append(min(ids,key=key))
        subset=tuple(sorted(chosen))
        if subset and subset!=active and subset not in result:result.append(subset)
    return result


def fps(points, count, anchor):
    points=np.asarray(points,float).reshape(-1,3)
    if not len(points): return np.empty(0,int)
    selected=[int(np.argmin(np.linalg.norm(points-anchor,axis=1)))]
    distances=np.linalg.norm(points-points[selected[0]],axis=1)**2
    while len(selected)<min(count,len(points)):
        distances[selected]=-1
        nxt=int(np.argmax(distances)); selected.append(nxt)
        distances=np.minimum(distances,np.linalg.norm(points-points[nxt],axis=1)**2)
    return np.array(selected)


class LocalLambdaPolicy:
    def __init__(self,surface,*,device='cpu',seed=0,friction=.5,force_limit=100.,iterations=20,
                 radii=(.005,.01),samples=8,capacity=64,pos_coef=1.,ori_coef=.0005,
                 require_closure_descent=False):
        self.surface=surface; self.device=device; self.rng=np.random.default_rng(seed)
        self.friction=friction; self.force_limit=force_limit; self.iterations=iterations
        self.radii=tuple(radii);self.samples=samples;self.capacity=capacity
        self.pos_coef=pos_coef;self.ori_coef=ori_coef;self.solvers={};self.previous=None
        self.rejections=[]
        self.require_closure_descent=bool(require_closure_descent)

    def solver(self,n):
        if n not in self.solvers:
            self.solvers[n]=BatchedLambdaLM(num_contacts=n,device=self.device,iterations=self.iterations,
                mu=self.friction,max_contact_force=self.force_limit,friction_cone='pyramidal',
                pos_coef=self.pos_coef,ori_coef=self.ori_coef,force_reg_coef=1e-8)
        return self.solvers[n]

    def _evaluate(self,points,normals,slots,current,target,inertia,velocity,external,dt,verify,
                  source,**support):
        points=np.asarray(points);normals=np.asarray(normals);n=len(slots)
        from planning.contact_friction import native_tangents
        r=rotation(current)
        tangents=native_tangents(normals@r.T,
            None if self.actual_tangents is None else self.actual_tangents@r.T)@r
        initial=None
        if self.previous is not None and tuple(slots)==self.previous['slots']:
            # Convert world/object vectors to each candidate's force frame.
            vec=self.previous['force_vectors_local']
            initial=np.zeros((len(points),n,3))
            for b in range(len(points)):
                for k,normal in enumerate(normals[b]):
                    normal=normalize(normal);tangent,second=tangents[b,k]
                    initial[b,k]=[vec[k]@normal,vec[k]@tangent,vec[k]@second]
        solver=self.solver(n)
        batch=solver.solve_device(points,normals,current_x=current,x_d=target,object_inertia=inertia,
            v_obj=velocity,tau_o=external,prediction_dt=dt,initial_forces=initial,
            tangent_directions=tangents,**support)
        valid=batch.valid.numpy().astype(bool);costs=batch.pose_costs.numpy()
        order=np.argsort(costs);best=None
        for row in order:
            if best is not None and costs[row]>best['rank'][0]+1e-9: break
            if not valid[row] or not np.isfinite(costs[row]): continue
            result=solver.gather_selected(int(row),batch=batch)
            result.pop('batch',None)  # Reused device buffers are not persistent plans.
            f=result['control_input']
            if (np.any(f[:,0]<-1e-6) or np.any(np.sum(np.abs(f[:,1:]),axis=1)>self.friction*f[:,0]+1e-5)
                or np.any(np.linalg.norm(f,axis=1)>self.force_limit+1e-5)): continue
            closure=verify_grasp(points[row],normals[row],self.friction,edges=4,tangent_directions=tangents[row])
            cost=float(result.get('pose_cost',costs[row]))
            current_cost=float(result.get('current_pose_cost',np.inf))
            at_target=current_cost<=1e-12 and cost<=1e-12
            decreasing=float(result.get('pose_reduction',-np.inf))>1e-12
            if self.require_closure_descent and (not closure['force_closure'] or not (decreasing or at_target)):
                if len(self.rejections)<8:self.rejections.append(dict(source=source,
                    reason='not_force_closed' if not closure['force_closure'] else 'no_pose_cost_descent',
                    pose_cost=cost,pose_reduction=result.get('pose_reduction')))
                continue
            check=verify(points[row],normals[row])
            if not check['valid']:
                if len(self.rejections)<8:
                    self.rejections.append(dict(source=source,pose_cost=float(costs[row]),
                        max_contact_error=check.get('max_contact_error'),max_shape_error=check.get('max_shape_error'),
                        collisions=check.get('collisions',[])))
                continue
            deviation=float(np.linalg.norm(points[row]-self.original,axis=1).sum())
            switching=0. if self.previous is None or tuple(slots)!=self.previous['slots'] else float(np.linalg.norm(points[row]-self.previous['points']))
            rank=(float(costs[row]),deviation,float(check.get('material_offset',0.)),switching)
            if self.require_closure_descent:rank+=(-closure['positive_span_margin'],)
            candidate=dict(result,slots=tuple(slots),valid=True,source=source,ik=check,
                force_closure_certificate=closure,candidate_count=len(points),rank=rank,
                tangent_directions=tangents[row].copy(),friction_cone='pyramidal')
            candidate.update(pose_cost_decreased=decreasing,target_already_reached=at_target,
                closure_descent_required=self.require_closure_descent)
            if best is None or rank<best['rank']: best=candidate
        return best

    def solve(self,anchors,normals,slots,*,current,target,inertia,velocity,external,dt,verify,
              actual_points=None,actual_normals=None,actual_tangents=None,projected_points=None,projected_normals=None,seed_sets=(),**support):
        started=time.perf_counter();slots=tuple(slots);self.rejections=[]
        if not slots:
            return dict(valid=True,source='no_contact',slots=(),points=np.empty((0,3)),normals=np.empty((0,3)),force_vectors_local=np.empty((0,3)),solve_time=0.)
        self.original=np.asarray(anchors,float)
        self.actual_tangents=None if actual_tangents is None else np.asarray(actual_tangents,float)
        original=self._evaluate(self.original[None],np.asarray(normals)[None],slots,current,target,inertia,
            velocity,external,dt,verify,'mano_original',**support)
        if original is not None:
            original['exploration_radius']=0.;original['solve_time']=time.perf_counter()-started
            self.previous=original;return original
        reason=(self.rejections[0].get('reason','original_physical_or_ik_infeasible')
            if self.rejections else 'original_physical_or_ik_infeasible')
        for radius in self.radii:
            pools=[];npools=[]
            for k,anchor in enumerate(anchors):
                ids=np.flatnonzero(np.linalg.norm(self.surface.points-anchor,axis=1)<=radius)
                chosen=ids[fps(self.surface.points[ids],self.samples,anchor)]
                pool=[np.asarray(anchor)];normal_pool=[normals[k]]
                for extra,extra_normal in ((actual_points,actual_normals),(projected_points,projected_normals)):
                    if extra is not None and np.isfinite(extra[k]).all() and np.linalg.norm(extra[k]-anchor)<=radius:
                        pool.append(extra[k]);normal_pool.append(extra_normal[k])
                pool.extend(self.surface.points[chosen]);normal_pool.extend(self.surface.normals[chosen])
                pools.append(np.array(pool));npools.append(np.array(normal_pool))
            combinations=[tuple(0 for _ in slots)];seen=set(combinations)
            # Preserve every unaffected MANO anchor first. Interleave fingers
            # so a small capacity does not explore only the first finger.
            for choice in range(1,max(map(len,pools))):
                for k,pool in enumerate(pools):
                    if choice>=len(pool) or len(combinations)>=max(1,self.capacity//2):continue
                    row=[0]*len(slots);row[k]=choice;row=tuple(row)
                    combinations.append(row);seen.add(row)
            for _ in range(self.capacity*20):
                row=tuple(int(self.rng.integers(len(pool))) for pool in pools)
                if row not in seen: combinations.append(row);seen.add(row)
                if len(combinations)>=self.capacity: break
            protected=[]
            for seed_points,seed_normals in [(projected_points,projected_normals),*seed_sets]:
                if seed_points is None:continue
                seed_points=np.asarray(seed_points);seed_normals=np.asarray(seed_normals)
                if np.isfinite(seed_points).all() and np.isfinite(seed_normals).all() and np.all(np.linalg.norm(seed_points-self.original,axis=1)<=radius+1e-12):
                    protected.append((seed_points,seed_normals))
            points=np.array([[pools[k][row[k]] for k in range(len(slots))] for row in combinations])
            inward=np.array([[npools[k][row[k]] for k in range(len(slots))] for row in combinations])
            if protected:
                room=max(0,self.capacity-len(protected))
                points=np.concatenate((points[:room],np.array([v[0] for v in protected[:self.capacity]])))
                inward=np.concatenate((inward[:room],np.array([v[1] for v in protected[:self.capacity]])))
            result=self._evaluate(points,inward,slots,current,target,inertia,velocity,external,dt,verify,'local_fps',**support)
            if result is not None:
                result.update(exploration_radius=radius,exploration_reason=reason,solve_time=time.perf_counter()-started)
                self.previous=result;return result
        if self.previous is not None and slots==self.previous['slots']:
            result=self._evaluate(self.previous['points'][None],self.previous['normals'][None],slots,current,target,
                inertia,velocity,external,dt,verify,'verified_incumbent',**support)
            if result is not None:
                result.update(exploration_reason=reason,solve_time=time.perf_counter()-started)
                self.previous=result;return result
        return dict(valid=False,source='infeasible',slots=slots,exploration_reason=reason,
            solve_time=time.perf_counter()-started,points=np.asarray(anchors),normals=np.asarray(normals),
            force_vectors_local=np.zeros((len(slots),3)),rejected_candidates=self.rejections.copy())
