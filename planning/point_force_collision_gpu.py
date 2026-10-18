"""MuJoCo 3.6 plane-mesh contact selection for the free-object benchmark.

The support hill climb and at-most-three contacts follow mjc_PlaneConvex,
including its 0.3*rbound separation from the first contact. The vendored Warp
plane_convex selects four points within a 1 mm band and is not equivalent.
Reference: https://github.com/google-deepmind/mujoco/blob/3.6.0/src/engine/engine_collision_convex.c
Algorithm adapted from MuJoCo, Copyright Google DeepMind, Apache-2.0.
"""
import mujoco
import numpy as np
import warp as wp


@wp.struct
class PlaneMesh:
    vertices:wp.array(dtype=wp.vec3d)
    graph:wp.array(dtype=wp.int32)
    graphadr:wp.int32
    geom_pose:wp.transformd
    normal:wp.vec3d
    plane_pos:wp.vec3d
    rbound:wp.float64
    margin:wp.float64


@wp.func
def plane_mesh_contacts(cfg:PlaneMesh,body:wp.transformd):
    transform=body*cfg.geom_pose
    origin=wp.transform_get_translation(transform)
    rotation=wp.transform_get_rotation(transform)
    direction=wp.quat_rotate_inv(rotation,-cfg.normal)
    threshold=wp.dot(cfg.normal,origin-cfg.plane_pos)-cfg.margin
    selected=wp.vec3i(-1,-1,-1)
    best=int(0)
    global_best=int(0)
    if cfg.graphadr<0:
        support=wp.float64(-1.e10)
        for i in range(cfg.vertices.shape[0]):
            value=wp.dot(cfg.vertices[i],direction)
            if value>support:
                support=value
                global_best=i
    else:
        adr=cfg.graphadr
        n=cfg.graph[adr]
        edges=adr+2
        globals=edges+n
        neighbors=globals+n
        support=wp.dot(cfg.vertices[cfg.graph[globals]],direction)
        change=int(1)
        while change!=0:
            change=int(0)
            i=cfg.graph[edges+best]
            while cfg.graph[neighbors+i]>=0:
                local=cfg.graph[neighbors+i]
                value=wp.dot(cfg.vertices[cfg.graph[globals+local]],direction)
                if value>support:
                    support=value
                    best=local
                    change=int(1)
                i+=1
        global_best=cfg.graph[globals+best]
    first=wp.transform_point(transform,cfg.vertices[global_best])
    first_dist=wp.dot(cfg.normal,first-cfg.plane_pos)
    first_contact=first-wp.float64(.5)*first_dist*cfg.normal
    count=int(0)
    if first_dist<cfg.margin:
        selected[0]=global_best
        count=int(1)
        if cfg.graphadr<0:
            for i in range(cfg.vertices.shape[0]):
                if count>=3:
                    break
                if i!=global_best and wp.dot(cfg.vertices[i],direction)>threshold:
                    p=wp.transform_point(transform,cfg.vertices[i])
                    if wp.length(p-first_contact)>=wp.float64(.3)*cfg.rbound:
                        selected[count]=i
                        count+=1
        else:
            adr=cfg.graphadr
            n=cfg.graph[adr]
            globals=adr+2+n
            neighbors=globals+n
            i=cfg.graph[adr+2+best]
            while cfg.graph[neighbors+i]>=0 and count<3:
                index=cfg.graph[globals+cfg.graph[neighbors+i]]
                if wp.dot(cfg.vertices[index],direction)>threshold:
                    p=wp.transform_point(transform,cfg.vertices[index])
                    if wp.length(p-first_contact)>=wp.float64(.3)*cfg.rbound:
                        selected[count]=index
                        count+=1
                i+=1
    distances=wp.vec3d(wp.float64(1.e6))
    positions=wp.mat33d(wp.float64(0.))
    for i in range(count):
        p=wp.transform_point(transform,cfg.vertices[selected[i]])
        distance=wp.dot(cfg.normal,p-cfg.plane_pos)
        point=p-wp.float64(.5)*distance*cfg.normal
        distances[i]=distance
        for k in range(3):
            positions[i,k]=point[k]
    return distances,positions,count


class GPUPlaneContacts:
    def __init__(self,model,device):
        obj,table=model.geom('obj').id,model.geom('table').id
        if model.nv!=6 or model.geom_type[obj]!=mujoco.mjtGeom.mjGEOM_MESH or model.geom_type[table]!=mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError('Plane-mesh benchmark geometry required')
        if np.any(model.geom_margin) or np.any(model.geom_gap):
            raise ValueError('Nonzero contact margins require an explicit GPU contact extension')
        cfg=PlaneMesh()
        mesh=int(model.geom_dataid[obj]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
        cfg.vertices=wp.array(model.mesh_vert[start:start+count],dtype=wp.vec3d,device=device)
        cfg.graph=wp.array(model.mesh_graph,dtype=wp.int32,device=device)
        cfg.graphadr=int(model.mesh_graphadr[mesh])
        cfg.geom_pose=wp.transformd(wp.vec3d(*model.geom_pos[obj]),wp.quatd(*model.geom_quat[obj][[1,2,3,0]]))
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        cfg.normal=wp.vec3d(*data.geom_xmat[table].reshape(3,3)[:,2])
        cfg.plane_pos=wp.vec3d(*data.geom_xpos[table])
        cfg.rbound=float(model.geom_rbound[obj]);cfg.margin=0.
        self.config=cfg
        self.settings=wp.array([cfg],dtype=PlaneMesh,device=device)
        self.device=device
        wp.load_module(module=__name__,device=device)
