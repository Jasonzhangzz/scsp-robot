"""Residual form of the upstream FA loss, using its own Warp functions."""
import warp as wp
from forceaware.mpc_loss import (LossJob,_normalized_quaternion_angle,_contact,
    _query_world,_query_target_world,_collision_output,_finger_joint_world,
    _segment_distance_squared)


@wp.kernel
def residual(job:LossJob,output:wp.array2d(dtype=float)):
    world=wp.tid()
    for a in range(job.model.action_dim):
        weight=job.weights.action_finger
        if a<3:weight=job.weights.action_base_position
        elif a<6:weight=job.weights.action_base_rotation
        output[world,a]=wp.sqrt(job.reference_scale)*1000.0*weight*(job.state.qpos[world,job.model.hand_qpos[a]]-job.target.action[a])
    for a in range(3):
        output[world,job.model.action_dim+a]=wp.sqrt(job.reference_scale*job.weights.reference_pose_scale)*1000.0*job.weights.object_position*(job.state.qpos[world,job.model.object_qpos+a]-job.target.object_qpos[a])
    a=job.model.object_qpos+3
    angle=_normalized_quaternion_angle(wp.vec4(job.state.qpos[world,a],job.state.qpos[world,a+1],job.state.qpos[world,a+2],job.state.qpos[world,a+3]),
        wp.vec4(job.target.object_qpos[3],job.target.object_qpos[4],job.target.object_qpos[5],job.target.object_qpos[6]))
    output[world,job.model.action_dim+3]=wp.sqrt(job.reference_scale*job.weights.reference_pose_scale)*1000.0*job.weights.object_rotation*angle
    for k in range(job.model.guidance_count):
        phi,_=_contact(job,world,k)
        weight=job.target.contact_mask[k]*job.target.contact_weight[k]
        gate=float(job.guidance_enabled)
        offset=job.model.action_dim+4+4*k
        output[world,offset]=gate*wp.sqrt(job.weights.contact_robustness*weight)*wp.max(1000.0*(phi-job.weights.target_phi),0.0)
        error=1000.0*(_query_world(job,world,k)-_query_target_world(job,world,k))
        weight=job.target.query_mask[k]*job.target.query_weight[k]*job.auxiliary.marked_query_weight[world,k]
        factor=float(job.query_enabled)*wp.sqrt(job.weights.marked_query*weight)
        for axis in range(3):output[world,offset+1+axis]=factor*error[axis]
    for k in range(job.model.output_count):
        phi,_=_collision_output(job,world,k)
        gate=float(job.penetration_enabled)
        offset=job.model.action_dim+4+4*job.model.guidance_count+2*k
        output[world,offset]=gate*wp.sqrt(job.weights.soft_penetration)*wp.max(1000.0*(-phi-job.weights.soft_penetration_limit),0.0)
        output[world,offset+1]=gate*wp.sqrt(job.weights.penetration)*wp.max(1000.0*(-phi-job.weights.penetration_limit),0.0)
    for pair in range(job.model.finger_joint_pair_count):
        left=job.model.finger_joint_pair_left[pair];right=job.model.finger_joint_pair_right[pair]
        a0=_finger_joint_world(job,world,left,job.model.finger_joint_start[left]);a1=_finger_joint_world(job,world,left,job.model.finger_joint_end[left])
        b0=_finger_joint_world(job,world,right,job.model.finger_joint_start[right]);b1=_finger_joint_world(job,world,right,job.model.finger_joint_end[right])
        axis_mm=1000.0*wp.sqrt(_segment_distance_squared(a0,a1,b0,b1)+1.0e-12)
        radius_mm=1000.0*(job.model.finger_joint_radius[left]+job.model.finger_joint_radius[right])
        offset=job.model.action_dim+4+4*job.model.guidance_count+2*job.model.output_count+pair
        output[world,offset]=wp.sqrt(job.weights.finger_joint_collision)*wp.max(job.weights.finger_joint_surface_margin_mm-(axis_mm-radius_mm),0.0)


@wp.kernel(enable_backward=False)
def freeze_marked(weights:wp.array2d(dtype=float)):
    world,k=wp.tid();weights[world,k]=weights[0,k]
