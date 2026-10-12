"""Batched MuJoCo physics with the Isaac example's Franka OSC executor.

Physics runs on MJWarp. CPU MuJoCo computes kinematics/contact geometry for
OSC and CSO, but never advances physics or writes constraint forces to MJWarp.
The backend is deliberately identified separately from PhysX in trial metrics.
"""
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import warp as wp
import mujoco_warp as mjwarp
from scipy.spatial.transform import Rotation

from examples.mpc.franka.ik2.test_mpc_isaac import (
    IsaacFrankaOSCSimulator, POLICY_INTERVAL, DYWA_SIM_DT,
    FRANKA_QD_NULLSPACE, _apply_dywa_physics_to_param,
    adapt_param_for_cartesian_solver, _configure_rollout_param,
)
from examples.mpc.franka.ik2.params import ExplicitMPCParams
from examples.mpc.franka.ik2.isaac_bus import PolicyActionSchedule
from examples.mpc.franka.ik2.contact_frames import (
    _contact_jacobian_np, tangent_basis_from_normal, mesh_table_contact_points,
    support_patch_jacobians, contact_jacobian_angular_body_frame,
)
from planning.runtime_compat import make_runtime_profile

ROOT = Path(__file__).resolve().parents[4]


def isaac_robot_dynamics():
    """Measured properties of the exact URDF loaded by the Isaac example.

    That URDF has no inertial tags: PhysX infers inertias from its collision
    meshes. The Menagerie Panda's authored inertias are a different plant.
    """
    path = ROOT/'examples/mpc/franka/ik2/assets/isaac_franka_dynamics.json'
    data = json.loads(path.read_text())
    urdf = ROOT/data['source_urdf']
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != data['source_urdf_sha256']:
        raise RuntimeError('Franka URDF changed; re-export Isaac rigid-body properties')
    return data


def build_param(args, trial):
    param = ExplicitMPCParams(args, rand_seed=trial, target_type=args.target_type, build_optimizer=False)
    return _configure_rollout_param(
        adapt_param_for_cartesian_solver(_apply_dywa_physics_to_param(param), args, build_optimizer=False), args)


def scene_xml(param, args):
    # Reuse the repository's Panda joint transforms. The default scene replaces
    # its authored inertias with the properties exported from Isaac below.
    source = ROOT / 'examples/mpc/franka/scm/assets/franka_cube_grasp/panda.xml'
    root = ET.parse(source).getroot()
    root.set('model', 'ik2_mjwarp_franka')
    for element in list(root):
        if element.tag not in ('compiler', 'default', 'worldbody'):
            root.remove(element)
    root.find('compiler').set('meshdir', '')
    world = root.find('worldbody')
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ('geom', 'site', 'light'):
                parent.remove(child)
    link7 = world.find('.//body[@name="link7"]')
    for child in list(link7):
        if child.tag == 'body':
            link7.remove(child)
    attachment = ET.SubElement(link7, 'body', name='attachment', pos='0 0 0.107',
                               quat='0.3826834 0 0 0.9238795')
    ET.SubElement(attachment, 'inertial', mass='.001', pos='0 0 .03', diaginertia='1e-6 1e-6 1e-6')
    tip = ET.SubElement(attachment, 'body', name='fingertip', pos='0 0 .06')
    ET.SubElement(tip, 'geom', name='fingertip', type='sphere', size='.01', mass='.01',
                  contype='2', conaffinity='1', friction='1.5 .005 .0001', condim='3', margin='.001')
    for body in world.iter('body'):
        body.set('gravcomp', '1')  # Isaac disables gravity on the Franka asset.
    for joint in world.iter('joint'):
        joint.set('damping', str(args.effort_joint_damping))
        joint.set('armature', '0')
    option = ET.SubElement(root, 'option', timestep=str(DYWA_SIM_DT), integrator='implicitfast',
                           gravity='0 0 -9.81', cone='pyramidal', iterations='20')
    ET.SubElement(option, 'flag', sensor='disable')
    ET.SubElement(world, 'geom', name='table', type='box', size='1 1 .175', pos='1.2 0 .175',
                  contype='1', conaffinity='3', friction='.5 .005 .0001', condim='3')
    asset = ET.SubElement(root, 'asset')
    parts = param.collision_mesh_paths_ or [param.mesh_path_]
    obj = ET.SubElement(world, 'body', name='obj')
    ET.SubElement(obj, 'freejoint', name='object_free')
    values = lambda arr: ' '.join(str(float(v)) for v in arr)
    ET.SubElement(obj, 'inertial', mass=str(param.sim_obj_mass_), pos='0 0 0',
                  diaginertia=values(param.sim_obj_inertia_diag_))
    for i, path in enumerate(parts):
        name = 'object_part_%s' % i
        ET.SubElement(asset, 'mesh', name=name, file=os.path.abspath(path))
        ET.SubElement(obj, 'geom', name=name, mesh=name, type='mesh', contype='1', conaffinity='3',
                      friction='.9 .005 .0001', condim='3')
    actuator = ET.SubElement(root, 'actuator')
    for i, limit in enumerate([87, 87, 87, 87, 12, 12, 12], 1):
        ET.SubElement(actuator, 'motor', name='motor%s' % i, joint='joint%s' % i,
                      ctrllimited='true', ctrlrange='%s %s' % (-limit, limit))
    if getattr(args, 'mjwarp_scene', 'isaac') == 'isaac':
        dynamics = isaac_robot_dynamics()
        for properties in dynamics['bodies']:
            name = properties['name'].replace('panda_link', 'link')
            body = world.find('.//body[@name="%s"]' % name)
            inertial = body.find('inertial')
            if inertial is None:
                inertial = ET.SubElement(body, 'inertial')
            matrix = np.asarray(properties['inertia'])
            inertial.attrib.clear()
            inertial.set('mass', str(properties['mass']))
            inertial.set('pos', values(properties['com_pos']))
            inertial.set('fullinertia', values(matrix[[0,1,2,0,0,1], [0,1,2,1,2,2]]))
        # MuJoCo's implicit geom-pair rule takes max(friction), whereas
        # PhysX defaults to the average. Use explicit sliding coefficients
        # rather than copying identical numbers with different semantics.
        contact = ET.SubElement(root, 'contact')
        for i in range(len(parts)):
            for other, friction in (('table', .5*(.5+.9)), ('fingertip', .5*(1.5+.9))):
                ET.SubElement(contact, 'pair', geom1='object_part_%s' % i, geom2=other,
                              condim='3', margin='.001' if other == 'fingertip' else '0',
                              friction=values([friction, friction, .005, .0001, .0001]))
        ET.SubElement(contact, 'pair', geom1='fingertip', geom2='table', condim='3',
                      margin='.001', friction=values([1., 1., .005, .0001, .0001]))
    return ET.tostring(root, encoding='unicode')


class MuJoCoContact:
    def __init__(self, param):
        self.param_ = param
        self._last_fingertip_contact = None

    def get_actual_fingertip_contact(self):
        return self._last_fingertip_contact

    def detect_once(self, env):
        p, nv, count = self.param_, self.param_.n_qvel_, self.param_.max_ncon_
        phi = np.ones(4*count)
        jac, jac_env = np.zeros((4*count, nv)), np.zeros((4*count, nv))
        self.env_phi = np.ones(4*count)
        state = env.get_policy_state()
        pos = state[:3]
        rot = Rotation.from_quat(state[3:7][[1, 2, 3, 0]]).as_matrix()
        measured = env.measured_contact()
        self._last_fingertip_contact = measured
        row = 0
        if measured is not None:
            point, normal = measured['point_world'], measured['normal_world']
            rel = np.zeros((3, nv))
            rel[:, :3] = np.eye(3)
            rel[:, 3:6] = -env._skew(point-pos)
            rel[:, 6:9] = -np.eye(3)
            n, t1, t2 = tangent_basis_from_normal(-normal)
            jac[:4] = _contact_jacobian_np(n, t1, t2, rel, p.mu_object_)
            phi[:4] = measured['dist']
            row = 1
        points = mesh_table_contact_points(pos, rot, p.object_vertices_)
        con_pos = []
        if min(points[:, 2]) - p.table_height < .02:
            for i, point in enumerate(points[:count-row]):
                world_j, body_j = support_patch_jacobians(pos, rot, point, nv, p.mu_object_)
                gap = float(point[2]-p.table_height)
                phi[4*row:4*row+4] = gap
                jac[4*row:4*row+4] = world_j
                jac_env[4*i:4*i+4, :6] = body_j
                self.env_phi[4*i:4*i+4] = gap
                con_pos.append(rot.T @ (point-pos))
                row += 1
        jac = contact_jacobian_angular_body_frame(jac, rot)
        return phi, jac, con_pos, jac_env, measured is not None and measured['dist'] <= 0.


class MuJoCoFrankaOSC(IsaacFrankaOSCSimulator):
    """Reuse action scheduling and OSC; override only simulator operations."""
    def __init__(self, model, args, param):
        self.model, self.args, self.param_ = model, args, param
        self.data = mujoco.MjData(model)
        self.sim_dt_ = DYWA_SIM_DT
        self.viewer_ = None
        self.break_out_signal_ = self.dyn_paused_ = False
        self.tip_id = model.body('fingertip').id
        self.obj_id = model.body('obj').id
        joint = model.joint('object_free').id
        self.obj_qadr = int(model.jnt_qposadr[joint])
        self.obj_vadr = int(model.jnt_dofadr[joint])
        self.torque_limits = np.array([87, 87, 87, 87, 12, 12, 12])
        self.nullspace_stiffness = args.nullspace_stiffness
        self.activate_tool_compensation = False
        self.tool_compensation_force = np.zeros(6)
        self._policy_schedule = PolicyActionSchedule(self.sim_dt_, POLICY_INTERVAL, float(param.mpc_u_ub_))
        self.runtime_profile_ = make_runtime_profile(args.solver_backend)
        self.contact = MuJoCoContact(param)
        self.reset(param)

    def reset(self, param):
        self.param_ = self.contact.param_ = param
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.obj_qadr:self.obj_qadr+7] = param.init_obj_qpos_
        q = FRANKA_QD_NULLSPACE.copy()
        target = param.init_fingertip_pos_
        q[0] = np.arctan2(target[1], max(target[0], .05))
        for _ in range(80):
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            err = target-self.get_end_effector_pos()[0]
            if np.linalg.norm(err) < .008:
                break
            jac = self._get_position_jacobian()
            q += jac.T @ np.linalg.solve(jac @ jac.T+1e-3*np.eye(3), .45*err)
            q = np.clip(q, self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1])
        self.data.qpos[:7] = q
        mujoco.mj_forward(self.model, self.data)
        self.q_d_nullspace = q.copy()
        self.p_d, self.R_d_hold = self.get_end_effector_pos()
        self.R_d = self.R_d_hold.copy()
        self.position_d = self.p_d.copy()
        self.orientation_d = self.R_d_hold.copy()
        self._hold_dq = None
        self._hold_i = 0
        self._clear_mpc_action()
        self.seq = 0
        self._last_osc_force = self._last_osc_pd = None
        self._osc_upright_guard_active = False
        if param.contact_model == 'soft-scm' and (self.args.online_scm or self.args.scm_record_data):
            from models.scm_online import ExecutionTransitionBuffer
            self._scm_recorder = ExecutionTransitionBuffer(param.trial_seed_, self.sim_dt_)
        else:
            self._scm_recorder = None

    def get_state(self):
        return np.r_[self.data.qpos[self.obj_qadr:self.obj_qadr+7], self.data.qpos[:7]]

    def get_end_effector_pos(self):
        return self.data.xpos[self.tip_id].copy(), self.data.xmat[self.tip_id].reshape(3, 3).copy()

    def get_object_velocity_world(self):
        v = self.data.qvel[self.obj_vadr:self.obj_vadr+6]
        r = self.data.xmat[self.obj_id].reshape(3, 3)
        return v[:3].copy(), r @ v[3:]

    def get_fingertip_velocity_world(self):
        return self._get_position_jacobian() @ self.data.qvel[:7]

    def get_current_joint_position(self):
        return self.data.qpos[:7].copy()

    def get_current_joint_velocity(self):
        return self.data.qvel[:7].copy()

    def _get_position_jacobian(self, q=None):
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jp, jr, self.tip_id)
        return jp[:, :7]

    def measured_contact(self):
        best = None
        tip_pos = self.data.xpos[self.tip_id]
        for c in self.data.contact:
            bodies = self.model.geom_bodyid[c.geom]
            if set(bodies) != {self.obj_id, self.tip_id}:
                continue
            n = np.asarray(c.frame[:3]).copy()
            if np.dot(n, tip_pos-c.pos) < 0:
                n = -n
            if best is None or c.dist < best['dist']:
                best = dict(dist=float(c.dist), normal_world=n,
                            point_world=np.asarray(c.pos)-.5*float(c.dist)*n,
                            midpoint_world=np.asarray(c.pos).copy())
        return best

    def _fingertip_contact_info(self):
        c = self.measured_contact()
        return (c is not None and c['dist'] <= 0., None if c is None else c['normal_world'])

    def _refresh_osc_tensors(self):
        pass

    def _apply_arm_torque(self, tau):
        self.data.ctrl[:] = tau

    def _simulate_once(self, **kwargs):
        pass  # GPU coordinator advances every world together.

    def show_target(self, *args):
        pass

    def show_best_contact(self, *args):
        pass

    def observation(self, trial):
        from examples.mpc.franka.ik2.test_mppi_isaac import _plan_payload
        fields, _, _, dist = _plan_payload(
            self, self.contact, self.param_.table_height+self.args.ground_height_threshold)
        linear, angular = self.get_object_velocity_world()
        fields.update(trial=trial, seq=self.seq, sim_time=float(self.data.time), contact_distance=dist,
                      object_linear_velocity=linear, object_angular_velocity=angular,
                      osc_force=self._last_osc_force, osc_pd=self._last_osc_pd,
                      osc_upright_guard_active=self._osc_upright_guard_active,
                      joint_velocity_projections=getattr(self, 'joint_velocity_projections', 0),
                      joint_velocity_max_unlimited=getattr(self, 'joint_velocity_max_unlimited', 0.),
                      physics_scene_sha256=getattr(self, 'physics_scene_sha256', None),
                      scm_transitions=[] if self._scm_recorder is None else self._scm_recorder.snapshot())
        return fields


class MJWarpBatch:
    def __init__(self, args, trials):
        self.args = args
        if args.mjwarp_nconmax <= 0 or args.mjwarp_njmax <= 0:
            raise ValueError('MJWarp contact/constraint capacities must be positive')
        wp.init()
        self.device = wp.get_device(args.sim_device)
        if not self.device.is_cuda:
            raise ValueError('MJWarp evaluation requires --sim-device cuda:N')
        params = [build_param(args, trial) for trial in trials]
        xml = scene_xml(params[0], args)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.scene_sha256 = hashlib.sha256(xml.encode()).hexdigest()
        self.envs = [MuJoCoFrankaOSC(self.model, args, p) for p in params]
        with wp.ScopedDevice(self.device):
            self.wmodel = mjwarp.put_model(self.model)
            self.wdata = mjwarp.put_data(self.model, self.envs[0].data, nworld=len(trials),
                                        nconmax=args.mjwarp_nconmax, njmax=args.mjwarp_njmax)
            self.wdata.qpos.assign(np.asarray([e.data.qpos for e in self.envs], dtype=np.float32))
            self.wdata.qvel.assign(np.asarray([e.data.qvel for e in self.envs], dtype=np.float32))
            mjwarp.forward(self.wmodel, self.wdata)
        self.steps = int(round(POLICY_INTERVAL/DYWA_SIM_DT))
        self.velocity_limits = (np.asarray(isaac_robot_dynamics()['joint_properties']['velocity'])
                                if getattr(args, 'mjwarp_scene', 'isaac') == 'isaac' else None)
        for env in self.envs:
            env.physics_scene_sha256 = self.scene_sha256
            env.joint_velocity_projections = 0
            env.joint_velocity_max_unlimited = 0.

    def reset(self, i, trial):
        self.envs[i].reset(build_param(self.args, trial))
        self.envs[i].joint_velocity_projections = 0
        self.envs[i].joint_velocity_max_unlimited = 0.
        mask = np.zeros(len(self.envs), dtype=bool)
        mask[i] = True
        with wp.ScopedDevice(self.device):
            mjwarp.reset_data(self.wmodel, self.wdata, wp.array(mask, dtype=wp.bool))
            for name in ('qpos', 'qvel'):
                values = getattr(self.wdata, name).numpy()
                values[i] = getattr(self.envs[i].data, name)
                getattr(self.wdata, name).assign(values)

    def command(self, i, cmd, record_transition=True):
        env = self.envs[i]
        env.set_via_action(cmd['via'], action=cmd.get('action'), seq=cmd.get('seq'),
                           policy_dt=cmd.get('policy_dt'), press=cmd.get('press'),
                           submitted_at=cmd.get('submitted_at'), planner_dt=cmd.get('planner_dt'),
                           path_blocked=cmd.get('path_blocked'), release_pressure=cmd.get('release_pressure', False),
                           preserve_upright=cmd.get('preserve_upright', False),
                           solver_backend=cmd.get('solver_backend'))
        if record_transition and env._scm_recorder is not None:
            env._scm_recorder.begin(env.get_policy_state(), env.observation(env.param_.trial_seed_))

    def step(self, indices, frames=None):
        active = set(indices)
        ctrl = np.zeros((len(self.envs), self.model.nu), dtype=np.float32)
        with wp.ScopedDevice(self.device):
            for _ in range(self.steps if frames is None else frames):
                for i in indices:
                    self.envs[i].step_control_frame(draw=False, sync_realtime=False)
                    ctrl[i] = self.envs[i].data.ctrl
                self.wdata.ctrl.assign(ctrl)
                mjwarp.step(self.wmodel, self.wdata)
                qpos, qvel, times = self.wdata.qpos.numpy(), self.wdata.qvel.numpy(), self.wdata.time.numpy()
                projected = False
                ncontact = int(self.wdata.nacon.numpy()[0])
                ncollision = int(self.wdata.ncollision.numpy()[0])
                constraints = self.wdata.nefc.numpy()
                if (max(ncontact, ncollision) > self.wdata.naconmax or np.any(constraints > self.wdata.njmax)):
                    raise RuntimeError('MJWarp contact/constraint buffer overflow; increase --mjwarp-nconmax/--mjwarp-njmax')
                for i, env in enumerate(self.envs):
                    if i not in active:
                        # Finished worlds stay frozen and cannot overflow the
                        # batch's buffers while the remaining trials run.
                        qpos[i], qvel[i], times[i] = env.data.qpos, env.data.qvel, env.data.time
                        continue
                    if not np.isfinite(qpos[i]).all() or not np.isfinite(qvel[i]).all():
                        raise FloatingPointError('MJWarp produced a nonfinite state in world %s '
                                                 '(previous joint velocity=%s)' % (i, env.data.qvel[:7]))
                    if self.velocity_limits is not None:
                        # MuJoCo has no equivalent of PhysX's max joint velocity
                        # constraint. Project the seven hinge velocities and
                        # their implicitfast Euler position increments together.
                        # This approximates the limit; it does not reproduce
                        # PhysX's coupled contact/limit impulse solve.
                        raw = qvel[i, :7].copy()
                        bounded = np.clip(raw, -self.velocity_limits, self.velocity_limits)
                        env.joint_velocity_max_unlimited = max(env.joint_velocity_max_unlimited,
                                                               float(np.max(np.abs(raw))))
                        if np.any(raw != bounded):
                            env.joint_velocity_projections += 1
                            qpos[i, :7] += self.model.opt.timestep * (bounded - raw)
                            qvel[i, :7] = bounded
                            projected = True
                    env.data.qpos[:], env.data.qvel[:], env.data.time = qpos[i], qvel[i], float(times[i])
                    mujoco.mj_forward(self.model, env.data)
                if projected or len(active) != len(self.envs):
                    self.wdata.qpos.assign(qpos)
                    self.wdata.qvel.assign(qvel)
                    self.wdata.time.assign(times)
        for i in indices if frames is None else []:
            env = self.envs[i]
            env.seq += 1
            if env._scm_recorder is not None:
                env._scm_recorder.finish(env.get_policy_state(), env.observation(env.param_.trial_seed_))

    def observation(self, i, trial):
        return self.envs[i].observation(trial)
