"""Reproduce lift failure; compare geometry, contact capacity, MPC and execution."""
from pathlib import Path
import copy
import json
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'planning').is_dir())
sys.path.insert(0, str(ROOT))
import casadi as cs
import mujoco
import numpy as np
from scipy.optimize import minimize

from examples.mpc.franka.trigrasp.trigrasp import AllegroTabletopTrigraspDemo, build_argparser
from planning.grasp_diagnostics import object_contact_wrenches, bounded_vertical_acceleration

OUT = Path(__file__).resolve().parent


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def observe(demo):
    pos, quat, rot = demo.get_object_pose()
    mass, external = demo.lambda_object_dynamics(rot)
    totals, contacts = object_contact_wrenches(demo.model, demo.data, demo.object_body_id,
        demo.fingertip_geom_id_to_name, demo.hand_collision_geom_ids)
    mixed = totals['all'].copy(); mixed[3:] = rot.T@mixed[3:]
    dofs = slice(demo.object_dof_adr, demo.object_dof_adr+6)
    balance_error = np.max(np.abs(mixed-demo.data.qfrc_constraint[dofs]))
    if balance_error > 1e-6:
        raise AssertionError(f'contact wrench sign/frame mismatch: {balance_error}')
    pads = [c for c in contacts if c['group']=='pads' and c['normal_force']>demo.args.contact_force_threshold]
    world = np.eye(6); world[3:, 3:] = rot
    capacity = bounded_vertical_acceleration(
        [c['point']-pos for c in pads], [c['normal'] for c in pads], [c['mu'] for c in pads],
        world@mass@world.T, external, [max(c['normal_force'], 0.) for c in pads],
        tangents=[c['tangents'] for c in pads] if demo.model.opt.cone==mujoco.mjtCone.mjCONE_PYRAMIDAL else None)
    return dict(pos=pos, quat=quat, wrench=totals, contacts=pads,
        contact_count=len(pads), measured_capacity=capacity, balance_error=balance_error,
        velocity=demo.data.qvel[dofs].copy(), actual_acceleration=demo.data.qacc[dofs].copy())


def run():
    args = build_argparser().parse_args(['--headless', '--screenshot-interval', '0', '--max-steps', '450',
        '--scene-output', '/tmp/trigrasp_lift_audit.xml', '--log-every', '25',
        '--diagnostics-output', str(OUT/'baseline.json')])
    demo = AllegroTabletopTrigraspDemo(args)
    records, snapshots, request = [], {}, {}
    native = demo.mpc._plan_once_acados
    def native_capture(curr_x, phi_vec, jac_mat, cost_params, sol_guess, u_lb, u_ub):
        request.update(cp=np.asarray(cost_params).copy(), bounds=np.stack((u_lb, u_ub)))
        result = native(curr_x, phi_vec, jac_mat, cost_params, sol_guess, u_lb, u_ub)
        request['raw_result'] = copy.deepcopy(result)
        return result
    demo.mpc._plan_once_acados = native_capture
    plan = demo.mpc.plan_contacts
    def plan_capture(state, phi, jac, **kw):
        result = plan(state, phi, jac, **kw)
        observed = observe(demo)
        mp = demo.mpc.model.runtime_parameters(demo.mpc_param.model_params)
        forces = np.asarray(demo.mpc.model.contact_force_fn(state, result['action'], phi, jac, mp)).reshape(-1, 3)
        ideal_forces = np.asarray(kw['desired_force_local'])@demo.get_object_pose()[2].T
        row = dict(step=len(records), loaded=kw['loaded'], **observed,
            model_pad_force_sum=forces.sum(0), model_pad_forces=forces,
            lambda_pad_force_sum=ideal_forces.sum(0), action=result['action'].copy(),
            predicted_velocity=np.asarray(result['rollout_q'])[0, demo.mpc.model.pose_size:demo.mpc.model.pose_size+6],
            selected_control=result.get('selected_control', 'mpc'), scale=result.get('nonlinear_step_scale'))
        records.append(row)
        if kw['loaded'] and 'first_loaded' not in snapshots:
            data = mujoco.MjData(demo.model); mujoco.mj_copyData(data, demo.model, demo.data)
            snapshots['first_loaded'] = dict(data=data, state=state.copy(), phi=phi.copy(), jac=jac.copy(),
                mp=mp.copy(), servo=demo.online_servo_target.copy(), cp=request['cp'].copy(),
                bounds=request['bounds'].copy(), result=copy.deepcopy(result), kw=copy.deepcopy(kw), row=row)
            print('[audit] captured first loaded state', row['step'], 'physical/model/lambda Fz',
                observed['wrench']['pads'][2], forces.sum(0)[2], ideal_forces.sum(0)[2], flush=True)
        return result
    demo.mpc.plan_contacts = plan_capture
    try:
        demo.run_online()
        (OUT/'wrenches.json').write_text(json.dumps(records, default=serial, indent=2))
        if not snapshots:
            raise RuntimeError('No verified physical grasp in the baseline')
        snap = snapshots['first_loaded']
        np.savez(OUT/'first_loaded.npz', qpos=snap['data'].qpos, qvel=snap['data'].qvel,
            ctrl=snap['data'].ctrl, servo=snap['servo'], state=snap['state'], phi=snap['phi'], jac=snap['jac'],
            mp=snap['mp'], cp=snap['cp'], bounds=snap['bounds'])
        print('[audit] counterfactual execution from identical measured grasp', flush=True)
        probes = {}
        for mode in ('hold_servo', 'base_lift', 'base_lift_extra_preload'):
            probes[mode] = execute_probe(demo, snap, mode)
            print('[audit]', mode, probes[mode]['height_gain'], probes[mode]['final_contact_count'], flush=True)
        offline = offline_solve(demo, snap)
        summary = dict(first_loaded=snap['row'], probes=probes, offline_optimizer=offline,
            physical_cone=int(demo.model.opt.cone), friction=args.object_friction,
            object_mass=args.obj_mass, gravity_force=args.obj_mass*9.81,
            baseline_loaded_steps=[r['step'] for r in records if r['loaded']],
            max_wrench_balance_error=max(r['balance_error'] for r in records))
        (OUT/'summary.json').write_text(json.dumps(summary, default=serial, indent=2))
    finally:
        demo.close()


def execute_probe(demo, snap, mode):
    original_data, original_servo = demo.data, demo.online_servo_target.copy()
    demo.data = mujoco.MjData(demo.model); mujoco.mj_copyData(demo.data, demo.model, snap['data'])
    demo.online_servo_target = snap['servo'].copy()
    start = demo.get_object_pose()[0].copy()
    if mode=='base_lift_extra_preload':
        tips, jac = demo.online_tip_kinematics()
        normals = snap['kw']['desired_force_local']
        # Use actual contact normals, not the direction of the ideal force.
        pad_normals = demo.online_contact_normals_local.copy()@demo.get_object_pose()[2].T
        # Restore the snapshot's physical normals in fingertip order.
        by_geom = {c['geom']: c['normal'] for c in snap['row']['contacts']}
        pad_normals = np.array([by_geom[g] for g in demo.fingertip_geom_id_to_name if demo.fingertip_geom_id_to_name[g] in demo.active_finger_names])
        jf = jac[:, 6:]
        squeeze = jf.T@np.linalg.solve(jf@jf.T+1e-5*np.eye(jf.shape[0]), (.003*pad_normals).reshape(-1))
        demo.online_servo_target[6:] += squeeze
    trace = []
    for step in range(120):
        target = demo.online_servo_target.copy()
        if mode!='hold_servo':
            target[2] += .001  # Same 0.04 m/s bound as contact MPC.
        demo.execute_online_joint_target(target)
        mujoco.mj_forward(demo.model, demo.data)
        trace.append(dict(step=step, **observe(demo)))
    pos, _, _ = demo.get_object_pose()
    report = dict(height_gain=float(pos[2]-start[2]), hand_height_gain=float(demo.get_hand_qpos()[2]-snap['state'][9]),
        final_contact_count=trace[-1]['contact_count'], trace=trace)
    demo.data, demo.online_servo_target = original_data, original_servo
    return report


def offline_solve(demo, snap):
    """Check whether RTI missed a better command for the very same objective."""
    model, p = demo.mpc.model, demo.mpc_param
    horizon, nu = p.mpc_horizon_, p.n_cmd_
    u = cs.MX.sym('offline_controls', horizon*nu)
    state = cs.DM(snap['state']); total = 0.; states = []
    for i in range(horizon):
        action = u[i*nu:(i+1)*nu]
        total += demo.mpc.path_cost_fn(state, action, snap['cp'])
        state = model.prediction_fn(state, action, snap['phi'], snap['jac'], snap['mp'])
        states.append(state)
    total += demo.mpc.final_cost_fn(state, snap['cp'])
    fn = cs.Function('offline_same_mpc_cost', [u], [total, cs.gradient(total, u), cs.vertcat(*states)])
    bounds = list(zip(np.tile(snap['bounds'][0], horizon), np.tile(snap['bounds'][1], horizon)))
    start = np.asarray(snap['result']['u_traj']).reshape(-1)
    def objective(controls):
        value, gradient, _ = fn(controls)
        return float(value), np.asarray(gradient).reshape(-1)
    print('[audit] independent nonlinear solve of the original MPC cost', flush=True)
    result = minimize(objective, start, jac=True, bounds=bounds, method='L-BFGS-B',
        options=dict(maxiter=100, ftol=1e-12, gtol=1e-7, maxls=30))
    value, gradient, states = fn(result.x)
    trajectory = np.asarray(states).reshape(horizon, p.n_qpos_)
    violation = max(float(np.max(p.mpc_q_lb_-trajectory)), float(np.max(trajectory-p.mpc_q_ub_)), 0.)
    first = result.x[:nu]
    force = np.asarray(model.contact_force_fn(snap['state'], first, snap['phi'], snap['jac'], snap['mp'])).reshape(-1, 3)
    report = dict(success=bool(result.success), message=result.message, iterations=result.nit,
        initial_cost=objective(start)[0], optimized_cost=float(value), action=first,
        pad_force_sum=force.sum(0), predicted_object_positions=trajectory[:, :3], state_bound_violation=violation)
    print('[audit] nonlinear comparison', report, flush=True)
    return report


if __name__=='__main__':
    run()
