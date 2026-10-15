"""Contact-model and actuator comparisons on the archived loaded state."""
from audit import *


def main():
    args = build_argparser().parse_args(['--headless', '--screenshot-interval', '0',
        '--scene-output', '/tmp/trigrasp_lift_probe.xml'])
    demo = AllegroTabletopTrigraspDemo(args)
    saved = np.load(OUT/'first_loaded.npz')
    demo.data.qpos[:] = saved['qpos']; demo.data.qvel[:] = saved['qvel']; demo.data.ctrl[:] = saved['ctrl']
    demo.online_servo_target = saved['servo'].copy()
    mujoco.mj_forward(demo.model, demo.data)
    np.savez(OUT/'probe_state.npz', qpos=demo.data.qpos, qvel=demo.data.qvel)
    summary = json.loads((OUT/'summary.json').read_text())
    row = observe(demo)
    model = demo.mpc.model; p = demo.mpc_param
    mp = saved['mp']; offset = 1+model.pose_size
    model.reference = mp[1:offset].copy()
    model.object_inverse = mp[offset:offset+36].reshape(6, 6, order='F')
    model.robot_inverse = mp[offset+36:offset+36+p.n_cmd_].copy()
    model.external_wrench = mp[-6:].copy()
    state, phi, jac = saved['state'], saved['phi'], saved['jac']
    data = mujoco.MjData(demo.model); mujoco.mj_copyData(data, demo.model, demo.data)
    by_name = {name: geom for geom, name in demo.fingertip_geom_id_to_name.items()}
    by_geom = {c['geom']: c for c in row['contacts']}
    points = np.array([by_geom[by_name[name]]['point'] for name in demo.active_finger_names])
    normals = np.array([by_geom[by_name[name]]['normal'] for name in demo.active_finger_names])
    rotation = demo.get_object_pose()[2]
    head = p.cost_param_head_size
    target = saved['cp'][head:head+3*p.num_contacts].reshape(-1, 3)@rotation.T
    u = cs.MX.sym('force_command', p.n_cmd_)
    force = model.contact_force_fn(state, u, phi, jac, mp)
    lift_force = sum(force[3*i+2] for i in range(p.num_contacts))
    residual = force-cs.DM(target.reshape(-1))
    loss = cs.sumsqr(residual)
    force_goal = cs.Function('coupled_force_allocation', [u], [loss, cs.gradient(loss, u)])
    force_max = cs.Function('coupled_max_lift', [u], [-lift_force, cs.gradient(-lift_force, u)])
    bounds = list(zip(saved['bounds'][0], saved['bounds'][1]))
    actions = {'zero': np.zeros(p.n_cmd_), 'mpc': np.asarray(summary['first_loaded']['action']),
               'offline_mpc': np.asarray(summary['offline_optimizer']['action'])}
    base = np.zeros(p.n_cmd_); base[2] = .001; actions['base_up'] = base
    force_solutions = {}
    for name, fn in [('force_tracking', force_goal), ('max_pad_lift', force_max)]:
        def objective(action):
            value, grad = fn(action)
            return float(value), np.asarray(grad).reshape(-1)
        best = None
        for seed in (actions['mpc'], actions['base_up'], np.zeros(p.n_cmd_)):
            result = minimize(objective, seed, jac=True, method='L-BFGS-B', bounds=bounds,
                options=dict(maxiter=100, ftol=1e-12, gtol=1e-7, maxls=30))
            if best is None or result.fun < best.fun:
                best = result
        actions[name] = best.x
        force_solutions[name] = dict(success=bool(best.success), iterations=best.nit, cost=best.fun,
            action=best.x, model_force_sum=np.asarray(model.contact_force_fn(state, best.x, phi, jac, mp)).reshape(-1,3).sum(0))
        print('[probe]', name, force_solutions[name], flush=True)
    one_step = {}
    for name, action in actions.items():
        mujoco.mj_copyData(demo.data, demo.model, data)
        demo.online_servo_target = saved['servo'].copy()
        start = demo.get_object_pose()[0].copy(); initial_velocity = demo.data.qvel[demo.object_dof_adr:demo.object_dof_adr+6].copy()
        command = demo.online_servo_target+action
        kp = demo.model.actuator_gainprm[:, 0]
        average = np.zeros(6)
        for _ in range(args.mj_steps_per_command):
            mujoco.mj_forward(demo.model, demo.data)
            bias = demo.data.qfrc_bias[demo.hand_dof_adr]-demo.data.qfrc_passive[demo.hand_dof_adr]-demo.data.qfrc_applied[demo.hand_dof_adr]
            demo.data.ctrl[:] = np.clip(command+bias/kp, demo.hand_joint_ranges[:,0], demo.hand_joint_ranges[:,1])
            mujoco.mj_step(demo.model, demo.data)
            measured, _ = object_contact_wrenches(demo.model, demo.data, demo.object_body_id,
                demo.fingertip_geom_id_to_name, demo.hand_collision_geom_ids)
            average += measured['pads']/args.mj_steps_per_command
        mujoco.mj_forward(demo.model, demo.data)
        after = observe(demo)
        predicted = np.asarray(model.prediction_fn(state, action, phi, jac, mp)).reshape(-1)
        pred_force = np.asarray(model.contact_force_fn(state, action, phi, jac, mp)).reshape(-1,3)
        one_step[name] = dict(model_pad_force_sum=pred_force.sum(0), actual_average_pad_wrench=average,
            actual_final_pad_wrench=after['wrench']['pads'], actual_object_displacement=after['pos']-start,
            predicted_object_displacement=predicted[:3]-state[:3], actual_contact_count=after['contact_count'],
            actual_delta_vz=after['velocity'][2]-initial_velocity[2], predicted_delta_vz=predicted[model.pose_size+2]-state[model.pose_size+2])
        print('[probe] one-step', name, one_step[name], flush=True)
    # Test response sensitivity without changing production defaults or codegen.
    from examples.mpc.franka.trigrasp.trigrasp import AllegroOnlineContactModel
    response = {}
    for tau in (.05, .02, .01, .005):
        params = copy.copy(p); params.soft_scm_response_time = tau; params.linearize_force_cost = False
        alternate = AllegroOnlineContactModel(params)
        f = np.asarray(alternate.contact_force_fn(state, actions['mpc'], phi, jac, mp)).reshape(-1,3)
        prediction = np.asarray(alternate.prediction_fn(state, actions['mpc'], phi, jac, mp)).reshape(-1)
        response[str(tau)] = dict(pad_force_sum=f.sum(0), prediction=prediction[:3],
            normal_forces=np.einsum('ij,ij->i', f, normals))
    import models.soft_scm_contact as scm
    sweeps = {}
    default_sweeps = scm.GROUPED_CONTACT_SWEEPS
    try:
        for count in (1, 4, 8, 16, 32, 64):
            scm.GROUPED_CONTACT_SWEEPS = count
            params = copy.copy(p); params.linearize_force_cost = False
            alternate = AllegroOnlineContactModel(params)
            f = np.asarray(alternate.contact_force_fn(state, actions['mpc'], phi, jac, mp)).reshape(-1,3)
            prediction = np.asarray(alternate.prediction_fn(state, actions['mpc'], phi, jac, mp)).reshape(-1)
            sweeps[str(count)] = dict(pad_force_sum=f.sum(0), prediction=prediction[:3],
                normal_forces=np.einsum('ij,ij->i', f, normals))
            print('[probe] GS sweeps', count, sweeps[str(count)], flush=True)
    finally:
        scm.GROUPED_CONTACT_SWEEPS = default_sweeps
    active_blocks = np.asarray(jac).reshape(p.max_ncon_,4,p.n_qvel_)
    report = dict(restored_state=row, desired_lambda_pad_forces=target, force_solutions=force_solutions,
        one_step=one_step, response_sensitivity=response, sweep_sensitivity=sweeps,
        modeled_object_hand_blocks=np.flatnonzero((active_blocks[:,:,:6]**2).sum((1,2))>0).tolist(),
        modeled_self_blocks=np.flatnonzero((active_blocks[:,:,:6]**2).sum((1,2))==0).tolist())
    (OUT/'model_probe.json').write_text(json.dumps(report, default=serial, indent=2))
    demo.close()


if __name__=='__main__':
    main()
