"""Diagnostic force-feedback oracle; not a replacement MPC policy."""
from audit import *


def main():
    args = build_argparser().parse_args(['--headless', '--screenshot-interval', '0',
        '--scene-output', '/tmp/trigrasp_feedback_probe.xml'])
    demo = AllegroTabletopTrigraspDemo(args)
    saved = np.load(OUT/'first_loaded.npz')
    source = json.loads((OUT/'summary.json').read_text())['first_loaded']
    by_name = {name: geom for geom, name in demo.fingertip_geom_id_to_name.items()}
    by_geom = {c['geom']: c for c in source['contacts']}
    original_rot = np.asarray(source['quat']); from examples.mpc.franka.trigrasp.trigrasp import quat_wxyz_to_mat
    r0 = quat_wxyz_to_mat(original_rot)
    local_normals = np.array([by_geom[by_name[name]]['normal'] for name in demo.active_finger_names])@r0
    target_loads = np.array([by_geom[by_name[name]]['normal_force'] for name in demo.active_finger_names])
    reports = {}
    for cone in ('pyramidal', 'elliptic'):
        demo.model.opt.cone = getattr(mujoco.mjtCone, 'mjCONE_'+cone.upper())
        for mode in ('none', 'feedback'):
            scale = 1.
            demo.data.qpos[:] = saved['qpos']; demo.data.qvel[:] = saved['qvel']; demo.data.ctrl[:] = saved['ctrl']
            demo.online_servo_target = saved['servo'].copy()
            mujoco.mj_forward(demo.model, demo.data)
            initial = demo.get_object_pose()[0].copy(); trace = []
            for step in range(300):
                mujoco.mj_forward(demo.model, demo.data)
                pos, _, rot = demo.get_object_pose()
                _, j = demo.online_tip_kinematics()
                summary = demo.get_fingertip_contact_summary()
                pressure = np.array([summary[name] for name in demo.active_finger_names])
                inward = local_normals@rot.T
                displacement = np.clip(.003*(scale*target_loads-pressure), -.001, .001)[:, None]*inward
                jf = j[:, 6:]
                delta = jf.T@np.linalg.solve(jf@jf.T+1e-5*np.eye(jf.shape[0]), displacement.reshape(-1))
                delta = np.clip(delta, -demo.mpc_param.contact_u_limit[6:], demo.mpc_param.contact_u_limit[6:])
                target = demo.online_servo_target.copy()
                if mode == 'feedback':
                    target[6:] += delta
                # Establish preload before lifting, and pause when pressure
                # collapses. This is a diagnostic oracle using measured force.
                can_lift = step>=20 and (mode == 'none' or np.all(pressure >= .5*scale*target_loads))
                if can_lift:
                    target[2] += .001
                demo.execute_online_joint_target(target)
                mujoco.mj_forward(demo.model, demo.data)
                frame = observe(demo); frame['lifting'] = bool(can_lift); trace.append(frame)
            key = f'{cone}_{mode}'
            reports[key] = dict(height_gain=float(trace[-1]['pos'][2]-initial[2]),
                max_height_gain=float(max(r['pos'][2]-initial[2] for r in trace)),
                hand_height_gain=float(demo.get_hand_qpos()[2]-saved['state'][9]), trace=trace)
            print('[feedback]', cone, mode, reports[key]['height_gain'],
                reports[key]['max_height_gain'], trace[-1]['contact_count'], flush=True)
    (OUT/'feedback_probe.json').write_text(json.dumps(reports, default=serial, indent=2))
    demo.close()


if __name__=='__main__':
    main()
