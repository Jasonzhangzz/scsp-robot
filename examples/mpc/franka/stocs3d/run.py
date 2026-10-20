"""Solve STOCS-3D offline, then execute the trajectory with test_mpc_isaac.py.

Contact selection is ``STOCS.oracle``. The CITO program is the vendored
``MPCC`` solved by Drake's SNOPT, the solver that repository calls.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR = Path(__file__).resolve().parent / 'vendor'


def import_stocs():
    """Load the vendored package in front of this repository's ``utils`` package."""
    import klampt
    import klampt.math.so3 as so3
    import numpy as np
    if tuple(int(part) for part in klampt.__version__.split('.')[:2]) != (0, 9):
        raise RuntimeError('STOCS-3D calls Klampt 0.9 VolumeGrid code; acc has Klampt %s' % klampt.__version__)
    # SNOPT quaternions are not exact unit quaternions. Project before Klampt's
    # rotation-vector conversion so its near-pi assertion is not tripped.
    original_interpolate = so3.interpolate

    def project(rotation):
        matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
        left, _, right = np.linalg.svd(matrix)
        projected = left @ right
        if np.linalg.det(projected) < 0:
            left = left.copy()
            left[:, -1] *= -1
            projected = left @ right
        return projected.reshape(-1).tolist()

    def interpolate(start, goal, fraction):
        return original_interpolate(project(start), project(goal), fraction)

    so3.interpolate = interpolate
    vendor = str(VENDOR)
    sys.path.insert(0, vendor)
    for name in list(sys.modules):
        if name == 'utils' or name.startswith('utils.') or name.startswith('semiinfinite') or name in ('STOCS', 'MPCC'):
            del sys.modules[name]
    import STOCS
    return STOCS


def validate_options(args):
    if getattr(args, 'contactsdf', False) or getattr(args, 'contact_model', '') == 'contactsdf':
        raise ValueError('STOCS-3D does not use ContactSDF')
    if args.physics_backend != 'isaac' or int(args.num_envs) != 1:
        raise ValueError('STOCS-3D plans one object trajectory and executes it in one Isaac scene')
    if args.floating_fingertip:
        raise ValueError('STOCS-3D execution uses the Franka OSC scene')
    if getattr(args, 'online_scm', False):
        raise ValueError('STOCS-3D does not identify a contact model online')


def solve_trial(stocs_module, scene, cache):
    from examples.mpc.franka.stocs3d.task import build_stocs_params
    from examples.mpc.franka.stocs3d.trajectory import APPROACH_S, fingertip_knots
    params = build_stocs_params(scene, cache)
    # SNOPT's print file is the relative path hard-coded in MPCC.solve.
    output = Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'tmp').mkdir(exist_ok=True)
    cwd = Path.cwd()
    os.chdir(output)
    started = time.perf_counter()
    try:
        solver = stocs_module.STOCS(params)
        result, average_contacts = solver.solve()
    finally:
        os.chdir(cwd)
    planning_time_s = time.perf_counter() - started
    knots = fingertip_knots(result, params, params['fingertip_radius'])
    knots.update(
        object=Path(scene.model_path_).name,
        planning_time_s=float(planning_time_s),
        average_contacts=float(average_contacts),
        approach_s=APPROACH_S,
        init_xyz=np_list(scene.init_obj_qpos_[:3]),
        init_quat_wxyz=np_list(scene.init_obj_qpos_[3:7]),
        radius=float(params['fingertip_radius']),
        goal_xyz=np_list(scene.target_p_),
        goal_quat_wxyz=np_list(scene.target_q_),
        table_height=float(scene.table_height),
    )
    return knots


def np_list(value):
    import numpy as np
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def main():
    # Parsing and the scene use the repository model. Isaac is a later process.
    os.environ['SCSP_PLANNER_ONLY'] = '1'
    from examples.mpc.franka.runtime.params import ExplicitMPCParams
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    import argparse
    parser = _add_rollout_policy_args(argparse.ArgumentParser(
        description='Offline STOCS-3D contact selection and CITO, then Isaac execution.'))
    parser.add_argument('--stocs3d', action='store_true', help='Solve STOCS-3D and track it in Isaac.')
    parser.add_argument('--stocs-solve-only', action='store_true',
                        help='Write the offline trajectory and skip Isaac execution.')
    args = parser.parse_args()
    try:
        validate_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.rollout_summary_json).parent if args.rollout_summary_json else Path('output/stocs3d')
    output.mkdir(parents=True, exist_ok=True)
    start = int(args.trial_start)
    scenes = [ExplicitMPCParams(args, rand_seed=trial, target_type=args.target_type, build_optimizer=False)
              for trial in range(start, start + max(1, int(args.trial_num)))]
    stocs_module = import_stocs()
    for trial, scene in zip(range(start, start + len(scenes)), scenes):
        print('[stocs3d] solving trial %d' % trial, flush=True)
        knots = solve_trial(stocs_module, scene, output / 'geometry')
        knots['trial'] = trial
        knots['object'] = args.obj
        knots['target_type'] = args.target_type
        path = output / ('trial_%03d.json' % trial)
        path.write_text(json.dumps(knots) + '\n')
        print('[stocs3d] trial %d solved=%s iterations=%d -> %s' % (
            trial, knots['solved'], knots['iterations'], path), flush=True)
    if args.stocs_solve_only:
        return
    flags = [flag for flag in sys.argv[1:] if flag not in ('--stocs3d', '--stocs-solve-only')]
    os.environ.pop('SCSP_PLANNER_ONLY', None)
    isaac = ROOT / 'examples/mpc/franka/ik2/test_mpc_isaac.py'
    os.execv(sys.executable, [sys.executable, '-u', str(isaac), '--stocs-trajectory', str(output), *flags])


if __name__ == '__main__':
    main()
