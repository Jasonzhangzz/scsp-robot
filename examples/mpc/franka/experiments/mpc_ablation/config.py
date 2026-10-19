"""Ablation selection without importing physics or numerical libraries."""
import argparse
import math


MODES = {
    'full': 'Unmodified test_mpc_isaac baseline.',
    'a1': 'MPC tracks only best_contact_point_world.',
    'a2': 'MPC tracks only p_arm_world; best-contact selection is bypassed.',
    'a3': 'Lambda searches a cached mesh OBB; the MPC and Isaac mesh remain unchanged.',
    'a4': 'MuJoCo native contact QP predicts the best contact.',
    'a5': 'CMA-ES solves the unchanged lambda force objective.',
    'a6': 'Direct AdamW MPC with ComFree rollouts and finite-difference gradients; no CSO.',
    'a8': 'Use the second (pose) stage cost throughout; never switch to flip.',
}


def add_ablation_args(parser):
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--ablation-mode', choices=tuple(MODES), default='full',
                           help='Ablation to run (default: full baseline).')
    for mode, description in MODES.items():
        if mode != 'full':
            selection.add_argument('--' + mode, dest='ablation_mode',
                                   action='store_const', const=mode, help=description)
    parser.add_argument('--a3-obb-cache-dir', default='',
                        help='Offline OBB cache directory (default: ~/.cache/scsp_robot/a3_obb).')
    parser.add_argument('--a4-qp-maxiter', type=int, default=12,
                        help='Maximum outer force-solve iterations for MuJoCo QP.')
    parser.add_argument('--a5-population-size', type=int, default=24)
    parser.add_argument('--a5-population-iterations', type=int, default=12)
    parser.add_argument('--a6-center-weight', type=float, default=.05,
                        help='A6 approach attraction relative to contact_coef (default .05).')
    parser.add_argument('--a6-adamw-iterations', type=int, default=30)
    parser.add_argument('--a6-adamw-lr', type=float, default=.1,
                        help='AdamW learning rate in units of the MPC command bound.')
    parser.add_argument('--a6-adamw-weight-decay', type=float, default=.01)
    parser.add_argument('--a6-fd-epsilon', type=float, default=1e-5,
                        help='Central-difference perturbation of each command, in meters.')
    parser.add_argument('--verify-accept-threshold', type=float, default=.45,
                        help='Accept a nearby same-patch p_arm when tightness is below this value.')
    parser.add_argument('--verify-release-threshold', type=float, default=.65,
                        help='Release tightness for an already accepted p_arm (default .65).')
    from .sensitivity import add_sensitivity_args
    add_sensitivity_args(parser)
    return parser


def configure_ablation(args):
    mode = getattr(args, 'ablation_mode', 'full')
    if mode not in MODES:
        raise ValueError('Unknown ablation mode: ' + str(mode))
    args.ablation_mode = mode
    defaults = dict(a3_obb_cache_dir='', a4_qp_maxiter=12, a5_population_size=24,
                    a5_population_iterations=12, a6_center_weight=.05,
                    a6_adamw_iterations=30, a6_adamw_lr=.1,
                    a6_adamw_weight_decay=.01, a6_fd_epsilon=1e-5,
                    verify_accept_threshold=.45, verify_release_threshold=.65)
    for name, default in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    enter, release = args.verify_accept_threshold, args.verify_release_threshold
    if not (math.isfinite(enter) and math.isfinite(release) and 0 <= enter <= release <= 1):
        raise ValueError('verify thresholds must satisfy 0 <= accept <= release <= 1')
    if mode == 'a4' and args.a4_qp_maxiter < 1:
        raise ValueError('--a4-qp-maxiter must be positive')
    if mode == 'a5' and (args.a5_population_size < 4 or args.a5_population_iterations < 1):
        raise ValueError('CMA-ES requires population size >= 4 and iterations >= 1')
    if mode == 'a6' and (not math.isfinite(args.a6_center_weight) or args.a6_center_weight <= 0):
        raise ValueError('--a6-center-weight must be finite and positive')
    if mode == 'a6':
        if getattr(args, 'physics_backend', 'isaac') != 'isaac':
            raise ValueError('A6 executes in Isaac Gym; use --physics-backend isaac')
        if args.a6_adamw_iterations < 1:
            raise ValueError('--a6-adamw-iterations must be positive')
        for name in ('a6_adamw_lr', 'a6_fd_epsilon', 'a6_adamw_weight_decay'):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0 or (name != 'a6_adamw_weight_decay' and value == 0):
                raise ValueError('--' + name.replace('_', '-') + ' has an invalid value')
        if args.a6_adamw_lr * args.a6_adamw_weight_decay >= 1:
            raise ValueError('A6 learning rate times weight decay must be less than one')
        if (getattr(args, 'online_scm', False) or getattr(args, 'scm_parameters', '')
                or getattr(args, 'scm_record_data', False)):
            raise ValueError('A6 uses ComFree and does not support online SCM')
        if getattr(args, 'tool_orientation_control', 'free') == 'mpc':
            raise ValueError('A6 currently optimizes Cartesian translation commands only')
        args.contact_model = 'comfree'
    if mode != 'full' and getattr(args, 'sampling_c3', False):
        raise ValueError('MPC ablations require MPC; --sampling-c3 replaces the planner')
    if mode != 'full' and getattr(args, 'floating_fingertip', False):
        raise ValueError('MPC ablations use the Franka OSC environment')
    if mode == 'a8':
        # Keep the same dynamics/layout as the baseline. Disabling the existing
        # two-stage switch via its CLI flag would also change the physics model.
        args.two_stage_object_cost = True
    if mode != 'full' and not getattr(args, 'ablation_variant', ''):
        args.ablation_variant = mode
    return args


def selection_parser():
    return add_ablation_args(argparse.ArgumentParser(add_help=False))
