"""Single-component counterfactuals against the repaired live controller."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.mpc.franka.trigrasp.trigrasp import build_argparser, AllegroTabletopTrigraspDemo
import planning.grasp_force_allocation as allocation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', choices=['clamp', 'fixed_load', 'stationary_base', 'noslip'], required=True)
    parser.add_argument('--online-pad-normal-weight', type=float, default=0., help=argparse.SUPPRESS)
    args, rest = parser.parse_known_args()
    demo = AllegroTabletopTrigraspDemo(build_argparser().parse_args(rest))
    pressure, allocator = allocation.normal_pressure_command, allocation.allocate_contact_wrench
    if args.ablation == 'clamp':
        def changed(jac, normals, measured, reference, *a, **kw):
            return pressure(jac, normals, measured, np.clip(reference, .25, .6), *a, **kw)
        allocation.normal_pressure_command = changed
    elif args.ablation == 'stationary_base':
        def changed(*a, **kw):
            return pressure(*a, **dict(kw, grasp_established=False))
        allocation.normal_pressure_command = changed
    elif args.ablation == 'noslip':
        demo.model.opt.noslip_iterations = 3
    else:
        def changed(points, normals, wrench, friction, **kw):
            closest = np.asarray(demo.online_contact_points_local)
            ids = np.linalg.norm(np.asarray(points)[:, None]-closest[None], axis=2).argmin(axis=1)
            summary = demo.get_fingertip_contact_summary()
            caps = np.array([summary[demo.active_finger_names[i]] for i in ids])
            return allocator(points, normals, wrench, friction,
                **dict(kw, minimum_normal=caps, normal_limits=caps))
        allocation.allocate_contact_wrench = changed
    try:
        print('Ablation:', args.ablation, flush=True)
        demo.run()
    finally:
        demo.close()


if __name__ == '__main__':
    main()
