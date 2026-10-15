"""Paired execution settings with one canonical startup equilibrium."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from examples.mpc.franka.bigrasp.bigrasp import BimanualPandaGrasper, build_argparser, PANDA_XML_PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-noslip', type=int, required=True)
    experiment, rest = parser.parse_known_args()
    args = build_argparser().parse_args(rest)
    args.scene_output.parent.mkdir(parents=True, exist_ok=True)
    assets = args.scene_output.parent/'assets'
    if not assets.exists() and not assets.is_symlink():
        assets.symlink_to(PANDA_XML_PATH.parent/'assets', target_is_directory=True)
    # Select the same initial resting pose before changing the execution solver.
    args.contact_noslip_iterations = 3
    demo = BimanualPandaGrasper(args)
    try:
        initial_pose = np.r_[demo.get_object_pose()[0], demo.get_object_pose()[1]]
        initial_qpos = demo.data.qpos.copy()
        demo.model.opt.noslip_iterations = experiment.experiment_noslip
        try:
            result = demo.run()
        except Exception as error:
            pos, quat, _ = demo.get_object_pose()
            failure = dict(success=False, runtime_error=str(error), error_type=type(error).__name__,
                simulation_seconds=float(demo.data.time), final_object_pos=pos, final_object_quat=quat,
                contact_solver_settings=dict(impratio=float(demo.model.opt.impratio),
                    noslip_iterations=int(demo.model.opt.noslip_iterations),
                    fingertip_object_friction=demo.fingertip_object_friction,
                    table_object_friction=demo.table_object_friction),
                experiment=dict(initial_pose=initial_pose, initial_qpos=initial_qpos,
                    startup_noslip=3, execution_noslip=experiment.experiment_noslip))
            args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostics_output.write_text(json.dumps(failure, default=lambda x: np.asarray(x).tolist()))
            raise
        result['experiment'] = dict(initial_pose=initial_pose, initial_qpos=initial_qpos,
            startup_noslip=3, execution_noslip=experiment.experiment_noslip,
            control_steps=args.control_steps, lift_height=args.lift_height)
        args.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_output.write_text(json.dumps(result, default=lambda x: np.asarray(x).tolist()))
    finally:
        demo.close()
