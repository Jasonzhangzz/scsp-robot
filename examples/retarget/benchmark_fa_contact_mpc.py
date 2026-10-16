"""Reproducible wall-time benchmark at saved, active-contact MPC states."""
import argparse
import cProfile
from dataclasses import replace
import json
import os
from pathlib import Path
from time import perf_counter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, default=Path('outputs/fa_admm_solver_20261016'))
    p.add_argument('--forge', type=Path, default=Path('thirdparty/retarget_assets/DexForge'))
    p.add_argument('--states', type=Path)
    p.add_argument('--dense-steps', type=int, nargs='+', default=[90])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--multistarts', type=int, default=2)
    p.add_argument('--admm-outer', type=int, default=2)
    p.add_argument('--contact-samples', type=int, default=7)
    p.add_argument('--contact-force-iterations', type=int, default=2)
    p.add_argument('--contact-solver', choices=['shooting', 'batch-lm'], default='shooting')
    p.add_argument('--hand-solver',choices=['lbfgs','slsqp','acados'],default='lbfgs')
    p.add_argument('--hand-rti-iterations',type=int,default=1)
    p.add_argument('--no-reuse-forward',action='store_true')
    p.add_argument('--no-fuse-adjoint-zero',action='store_true')
    p.add_argument('--output', type=Path, default=Path('outputs/fa_contact_timing/benchmark.json'))
    p.add_argument('--profile', action='store_true')
    p.add_argument('--profile-graphs',action='store_true',help='CUDA-event timings and graph-node counts after solving')
    args = p.parse_args()
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('COMFREE_WARP_ROOT', str(args.forge.resolve()/'forceaware/third_party/comfree_warp'))
    os.environ.setdefault('HAND_ASSETS_ROOT', str(args.forge.resolve()/'assets'))
    import numpy as np
    import mujoco
    from examples.retarget.matrix_comfree import ComFreeEngine
    e = ComFreeEngine(args.experiment, args.forge)
    import warp as wp
    from forceaware.targets import make_window
    from examples.retarget.fa_contact_admm import ADMMSettings, optimizer_class
    from examples.retarget.comfree_contact_search import write_object_scene
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene = write_object_scene(e.cfg.sequence.xml, args.output.parent/'object_scene.xml')
    cfg = replace(e.cfg, optimizer=replace(e.cfg.optimizer, multistart_count=args.multistarts, n_iter=args.iterations),
                  simulator=replace(e.cfg.simulator, exec_backend='mujoco'))
    settings = ADMMSettings(contact_mode='comfree-force', object_scene=str(scene), outer=args.admm_outer,
                            contact_samples=args.contact_samples, contact_force_iterations=args.contact_force_iterations,
                            contact_solver=args.contact_solver,hand_solver=args.hand_solver,reuse_forward=not args.no_reuse_forward,
                            hand_rti_iterations=args.hand_rti_iterations,fuse_adjoint_zero=not args.no_fuse_adjoint_zero)
    native = mujoco.MjModel.from_xml_path(str(args.experiment/'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    solver = optimizer_class(native, settings)(e.model, e.device_model, e.index,
        guidance_count=e.sequence.contact.mask.shape[1], cfg=cfg)
    states = args.states or args.experiment/'mujoco/force_contact_mujoco_verified/rollout.npz'
    saved = np.load(states)
    timings = {}
    def instrument(owner, name, label):
        original = getattr(owner, name)
        def measured(*a, **kw):
            t = perf_counter()
            try:
                return original(*a, **kw)
            finally:
                timings[label] = timings.get(label, 0.) + perf_counter() - t
        setattr(owner, name, measured)
    instrument(solver, '_evaluate', 'hand_evaluate_s')
    instrument(solver.backend, '_prepare_window', 'prepare_s')
    instrument(solver.objective.contact_search, 'update', 'contact_search_s')
    instrument(solver.objective.contact_search.force_solver, 'solve', 'force_s')
    instrument(solver.objective, 'remember_contacts', 'remember_s')
    rows = []
    graph_profiles=[]
    profile = cProfile.Profile()
    for dense in args.dense_steps:
        targets = make_window(e.sequence, executed_steps=dense, horizon=cfg.optimizer.horizon,
            time_grid=cfg.simulator.time_grid, start_frame=0., end_frame=71.)
        window = dense//cfg.simulator.time_grid.action_substeps
        for repeat in range(args.repeats+1):
            # Identical input, regularizer and contact memory for each repetition.
            solver.objective.contact_memory = None
            solver.objective.force_memory = None
            if dense >= 4:
                solver.objective.regularizer._previous_previous_cpu = saved['ctrl_traj'][dense-4].copy()
            timings.clear()
            wp.synchronize()
            if args.profile and repeat == 1:
                profile.enable()
            t = perf_counter()
            plan = solver.solve(saved['qpos_traj'][dense], saved['qvel_traj'][dense], targets,
                warm=saved['raw_warm_history'][window], previous_action=saved['ctrl_traj'][max(dense-1, 0)],
                seed=window, start_time=dense*.01, allow_init=False, execution_steps=3)
            wp.synchronize()
            elapsed = perf_counter()-t
            profile.disable()
            row = dict(dense=dense, repeat=repeat, cold=repeat == 0, wall_s=elapsed,
                       plan_s=plan.seconds, loss=plan.loss, **timings, solver=solver.last_solver_diag)
            rows.append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(dict(arguments=vars(args), rows=rows,graph_profiles=graph_profiles), default=str, indent=2)+'\n')
        if args.profile_graphs:
            import re
            for key,graph in solver.graphs.items():
                start=wp.Event(e.device,enable_timing=True);end=wp.Event(e.device,enable_timing=True)
                gpu=[];launch=[];wall=[]
                for _ in range(5):
                    wp.synchronize_device(e.device);t=perf_counter();wp.record_event(start)
                    t_launch=perf_counter();wp.capture_launch(graph);launch.append((perf_counter()-t_launch)*1000)
                    wp.record_event(end);wp.synchronize_device(e.device)
                    wall.append((perf_counter()-t)*1000);gpu.append(wp.get_event_elapsed_time(start,end))
                dot=args.output.parent/(args.output.stem+'_graph_'+str(key)+'.dot')
                wp.capture_debug_dot_print(graph,str(dot),verbose=False)
                nodes=len(re.findall(r'\[.*shape=',dot.read_text()))
                graph_profiles.append(dict(dense=dense,graph=str(key),gpu_ms=gpu,launch_ms=launch,wall_ms=wall,nodes=nodes))
            args.output.write_text(json.dumps(dict(arguments=vars(args),rows=rows,graph_profiles=graph_profiles),default=str,indent=2)+'\n')
    if args.profile:
        import pstats
        profile.dump_stats(str(args.output.with_suffix('.prof')))
        pstats.Stats(profile).sort_stats('cumtime').print_stats(35)


if __name__ == '__main__':
    main()
