"""Fixed-template FA/L-BFGS worker with native or compiled rollout gradients."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import mujoco
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--executor', choices=['mujoco', 'comfree'], required=True)
    parser.add_argument('--forge', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=5.4)
    parser.add_argument('--case-name', default='dexforge')
    parser.add_argument('--output-base', type=Path, help='Write new cases outside the source experiment')
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--multistarts', type=int)
    parser.add_argument('--evaluation-backend', choices=['native', 'compiled'], default='native', help='compiled uses generated ComFree equations and analytic FA control gradients')
    parser.add_argument('--compiled-cache', default='')
    parser.add_argument('--compiled-threads', type=int, default=8)
    parser.add_argument('--viewer', action='store_true')
    args = parser.parse_args()
    p = args.experiment.resolve()
    forge = args.forge.resolve()
    sys.path[:0] = [str(forge / 'forceaware'), str(forge)]
    from forceaware.config import load_retarget_config, config_dict
    from forceaware.runtime import setup_runtime
    from forceaware.time_grid import SolveRequest
    cfg = load_retarget_config(p / 'sequence', p / 'forceaware.yaml', hand='wuji_right', contactaware_dir=p / 'contactaware')
    if args.iterations is not None:
        cfg = replace(cfg, optimizer=replace(cfg.optimizer, n_iter=args.iterations))
    if args.multistarts is not None:
        cfg = replace(cfg, optimizer=replace(cfg.optimizer, multistart_count=args.multistarts))
    case = (args.output_base.resolve() if args.output_base is not None else p) / args.executor / args.case_name
    case.mkdir(parents=True, exist_ok=True)
    print(f'[case] {case}', flush=True)
    if any(((case / n).exists() for n in ('rollout.npz', 'partial.npz', 'admm_windows.jsonl', 'live_state.npz'))):
        raise FileExistsError(f'{case} contains a previous run; use a new --case-name')
    grid = replace(cfg.simulator.time_grid, exec_dt=0.002 if args.executor == 'mujoco' else 0.01)
    cfg = replace(cfg, sequence=replace(cfg.sequence, output_dir_name=str(case)), simulator=replace(cfg.simulator, time_grid=grid, exec_backend=args.executor))
    setup_runtime(cfg)
    from forceaware import workflow as workflow
    from forceaware.executor import ActionExecution, ExecutorRuntime
    from examples.retarget.fa.solver import ADMMSettings, optimizer_class
    native = mujoco.MjModel.from_xml_path(str(p / 'sequence/scene/wuji_right/sugar_box_mesh.xml'))
    settings = ADMMSettings(optimize_contacts=False, evaluation_backend=args.evaluation_backend,
        compiled_cache=args.compiled_cache, compiled_threads=args.compiled_threads)
    optimizer_base = optimizer_class(native, settings)
    snapshots = []
    snapshot_indices = []
    cursor = {'window': 0}
    class FixedInitialOptimizer(optimizer_base):

        def solve(self, *a, **kw):
            solve_started = time.perf_counter()
            kw['allow_init'] = False
            plan = super().solve(*a, **kw)
            from forceaware.mpc_loss import MpcObjective
            selected = plan.diag['selected']
            snap = self.objective.snapshot(selected)
            snapshots.append(snap)
            snapshot_indices.append(cursor['window'])
            MpcObjective.evaluate(self.objective, self.control, allow_init=False)
            fa = float(self.loss.total.numpy()[selected])
            mask = snap['mask'] > 0
            primal = np.linalg.norm(snap['c_before_surface_polish'] - snap['z'], axis=-1)[mask] * 1000
            shift = np.linalg.norm(snap['z'] - snap['template'], axis=-1)[mask] * 1000
            record = dict(window=cursor['window'], FA_loss=fa, selection_merit=plan.loss, primal_mean_mm=float(primal.mean()) if len(primal) else 0.0, primal_max_mm=float(primal.max()) if len(primal) else 0.0, object_contact_shift_mean_mm=float(shift.mean()) if len(shift) else 0.0, object_contact_shift_max_mm=float(shift.max()) if len(shift) else 0.0, seconds=time.perf_counter() - solve_started, solver_wall_seconds=plan.seconds, solver=self.last_solver_diag)
            with (case / 'admm_windows.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            cursor['window'] += 1
            return replace(plan, seconds=time.perf_counter() - solve_started)
    workflow.WindowOptimizer = FixedInitialOptimizer
    initial_run = workflow._initial_run

    def initial(*a, **kw):
        result = initial_run(*a, **kw)
        result = replace(result, qvel=np.zeros_like(result.qvel))
        if visualizer is not None:
            visualizer.update(result.qpos, result.qvel, 0.0)
        return result
    workflow._initial_run = initial
    visualizer = None
    if args.viewer:
        from examples.retarget.common.viewer_bridge import ViewerProcess
        visualizer = ViewerProcess(p, case)
    workflow.make_solve_request = lambda config: SolveRequest.create(config.simulator.time_grid, duration_seconds=args.duration)

    class MujocoExecutor:

        def __init__(self, runtime, index, cfg):
            self.model = runtime.cpu_model
            self.data = mujoco.MjData(self.model)
            self.index = index
            self.elapsed = 0.0
            self.substeps = round(cfg.simulator.time_grid.mpc_dt / self.model.opt.timestep)

        def execute_action_sequence(self, qpos, qvel, *, actions):
            d = self.data
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            qs = []
            vs = []
            for action in actions:
                d.ctrl[self.index.hand_ctrl] = action
                d.qfrc_applied[:] = 0.0
                for _ in range(self.substeps):
                    mujoco.mj_step(self.model, d)
                mujoco.mj_forward(self.model, d)
                qs.append(d.qpos.copy())
                vs.append(d.qvel.copy())
            self.elapsed += len(actions) * self.substeps * self.model.opt.timestep
            return ActionExecution(qs[-1], vs[-1], np.array(qs), np.array(vs))
    if args.executor == 'mujoco':

        def load_executor(config, comfree_warp, **kw):
            cpu = mujoco.MjModel.from_xml_path(str(p / 'sequence/scene/wuji_right/sugar_box_mesh.xml'))
            cpu.opt.timestep = 0.002
            return ExecutorRuntime(cpu, None)
        workflow._load_executor_runtime = load_executor
        workflow.build_action_executor = lambda runtime, index, cfg: MujocoExecutor(runtime, index, cfg)

    def save(config, recorder, *, compiled, cpu_model, index, sequence, **kw):
        from forceaware.output import _object_qpos
        arrays = recorder.arrays()
        arrays['object_reference_qpos'] = _object_qpos(sequence.object_pose_xyzw, np.r_[arrays['qpos_frames'][0], arrays['target_frames']])
        np.savez_compressed(case / 'rollout.npz', **arrays)
        if snapshots:
            np.savez_compressed(case / 'contact_admm.npz', window_indices=np.array(snapshot_indices), **{k: np.stack([s[k] for s in snapshots]) for k in snapshots[0]})
        (case / 'config.json').write_text(json.dumps(config_dict(config), default=str, indent=2))
        (case / 'audit.json').write_text(json.dumps(dict(executor=args.executor, closed_loop=True, initial_pose_refinement=False, initial_velocity='zero', duration=args.duration, original_FA_optimizer=True, rendering=False), indent=2))
        audit = json.loads((case / 'audit.json').read_text())
        audit.update(original_FA_optimizer=False, base_optimizer='independent batched L-BFGS hand block', variant='lbfgs', ADMM=vars(settings), force_QP=False, template_query_targets_optimized=False, contact_mask_unchanged=True, force_gates=False, FA_cost_formulas_and_weights_unchanged=True, MANO_role='fixed_template', final_ranking='pure_FA')
        audit.update(auxiliary_snapshot_windows=len(snapshots))
        (case / 'audit.json').write_text(json.dumps(audit, indent=2))
        return case
    workflow.save_outputs = save
    print_step = workflow._print_step

    def progress(window, total, plan, *, recorder):
        print_step(window, total, plan, recorder=recorder)
        if visualizer is not None:
            visualizer.update(recorder.qpos[-1], recorder.qvel[-1], (window + 1) * 0.03, snapshots[-1] if snapshots else None)
        if window % 10 == 0:
            temporary = case / 'partial_tmp.npz'
            np.savez_compressed(temporary, **recorder.arrays())
            temporary.replace(case / 'partial.npz')
            if snapshots:
                temporary = case / 'partial_contact_tmp.npz'
                np.savez_compressed(temporary, window_indices=np.array(snapshot_indices), **{k: np.stack([s[k] for s in snapshots]) for k in snapshots[0]})
                temporary.replace(case / 'partial_contact_admm.npz')
    workflow._print_step = progress
    started = time.time()
    try:
        workflow.rolling_mpc_retarget(cfg, config_path=p / 'forceaware.yaml')
    finally:
        if visualizer is not None:
            visualizer.close()
    (case / 'timing.json').write_text(json.dumps(dict(wall_seconds=time.time() - started)))
if __name__ == '__main__':
    main()
