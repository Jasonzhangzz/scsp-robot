"""Run matched IsaacGym switch/stay rollouts for ranking Value differences.

Examples (run from the repository root, using the simulation environment):
  python test_rs.py --obj foam_brick --rotation --trial_num 10 \
      --rs-samples-per-trajectory 5 --rs-output outputs/rs_rotation
  python test_rs.py --obj foam_brick --flipping --trial_num 1 \
      --headless --sync-planner --no-adaptive --solver-backend acados \
      --rs-event-stage flip --rs-output outputs/rs_flip

The original successful suffix supplies the factual Q sample. The opposite
branch reconstructs the complete prefix and changes ONLY the sampled decision;
subsequent SCSP replanning remains enabled. Extra repetitions replay both
branches. With deterministic physics these are identical, not independent
noise samples. T is remaining 20 ms control steps, including the chosen action.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from utils.ranking_cdr import (DecisionSampler, assert_matched, continuation_outcome,
                               evaluate_state, select_events, summarize_cdr,
                               select_decision_points, source_decision_statistics)


class MatchedRolloutBus:
    """Observe/verify the unforced prefix without altering the physics protocol."""
    def __init__(self, bus, sampler, *, reference=None):
        self.bus, self.sampler, self.reference = bus, sampler, reference
        self.observations, self.commands, self.summaries = [], [], []
        self.rs_decision_hook = sampler

    def wait_obs(self, timeout=180.):
        obs = self.bus.wait_obs(timeout=timeout)
        if obs is None or obs.get('cmd') == 'stop':
            return obs
        self.sampler.step = int(obs['seq'])
        self.sampler.sim_time = float(obs['sim_time'])
        contact = obs.get('actual_contact')
        distance = (contact.get('dist', obs.get('contact_distance', np.inf))
                    if contact else np.inf)
        self.sampler.in_physical_contact = bool(
            obs.get('if_contact', False) or (contact is not None and float(distance) <= 0))
        # Includes configuration, velocities and measured contact Jacobians,
        # rather than only the ten-dimensional policy pose.
        fields = ('seq', 'sim_time', 'policy_q', 'full_q', 'object_linear_velocity',
                  'object_angular_velocity', 'fingertip_linear_velocity',
                  'executed_action', 'phi_vec', 'jac_mat', 'jac_mat_env', 'env_phi',
                  'actual_contact', 'if_contact', 'contact_distance',
                  'osc_upright_guard_active')
        record = copy.deepcopy({k: obs[k] for k in fields if k in obs})
        env = getattr(self.bus, 'env', None)
        if env is not None:
            record['joint_position'] = np.asarray(env.get_current_joint_position()).copy()
            record['joint_velocity'] = np.asarray(env.get_current_joint_velocity()).copy()
        self.observations.append(record)
        if self.reference is not None and self.sampler.step <= self.sampler.target:
            index = len(self.observations)-1
            if index >= len(self.reference.observations):
                raise RuntimeError('Replay observation exceeds the original prefix')
            assert_matched(record, self.reference.observations[index],
                           atol=self.sampler.atol, path=f'observation[{index}]')
        return obs

    def publish_cmd(self, command):
        if not command.get('cmd'):
            # Exclude planner/submission wall-clock diagnostics from comparison.
            fields = ('kind', 'seq', 'via', 'action', 'policy_dt', 'press',
                      'release_pressure', 'preserve_upright', 'path_blocked')
            record = copy.deepcopy({k: command[k] for k in fields if k in command})
            self.commands.append(record)
            if self.reference is not None and self.sampler.step < self.sampler.target:
                index = len(self.commands)-1
                assert_matched(record, self.reference.commands[index],
                               atol=self.sampler.atol, path=f'command[{index}]')
        self.bus.publish_cmd(command)

    def record_summary(self, summary):
        self.summaries.append(copy.deepcopy(summary))

    def close(self):
        self.bus.close()


class MJWarpSingleBus:
    """Existing MJWarp executor with one synchronous world, no IPC delays."""
    def __init__(self, args):
        from examples.mpc.franka.runtime.mjwarp import MJWarpBatch
        self.trial = args.trial_start
        self.batch = MJWarpBatch(args, [self.trial])
        self.stopped = False

    @property
    def env(self):
        return self.batch.envs[0] if self.batch is not None else None

    def wait_obs(self, timeout=180.):
        return ({'cmd': 'stop', 'trial': self.trial} if self.stopped
                else self.batch.observation(0, self.trial))

    def publish_cmd(self, command):
        if command.get('cmd') in ('end_trial', 'stop'):
            self.stopped = True
        else:
            self.batch.command(0, command)
            self.batch.step([0])

    def close(self):
        self.stopped = True
        self.batch = None


def run_trial(args, trial, sampler, *, reference=None):
    from examples.mpc.franka.runtime.rollout import rollout_output
    from examples.mpc.franka.runtime.mpc import run_mpc_planner
    # Rebuild planners and execution state, including their histories/warm
    # starts. This avoids incomplete pose-only checkpoints of a contact sim.
    random.seed(100 + trial)
    np.random.seed(100 + trial)
    import torch
    torch.manual_seed(100 + trial)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(100 + trial)
    values = dict(vars(args), trial_start=trial, trial_num=1, num_envs=1,
                  headless=True, rollout=True, async_planner=False,
                  sync_planner=True, online_scm=False, scm_record_data=False,
                  ranking_metrics_jsonl='', rollout_summary_json='',
                  rollout_trace_jsonl='', rollout_planner_snapshots='',
                  rollout_cache_dir='', trial_output='')
    local = argparse.Namespace(**values)
    if args.physics_backend == 'isaac':
        from examples.mpc.franka.runtime.isaac import SynchronousMPCBus
        physics = SynchronousMPCBus(local)
    else:
        physics = MJWarpSingleBus(local)
    bus = MatchedRolloutBus(physics, sampler, reference=reference)
    try:
        with rollout_output(args.rollout_verbose):
            run_mpc_planner(bus, local)
    finally:
        bus.close()
    if len(bus.summaries) != 1 or bus.summaries[0].get('aborted'):
        raise RuntimeError('Expected one complete, non-aborted rollout')
    if sampler.target is not None and not sampler.applied:
        raise RuntimeError('Rollout did not reach the sampled decision')
    return bus


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, default=json_value, allow_nan=False)+'\n',
                    encoding='utf-8')
    temp.replace(path)


def csv_source_candidates(paths, *, obj, task, seed):
    """CSV summaries identify successful seeds; they are not simulator snapshots."""
    candidates = {}
    for path in paths:
        with Path(path).open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                # Historical Isaac five-trial CSV has only the flipping task.
                row_task = row.get('task', 'flipping')
                if (row.get('object') != obj or row_task != task
                        or row.get('success', '').lower() not in ('1', 'true')):
                    continue
                trial = int(row.get('seed', row.get('trial')))
                candidates.setdefault(trial, dict(trial=trial, csv=str(Path(path).resolve()), row=row))
    items = list(candidates.values())
    order = np.random.default_rng(seed).permutation(len(items))
    return [items[i] for i in order]


def write_statistics(output, result, states):
    write_json(output/'summary.json', result)
    metrics = result['table_vi']
    row = {key: metrics[key] for key in ('contact_switch_count', 'p_near', 'p_optimal',
           'f_near_hz', 'f_p_star_hz', 'p_advantage_positive')}
    row.update(advantage_mean=metrics['advantage']['mean'],
               advantage_std=metrics['advantage']['std'], n_states=len(states))
    with (output/'table_vi.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    fields = ('state_id', 'object', 'trial', 'step', 'sim_time', 'decision',
              'in_physical_contact', 'in_transit',
              'original_contact_world', 'opposite_contact_world', 'remaining_budget',
              'value_original', 'value_perturbed', 'advantage_rs', 'cdr')
    with (output/'states.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(states)


def run(args, *, trial_runner=run_trial):
    if not 0 < args.ranking_metric_gamma < 1:
        raise ValueError('ranking-metric-gamma must be in (0, 1) for CDR')
    if not np.isclose(float(getattr(args, 'ranking_metric_eta', 1.0)), 1.0):
        raise ValueError('CDR uses a unit success reward; ranking-metric-eta must be 1')
    if getattr(args, 'physics_backend', 'isaac') != 'isaac':
        raise ValueError('test_rs is intentionally IsaacGym-only; use test_mpc_isaac defaults')
    if getattr(args, 'solver_backend', 'acados') != 'acados':
        raise ValueError('test_rs requires the developer acados planner configuration')
    if args.trial_start < 0 or args.trial_num <= 0 or args.rs_repeats <= 0:
        raise ValueError('trial-start must be nonnegative; trial_num and rs-repeats positive')
    if args.rs_samples_per_trajectory <= 0 or args.rs_successful_trajectories <= 0:
        raise ValueError('Sample and successful-trajectory counts must be positive')
    if (args.rs_max_per_kind < 0 or not np.isfinite(args.rs_best_change_distance)
            or args.rs_best_change_distance <= 0 or not np.isfinite(args.rs_match_atol)
            or args.rs_match_atol < 0):
        raise ValueError('Invalid sampling or replay tolerances')
    if args.num_envs != 1 or args.async_planner or args.floating_fingertip or args.sampling_c3:
        raise ValueError('CDR requires one synchronous Franka SCSP world')
    if args.online_scm or args.scm_record_data:
        raise ValueError('Use fixed SCM parameters for matched CDR evaluation')
    if args.cso_guidance != 'cso' or args.cpo_variant != 'full':
        raise ValueError('CDR evaluates the normal SCSP ranking and full CPO')
    output = Path(args.rs_output)
    output.mkdir(parents=True, exist_ok=True)
    resume = bool(getattr(args, 'rs_resume', False))
    saved_states = {}
    if resume and (output/'states.jsonl').exists():
        previous_config = json.loads((output/'config.json').read_text())
        for key, value in vars(args).items():
            if key != 'rs_resume' and previous_config.get(key) != value:
                raise ValueError(f'Resume configuration differs: {key}')
        for line in (output/'states.jsonl').read_text().splitlines():
            state = json.loads(line)
            if state['state_id'] in saved_states:
                raise ValueError('Duplicate saved state during resume')
            saved_states[state['state_id']] = state
    write_json(output/'config.json', dict(vars(args), reward_origin='sampled_decision',
        reward_definition='gamma**remaining_control_steps * success',
        replay_method='complete prefix, verified observations, commands and ranking decisions',
        switching_definition='completed physical relocation, derived offline from execution contact trace',
        sampling_definition='uniform without replacement over eligible physical-contact decisions, excluding transit; includes A=0',
        csv_source_definition='successful seed selection only; reconstruct and re-evaluate a synchronous factual source before sampling',
        rs_trigger_definition='nearby-to-guide RS decision onset; not a physical contact switch'))
    states, trials = [], []
    paths = getattr(args, 'rs_source_csv', [])
    candidates = (csv_source_candidates(paths, obj=args.obj,
        task='rotation' if args.target_type == 'rotation' else 'flipping',
        seed=args.rs_sample_seed) if paths else
        [dict(trial=trial) for trial in range(args.trial_start, args.trial_start+args.trial_num)])
    if not candidates:
        raise RuntimeError(f'No successful CSV source for {args.obj}/{args.target_type}')
    selected_sources = 0
    # Resume retains completed pairs, after checking the regenerated source.
    with (output/'states.jsonl').open('a' if resume else 'w', encoding='utf-8') as state_file, \
         (output/'trials.jsonl').open('w', encoding='utf-8') as trial_file:
        for candidate in candidates:
            trial = candidate['trial']
            print(f'[rs] trial={trial}: collecting SCSP trajectory', flush=True)
            sampler = DecisionSampler(best_change_distance=args.rs_best_change_distance)
            if getattr(args, 'rs_source_cache', ''):
                from utils.rollout_cache import load_source_cache
                baseline, sampler = load_source_cache(args.rs_source_cache, args, trial)
                print(f'[rs] using cached original source: trial={trial}', flush=True)
            else:
                baseline = trial_runner(args, trial, sampler)
            summary = dict(baseline.summaries[0], rs_switch_events=sampler.switch_count,
                           rs_evaluations=sampler.evaluations, rs_candidate_events=len(sampler.events))
            summary.update(source_decision_statistics(sampler.decisions, summary,
                change_distance=args.rs_best_change_distance))
            summary['rs_source_provenance'] = candidate
            events = []
            summary['rs_source_selected'] = False
            if summary['success']:
                stage = args.rs_event_stage
                if args.rs_sampling == 'decisions':
                    try:
                        events = select_decision_points(sampler.decisions,
                            count=args.rs_samples_per_trajectory,
                            seed=args.rs_sample_seed+trial, stage=stage)
                    except ValueError as error:
                        summary['rs_source_excluded_reason'] = str(error)
                        print(f'[rs] trial={trial}: {error}; trying next source', flush=True)
                else:
                    events = select_events([e for e in sampler.events
                        if (stage == 'all' or e.get('stage') == stage)
                        and e.get('in_physical_contact', False) and not e.get('in_transit', False)],
                        max_per_kind=args.rs_max_per_kind, seed=args.rs_sample_seed+trial)
                summary['rs_source_selected'] = bool(events)
            if (summary['rs_source_selected'] and baseline.observations
                    and not getattr(args, 'rs_source_cache', '')):
                from utils.rollout_cache import save_source_cache
                save_source_cache(output/'source'/f'trial{trial}', baseline, sampler, args)
            from utils.rollout_cache import best_point_changes
            summary.update(best_point_changes(sampler.decisions, summary))
            with (output/f'ranking_decisions_trial{trial}.jsonl').open('w') as ranking_file:
                for decision in sampler.decisions:
                    ranking_file.write(json.dumps(decision, default=json_value, allow_nan=False)+'\n')
            if baseline.observations:
                # Persist existing execution observations/commands once the
                # original source ends; never use hypothetical CDR branches.
                from utils.executed_contact_switches import execution_trace, completed_contact_switches
                trace = execution_trace(baseline.observations, baseline.commands)
                trace_path = output/f'execution_contact_trace_trial{trial}.jsonl'
                with trace_path.open('w', encoding='utf-8') as trace_file:
                    for observation in trace:
                        trace_file.write(json.dumps(observation, allow_nan=False)+'\n')
                try:
                    physical = completed_contact_switches(trace)
                    summary['mpc_contact_switches'] = physical['mpc_contact_switches']
                    summary['mpc_contact_switch_events'] = physical['events']
                    summary['mpc_contact_switch_definition'] = physical['definition']
                    summary['mpc_contact_switch_parameters'] = physical['parameters']
                    write_json(output/f'execution_contact_switches_trial{trial}.json', physical)
                except ValueError as error:
                    summary['mpc_contact_switches'] = None
                    summary['mpc_contact_switches_unavailable_reason'] = str(error)
            trials.append(summary)
            trial_file.write(json.dumps(summary, default=json_value, allow_nan=False)+'\n')
            trial_file.flush()
            if summary['rs_source_selected']:
                selected_sources += 1
                valid_ids = {f'trial_{trial}_step_{e["step"]}' for e in events}
                if any(s['trial'] == trial and key not in valid_ids
                       for key, s in saved_states.items()):
                    raise RuntimeError('Regenerated source has different sampled timestamps')
                for event in events:
                    event = dict(event, state_id=f'trial_{trial}_step_{event["step"]}', trial=trial,
                                 remaining_budget=args.max_rollout_length-event['step'])
                    rs = event['decision']
                    opposite = 'stay' if rs == 'switch' else 'switch'
                    rollouts = {'switch': [], 'stay': []}
                    rollouts[rs].append(continuation_outcome(summary, event['step'],
                                                           gamma=args.ranking_metric_gamma))
                    saved = saved_states.get(event['state_id'])
                    if saved is not None:
                        assert_matched({k: saved[k] for k in event}, event,
                                       atol=args.rs_match_atol, path='resumed decision')
                        assert_matched(saved['rollouts'][rs][0], rollouts[rs][0],
                                       atol=args.rs_match_atol, path='resumed source outcome')
                        states.append(saved)
                        print(f'[rs] reused completed {event["state_id"]}: '
                              f'Q={saved["q"]}, CDR={saved["cdr"]:.6g}', flush=True)
                        continue
                    for repeat in range(args.rs_repeats):
                        # The first factual sample is the already-observed suffix.
                        for branch in ((opposite,) if repeat == 0 else (rs, opposite)):
                            print(f'[rs] {event["state_id"]} {event["event_kind"]}: '
                                  f'{branch}, sample={repeat+1}', flush=True)
                            replay_sampler = DecisionSampler(
                                best_change_distance=args.rs_best_change_distance,
                                target=event['step'], expected=sampler.decisions,
                                forced=branch if branch != rs else None,
                                match_atol=args.rs_match_atol)
                            replay = trial_runner(args, trial, replay_sampler, reference=baseline)
                            rollouts[branch].append(continuation_outcome(replay.summaries[0],
                                event['step'], gamma=args.ranking_metric_gamma))
                    state = evaluate_state(event, rollouts, gamma=args.ranking_metric_gamma)
                    state['object'] = str(getattr(args, 'obj', ''))
                    state['original_contact_world'] = event.get(
                        'near_world' if rs == 'stay' else 'switch_world')
                    state['opposite_contact_world'] = event.get(
                        'switch_world' if rs == 'stay' else 'near_world')
                    states.append(state)
                    state_file.write(json.dumps(state, default=json_value, allow_nan=False)+'\n')
                    state_file.flush()
                    write_statistics(output, summarize_cdr(states, trials,
                        gamma=args.ranking_metric_gamma), states)
                    print(f'[rs] {event["state_id"]}: Q={state["q"]}, '
                          f'A_RS={state["advantage_rs"]:.6g}, CDR={state["cdr"]:.6g}', flush=True)
            elif any(s['trial'] == trial for s in saved_states.values()):
                raise RuntimeError('Previously successful source failed during resume')
            result = summarize_cdr(states, trials, gamma=args.ranking_metric_gamma)
            write_statistics(output, result, states)
            if selected_sources >= args.rs_successful_trajectories:
                break
    if selected_sources < args.rs_successful_trajectories:
        raise RuntimeError(f'Only {selected_sources} usable successful sources; '
                           f'need {args.rs_successful_trajectories}. See trials.jsonl.')
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    from planning.cpu_runtime import configure_compute_threads
    configure_compute_threads()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--physics-backend', choices=('isaac', 'mjwarp'), default='isaac')
    backend, _ = pre.parse_known_args()
    # --help can be inspected outside the Isaac Python environment.
    if backend.physics_backend == 'mjwarp' or any(a in sys.argv for a in ('-h', '--help')):
        os.environ['SCSP_PLANNER_ONLY'] = '1'
    else:
        os.environ.pop('SCSP_PLANNER_ONLY', None)
        from isaacgym import gymapi, gymtorch  # Must precede Torch imports.
    from examples.mpc.franka.runtime.mpc import _add_rollout_policy_args
    parser = _add_rollout_policy_args(argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument('--rs-output', default='outputs/ranking_cdr')
    parser.add_argument('--rs-source-cache', default='',
                        help='Trusted local source cache generated by test_mpc_isaac --rollout-cache-dir; original success is not re-estimated.')
    parser.add_argument('--rs-source-csv', nargs='+', default=[], metavar='CSV',
                        help='Randomly order successful seeds from these CSV summaries; reconstruct a matched synchronous source.')
    parser.add_argument('--rs-samples-per-trajectory', type=int, default=5)
    parser.add_argument('--rs-successful-trajectories', type=int, default=1)
    parser.add_argument('--rs-sampling', choices=('decisions', 'events'), default='decisions',
                        help='Default: uniformly sample all eligible decision timestamps; events retains the older event-stratified analysis.')
    parser.add_argument('--rs-resume', action='store_true',
                        help='Reuse completed state pairs after verifying source/configuration.')
    parser.add_argument('--rs-max-per-kind', type=int, default=10,
                        help='Max sampled states per event kind per successful trial; 0 keeps all.')
    parser.add_argument('--rs-repeats', type=int, default=1,
                        help='Rollouts per decision; deterministic repetitions do not add noise.')
    parser.add_argument('--rs-sample-seed', type=int, default=0)
    parser.add_argument('--rs-best-change-distance', type=float, default=.01,
                        help='Minimum local best-point displacement (m) to reject mesh jitter.')
    parser.add_argument('--rs-match-atol', type=float, default=1e-7,
                        help='Absolute prefix-matching tolerance; mismatches abort evaluation.')
    parser.add_argument('--rs-event-stage', choices=('all', 'flip'), default='all',
                        help='Sample only ranking decisions from this object-cost stage.')
    parser.set_defaults(headless=True, async_planner=False, sync_planner=True,
                        num_envs=1, rollout=True, physics_backend='isaac',
                        solver_backend='acados', online_scm=False,
                        contact_model='soft-scm', two_stage_object_cost=True,
                        ranking_metric_gamma=.99, ranking_metric_eta=1.0,
                        no_adaptive=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
