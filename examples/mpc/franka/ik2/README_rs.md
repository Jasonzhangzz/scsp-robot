# Ranking strategy: Q, advantage and CDR

Run from the repository root in the existing simulation environment:

```bash
python test_rs.py --obj foam_brick --rotation --trial_num 10 \
  --solver-backend portable --ranking-metric-gamma 0.999 \
  --rs-output outputs/rs_rotation

python test_rs.py --obj bunny2 --flipping --trial_num 10 \
  --solver-backend portable --ranking-metric-gamma 0.999 \
  --rs-output outputs/rs_flipping
```

`--rotation` starts upright; `--flipping` starts tilted. The legacy
`--target-type ground-rotation` means flipping. The default physics is Isaac;
`--physics-backend mjwarp` selects the existing MJWarp executor. The script uses
one synchronous world and fixed SCM parameters. It inherits the existing object,
contact, solver, initialization and task-limit flags, including
`--max_rollout_length` (default 5000). Ordinary MPC entry points are unchanged.

A successful source trajectory contributes two kinds of timestamps:

- A new RS contact-switch commitment, merging consecutive commands pursuing
  the same destination.
- A displaced CSO optimum while RS retains the nearby contact.

The two branches must have distinct patches (over 3 cm geodesic distance, with
Euclidean fallback), a reachable/observed nearby contact and a feasible ranked
switch candidate. Initial ranking and free-space travel do not create samples.
`--rs-best-change-distance` defaults to 0.01 m in object-local coordinates to
filter mesh jitter. `--rs-max-per-kind 10` uniformly samples up to ten timestamps
of each kind per successful trial; set it to `0` to keep all candidates.
`--rs-sample-seed` makes subsampling reproducible. Aggregates describe these
sampled states; `by_event` keeps the two sampling strata separate.

For each sample the successful source suffix supplies the factual outcome.
The opposite branch rebuilds the simulator and planner from the same episode
seed and replays the full prefix, preserving controller history, solver warm
starts and random-number history. Observations, physical velocities/contact
geometry, issued commands and RS decisions must match before the intervention;
a mismatch aborts instead of producing a spurious paired value.
`--rs-match-atol` defaults to `1e-7`, with zero relative tolerance.

At the sampled timestamp only the switch/stay commitment is overridden, before
CPO constructs its target and MPC solves. Every following timestamp uses normal
SCSP replanning. Both branches retain the original absolute task deadline, so
the remaining budget is `max_rollout_length - sampled_step`. An aborted rollout
is an error, not a task failure.

The reported quantities are:

```text
R = gamma ** remaining_control_steps * success
Q_switch, Q_stay = mean(R over rollouts for that decision)
advantage_rs = Q_RS - Q_opposite
advantage_switch = Q_switch - Q_stay
CDR = max(Q_switch, Q_stay) - Q_RS
```

The chosen action counts as the first remaining step. Each step is 20 ms;
wall-clock planner time is excluded. Failure has zero reward. CDR uses no reward
scale or relative-advantage denominator. `--ranking-metric-gamma` inherits the
existing default `0.9`; the examples choose `0.999` because `0.9` per step makes
rewards extremely small for long trajectories. Fix gamma across comparisons.
`--rs-repeats` defaults to one outcome per decision. Extra repetitions replay
both branches with the same seed; deterministic repetitions are not independent
noise samples or evidence of confidence in an expectation.

Outputs under `--rs-output`:

- `config.json`: experimental flags, reward definition and replay method.
- `trials.jsonl`: every source trial, success, physical relocation count and
  separately labelled RS decision-trigger counts.
- `execution_contact_trace_trialN.jsonl`: full original-source measured contacts
  and executed MPC targets, expressed in the object's local frame.
- `execution_contact_switches_trialN.json`: offline-detected physical relocation
  timestamps, old/new patches and explicit detector parameters.
- `states.jsonl`: sampled timestamps, remaining budgets, switch/stay outcomes,
  Q estimates, signed advantages and CDR.
- `summary.json`: sampled CDR/advantage mean and sample standard deviation,
  separate event strata, completed physical switches per simulated second,
  their fraction per ranking evaluation and switches per successful source trial.
  `rs_decision_trigger_frequency_hz` separately describes decision triggers;
  it is not the physical switching frequency.

Frequency is computed from original successful trials, including zero-switch
trials, never from replay branches or capped samples. No successful source or
no eligible samples produces `null` measurements, rather than a measured zero.
Files are refreshed for each invocation; partial trial/state records are flushed
and the summary is updated after each completed source trial. Use distinct
output directories for tasks, objects or parameter settings.

`--rs-resume` retains completed state pairs after interruption. It checks the
saved configuration, regenerates the source trajectory, and verifies sampled
decisions and factual outcomes before reusing those pairs. A mismatch aborts.
Independent object/seed cases can run in separate Isaac Gym processes; keep
each case synchronous with `--num-envs 1` and give it a separate output directory.

Offline switching statistics can be refreshed from already-cached original
rollouts, without starting a simulator:

```bash
python -m utils.switching_frequency outputs/rs_cached_success_20261013
```

This writes `switching_frequency.json` and `.csv` with overall, per-object and
per-task statistics. Only `mpc_contact_switches` or full execution contact
traces supply counts. `rs_switch_events`, best/selected sample-index changes,
and MPC warm-start resets must never be used as physical switch counts.
`switches_per_rollout` is total completed physical switches divided by the
number of measured original rollouts (with sample standard deviation); Hz uses
total switches divided by total simulated execution seconds. Recorded execution
time is preferred; synchronous records can fall back to steps times the policy
interval. The ranking fraction uses total switches divided by total ranking
evaluations. The default includes successful, non-aborted source rollouts;
`--include-failures` also includes non-aborted failures.

The case manifest restricts inputs to the scheduled source trajectories, so
archived retries and counterfactual CDR branches are excluded. Missing switch
counts and uncached trajectories are explicitly pending and never treated as
zero-switch trials. Rerun the command as further source caches become available;
CDR branches need not finish before a completed source can enter this statistic.

A completed physical switch requires an actually issued far MPC contact target,
loss of measured fingertip-object contact, then stable measured contact on a
distinct patch at that target. Default thresholds are 3 cm patch separation,
2 cm arrival tolerance, at least two detached control observations and three
stable new-contact observations. Initial contact, uninterrupted sliding,
same-patch recontact, target jitter and travel that never reaches a new contact
do not count. All comparisons use object-local coordinates, so object motion
does not itself create switches. Insufficient execution traces produce pending
measurements; existing RS-trigger counts cannot reconstruct physical switches.

For a short execution check (which is not a complete CDR experiment):

```bash
python test_rs.py --obj foam_brick --rotation --trial_num 1 \
  --max_rollout_length 2 --sample_num 8 --solver-backend portable \
  --rs-output outputs/rs_smoke
```
