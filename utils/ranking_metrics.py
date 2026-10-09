"""Ranking-strategy metrics from Reviewer 5 Q1, TRO rebuttal p. 7.

J = eta * gamma**completion_steps * I(success)
A_RS-b = (J_RS - J_b) / (abs(J_b) + epsilon)

Steps are CONTROL steps elapsed from the saved trigger state, not absolute
episode indices, simulator substeps, seconds, or planner wall-clock time.
Failed rollouts have zero reward. No pose-error monotonicity is assumed.
``CSO`` is the rebuttal's ``optimal`` baseline; it need not be globally optimal.

Counterfactual rollouts must restore the same simulator AND controller state,
goal, physical parameters, and random state, with the same rollout budget.
This module evaluates recorded outcomes; it does not run those rollouts.

JSON input for the CLI (each state_id must be unique):
    {
      "paired_rollouts": [
        {"state_id": "trial_0_trigger_10", "rollouts": {
          "RS":   {"success": true,  "steps": 12},
          "CSO":  {"success": true,  "steps": 20},
          "near": {"success": false, "steps": null}
        }}
      ],
      "switch_counts": [2, 3]
    }

switch_counts is optional: one number of actual RS-triggered contact switches
per ORIGINAL trial, including zeros. Do not repeat it for each trigger state
or count nearest-sample index jitter as physical contact switches. For separate
rotation/flipping tables, supply one input file per task.

Usage:
    python -m utils.ranking_metrics outcomes.json --latex table.tex

Defaults eta=1 and epsilon=1e-12 specify values omitted in the rebuttal.
Keep them fixed across comparisons. With gamma=0.9 per 20 ms control step,
long-rollout rewards become very small; epsilon then affects advantage sizes.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def discounted_terminal_reward(success, completion_steps=None, *, gamma=0.9, eta=1.0):
    """Return J for a scalar or broadcastable arrays of rollout outcomes.

    ``completion_steps=None`` is allowed only when all outcomes are failures.
    Successful completion at the initial state has T_s=0. Finite, nonnegative
    integer steps are required; a failure's supplied step count is ignored in J.
    """
    gamma, eta = float(gamma), float(eta)
    if not np.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("gamma must be finite and in (0, 1]")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    outcomes = np.asarray(success)
    if outcomes.dtype.kind not in "biuf" or not np.all((outcomes == 0) | (outcomes == 1)):
        raise ValueError("success must contain booleans or 0/1 values")
    if completion_steps is None:
        if np.any(outcomes):
            raise ValueError("successful rollouts require completion_steps")
        completion_steps = 0
    steps = np.asarray(completion_steps, dtype=np.float64)
    if not np.all(np.isfinite(steps) & (steps >= 0) & (steps == np.floor(steps))):
        raise ValueError("completion_steps must be finite nonnegative integers")
    outcomes, steps = np.broadcast_arrays(outcomes.astype(bool), steps)
    reward = np.where(outcomes, eta * np.power(gamma, steps), 0.0)
    return float(reward) if reward.ndim == 0 else reward


def relative_advantage(rs_reward, baseline_reward, *, epsilon=1e-12):
    """Return the paired relative advantage, including zero-reward baselines."""
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    rs, baseline = np.broadcast_arrays(
        np.asarray(rs_reward, dtype=np.float64),
        np.asarray(baseline_reward, dtype=np.float64),
    )
    if not np.all(np.isfinite(rs) & np.isfinite(baseline) & (rs >= 0) & (baseline >= 0)):
        raise ValueError("terminal rewards must be finite and nonnegative")
    advantage = (rs - baseline) / (np.abs(baseline) + epsilon)
    if not np.all(np.isfinite(advantage)):
        raise ValueError("advantage overflow; increase epsilon or reduce eta")
    return float(advantage) if advantage.ndim == 0 else advantage


def _mean_std(values):
    """Sample standard deviation is undefined for fewer than two samples."""
    values = np.asarray(values, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)) if values.size else None,
        "std": float(np.std(values, ddof=1)) if values.size > 1 else None,
    }


def summarize_paired_rollouts(paired_rollouts, *, switch_counts=None,
                             gamma=0.9, eta=1.0, epsilon=1e-12):
    """Summarize matched trigger states, with separate per-trial switch stats.

Each record contains a unique state_id and RS/CSO/near rollout outcomes.
Missing branches are errors, never failures or silently unpaired comparisons.
Advantages are computed per state BEFORE averaging; positive fractions use
strict A > 0 over all paired states, including double-failure ties.
    """
    # Validate parameters even for empty data.
    discounted_terminal_reward(False, gamma=gamma, eta=eta)
    relative_advantage(0.0, 0.0, epsilon=epsilon)
    rewards = {strategy: [] for strategy in ("RS", "CSO", "near")}
    seen = set()
    for record in paired_rollouts:
        state_id = record["state_id"]
        if state_id in seen:
            raise ValueError(f"duplicate state_id: {state_id}")
        seen.add(state_id)
        for strategy in rewards:
            outcome = record["rollouts"][strategy]
            rewards[strategy].append(discounted_terminal_reward(
                outcome["success"], outcome.get("steps"), gamma=gamma, eta=eta,
            ))
    summary = {
        "gamma": float(gamma), "eta": float(eta), "epsilon": float(epsilon),
        "n_states": len(seen), "comparisons": {}, "switches_per_trial": None,
    }
    for baseline in ("CSO", "near"):
        advantages = relative_advantage(rewards["RS"], rewards[baseline], epsilon=epsilon)
        summary["comparisons"][baseline] = {
            **_mean_std(advantages),
            "positive_fraction": float(np.mean(advantages > 0)) if advantages.size else None,
        }
    if switch_counts is not None:
        counts = np.asarray(switch_counts, dtype=np.float64)
        if counts.ndim != 1 or not np.all(
                np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts))):
            raise ValueError("switch_counts must be a 1D sequence of nonnegative integers")
        summary["switches_per_trial"] = _mean_std(counts)
    return summary


def latex_table(summary):
    """Render a booktabs table; missing measurements are explicitly pending."""
    def number(value):
        formatted = f'{value:.3g}'
        if 'e' in formatted:
            mantissa, exponent = formatted.split('e')
            power = f'10^{{{int(exponent)}}}'
            return power if mantissa == '1' else f'{mantissa} \\times {power}'
        return formatted

    def mean_std(stats):
        if stats is None or stats["mean"] is None:
            return r"\textit{Pending}"
        mean = number(stats["mean"])
        std = "--" if stats["std"] is None else number(stats["std"])
        return f"${mean} \\pm {std}$"

    rows = []
    for baseline, label in (("CSO", r"$A_{\mathrm{RS}-\mathrm{CSO}}$"),
                            ("near", r"$A_{\mathrm{RS}-\mathrm{near}}$")):
        stats = summary["comparisons"][baseline]
        fraction = stats["positive_fraction"]
        positive = r"\textit{Pending}" if fraction is None else f"{100 * fraction:.1f}\\%"
        rows.append(f'{label} & {stats["n"]} & {mean_std(stats)} & {positive} ' + r"\\")
    switches = summary["switches_per_trial"]
    switch_n = "--" if switches is None else str(switches["n"])
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Paired ranking-strategy advantages (mean $\pm$ sample std),",
        f'with $\\gamma={summary["gamma"]:g}$, $\\eta={summary["eta"]:g}$,',
        f'$\\epsilon={number(summary["epsilon"])}$. $N$ counts trigger states for advantages',
        r"and original trials for switches. Pending denotes unmeasured values.}",
        r"\label{tab:ranking_counterfactual}", r"\small",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Statistic & $N$ & Mean $\pm$ std & $A>0$ \\",
        r"\midrule", *rows, r"\midrule",
        f"RS switches/trial & {switch_n} & {mean_std(switches)} & -- " + r"\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("outcomes", type=Path, help="JSON with paired_rollouts and optional switch_counts")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--latex", type=Path, help="Write a table for inclusion in experiment.tex")
    args = parser.parse_args()
    data = json.loads(args.outcomes.read_text(encoding="utf-8"))
    summary = summarize_paired_rollouts(
        data["paired_rollouts"], switch_counts=data.get("switch_counts"),
        gamma=args.gamma, eta=args.eta, epsilon=args.epsilon,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    if args.latex:
        args.latex.parent.mkdir(parents=True, exist_ok=True)
        args.latex.write_text(latex_table(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
