"""Run one row of the Franka SCSP component-ablation table.

The table is ``Functional Analysis of SCSP Components`` in
``tro_revision/experiment.tex``. CSO is the contact selector in
``planning/mlqp_point.py``. CPO is the explicit MPC in
``planning/mpc_explicit.py``. Every variant keeps CPO's terminal objective,
the guidance update schedule, and the ranking strategy.

Variants
--------
full
    CSO minimizes the stable-pose objective. CPO uses lift/place potentials
    and object-center attraction.
cso-pose-only
    CSO minimizes the direct pose objective. CPO's terminal objective is
    unchanged.
cso-random
    The CSO prior is drawn uniformly from the feasible candidate set.
    Every candidate is still evaluated for the ranking strategy.
cso-nearest
    The CSO prior is the feasible candidate closest to the end effector.
    Every candidate is still evaluated for the ranking strategy.
cpo-no-object-center
    CPO sets the object-center weight to zero and keeps reference tracking
    together with the lift/place potentials.
cpo-direct
    CPO solves reference tracking, object-center attraction, control effort,
    and the unchanged terminal objective, without lift/place potentials.

``--task rotation`` is on-ground yaw. ``--task flipping`` is the side-start
ground flip. Remaining arguments are forwarded to ``run.py --planner mpc``.

Examples
--------
python examples/mpc/franka/ik2/run_ablation.py --variant full --task flipping \\
    --trial_num 5 --headless --ranking-metrics-jsonl ablations/full_flip.jsonl
python examples/mpc/franka/ik2/run_ablation.py --variant cso-pose-only --task rotation \\
    --obj stanford_bunny2 --trial-start 0 --trial_num 5 --headless
"""

from __future__ import annotations

import argparse
import os
import sys


VARIANTS = {
    "full": ("cso", "full"),
    "cso-pose-only": ("pose", "full"),
    "cso-random": ("random", "full"),
    "cso-nearest": ("nearest", "full"),
    "cpo-no-object-center": ("cso", "no_object_center"),
    "cpo-direct": ("cso", "direct"),
}

TASK_TARGET = {
    "rotation": "rotation",
    "flipping": "ground-rotation",
}

_DESCRIPTIONS = {
    "full": "CSO minimizes ell_cso; CPO keeps lift/place and object-center attraction.",
    "cso-pose-only": "CSO minimizes ell_pose; CPO terminal objective stays ell_cso.",
    "cso-random": "CSO prior is uniform on the feasible set; candidates stay available to RS.",
    "cso-nearest": "CSO prior is the nearest feasible sample; candidates stay available to RS.",
    "cpo-no-object-center": "CPO sets w_o=0 and keeps reference tracking plus lift/place.",
    "cpo-direct": "CPO drops lift/place potentials and solves reference, object-center, effort, and the terminal objective.",
}


def ablation_forward_argv(variant, task, rest):
    """Arguments for ``run.py`` that implement one table cell."""
    if variant not in VARIANTS:
        raise ValueError("unknown variant %r" % (variant,))
    if task not in TASK_TARGET:
        raise ValueError("unknown task %r" % (task,))
    guidance, cpo = VARIANTS[variant]
    flags = ["--planner", "mpc", "--target-type", TASK_TARGET[task]]
    if task == "rotation":
        flags.extend(["--no_random_init_tilt"])
    flags.extend(list(rest))
    # Variant flags come last so they win over forwarded overrides.
    flags.extend([
        "--planner", "mpc",
        "--cso-guidance", guidance,
        "--cpo-variant", cpo,
        "--ablation-variant", variant,
    ])
    return flags


def _repo_root():
    current = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.abspath(current)
    while os.path.basename(parent) != "scsp-robot":
        nxt = os.path.dirname(parent)
        if nxt == parent:
            raise RuntimeError("scsp-robot repo root not found from %s" % current)
        parent = nxt
    return parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=tuple(VARIANTS))
    parser.add_argument("--task", required=True, choices=tuple(TASK_TARGET))
    args, rest = parser.parse_known_args(argv)
    forwarded = ablation_forward_argv(args.variant, args.task, rest)
    print("[ablation] variant=%s task=%s target=%s" % (
        args.variant, args.task, TASK_TARGET[args.task]), flush=True)
    print("[ablation] %s" % _DESCRIPTIONS[args.variant], flush=True)
    print("[ablation] success, completion_time_s, and solve_dt_mean_ms are in trial_summary"
          " and --ranking-metrics-jsonl.", flush=True)
    root = _repo_root()
    sys.path = [p for p in sys.path if os.path.abspath(p or os.curdir) != root]
    sys.path.insert(0, root)
    sys.argv = [os.path.join(root, "examples", "mpc", "franka", "ik2", "run.py"), *forwarded]
    from examples.mpc.franka.ik2.run import main as run_main
    return run_main()


if __name__ == "__main__":
    main()
