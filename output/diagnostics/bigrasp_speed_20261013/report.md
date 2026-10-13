# BigRasp approach speed comparison

Both trials use native acados, frozen initial contact IDs [0, 20], an 8-step horizon, 10 ms physics/control interval, 600 control cycles and `--fingertip-max-pitch 0.9`. The baseline is the preceding `bigrasp_grasp_recovery_20261013/soft_trial.json`; the new run is `fast_trial.json`. Loop wall times exclude initialization/code generation and can vary with machine load.

| Metric | Before | After |
| --- | ---: | ---: |
| Both tips within 30 mm of anchors, simulation seconds | 2.87 | 1.01 |
| Accumulated loop wall seconds at that point | 9.78 | 3.78 |
| First actual bilateral contact, simulation seconds | 1.46 | 1.19 |
| Median plan time, ms | 9.69 | 10.39 |
| Solver failures | 0 | 0 |
| Final height gain, mm | 3.23 | 43.66 |
| Final horizontal displacement, mm | 14.78 | 41.27 |

The shared predictor/executor adds action/interval velocity feedforward, capped at a 0.25 m/s reference norm. It fades over the last 2 mm of collision gap and is disabled on physical contact or an active grasp latch. OSC force limits and unilateral contact protection still apply. The new online MPC CLI default uses 8 steps instead of 20; this table isolates the velocity-reference change by holding horizon=8 for both trials.

The approach metric is proximity to the anchors, not confirmation of loaded contacts at both selected surface patches. Neither run completed the target 60 mm stable lift. The new run has more horizontal displacement; contact retention and horizontal stability remain unresolved. Experimental 20 ms control trials (`interval_trial.json`, `cadence_trial.json`) reduced loop count but degraded contact response, so the control interval remains 10 ms.

Validation: 56 native acados regression tests passed before the final mode-specific CLI default change; the CLI/executor suite was rerun after that change. Base environment: 35 passed, 1 skipped (native acados unavailable).
