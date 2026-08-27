# P2V Fresh Visited-State Transfer Result

## Canonical result

`run://worldsim_v65/WS-V65-P2V-VISITED-STATE-TRANSFER-01/20260827T142000Z__fresh-visited-transfer-s0-r2`

Verdict: `supported_fresh_trajectory_visited_state_qagg`.

## Formal metrics

The frozen cohort contained 72 trajectory units across six previously unused scenes. Nine units did not meet the observable minimum of 16 visited boundary samples, leaving 63 eligible units, 8,862 visited samples, 1,055 hidden-FREE outcomes, and 57 unsafe trajectories.

| metric | value | frozen gate | result |
| --- | ---: | ---: | --- |
| Qmean versus visited-error-rate Spearman | 0.633963 | >=0.60 | pass |
| unsafe AUROC | 0.994152 | >=0.85 | pass |
| unsafe AUPRC | 0.999390 | descriptive | - |
| all-trajectory actual cost | 0.102965 | - | - |
| selected-40% actual cost | 0.0522594 | - | - |
| selected actual-cost reduction | 49.25% | >=40% | pass |
| scene lower/equal/higher | 5/1/0 | at least 5 lower | pass |

All four preregistered gates passed. Scene 0219 retained only three eligible trajectories and had zero realized cost, so it was equal rather than lower; the other five scenes all improved at matched per-scene coverage.

The run materialized the compact cache once, used 0.0236 GiB peak allocated GPU memory and 1.143 GiB peak RSS, and completed in 9.175 seconds.

## Entry disclosure

The preceding r1 entry (`20260827T141500Z__fresh-visited-transfer-s0-r1`) is preserved as `V65-F15`. It loaded the first formal input/target unit but failed on a dimension-specific squeeze before any Qagg, target value, metric, gate, verdict, or compact cache was emitted. The r2 recovery changed only the tensor view from `.squeeze(1)` to `.reshape(-1)` and kept the candidate, cohort, sampling, target, seed, and gates fixed.

## Supported claim boundary

The result supports a deterministic task-conditioned representation statement:

> Given a future Ego trajectory, arithmetic-mean aggregation of frozen state risk over the world states that trajectory will visit transfers its ranking of realized visited-state reliability to six fresh scenes.

It does not establish Actor-state reliability, learned conditional risk, admission, independent calibration, candidate-trajectory planning utility, or safety. The R7 monotone calibration form remains eligible only for a separately frozen unused calibration cohort.
