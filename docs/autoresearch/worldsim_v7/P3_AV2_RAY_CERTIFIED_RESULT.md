# WorldSim V7 P3 — Ray-Certified Hard Physical Evidence

## Verdict

- canonical run: `run://worldsim_v7/WS-V7-P3-AV2-HARD-EVIDENCE-01/20260902T143000Z__ray-certified-s0-r3`
- verdict: `supported_zero_shot_av2_hard_physical_evidence`
- gates: quantitative `8/8`, qualitative `8/8`, visual selection `3/3`
- coverage: 30/30 frozen AV2 logs; 8 main and 30 supplement point/surface/ray cases

## Quantitative evidence

| Evidence | Before | Ray-certified | Change |
|---|---:|---:|---:|
| Paired free-space violation | 1.000000 | 0.000000 | -1.000000 |
| Paired ghost components | 10,983 | 0 | -10,983 |
| Target LiDAR depth error (m) | 0.206731 | 0.154107 | -0.052624 |
| Ray termination consistency | 0.569444 | 0.643795 | +0.074351 |
| Canonical zero-level error (m) | 0.140003 | 0.058331 | 0.416643x |
| Target recall | 0.533573 | 0.595594 | +0.062021 |
| Collision-shell precision | 0.795861 | 0.974935 | +0.179074 |
| Symmetric Chamfer (m) | 0.254369 | 0.175268 | 0.689030x |

The quantitative cohort has 433 Actors, including 147 hazardous Actors. Mean temporal surface jitter is 0.042402 m.
Actor identity/lifecycle, trajectory, speed, acceleration, TTC, dimensions, hazard label, and hazard event counts do not change.

## Qualitative confirmation

The 10 frozen qualitative logs contain 201 Actors, including 83 hazardous Actors. Free-space violation and residual ghost
components both reach zero; depth error changes from 0.216718 to 0.162004 m, ray consistency from 0.538437 to 0.632761,
zero-level error from 0.145743 to 0.059975 m, and Chamfer from 0.267530 to 0.180854 m. All eight gates pass.

## Safety boundary and trade-off

The rejected nearest-canonical r2 attains 0.168063 m Chamfer but leaves 84.91% matched-ray free-space violations. The
ray-certified candidate attains 0.175268 m, paying about 7.2 mm Chamfer to remove those early terminations. A primitive may
PROJECT only to its direct observed LiDAR hit; without matched termination provenance it becomes `UNKNOWN` and emits no
collision surface.

Paired ghosts are synthetic contract interventions. Target-only depth, ray, zero-level, recall, precision, and Chamfer are real
zero-shot AV2 geometry evidence. Immutable Actor-state retention does not establish upstream perception or prediction accuracy,
planning quality, closed-loop behavior, or real-road safety.

Runtime is 99.011 seconds with 0.06924 GiB peak GPU memory and 1.2111 GiB peak RSS. The run occupies 5.1 MiB.
