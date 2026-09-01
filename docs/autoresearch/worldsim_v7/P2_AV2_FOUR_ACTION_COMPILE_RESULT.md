# WorldSim V7 P2 — AV2 Four-Action Physical Compile

## Verdict

- canonical run: `run://worldsim_v7/WS-V7-P2-AV2-FOUR-ACTION-COMPILE-01/20260902T125000Z__four-action-s0-r2`
- verdict: `supported_zero_shot_av2_four_action_physical_compile`
- cohort: frozen AV2 Sensor val, 20 quantitative + 10 qualitative logs
- gates: quantitative `13/13`, qualitative `13/13`

P2 compiles single-frame query primitives against a build-only Actor-local surface using `KEEP`, `PROJECT`, `COMPLETE`,
and `UNKNOWN`. The first held-out frame is the query; later held-out frames are target-only and cannot influence actions.

## Quantitative result

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Target recall | 0.533573 | 0.625960 | +0.092387 |
| Surface precision | 0.795861 | 0.976750 | +0.180889 |
| Symmetric Chamfer (m) | 0.254369 | 0.168063 | 0.660706x |

The 20 logs contain 433 compiled Actors, including 147 hazardous Actors. Action counts are 143,633 `KEEP`, 20,313
`PROJECT`, 15,004 `COMPLETE`, and 47,376 `UNKNOWN`. Clean KEEP is 0.957694, observed-free ghost PROJECT is
0.993398, unsupported-artifact UNKNOWN is 1.0, completion independent-target support is 0.986004, and overall artifact
repair-or-reject is 0.997799. Actor identity/trajectory/size retention and hazard-label retention are both 1.0.

## Frozen qualitative confirmation

All 10 frozen qualitative logs are retained, covering 201 Actors including 83 hazardous Actors. Recall changes from 0.502263
to 0.611304, precision from 0.781215 to 0.970182, and Chamfer from 0.267530 m to 0.173570 m (0.648788x).
All 13 gates pass; no result-based scene or Actor selection occurs.

## Failure and recovery boundary

r1 (`20260902T120000Z__four-action-s0-r1`) stopped after first-log metadata because the P2 YAML omitted the frozen P1
`hazard` block (`V7-F06`). No point association, surface, action, target metric, or gate was evaluated. r2 restored exactly
that existing block without changing the cohort, split, action thresholds, gates, seed, or claim boundary.

## Claim boundary and resources

The before/after target metrics are real zero-shot AV2 LiDAR geometry evidence. Ghost, duplicate-shell, temporal-flicker,
and supported-hole inputs are paired synthetic interventions and validate only the deterministic compiler contract. This is not
evidence for natural-artifact generalization, learned completion, appearance repair, planning, closed-loop behavior, or safety.

Runtime is 56.228 seconds with 0.06924 GiB peak GPU memory and 1.1939 GiB peak RSS. The run occupies 3.2 MiB.
