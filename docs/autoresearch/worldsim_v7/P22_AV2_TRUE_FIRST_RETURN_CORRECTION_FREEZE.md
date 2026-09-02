# P22 Consumed AV2 Literal First-Return Correction Freeze

## Scope

P20 showed on consumed nuScenes source Actors that the historical target-nearest attribution is not literal ray termination and
understates new-early exposure by about sixfold. P22 applies the already frozen minimum-positive-depth operator to the already
consumed 20-log AV2 recovery cohort. It is a metric correction and cross-sensor diagnostic, not a new method or fresh test.

## Frozen inputs

- Cohort: `configs/worldsim_v7/av2_zero_shot_recovery_cohort_v1.json`, all 20 `fresh_external_confirmation` logs.
- Compiler: frozen P2 four-action compiler with `projection_output=observed_lidar_hit`.
- Legacy comparator: P15 raw target-nearest attribution, canonical
  `run://worldsim_v7/WS-V7-P15-FRESH-HAZARD-ACTION-ATTRIBUTION-01/20260903T043000Z__fresh-hazard-action-raw-s0-r1`.
- Literal operator: minimum positive projected depth inside the unchanged `0.20 m` beam tube.
- Hit/early tolerance: unchanged `0.20 m`.
- Run ID: `20260903T160000Z__consumed-av2-true-first-return-s0-r1`.

## Report

The single run reports all/hazard/clear proxy and literal new-early counts/rates, literal-to-proxy multiplier, new-hit counts,
and KEEP/PROJECT/COMPLETE attribution under literal first return. Every log and Actor remains in the denominator.

No threshold, model, checkpoint, action, tolerance, cohort, or policy is fitted or selected. The third 10-log AV2 cohort remains
unread. Results cannot be described as an independent confirmation, formal transfer guarantee, collision guarantee, planning
result, or road-safety certificate.

## Literature migration

The operator follows the raw-LiDAR first-occupied-return evaluation used by CVPR 2024 evidential occupancy and CVPR 2025
EvOcc. P22 migrates only the ray-depth measurement convention; it does not inherit their semantic occupancy or safety claims.
