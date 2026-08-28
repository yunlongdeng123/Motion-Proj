# P7R Exact-hit Sensor-supported Surface Repair 结果

Task：`WS-V66-P7R-SENSOR-SUPPORTED-ACTOR-REPAIR-01`

Canonical：`run://worldsim_v66/WS-V66-P7R-SENSOR-SUPPORTED-ACTOR-REPAIR-01/20260828T093710Z__sensor-surface-repair-s0-r1`

冻结L0 290 action states对23,580 actor-owned native boundary points执行same-Actor exact evidence-voxel hit保留。

| 指标 | 结果 | Gate |
|---|---:|---:|
| conflict point reduction | 0.847660 | >=0.50 PASS |
| overall boundary retention | 0.383588 | >=0.40 FAIL |
| clean boundary retention | 0.395715 | >=0.40 FAIL |
| Actor/shell/ID-track-trajectory retention | 1/1/1 | PASS |
| Actor removed / hazard proxy shift | 0 / 0 | PASS |
| scene yield | 1.0 | PASS |

7/9 gates通过但formal verdict=`rejected_consumed_legacy_sensor_surface_repair`。exact 0.2m evidence-voxel匹配过稀，
虽然把1,175个conflict points降到179，却把22,405个clean points降到8,866；不得降低0.40 retention gates。
