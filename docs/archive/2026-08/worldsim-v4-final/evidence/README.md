# V4 canonical 轻量证据映射

| 归档目录 | 原始 canonical run | 用途 |
|---|---|---|
| `m1-validation-r200/` | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T204156Z__m1-validation-six-scene-confirmation-s0-r200` | M1 scene-disjoint validation 负结果、metrics、calibration、manifest |
| `m1-rejection-r201/` | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M1-EVIDENCE-FIELD-01/20260812T210150Z__m1-validation-rejection-audit-s0-r201` | M1 rejected 终态审计 |
| `m2-development-r212/` | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M2-REPAIR-ROUTER-01/20260812T233139Z__m2-development-router-selection-s0-r212` | router、threshold、baseline 与 tie priority 冻结 |
| `m2-validation-r222/` | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M2-REPAIR-ROUTER-01/20260813T064330Z__m2-validation-confirmation-s0-r222` | selective routing、coverage、geometry caveat、逐 request 决策 |
| `m3-validation-r238/` | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T075800Z__m3-validation-confirmation-s0-r238` | frozen M3 参数与 validation gate |
| `m3-test-r335/` | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T225624Z__m3-test-aggregate18-s0-r335` | 18-scene exact-once 聚合、scene/paired metrics、freeze |
| `closeout-r336/` | `/root/autodl-tmp/runs/worldsim_v4/WS-V4-M3-TEMPORAL-DELTA-01/20260813T225805Z__m1-m2-m3-test-closeout-audit-s0-r336` | M1/M2/M3 联合终局审计 |

`m1-validation-r200` 原始 run 约 119 MiB；本归档有意只保留根级 JSON 与 resolved config，不复制大体积诊断数组。其他目录也只作为轻量可复核副本，不替代原始 canonical run。
