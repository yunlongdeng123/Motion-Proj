# P3L Instance-evidence Local Geometry Head 结果

Task：`WS-V66-P3L-ACTOR-LOCAL-GEOMETRY-HEAD-01`

Canonical：`run://worldsim_v66/WS-V66-P3L-ACTOR-LOCAL-GEOMETRY-HEAD-01/20260828T091036Z__local-geometry-head-s0-r1`

固定8维instance-evidence summary与2x32 MLP只对Actor-owned local geometry conflict排序，不拥有Actor existence
删除权限。P10V consumed六场景用于训练，scene-disjoint P10X consumed六场景只执行一次selection；没有architecture、
feature、seed、threshold sweep，也没有hash/checksum/fingerprint。

| 指标 | 结果 |
|---|---:|
| train actor-unit / conflict / clean | 409 / 243 / 166 |
| selection actor-unit / conflict / clean | 891 / 498 / 393 |
| train AUROC / AUPRC | 0.999504 / 0.999663 |
| selection AUROC / AUPRC | 0.652365 / 0.692384 |
| deterministic baseline AUROC / AUPRC | 0.500000 / 0.558923 |
| selection improvement over deterministic | +0.152365 / +0.133461 |
| q0 AUROC / AUPRC | 0.543745 / 0.612874 |
| selection improvement over q0 | +0.108620 / +0.079510 |
| above-chance evaluable scenes | 6 / 6 |
| final weighted BCE | 0.053289 |
| wall / peak GPU / RSS | 13.07s / 0.02359GiB / 1.08285GiB |

4/4 gates通过，支持`consumed legacy selection local-geometry ranking`。该结果只使`V66-F01`进入恢复中：仍需
独立cohort exact-once确认，且不支持fresh V6.6 generalization、Actor删除、真实geometry repair、planning、policy、
closed-loop、RL或safety claim。
