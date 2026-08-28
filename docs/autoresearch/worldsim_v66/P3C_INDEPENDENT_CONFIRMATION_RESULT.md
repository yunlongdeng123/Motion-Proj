# P3C Independent Local Geometry Confirmation 结果

Task：`WS-V66-P3C-INDEPENDENT-LOCAL-GEOMETRY-CONFIRM-01`

Canonical：`run://worldsim_v66/WS-V66-P3C-INDEPENDENT-LOCAL-GEOMETRY-CONFIRM-01/20260828T091611Z__independent-local-geometry-confirm-s0-r1`

冻结P3L checkpoint与normalization在scene-disjoint V65 P2V consumed cohort上exact-once评估；没有refit、threshold、
feature/seed/architecture sweep或第二confirmation。Actor existence authority保持关闭。

| 指标 | 结果 |
|---|---:|
| source units / eligible actor-unit | 72 / 581 |
| conflict / clean | 307 / 274 |
| confirmation AUROC / AUPRC | 0.761644 / 0.767165 |
| deterministic baseline AUROC / AUPRC | 0.500000 / 0.528399 |
| improvement over deterministic | +0.261644 / +0.238766 |
| q0 AUROC / AUPRC | 0.699517 / 0.738965 |
| improvement over q0 | +0.062127 / +0.028200 |
| above-chance evaluable scenes | 6 / 6 |
| wall / peak GPU / RSS | 10.77s / 0.02359GiB / 0.93375GiB |

4/4 gates通过。`V66-F01`由“两级证书”恢复关闭：deterministic证据继续保护Actor existence，冻结learned head只为
local geometry REPAIR/ABSTAIN排序。原deterministic certificate本身仍不覆盖natural local conflict；不将恢复解释为
Actor删除权、fresh V6.6 generalization、真实geometry repair、planning、policy、closed-loop、RL或safety。
