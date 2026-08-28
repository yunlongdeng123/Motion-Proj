# P2N Natural Actor-owned Local Geometry Conflict 诊断

Task：`WS-V66-P2N-NATURAL-ACTOR-CONFLICT-DIAGNOSTIC-01`

Canonical：`run://worldsim_v66/WS-V66-P2N-NATURAL-ACTOR-CONFLICT-DIAGNOSTIC-01/20260828T090228Z__natural-actor-conflict-s0-r1`

六个P10X consumed scenes与P1-D场景不重叠。每个Actor-unit只有在sensor hit、current/swept envelope存在且至少有4个
grounded boundary points时进入分母；任一boundary point被target观测为FREE即标记local geometry conflict。该label不判断
Actor existence。

| 指标 | 结果 |
|---|---:|
| eligible actor-unit | 891 |
| conflict / clean | 498 / 393 |
| conflict prevalence | 0.558923 |
| q0 AUROC / AUPRC | 0.543745 / 0.612874 |
| q0-rate Spearman | 0.267650 |
| deterministic certificate recall | 0 |
| deterministic AUROC / AUPRC | 0.5 / 0.558923 |
| clean false conflict | 0 |

`V66-F01`记录natural local geometry transfer ceiling。Actor existence证据是强保护证据，但不能证明Actor-owned每个局部
primitive都可靠。下一路线必须分层：保留Actor，同时只对local geometry做REPAIR/ABSTAIN。
