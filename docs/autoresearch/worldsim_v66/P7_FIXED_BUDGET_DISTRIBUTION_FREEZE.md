# P7 Fixed-budget Hazard-preserving Distribution Audit Freeze

Task：`WS-V66-P7-HAZARD-PRESERVING-DISTRIBUTION-01`

本阶段是consumed legacy triage capability，不冒充真实physical repair。输入为P3C exact-once rows与P6 package；
local action budget在读本阶段指标前固定为Actor states的50%，不扫budget或score threshold。

Matched arms：N0不处理、Q0按q0排序、D0按deterministic reason、L0按冻结learned score、O0按label oracle上界。
所有臂保留完全相同的Actor/state和hazard attributes；action只表示`RANK_REPAIR_OR_ABSTAIN`候选，不删除Actor，不改
physical geometry。primary estimand是原始分母上的unhandled local-conflict exposure，不把ABSTAIN写成已验证repair。

L0 gates固定：conflict exposure reduction至少50%、Actor retention=1、removed=0、hazard proxy shift=0、emitted local
geometry fraction至少50%、六场景world yield=1。即使通过，也只授权固定预算triage；完整P7 physical distribution、
RL-ready、planning/policy/closed-loop/RL/safety仍不成立。
