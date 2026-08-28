# P6 Actor-preserving HARP Bake 结果

Task：`WS-V66-P6-HARP-BAKE-01`

Canonical：`run://worldsim_v66/WS-V66-P6-HARP-BAKE-01/20260828T092421Z__harp-bake-s0-r1`

| 指标 | 结果 |
|---|---:|
| units / unique Actors / Actor states | 72 / 127 / 581 |
| Actor primitives | 1,623,503 |
| Actor state retention | 1.0 |
| Actor metadata completeness | 1.0 |
| Actor removed | 0 |
| hidden-target fields | 0 |
| package files / bytes | 8 / 16,321,358 |
| wall / peak RSS | 15.57s / 0.82530GiB |

6/6 gates通过。runtime manifest固定`learned_model_loaded=false`、`hidden_target_loaded=false`、
`hazard_controls_actor_existence=false`，Actor/static与physical/appearance layers分离。local geometry保持连续
`RANK_REPAIR_OR_ABSTAIN`，无事后binary threshold，也未执行physical geometry mutation。

该结果只证明consumed legacy package capability；不证明真实geometry/appearance artifact已修复、fresh quality、
distribution improvement、planning、policy、closed-loop、RL或safety。
