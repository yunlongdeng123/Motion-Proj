# P1 Signal Atlas Result

Canonical run：
`run://worldsim_v65/WS-V65-P1-CONDITION-SIGNAL-ATLAS-01/20260827T074500Z__signal-atlas-s0-r1`。

结论：`WS-V65-H-P1-001` rejected。

- q0 AUROC/AUPRC：`0.871759/0.407081`；
- T0 AUROC/AUPRC：`0.871576/0.405639`；
- fixed-route density：`0.00299581→0.00314560`（风险相对 `+5%`）；
- scene lower/equal/higher：`1/13/2`；
- true-vs-shuffled trajectory AUROC：`+0.009591`。

网络确实使用了 trajectory，但把 task query 注入 task-agnostic hidden-FREE 目标没有形成增量。T1 attention、
seed/capacity sweep 均关闭。P1R 是新的输出语义假设，不改写本结论。
