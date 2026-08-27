# P1R Task-Aligned Monotone Risk Result

Canonical：
`run://worldsim_v65/WS-V65-P1R-TASK-ALIGNED-RISK-01/20260827T075500Z__task-risk-s0-r1`。

Verdict：`positive_train_only_task_risk_signal`。

- fixed-route density：`0.00299581→0.00284602`（20→19/6676，`-5%` relative）；
- worst-tail：`0.01643968→0.01559935`；
- scene lower/equal/higher：`1/15/0`；
- non-route emitted risk relative change：`-0.00665%`；
- shuffled query density：`0.00299581`；
- wall：`12.47s`；peak GPU：`0.137GiB`。

这是小 denominator 上的弱机制证据，只解锁 frozen T0 的 fresh P2。Attention、actor、admission 均未解锁。
