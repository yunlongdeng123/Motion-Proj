# P2 Fresh Trajectory-Conditioned Risk Result

Canonical run：

```text
run://worldsim_v65/WS-V65-P2-TRAJECTORY-CONDITIONED-RISK-01/20260827T093900Z__trajectory-selection-s0-r1
```

这是 frozen q0 与 frozen P1R monotone task risk 在 6 个 fresh scenes、72 cases 上唯一一次正式
representation-selection read。未 refit、未 sweep、未执行第二次 selection read。

| metric | q0 | task risk | result |
| --- | ---: | ---: | --- |
| fixed-route conflicts / eligible | 18 / 6975 | 18 / 6975 | reduction 0% |
| pooled fixed-route density | 0.00258065 | 0.00258065 | equal |
| worst-tail CVaR | 0.02935237 | 0.02935237 | equal |
| scene lower/equal/higher | - | 0/6/0 | support failed |
| non-route conflicts / selected | 4538 / 350093 | 4542 / 350093 | +0.0881% relative |

Gates：risk reduction=false、scene support=false；scene regression=true、non-route bound=true、coverage matched=true、
monotone semantics=true。Verdict=`rejected_fresh_trajectory_condition`。wall=`10.01s`、peak GPU=`0.0409GiB`。

P1R 在 legacy train-only denominator 上只减少 1/20 sampled conflicts；在 fresh 40% ranking boundary 没有
改变任何 route decision，六个 scenes 全部 equal。因此关闭当前 trajectory-only score-ranking family，不执行
attention/seed/capacity rescue。该结果不直接否定以 ego action 与 Actor future response 为监督的 actor-time/
action-outcome 模型；后者必须作为新 hypothesis、使用 train-only 诊断和未消费的新 selection cohort。
