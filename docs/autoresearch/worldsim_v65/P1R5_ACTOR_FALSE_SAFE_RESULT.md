# V6.5 P1R5 Actor False-Safe Result

- task: `WS-V65-P1R5-ACTOR-FALSE-SAFE-01`
- canonical run: `run://worldsim_v65/WS-V65-P1R5-ACTOR-FALSE-SAFE-01/20260827T123100Z__actor-false-safe-s0-r1`
- verdict: `no_clear_train_only_actor_trajectory_forecast`
- formal V6.5 selection read: `false`

Frozen A0/A1 models were scored on GPU while P2V archive/preprocess I/O ran concurrently. Reducing 302 evaluation Actor
tokens to 24 trajectory units exposed an aggregation failure: snapshot-vs-target Spearman was 0.626087, below the
preregistered 0.70 gate, although lowest-risk 40% target cost fell 26.07% and passed its gate. A1 descriptive Spearman
was lower at 0.488696.

The positive-disagreement false-safe monitor failed all gates: gap Spearman -0.054402, positive-gap AUROC 0.522222, and
lowest-monitor 40% realized gap increased 73.40%. There were 9 positive-gap trajectories and 6 zero-monitor trajectories,
so the failure is not a zero-support artifact. No threshold or learned monitor is attempted.

