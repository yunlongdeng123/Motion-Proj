# P2C Actor-Time Continuous Cost Result

Canonical run:

```text
run://worldsim_v65/WS-V65-P2C-ACTOR-TIME-COST-01/20260827T102000Z__actor-time-cost-s0-r2
```

Claim boundary: legacy train-only continuous Actor proximity-cost diagnostic. This run did not read V6.5
representation/admission selection, calibration, confirmation, or test data and makes no method, planning, or safety claim.

## Frozen comparison

| metric | snapshot A0 | Actor×time A1 | A1 delta |
| --- | ---: | ---: | ---: |
| Spearman | 0.872281 | 0.857392 | -0.014889 |
| MSE | 0.006247 | 0.008407 | -34.59% reduction (worse) |
| MAE | 0.059017 | 0.065177 | +0.006160 |
| matched-40% mean cost | 0.023950 | 0.021468 | -10.37% |

The Actor×time model retained a genuine temporal response: shuffling the time features within each evaluation scene reduced
its Spearman to 0.758575, a real-minus-shuffled gap of 0.098817. That sensitivity did not become useful incremental
generalization. Both evaluation scenes had higher selected mean cost under A1 (`+0.000747`, `+0.009296`). The frozen gates
passed only global selected-cost reduction and temporal-shuffle response; Spearman gain, MSE reduction, and 2/2 scene
support failed. Verdict: `no_clear_train_only_continuous_actor_time_cost`.

## Decision

`WS-V65-H-P2C-001` is rejected and the Actor/time family is closed. No distance-scale, seed, capacity, split, or threshold
rescue is authorized. P3 is not unlocked. Together with the fresh P2 result, this also leaves trajectory attention and
end-to-end task-conditioned representation locked. A subsequent admission feasibility audit may only keep the V6.4 risk
representation frozen and must use an unexposed admission split; it cannot reinterpret P2/P2C as positive.

The failed pre-training entry `20260827T101500Z__actor-time-cost-s0-r1` is preserved separately as `V65-F05`; it read no
evidence, trained no model, and produced no scientific result.
