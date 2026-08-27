# V6.5 P1R4 Trajectory-Visited-State Result

- task: `WS-V65-P1R4-TRAJECTORY-VISITED-STATE-01`
- canonical run: `run://worldsim_v65/WS-V65-P1R4-TRAJECTORY-VISITED-STATE-01/20260827T121500Z__visited-state-s0-r1`
- verdict: `positive_train_only_visited_state_object_q0_aggregation_only`
- formal V6.5 selection read: `false`

Changing the prediction object was successful. On 58 eligible nested-evaluation trajectory units, direct mean q0 risk over
the future 2 s Ego corridor achieved Spearman 0.751487 and unsafe-unit AUROC 0.978261. Selecting the lowest predicted-risk
40% reduced realized visited hidden-FREE cost from 0.103005 to 0.038137, a 62.98% reduction. All three prediction-object
viability gates passed.

| metric | Qagg | learned V1 | delta |
| --- | ---: | ---: | ---: |
| Spearman | 0.751487 | 0.635127 | -0.116360 |
| unsafe-unit AUROC | 0.978261 | 0.909420 | -0.068841 |
| MSE | 0.0273778 | 0.00346444 | -87.35% |
| lowest-risk 40% realized cost | 0.0381365 | 0.0577178 | +51.35% |
| scene lower/equal/higher | - | 2/7/6 | fail |
| real-minus-shuffled Spearman | - | +0.069117 | pass |

The context head improved point calibration but damaged the decision-relevant ordering. It is rejected without a seed,
loss, or capacity rescue. The retained mechanism is the deterministic, interpretable Qagg reduction from frozen state
risk to trajectory-level visited-state reliability. The run used 108 train and 58 evaluation units; 26 units were excluded
by the preregistered minimum footprint, and evaluation contained 6,651 visited states with 754 hidden-FREE outcomes.

