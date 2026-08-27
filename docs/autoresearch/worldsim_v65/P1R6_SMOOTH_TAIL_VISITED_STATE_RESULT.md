# P1R6 Smooth-Tail Visited-State Result

## Canonical run

`run://worldsim_v65/WS-V65-P1R6-SMOOTH-TAIL-VISITED-STATE-01/20260827T124500Z__smooth-tail-s0-r1`

Verdict: `no_clear_train_only_smooth_tail_visited_state_increment`.

## Result

The read reused exactly the 108 train / 58 nested-evaluation trajectory units from R4. The evaluation set contained 6,651 visited samples and 754 hidden-FREE outcomes.

| metric | Qmean | Qsoft-tail | delta / gate |
| --- | ---: | ---: | --- |
| Spearman | 0.751487 | 0.708230 | -0.043256, fail (`>=-0.02`) |
| unsafe AUROC | 0.978261 | 1.000000 | +0.021739, pass |
| unsafe AUPRC | 0.994327 | 1.000000 | descriptive |
| MSE | 0.027378 | 0.183788 | tail emphasis is not calibrated to the rate target |
| selected-40% actual cost | 0.038137 | 0.048535 | +27.27%, fail |
| reduction versus all-unit cost | 62.98% | 52.88% | Qmean remains stronger |
| scene lower/equal/higher | - | 4/6/5 | fail |

Only one of four incremental gates passed. Qsoft-tail perfectly separated the binary any-error event on this small nested set, but overemphasized isolated high-q0 states and damaged both ordering of the continuous visited-error rate and the actual cost of the retained 40% trajectories.

Runtime was 0.562 seconds on one RTX 3090; peak allocated GPU memory was 0.00195 GiB and peak RSS was 0.719 GiB. Native hidden was not loaded. The run overlapped the continuing fresh archive scan.

## Literature response and decision

MIDAM shows that smoothed-max or attention pooling can be useful when trained against a bag-level AUC objective, but that is a new learned family; the already frozen R4 learned head damaged decision ranking. RAP couples risk-aware prediction to robust planning, rather than applying an after-the-fact pooling temperature. TAT aggregates multiple sampled trajectories and their history, not states inside a single fixed trajectory. These methods therefore motivate a future candidate-set planner interface, not a temperature rescue on the current data.

References: Zhu et al., [Provable Multi-instance Deep AUC Maximization with Stochastic Pooling](https://proceedings.mlr.press/v202/zhu23l.html); Nishimura et al., [RAP: Risk-Aware Prediction for Robust Planning](https://proceedings.mlr.press/v205/nishimura23a.html); Feng et al., [Resisting Stochastic Risks in Diffusion Planners with the Trajectory Aggregation Tree](https://proceedings.mlr.press/v235/feng24b.html).

Register `V65-F12`, close smooth-tail pooling without changing temperature or target, and retain deterministic Qmean as the sole frozen P2V candidate.
