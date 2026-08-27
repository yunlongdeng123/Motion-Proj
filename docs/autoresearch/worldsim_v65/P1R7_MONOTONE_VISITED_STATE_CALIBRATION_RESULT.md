# P1R7 Monotone Visited-State Calibration Result

## Canonical run

`run://worldsim_v65/WS-V65-P1R7-MONOTONE-VISITED-STATE-CALIBRATION-01/20260827T125000Z__monotone-calibration-s0-r1`

Verdict: `positive_train_only_monotone_visited_state_calibration`.

## Result

The fit used 108 legacy train trajectories and the single read used 58 nested-evaluation trajectories. The fitted map is:

`calibrated_error = sigmoid(1.703977 * logit(Qmean) - 0.479222)`.

| metric | raw Qmean | monotone calibrated | delta |
| --- | ---: | ---: | ---: |
| MSE | 0.0273778 | 0.00210441 | -92.31% |
| MAE | 0.156639 | 0.0355369 | -77.31% |
| five-bin absolute calibration error | 0.156639 | 0.0177814 | -88.65% |
| Spearman | 0.751487 | 0.751487 | 0 |
| unsafe AUROC | 0.978261 | 0.978261 | 0 |
| unsafe AUPRC | 0.994327 | 0.994327 | 0 |
| scene MSE lower/equal/higher | - | 15/0/0 | all scenes improve |
| selected-40% units | 23 | same 23 | exact identity |
| selected actual cost | 0.0381365 | 0.0381365 | unchanged |

All six preregistered gates passed. The result resolves the R4 tension: a scalar monotone map can improve expected visited-error calibration without sacrificing the decision ranking, whereas the unconstrained 25D R4 head changed ordering and worsened selected cost.

Runtime was 2.319 seconds on one RTX 3090; peak allocated GPU memory was 0.00195 GiB and peak RSS was 0.954 GiB. Native hidden was not loaded, and the run overlapped the fresh public-archive scan.

## Claim boundary and next use

This is a legacy train-only mechanism result, not formal V6.5 calibration. The fitted map must not be reported as independently calibrated and is not added post hoc to the already frozen P2V candidate. If Qmean first transfers on the P2V fresh ranking read, the map is eligible only as a frozen starting form for a separately selected, previously unused calibration cohort.

References: Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html); Zhang et al., [Instance-Wise Monotonic Calibration by Constrained Transformation](https://proceedings.mlr.press/v286/zhang25c.html).
