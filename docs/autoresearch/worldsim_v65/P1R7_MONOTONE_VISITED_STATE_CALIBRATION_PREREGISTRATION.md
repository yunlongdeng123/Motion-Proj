# P1R7 Monotone Visited-State Calibration Preregistration

## Question

R4 showed that arithmetic-mean Qagg strongly ranks the reliability of world states visited by a future Ego trajectory, while its raw MSE is high. R4's unconstrained learned head improved MSE but damaged ranking and selected cost. This diagnostic asks whether one strictly monotone two-parameter map can estimate the expected visited hidden-FREE rate without changing the retained trajectory order.

## Frozen contract

- Reuse the R4 compact cache and its 108 train / 58 nested-evaluation trajectory units.
- Keep the 20-frame / 2.0-second horizon, 1.5 m footprint, minimum 16 visited samples, and continuous `visited_hidden_free_fraction` target.
- Input is only `Qmean=mean(q0 | visited by tau)`.
- Fit `sigmoid(a * logit(Qmean) + b)` with `a>0`, enforced by softplus. Optimize MSE for 800 full-batch Adam epochs at learning rate 0.02, seed 0.
- This is a legacy train-only calibration mechanism read. It neither consumes a formal V6.5 calibration split nor changes the frozen P2V candidate or selection ordering.

The constrained map follows the practical post-hoc scaling principle of Guo et al. and the ranking-preservation requirement formalized by constrained monotonic calibration. It intentionally excludes context, bins learned from evaluation data, isotonic knots, ensembles, and capacity or seed sweeps.

References: Guo et al., [On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html); Zhang et al., [Instance-Wise Monotonic Calibration by Constrained Transformation](https://proceedings.mlr.press/v286/zhang25c.html).

## Gates

- nested-evaluation MSE reduction at least 50%;
- five equal-count-bin absolute calibration error reduction at least 30%;
- per-scene MSE improves in at least 8 of 15 evaluation scenes;
- Spearman and unsafe AUROC each change by no less than `-1e-6`;
- lowest-risk 40% selected unit indices are exactly unchanged.

Failure closes this scalar calibration form without changing optimizer, loss, feature, bins, seed, or capacity. Success only retains it for a future independently frozen calibration cohort.
