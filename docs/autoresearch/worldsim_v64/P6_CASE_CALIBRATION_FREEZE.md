# P6 Case-Level Calibration Freeze

Date: 2026-08-26

Task/hypothesis: `WS-V64-P6-CALIBRATION-01 / WS-V64-H-P6C-001`.

The P5 U3 logistic risk model is immutable. Calibration may only choose a single
case-level selective coverage from `[0.05, 0.10, 0.20, 0.30, 0.40, 0.50]`; it may
not refit the model, change the score, or choose scene/stratum-specific policies.

The independent calibration denominator is the frozen 16-scene, four-stratum
cohort with 12 target cases per scene (`192` cases). Within each case, the policy
emits the lowest-U3-risk eligible native boundary voxels at the nominated coverage.
Case loss is one when selected hidden-FREE conflict exceeds `0.05`.

For each candidate, report empirical case risk and a one-sided Clopper-Pearson
upper bound. Because one of six coverages is selected, use Bonferroni-adjusted
confidence `1 - (1 - 0.95) / 6`. Select the largest coverage whose simultaneous
upper bound is at most `epsilon=0.05`; if none passes, reject the calibration
mechanism without reading confirmation.

This milestone only supports or rejects independent case-level calibration of the
frozen U3 ranking. It does not claim full-plan UNKNOWN/retention gates, compiler
authority, formal conditional coverage, safety, confirmation, or test validity.
No hashes, checksums, fingerprints, parameter sweeps, smoke matrix, or regression
matrix are added.
