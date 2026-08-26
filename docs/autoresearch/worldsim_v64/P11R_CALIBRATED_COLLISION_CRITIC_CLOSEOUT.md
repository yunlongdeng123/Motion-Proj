# P11R Independently Calibrated Collision-Critic Closeout

Date: 2026-08-27  
Task: `WS-V64-P11R-CALIBRATED-COLLISION-CRITIC-01`

Canonical run:

`run://worldsim_v64/WS-V64-P11R-CALIBRATED-COLLISION-CRITIC-01/20260827T034500Z__calibrated-collision-critic-s0-r1`

Verdict: `rejected_independently_calibrated_collision_critic`.

## Calibration

The frozen P11 critic models were not retrained. P10R2 provided 1,248 calibration actions with 88 actor-envelope unsafe labels.
The analytic thresholds for Real-only / naive / UNC-verified were `4.248e-18 / 0.191678 / 0.084891`. Naive and verified each
recalled 70/88 unsafe actions (0.795455). The discrete quantile did not reach exactly 0.80; it was not corrected after observation.

Real-only classified every action unsafe and fell back to stop in all 96 cases. Naive retained progress 1.0. Verified had
progress 0.8125 and stuck rate 0.1875.

## Exact evaluation

P4C supplied 1,248 previously unread downstream action labels with 137 unsafe actions.

| Metric | Real-only | Real + naive | Real + UNC verified |
| --- | ---: | ---: | ---: |
| Unsafe-action recall | 1.000000 | 0.613139 | 0.620438 |
| Safe-action precision | 1.000000 | 0.926389 | 0.911414 |
| Policy false-safe | 0 | 3 | 2 |
| Mean progress | 0 | 1.000000 | 0.872396 |
| Stuck rate | 1.000000 | 0 | 0.114583 |
| Mean reward | 0 | 0.922917 | 0.807813 |

Verified failed the 0.80 recall gate and could not be no-worse than the all-stop Real-only false-safe count. It passed only the
progress and stuck gates. This demonstrates the central tradeoff rather than a useful recovery: a near-zero operating point can
eliminate false-safe only by rejecting all actions, while thresholds that preserve progress do not generalize unsafe recall.

Runtime was 26.552831 seconds, peak RSS 0.906693 GiB, and peak CUDA allocation 0.065771 GiB.

## Terminal boundary

V64-F28 closes negative after its single recovery. P11 does not support actor-envelope collision authority, physical collision,
planning, closed-loop, RL, or safety claims. No source critic retraining, threshold grid, P10R4 reuse, threshold change during
evaluation, second evaluation, large NWM, hash, checksum, fingerprint, smoke suite, or regression matrix was used.

Only rows-only failure characterization and the V6.4 report/evidence index remain authorized.
