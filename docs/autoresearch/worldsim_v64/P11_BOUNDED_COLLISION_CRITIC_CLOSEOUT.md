# P11 Bounded Collision-Critic Closeout

Date: 2026-08-27  
Branch: `research/worldsim-v6.4-native-uq`  
Task: `WS-V64-P11-BOUNDED-COLLISION-CRITIC-01`

## Formal outcome

Canonical run:

`run://worldsim_v64/WS-V64-P11-BOUNDED-COLLISION-CRITIC-01/20260827T033000Z__bounded-collision-critic-s0-r1`

The frozen formal verdict is `supported_bounded_unc_verified_collision_critic` because all three preregistered gates passed.
Selected-policy collision false-safe counts for Real-only / naive / UNC-verified were 13/12/12; all arms had mean progress 1.0
and stuck rate 0.

## Full primary-metric interpretation

| Metric | Real-only | Real + naive | Real + UNC verified |
| --- | ---: | ---: | ---: |
| Training rows / positives | 384 / 3 | 1,152 / 191 | 768 / 96 |
| Unsafe-action recall | 0.021739 | 0 | 0.010870 |
| Action false-safe | 180 | 184 | 182 |
| Safe-action precision | 0.846547 | 0.844463 | 0.841877 |
| Brier score | 0.180185 | 0.161755 | 0.174284 |
| ECE | 0.194078 | 0.144119 | 0.180180 |
| Policy false-safe | 13 | 12 | 12 |
| Mean policy reward | 0.820833 | 0.833333 | 0.833333 |

The complete 1,248-action denominator contains 184 unsafe actions. All three critics miss almost all of them at the frozen 0.5
threshold. UNC verification does not improve the selected policy over naive augmentation and has worse Brier/ECE. Therefore the
formal narrow gate pass is preserved, but the critic is rejected as collision authority and V64-F28 is active.

Runtime was 26.646745 seconds, peak RSS 1.080910 GiB, and peak CUDA allocation 0.073706 GiB. No large NWM or RL policy was trained.

## Recovery boundary

The P10R4 evaluation labels may not be used to tune the threshold, action lattice, architecture, loss, or augmentation. Literature
review indicates that the narrowest relevant migration is independent long-tail/logit calibration rather than another backbone:
class-balanced loss (CVPR 2019), logit adjustment (ICLR 2021), and the offline violation-critic separation in Recovery RL.

One recovery may keep the already-fitted critics, derive an analytic unsafe-recall threshold from a separate cohort whose action
labels have never been generated, and evaluate once on another unread action-label cohort. Failure of recall or anti-trivial
progress/stuck closes P11; it does not unlock large-NWM training.
