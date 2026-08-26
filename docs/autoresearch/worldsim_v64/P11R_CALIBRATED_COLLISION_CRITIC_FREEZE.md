# P11R Independently Calibrated Collision-Critic Freeze

Date: 2026-08-27  
Task/Hypothesis: `WS-V64-P11R-CALIBRATED-COLLISION-CRITIC-01 / WS-V64-H-P11R-001`

## Recovery scope

V64-F28 is a long-tail operating-point failure: the frozen 0.5 threshold misses 98--100% of unsafe actions even though the
selected-policy progress gates pass. This recovery does not retrain the critic, change its features, alter the 13-action lattice,
or reuse P10R4 labels for threshold choice.

The three P11 critic models are loaded unchanged. P10R2 supplies a separate 96-case downstream action-label calibration cohort.
For each arm, one threshold is computed analytically as the 20th percentile of unsafe-action scores, corresponding to target
unsafe recall 0.80. There is no threshold grid or candidate selection. Thresholds and calibration metrics must be written before
the P4C downstream action labels are generated.

P4C supplies a different 96-case exact-once action-label evaluation cohort. Its occupancy quality was consumed by older tasks,
but its 13-action actor-envelope collision labels have never been generated and are unread at this freeze.

## Gates

- verified unsafe-action recall >= 0.80;
- verified selected-policy false-safe count no worse than both comparators;
- verified mean progress >= 0.50;
- verified stuck rate <= 0.20.

Safe-action precision, Brier/ECE, comfort and reward remain reported rather than gated. If calibration causes all-brake behavior,
the progress/stuck gates reject. If Real-only remains equally good, the verdict is no increment.

## Boundary

This remains a bounded actor-envelope collision proxy. It is not physical collision, population, planning, closed-loop, RL, or
safety evidence. No large NWM, critic retraining, threshold sweep, second evaluation, hash, checksum, fingerprint, smoke suite, or
regression matrix is authorized. Formal run: `20260827T034500Z__calibrated-collision-critic-s0-r1`.
