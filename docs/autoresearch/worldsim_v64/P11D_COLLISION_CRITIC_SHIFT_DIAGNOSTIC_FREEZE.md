# P11D Collision-Critic Shift Diagnostic Freeze

Date: 2026-08-27  
Task/Hypothesis: `WS-V64-P11D-COLLISION-CRITIC-SHIFT-DIAGNOSTIC-01 / WS-V64-H-P11D-001`

P11 and its one recovery are terminal negative. This diagnostic only reads the already-written P11R calibration/evaluation action
rows and frozen thresholds. It reports per-arm unsafe priors, average precision, AUROC, safe/unsafe score quantiles and their
cross-cohort deltas.

There is no confirmatory gate, native/evidence reread, GPU use, refit, threshold change, policy selection, second evaluation, or
P11 reopening. The result is failure characterization for the V6.4 technical report only. Formal run:
`20260827T040000Z__collision-critic-shift-s0-r1`.
