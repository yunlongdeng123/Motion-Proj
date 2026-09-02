# P10 Frozen Physical-Authority Audit Result

Date: 2026-09-02

Canonical run:

```text
run://worldsim_v7/WS-V7-P10-FROZEN-PHYSICAL-AUTHORITY-AUDIT-01/20260903T010000Z__physical-authority-s0-r2
```

The earlier r1 run is retained for audit but superseded because it interpreted the safe predicate
`nonnew_visible_violation=true` as a failure. The r2 analyzer changes only the predicate direction; cohort, exact join,
stored selections, gates, confidence formula, and all Chamfer/hazard quantities are unchanged.

## Result

The exact frozen join contains 523 Actors from 20 fresh AV2 logs. Always-repair visible-failure risk is 195/523=37.28%.
P4 selects 404 Actors (77.25% coverage), with selected risk 147/404=36.39% and a one-sided 95% Wilson upper bound of
40.40%. Abstained risk is 48/119=40.34%; abstention captures 48/195=24.62% of visible failures. Safe-visible AUROC is
0.533 and visible-failure AURC is 0.360.

P4 retains its original target: selected Chamfer-worsening is 14.60%, below the 19.50% always-repair rate, with
Chamfer-nonworse AUROC 0.655. Hazard/non-hazard coverage is 93.66%/71.13%. The P6-C context score is similarly weak for
visibility: selected risk 36.67%, Wilson upper 40.53%, and safe-visible AUROC 0.537, despite 14.81% selected
Chamfer-worsening.

## Decision

The point-risk and Chamfer gates pass, but the strong confidence gate fails because 40.40% is not below the 37.28%
always-repair point risk. The frozen verdict remains `p4_not_a_physical_safety_certificate`, and `V7-F20` is retained
with corrected semantics: the source-only repair score gives weak visibility ranking and no finite-cohort confidence
separation, even though it improves its intended Chamfer target.

No target refit, recalibration, threshold scan, cohort filtering, or recovery is allowed on these 523 Actors. The result
is an empirical fresh-AV2 boundary, not a conformal, exchangeable, per-Actor, or deployment-safety guarantee.
