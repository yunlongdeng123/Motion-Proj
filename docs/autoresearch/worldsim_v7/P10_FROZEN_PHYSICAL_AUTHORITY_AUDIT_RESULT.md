# P10 Frozen Physical-Authority Audit Result

Date: 2026-09-02

Canonical run:

```text
run://worldsim_v7/WS-V7-P10-FROZEN-PHYSICAL-AUTHORITY-AUDIT-01/20260902T234500Z__physical-authority-s0-r1
```

## Result

The exact frozen join contains 523 Actors from 20 fresh AV2 logs. P4 selects 404 Actors (77.25% coverage). The selected
non-new-visible-violation rate is 63.61% (257/404), versus 62.72% (328/523) under always-repair; its one-sided 95% Wilson
upper bound is 67.45%. The safe-visible AUROC is 0.467 and abstention captures only 21.65% of visible failures.

P4 does retain its original target: selected Chamfer-worsening is 14.60%, below the 19.50% always-repair rate, with
Chamfer-nonworse AUROC 0.655. Hazard/non-hazard coverage is 93.66%/71.13%. The context-only P6-C score exhibits the same
separation: 63.33% selected visible violation and safe-visible AUROC 0.463 despite 14.81% selected Chamfer-worsening.

## Decision

Only the fixed Chamfer gate passes. The two visible-risk gates fail, so the frozen verdict is
`p4_not_a_physical_safety_certificate` and failure `V7-F20` is registered. This is not a contradiction of the P4
repair-or-abstain result: the score predicts target Chamfer non-worsening, not bidirectional visibility consistency.

No target refit, recalibration, threshold scan, cohort filtering, or recovery is allowed on these 523 Actors. The result is
an empirical fresh-AV2 safety boundary, not a conformal, exchangeable, per-Actor, or deployment-safety guarantee.
