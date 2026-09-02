# P10 Frozen Physical-Authority Audit Freeze

Date: 2026-09-02

## Question

On the exact-once fresh AV2 cohort, does the frozen nuScenes-trained P4 validity score place Actor-level visible physical
failures into abstention, or does “repairable” remain distinct from a physical safety certificate?

## Fixed analysis

- Join only the already frozen P3-C fresh visibility rows and P6-C fresh score rows by exact `(log_id, track_id)`.
- P4 is the primary paper selector and its stored `p4_selected` decision is immutable. P6-C is context only and cannot
  replace, recalibrate, or reverse the P4 verdict.
- Primary physical failure is `nonnew_visible_violation`; `chamfer_worsened_vs_query` is the fixed secondary failure.
- Report always-repair, selected, and abstained coverage/risk; abstention capture; safe-visible and Chamfer-nonworse AUROC;
  descriptive AURC; hazard/non-hazard coverage; and a one-sided 95% Wilson upper bound on selected visible-failure risk.
- Strong empirical containment requires all three pre-frozen gates: selected visible risk below always-repair, its Wilson
  upper bound below the always-repair point risk, and selected Chamfer-worsening no higher than always-repair.
- No training, target read, refit, recalibration, threshold scan, bootstrap, case removal, or new rendering is permitted.

The interval is a finite-cohort descriptive bound, not a cross-domain conformal/exchangeability or deployment-safety
guarantee. A failed gate is retained as an interpretability/safety boundary and cannot trigger a recovery on this cohort.
