# P11 Provenance-Conditioned Authority Audit Freeze

Date: 2026-09-02

## Literature-to-project migration

- Savinov et al. (CVPR 2016) formulate reconstruction with viewing-ray potentials and an explicit visibility-consistency
  constraint; this motivates using measurement witnesses rather than a generic confidence feature.
- SelectiveNet (ICML 2019) evaluates rejection against the actual selective risk; P10 showed that P4's Chamfer target is not
  the visible-ray risk, so the two authorities must be evaluated separately.
- DGLSS (CVPR 2023) uses source-only sparsity consistency for LiDAR domain generalization; V7-F19 already shows that this
  improves one external ranking but reverses on another, so P11 does not add another feature-consistency head.

## Fixed analysis

- A compiled Actor is provenance-certified iff `completion_decision_count == action_counts.COMPLETE == 0`. Its output then
  consists only of sensor-supported `KEEP` points and exact matched-termination `PROJECT` points; `UNKNOWN` inputs are not
  inserted into the compiled surface.
- The deployable dual authority is the frozen P4 decision AND the provenance certificate. P4 and the compiler are unchanged.
- Compare always-repair, P4-only, provenance-only, and dual groups on the already frozen 523-Actor fresh AV2 join.
- The five gates are frozen before aggregate read: dual visible risk below P4, dual one-sided 95% Wilson upper below the P4
  point risk, dual Chamfer-worsening no higher than P4, at least 10% coverage, and at least 50% hazardous-Actor coverage.
- Report completion-count unsafe-visible AUROC as a descriptive ranking diagnostic. No count threshold is scanned.
- No training, target fit, recalibration, threshold change, compiler edit, or case deletion is permitted.

This is a consumed-cohort development audit. Even a pass would certify only provenance to already observed ray terminations,
not unseen-surface completeness, future-view consistency, sensor noise, collision, planning, or road safety.
