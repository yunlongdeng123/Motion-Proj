# WorldSim V7.1 M43 — frozen M39 nuScenes→AV2 zero-shot plan

## Question

Does the M39 categorical surface-return interpretation transfer to untouched AV2 sensor logs when both evidential heads,
M8 geometry, metric tolerances, and the CDF readout are frozen?

## Frozen candidate and input boundary

- Candidate: M8 centers/scales + M35 anchor F/O/U head + M38 child F/O/U head, composed exactly as the M39 direct
  categorical Gaussian surface-return measure.
- Anchor inputs reproduce the M33 31-dimensional contract from AV2 `query` and `build_frame_points`: KEEP/PROJECT
  provenance, source ray, canonical support, and build-ray F/O/U evidence. Child inputs reproduce the M37 parent-feature,
  child-local, residual, scale, and slot contract.
- AV2 target sweeps are unavailable to both heads and all geometry construction. They are used only once for final early/hit
  evaluation after all 20 frozen logs complete.
- No target-domain fine-tuning, pseudo-labeling, calibration, feature normalization fitting, threshold selection, log
  replacement, or failed-case deletion.

The sensor gap is deliberately not hidden. DGLSS (CVPR 2023) shows that LiDAR beam configuration and sparsity are major
cross-domain shifts; M43 therefore keeps metric ray evidence and source-trained feature scaling fixed rather than fitting an
AV2 correction. 3DLabelProp (ICCV 2023) motivates retaining sequentially accumulated geometry, which is already supplied by
the build-only canonical surfels.

## Evaluation and decisions

Baseline is the same frozen M8 unit-energy categorical readout used by M20/M21. Report all, hazard, and clear early-return
and ±0.20m hit rates over the complete frozen cohort. The three preregistered decisions are:

1. all-stratum M39 early delta versus baseline ≤ 0;
2. both hazard and clear early deltas ≤ 0;
3. all-stratum hit delta ≥ -1 percentage point.

Point-surface metrics and authority mass means are descriptive. No partial AV2 quality is read. If any decision fails,
register `V71-F43`, retain M39 only as a development-exposed mechanism result, and do not tune on AV2. If all pass, freeze
the external confirmation and move to explanation/safety-bound synthesis rather than a second target-domain experiment.
