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

## Execution state at final V7.1 handoff（2026-09-05）

- canonical run=`run://worldsim_v71/WS-V71-M43-M39-AV2-ZERO-SHOT-01/20260905T091500Z__m43-m39-av2-zero-shot-r1`；
- state=`running/waiting_fresh_av2`, completed=`16/20` logs、276 Actors；current log=
  `c85a88a8-c916-30a7-923c-0c66bd3ebbd3`；
- one evaluator and one downloader were alive; recent execution log had no error/retry; AV2 data=`~79GiB`, disk free=`86GiB`；
- final `summary.json` absent；partial Actor output and all partial quality remained unread；external verdict=`no_verdict`；
- user-requested shutdown pauses this run after repository/paper push. Resume only the same frozen run/data with one
  downloader; read quality once after 20/20 and normal evaluator exit. No target adaptation is authorized.

## Exact-once resume protocol（2026-09-05）

After the server restart, resume the same canonical run with `--resume`. The evaluator loads its own partial rows only for
final concatenation, skips the cohort prefix recorded by `status.completed_logs`, and starts at the next log. It must not
aggregate or expose partial quality. The downloader independently skips existing `.complete` markers. Exactly one evaluator
and one downloader may run. This is an execution recovery only: all checkpoints, features, metrics, decisions, and AV2
prohibitions remain unchanged. The missing original resume path is recorded as `V71-F50` and resolved before new target reads.
