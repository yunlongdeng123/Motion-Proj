# P21 Monotone Safety Boundary Freeze

Date: 2026-09-02

## Formal object

For ray `r`, surface set `S`, frozen lateral tolerance, and positive projected depth, define `d_S(r)` as the minimum valid depth or
infinity. If `S'` is a subset of `S`, then `d_{S'}(r) >= d_S(r)`. Therefore the early predicate
`1[d_S(r) < d_gt(r) - tau]` and new-early relative to a fixed query surface are monotone non-increasing under deletion.

Neither hit retention nor symmetric Chamfer has this property. A deleted first return may expose a hit behind it or remove the only
hit; Chamfer's predicted-to-target term can improve when an outlier is removed while the target-to-predicted term can worsen when
support disappears. These are structural limits, not calibration errors.

## Frozen analysis

P21 reads only the completed P20 summary and computes, for P17/P17R/P19, early events removed, new hits lost, Chamfer penalty,
events removed per hit lost, and events removed per Chamfer millimeter. It does not recompile data, train a model, choose a policy,
or read AV2. Undefined zero-denominator ratios remain null; no smoothing or epsilon is introduced.

The theorem is exact for deletion-only compiled sets under the true first-return operator. Empirical ratios remain cohort-specific and
must not be called a road-safety guarantee. The artifact is intended for the paper's interpretability/safety-boundary section and to
close further deletion-only policy sweeps.
