# P11 Provenance-Conditioned Authority Audit Result

Date: 2026-09-02

Canonical run:

```text
run://worldsim_v7/WS-V7-P11-PROVENANCE-AUTHORITY-AUDIT-01/20260903T001500Z__provenance-authority-s0-r1
```

## Result

The no-COMPLETE provenance witness covers 193/523 Actors (36.90%) but has 72.02% visible violation, 48.19% Chamfer
worsening, and only 6 hazardous Actors. Conjoining it with frozen P4 retains 124 Actors (23.71% coverage) and only 5/142
hazardous Actors (3.52% hazard coverage). Dual visible violation is 79.84% with a one-sided 95% Wilson upper bound of
85.10%; Chamfer worsening is 43.55% and mean Chamfer gain is -0.0037 m.

All risk and hazard gates fail; only minimum overall coverage passes. Completion count has unsafe-visible AUROC 0.411, so
more completion is not an unsafe ranking signal on this cohort. The opposite association reflects opportunity: Actors with
enough multi-view support to complete are also easier to evaluate against held-out rays, while `COMPLETE==0` concentrates
sparse, low-support Actors.

## Decision

The frozen verdict is `provenance_witness_does_not_certify_future_visibility`; register `V7-F21` and close provenance-gate
recovery. The deterministic witness remains valid for its narrow object—all emitted points arise from KEEP or matched-hit
PROJECT—but that observed-ray statement cannot certify future views or preserve hazard authority.

No completion-count threshold, conjunction, compiler operator, case set, or target-calibrated rule may be tried on these rows.
