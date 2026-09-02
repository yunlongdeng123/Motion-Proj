# P11 Provenance-Conditioned Authority Audit Result

Date: 2026-09-02

Canonical run:

```text
run://worldsim_v7/WS-V7-P11-PROVENANCE-AUTHORITY-AUDIT-01/20260903T011500Z__provenance-authority-s0-r2
```

The earlier r1 run is retained for audit but superseded because it interpreted the safe predicate
`nonnew_visible_violation=true` as a failure. The r2 analyzer changes only the predicate direction; provenance groups,
P4 selection, coverage, Chamfer, hazard counts, and frozen gates are unchanged.

## Result

The no-COMPLETE provenance witness covers 193/523 Actors (36.90%) and reduces visible-failure risk from the 37.28%
always-repair baseline to 54/193=27.98% (one-sided 95% Wilson upper 33.57%). Conjoining it with frozen P4 retains
124/523 Actors (23.71% coverage) and reduces visible-failure risk from P4's 36.39% to 25/124=20.16%, with one-sided 95%
upper 26.70%.

That visibility gain is incompatible with the other authority requirements. Provenance-only Chamfer-worsening is 48.19%.
The dual group worsens Chamfer for 54/124=43.55%, has mean Chamfer gain -0.0037 m, and retains only 5/142 hazardous Actors
(3.52% hazard coverage). Completion count ranks unsafe visibility with AUROC 0.589.

## Decision

The two visible-risk gates and minimum overall coverage pass; Chamfer non-regression and minimum hazard coverage fail. The
frozen verdict remains `provenance_witness_does_not_certify_future_visibility`, and `V7-F21` is retained with corrected
semantics: observed-ray provenance is useful visibility evidence, but it is not a jointly valid repair authority and cannot
be obtained by discarding nearly all hazardous Actors.

The deterministic witness remains valid for its narrow object--retained output points arise from KEEP or matched-hit
PROJECT--but it does not certify unseen surfaces, future views, collision, planning, or road safety. No completion-count
threshold, conjunction, compiler operator, case set, or target-calibrated rule may be tried on these rows.
