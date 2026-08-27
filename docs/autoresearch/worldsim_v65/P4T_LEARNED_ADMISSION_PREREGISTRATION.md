# P4T Learned Admission Preregistration

`WS-V65-H-P4T-001` asks a scientifically separate question after trajectory and Actor/time representation closure: can a
low-capacity selector replace V6.4's hand-written stratum coverage lookup while keeping the V6.4 risk model and route cap
frozen? This is a Tier-L train-only feasibility experiment. It cannot reopen P2/P3 or create a V6.5 admission claim.

## Data and visibility

Eight V6.4 calibration scenes (96 cases) form train; eight different V6.4 conditional-confirmation scenes (96 cases) form
nested evaluation. All are Tier L in V6.5. The selector sees only continuous, method-visible summaries of the frozen risk
distribution, eligible count, route fraction, route-score summaries, and normalized target time. It never receives scene ID,
stratum ID, or hidden truth at inference. Stratum is retained only for M1 comparison and reporting.

GPU scoring and route geometry for the current case overlap one-thread prefetch of the next evidence/native unit. A compact
case cache is materialized once. No V6.5 representation/admission selection, calibration, confirmation, or test partition is
read.

## Frozen model and target

The V6.4 selective MLP remains frozen. For training cases only, the target is the largest prefix coverage in `[0.30, 0.55]`
whose cumulative hidden-FREE conflict is at most 0.05 after applying the already-supported M1 route cap of 0.40. A single
`context → 32 → 16 → coverage` MLP uses seed 0, 400 full-batch epochs, AdamW, and SmoothL1. There is no seed, capacity,
coverage-bound, or threshold sweep.

## Frozen evaluation

Comparator M1 retains its fixed stratum coverages (`0.50/0.50/0.40/0.50`) and the same route cap. G0 uses only predicted
continuous coverage and the same route cap. Positive train-only support requires:

- either mean coverage uplift at least 0.05 or fixed-route worst-10% tail reduction at least 10%;
- no more overall case failures than M1;
- pooled fixed-route conflict density regression at most 5%;
- at least 5/8 evaluation scenes with nonnegative utility (coverage gain without added failure, or lower fixed-route density).

Failure closes learned admission without a fresh cohort. Success only unlocks a separately frozen six-scene
D-Selection-Admission cohort and does not itself support the method.

Migration basis: SelectiveNet (ICML 2019) explicitly optimizes selective risk under a coverage constraint, while calibrated
selective classification separates model fitting from selector calibration. This probe keeps that separation and does not
use V6.5 selection data to fit or tune the selector.

- https://proceedings.mlr.press/v97/geifman19a
- https://github.com/ajfisch/calibrated-selective-classification
