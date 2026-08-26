# P10R2 Route-Aware Conditional Compiler Freeze

Date: 2026-08-26  
Task: `WS-V64-P10R2-ROUTE-AWARE-COMPILER-01`  
Hypothesis: `WS-V64-H-P10R2-001`  
Run ID: `20260826T191500Z__route-aware-compiler-s0-r1`

## Motivation and authority boundary

P10T rejected the current frozen M0 empirical route-tail claim at worst10/96 CVaR `0.0517085 > 0.05`. That result remains
canonical and P11 remains locked for M0. P10R2 is a new M1 policy version, not a repair or reinterpretation of P10T.

The already-consumed P6R confirmation cohort is explicitly reclassified as M1 development/calibration data. No fresh
confirmation data may be inspected during this run. A passing result only creates a candidate for later exact-once evaluation.

## Frozen policy

- denominator: the same target-free native occupied-boundary voxels used by the compiler;
- risk score: unchanged frozen full-native selective MLP; no refit;
- M0 total nominal coverage: rain `0.40`; night/construction/vulnerable transit `0.50`;
- M1 total selected count: exactly the M0 count in every case;
- route definition: future 20 lidar poses / 2 seconds in target-lidar frame, radius `1.5m`;
- route selection cap: at most `floor(0.40 * route-eligible-count)` selected route voxels;
- reallocation: rank all voxels by the unchanged risk score and transfer unused route budget to the next lowest-risk non-route voxels;
- empirical tail: fixed worst `ceil(0.10 * 96)=10` case route hidden-FREE conflict mean.

No route-cap, total-coverage, corridor, horizon, tail-fraction, model, or seed sweep is allowed. The current M0 negative result
cannot be rewritten. No hash, checksum, fingerprint, smoke suite, or regression matrix is added.

## Frozen gates

1. M1 empirical route worst-decile mean `<=0.05`.
2. Absolute M1-minus-M0 mean realized total coverage `<=1e-6`.

Failure rejects this candidate without tuning. Passing supports only a consumed-cohort calibration candidate and requires a
metadata-only frozen, previously unread temporal-member confirmation cohort before any exact-once route-tail claim.

## Resource scheduling

The formal run uses the single RTX 3090 for route-corridor construction. Background raw-shard scanners remain paused while it
runs so slow local I/O cannot idle the GPU path. Cleanup resumes after the result is persisted.
