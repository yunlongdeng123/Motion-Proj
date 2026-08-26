# P10R4 Fixed-Denominator Exact-Once Closeout

Date: 2026-08-27  
Branch: `research/worldsim-v6.4-native-uq`  
Task: `WS-V64-P10R4-FIXED-DENOMINATOR-EXACT-ONCE-01`

## Outcome

The single preregistered untouched-test run completed with verdict
`supported_exact_once_fixed_denominator_relative_confirmation`:

`run://worldsim_v64/WS-V64-P10R4-FIXED-DENOMINATOR-EXACT-ONCE-01/20260827T025000Z__exact-once-fixed-denominator-s4-r1`

| Measure | M0 conditional | M1 route-aware | M1 - M0 |
| --- | ---: | ---: | ---: |
| Mean realized coverage | 0.474969689 | 0.474969689 | 0 |
| Fixed-denominator worst-10 CVaR | 0.020725740 | 0.010821074 | -0.009904666 |
| Pooled fixed-denominator conflict density | 0.004944667 | 0.002001413 | -0.002943254 |
| Route selected voxels | 8,760 | 6,425 | -2,335 |
| Route hidden-FREE conflicts | 84 | 34 | -50 |

All three frozen gates passed. Paired case directions were M1 lower/equal/higher = 18/78/0; the preregistered half-tie
probability was 0.59375 and remains descriptive, not a significance gate. Runtime was 11.623258 s with 0.879780 GiB peak RSS.

No model refit, runtime policy selection, route/coverage/tail/denominator sweep, bootstrap gate, significance gate, second test,
hash, checksum, fingerprint, smoke suite, or regression matrix was used.

## Authority boundary

V64-F25 is resolved only at the independent 96-case exact empirical fixed-opportunity layer. This result does not rewrite the
P10R2 selected-denominator formal outcome and does not erase V64-F21's negative result for current M0. It supplies no population
bound and no physical collision, planning, closed-loop, or safety authority.

The P11 lock caused by the unresolved relative fixed-denominator question is removed only for bounded design work. Large-NWM
training remains locked; the next step must first freeze a single-GPU collision-critic protocol with anti-trivial completion/stuck
controls.
