# P10R3 Fixed Route-Denominator Audit Closeout

Date: 2026-08-27  
Canonical run: `run://worldsim_v64/WS-V64-P10R3-FIXED-DENOMINATOR-AUDIT-01/20260827T013000Z__fixed-denominator-audit-s0-r1`  
Verdict: `diagnosed_fixed_denominator_direction_consistent`

The rows-only diagnostic completed on 96 consumed-calibration cases and 96 fresh-confirmation cases without rereading target,
model, or evidence artifacts. M0/M1 shared the same route-eligible denominator in each cohort.

On consumed calibration, pooled fixed-denominator conflict density changed from `0.00236358` to `0.000924879`, while
worst10 CVaR changed from `0.01323514` to `0.004552403` (`M1-M0=-0.008682740`). On fresh confirmation, pooled density
changed from `0.004217761` to `0.001562134`, while worst10 CVaR changed from `0.02164704` to `0.01498322`
(`M1-M0=-0.006663816`).

This direction is descriptive evidence that the selected-only tail reversal was sensitive to the arm-dependent selected
denominator. It is not an independent confirmation because the diagnostic was designed after reading P10R2 confirmation.
V64-F25 therefore remains active and P11 comparative authority remains locked. No policy, route, tail fraction, denominator,
model, or threshold was changed or swept. Runtime was `0.00264s` CPU-only with `0.1680GiB` peak RSS.
