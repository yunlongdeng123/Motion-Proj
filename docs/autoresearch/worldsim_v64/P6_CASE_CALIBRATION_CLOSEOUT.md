# P6 Case-Level Calibration Closeout

Date: 2026-08-26

Canonical run:
`run://worldsim_v64/WS-V64-P6-CALIBRATION-01/20260826T131000Z__case-calibration-s0-r1`.

Verdict: `rejected_no_positive_coverage`.

The frozen U3 model was evaluated once on 192 independent target cases from 16
scenes and four metadata strata. Calibration target was read; confirmation and
test targets were not read.

| Nominal coverage | Failed cases | Empirical case risk | Simultaneous upper bound |
| ---: | ---: | ---: | ---: |
| 0.05 | 41/192 | 0.213542 | 0.292860 |
| 0.10 | 54/192 | 0.281250 | 0.365775 |
| 0.20 | 62/192 | 0.322917 | 0.409553 |
| 0.30 | 74/192 | 0.385417 | 0.473895 |
| 0.40 | 80/192 | 0.416667 | 0.505518 |
| 0.50 | 93/192 | 0.484375 | 0.572863 |

At 5% coverage, construction/night/rain/vulnerable-transit failures were
`4/16/8/13` out of 48 cases per stratum. Thus the failure is not caused by the
finite-sample correction: empirical risk alone is more than four times the 0.05
target, and night is the largest shift.

No policy was selected and confirmation remains locked. The P5 PCA16 linear U3
ranking is rejected as a case-level risk controller on the new cohort. Legal
recovery requires a newly frozen selective model trained on these now-consumed
development scenes, a still-unread calibration cohort, and a newly selected
metadata-only confirmation cohort. The old calibration result may not be used to
lower epsilon, raise the conflict threshold, remove strata, or choose a smaller
unregistered coverage.
