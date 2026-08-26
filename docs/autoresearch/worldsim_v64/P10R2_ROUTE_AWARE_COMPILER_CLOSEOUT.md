# P10R2 Route-Aware Conditional Compiler Closeout

Date: 2026-08-26  
Canonical run: `run://worldsim_v64/WS-V64-P10R2-ROUTE-AWARE-COMPILER-01/20260826T191500Z__route-aware-compiler-s0-r1`  
Verdict: `supported_route_aware_candidate_on_consumed_calibration`

The single frozen run consumed 96 cases from the already-read P6R cohort. M0 and M1 both realized mean total coverage
`0.4749504873`, so M1 preserved the selected count exactly. The route-aware constraint reduced route-selected voxels from
`5912` to `3826` and route hidden-FREE conflicts from `23` to `9` while reallocating the remaining budget outside the route.

M0 and M1 empirical worst10 route conflict means were `0.0220498646` and `0.0114782638`. M1's maximum case route conflict
was `0.0454545455`, with no case above `0.05`. The CVaR gate and total-coverage preservation gate both passed. GPU wall time
was `11.3438444s` and peak RSS was `0.8849335GiB`.

The risk model was not refit, policy parameters were not selected during the run, and no new confirmation evidence was read.
This closeout supports only a calibration candidate. It does not replace the canonical P10T rejection for current M0, close
V64-F21, unlock P11, or establish collision, planning, closed-loop, population-tail, or safety authority. The only admissible
next step is a metadata-only freeze of a previously unread temporal-member confirmation cohort followed by one fixed M1 run.
