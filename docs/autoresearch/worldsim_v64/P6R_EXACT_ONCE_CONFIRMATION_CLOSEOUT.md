# P6R exact-once confirmation closeout

Date: 2026-08-26

Canonical run: `run://worldsim_v64/WS-V64-P6R-EXACT-ONCE-CONFIRMATION-01/20260826T153500Z__exact-once-confirmation-s0-r1`.

The frozen full-native selective MLP was applied once to the 96 confirmation cases at the independently calibrated nominal coverage of 0.40. Mean realized coverage was 0.3999405. One case failed, giving empirical case risk 1/96 = 0.0104167. The sole failure was in the night stratum (1/24); construction, rain, and vulnerable-transit each had 0/24 failures.

Both preregistered gates passed: overall failures were at most 4/96 and every stratum had at most 1/24. There was no model refit and no confirmation-time policy or coverage selection. GPU scoring took 12.5902 seconds with peak RSS 0.7907 GiB.

Verdict: `supported_exact_once_confirmation`. This is exact-once observed case-risk evidence for the frozen selective policy only; it is not a real-world safety claim and does not establish a downstream compiler result.
