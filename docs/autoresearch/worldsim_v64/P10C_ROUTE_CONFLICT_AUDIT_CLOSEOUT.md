# P10C Route-Local Conflict Severity Closeout

Date: 2026-08-26

Canonical run: `run://worldsim_v64/WS-V64-P10C-ROUTE-CONFLICT-AUDIT-01/20260826T184500Z__route-conflict-audit-s0-r1`

Verdict: `supported_route_local_conflict_severity`

C0 emitted 9,450 route-local voxels with 34 hidden-FREE conflicts, rate 0.0035978835978835977. M0 emitted 10,013 with 43 conflicts, rate 0.004294417257565165. M0 therefore added 563 route-local states and nine conflicts while remaining far below the preregistered pooled 0.05 gate. Both minimal gates passed.

Five of 96 M0 cases nevertheless exceeded 0.05 locally; the maximum was 0.10638297872340426. P10C resolves V64-F20's binary-metric saturation by providing cell-level severity, but it does not resolve tail risk. V64-F21 records that boundary.

Target evidence was read once after policy and route freeze. The model was not refit, and collision ground truth was not read. GPU wall time was 4.169704079627991 seconds. The result supports pooled route-local hidden-FREE severity only, not physical collision, planning, closed-loop behavior, or safety.
