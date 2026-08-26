# P10T Empirical Route-Tail Audit Closeout

Date: 2026-08-26

Canonical run: `run://worldsim_v64/WS-V64-P10T-ROUTE-TAIL-AUDIT-01/20260826T190000Z__route-tail-audit-s0-r1`

Verdict: `rejected_empirical_route_tail`

The frozen worst-decile audit averaged the ten largest route-local hidden-FREE conflict rates among 96 cases. C0 empirical CVaR was 0.05042978771947152. M0 was 0.05170853114753883, exceeding the 0.05 gate and worsening C0 by 0.0012787434280673096. Five M0 cases exceeded 0.05 pointwise, and the maximum was 0.10638297872340426.

Target evidence was not reread. The model, policy, route corridor, and tail fraction did not change. The run used only frozen P10C rows and completed on CPU in 0.000759761780500412 seconds.

This negative result closes route/collision tail authority for the current frozen M0 and locks P11. It does not rewrite the supported scopes of P4C exact-once case risk, P10M target-free materialization, P10G sparse Gaussian splatting, P10R bounded route exposure, or P10C pooled route-local severity.

No post-quality retuning is permitted. A legal recovery requires a new route-aware policy version with independent calibration and a new confirmation cohort. This audit is empirical and does not claim a population CVaR bound.
