# P10T Empirical Route-Tail CVaR Audit Freeze

Date: 2026-08-26

P10C passes pooled risk but has five pointwise case failures. CVaR is frozen as a worst-tail summary based on risk-sensitive decision-making and CVaR generalization literature:

- https://proceedings.neurips.cc/paper/2015/hash/64223ccf70bbb65a3a4aceac37e21016-Abstract.html
- https://proceedings.neurips.cc/paper/2020/hash/d02e9bdc27a894e882fa0c9055c99722-Abstract.html

Task: `WS-V64-P10T-ROUTE-TAIL-AUDIT-01`

Run: `20260826T190000Z__route-tail-audit-s0-r1`

Hypothesis: `WS-V64-H-P10T-001`

Read only the frozen 96 P10C rows. Treat a case with no emitted route voxel as zero conflict, sort M0 case conflict rates descending, and average the worst `ceil(96*0.10)=10`. The only scientific gate is empirical M0 CVaR at most 0.05.

The audit does not reread target evidence, change the model, change policy or route, optimize CVaR, or sweep the tail fraction. A pass is an empirical tail diagnostic only and not a PAC-Bayesian or population CVaR bound. A failure closes route/collision authority for this frozen V6.4 policy; it must not trigger post-quality retuning.
