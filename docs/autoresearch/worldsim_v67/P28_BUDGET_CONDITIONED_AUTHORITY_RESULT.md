# P28 Budget-Conditioned Authority Result

Canonical run: `run://worldsim_v67/WS-V67-P28-BUDGET-CONDITIONED-AUTHORITY-01/20260828T162000Z__budget-conditioned-s0-r1`.

P28 trained 1,708 case-budget rows at fractions `0.25` and `0.50` while excluding P10R4 from training. At the held-out
one-third fraction, all 96 P10R4 cases were evaluable. The compiler selected exactly 363 actions, covered 68 cases (`70.8333%`),
and attained group coverages `83.3333%`, `58.3333%`, `83.3333%`, and `58.3333%`.

Relative selected-cost reduction was `0.674930`, versus `0.281451` for fixed P20 at the same total budget, an improvement of
`+0.393479`. All eight scenes are non-increasing and all six gates pass. Wall time was `53.868s`, peak allocated GPU memory
`0.01713GiB`, and peak RSS `1.1817GiB`.

This supports interpolation to an unseen budget condition on a globally consumed cohort. It does not establish fresh-cohort,
collision, planning, policy, closed-loop, population, or safety guarantees.
