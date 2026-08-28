# V6.7 P23 entropic action compiler freeze

P23 replaces the saturated binary any-event auxiliary with a continuous selected-cost entropic objective:
`log E[exp(10 * cost)] / 10`. Its weight is fixed at 0.25. Mean, pairwise and listwise losses, the `32/16` architecture,
case-centered `±0.02` residual and 0.25 selection fraction remain unchanged.

Seven consumed domains train the compiler. P20 and P22 baselines plus P23 are frozen before materializing V64 P10R2 scenes
`1020/1016/0596/0590/0006/0472/0070/0371`. Gates require mean reduction >=0.35, selected top-10% tail mean <=95% of P20,
pairwise >=0.70 and six non-increasing scenes.

This is a continuous risk-sensitive objective and descriptive selected tail, not an OCE/CVaR guarantee, collision metric, policy,
closed-loop or safety claim. No risk-aversion, weight, tail-fraction, model, temperature or gate sweep is allowed.
