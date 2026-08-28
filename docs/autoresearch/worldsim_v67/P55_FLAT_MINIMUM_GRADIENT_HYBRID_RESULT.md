# P55 Flat-Minimum Gradient Hybrid Result

Canonical: `run://worldsim_v67/WS-V67-P55-FLAT-MINIMUM-GRADIENT-HYBRID-01/20260829T053000Z__flat-minimum-gradient-s0-r1`.

P55 averaged the fixed final 1,200 checkpoints while training the same 5,320 cases / 59,608 rows / 14 domains as P53.
On P10R4 H=`.8s`, 95 cases were evaluable; selection was exact `328/328`, coverage `0.610526`, minimum group `0.50`,
and all eight scenes were non-increasing.

P55/P53/P31/fixed reduction was `0.688694/0.698266/0.694007/0.203041`. Deltas over P53 and P31 were
`-0.009572/-0.005313`; both decision gates failed, so 3/5 gates passed and verdict is
`rejected_flat_minimum_gradient_hybrid`. The averaging window/gate is not changed and the family closes.

Wall/peak GPU/peak RSS: `496.882s / 0.18637GiB / 1.36549GiB`.
