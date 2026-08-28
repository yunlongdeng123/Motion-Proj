# P57 SAM Gradient Hybrid Result

Canonical: `run://worldsim_v67/WS-V67-P57-SAM-GRADIENT-HYBRID-01/20260829T070000Z__sam-gradient-s0-r1`.

Standard two-pass SAM with fixed radius `.05` trained the same 5,320 cases / 59,608 rows / 14 domains as P53. On P10R2
H=`.8s`, 93 cases were evaluable; selection was exact `344/344`, coverage `0.645161`, minimum group `0.50`, and all eight
scenes were non-increasing.

P57/P53/P31/fixed reduction was `0.731922/0.723709/0.727373/0.182775`. P57 improved P53 by `+0.008213`, but improved
P31 by only `+0.004549`, missing the frozen `+0.005` gate by `0.000451`. Four of five gates passed; verdict is
`rejected_sam_gradient_hybrid`. The gate is not reduced and rho/ASAM are not swept; sharpness optimization closes.

Wall/peak GPU/peak RSS: `989.278s / 0.17979GiB / 1.35990GiB`.
