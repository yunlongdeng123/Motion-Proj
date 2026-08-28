# P52 Frozen Horizon Extrapolation Result

Canonical: `run://worldsim_v67/WS-V67-P52-FROZEN-HORIZON-EXTRAPOLATION-01/20260829T034000Z__short-horizon-extrapolation-s0-r1`.

Frozen P51/P31/P20 was read once on P10V H=`0.8s`, below every training horizon. The new cache retained `694/864`
actions and 67 cases were evaluable. Selection was exact `229/229`, case coverage `0.671642`, minimum group coverage
`0.521739`, and all six scenes were non-increasing.

P51/P31/fixed reduction was `0.761914/0.680754/0.222998`; delta over P31 was `+0.081161`. All four gates passed;
verdict=`supported_frozen_short_horizon_extrapolation`. No training/refit/sweep. Wall/peak GPU/peak RSS was
`0.500s / 0.00913GiB / 0.71574GiB`.
