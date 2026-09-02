# P14 Hazard-Stratified Defer-to-Query Result

## Canonical

`run://worldsim_v7/WS-V7-P14-HAZARD-STRATIFIED-DEFER-01/20260903T031500Z__hazard-stratified-s0-r1`

Status is `done` on the frozen 20-log, 523-Actor P13 composite rows. The run is CPU-only (`0.0101 s`), reads no dataset, and performs no training, fitting, calibration, thresholding, or policy search.

## Exact accounting

There are 142 hazardous Actors (`27.15%`) and 381 clear Actors. For every frozen policy and both strata, the two accounting identities hold to maximum absolute residual `5.55e-17`:

- `composite gain = repair coverage * selected mean gain`;
- `introduced visible-failure mass = repair coverage * selected conditional failure`.

The population quantities are the Actor-share-weighted sums of the two strata. This numerical residual verifies only the implementation of an algebraic identity; it is not a statistical or safety guarantee.

## Primary results

| Policy | Hazard / clear coverage | Hazard / clear selected visible risk | Hazard failure share / burden amplification | Hazard gain share | Population introduced failure / Chamfer gain |
|---|---:|---:|---:|---:|---:|
| always repair | 100.00 / 100.00% | 46.48 / 33.86% | 33.85% / 1.247x | 56.38% | 37.28% / .08838 m |
| P4 defer | 93.66 / 71.13% | 47.37 / 31.00% | 42.86% / 1.578x | 56.42% | 28.11% / .08311 m |
| P6-C defer | 99.30 / 78.22% | 46.81 / 31.88% | 40.99% / 1.510x | 56.01% | 30.78% / .08895 m |
| P4 and visibility | 3.52 / 8.92% | 0 / 11.76% | 0 / 0x | 4.23% | .76% / .00122 m |

P4 reduces introduced failures from 195 to 147. Only three of the 48 removed failures are hazardous (`66 -> 63`); 45 are clear (`129 -> 84`). Its hazardous selected risk is `+0.89 pp` versus always-repair while clear selected risk is `-2.86 pp`. P6-C repairs 141/142 hazardous Actors and leaves all 66 hazardous failures, so its entire 34-failure reduction is in the clear stratum. Hazardous Actors provide about 56% of P4/P6-C Chamfer gain while carrying 41--43% of introduced failures despite being 27.15% of Actors.

P4-and-visibility has zero observed hazardous failure only because it repairs five hazardous Actors (`3.52%` hazard coverage). Its total gain is `.00122 m`; this confirms the previously closed safety-through-abstention boundary rather than recovering the P12 head.

## Verdict

The result supports an interpretable decomposition, not a new selector. P4 and P6-C preserve every Actor and the frozen hazard state, but that preservation must not be described as reduced hazard-stratum visibility risk. Their aggregate failure reduction is driven almost entirely by clear Actors. Conversely, the lowest-risk policy largely declines hazardous repairs and has negligible utility.

This does not contradict the existing V7 claim, which scopes “hazard preserving” to immutable identity, trajectory, extent, lifecycle, and analytic hazard fields and explicitly withholds road-safety authority. No `V7-F24` is registered; the result tightens the explanation and leaves the next available failure ID unchanged.

## Claim boundary

The hazardous field is a deterministic project proxy, not collision ground truth, planner outcome, demographic group, or real-road safety label. The consumed AV2 cohort is descriptive. No conformal, exchangeability, causal, planning, closed-loop, or deployment-safety claim is made.
