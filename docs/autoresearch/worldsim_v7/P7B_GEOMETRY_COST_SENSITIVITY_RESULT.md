# P7-B Geometry-to-Cost Sensitivity Result

Date: 2026-09-02

Canonical run:
`run://worldsim_v7/WS-V7-P7B-GEOMETRY-COST-SENSITIVITY-01/20260902T191500Z__geometry-cost-sensitivity-s0-r2`

## Verdict

`supported_deterministic_geometry_cost_bound`

FP64 evaluation covers all 575,596 retained source rows, six frozen signed-clearance shifts, and the exact-match P5 test strata.
There are zero bound violations across every reported group/shift; the maximum numerical `shift - bound` is
`1.42e-14`. This resolves the FP32 roundoff failure `V7-F14` without changing the `1e-6` tolerance, perturbations, clearance
floor, rows, or groups.

## Full-source envelope

For the largest inward shift (`-0.20 m`), mean/q99 absolute cost shift is `0.02922/0.14501`; mean/q99 row-local bound is
`0.70064/10.02198`. Only `0.8409%` of rows cross signed clearance and `0.8798%` change denominator-floor state, but those rare
rows dominate the loose bound. The median and q90 tightness are numerically one because uniform shifts usually preserve the
maximizing profile point; the mean bound remains intentionally conservative at floor crossings.

## P5 selected versus abstained

| signed shift | selected mean / q99 cost shift | abstained mean / q99 cost shift | selected sign/floor crossing | abstained sign/floor crossing |
|---:|---:|---:|---:|---:|
| `-0.20 m` | `0.000847 / 0.005664` | `0.004560 / 0.019125` | `0 / 0` | `0.3333% / 0.3600%` |
| `+0.20 m` | `0.000790 / 0.005224` | `0.006830 / 0.017099` | `0 / 0` | `0.2640% / 0.2586%` |

P5 selected Actors are therefore `5.38x` less sensitive in mean cost under the largest inward perturbation and `8.64x` less
sensitive under the largest outward perturbation, even though P5-B found their long-horizon motion error to be `2.28x` larger.
This orthogonality is the main scientific result: geometric sensitivity, physical repairability, and motion uncertainty cannot be
collapsed into a scalar confidence.

## Boundary

The algebraic inequality is deterministic. Uniform shifts are not a learned or measured sensor-error distribution, the retained
profiles are not a causal repair intervention, and zero inequality violations are not a collision or road-safety guarantee.
