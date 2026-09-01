# V7 P1 AV2 Actor-surface factorial atlas result

Canonical run:
`run://worldsim_v7/WS-V7-P1-AV2-FACTORIAL-ATLAS-01/20260902T104500Z__av2-factorial-s0-r1`.

## Verdict

`supported_zero_shot_av2_actor_surface_factorial_atlas`; all 10 frozen gates pass.

| Evidence | Result |
| --- | ---: |
| AV2 quantitative logs | 20/20 |
| eligible / hazard / safe Actors | 1,046 / 241 / 805 |
| stable Actor-local surfels | 1,074,935 |
| paired probes | 134,914 |
| single-frame target distance | 0.864454 m |
| fused target distance | 0.130623 m |
| distance ratio | 0.151105 |
| single-frame / fused recall @ 0.20 m | 0.261650 / 0.857224 |
| action accuracy / artifact recall | 1.0 / 1.0 |
| clean-hazard false-artifact | 0.0 |
| ghost FREE violation before / after | 0.15 / 0.0 m |
| Actor and hazard retention | 1.0 |
| paired validity / hazard leakage | 0.0 / 0.0 |

The held-out surface comparison uses real AV2 LiDAR returns: frame indices divisible by three are excluded from fusion and used
only as target geometry. The factorial action scores use frozen paired synthetic corruptions of those real surfels. Accordingly,
the former supports Actor-local canonical surfel fusion, while the latter verifies compiler semantics and hazard preservation; it
does not establish natural-artifact detection accuracy.

Hazard Actors have descriptive fused distance/recall `0.0703 m / 0.9664`, versus `0.1487 m / 0.8245` for safe Actors. This
all-row breakdown is not a gate and is not interpreted causally; closer hazard Actors may simply receive denser LiDAR support.

Resources: one RTX 3090, wall `43.19 s`, peak allocated GPU memory `0.0693 GiB`, peak RSS `1.418 GiB`, run size `78 MiB`.
No AV2 training, calibration, threshold search, failed-log deletion, mirrored Actor completion, hash, checksum, or fingerprint.
