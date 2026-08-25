# WorldSim V6.3 P6 Surface-Family Closeout

- Task: `WS-V63-P6-DEVELOPMENT-AB-01`
- Terminal status: `H-P6-001 rejected; surface architecture family closed by Stop 2`
- Quality boundary: P6 selection read; no threshold fit, legacy, calibration, confirmation or exact-once test read
- Downstream status: B4/B5/M0 not executed; P7–P11 locked

## Frozen evidence

| Stage | Canonical run | Result |
|---|---|---|
| B0/B1/B2 | `run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260825T151200Z__baselines-s0-r1` | Native B2 comparator frozen |
| B3 mean train | `run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260825T152000Z__b3-mean-s0-r1` | epoch 1 training candidate frozen |
| B3 common eval | `run://worldsim_v63/WS-V63-P6-DEVELOPMENT-AB-01/20260826T014500Z__b3-eval-s0-r1` | stage failed, supporting scenes `0/2` |

B3 training completed five epochs and 1280 optimizer steps in `9181.220 s`, with finite gradients, zero hard violations and
`0.400373 GiB` peak GPU memory. Epoch 1 was the only best feasible training checkpoint: safe-OCC retention `0.636863`, emitted-OCC
coverage `0.285326`, UNKNOWN `0.550411`, hidden-FREE tail `0.608174` and matched rank `0.080258`. Epochs with lower tail but failed
coverage/retention/UNKNOWN gates did not replace it.

## Common matched evaluation

| Metric | Native B2 | B3 | B3 relative/result |
|---|---:|---:|---:|
| Pooled common surface CVaR | 0.491496 | 0.608174 | -23.739% |
| Pooled emitted OCC points | 2,298,450 | 1,047,186 | area ratio 0.455605 |
| Pooled proposal false-safe surrogate | 0.396840 | 0.515384 | -29.872% |
| Pooled safe-OCC retention | 0.851056 | 0.636863 | pass absolute gate only |
| Pooled source-valid UNKNOWN | 0.266284 | 0.554227 | pass pooled gate only |

| Scene | B2 tail | B3 tail | Relative improvement | Area ratio | Retention | Source-valid UNKNOWN | Support |
|---|---:|---:|---:|---:|---:|---:|---|
| scene-0450 | 0.497850 | 0.596685 | -19.852% | 0.406270 | 0.601623 | 0.651678 | no |
| scene-1089 | 0.465122 | 0.655861 | -41.008% | 0.499323 | 0.704815 | 0.445030 | no |

Both scenes retained zero hard violations, full case coverage and nonzero actor/static coverage. The rejection is therefore not a hard
projection regression or an all-UNKNOWN loophole. It is a matched architecture failure: after preserving useful OCC emission, B3 still
increased hidden-FREE tail risk and emitted less than half the Native B2 area in each scene.

## Verdict and locks

`WS-V63-H-P6-001` is rejected. Main-plan Stop 2 closes the surface architecture family, so Surface-Max, Surface-CVaR and the M0 authority
arm are not executed. `WS-V63-H-P6-002` and `WS-V63-H-P6-003` close without quality read. P7 cannot start because P6 did not freeze a
promotable M0. No V6.3 seed, model-size, alpha, epoch, threshold, gate or downstream-data recovery is authorized.

## Report-writing boundary

The P5 epoch-3 artifact is only the best scalar training-objective checkpoint and is not a candidate because its safe-OCC retention is
zero. P5R epoch 6 is a promotable training-side candidate that legally unlocked P6, not a final method result. B3 epoch 1 is likewise only
a feasible B3 training checkpoint: the common evaluator rejected it on both scenes. V6.3 therefore has no version-level best SurfNCC
candidate.

For the technical report, pooled values are descriptive and the frozen scene-level gate determines the hypothesis verdict. B4/B5/M0 and
P7--P11 must be reported as not executed/locked, not as rejected arms. Legacy28 quality in P6, calibration, confirmation and exact-once
test were never read, so this closeout supports no calibrated risk bound, held-out confirmation, deployment, or real-world safety claim.
The complete report evidence map is `ARXIV_EVIDENCE_INDEX.md`.

## Future-only migration audit

The terminal search points to a new-version direction rather than a V6.3 recovery:

- [EvOcc (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html)
  suggests evidential targets that explicitly retain unobserved and contradictory uncertainty.
- [ReliOcc (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/220) and the open-source
  [OCCUQ (ICRA 2025)](https://github.com/ika-rwth-aachen/OCCUQ) suggest hybrid and feature-level aleatoric/epistemic uncertainty rather
  than relying on a surface state decoder alone.
- [End-to-end Conditional Robust Optimization (UAI 2024)](https://proceedings.mlr.press/v244/chenreddy24a.html) motivates optimizing
  conditional coverage together with decision risk; this directly targets the observed pooled-versus-scene coverage mismatch.

A legal continuation requires a new hypothesis/version, fresh development scenes, a frozen uncertainty interface and scene/stratum-
conditional coverage constraints. These sources do not justify changing or extending the completed V6.3 experiment.
