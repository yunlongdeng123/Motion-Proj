# P5 fit-only supervised hidden-FREE risk closeout

- Task: `WS-V64-P5-SUPERVISED-RISK-01`
- Hypothesis: `WS-V64-H-P5-001`
- Canonical run: `run://worldsim_v64/WS-V64-P5-SUPERVISED-RISK-01/20260826T093000Z__supervised-risk-s0-r1`
- Verdict: `supported_ranking_only`

## Result

The fixed logistic head trained on 200,000 fit points, including 18,242 hidden-FREE positives (prevalence `0.091210`). It evaluated the identical P4N denominator of 333,009 points and 27,495 positives.

| Scope | U2 AUROC | U3 AUROC | U3 AUPRC | U3 FPR@95TPR | U3 risk at 50% coverage |
|---|---:|---:|---:|---:|---:|
| pooled | 0.518545 | 0.658118 | 0.148720 | 0.867738 | 0.049098 |
| scene-0359 | 0.498387 | 0.640682 | 0.171993 | 0.859069 | 0.069888 |
| scene-0998 | 0.498295 | 0.636266 | 0.102831 | 0.907021 | 0.036493 |

Both frozen gates passed: pooled AUROC is `0.658118 >= 0.60`, and both scene AUROCs exceed `0.55`. U3 improves pooled AUROC over U2 by `+0.139573`; pooled AUPRC rises from `0.085650` to `0.148720`. At 50% coverage, pooled hidden-FREE risk falls from the prevalence `0.082565` and U2's `0.076917` to `0.049098`.

The CPU-only run completed in `17.3115 s`, with peak RSS `0.8592 GiB` and a `80 KiB` run leaf. No multi-GPU resource was required.

## Boundary and next scientific question

The result supports fit-only supervised ranking transfer on two fresh scenes. It does not establish calibrated probabilities, authority, conditional coverage, or safety. FPR@95TPR remains `0.867738` pooled and reaches `0.907021` on scene-0998, so a low-FPR authority claim is explicitly blocked as `V64-F11`.

The ICLR 2024 Conformal Risk Control method and its official implementation provide the appropriate next abstraction: choose a monotone selective set on independent calibration units to control an expected bounded risk, while preserving a separate untouched confirmation set. Current evaluation labels must not be reused to choose a threshold. The next stage therefore starts with metadata-only selection of new, scene-disjoint calibration and confirmation scenes; it does not scan the current evaluation scores.

- ICLR 2024 proceedings: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html>
- Official implementation: <https://github.com/aangelopoulos/conformal-risk>

