# WorldSim V7.1 M31 — Anchor Contradiction Attribution Plan

状态：`frozen / one diagnostic read / no training`  
任务：`WS-V71-M31-ANCHOR-CONTRADICTION-ATTRIBUTION-01`

## 1. Why this experiment exists

M30 found a `19.58%` held-out free-before-hit contradiction rate before learned completion is added: the
nominally known surface already contains immutable anchors that return too early. Continuing to tune a completion
head cannot identify or repair that upstream supervision error. M31 therefore re-materializes the exact 66 exposed
development Actors and attributes literal first returns to the two existing anchor construction paths:

1. `KEEP`: points from the first held-out query frame that survive the compiler's evidence rule;
2. `PROJECT`: perturbed query probes projected back to supported canonical geometry;
3. their ordered union, which is the `anchors` tensor consumed by V7.1 training.

This is diagnosis of the GT/supervision producer, not a post-hoc filter and not evidence for a new model.

## 2. Frozen protocol

- Cohort: the same sorted first-1,024 corpus eligibility and every-tenth holdout rule used by M30; expected 66 Actors.
- Recovery: compile only their source scenes with the already frozen S2 source split, Actor rules and P2 compiler.
- Operator: shared literal minimum positive depth inside a `0.20m` beam tube; early/hit tolerance `0.20m`.
- Surfaces: raw query, rejected/unknown query, KEEP-only, PROJECT-only and ordered `KEEP+PROJECT` anchors.
- Attribution: use the union first-return index, not nearest-point assignment, to label each early return KEEP or PROJECT.
- Temporal view: preserve target-ray order and group repeated sensor origins by first appearance to report later-frame ordinal.
- Strata: all, hazard/clear, moving/quasi-static (`0.50m` trajectory displacement), and source category.
- Outputs: per-Actor JSONL, aggregate JSON, point/ray counts, early/hit/observable rates, union early provenance,
  and target-frame-ordinal rates.

No learned checkpoint is loaded, no parameter is fitted, and no surface point is removed from the measurements.
There is no threshold, seed, source, or evaluator sweep. Missing source Actors, if any, are reported rather than
substituted.

## 3. Interpretation contract

- KEEP-dominant contradiction means current query endpoint selection/box-frame supervision is itself inconsistent.
- PROJECT-dominant contradiction means cross-frame canonical projection introduces the contradiction.
- Increasing early rate at later target ordinals, especially for moving Actors, supports temporal pose/rigidity label noise.
- Similar rates across provenance/time imply the error is more likely shared pose/box annotation or the beam-tube point
  representation rather than completion.

All conclusions are descriptive on a pretrained-exposed development cohort. M31 cannot establish causal blame,
GT completeness, corrected geometry, external generalization, collision correctness, or safety.

## 4. Literature-guided response boundary

- [SCPNet (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Xia_SCPNet_Semantic_Scene_Completion_on_Point_Cloud_CVPR_2023_paper.html)
  identifies moving-object traces as completion-label noise and rectifies labels before learning.
- [DualAD (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Doll_DualAD_Disentangling_the_Dynamic_and_Static_World_for_End-to-End_Driving_CVPR_2024_paper.html)
  separates dynamic and static representations and explicitly compensates ego/object motion.
- [SCOOP (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Lang_SCOOP_Self-Supervised_Correspondence_and_Optimization-Based_Scene_Flow_CVPR_2023_paper.html)
  motivates correspondence-space motion diagnosis instead of unconstrained free-space regression.

M31 adopts only the upstream provenance question. Any later rectification must become a separately frozen supervision
construction followed by training; M31 itself must not turn these findings into inference-time deletion.
