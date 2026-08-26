# P5 fit-only supervised hidden-FREE risk freeze

- Task: `WS-V64-P5-SUPERVISED-RISK-01`
- Hypothesis: `WS-V64-H-P5-001`
- Formal run: `20260826T093000Z__supervised-risk-s0-r1`
- Evaluation score read at freeze: `false`

## Motivation and migration

P4N's feature-density U2 passed relative gates but produced within-scene AUROC `0.498387/0.498295` and FPR@95TPR near `0.96`. This blocks authority and calibration. It also makes another GMM/PCA/seed sweep scientifically invalid on the same evaluation scenes.

The recovery follows the supervised/hybrid direction of three relevant methods: OCCUQ combines a learned dense uncertainty head with a separate feature GMM; ReliOcc proposes hybrid voxel uncertainty with explicit training-time uncertainty construction; EvOcc explicitly supervises evidence for unobserved and contradicting occupancy. The migration here is intentionally smaller: a single linear hidden-FREE risk head trained only on the existing four fit scenes.

References:

- OCCUQ official repository: <https://github.com/ika-rwth-aachen/OCCUQ>
- ReliOcc official IJCAI 2025 entry: <https://doi.org/10.24963/ijcai.2025/220>
- EvOcc CVPR 2025 paper: <https://openaccess.thecvf.com/content/CVPR2025/papers/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.pdf>

## Frozen representation and head

- Denominator, scenes, targets, and sampling are identical to P4N.
- Fit: `scene-0139, scene-0230, scene-0255, scene-0994`, exactly `50,000` points per scene and seed `0`.
- Evaluation: `scene-0359, scene-0998`; their labels are not used by fit.
- Representation: load P4N r2's fitted StandardScaler and PCA-16; no refit and no scene ID.
- Head: scikit-learn logistic regression, `C=1.0`, `class_weight=balanced`, `solver=lbfgs`, `max_iter=200`, `random_state=0`.
- Comparator: reproduce U2 and U0 on the identical evaluation points in the same pass.

## Frozen decision

Only two absolute gates are used:

1. pooled U3 AUROC `>= 0.60`;
2. U3 AUROC `>= 0.55` in both evaluation scenes.

Both must pass. All other metrics are report-only. No coefficient/regularization/feature/seed/denominator/gate sweep, no extra split, and no repeat run is allowed after evaluation quality is read. Passing supports only a supervised ranking mechanism; it does not establish calibrated probability, authority, conditional coverage, or safety. Failure closes this fixed linear/PCA-16 recovery on the current cohort.

Execution is CPU-only and expected to fit within the existing single-3090 machine; no multi-GPU resource is needed.

