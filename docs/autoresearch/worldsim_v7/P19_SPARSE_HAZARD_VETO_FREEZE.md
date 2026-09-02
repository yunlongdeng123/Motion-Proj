# P19 Sparse Hazard Veto Freeze

Date: 2026-09-02

## Rationale

P18 shows that Actor-level P17R dominance is a `3/85` source-fit rare event and a `2/228` consumed-test rare event. A learned router
therefore collapses to always-COMPLETE even though a two-Actor oracle has a tiny valid Pareto gain. SparseOcc (CVPR 2024) reports
that sparse representations can avoid empty-space hallucination, while EvOcc (CVPR 2025) treats unobserved/conflicting occupancy as
UNKNOWN and evaluates first-return rays. P19 transfers those principles as a minimal, fixed-capacity action rather than another model.

## Frozen policy

For a hazardous Actor only, inspect the frozen P17R candidate scores. If at least one score is strictly below P17R's already-frozen
`.5` selection threshold, mark exactly the minimum-score candidate UNKNOWN and keep every other completion candidate OCCUPIED.
Ties use first candidate order. Clear Actors remain always-COMPLETE. P19 trains no parameters, changes neither candidate coordinates
nor KEEP/PROJECT points, and cannot veto more than one candidate per hazardous Actor.

## Evidence boundary

The consumed nuScenes test role is only a prospective source-development gate. Hazard new-early must strictly decrease and population
mean Chamfer must not exceed always-COMPLETE. Failure registers `V7-F28` and closes score-ranked candidate vetoes without sweeping
capacity, threshold, strata, or ranking. Passing freezes the policy and permits exactly one read of the still-unread 10-log AV2
cohort after download; only that external result can support transfer. No formal road-safety guarantee is claimed.

References: [SparseOcc](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_SparseOcc_Rethinking_Sparse_Latent_Representation_for_Vision-Based_Semantic_Occupancy_Prediction_CVPR_2024_paper.html),
[EvOcc](https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html).
