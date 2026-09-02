# P18 Two-Expert Completion Router Freeze

Date: 2026-09-02

## Rationale

P17R is a useful but non-dominating expert: it reduces hazardous first returns while slightly worsening surface Chamfer. Following
two-stage learning-to-defer/multiple-expert routing, P18 freezes both always-COMPLETE and P17R and learns only which expert executes
for an Actor. It does not alter a completion score, threshold, point, KEEP/PROJECT output, or geometry tolerance.

## Router contract

Inputs are 13 existing runtime-validity features, 5 analytic hazard features, and 5 frozen P17R score statistics (completion
fraction, mean, standard deviation, minimum, maximum), all available without held-out target rays. A `23-32-2` ReLU router with
seed `71801` trains for 120 epochs, batch 32, AdamW `.001/.0001`, with inverse-square-root source-frequency cross-entropy.

The source label has no scalar utility weight: select P17R only if its Actor Chamfer and new-early count are both no worse than
always-COMPLETE and at least one is strict; otherwise label always-COMPLETE. Inference is argmax with no threshold.

## Evidence boundary

Fit uses nuScenes train+calibration. The prior nuScenes test role is already consumed by P17/P17R; P18 may use it only as a
prospective-router development gate, not a new independent source confirmation. Both aggregate Pareto directions passing freeze the
router and authorize exactly one read of the still-unread 10-log AV2 cohort after download. Failure closes routing without feature,
loss, threshold, seed, or expert sweep. Only AV2 can support a new transfer claim; no formal L2D or road-safety guarantee is made.
