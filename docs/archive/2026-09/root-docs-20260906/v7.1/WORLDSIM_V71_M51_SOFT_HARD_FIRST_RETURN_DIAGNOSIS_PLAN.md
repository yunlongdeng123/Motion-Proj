# WorldSim V7.1 M51 — smooth/hard first-return non-implication diagnosis

Date: 2026-09-05

## Question

M50 lowers its differentiable first-return loss but worsens both frame-wise and aggregate literal earliest-return metrics.
M51 asks whether this is an operator-level surrogate mismatch or only the later full-anchor/voxel deployment realization.
It does not train, choose, or rescue a model.

## Exact boundary

The smooth renderer returns an alpha-composited expected depth, including fallback survival. The literal evaluator returns the
minimum positive depth inside a fixed beam tube. There is no monotone implication from lower expected-depth error to a safer
minimum: an arbitrarily small shallow component can change the minimum discontinuously while changing the expectation only
slightly, or even toward the target when other mass is late. A guarantee requires an explicit support margin, not only a mean
depth loss.

## Frozen paired read

- models: frozen M8 and rejected M50 on the same 66 exposed holdout Actors;
- same-support arm: 512 capped anchors plus all children, without voxelization, rendered by both smooth alpha composition and
  literal beam-tube minimum;
- deployment arm: full anchors plus children after the frozen 0.06 m voxel realization;
- report smooth absolute-error change, added/removed hard-early rays, the fraction of added-early rays whose smooth error
  improved, smooth/hard depth-delta correlations, and train-support versus deployment-event flips;
- strata: all, hazardous, clear, moving, quasi-static;
- no target-dependent filter, gate, model selection, threshold change, training, or partial M43 read.

This diagnosis may explain `V71-F49`; it cannot upgrade M50 or create a new external candidate.

## Outcome

Confirmed on 99,208 rays. Same-support hard early rises by 0.490 points; deployment hard early rises by 0.545 points.
Among newly early rays, 38.03% (same support) and 36.54% (deployment) nevertheless improve smooth absolute depth error.
Smooth-error delta versus hard-train depth delta has correlation 0.038, while full-anchor/voxel realization flips 1.280%
of events. The non-implication therefore exists before deployment discretization, with an additional support-realization term.
M51 closes `V71-F49` as an explained negative result and is never converted into a selector.
