# P1R6 Smooth-Tail Visited-State Preregistration

## Question

R4 established that the prediction object should be the reliability of world states actually visited by a future Ego trajectory, rather than correctness of an arbitrary voxel. This train-only diagnostic asks one narrower question: does a differentiable upper-tail aggregation improve trajectory admission over the retained arithmetic mean Qagg?

## Frozen input and target

- Reuse `/root/autodl-tmp/cache/worldsim_v65/p1r3_map_context_compact.npz`; no native hidden or new sidecar read.
- Keep the R4 future horizon at 20 frames / 2.0 seconds, the Ego corridor radius at 1.5 m, and eligibility at at least 16 visited samples.
- Keep the continuous target `visited_hidden_free_fraction` and the binary unsafe event `hidden_free_count > 0`.
- Evaluate only the existing P1 nested-evaluation units. This is not a formal V6.5 selection read and cannot change the already frozen P2V candidate.

## Arms

1. `Qmean`: arithmetic mean of frozen q0 risk over states visited by the trajectory.
2. `Qsoft-tail`: `sum softmax(q0 / 0.10) * q0` over the same visited states.

The probability-scale temperature is fixed once at `0.10`. There is no temperature, seed, horizon, corridor, threshold, or capacity sweep and no learned head.

The migration is motivated by risk-sensitive planning and soft robust risk aggregation: a smooth tail can emphasize a locally unreliable part of a path while remaining differentiable. It is deliberately tested against mean aggregation rather than introduced as a post-hoc rescue.

References: Filos et al., [Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?](https://proceedings.mlr.press/v119/filos20a.html); Wang and Chapman, [Adaptive and Risk-Aware Target Tracking for Robot Teams with Heterogeneous Sensors](https://proceedings.mlr.press/v144/wang21b.html).

## Gates

At matched lowest-risk 40% coverage, all of the following must hold:

- realized visited-state cost is at least 10% lower than Qmean;
- unsafe AUROC does not decrease;
- Spearman decreases by no more than 0.02;
- more evaluation scenes improve than worsen.

If any gate fails, retain Qmean, register one negative result, and close smooth-tail aggregation without a temperature or threshold rescue.
