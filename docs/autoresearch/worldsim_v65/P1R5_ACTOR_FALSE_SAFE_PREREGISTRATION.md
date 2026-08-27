# V6.5 P1R5 Actor False-Safe Preregistration

## Question

Given the Ego trajectory, is the future Actor-route interaction forecast reliable, and can frozen temporal disagreement
identify false-safe cases where realized Actor proximity cost exceeds the snapshot forecast?

## Frozen construction

- Reuse the P2C legacy 4-scene train / 2-scene nested-evaluation cache and its already-frozen A0 snapshot and A1
  Actor-time models. No model is refit.
- Score all Actor tokens on GPU, then reduce each `(scene, unit, trajectory)` by maximum proximity cost.
- Forecast target: maximum realized Actor-route proximity cost.
- Reliability target: `max(realized target - A0 forecast, 0)`.
- Deterministic monitor: `max(A1 forecast - A0 forecast, 0)`. It asks whether method-visible temporal evidence raises
  risk above the strong snapshot forecast; no outcome label enters the monitor.

## Frozen reads

The Actor-route prediction object is viable when A0-vs-target Spearman >=0.70 and its lowest-risk 40% reduces realized
target cost by >=25% versus all evaluation trajectories.

The false-safe monitor is supported only when disagreement-vs-gap Spearman >=0.30, positive-gap AUROC >=0.65, and the
lowest-monitor 40% reduces realized false-safe gap by >=25% versus all. Failure closes this monitor without threshold,
learned-head, seed, or model rescue; it does not erase the independently measured Actor-route forecast viability.

## Boundary

This is a legacy train-only diagnostic on 72 trajectories. It makes no fresh V6.5 selection, planning, calibration, or
safety claim and does not reopen the rejected A1 target-cost model family.

