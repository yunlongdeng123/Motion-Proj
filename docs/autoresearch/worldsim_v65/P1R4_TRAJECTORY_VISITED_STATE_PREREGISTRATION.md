# V6.5 P1R4 Trajectory-Visited-State Preregistration

## Question

Given the frozen future Ego trajectory `tau` over 20 frames (2.0 s), can the system predict whether the world states
visited by its 1.5 m corridor are reliable? The supervised object is one continuous outcome per `(scene, unit, tau)`, not
one label per voxel.

## Data and leakage boundary

- Reuse only the already-consumed P1/R3 legacy train-only cache: first eight units per each of 16 scenes train, final
  four units nested evaluation.
- A unit is eligible when its observable trajectory footprint contains at least 16 sampled boundary states.
- Target: mean hidden-FREE outcome inside the frozen future trajectory footprint.
- `hidden_free` is never an input. The hard corridor defines the prediction object but is not a predictor feature.
- This does not consume a fresh V6.5 selection, calibration, confirmation, or test cohort.

## Arms

`Qagg` is the mean frozen q0 risk over visited states. `V1` is a `25 -> 32 -> 16 -> 1` sigmoid MLP over frozen q0
route-distribution statistics, observable footprint/global context, and the mean of the 14 preregistered R3 map/context
features inside the footprint. There is one seed, one capacity, and no sweep.

The cache reader pipelines the 14D map-context array load with GPU sigmoid computation for base q0 logits. No native
hidden tensor or source sidecar is reread.

## Frozen reads

Prediction-object viability is supported when all hold on nested evaluation:

- `Qagg` Spearman >= 0.30;
- unsafe-unit AUROC >= 0.65, where unsafe means at least one visited hidden-FREE outcome;
- selecting the lowest-risk 40% by `Qagg` reduces mean realized visited-state cost by >=25% versus all eligible units.

The learned head is incrementally supported only if Spearman gain >=0.03, MSE reduction >=10%, selected realized cost
reduction versus `Qagg` >=10%, scene lower > higher, and real features outperform a within-scene shuffled-trajectory
control. If viability passes but increment fails, retain the trajectory-level object with direct q0 aggregation and close
the learned residual. If viability fails, close this world-state target and move only to the already-supported continuous
Actor-route outcome object.

## Literature migration

Task-Relevant Failure Detection (CoRL 2022) propagates prediction errors into planning cost; PRECOG (ICCV 2019)
conditions other agents on the controlled agent goal. The migration here is deliberately smaller: change the outcome
and aggregation boundary first, without importing a new planner, recurrent world model, or large interaction network.

