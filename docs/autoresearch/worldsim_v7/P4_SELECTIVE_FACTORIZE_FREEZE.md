# WorldSim V7 P4 selective repair certificate freeze

Date: 2026-09-02  
Task: `WS-V7-P4-NUSCENES-SELECTIVE-FACTORIZE-01`  
Hypothesis: `WS-V7-H-P4-001`  
Status: frozen before formal method-quality read

## Question

Can a nuScenes-only, low-capacity validity model identify Actors for which the frozen
ray-certified four-action compiler is no worse than the untouched clean query, while an
input-separated hazard head retains analytic hazard prediction and does not leak task risk
into repair authority?

## Frozen data boundary

- Train: 11 complete nuScenes train-split scenes, 440 keyframes.
- Calibration: 14 disjoint nuScenes train-split scenes, 561 keyframes.
- Test: 38 disjoint nuScenes val-split scenes, 1,529 keyframes.
- The three planned development scenes `scene-0230/0242/0255` are excluded only because
  their raw LIDAR files are absent locally. No method output or quality was read for this choice.
- External test: all 30 previously frozen AV2 Sensor val logs. AV2 is never used for training,
  standardization, threshold fitting, model selection, or recovery.
- Waymo is deferred because no frozen local Waymo sensor corpus is available; P4 deepens the
  already downloaded AV2 external-domain evidence instead of adding an under-supported adapter.

## Frozen target and interface

- Clean fallback is the original single-frame query, without synthetic ghost/duplicate/flicker.
- Per-Actor repairability label is
  `Chamfer(compiled,target) <= Chamfer(clean-query,target)`.
- Validity input contains only runtime-visible surface/ray/provenance/support features.
- Hazard input contains only TTC, clearance, closing speed, hard-brake, and crossing geometry.
- Factorized model has structurally independent validity and hazard encoders. The shared-input
  two-head model is the sole learned baseline.
- At inference, accepted Actors use the compiled surface; abstained Actors retain the clean query.
  Actor identity, trajectory, size, and hazard label remain immutable.

## Training and risk boundary

- One fixed seed (`70401`), 32 hidden units, 80 epochs; no architecture, seed, epoch, or threshold sweep.
- Feature standardizers fit on nuScenes train only.
- A monotone false-repair threshold is selected on nuScenes calibration only at `alpha=.05`,
  using the finite-sample adjusted empirical loss `(selected failures + 1)/(n + 1)`.
- The finite-sample interpretation requires nuScenes calibration/test exchangeability. The
  unknown nuScenes-to-AV2 shift breaks that premise, so AV2 receives only descriptive zero-shot
  risk--coverage and geometry metrics, never a formal conformal or road-safety guarantee.

## Decision rule

The candidate is supported only if the factorized model is within `.02` AUROC of the shared
baseline on both nuScenes test tasks, both cross-input score shifts are `<=1e-8`, AV2 coverage is
at least `.10`, AV2 false repairs are fewer than always-repair failures, selective AV2 mean
Chamfer is no worse than the clean query, and Actor/hazard retention is exact. These seven tests
are the complete scientific decision surface; no extra smoke/regression matrix is added.

## Implementation and claim limits

- `motion_proj/worldsim_v7/nuscenes_actor_surface.py` streams nuScenes metadata and transforms
  official LIDAR_TOP returns sensor-to-ego-to-global, then global-to-Actor local coordinates.
- `motion_proj/worldsim_v7/selective_validity_hazard.py` implements the two fixed low-capacity baselines.
- `scripts/run_worldsim_v7_p4_selective_factorization.py` performs one nuScenes build/train/calibration/test
  pass and then freezes the model before the single AV2 zero-shot pass.
- No hash, checksum, fingerprint, fine-tuning, target-conditioned action, planner, policy,
  closed-loop, collision, or population-safety claim is introduced.
