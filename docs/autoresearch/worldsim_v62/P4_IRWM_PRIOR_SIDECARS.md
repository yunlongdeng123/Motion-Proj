# P4 formal frozen IR-WM prior sidecars

Date: 2026-08-24  
Task: `WS-V62-P4-IRWM-PRIOR-SIDECAR-01`  
Outcome: `done_formal_prior_sidecars_passed`

## Canonical run

```text
run://worldsim_v62/WS-V62-P4-IRWM-PRIOR-SIDECAR-01/20260824T090444Z__prior-sidecars-s1-r1
```

Source implementation: `ec68ced`.

## Denominator and coverage

All six frozen development scenes and all 72 P2 targets completed, yielding one sidecar for every 100,000-query unit. Of
7,200,000 queries, 6,811,702 (94.60697%) fall inside the native IR-WM source grid and 388,298 are explicitly prior-invalid.
The per-unit valid count ranges from 91,305 to 97,434.

Deduplication retains 23,129 to 38,500 unique 3D prior cells per unit and 4,973 to 10,364 unique 2D BEV cells per unit. No
unit has an empty logit or latent mapping. Invalid queries are not extrapolated and can still be governed by hard evidence and
the UNKNOWN state in CPSC.

## Resources

- 72 sidecars: 368,162,079 bytes.
- Sum of 72 official current-state forwards: 119.406 seconds.
- Per-forward time: 1.036 to 2.213 seconds.
- Formal controller wall: 176.271 seconds.
- Maximum single-worker GPU peak: 4.1265 GiB.
- Two-worker peak-sum upper bound: 8.2523 GiB.

Each scene worker loaded the model once, processed 12 targets, and exited. IR-WM therefore does not remain resident during P5.

## Model and information boundary

All six workers reported zero unexpected model keys. Each reported the same two officially deleted
`pts_bbox_head.transformer.reference_points` keys already resolved in V6.1; current BEV and occupancy extraction does not call
that detector decoder path.

The workers read camera images, calibration, ego-motion metadata, and only `query_indices`, `grid_origin_m`, `voxel_size_m`,
and `grid_shape` from the P2 query archives. Target evidence, occupancy ground truth, O_method/O_eval, confirmation, and
exact-once test content remained unread. IR-WM training, future decoding, and planning were not started.

Artifacts use logical path, semantic version, backend identity, task/run, and Git identity. No hash, checksum, fingerprint,
content addressing, repeated build, or additional quality gate was added.

Decision: close P4 and start `WS-V62-P5-CPSC-LITE-TRAIN-01` using only the frozen P2 evidence/query dataset and these P4
sidecars.
