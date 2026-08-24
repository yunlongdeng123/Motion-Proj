# P4 frozen IR-WM sidecar interface

Date: 2026-08-24  
Task: `WS-V62-P4-IRWM-PRIOR-SIDECAR-01`  
Status: `probe_ready_no_gpu_run`

## Frozen source

P4 reuses the V6.1 capability-passed IR-WM environment, official source, fully-decoupled checkpoint, image normalization,
camera ordering, and two-history-plus-current forward. IR-WM stays in evaluation mode and is never trained. P4 reads camera
images, calibration, ego-motion metadata, and four coordinate arrays from each P2 query archive. It does not read target
evidence, occupancy ground truth, O_method/O_eval, confirmation, or exact-once test content.

Per user constraint, this stage adds no hash, checksum, fingerprint, content addressing, model-before/after hash, repeated
build, or new audit gate. Identity is the logical source path, backend semantic identity, task/run ID, branch Git commit, and
sidecar schema version.

## Native tensors

The official current-state path exposes:

- final decoder occupancy logits: `200 x 200 x 16 x 17`;
- current reference BEV latent: `200 x 200 x 256`;
- source grid origin `[-51.2, -51.2, -5.0] m` and voxel size `0.512 m`.

P2 query centers are mapped directly into that source grid. Queries above or outside the IR-WM source extent receive an
explicit invalid prior mapping; the adapter does not extrapolate a class or call invalid space FREE.

## Deduplicated query alignment

Each target sidecar stores:

```text
query_to_prior_cell -> unique 3D source cell -> 17 FP16 logits
query_to_bev_cell   -> unique 2D BEV cell    -> 256 FP16 latent
query_source_valid
query/source coordinate metadata
input frames, metadata indices, target LiDAR pose
```

This is exactly recoverable at every frozen P2 query coordinate while avoiding repeated copies for replacement-sampled
queries, multiple fine-grid queries mapping to one 0.512 m source cell, and all 16 heights sharing one BEV feature.

## Probe and formal sequence

The only capability probe is `scene-0071/f017`, frames `[7,12,17]`, metadata indices `[1,2,3]`, batch1, one worker. It must
produce finite, nonempty logits and latent mappings below the 22 GiB ceiling without target evidence. If it passes, P4 runs
all six scenes and 72 targets with at most two scene workers; no additional smoke stage is inserted.
