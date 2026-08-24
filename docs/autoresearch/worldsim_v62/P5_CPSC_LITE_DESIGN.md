# P5 CPSC-Lite design freeze

Date: 2026-08-24  
Task: `WS-V62-P5-CPSC-LITE-TRAIN-01`  
Status: `capacity_probe_ready`

## Development split

The split is scene-disjoint and frozen from metadata before P5 training:

- train: `scene-0071`, `scene-0317`, `scene-0862`, `scene-1012` (48 units);
- selection: `scene-0450`, `scene-1089` (24 units).

Training covers Boston day and Singapore day/dusk/night. Selection covers Boston rain and Singapore Holland Village night. No
model, occupancy, proposal, or target-quality result was used to choose the split.

## Feature boundary

Model inputs are P4's 17 semantic logits, prior entropy, tri-state prior, source-valid flag, 256-dimensional selected BEV
latent, normalized query coordinates, P2 method-evidence one-hot state, method contradiction, actor bound/current/swept
support, and prior-minus-method residual.

Query type remains available only for stratified metrics. The held-out dropout state and target-evidence state are supervision
only and never model inputs. Target evidence conflicts with hard method evidence are excluded from the query loss instead of
asking the network to violate the method-time hard constraint.

## Network

```text
prior adapter + query adapter
-> four 256-wide MLP layers
-> two residual blocks
-> three-state evidential head
-> trust-scaled residual update
-> exact hard projection, repeated three times
```

The trust head scales only a learned residual. It cannot disable observed FREE/OCC or contradiction-to-UNKNOWN projection.
IR-WM is frozen, absent from the training process, and represented only by P4 sidecars.

## Loss and selection

Fixed weights for query/evidential/hidden-FREE/safe-OCC/actor-temporal/prior-preserve are
`1.0/0.05/2.0/1.5/0.25/0.05`; class weights are FREE/OCC/UNKNOWN=`1.0/1.5/0.5`. Early stopping uses only the weighted total
loss on the two selection scenes. The report also compares the learned model with projection-only prior behavior and records
hidden-FREE false-OCC rate, safe-OCC retention, UNKNOWN fraction, class fractions, and exact hard-constraint violations.

## Training and resource contract

The single bounded configuration is seed0, FP16, 16,384 queries per batch, accumulation2, AdamW at `3e-4`, maximum12 epochs,
minimum4 epochs, and patience3. Peak memory must remain at or below18 GiB, wall below12 hours, and incremental disk below20
GiB.

To honor the user's minimal-validation constraint, the plan's three-seed smoke suggestion is not used. One 8-optimizer-step
capacity probe on one train and one selection unit checks loader/forward/backward/resource feasibility. If it passes, the next
run is the full 48/24-unit training configuration; no smoke matrix or regression suite is inserted.

Legacy O_eval, confirmation, and exact-once test content remain unread. No hash, checksum, fingerprint, or content-addressing
machinery is introduced.
