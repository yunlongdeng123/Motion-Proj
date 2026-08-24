# P2 formal evidence-query dataset

Date: 2026-08-24  
Task: `WS-V62-P2-EVIDENCE-QUERY-DATASET-01`  
Outcome: `done_formal_evidence_query_dataset_passed`

## Canonical run

```text
run://worldsim_v62/WS-V62-P2-EVIDENCE-QUERY-DATASET-01/20260824T083654Z__query-dataset-s20260824-r2
```

Source implementation: `fc5a5f7` on `research/worldsim-v6.2-cpsc`.

## Frozen denominator

- Scenes: `scene-0071`, `scene-0317`, `scene-0450`, `scene-0862`, `scene-1012`, `scene-1089`.
- Targets: 12 per scene, 72 total.
- Queries: 100,000 per target, 7,200,000 total.
- Query totals: hard FREE 1,800,000; hard OCC 1,080,000; behind-hit UNKNOWN 1,800,000; evidence boundary
  1,080,000; actor envelope 1,080,000; contradiction 360,000.
- Source roles: 216 method sweeps, 72 held-out dropout sweeps, 288 target-evidence sweeps; cross-role overlap 0.
- Confirmation and exact-once test remained unread.

Minimum candidate-pool sizes across the 72 units were 156,406 hard FREE, 6,860 hard OCC, 6,533 behind-hit UNKNOWN,
382,175 boundary, 167 actor envelope, and 2,446 contradiction. All frozen query types therefore retained a nonempty
denominator. Sparse pools below their quota use the pre-registered unit-local sampling with replacement; no unit or query type
was deleted.

## Actor and target supervision

One unit had no instantaneous actor envelope in the current ROI. Visible method-sweep support resolved it, yielding zero empty
combined actor pools across all 72 units. The dataset contains 1,383,331 actor-bound rows and 103,946 motion-compensated actor
hit voxels. Swept actor boxes remain query support rather than hard occupied evidence.

Target-evidence supervision covers 2,639,153 query rows in total, ranging from 30,254 to 45,396 per unit. The target evidence
is stored separately from method evidence and is not available to the P4 prior extractor.

## Artifacts and resources

Top-level artifacts:

```text
METHOD_EVIDENCE.jsonl
DROPOUT_TARGETS.jsonl
TARGET_EVIDENCE.jsonl
QUERY_MANIFEST.jsonl
SPLIT_MANIFEST.json
P2_SUMMARY.json
```

The dataset occupies 155,249,746 bytes. Materialization used two CPU workers, completed in 151.469 seconds, and had a maximum
unit wall time of 8.772 seconds. GPU remained idle. Per user constraint, no hash, checksum, fingerprint, repeated byte-exact
build, or additional quality gate was added.

Decision: close P2 and start `WS-V62-P4-IRWM-PRIOR-SIDECAR-01` with frozen V6.1 IR-WM weights and no target-evidence read.
