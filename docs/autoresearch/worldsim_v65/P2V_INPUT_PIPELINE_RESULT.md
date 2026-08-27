# P2V Fresh Visited-State Input Pipeline Result

## Canonical inputs

- preparation: `run://worldsim_v65/WS-V65-P2V-FRESH-PREPARATION-01/20260827T123000Z__fresh-visited-prep-s0-r1`
- evidence: `run://worldsim_v65/WS-V65-P2V-FRESH-EVIDENCE-01/20260827T133000Z__fresh-visited-evidence-s0-r1`
- native aggregate: `run://worldsim_v65/WS-V65-P2V-FRESH-NATIVE-SIDECAR-01/20260827T133000Z__fresh-visited-native-aggregate-s0-r1`

## Preparation and overlap

Ten public gzip tar shards were read sequentially by ten shard workers. They extracted 10,705 previously unindexed sensor members in 4,108.35 seconds. Completed shards each released a complete scene, so the preparation parent alone was paused while the archive children continued; three two-scene preprocessing waves then ran before the parent performed its normal index merge and cleanup.

All six scenes completed, the parent reused those completed directories, and the temporary raw root was removed after success. No quality was read. This avoided waiting for global archive completion before preprocessing and allowed native inference for the first four scenes to finish while later scene conversion was still active.

## Native sidecars

| scene | targets | wall seconds | output bytes | peak GPU GiB |
| --- | ---: | ---: | ---: | ---: |
| scene-0001 | 12 | 56.84 | 552,980,730 | 4.1305 |
| scene-0219 | 12 | 56.68 | 552,980,764 | 4.1314 |
| scene-0402 | 12 | 44.39 | 552,980,701 | 4.1314 |
| scene-0594 | 12 | 53.70 | 552,980,737 | 4.1314 |
| scene-0822 | 12 | 55.04 | 552,980,740 | 4.1314 |
| scene-1110 | 12 | 48.16 | 552,980,774 | 4.1309 |

The aggregate contains 72 targets and 3,317,884,446 bytes. All native features are complete; inference was not repeated. Target evidence, calibration quality, confirmation content, and exact-once test content were not read.

## Evidence

The first 24 units for scenes 0001/0219 were built during later scene preprocessing. The final canonical 72-unit evidence run reused those 24 units by hardlink and computed the remaining 48 in 58.72 seconds. It contains 76,067,478 bytes, has zero source-role overlap, and passed.

## Entry recoveries

`V65-F13` and `V65-F14` occurred before any model or quality read: a missing task parent, then a launcher that bypassed the established base-plus-overlay resolver. Both are preserved in the failure ledger. The final six scene runs use the originally frozen r1 paths and unchanged scientific config.

The next operation is the single formal P2V Qmean prediction-object transfer read.
