# P10X combined-confirmation input pipeline result

Date: 2026-08-28  
Status: inputs ready; formal combined quality unread

## Preparation

Canonical run:

`run://worldsim_v65/WS-V65-P10X-CONFIRMATION-PREPARATION-01/20260828T010000Z__confirmation-prep-s0-r1`

Targeted public-tar shards 3/7/8 found all 10,718 required members: 3,533/3,589/3,596 members in approximately
1010.9/982.5/1014.7 seconds. The same-cohort full-ten-shard fallback was not used. Parent wall was 1564.1072 seconds;
it reused all six asynchronously preprocessed scenes and removed the temporary raw batch after success.

The scene-ready feeder overlapped the three archive scans with two preprocess workers. Per-scene preprocessing walls for
indices 194/229/538/563/634/657 were `160.52/151.46/175.19/159.90/199.32/196.81s`. The extractor's existing
`.partial.<pid>` to atomic rename delivery was reused without hashes, checksums, fingerprints, or content gates.

## Native and evidence

All six native scene runs passed with 12 targets each. Walls for `0245/0287/0686/0718/0817/0868` were
`74.11/80.51/49.38/47.15/57.61/57.41s`; the longer first pair overlapped the four-worker partial evidence job.
Maximum worker peak GPU memory was `4.1314GiB`.

Canonical native aggregate:

`run://worldsim_v65/WS-V65-P10X-CONFIRMATION-NATIVE-SIDECAR-01/20260828T010500Z__confirmation-native-aggregate-s0-r1`

- scenes/targets: 6/72
- output bytes: 3,317,884,470
- inference repeated: false
- target/calibration/confirmation/test read: false

The first four config-order scenes produced a 48-unit partial evidence run while the final two scenes were still being
preprocessed and inferred. Canonical evidence reused those 48 units and computed only the final 24:

`run://worldsim_v65/WS-V65-P10X-CONFIRMATION-EVIDENCE-01/20260828T010500Z__confirmation-evidence-s0-r1`

- scenes/units/reused: 6/72/48
- disk bytes: 81,763,088
- wall: 35.1403s
- source-role overlap: 0

`V65-F18` was limited to the first feeder import before run or input read; the process-local `PYTHONPATH=.` recovery did not
change any frozen scientific input. No further failure occurred. The next operation is the single frozen combined quality read.
