# P10V fixed-action input pipeline result

Date: 2026-08-28  
Status: inputs ready; formal action quality unread

## Preparation

Canonical run:

`run://worldsim_v65/WS-V65-P10V-ACTION-PREPARATION-01/20260828T001000Z__action-prep-s0-r1`

Targeted public-tar shards 2/6/9 found all 10,709 required members. Their scan walls were approximately
1011.8/936.3/1069.6 seconds. The same-cohort full-ten-shard fallback was not used. Parent wall was 1462.0184 seconds,
and temporary raw was removed after all six processed scenes were complete.

The archive extractor already writes `.partial.<pid>` and atomically renames only complete files. A scene-ready feeder therefore
launched each scene as soon as every metadata-required target existed, without hashes, checksums, fingerprints, or content gates.
Per-scene preprocessing walls for indices 119/143/459/479/722/745 were
`161.16/135.51/141.27/154.00/190.33/197.18s`.

## Native and evidence

All six native runs passed with 12 targets per scene. Walls for `0159/0184/0577/0599/0955/0983` were
`48.71/45.81/51.14/52.43/45.43/52.69s`; maximum worker peak GPU memory was `4.1314GiB`.

Canonical native aggregate:

`run://worldsim_v65/WS-V65-P10V-ACTION-NATIVE-SIDECAR-01/20260828T001500Z__action-native-aggregate-s0-r1`

- targets: 72
- output bytes: 3,317,884,673
- inference repeated: false
- target/calibration/confirmation/test read: false

Canonical evidence:

`run://worldsim_v65/WS-V65-P10V-ACTION-EVIDENCE-01/20260828T001500Z__action-evidence-s0-r1`

- units: 72
- reused partial units: 48
- disk bytes: 75,306,035
- wall: 32.1748s
- source-role overlap: 0

No new failure occurred. The next operation is the single frozen fixed-lattice action-level visited-state quality read.
