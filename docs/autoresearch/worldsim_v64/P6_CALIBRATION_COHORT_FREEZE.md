# P6 calibration / confirmation cohort freeze

- Task: `WS-V64-P6-CALIBRATION-SIDECAR-01`
- Hypothesis: `WS-V64-H-P6C-001`
- Selection seed: `0`
- Quality read at freeze: `false`

## Metadata-only selection

The pool is the 700-scene frozen IR-WM train temporal metadata. It excludes the exact 21 scenes mentioned by the V6.1–V6.3 quality/config ledger and the six P2/P4N/P5 fit/evaluation scenes. Requiring at least 40 nuScenes samples leaves 612 candidates. No occupancy, uncertainty, hidden-FREE, false-safe, or model score was read.

Four disjoint description strata are used: night, rain, construction/work-zone, and vulnerable-road-user/transit. With seed 0, six scenes are selected per stratum; the first four enter calibration and the last two remain untouched confirmation. Each scene keeps the existing 12 target frames `17,32,47,62,77,92,107,122,137,152,167,182`.

## Calibration scenes (16)

- night: `scene-1045(785), scene-1055(795), scene-1051(791), scene-1006(764)`;
- rain: `scene-0901(687), scene-0810(628), scene-0458(372), scene-0885(671)`;
- construction: `scene-0047(44), scene-0203(157), scene-0768(596), scene-0259(208)`;
- vulnerable/transit: `scene-0405(320), scene-0439(353), scene-0060(57), scene-0656(510)`.

## Untouched confirmation scenes (8)

- night: `scene-1079(819), scene-1097(837)`;
- rain: `scene-1106(845), scene-0576(458)`;
- construction: `scene-0067(64), scene-0258(207)`;
- vulnerable/transit: `scene-1083(823), scene-0738(575)`.

The numbers in parentheses are the official trainval scene-table indices consumed by DriveStudio. Confirmation target evidence remains locked until a later selective-risk rule has been frozen from calibration only.

## Resource and preparation contract

The 24 scenes are extracted from the local read-only official nuScenes tar archives into exactly `/root/autodl-tmp/tmp/worldsim_v64_p6_raw_batch`. After all 24 DriveStudio outputs have 1,176 images and 196 lidar frames, that temporary raw batch is removed; it is recoverable from the official archives. Existing raw roots are never modified or deleted. Only processed data, native sidecars, and formal summaries persist.

Expected persistent cost is about `21.6 GiB` for processed plus native sidecars, leaving roughly `34 GiB` from the observed `56 GiB` free. Two IR-WM workers have the previous upper-bound peak `8.27 GiB`, so the single RTX 3090 is sufficient. No hash, checksum, fingerprint, smoke, or regression matrix is added.

This stage materializes capability only. Calibration and confirmation quality remain unread.

