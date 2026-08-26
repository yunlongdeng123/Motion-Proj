# P6R confirmation-evidence empty-frame recovery freeze

Date: 2026-08-26

Failed run: `run://worldsim_v64/WS-V64-P6R-CONFIRMATION-EVIDENCE-01/20260826T151500Z__confirmation-evidence-s0-r1`.

The first exact-once evidence entrance completed 33 units, then scene-1105 target 62 raised `KeyError: '62'` while loading `frame_instances.json`. The processed scene has no keys for frames 0--9 and 56--64. The corresponding `instances_info.json` has zero actor annotations on every missing frame; `missing_with_annotations=[]`. Thus the missing key encodes an empty actor set, not missing sensor data or an unknown target.

The nuScenes devkit defines non-keyframe boxes by interpolation between sample annotations and returns current annotations when no previous annotation is available. The recovery changes only `frame_instances[str(frame)]` to `frame_instances.get(str(frame), [])` in the common box loader.

The 33 complete r1 units are retained by hard link into r2. They are not recomputed or copied. Their NPZ-derived summaries preserve semantic and sparse-array counts; three values not stored in the NPZ (`actor_count`, raw point count, and motion-compensated dynamic point count) are explicitly null in reused manifest rows. The remaining 63 units are computed once. No model score or policy outcome was read in r1.

No hash, checksum, fingerprint, parameter change, coverage change, smoke suite, or regression matrix is added.
