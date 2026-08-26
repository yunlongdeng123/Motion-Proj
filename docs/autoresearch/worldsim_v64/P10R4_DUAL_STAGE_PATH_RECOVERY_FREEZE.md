# P10R4 Dual-Stage Path Recovery Freeze

Date: 2026-08-27  
Failure: `V64-F27`  
Test quality read: `false`

The dual-preprocess feeder completed scene-1084 and scene-1081 conversion, but DriveStudio rewrites a target named
`..._processed_824` to `..._processed_10Hz_824/trainval/824`. The feeder incorrectly expected
`..._processed_824_10Hz/trainval/824` and therefore raised before installing either canonical scene or starting its native
worker. Both staged scenes are complete (`1206/201` images/lidar for 824 and `1176/196` for 821); no native partial exists.

The feeder parent is stopped while its two unique in-flight preprocess children are allowed to finish. The path resolution is
changed only to mirror DriveStudio's existing `_processed_ -> _processed_10Hz_` rewrite. Complete staged scenes 824, 821,
424, and 522 are installed atomically after their processes exit. To avoid another GPU gap, scene-1084 and scene-1081 use the
same frozen per-scene native command and planned run directories directly while the remaining preprocessing completes. The
patched feeder is then restarted with the same prefix and reuses every complete canonical/native leaf.

No scene, model, policy, target, route, denominator, tail, gate, or run prefix changes. Target evidence and model score remain
unread. There is no second owner for any scene, no hash/checksum/fingerprint, and no additional test suite.
