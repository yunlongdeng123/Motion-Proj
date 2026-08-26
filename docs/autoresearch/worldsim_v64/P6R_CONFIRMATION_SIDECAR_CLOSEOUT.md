# P6R exact-once confirmation sidecar closeout

Date: 2026-08-26

Canonical aggregate:

`run://worldsim_v64/WS-V64-P6R-CONFIRMATION-SIDECAR-01/20260826T150000Z__native-aggregate-s0-r1`

All eight frozen scenes completed 12 blind IR-WM targets each. The aggregate contains 96 targets and 4,423,846,018 bytes through symlinks to per-scene outputs; native arrays were not copied. Maximum worker GPU memory was 4.1314 GiB. Confirmation target evidence, hidden-FREE quality, and exact-once case scores remained unread.

The compressed-tar scan was scheduled by scene readiness. Shard10 was prioritized to feed scene-1105, shards8/9 fed scene-0903, shards4/5 fed scene-0451, shard6 fed scene-0537, and shards1/2 fed scene-0157. DriveStudio preprocessing had exclusive I/O windows, while unfinished scans resumed during GPU extraction. This produced GPU work before the full ten-shard barrier and retained a superset member-to-shard catalog for later sparse batches.

Verdict: `blind_confirmation_sidecar_complete`. This is an input-capability result only. Exact-once target quality remains locked until the milestone commit is pushed.

## Catalog finalization

After exact-once scoring, the three paused EOF scanners resumed and the original preparation controller completed normally. Canonical prep is `run://worldsim_v64/WS-V64-P6R-CONFIRMATION-SIDECAR-01/20260826T143000Z__confirmation-prep-s0-r1`: eight complete processed scenes, quality unread, wall 5,872.4206 seconds. The persistent superset catalog now contains 57,338 member-to-shard entries (6,880,063 bytes), and `/root/autodl-tmp/tmp/worldsim_v64_p6_raw_batch` was removed by the controller. V64-F16 is resolved; the long wall time records the unavoidable first sparse scan, while scene-ready scheduling prevented it from becoming a GPU barrier.
