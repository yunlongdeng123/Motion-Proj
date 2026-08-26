# P10R2 Fresh Confirmation Sidecar Closeout

Date: 2026-08-27  
Canonical native aggregate: `run://worldsim_v64/WS-V64-P10R2-CONFIRMATION-SIDECAR-01/20260826T201000Z__native-aggregate-s3-r1`

All eight frozen scenes completed canonical DriveStudio processing and 12 target-free IR-WM native targets. The aggregate
contains 96 targets and 4,423,846,005 logical output bytes, passed, and reports maximum per-worker GPU memory
4.131403446 GiB. Per-scene native wall time ranged from 44.453197 to 59.723026 seconds.

The ready-first restart reused six complete native leaves and ran only scene-0006 and scene-0371 concurrently. A prep/feeder
producer overlap on scene-0371 was stopped before duplicate canonical write. The only incomplete native run, scene-0006,
contained no summary and one partial file; after confirming no process owned it, that exact partial run was removed and rebuilt
from its complete processed scene. No complete native leaf was deleted or recomputed.

The original prep run remains an operations-failed record. The recovery prep
`run://worldsim_v64/WS-V64-P10R2-CONFIRMATION-SIDECAR-01/20260827T011800Z__confirmation-prep-reuse-s3-r2`
reused all eight complete processed scenes, finished in 0.817105 seconds, and removed the recoverable temporary raw directory.
The persistent semantic member-to-shard catalog now has 71,555 entries and 8,585,986 bytes.

Confirmation target evidence, route conflict, and model score remained unread throughout sidecar generation. Cohort, model,
M0/M1 policy, target frames, route, and gates did not change. The next action is exactly one 96-unit evidence materialization,
followed by the preregistered exact-once M1 route-tail comparison.
