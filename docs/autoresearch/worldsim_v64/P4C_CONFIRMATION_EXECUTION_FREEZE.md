# P4C confirmation execution freeze

Date: 2026-08-26

The eight-scene, 96-case metadata-only cohort and both C0/M0 policies were frozen in `P4C_CONDITIONAL_COMPILER_FREEZE.md` before any new target quality read. This execution version additionally fixes the canonical leaves:

- blind prep: `20260826T161500Z__confirmation-prep-s0-r1`;
- per-scene native prefix: `20260826T162000Z__confirmation-native`;
- native aggregate: `20260826T170000Z__native-aggregate-s0-r1`;
- target evidence: `20260826T171500Z__confirmation-evidence-s0-r1`;
- exact-once score: `20260826T173000Z__exact-once-confirmation-s0-r1`.

The new raw batch has the exact recoverable path `/root/autodl-tmp/tmp/worldsim_v64_p4c_raw_batch`. Tar scan and DriveStudio preprocessing are CPU/I/O work. A scene-ready feeder starts preprocessing when all required members for one scene exist, then immediately starts its native GPU worker; at most one preprocess and two GPU workers run concurrently. The prep controller remains the owner of final temporary-raw deletion and superset catalog writeback.

Sidecars do not read target quality. Target evidence is generated once after all native units exist, followed by one C0/M0 score run. The model and coverages cannot change, and no second confirmation, hash, checksum, fingerprint, smoke matrix, or regression matrix is permitted.

## V2 metadata-only correction

V1 omitted IR-WM temporal-pickle membership and therefore rejected `scene-0276` before native output. Per `P4C_TEMPORAL_MEMBERSHIP_RECOVERY_FREEZE.md`, only that scene is replaced by token-valid, temporal-member `scene-0813(631)`. The seven complete native leaves and all scientific locks are retained.
