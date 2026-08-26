# P4C Conditional Confirmation Sidecar Closeout

Date: 2026-08-26

## Outcome

The corrected blind native cohort is complete and passed the capability contract:

- canonical aggregate: `run://worldsim_v64/WS-V64-P4C-CONDITIONAL-CONFIRMATION-SIDECAR-01/20260826T170000Z__native-aggregate-s0-r1`;
- cohort: `scene-0992, scene-1101, scene-0454, scene-1102, scene-0876, scene-0895, scene-0321, scene-0813`;
- retained valid v1 leaves: 7;
- frozen replacement leaf: `scene-0813`, processed index 631;
- result: 8 scenes, 96 targets, 4,423,846,027 logical output bytes;
- maximum worker peak GPU memory: 4.13140344619751 GiB;
- aggregate verdict: `passed`.

The replacement waited 716.9760900679976 seconds for its exact raw members and required 45.253681937232614 seconds for blind native inference. Scene-ready execution handed the scene to DriveStudio and then IR-WM without waiting for unrelated shard EOF.

## Recovery closure

`scene-0276` was absent from the IR-WM train temporal infos and failed before native output. The recovery kept all seven valid native leaves and changed only the vulnerable-transit replacement to the preregistered, token-valid temporal member `scene-0813`. C0, M0, the frozen MLP, confirmation gates, and the 96-case denominator did not change.

V64-F18 is therefore `resolved_pre_quality`. The original and replacement catalog writers remain isolated until controller EOF; their semantic union is operational cleanup and is not on the evidence critical path.

## Read and claim boundary

At this closeout:

- confirmation target read: false;
- confirmation quality read: false;
- confirmation model-score read: false;
- model refit: false;
- policy or coverage selection: false;
- hash, checksum, or fingerprint: none;
- extra smoke or regression matrix: none.

This artifact supports blind native-sidecar capability only. It does not support the conditional compiler until the frozen 96-unit evidence is generated and C0/M0 are scored exactly once.

## Next action

Run only `WS-V64-P4C-CONDITIONAL-CONFIRMATION-EVIDENCE-01/20260826T171500Z__confirmation-evidence-s0-r1`, then execute the preregistered exact-once conditional scorer once.
