# P10R4 Untouched-Test Native Sidecar Closeout

Date: 2026-08-27  
Branch: `research/worldsim-v6.4-native-uq`  
Task: `WS-V64-P10R4-TEST-SIDECAR-01`

## Outcome

The frozen seed-4 untouched cohort completed native sidecar generation before any test target quality or model score was read.
The canonical aggregate is:

`run://worldsim_v64/WS-V64-P10R4-TEST-SIDECAR-01/20260827T023000Z__native-aggregate-s4-r1`

- scenes: 8/8
- targets: 96/96
- output bytes: 4,423,846,058
- maximum worker peak GPU memory: 4.131403446 GiB
- aggregate status: passed

All eight canonical processed scenes were registered by
`20260827T024500Z__test-prep-finalize-s4-r1`; it reused complete scenes, finished in 0.832848 s, and removed the approximately
6.2 GiB reconstructable temporary raw directory.

## I/O and GPU scheduling result

The bounded recovery used two independent preprocessing slots and two native GPU slots. Complete native leaves were reused.
For the final two scenes, the feeder's recorded raw/stage-ready waits before native launch were 0.064590 s and 0.062485 s.
Per-scene native wall time ranged from 45.384476 s to 60.511395 s. No second producer, scene replacement, or full archive
rescan was introduced.

V64-F27 is resolved by mirroring DriveStudio's actual `_processed_` to `_processed_10Hz_` target rewrite, atomically installing
the complete stage, and reusing valid native leaves. The recovery did not change the cohort, targets, model, policies, route,
fixed denominator, worst-10 tail, or gates.

## Authority boundary

This milestone proves only native sidecar capability on the frozen untouched cohort. Test target quality and model scores remain
unread. It does not establish a fixed-denominator result, population guarantee, collision result, planning benefit, closed-loop
benefit, or safety claim. No hash, checksum, fingerprint, bootstrap gate, significance gate, extra smoke suite, or regression
matrix was added.

The only next scientific action is the single frozen 96-unit evidence generation, followed by the preregistered exact-once scorer.
