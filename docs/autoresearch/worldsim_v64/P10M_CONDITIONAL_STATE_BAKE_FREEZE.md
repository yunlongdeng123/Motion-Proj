# P10M Target-Free Conditional State Bake Freeze

Date: 2026-08-26

## Objective

Materialize the exact-once-supported frozen C0/M0 policies into metric native point-state packages that a downstream semantic runtime can consume without loading target evidence or the risk model.

Task: `WS-V64-P10M-CONDITIONAL-STATE-BAKE-01`

Run: `20260826T180000Z__conditional-state-bake-s0-r1`

Hypothesis: `WS-V64-H-P10M-001`

## Frozen inputs and policy

The run uses the corrected 8-scene, 96-case native confirmation cohort, each unit's `METHOD_EVIDENCE`, native IR-WM logits and BEV features, and the already frozen full-native MLP. It must not open `TARGET_EVIDENCE.npz`.

C0 remains global nominal coverage 0.40. M0 remains 0.40 for rain and 0.50 for construction, night, and vulnerable transit. There is no refit, threshold search, coverage sweep, scene selection, or second mapping.

## Package and consumer

Each case produces one `PHYSICAL_STATE.npz` containing native voxel indices, metric voxel centers, frozen risk score, and C0/M0 ternary decisions restricted to `OCCUPIED` or `UNKNOWN`. The semantic runtime then reopens only this package and reports emitted and abstained counts. It does not load model, native arrays, method evidence, or target evidence.

## Gates

Only two gates are used:

1. M0 mean realized coverage uplift over C0 is at least 0.05.
2. All 96 packages are consumed and M0 emits a positive number of additional voxels.

No hash, checksum, fingerprint, smoke matrix, regression matrix, repeated bake, or bit-exact replay is added.

## Claim boundary

A pass supports target-free physical-state materialization and package-only semantic consumption. It does not establish GS rendering, sensor replay, collision validity, planning utility, closed-loop behavior, or real-world safety.
