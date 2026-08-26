# P10M Target-Free Conditional State Bake Closeout

Date: 2026-08-26

## Outcome

Canonical run: `run://worldsim_v64/WS-V64-P10M-CONDITIONAL-STATE-BAKE-01/20260826T180000Z__conditional-state-bake-s0-r1`

Verdict: `supported_target_free_conditional_state_bake`

The single formal run produced and consumed 96 metric native point-state packages over 1,150,300 eligible boundary voxels. C0 emitted 460,082 voxels and M0 emitted 534,581, a gain of 74,499. Mean realized coverage was 0.39994442456469775 for C0 and 0.4749608367012231 for M0, preserving the frozen uplift of 0.07501641213652532.

Additional emitted voxels by stratum were 25,199 construction, 35,221 night, zero rain, and 14,079 vulnerable transit. At the frozen 0.512 metre voxel size, their nominal summed voxel volume is 9,999.086518272 cubic metres. This is a package statistic across case-time units, not unique verified world volume.

## Target-free and runtime boundary

The state bake opened `METHOD_EVIDENCE`, native logits/BEV, and the frozen MLP. It did not open target evidence. Each package contains native indices, metric centres, frozen risk scores, and C0/M0 `OCCUPIED` or `UNKNOWN` states. The semantic runtime consumer reopened only the package and did not load the model, native arrays, or evidence.

Both minimal gates passed: mean coverage uplift was at least 0.05, and all 96 packages were consumed with positive additional M0 state. Output size was 27,780,960 bytes. GPU wall time was 9.399617448449135 seconds and peak RSS was 0.7806472778320312 GiB.

## Claim boundary

This supports target-free state materialization and package-only semantic consumption. It does not support Gaussian rendering, sensor replay, collision validity, planning utility, closed-loop behavior, or safety. No refit, policy selection, hash, checksum, fingerprint, smoke matrix, or regression matrix was used.

The next direct task is the smallest faithful Gaussian-state adapter that can consume these metric point states without reviving the old hash-heavy governance pipeline.
