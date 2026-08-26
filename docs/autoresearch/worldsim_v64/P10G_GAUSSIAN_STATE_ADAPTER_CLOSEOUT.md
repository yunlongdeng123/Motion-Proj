# P10G Sparse Gaussian State Adapter Closeout

Date: 2026-08-26

## Outcome

Canonical run: `run://worldsim_v64/WS-V64-P10G-GAUSSIAN-STATE-ADAPTER-01/20260826T181500Z__gaussian-state-adapter-s0-r1`

Verdict: `supported_sparse_gaussian_state_adapter`

All 96 target-free P10M packages were converted and rendered. M0 produced 534,581 sparse semantic Gaussians versus 460,082 for C0, preserving 74,499 additional authorized states. At the frozen BEV density threshold, aggregate support increased from 553,756 to 594,772 cells, a gain of 41,016.

Each Gaussian used the metric voxel centre as mean, isotropic scale 0.256 metres, identity rotation, opacity 0.95, and `OCCUPIED` semantic state. Output size was 40,148,486 bytes. Batched GPU splatting took 0.9840179476886988 seconds and peak RSS was 0.8689231872558594 GiB.

## Recovery closure

V64-F19 is `resolved_by_sparse_gaussian_adapter`: the fresh cohort can enter a faithful sparse semantic Gaussian consumer without a mismatched StreetGS checkpoint or the old hash-heavy runtime. The adapter and consumer accessed neither target evidence, the risk model, nor a StreetGS checkpoint.

This closure does not claim that the fresh scenes have photorealistic Gaussian radiance fields. It only closes the representation-and-splat integration block.

## Claim boundary

The result supports sparse semantic Gaussian parameterization and probabilistic BEV Gaussian superposition. Sensor replay, appearance, collision validity, route validity, planning, closed-loop behavior, and safety remain untested. No parameter sweep, hash, checksum, fingerprint, smoke matrix, or regression matrix was used.

The next direct consumer should use the Gaussian BEV state for a bounded route or collision semantic calculation rather than introduce another rendering test matrix.
