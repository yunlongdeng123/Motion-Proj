# P10G Sparse Gaussian State Adapter Freeze

Date: 2026-08-26

## Block and migration

The fresh P10M cohort has no same-scene StreetGS or SceneIR checkpoint. The old V6 Gaussian runtime is tied to different scenes and hash-heavy package governance, so direct reuse would violate both semantic alignment and the V6.4 no-hash constraint.

The recovery is based on three primary implementations/papers:

- GaussianFormer represents driving scenes as sparse semantic 3D Gaussians: https://github.com/huang-yh/GaussianFormer
- GaussianWorld uses explicit Gaussian position, scale, rotation, semantic probability, and feature fields: https://openaccess.thecvf.com/content/CVPR2025/papers/Zuo_GaussianWorld_Gaussian_World_Model_for_Streaming_3D_Occupancy_Prediction_CVPR_2025_paper.pdf
- GaussianOcc demonstrates voxel-grid Gaussian splatting with uniform scale and identity rotation: https://openaccess.thecvf.com/content/ICCV2025/papers/Gan_GaussianOcc_Fully_Self-supervised_and_Efficient_3D_Occupancy_Estimation_with_Gaussian_ICCV_2025_paper.pdf

V64-F19 is `recovery_frozen_pre_run`.

## Frozen task

Task: `WS-V64-P10G-GAUSSIAN-STATE-ADAPTER-01`

Run: `20260826T181500Z__gaussian-state-adapter-s0-r1`

Hypothesis: `WS-V64-H-P10G-001`

The adapter reads only the 96 P10M `PHYSICAL_STATE.npz` packages. For every M0 `OCCUPIED` voxel it emits one semantic Gaussian with:

- mean: frozen metric voxel centre;
- scale: isotropic 0.256 metres, half the native voxel size;
- rotation: identity quaternion;
- opacity: 0.95;
- semantic state: `OCCUPIED`.

It then renders C0 and M0 as a batched 200 by 200 GPU BEV probabilistic Gaussian superposition at 0.512 metres per cell. The consumer does not access target evidence, the risk model, native features, or a StreetGS checkpoint.

## Gates and claim boundary

Only two gates are used: all 96 packages render, and aggregate M0 BEV support is strictly greater than C0 support at frozen density threshold 0.05. There is no scale, opacity, kernel, threshold, or seed sweep and no extra smoke/regression matrix.

A pass supports sparse semantic Gaussian parameterization plus BEV splatting. It is not photorealistic Gaussian rendering, sensor replay, collision validity, planning utility, closed-loop behavior, or safety.
