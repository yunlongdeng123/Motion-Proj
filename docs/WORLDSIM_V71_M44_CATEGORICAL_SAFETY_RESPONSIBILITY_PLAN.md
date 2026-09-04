# WorldSim V7.1 M44 — categorical safety boundary and family responsibility

## Purpose

M44 does not change M39 and does not claim that an explanation creates physical consistency. The consistency signal already
comes from held-out LiDAR F/O/U supervision. This milestone exposes the learned representation's exact decision boundary and
attributes its probability measure to observed anchors versus completion children.

For depth samples `d_i`, primitive occupied masses `o_j`, Gaussian kernels `k_ij`, define the normalized joint measure

`q(i,j) = o_j k_ij / sum_(a,b) o_b k_ab`, so `p_i = sum_j q(i,j)`.

With the same linear CDF interpolation as deployment, the predicted median is early exactly when

`CDF(d_gt - 0.20m) > 0.5`.

The boundary CDF decomposes without approximation as anchor mass plus child mass. M44 reports this identity, its numerical
residual, and the paired change from unit-energy baseline to frozen M39 for all/hazard/clear strata.

## Protocol

- Same 66 development-exposed holdout Actors, frozen M8/M35/M38/M39, 64 samples, CDF threshold 0.5.
- No training, threshold/scale/bin sweep, filtering, deletion, model selection, or external/partial read.
- Report median-boundary equivalence, mean boundary CDF, and anchor/child pre-boundary mass deltas.
- This is an explanation/safety-bound audit, not an acceptance gate and not a reason to alter the frozen M43 candidate.
