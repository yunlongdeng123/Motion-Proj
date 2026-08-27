# V6.5 P1R3 Map/Context Result

- task: `WS-V65-P1R3-MAP-CONTEXT-TRAIN-ONLY-01`
- canonical run: `run://worldsim_v65/WS-V65-P1R3-MAP-CONTEXT-TRAIN-ONLY-01/20260827T114500Z__map-context-s0-r1`
- verdict: `no_clear_train_only_map_context_signal`
- formal V6.5 selection read: `false`

The single preregistered seed-0 run materialized 192 units with next-unit evidence/native I/O prefetched while the
current unit ran q0 and trajectory geometry on the RTX 3090. It trained on 523,910 points and evaluated 497,892 points.

| metric | frozen q0 | q0 + R3 map/context | delta |
| --- | ---: | ---: | ---: |
| AUROC | 0.871759 | 0.871264 | -0.000496 |
| AUPRC | 0.407081 | 0.404801 | -0.002280 |
| pooled fixed-route conflict density | 0.00299581 | 0.00299581 | 0.00% |
| worst-tail fixed-route CVaR | 0.0164397 | 0.0162584 | -1.10% descriptive |
| non-route emitted conflict rate | 0.00595363 | 0.00590859 | -0.756% |

Scene lower/equal/higher was `1/14/1`. Real-map AUROC exceeded within-unit shuffled-map AUROC by only `0.000625`.
Thus only the non-route and shuffle gates passed; AUROC gain, route-risk reduction, and scene-direction gates failed.
The 14D map/context residual is rejected without seed, capacity, threshold, or feature rescue. Peak allocated GPU memory
was 0.1397 GiB, peak RSS 1.9598 GiB, and wall time 85.42 s; resource capacity was not the limiting factor.

The next hypothesis changes the prediction object from per-voxel correctness to trajectory-level visited-state
reliability over the future Ego corridor. This follows task-relevant failure detection: errors matter through their
effect on the executed plan, rather than as uniformly weighted state errors.

