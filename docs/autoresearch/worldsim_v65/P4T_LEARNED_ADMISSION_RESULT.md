# P4T Learned Admission Result

Canonical run:

```text
run://worldsim_v65/WS-V65-P4T-LEARNED-ADMISSION-TRAIN-ONLY-01/20260827T110000Z__learned-admission-s0-r1
```

The frozen V6.4 M1 comparator emitted mean coverage 0.474961 with zero case failures. G0 emitted 0.541329, an absolute
uplift of 0.066368, but introduced one case failure. Its pooled fixed-route density increased from 0.00181015 to
0.00196987 (`+8.82%`) and worst-10% fixed-route CVaR increased from 0.0158854 to 0.0170938 (`+7.61%`). Seven of eight
evaluation scenes had nonnegative scene utility, but only the coverage/scene-support gates passed.

The new failure occurred in the first night evaluation scene: predicted coverage 0.521873 exceeded that case's train-only
oracle-safe coverage 0.510822; conflict rose from M1's 0.047814 to 0.050764. This is a real selective-risk violation, even
though it is numerically close to the 0.05 boundary. It is not rounded down or post-calibrated after observation.

Verdict: `no_clear_train_only_learned_admission`. `WS-V65-H-P4T-001` is rejected, no fresh admission cohort is prepared,
and P5 SOFT top-k / allocator is not unlocked. The 16 MiB case cache, learned model, case rows, and failed gates are retained.
No V6.5 admission selection, calibration, confirmation, or test partition was read.
