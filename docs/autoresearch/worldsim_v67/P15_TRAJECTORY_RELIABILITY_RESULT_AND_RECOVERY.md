# V6.7 P15 trajectory reliability result and single recovery

P15 canonical=`run://worldsim_v67/WS-V67-P15-TRAJECTORY-RELIABILITY-TRAIN-01/20260828T122000Z__trajectory-reliability-s0-r1`;
verdict=`rejected_trajectory_conditioned_visited_state_reliability_selection`.

The free `64/64` residual MLP reaches train Spearman/pairwise/selected reduction `0.8633/1.0/0.5015`, but transfers at only
`0.2102/0.5866/0.1099`, versus qmean `0.7729/0.6557/0.1638`. Unsafe AUROC is `0.6464` versus qmean `0.9727`; all six
selection gates fail. The trajectory-level prediction object remains valid, but the unrestricted action/context residual learns a
source-scene shortcut.

[ResAD](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_ResAD_Normalized_Residual_Trajectory_Modeling_for_End-to-End_Autonomous_Driving_CVPR_2026_paper.html)
motivates anchoring learning to a deterministic reference, while [instance-wise monotonic calibration](https://proceedings.mlr.press/v286/zhang25c.html)
motivates preserving a strong base ranking. P15R therefore replaces the MLP with 12 learned action-lattice biases. Biases are
centered within each case and the score-space residual is bounded to `±0.02`, equal to the preregistered meaningful pairwise gap.
qmean remains dominant; data, lattice, loss family and six gates are unchanged. This is the only P15 recovery.
