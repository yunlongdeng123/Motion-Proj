# V6.7 P15R bounded lattice residual result

Canonical=`run://worldsim_v67/WS-V67-P15R-LATTICE-RESIDUAL-RELIABILITY-01/20260828T123000Z__bounded-lattice-s0-r1`;
verdict=`rejected_trajectory_conditioned_visited_state_reliability_selection`; 4/6 gates pass.

The 12-parameter bounded adapter preserves and slightly improves transfer: selection Spearman `0.780370` vs qmean `0.772946`,
unsafe AUROC `0.973522` vs `0.972730`, and pairwise concordance `0.672834` vs `0.655686`. Bottom-quartile selected-cost
reduction is only `0.170481` vs qmean `0.163836` (`+0.006645`), below the frozen `0.25/+0.05` gates. P15R is rejected and
`V67-F05` closes after its single recovery; the result nevertheless establishes that constrained trajectory residuals transfer
far better than the free MLP.

P16 does not retune P15R on the consumed selection. It treats both P10V and P10X as two development domains, trains one
domain-balanced bounded adapter, freezes it, and only then materializes action-visited targets on the P9 fresh scenes. This is a
new fresh action-task confirmation, not another P15 selection attempt.
