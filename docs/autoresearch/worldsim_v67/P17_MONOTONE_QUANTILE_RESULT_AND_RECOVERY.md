# V6.7 P17 monotone quantile result and recovery

Canonical run: `run://worldsim_v67/WS-V67-P17-MONOTONE-QUANTILE-TRAJECTORY-01/
20260828T125500Z__quantile-trajectory-s0-r1`.

The model used eight frozen quantiles of q0 along each two-second trajectory corridor and learned only a convex quantile pool plus
a bounded mix with qmean. Development materialization retained 813 P10V and 739 P10X actions; P9 selection retained 846 actions.
After 3,000 GPU epochs the learned distribution mix was 0.497368 and the largest quantile weight was on the minimum score.

On P9, learned/qmean Spearman was `0.645502/0.658731`, unsafe AUROC `0.814677/0.826644`, pairwise concordance
`0.749190/0.779650`, and selected-cost reduction `0.387839/0.418184`. Only scene support passed (1/6), so the candidate is
rejected. No quantile, mix, loss or gate sweep follows.

The recovery changes the downstream object: qmean ranking is frozen, and P18 learns only whether a case has enough observable
selection margin to grant action authority or should abstain. This follows selective/conformal planning work without claiming a
finite-sample conformal guarantee on the non-exchangeable cohorts.
