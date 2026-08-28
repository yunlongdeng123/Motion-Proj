# V6.7 P20 listwise action compiler result

Canonical run: `run://worldsim_v67/WS-V67-P20-LISTWISE-ACTION-COMPILER-01/
20260828T133000Z__listwise-action-s0-r1`.

The bounded `32/16` compiler trained on 284 cases and 3,227 actions across four development domains. Its 5,000-epoch objective
combined domain-balanced regression, pairwise ranking and differentiable bottom-quartile selected cost. The artifact was frozen
before materializing 715 eligible confirmation actions on V67 P1.

Learned/qmean Spearman was `0.734143/0.718365`, pairwise concordance `0.826230/0.792037`, and selected-cost reduction
`0.460084/0.429361`, a `+0.030723` gain. All five scenes with eligible actions improved; scene-1046 had no action meeting the
unchanged 16-point footprint. All four gates passed, resolving `V67-F08`.
