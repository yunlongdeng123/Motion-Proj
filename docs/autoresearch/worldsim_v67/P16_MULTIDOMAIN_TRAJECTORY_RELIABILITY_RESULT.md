# V6.7 P16 multidomain trajectory reliability result

Canonical run: `run://worldsim_v67/WS-V67-P16-MULTIDOMAIN-TRAJECTORY-RELIABILITY-01/
20260828T124500Z__multidomain-fresh-action-s0-r2`.

P10V and P10X supplied 1,552 development actions. The bounded action-lattice adapter was frozen before P9 action-target
materialization. P9 produced 846 eligible actions from 864 candidates across 72 cases.

| metric | learned | qmean |
| --- | ---: | ---: |
| Spearman | 0.651518 | 0.658731 |
| unsafe AUROC | 0.823949 | 0.826644 |
| pairwise concordance | 0.734932 | 0.779650 |
| selected-cost reduction | 0.417033 | 0.418184 |

The adapter passed pairwise, direct reduction and scene-support gates, but failed Spearman, AUROC and non-negative improvement:
3/6 gates, rejected. The stable result is the fixed qmean trajectory aggregate, not the learned action-ID residual. r1 stopped on
an unbound training variable before model freeze or target materialization; r2 resolved `V67-F06` without changing the experiment.
