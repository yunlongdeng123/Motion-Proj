# V6.7 P16 multidomain trajectory reliability freeze

P16 combines the consumed P10V/P10X compact action rows as two development domains. Case and scene identifiers are offset
before concatenation. The same 12-parameter, case-centered, `±0.02` qmean residual adapter is optimized with equal-domain Huber
risk, pairwise ranking and a frozen domain-loss variance penalty.

After training, the runner writes `model_frozen.json`. Only then may it materialize the P9 six-scene fixed-lattice action cache
from the already available native/evidence sidecars. P9 has been consumed for analytic surface evaluation but its candidate-action
visited-state targets have never been read, so its role is fresh action-task confirmation.

The one-shot gates are Spearman >=0.70, unsafe AUROC >=0.90, pairwise concordance >=0.65, bottom-quartile selected-cost
reduction >=0.20, no reduction loss relative to qmean, and at least five non-increasing scenes. No second confirmation, lattice,
residual-bound, loss or gate sweep is allowed.
