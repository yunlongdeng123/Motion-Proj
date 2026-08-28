# P53 Joint Budget-Horizon Gradient Hybrid Freeze

P53 keeps P51's model, double endpoint anchor, gradient-direction penalty, and losses. It adds consumed P6E H=`1.5s` as
development domain 14 and adds one fixed active interior training budget `.40`, producing training budgets
`{.25, 1/3, .40, .50}`. There is no budget or hyperparameter sweep.

P10X H=`0.8s` is materialized for the first time in parallel: `662/864` eligible actions from 72 source cases. Confirmation
uses jointly unseen budget `.375` and below-range horizon `.8s`. Gates are exact total, minimum group coverage `.50`, delta
over P31 `+.005`, and five non-increasing scenes.

No gradient-weight/budget-set/anchor/peak/model/loss/gate sweep; no full-Fishr equivalence, fresh-population, collision,
planning, policy, closed-loop, or safety claim; no hash/checksum/fingerprint.
