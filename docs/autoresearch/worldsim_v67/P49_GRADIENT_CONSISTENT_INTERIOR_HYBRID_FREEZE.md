# P49 Gradient-Consistent Interior Hybrid Freeze

P49 preserves P48's zero residual amplitude at budget `.25/.50` and fixed peak at `1/3`. It adds P10R2 H=`1.5s` as
the twelfth consumed development domain and changes only the training objective: a fixed `.01` penalty on dispersion of
normalized domain-loss gradients for the adapter's final linear layer.

This is a bounded Fishr-inspired transfer, not an implementation/equivalence claim for Fishr's per-sample gradient-variance
matching. P3C H=`1.5s` targets are materialized for the first time in parallel with GPU training and read once at budget
`1/3`. Gates remain exact total, minimum group coverage `0.50`, delta over P31 `+0.005`, and four non-increasing scenes.

No gradient-weight/layer/anchor/peak/model/loss/gate sweep; no fresh-population, collision, planning, policy, closed-loop,
or safety claim; no hash/checksum/fingerprint.

References: Fishr, ICML 2022: https://proceedings.mlr.press/v162/rame22a.html ; arXiv: https://arxiv.org/abs/2109.02934 .
