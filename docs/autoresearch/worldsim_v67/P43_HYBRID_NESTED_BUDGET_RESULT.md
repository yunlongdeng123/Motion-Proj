# P43 Hybrid Nested Budget Result

Canonical run: `run://worldsim_v67/WS-V67-P43-HYBRID-NESTED-BUDGET-01/20260828T221000Z__hybrid-nested-s0-r1`.

Low/high totals were exactly `222/222` and `438/438`; all 222 low actions were retained in the high set. Minimum group
coverage was `0.50/0.916667`, with `6/7` and `7/7` non-increasing scenes.

Hybrid low/high reductions were `0.808732/0.641285`, versus `0.833218/0.638464` for frozen P31 nested selection.
Deltas were `-0.024486/+0.002821`; low-budget nonregression failed. Verdict `rejected_hybrid_nested_budget`.

Exact nesting and broad coverage do not override the quarter-budget cost regression. P42 remains supported at one-third,
but cannot be claimed budget-uniform. No budget or blend-weight sweep follows this consumed read.
