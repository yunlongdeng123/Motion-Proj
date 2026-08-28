# P45 Anchored Hybrid Nested Budget Result

Canonical run: `run://worldsim_v67/WS-V67-P45-ANCHORED-HYBRID-NESTED-BUDGET-01/20260828T231000Z__anchored-hybrid-nested-s0-r1`.

Low/high totals were exactly `220/220` and `436/436`; every low action was retained in the high set. At quarter budget,
anchored hybrid and P31 reduction were exactly `0.802420`, as required by the structural fallback. At half budget, anchored
hybrid reduction was `0.700183` versus P31 `0.694720`, a `+0.005463` gain.

Minimum group coverage was `0.50/0.625`; non-increasing scene support was `5/7` and `7/7`. All five gates passed;
verdict `supported_anchored_hybrid_nested_budget`.

The anchor resolves P43's low-budget regression while preserving a high-budget gain under strict nesting. Evidence remains a
new task condition on a consumed cohort, without fresh-population, planning, closed-loop, collision, or safety claims.
