# P44 Low-Budget Anchored Hybrid Result

P6R H=`1.5s` materialization completed concurrently with training: 1,152 source actions, 881 eligible, 271 excluded by the
fixed footprint rule, and 76 evaluable cases. The scientific canonical run is
`run://worldsim_v67/WS-V67-P44-LOW-BUDGET-ANCHORED-HYBRID-01/20260828T223000Z__anchored-hybrid-s0-r1`.

At budget `1/3`, P44 selected exactly `292/292` actions, covered 51/76 cases (`0.671053`), retained minimum group coverage
`0.50`, and improved or tied all six evaluable scenes. Reduction was `0.809547`, versus `0.789186` for frozen P31 and
`0.502915` for fixed P20: deltas `+0.020361` and `+0.306632`. All gates passed; verdict
`supported_low_budget_anchored_hybrid`.

The fixed low-budget amplitude anchor preserves a structural fallback while retaining meaningful one-third-budget gain on a
previously unmaterialized horizon condition. This is not fresh-population, collision, planning, closed-loop, or safety evidence.
