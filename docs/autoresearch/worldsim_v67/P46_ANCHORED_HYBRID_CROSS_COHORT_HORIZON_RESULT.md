# P46 Anchored Hybrid Cross-Cohort Horizon Result

P10R4 H=`1.5s` materialization completed concurrently with training: 1,152 source actions, 1,077 eligible, 75 excluded,
and 96 evaluable cases. Canonical run:
`run://worldsim_v67/WS-V67-P46-ANCHORED-HYBRID-CROSS-COHORT-HORIZON-01/20260828T233000Z__cross-cohort-horizon-s0-r1`.

At budget `1/3`, P46 selected exactly `353/353` actions, covered 64/96 cases (`0.666667`), retained minimum group coverage
`0.583333`, and improved or tied all eight scenes. Reduction was `0.683908`, versus `0.672419` for P31 and `0.252624`
for fixed P20: deltas `+0.011489` and `+0.431285`. All gates passed; verdict
`supported_cross_cohort_horizon_anchored_hybrid`.

This replicates anchored-hybrid gain on a second cohort's newly materialized task condition. The source cohort was present at
H=2 in development, so the result is cross-condition replication rather than fresh-population confirmation.
