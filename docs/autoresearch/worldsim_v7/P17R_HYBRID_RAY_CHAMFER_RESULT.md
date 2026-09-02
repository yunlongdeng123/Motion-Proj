# P17R Hybrid Ray--Chamfer Result

Date: 2026-09-02

Canonical: `run://worldsim_v7/WS-V7-P17R-HYBRID-RAY-CHAMFER-FIT-01/20260903T094500Z__hybrid-ray-chamfer-s71701-r1`

The normalized hybrid objective recovers most, but not all, of P17's utility loss. Completion coverage rises from P17's 71.43% to
88.96%. On 228 disjoint nuScenes test Actors, total new-early changes from `.96618%` to `.94836%`, hazardous new-early from
`1.43622%` to `1.41532%`, and clear new-early from `.44250%` to `.42811%`.

Mean Chamfer nevertheless remains worse than frozen always-COMPLETE: `.1945868 m` versus `.1957160 m`. New hits decrease from
39,255 to 37,518. The hybrid term reduces P17's Chamfer regression by about 76.6% but does not cross the pre-frozen no-regression
boundary. Register `V7-F26`, close further ray-loss/mixture/threshold recovery, and keep fresh AV2 unread.

The remaining valid system move is two-stage expert routing rather than another surface optimizer. Treat frozen always-COMPLETE and
frozen P17R as experts; choose P17R only when a source-trained Actor router predicts its per-Actor joint dominance, otherwise retain
the baseline expert. This changes the complete action system, not P17R's point set or threshold.
