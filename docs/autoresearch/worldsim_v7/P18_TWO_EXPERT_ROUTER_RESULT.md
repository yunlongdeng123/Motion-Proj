# P18 Two-Expert Completion Router Result

Date: 2026-09-02

Canonical: `run://worldsim_v7/WS-V7-P18-TWO-EXPERT-ROUTER-FIT-01/20260903T104500Z__two-expert-router-s71801-r1`

P18 is rejected on consumed source development without reading fresh AV2. The frozen router fits the 85 training/calibration Actors
(`82` always-COMPLETE, `3` P17R) but the P17R-dominant class is too sparse to transfer: among 228 test Actors the dominance oracle
labels only two P17R, while the router selects always-COMPLETE for all 228. Test routing accuracy is `99.12%`, but this misleading
aggregate is exactly the majority solution.

The routed system therefore equals always-COMPLETE: hazardous new-early remains `1.4362216%`, population Chamfer remains
`.1945868 m`, and only the non-regression gate passes. Wall time is `399.09 s`, peak GPU memory `.0428 GiB`, peak RSS
`1.293 GiB`, and `target_data_read=false`.

A post-verdict oracle audit bounds the current two-expert family: only `2/228` test Actors are jointly P17R-dominant, and selecting
both changes hazard new-early by one event and mean Chamfer by only `-0.0000061 m` while keeping `99.94%` completion coverage.
Register `V7-F27` and close Actor-level expert routing. P19 moves the action unit down to a fixed-capacity candidate veto rather than
trying another rare-class router.
