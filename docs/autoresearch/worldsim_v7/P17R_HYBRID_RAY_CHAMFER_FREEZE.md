# P17R Hybrid Ray--Chamfer Completion Freeze

Date: 2026-09-02

P17R is the only recovery authorized by `V7-F25`. It reuses P17's 11 features, `64-64-1` network, seed `71701`, hard `.5`
forward selection, straight-through sigmoid, 160 epochs, Actor batch 8, 1024-ray cap, optimizer, source roles, and external cohort.

The sole change follows OccFlowNet's hybrid rendering/3D supervision: add expected bidirectional Chamfer of the hard-forward
completion set. Surface-to-target distance weights each completion candidate by its occupancy decision. Target-to-surface distance
uses the same nearest-set transmittance construction with the frozen KEEP/PROJECT core as fallback. The objective is
`ray_loss / always_complete_ray_loss + expected_chamfer / always_complete_chamfer`; there is no fitted mixture coefficient.

The same source Pareto decision is binding. Both directions passing authorize the frozen checkpoint's one fresh AV2 read after
download completion; either failing closes the ray-completion family without threshold/loss/normalization recovery. Scores remain
uncalibrated empirical geometry decisions, not occupancy probabilities or safety guarantees. Next failure is `V7-F26`.
