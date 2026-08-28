# P26 Large-Cohort Coverage Transfer Result

Canonical run: `run://worldsim_v67/WS-V67-P26-LARGE-COHORT-COVERAGE-TRANSFER-01/20260828T154000Z__large-cohort-coverage-s0-r1`.

The ten-domain model was frozen before P6E target materialization. Of 2,304 source actions, 2,077 were eligible, yielding 180
evaluable cases. The allocator selected exactly 511 actions, matching the fixed-quarter baseline, and covered 116 cases
(`64.4444%`) with 0--6 actions per case.

| allocator | relative selected-cost reduction |
| --- | ---: |
| fixed P20 quarter per case | 0.400589 |
| P24 min-one adaptive budget | 0.683927 |
| P26 coverage-constrained budget | 0.792541 |

P26 improves over fixed P20 by `+0.391952` and over P24 by `+0.108614`. All 15 evaluable scenes are non-increasing; scene 0
had no case meeting the frozen action-footprint rule. All five gates pass. Wall time was `76.996s`, peak allocated GPU memory
`0.04711GiB`, and peak RSS `1.4924GiB`.

This supports large-cohort transfer of visited-state action reliability allocation, not collision, planner, policy, closed-loop,
population, or safety guarantees.
