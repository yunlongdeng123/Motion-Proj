# P27 Stratum-Balanced Authority Result

Canonical run: `run://worldsim_v67/WS-V67-P27-STRATUM-BALANCED-AUTHORITY-01/20260828T160000Z__stratum-balanced-s0-r1`.

The eleven-domain model was evaluated on the consumed P6R cache. It selected exactly 222 actions for 78 evaluable cases,
matching the fixed-quarter total, and covered 49 cases (`62.8205%`). Coverage by the four frozen strata was `62.5%`, `50.0%`,
`75.0%`, and `55.5556%`.

| allocator | relative selected-cost reduction |
| --- | ---: |
| fixed P20 quarter per case | 0.596770 |
| P24 min-one adaptive budget | 0.758380 |
| P27 stratum-balanced budget | 0.800447 |

P27 improves over fixed P20 by `+0.203678` and over P24 by `+0.042068`. All six evaluable scenes are non-increasing and all
six gates pass. Wall time was `58.242s`, peak allocated GPU memory `0.01709GiB`, and peak RSS `1.1900GiB`.

Because P6R was previously consumed, this supports the context-balanced allocation mechanism only, not fresh generalization or
collision, planning, policy, closed-loop, population, or safety guarantees.
