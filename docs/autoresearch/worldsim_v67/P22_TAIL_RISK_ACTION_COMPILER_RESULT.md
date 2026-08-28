# V6.7 P22 tail-risk action compiler result

Canonical run: `run://worldsim_v67/WS-V67-P22-TAIL-RISK-ACTION-COMPILER-01/
20260828T140000Z__tail-risk-action-s0-r1`.

P22 trained on 415 cases and 4,729 actions across six domains with a fixed 0.25 soft selected binary-unsafe loss. P10R4
confirmation retained 1,105 actions.

P22/P20/qmean mean-cost reduction was `0.329362/0.332863/0.286027`; unsafe-rate reduction was
`0.112825/0.108106/0.060916`. The unsafe gain over P20 was only 0.004719 while mean reduction fell 0.003501. Only the 8/8
scene-support gate passed, so the candidate is rejected and the binary any-event tail family is closed without a weight sweep.
