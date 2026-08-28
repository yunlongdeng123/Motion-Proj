# V6.7 P24 adaptive budget compiler result

Canonical run: `run://worldsim_v67/WS-V67-P24-ADAPTIVE-BUDGET-COMPILER-01/
20260828T150500Z__adaptive-budget-s0-r2`.

r2 reused the r1 frozen offset artifact and action cache after correcting a pre-metric single-action case alignment. It evaluated
78 cases and selected exactly 222 actions, matching the fixed-quarter P20 budget. Per-case allocation ranged from one to five.

Adaptive/fixed-P20/qmean cost reduction was `0.758380/0.596770/0.569662`; adaptive allocation added 0.161610 over P20.
All seven evaluable scenes were non-increasing and all four gates passed. r1 training was not repeated.
