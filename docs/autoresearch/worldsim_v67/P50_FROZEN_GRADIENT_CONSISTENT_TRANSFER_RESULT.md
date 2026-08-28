# P50 Frozen Gradient-Consistent Transfer Result

Canonical: `run://worldsim_v67/WS-V67-P50-FROZEN-GRADIENT-CONSISTENT-TRANSFER-01/20260829T020000Z__frozen-gradient-transfer-s0-r1`.

The frozen P49/P31/P20 composition was read once on P2V H=`1.5s`: 70 evaluable cases and exactly `252/252` selected
actions. Case coverage was `0.714286`, minimum group coverage `0.541667`, and all six scenes were non-increasing.

P49/P31/fixed-P20 reduction was `0.789696/0.739907/0.301221`; delta over P31 was `+0.049789`. All four gates passed,
so verdict is `supported_frozen_gradient_consistent_cross_condition_transfer`. There was no training or refit.

Wall/peak GPU/peak RSS: `0.457s / 0.00913GiB / 0.71492GiB`.
