# P7 Fixed-budget Hazard-preserving Triage 结果

Task：`WS-V66-P7-HAZARD-PRESERVING-DISTRIBUTION-01`

Canonical：`run://worldsim_v66/WS-V66-P7-HAZARD-PRESERVING-DISTRIBUTION-01/20260828T092919Z__fixed-budget-distribution-s0-r1`

581 actor states含307 conflicts，local-action budget固定为290。所有臂Actor retention=1、removed=0；hazard proxy
distribution shift=0。

| Arm | action | handled / 307 | exposure reduction | emitted local fraction | scene yield |
|---|---:|---:|---:|---:|---:|
| N0 naive | 0 | 0 | 0 | 1.0000 | 1.0000 |
| Q0 q0 rank | 290 | 193 | 0.6287 | 0.5009 | 0.8333 |
| D0 deterministic | 0 | 0 | 0 | 1.0000 | 1.0000 |
| L0 learned HARP | 290 | 210 | 0.6840 | 0.5009 | 1.0000 |
| O0 oracle | 290 | 290 | 0.9446 | 0.5009 | 1.0000 |

L0 6/6 gates通过，且避免q0把一个scene的local geometry全部选入action造成的yield下降。但action仍只是triage
候选，physical geometry没有修改；因此完整P7 artifact repair/hazard-event preservation尚未成立，登记`V66-F02`并进入
sensor-supported actor-local surface repair。
