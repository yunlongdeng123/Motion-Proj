# P8R single-rate-limiter reactive-Actor result

Canonical run:
`run://worldsim_v66/WS-V66-P8R-STOP-STATE-JERK-RECOVERY-01/20260828T095839Z__stop-state-jerk-recovery-s0-r1`.

P8R changed only the numerical stop-state update: stopped desired acceleration is zero and the unchanged
`6m/s^3` command rate limiter is applied exactly once. Actor selection, scenario, trajectories, horizon, IDM/AV
parameters, outputs and gates were exact with P8.

Results:

- selected/supported scenes: `6/6`;
- pooled X0/X1 collision steps: `306/0`;
- minimum X1 bumper gap: `1.948192m`;
- maximum absolute command jerk: `6.000000m/s^3`;
- identity/lifecycle and logged-path retention: PASS;
- wall/RSS/GPU: `0.83402s / 0.50465GiB / false`.

All four pooled gates and all seven per-scene gates pass. The verdict is
`supported_synthetic_lead_brake_reactive_actor_capability`, and `V66-F03` is resolved by the single implementation
recovery.

The claim remains narrow: a retained Actor can execute this fixed bounded response on its logged path (plus explicit
terminal-tangent extension) in a synthetic lead-brake intervention. This is not natural interaction validation,
physical geometry repair, a planner or policy, RL, or safety. Because P7 physical repair is terminal negative, P9
remains locked despite P8R support.
