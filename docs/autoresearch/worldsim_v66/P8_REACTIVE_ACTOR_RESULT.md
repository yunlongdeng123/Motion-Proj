# P8 deterministic reactive-Actor result

Canonical run:
`run://worldsim_v66/WS-V66-P8-REACTIVE-ACTOR-01/20260828T095440Z__reactive-actor-s0-r1`.

The fixed six-scene synthetic lead-brake audit selected all six Actors and reduced pooled collision steps from `306`
in X0 to `0` in X1. Identity/lifecycle and logged-path retention passed, and the minimum X1 bumper gap was
`1.948192m`. Four of six scenes passed every gate.

The run is rejected because the two very-low-speed scenes violated the unchanged `6m/s^3` command-jerk gate:

- scene-0001: `9.637574m/s^3`;
- scene-0219: `7.400627m/s^3`.

Inspection found a numerical stop-state defect: after the normal jerk rate limiter, the implementation applied a
second acceleration increment when speed reached zero. The recovery may only replace this with one rate limiter and
an explicit stopped-state target acceleration of zero. It may not change any controller constant, Actor, scenario,
horizon, gate or claim boundary. See `P8R_STOP_STATE_JERK_RECOVERY_FREEZE.md`.
