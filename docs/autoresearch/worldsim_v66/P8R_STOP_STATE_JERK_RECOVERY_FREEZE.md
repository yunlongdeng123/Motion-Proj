# P8R single-rate-limiter stop-state recovery freeze

## Blocker and external migration

P8 reached collision-free X1 rollouts in all six scenes, but two low-speed scenes failed the jerk gate because the
implementation advanced acceleration twice at the zero-speed boundary. This is an implementation/numerical failure,
not authorization to tune the IDM scenario.

Autoware's production longitudinal controller separates DRIVE, STOPPING and STOPPED states, uses smooth-stop behavior
to decrease jerk, and exposes explicit acceleration/jerk limits. Its command gate likewise applies longitudinal jerk
limits to acceleration commands:

- <https://autowarefoundation.github.io/autoware_universe/latest/control/autoware_pid_longitudinal_controller/>
- <https://autowarefoundation.github.io/autoware_universe/main/control/autoware_vehicle_cmd_gate/>

## Sole recovery

- When both Actor and lead speed are zero, set desired acceleration to zero.
- Apply the same existing `6m/s^3` rate limiter exactly once between previous and desired acceleration.
- Keep velocity nonnegative during integration.
- Do not change Actor selection, trajectories, terminal-tangent extension, X0, horizon, IDM/AV parameters, response
  latency, output metrics or gates.
- Run exactly one recovery. Failure closes the P8 family.

The recovery remains a synthetic lead-brake capability audit and cannot recover P7 or unlock P9/RL.
