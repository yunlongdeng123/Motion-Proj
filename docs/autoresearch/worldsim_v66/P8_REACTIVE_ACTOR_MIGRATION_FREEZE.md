# P8 deterministic reactive-Actor capability freeze

## Claim boundary

P8 is an independent six-scene capability audit. It cannot recover P7 physical repair and cannot unlock P9/RL because
P7R2 closed the physical surface-repair family negatively. The only permitted claim is that a retained Actor can be
compiled into a bounded deterministic longitudinal response in a narrow synthetic lead-brake intervention.

## External migration basis

- UniSim (CVPR 2023) represents dynamic actors and supports closed-loop sensor simulation:
  <https://waabi.ai/unisim/>.
- SMARTS (CoRL/PMLR) provides multi-agent driving behaviors and scenario-based interaction:
  <https://github.com/huawei-noah/SMARTS>.
- Waymax provides differentiable multi-agent simulation plus fixed policy/agent interfaces:
  <https://github.com/waymo-research/waymax> and
  <https://waymo-research.github.io/waymax/docs/autoapi/waymax/agents/>.

The migration deliberately uses only the smallest common idea: preserve Actor identity and logged path, then replace
the nonreactive longitudinal rollout with one fixed, bounded response policy. It does not import a learned simulator,
planner or policy optimizer.

## Frozen experiment

- Source: P6 `ACTORS.jsonl`; six real scenes.
- Selection: within each scene choose the highest median-speed Actor having at least six samples and nondegenerate
  logged motion. This is metadata-only and is frozen before outcomes are read.
- Intervention: an independent AV lead vehicle starts with `12m` bumper headway, then brakes from `t=3.0s`.
- X0: selected Actor continues at its frozen initial constant speed.
- X1: selected Actor follows the same logged polyline using one deterministic IDM-style bounded longitudinal response.
- Fixed constants: `dt=0.1s`, reaction latency `0.5s`, comfortable deceleration `3m/s^2`, maximum acceleration
  `2m/s^2`, maximum jerk `6m/s^3`, initial headway `12m`, minimum gap `2m`, IDM time headway `1s` and exponent
  `4`, AV brake start `3s`, AV deceleration `2m/s^2`, and `5s` post-X0-collision observation. No parameter sweep.
- Spatial rollout stays on the observed polyline and uses its terminal tangent only after logged arc length is exhausted;
  this extension is reported explicitly and is not treated as new map or trajectory evidence.
- Outputs: both trajectories, per-scene gap/collision/kinematic metrics and pooled summary.

## Gates

For a supported scene, X1 must have fewer collision steps than X0, nonnegative minimum gap, bounded acceleration and
jerk, zero logged-path lateral deviation by construction, exact Actor identity/lifecycle retention, and observed
response latency in `[0.3, 1.0]s`. The capability gate is at least five of six supported scenes.

Any success is confined to this synthetic lead-brake protocol. It is not evidence for natural interactive behavior,
physical geometry repair, planning, policy learning, RL or safety.
