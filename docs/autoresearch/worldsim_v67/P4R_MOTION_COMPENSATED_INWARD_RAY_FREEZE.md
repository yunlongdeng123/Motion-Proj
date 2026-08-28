# V6.7 P4R motion-compensated inward-ray recovery freeze

- Date: 2026-08-28
- Failure addressed: `V67-F01`
- Task: `WS-V67-P4R-MOTION-COMPENSATED-INWARD-RAY-01`
- Hypothesis: `WS-V67-H-P4R-001`

## Blocker research and migration

The post-P4 search found a consistent geometric prescription: occupancy supervision distinguishes free space before a sensor
termination from unknown/occupied support behind it; recent sparse occupancy reconstruction also extends a measured surface
inward along its viewing ray. The relevant primary sources are ALSO (CVPR 2023), evidence-theory occupancy labels (CVPR
2024), a continuous occlusion model (CVPR 2016), and GPOcc's inward ray samples (CVPR 2026).

P4 did not preserve the source ray identity after dynamic-point motion compensation. P4R therefore computes the ray
analytically in the target frame for the nearest motion-compensated same-Actor hit, using the target LiDAR origin already
defined by that coordinate frame.

## Frozen recovery

```text
delta = query_center - nearest_motion_compensated_same_actor_hit
ray   = normalize(nearest_motion_compensated_same_actor_hit - target_lidar_origin)
KEEP  = exact same-Actor hit
     OR (distance <= 0.512m AND dot(delta, ray) >= 0)
otherwise UNKNOWN
```

This is a directional inward half-ball, not an intermediate radius or threshold selection. P3's 258 fixed actions, the
`0.512m` representation scale, target-only evaluation, Actor/shell/identity/trajectory/hazard protection, and all nine P4
gates remain unchanged. There is one formal P4R run and no rule/radius/budget/gate sweep.

Success only supports task-untouched legacy motion-compensated inward-ray surface capability. It does not support fresh
population generalization, RL readiness, planning, closed-loop policy or safety.
