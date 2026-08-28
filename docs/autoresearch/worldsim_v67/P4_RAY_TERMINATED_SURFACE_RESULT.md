# V6.7 P4 ray-terminated surface result

- Canonical run: `run://worldsim_v67/WS-V67-P4-RAY-TERMINATED-SURFACE-01/20260828T105253Z__ray-surface-s0-r1`
- Verdict: `rejected_task_untouched_ray_terminated_surface_repair`
- Failure: `V67-F01 active`

The frozen rule retained an exact same-Actor hit or a same-Actor hit within `0.512m` whose query voxel was marked by the
aggregated source `behind_hit` grid. It processed the unchanged P3 L0 actions and did not use target evidence for an action or
retention decision.

## Evidence

- units / Actor states / acted states: `72 / 517 / 258`
- baseline / repaired boundary points: `18,238 / 7,156`
- overall retention: `0.392368 < 0.40`
- conflict points: `1,003 -> 322`; reduction=`0.678963 >= 0.50`
- clean points: `17,235 -> 6,834`; retention=`0.396519 < 0.40`
- Actor / shell / ID-track-trajectory retention: `1 / 1 / 1`
- removed Actors / maximum hazard shift / scene yield: `0 / 0 / 1`
- exact / radius / source-behind-hit support counts: `112 / 8,030 / 1,018`
- gates: `7/9`; wall / RSS / GPU: `10.6065s / 0.6087GiB / false`

The failure is narrow but decisive: both preregistered retention gates remain binding and are not lowered. The raw
`behind_hit` grid is generated from original source-frame endpoints before Actor motion compensation, while the same-Actor
hit positions used by repair are motion-compensated into the target Actor frame. Their post-hoc intersection therefore loses
the ray-to-Actor association and is not the intended compensated ray continuation.

No rule, radius, action budget or gate sweep was run.
