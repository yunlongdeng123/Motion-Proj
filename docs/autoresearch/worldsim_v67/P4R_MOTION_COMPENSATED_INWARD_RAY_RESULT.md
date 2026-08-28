# V6.7 P4R motion-compensated inward-ray result

- Canonical run: `run://worldsim_v67/WS-V67-P4R-MOTION-COMPENSATED-INWARD-RAY-01/20260828T105920Z__inward-ray-s0-r1`
- Verdict: `supported_task_untouched_motion_compensated_inward_ray_surface_repair`
- Failure delta: `V67-F01 resolved_by_single_structural_recovery`

## Result

- units / Actor states / acted states: `72 / 517 / 258`
- baseline / repaired boundary points: `18,238 / 9,652`
- overall retention: `0.529225`
- conflict points: `1,003 -> 484`; reduction=`0.517448`
- clean points: `17,235 -> 9,168`; retention=`0.531941`
- Actor / shell / ID-track-trajectory retention: `1 / 1 / 1`
- removed Actors / maximum hazard shift / scene yield: `0 / 0 / 1`
- exact / one-voxel / inward-ray support points: `112 / 8,030 / 4,778`
- gates: `9/9`; wall / RSS / GPU: `10.2274s / 0.6089GiB / false`

P4R changed only the directional association of the frozen one-voxel support: the inward half-ball is constructed behind the
nearest motion-compensated same-Actor hit in the target frame. It did not read target evidence for the rule, alter P3 actions,
change the radius or lower any gate.

This supports a task-untouched-but-globally-consumed legacy capability. An independent cohort is still required before any
cross-cohort surface claim, and no fresh-population, RL, planning, policy or safety claim is authorized.
