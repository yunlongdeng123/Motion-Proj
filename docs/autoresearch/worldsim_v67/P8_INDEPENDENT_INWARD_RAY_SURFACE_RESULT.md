# V6.7 P8 independent inward-ray surface result

- Canonical run: `run://worldsim_v67/WS-V67-P8-INDEPENDENT-INWARD-RAY-SURFACE-01/20260828T110941Z__independent-inward-s0-r1`
- Verdict: `supported_independent_legacy_motion_compensated_inward_ray_surface_confirmation`
- Failure delta: `none`

On the independent V65 P2 legacy cohort, P8 processes 285/570 acted states. It retains 10,882/19,654 boundary points
(`0.553679`), reduces conflict points from 1,021 to 509 (`0.501469`), and retains 10,373/18,633 clean points (`0.556700`).
Actor, collision shell and ID-track-trajectory retention are all `1`; removed Actors and hazard shift are `0`; scene yield is
`1`. Exact/radius/inward support counts are `49/8,111/4,913`. All 9/9 gates pass; wall/RSS/GPU=
`10.7403s/0.5892GiB/false`.

Together with P4R, this supports the same physical rule on two surface-task-independent legacy cohorts. It does not establish a
fresh-population claim. P9 therefore freezes a new unprocessed, repository-unmentioned six-scene cohort before preparing any
scene or reading any target quality.
