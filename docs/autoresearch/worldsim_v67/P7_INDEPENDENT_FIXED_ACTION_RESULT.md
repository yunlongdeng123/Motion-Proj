# V6.7 P7 independent fixed-action result

- Canonical run: `run://worldsim_v67/WS-V67-P7-INDEPENDENT-FIXED-ACTIONS-01/20260828T110742Z__independent-actions-s0-r1`
- Verdict: `supported_v67_fixed_budget_action_set`
- Failure delta: `none`

The fixed budget is 285/570 Actor states (`0.5`). L0 handles 191/312 conflicts for exposure reduction `0.612179`; q0
handles 203/312 (`0.650641`) and the oracle handles 285/312 (`0.913462`). L0 retains every Actor, removes none, changes no
hazard proxy, emits exactly half of local geometry and yields all six scenes. All 6/6 gates pass; wall/RSS/GPU=
`0.02574s/0.4857GiB/false`.

The q0 pooled comparator is stronger, consistent with P5. L0 nevertheless exceeds 0.5 conflict reduction in all six scenes;
q0 is below 0.5 in scene-0072 and scene-0443. This is descriptive robustness evidence, not a post-hoc gate. P8 keeps L0 because
the full learned-arm confirmation chain was frozen before P5 quality was read.
