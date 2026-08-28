# P37 Conditioned Action Transfer Result

Canonical run: `run://worldsim_v67/WS-V67-P37-CONDITIONED-ACTION-TRANSFER-01/20260828T194500Z__conditioned-transfer-s0-r1`.

The exact P36 model and normalizer were applied to consumed P10X without training or refitting. It selected exactly
`236/236` actions, covered 52/66 cases (`0.787879`), retained minimum group coverage `0.666667`, and had five of six
non-increasing scenes. Coverage exceeded frozen P31's `0.727273`, with minimum group `0.583333`.

Selected cost reduction was `0.656886`, below frozen P31's `0.690636` by `-0.033750`; fixed P20 achieved `0.190718`.
The decision-improvement gate failed while exact, group, and scene gates passed. Verdict:
`rejected_second_cohort_conditioned_action_transfer`.

Thus P36's P4C decision gain does not transfer unchanged to P10X. Higher case coverage is not treated as a substitute for
selected-cost utility. This consumed transfer makes no fresh-population, planning, closed-loop, or safety claim.
