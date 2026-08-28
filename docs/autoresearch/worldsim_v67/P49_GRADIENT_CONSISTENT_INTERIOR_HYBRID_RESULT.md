# P49 Gradient-Consistent Interior Hybrid Result

Canonical: `run://worldsim_v67/WS-V67-P49-GRADIENT-CONSISTENT-INTERIOR-HYBRID-01/20260829T010000Z__gradient-consistent-interior-s0-r1`.

Twelve consumed development domains and three budgets produced 3,240 conditioned cases / 36,237 action rows. The fixed
final-layer domain-gradient direction penalty ended at dispersion `0.003184`; residual RMS was `0.017360`.

On newly materialized P3C H=`1.5s`, P49 selected exactly `236/236` actions, covered `0.70` of cases, retained minimum group
coverage `0.666667`, and improved all five evaluable scenes. P49/P31/fixed-P20 reduction was
`0.710322/0.695815/0.392525`; delta over P31 was `+0.014506`. All four gates passed and verdict is
`supported_gradient_consistent_interior_hybrid`.

Wall/peak GPU/peak RSS: `450.118s / 0.11199GiB / 1.35390GiB`. The claim remains Fishr-inspired final-layer update-direction
consistency, not full Fishr equivalence or a fresh-population/planning/safety result.
