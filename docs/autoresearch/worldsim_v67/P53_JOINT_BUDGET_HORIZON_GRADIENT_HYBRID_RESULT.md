# P53 Joint Budget-Horizon Gradient Hybrid Result

Canonical: `run://worldsim_v67/WS-V67-P53-JOINT-BUDGET-HORIZON-GRADIENT-HYBRID-01/20260829T040000Z__joint-budget-horizon-gradient-s0-r1`.

Fourteen domains and four budgets trained 5,320 conditioned cases / 59,608 action rows. Final gradient-direction dispersion
was `0.002873`; residual RMS was `0.017443`. On jointly unseen budget `.375` and below-range H=`.8s`, 61 P10X cases were
evaluable; selection was exact `218/218`, coverage `0.655738`, minimum group coverage `0.541667`, and 6/6 scenes improved.

P53/P31/fixed reduction was `0.733916/0.724912/0.214324`; delta over P31 was `+0.009004`. All four gates passed;
verdict=`supported_joint_budget_horizon_gradient_hybrid`. Wall/peak GPU/peak RSS was
`503.231s / 0.18637GiB / 1.35929GiB`.
