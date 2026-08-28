# P41 Continual-Domain Conditioned Action Result

Canonical run: `run://worldsim_v67/WS-V67-P41-CONTINUAL-DOMAIN-CONDITIONED-ACTION-01/20260828T211000Z__continual-domain-topk-s0-r1`.

Eight domains and three budgets produced 2,124 conditioned cases and 23,760 action rows. Final soft selected cost was
`0.061272`, residual RMS `0.037539`, and domain losses ranged from `0.071782` to `0.115753`.

On action-target-untouched P10R2, P41 selected exactly `365/365` actions, covered 68/96 cases (`0.708333`), retained
minimum group coverage `0.50`, and improved or tied all eight scenes. Reduction was `0.747149`, versus `0.743093` for P31
and `0.402325` for fixed P20. The delta over P31 was positive `+0.004055`, but below the frozen `+0.005` gate by
`0.000945`. Verdict remains `rejected_continual_domain_conditioned_action`.

The gate is not relaxed after observation. Per the terminal freeze, the pure conditioned action-scorer cross-cohort family
is closed. The narrow positive descriptive delta is retained for technical-report analysis.
