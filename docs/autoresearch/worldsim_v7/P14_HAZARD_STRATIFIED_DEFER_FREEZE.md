# P14 Hazard-Stratified Defer-to-Query Freeze

## Question

Does a favorable aggregate risk--utility point remain favorable for hazardous Actors, or is the apparent safety boundary created by coverage and stratum mixing?

## Literature migration

- SelectiveNet (ICML 2019) defines selective risk jointly with coverage, so conditional risk cannot stand alone.
- Learning to Defer (NeurIPS 2018) evaluates the complete two-stage system, which maps here to compiled repair followed by the original-query fallback.
- Selective Regression under Fairness Criteria (ICML 2022) shows that abstention can improve aggregate performance while worsening a subgroup. We migrate only its subgroup-accounting warning; `hazardous` is a physical project proxy, not a demographic attribute.
- Conformal Risk Control (ICLR 2024) is not invoked: this audit neither chooses a monotone threshold nor assumes cross-domain exchangeability.

## Frozen proposition

For Actor `i`, selector `s_i`, query Chamfer `q_i`, compiled Chamfer `c_i`, and newly introduced visible failure `v_i`, the defer-to-query result is

`L_i = s_i c_i + (1-s_i) q_i`.

Within each frozen stratum `z in {hazardous, clear}`:

- composite gain `G_z = mean_z[s_i(q_i-c_i)] = coverage_z * selected_gain_z`;
- introduced failure mass `M_z = mean_z[s_i v_i] = coverage_z * selected_risk_z`;
- the population quantity is the Actor-share-weighted sum over the two strata.

These are exact finite-sample accounting identities, not probabilistic guarantees.

## Inputs and policies

- Only P13 canonical `COMPOSITE_POLICY_ROWS.jsonl` and `summary.json` are read.
- Policies remain exactly `query_only`, `always_repair`, P4, P6-C, provenance, visibility, and P4-and-visibility.
- Primary interpretation is restricted to always-repair, P4, P6-C, and P4-and-visibility.
- No dataset, checkpoint, threshold, score, model, or compiler is read or changed.

## Report

For every policy and stratum, report repair coverage, conditional visible-failure risk, population introduced-failure mass, conditional Chamfer gain, composite gain, and the two identity residuals. Also report hazardous-Actor share, hazardous share of introduced failures, burden amplification, and hazardous share of composite gain when total gain is positive.

No pass/fail gate is added. A new failure is registered only if the audit contradicts an existing V7 claim; otherwise it is a descriptive safety boundary and the next available failure remains `V7-F24`.

## Claim boundary

The `hazardous` field is the existing deterministic proxy based on frozen Actor/world state. It is not collision ground truth, a planner outcome, a demographic group, or a real-road safety label. P14 makes no conformal, exchangeability, causal, closed-loop, or deployment-safety claim.
