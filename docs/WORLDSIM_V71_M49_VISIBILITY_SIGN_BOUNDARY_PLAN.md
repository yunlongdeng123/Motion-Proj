# WorldSim V7.1 M49 — Exact visibility attenuation sign boundary

## Analytic boundary

For a frozen ray, let primitive `j` have total unnormalized mass `D_j`, pre-boundary mass `N_j`, component CDF `C_j=N_j/D_j`, responsibility `r_j=w_j D_j / sum_a w_a D_a`, and global pre-boundary CDF `C`. Direct differentiation gives

`dC / d log(w_j) = r_j (C_j - C)`.

For attenuation `d log(w_j) <= 0`, the safety boundary CDF decreases only when `C_j >= C`; it increases when the attenuated component lies later than the current mixture (`C_j < C`). Therefore “lower visibility is safer” is not a monotone property of a jointly normalized categorical surface measure.

## Frozen audit

- Reuse the 66 exposed holdout Actors and exact 64-bin M45 oriented categorical measure.
- Load M48 visibility without retraining and compare the exact CDF change against its first-order derivative decomposition.
- Report favorable/adverse child attenuation pressure, rays for which uniform child attenuation has either sign, actual CDF increase/decrease, and early-label additions/removals for all/hazard/clear.
- This milestone has no gate, model selection, training, threshold tuning, or external read. It proves an interpretability and safety boundary; it cannot rescue M48.

The result determines the paper claim and permanently closes learned child attenuation in V7.1.
