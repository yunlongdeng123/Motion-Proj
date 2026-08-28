# P34 Heteroscedastic Authority Result

Canonical run: `run://worldsim_v67/WS-V67-P34-HETEROSCEDASTIC-AUTHORITY-01/20260828T184000Z__heteroscedastic-s0-r1`.

The bounded heteroscedastic head trained on 788 joint budget/horizon case rows. Its confirmation scale had
Spearman `0.190272` with absolute mean error, passing the frozen `0.15` diagnostic gate. The consumed P10X
selection nevertheless obtained relative cost reduction `0.610037`, below the frozen P31 mean compiler's
`0.690636` by `-0.080599`. The fixed P20 reduction was `0.190718`.

Exact total budget, minimum group coverage, uncertainty tracking, and five-scene support passed; improvement over
the mean compiler failed. Verdict: `rejected_heteroscedastic_conservative_authority`.

Interpretation: a weakly informative aleatoric scale does not imply that adding one scale unit to the priority improves
a constrained selection decision. Scale bounds, uncertainty weight, loss, model, and gates remain unswept. The
aleatoric-priority family is closed. No epistemic, calibrated-interval, OOD, collision, planning, closed-loop, or safety claim.
