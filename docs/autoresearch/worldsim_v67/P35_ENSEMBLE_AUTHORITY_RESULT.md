# P35 Ensemble Authority Result

Canonical run: `run://worldsim_v67/WS-V67-P35-ENSEMBLE-AUTHORITY-01/20260828T190000Z__ensemble-authority-s0-r1`.

Three independently initialized members each trained for 5,000 GPU epochs on 788 joint-condition rows. Disagreement had
Spearman `0.144178` with absolute ensemble-mean error, passing the frozen `0.10` diagnostic gate. Mean disagreement was
only `3.53e-6` and maximum disagreement `4.90e-5`; all members reached residual RMS approximately `0.05`.

The fixed mean-plus-one-disagreement priority selected exactly the same 315 actions as the frozen P33 mean compiler.
Both obtained relative cost reduction `0.698243`; delta was `0.0`, below the required `0.005`. Exact budget, minimum group
coverage `0.50`, uncertainty tracking, and 8/8 scene support passed. Verdict: `rejected_ensemble_disagreement_authority`.

The bounded full-batch members collapsed to the same boundary solution, so their weak diagnostic correlation did not create
a decision-relevant ordering change. No member, seed, weight, architecture, loss, or gate sweep is allowed. The entire
uncertainty-for-decision priority family is closed; no posterior calibration, OOD, planning, closed-loop, or safety claim.
