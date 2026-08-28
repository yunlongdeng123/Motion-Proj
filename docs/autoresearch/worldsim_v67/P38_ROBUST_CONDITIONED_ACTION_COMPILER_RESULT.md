# P38 Robust Conditioned Action Compiler Result

Canonical run: `run://worldsim_v67/WS-V67-P38-ROBUST-CONDITIONED-ACTION-COMPILER-01/20260828T200000Z__robust-conditioned-topk-s0-r1`.

The fixed-temperature smooth worst-domain objective ended with domain losses from `0.085743` to `0.100548`, soft selected
cost `0.056995`, and residual RMS `0.038812`. On consumed P10X it selected exactly `236/236` actions, covered 53/66 cases
(`0.803030`), retained minimum group coverage `0.708333`, and had five of six non-increasing scenes.

Relative cost reduction was only `0.629974`, versus `0.690636` for frozen P31 and `0.190718` for fixed P20. The delta
over P31 was `-0.060662`; the key decision gate failed. Verdict `rejected_robust_conditioned_action_compiler`.

Worst-domain aggregation increased coverage but degraded action ordering, including relative to the frozen P36 transfer.
No temperature, aggregation, architecture, loss, or gate sweep is permitted; objective reweighting is closed.
