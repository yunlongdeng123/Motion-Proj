# P35 Ensemble Authority Freeze

P35 trains exactly three independent `BoundedCaseOffset` members with seeds `0/1/2`, the same 788-row joint
budget/horizon training set, architecture, and loss used by P31/P33. Members run sequentially on the single RTX 3090.

On consumed P4C at `(budget=1/3,H=1.5s)`, priority is frozen as `ensemble_mean + 1.0 * ensemble_std`. The comparator
is the frozen P33 mean compiler under identical exact-total, case-coverage, and four-group coverage constraints.

Gates: exact total; minimum group coverage `>=0.50`; disagreement-error Spearman `>=0.10`; reduction delta over P33
`>=0.005`; at least six non-increasing scenes. No member-count, seed, weight, model, loss, or gate sweep. A failure closes
the uncertainty-for-decision family. Disagreement is only an epistemic proxy, not a calibrated posterior, OOD guarantee,
collision model, planner, closed-loop policy, or safety certificate. No hash/checksum/fingerprint.
