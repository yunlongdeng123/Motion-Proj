# P38 Robust Conditioned Action Compiler Freeze

P38 follows the worst-group motivation of GroupDRO (ICLR 2020) and the cross-environment risk focus of REx (ICML 2021).
It changes exactly one P36 component: four development-domain losses are aggregated by a temperature `0.02` log-sum-exp
smooth maximum instead of mean plus variance. Features, budgets, horizons, model, soft top-k objective, auxiliary weights,
residual bound, optimizer, seed, and 6,000 epochs remain unchanged.

The one consumed P10X read is at `(budget=1/3,H=1.5s)`, with frozen P31 as comparator. Gates: exact total; minimum group
coverage `>=0.50`; reduction delta over P31 `>=0.005`; at least five non-increasing scenes. No aggregation-temperature,
model, loss, or gate sweep. No population, collision, planning, closed-loop, or safety claim; no hash/checksum/fingerprint.
