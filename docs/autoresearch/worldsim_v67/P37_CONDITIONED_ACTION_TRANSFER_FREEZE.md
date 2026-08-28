# P37 Conditioned Action Transfer Freeze

P37 loads the exact P36 model and normalizer without training or refitting. It scores consumed P10X at
`(budget=1/3,H=1.5s)` and uses the same exact-total, case-coverage, and three-group constraints as P31.

The comparator is frozen P31. Gates: exact total; minimum group coverage `>=0.50`; relative-cost-reduction delta over P31
`>=0.005`; at least five non-increasing scenes. No parameter, temperature, model, loss, or gate sweep. This is a second
consumed-cohort transfer, not a fresh population confirmation, and makes no planning or safety claim.
