# P32 Joint Nested Budget-Horizon Authority Freeze

P32 uses the P31 joint budget/H training set and evaluates at held-out H=`1.5s`. The exact quarter-budget action set is formed
first and must be a strict subset of the exact half-budget set. Both budgets maintain at least 50% coverage per context group.
No model, condition, nesting, coverage, loss, or gate sweep; P10X is consumed; no hash/checksum/fingerprint.
