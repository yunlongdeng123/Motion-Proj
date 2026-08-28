# P43 Hybrid Nested Budget Freeze

P43 freezes P42, P31, and P20. On consumed P6R it evaluates quarter and half action budgets. Each budget receives its own
budget-conditioned hybrid action scores and P31 case offsets; the low selection is then preserved exactly while the high
budget is extended using high-budget priorities.

Gates: exact low and high totals; strict low subset of high; minimum group coverage `>=0.50` at both budgets; nonnegative
reduction delta versus the corresponding frozen-P31 nested baseline at both budgets; at least five non-increasing scenes at
both budgets. No training, refit, budget, blend-weight, model, or gate sweep. This is a structural consumed-cohort claim only.
