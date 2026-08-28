# P45 Anchored Hybrid Nested Budget Freeze

P45 freezes P44, P31, and P20 on the new P6R H=`1.5s` cache. Quarter and half budgets use their own anchored hybrid scores
and P31 offsets. The low set is retained exactly while high-budget priorities extend it. At quarter budget the anchored
hybrid action scores are structurally identical to P20 scores.

Gates: exact totals at both budgets; strict low subset high; minimum group coverage `>=0.50` at both; nonnegative reduction
delta versus the corresponding P31 nested baseline at both; at least five non-increasing scenes at both. No training, refit,
budget, amplitude, weight, model, or gate sweep. New task condition on a consumed cohort only; no safety claim.
