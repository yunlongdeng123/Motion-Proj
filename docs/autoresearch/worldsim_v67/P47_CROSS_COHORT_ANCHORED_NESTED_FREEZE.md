# P47 Cross-Cohort Anchored Nested Freeze

P47 freezes P46, P31, and P20 on P10R4 H=`1.5s`. Quarter and half selections use their own anchored scores and P31 offsets;
the low set is preserved exactly while high priorities extend it. Quarter action scores equal P20 by construction.

Gates: exact both totals; strict low subset high; minimum group coverage `>=0.50` at both; nonnegative reduction delta versus
the corresponding P31 nested baseline at both; at least six non-increasing scenes at both. No training, refit, budget, anchor,
weight, model, or gate sweep. No fresh population, planning, closed-loop, or safety claim.
