# P44 Low-Budget Anchored Hybrid Freeze

P44 applies a fixed budget amplitude to the P42 case-centered residual:
`clip((selected_fraction - 0.25) / (0.50 - 0.25), 0, 1)`. At quarter budget the action scores are exactly frozen P20;
at half budget the full hybrid residual is active; intermediate budgets interpolate continuously. P31 case allocation stays frozen.

Nine consumed domains and three budgets train the residual. In parallel, P6R is rematerialized at the previously unread
H=`1.5s` task condition. Confirmation uses budget `1/3`, frozen P31 comparison, exact total, minimum group coverage `0.50`,
reduction delta `>=0.005`, and at least five non-increasing scenes.

The first materializer entry omitted its required run ID and exited during argparse before cache creation or target reading;
the same frozen config resumed with a run ID. No anchor, full fraction, amplitude, model, loss, or gate sweep. Consumed cohort
with a new task condition only; no fresh population, planning, closed-loop, or safety claim.
