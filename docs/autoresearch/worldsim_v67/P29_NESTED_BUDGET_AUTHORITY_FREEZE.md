# P29 Nested-Budget Authority Freeze

P29 jointly trains at fractions `0.25` and `0.50` while excluding P4C from its ten training domains. On the consumed P4C cache,
the low-budget set is selected first. The high-budget set is then formed by retaining every low-budget action and extending by
the high-budget conditioned priority until the exact half-budget total is reached.

Both sets must meet at least 50% coverage in every pre-existing two-scene context group. P20 within-case order remains frozen.
Gates require exact totals at both budgets, strict low-subset-of-high nesting, group coverage at both budgets, low/high cost
reduction at least `0.55/0.30`, improvement over fixed P20 at least `0.10/0.05`, and at least six non-increasing scenes at both
budgets.

No fraction, nesting rule, architecture, offset, coverage, group, loss, or gate sweep is allowed. P4C is globally consumed, so
the claim is budget-path mechanism only. No hash, checksum, or fingerprint is introduced.
