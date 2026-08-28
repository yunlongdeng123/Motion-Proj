# V6.7 P24 adaptive budget compiler freeze

P24 freezes P20 within-case action order. A 16-hidden case offset bounded to `±0.05` is trained on eight development domains to
calibrate cross-case slot priority. Confirmation allocates at least one and at most five actions per case, while total selected
actions exactly equal the fixed 0.25-per-case P20 baseline budget.

The offset model is frozen before materializing V64 P6R scenes `1023/1105/0903/0451/0981/0537/0789/0157`. Gates require exact
total budget equality, adaptive reduction >=0.40, gain over fixed P20 >=0.03 and six non-increasing scenes.

One run only; no maximum-action, offset-bound, architecture, fraction or gate sweep. This is task-conditioned action-budget
allocation, not a planner policy, collision, closed-loop or safety claim.
