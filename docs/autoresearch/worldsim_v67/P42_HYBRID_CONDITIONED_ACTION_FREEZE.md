# P42 Hybrid Conditioned Action Freeze

P42 composes two supported mechanisms instead of replacing either. Frozen P31 case offsets allocate the exact total across
cases. Frozen P20 supplies each action's base score. A new budget/H-conditioned head outputs a residual centered within each
case, so training refines within-case ordering without learning a free case-allocation substitute.

The head trains on nine consumed domains and three budgets using the P41 soft top-k objective. P6R action targets are
excluded from P20, P31, P41, and P42 training. The single P6R read is `(budget=1/3,H=2s)`, with frozen P31 as comparator.

Gates: exact total; minimum group coverage `>=0.50`; reduction delta over P31 `>=0.005`; at least five of seven
non-increasing scenes. Composition weights are fixed by direct addition; no weight, architecture, loss, temperature, or gate
sweep. Globally consumed evidence only; no planning, closed-loop, collision, or safety claim; no hash/checksum/fingerprint.
