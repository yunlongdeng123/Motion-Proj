# P41 Continual-Domain Conditioned Action Freeze

P41 is the terminal action-scorer domain-expansion trial. Following DomainBed's strong carefully implemented ERM baseline,
it adds consumed P3C and P10R4 to P39 development, yielding eight domains and three budgets. All architecture, loss,
temperature, residual-bound, optimizer, seed, and epoch choices remain P36/P39-exact.

P10R2 action targets are excluded from P20, P31, P39, and P41 training. The single heldout task read is at
`(budget=1/3,H=2s)`, compared with frozen P31 under exact-total and four-group constraints. Gates: exact total; minimum
group coverage `>=0.50`; reduction delta over P31 `>=0.005`; at least six non-increasing scenes.

There is no cohort-combination or parameter sweep. If the decision-gain gate fails, the conditioned action-scorer
cross-cohort family closes. Globally consumed evidence only; no fresh population, planning, closed-loop, or safety claim.
