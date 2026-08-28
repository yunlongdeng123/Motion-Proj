# P39 Expanded-Domain Conditioned Action Freeze

P39 restores the P36 mean-plus-domain-variance objective and changes only the development denominator. Consumed P4C and
P10X H=`1.5s` caches join the original four domains; budgets expand from `.25/.50` to `.25/1/3/.50`. The architecture,
soft top-k and auxiliary losses, temperatures, residual bound, optimizer, seed, and 6,000 epochs remain P36-exact.

P3C action targets are excluded from P39 training. The single third-cohort read is P3C at `(budget=1/3,H=2s)`, compared
with frozen P31 under exact-total and three-group coverage constraints. Gates: exact total; minimum group coverage
`>=0.50`; reduction delta over P31 `>=0.005`; at least four of five non-increasing scenes.

This is action-target-untouched but globally consumed cohort evidence, not fresh population validation. No cohort-combination,
model, loss, temperature, or gate sweep; no planning or safety claim; no hash/checksum/fingerprint.
