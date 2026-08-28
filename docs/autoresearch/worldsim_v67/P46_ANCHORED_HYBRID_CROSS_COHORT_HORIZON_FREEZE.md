# P46 Anchored Hybrid Cross-Cohort Horizon Freeze

P46 retains P44's architecture, P20 base, P31 allocator, quarter-to-half residual amplitude, losses, optimizer, and gates.
The only training-denominator change is adding consumed P6R H=`1.5s` as a tenth domain.

In parallel with GPU training, P10R4 is materialized at H=`1.5s` for the first time. P10R4 H=`2s` appears in development,
but the H=`1.5s` task target is unread before this run. Confirmation uses budget `1/3`, exact total, minimum group coverage
`0.50`, reduction delta over P31 `>=0.005`, and at least six non-increasing scenes.

This is cross-cohort task-condition replication, not a fresh source-population confirmation. No anchor, architecture, model,
loss, temperature, or gate sweep; no planning or safety claim; no hash/checksum/fingerprint.
