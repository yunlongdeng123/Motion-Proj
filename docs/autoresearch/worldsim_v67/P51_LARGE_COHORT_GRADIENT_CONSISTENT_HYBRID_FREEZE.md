# P51 Large-Cohort Gradient-Consistent Hybrid Freeze

P51 keeps every P49 method hyperparameter, including the double endpoint anchor and fixed `.01` final-layer domain-gradient
direction penalty. The only development change is adding consumed P2V H=`1.5s` as domain 13. Six thousand GPU epochs run
while P6E H=`1.5s` is materialized for the first time across 16 scenes / 192 source cases.

The one decision read is budget `1/3`. The 16 ordered scenes are frozen into four contiguous four-scene groups. Gates are
exact total, minimum group coverage `0.50`, delta over P31 `+0.005`, and at least 12 non-increasing scenes.

No gradient-weight/anchor/peak/model/loss/group/gate sweep; no full-Fishr equivalence, fresh-population, collision, planning,
policy, closed-loop, or safety claim; no hash/checksum/fingerprint.
