# P30 Horizon-Conditioned Authority Result

Canonical run: `run://worldsim_v67/WS-V67-P30-HORIZON-CONDITIONED-AUTHORITY-01/20260828T172000Z__horizon-conditioned-s0-r1`.

P30 trained 394 cases with explicit H=`1.0/2.0s` conditions. P10X was excluded from training and materialized at held-out
H=`1.5s`, yielding 717/864 eligible actions and 66 evaluable cases. The compiler selected exactly 176 actions, covered 42 cases
(`63.6364%`), and maintained at least 50% coverage in every context group.

Relative selected-cost reduction was `0.740743`, versus `0.235554` for fixed P20, a delta of `+0.505189`. Five of six scenes
were non-increasing; the sixth differed by only `+1.86e-9` but remains higher under the frozen exact comparison. All six gates
pass. Wall time was `39.661s`, peak allocated GPU memory `0.03954GiB`, and peak RSS `1.5052GiB`.

This supports held-out-horizon trajectory visited-state authority on a globally consumed cohort, not fresh or safety claims.
