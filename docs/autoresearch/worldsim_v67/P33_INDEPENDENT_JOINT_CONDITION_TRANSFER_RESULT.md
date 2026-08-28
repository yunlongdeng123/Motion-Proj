# P33 Independent Joint-Condition Transfer Result

Canonical: `run://worldsim_v67/WS-V67-P33-INDEPENDENT-JOINT-CONDITION-TRANSFER-01/20260828T182000Z__second-joint-cohort-s0-r1`.

P4C was excluded from training and materialized at H=`1.5s`, yielding 973/1,152 eligible actions and 89 cases. At budget
`1/3`, the compiler selected exactly 315 actions, covered `69.6629%` of cases, and maintained minimum group coverage `50%`.

Reduction was `0.698243`, versus fixed P20 `0.258655` (`+0.439588`). All eight scenes were non-increasing and all six gates
pass. Wall/GPU/RSS=`44.414s/0.03954GiB/1.4932GiB`. Joint condition transfer is now supported on P10X and P4C, both globally
consumed legacy cohorts; no fresh-population or safety claim.
