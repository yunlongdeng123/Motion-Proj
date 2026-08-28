# P29 Nested-Budget Authority Result

Canonical run: `run://worldsim_v67/WS-V67-P29-NESTED-BUDGET-AUTHORITY-01/20260828T164000Z__nested-budget-s0-r1`.

P29 trained 1,722 case-budget rows while excluding P4C. On 89 evaluable P4C cases, the low and high action totals were exactly
`243/243` and `494/494`. Every one of the 243 low-budget actions was retained at the high budget. Minimum group coverage was
`0.50` at the low budget and `0.708333` at the high budget.

| budget | nested compiler reduction | fixed P20 reduction | delta |
| --- | ---: | ---: | ---: |
| 0.25 | 0.758868 | 0.312205 | +0.446663 |
| 0.50 | 0.387925 | 0.205116 | +0.182809 |

Both budgets are non-increasing on all eight scenes and all seven gates pass. Wall time was `51.817s`, peak allocated GPU
memory `0.01711GiB`, and peak RSS `1.1806GiB`.

This supports nested budget-path authority on a consumed cohort, not fresh generalization or collision/planning/safety claims.
