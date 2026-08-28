# P25 Coverage-Constrained Budget Compiler Result

Canonical run: `run://worldsim_v67/WS-V67-P25-COVERAGE-BUDGET-COMPILER-01/20260828T152000Z__coverage-budget-s0-r1`.

P25 trains a bounded case offset on nine development domains while freezing P20 within-case action order. On the V64 P4C
cohort, 1,000/1,152 actions and 89 cases were evaluable. The selected action count exactly matched the fixed-quarter baseline
(`243/243`), while authority covered 54 cases (`60.6742%`) with 0--6 actions per case.

| allocator | relative selected-cost reduction |
| --- | ---: |
| fixed P20 quarter per case | 0.312205 |
| P24 min-one adaptive budget | 0.594446 |
| P25 coverage-constrained budget | 0.694998 |

P25 improves over fixed P20 by `+0.382792` and over P24 by `+0.100552`; all eight scenes are non-increasing and all five
pre-registered gates pass. Wall time was `56.289s`, peak allocated GPU memory `0.04711GiB`, and peak RSS `1.4823GiB`.

The supported claim is limited to fixed-lattice Ego actions and future two-second visited world-state reliability. It does not
establish collision, planning, policy, closed-loop, population, or safety guarantees.
