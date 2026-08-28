# P36 Conditioned Action Compiler Result

Canonical run: `run://worldsim_v67/WS-V67-P36-CONDITIONED-ACTION-COMPILER-01/20260828T192000Z__conditioned-topk-s0-r1`.

The budget/H-conditioned scorer trained on 788 conditioned cases and 8,820 action rows. Final soft selected cost was
`0.056551`, pairwise loss `0.256469`, regression loss `0.156436`, and residual RMS `0.038763`.

On consumed P4C at the unseen `(budget=1/3,H=1.5s)` pair, it selected exactly `315/315` actions, covered 63/89 cases
(`0.707865`), retained minimum group coverage `0.50`, and improved all or tied all eight scenes. Relative cost reduction
was `0.719901`, versus `0.698243` for frozen P33 and `0.258655` for fixed P20: deltas `+0.021658` and `+0.461246`.
All four gates passed; verdict `supported_conditioned_action_compiler`.

This supports direct differentiable selected-cost training under budget/H conditions on consumed method-selection evidence.
It does not establish fresh-population, collision, planning, closed-loop, or safety performance.
