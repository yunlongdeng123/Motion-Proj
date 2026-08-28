# P36 Conditioned Action Compiler Freeze

P36 migrates the continuous-ranking idea of NeuralSort (ICLR 2019) and SoftSort (ICML 2020) to the existing compiler.
It adds selected budget and future horizon H to the 13 P20 action features, then trains a bounded residual scorer directly
against soft top-k selected visited-state cost. Small pairwise and regression terms stabilize ordering but are not primary.

Training uses four existing domains at H=`1/2s` and both selected fractions `.25/.50`. The single consumed P4C read is at
the unseen pair `(1/3,1.5s)`. Its scores enter the same exact-total and four-group coverage selector used by P33; the frozen
P33 joint budget/H compiler is the comparator.

Gates: exact total; minimum group coverage `>=0.50`; relative-cost-reduction delta over P33 `>=0.005`; at least six
non-increasing scenes. Temperature, residual bound, architecture, loss weights, and gates are fixed with no sweep. No
population, collision, planning, closed-loop, or safety claim. No hash/checksum/fingerprint.
