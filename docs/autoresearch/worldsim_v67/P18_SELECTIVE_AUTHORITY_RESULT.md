# V6.7 P18 selective authority result

Canonical run: `run://worldsim_v67/WS-V67-P18-SELECTIVE-AUTHORITY-COMPILER-01/
20260828T130500Z__selective-authority-s0-r1`.

The seven-feature monotone head was trained for 5,000 GPU epochs on P10V/P10X case benefit while qmean action order remained
immutable. P9 had 71 evaluable cases; the frozen 0.50 authority fraction selected 35.

Authorized all-action cost was 0.072745 and bottom-quartile qmean-selected cost was 0.037254, a 0.487876 reduction. Ungated
qmean reduction was 0.418184, so selective authority added 0.069693. Positive-benefit rate was 0.914286, benefit Spearman was
0.619743, and all six scenes improved. All four gates passed.

This is consumed-cohort method selection. P19 freezes the artifact and confirms once on the independent V65 P2 cohort.
