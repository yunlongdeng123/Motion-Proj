# P11D Collision-Critic Shift Diagnostic Closeout

Date: 2026-08-27  
Task: `WS-V64-P11D-COLLISION-CRITIC-SHIFT-DIAGNOSTIC-01`

Canonical run:

`run://worldsim_v64/WS-V64-P11D-COLLISION-CRITIC-SHIFT-DIAGNOSTIC-01/20260827T040000Z__collision-critic-shift-s0-r1`

The rows-only diagnostic found that unsafe prevalence increased from 0.070513 in calibration to 0.109776 in evaluation. For the
UNC-verified critic, unsafe q20 and median scores decreased by 0.053282 and 0.137452, while the safe median changed by only
-0.000025. Average precision fell from 0.247103 to 0.137399 and AUROC from 0.711648 to 0.562740. Naive augmentation also lost
0.090305 AP and 0.093099 AUROC.

The P11R failure is therefore not explained by a global class-prior or operating-point shift alone. Unsafe ranking itself degrades
across cohorts, especially for the uncertainty-verified arm. Another threshold calibration is not a supported recovery.

This run had no gate, GPU, native/evidence reread, model or threshold change, policy selection, or P11 reopening. Wall time was
0.098615 s and peak RSS 0.195736 GiB. The result is post-hoc failure characterization only.
