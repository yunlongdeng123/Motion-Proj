# V6.7 P14 directional surface result and single recovery

P14 canonical=`run://worldsim_v67/WS-V67-P14-DIRECTIONAL-SURFACE-TRAIN-01/20260828T120000Z__directional-surface-s0-r1`;
verdict=`rejected_learned_residual_directional_surface_selection`.

The 284-D `512/256/128` head trained for 600 GPU epochs on 14,250 analytic-rejected points. In-sample ranking was perfect and
the in-sample threshold remained 0.5, but selection rescued 7,612 clean and 420 conflict points. Clean retention rose from
0.5478 to 0.9745 while conflict reduction collapsed from 0.4924 to 0.0939; 5/6 gates pass, so the method is rejected. The
selection denominator also included 654 points from Actors absent from the frozen action rows, explaining why the reported
analytic comparator was 0.4924 rather than the P4R canonical 0.5174, but it does not explain the learned collapse.

External recovery evidence: selective classification controls risk by abstaining to a target risk/coverage operating point;
Conformal Risk Control calibrates a monotone risk on held-out data; SENTRY shows that consistency under domain shift is useful
for selective decisions. P14R therefore makes one structural recovery: compute the rescue threshold from leave-one-training-scene-
out conflict scores, train the final model on all six training scenes, and evaluate only the exact action-eligible denominator.
Architecture, loss, analytic core, 1% training conflict-rescue quantile and selection gates remain unchanged. No selection score
is used to set the threshold.
