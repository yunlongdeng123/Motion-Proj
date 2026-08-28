# V6.7 P14R crossfit directional surface result

Canonical=`run://worldsim_v67/WS-V67-P14R-CROSSFIT-DIRECTIONAL-SURFACE-01/20260828T121000Z__crossfit-directional-s0-r1`;
verdict=`rejected_learned_residual_directional_surface_selection`.

Leave-one-scene-out calibration raises the rescue threshold to `0.9999187` and fixes the action denominator exactly: the analytic
comparator returns to the P4R values (`0.517448` conflict reduction, `0.531941` clean retention). The final all-scene model still
rescues 6,382 clean and 284 conflict points on selection, producing `0.234297` conflict reduction and `0.902234` clean retention.
Five of six gates pass, so the learned point-rescue family closes as a negative after its single recovery (`V67-F04 terminal`).

The next method changes the prediction object, as preregistered by the user: given an Ego trajectory over two seconds, predict
the reliability/cost of the world and Actor states visited by that trajectory. This follows planning-oriented conditioning in
[UniAD](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html),
trajectory-conditioned occupancy in [SparseWorld-TC](https://openaccess.thecvf.com/content/CVPR2026/html/Du_SparseWorld-TC_Trajectory-Conditioned_Sparse_Occupancy_World_Model_CVPR_2026_paper.html),
and identity-aware dynamic occupancy in [Occupancy Flow Fields](https://waymo.com/research/occupancy-flow-fields-for-motion-forecasting-in-autonomous-driving/).
