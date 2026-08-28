# V6.7 P13 fresh inward-ray surface result

Canonical=`run://worldsim_v67/WS-V67-P13-FRESH-INWARD-RAY-SURFACE-01/20260828T115200Z__fresh-inward-s0-r1`;
verdict=`supported_fresh_empirical_motion_compensated_inward_ray_surface_confirmation`; 9/9 gates pass.

Across 72 units and 938 Actor states, the fixed 469-state action set retains 16,929/30,529 boundary points. Conflict points
fall from 1,812 to 853 (`0.529249` reduction), while overall/clean retention are `0.554522/0.559808`. Actor, collision-shell,
identity/track/trajectory and hazard contracts remain exact; zero Actors are removed and all six scenes yield repaired geometry.

This closes the fresh empirical capability chain, not a population/RL/safety claim. P14 moves to actual GPU training: the
analytic inward-ray core is frozen, and a differentiable residual head may only rescue high-confidence clean points rejected by
that core. P13 is not used for training or model selection.
