# V6.7 P11 fresh Actor package result

Canonical=`run://worldsim_v67/WS-V67-P11-FRESH-ACTOR-PACKAGE-01/20260828T114417Z__fresh-package-s0-r1`;
verdict=`supported_v67_actor_preserving_package`; 6/6 gates pass.

The 72-unit package retains all 938 Actor states across 186 Actors and 1,868,749 primitives. Actor state retention and
metadata completeness are both 1.0; Actor removals and hidden-target fields are both zero. Runtime model loading, hidden-target
loading and hazard-conditioned Actor existence are disabled. Eight package files occupy 17,405,615 bytes; wall/RSS are
23.1121s/0.8441GiB.

P12 mechanically binds this package and the P10 scores at the unchanged 50% action budget. No new failure is registered.
