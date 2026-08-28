# V6.7 P6 independent Actor package result

- Canonical run: `run://worldsim_v67/WS-V67-P6-INDEPENDENT-ACTOR-PACKAGE-01/20260828T110547Z__independent-package-s0-r1`
- Verdict: `supported_v67_actor_preserving_package`
- Failure delta: `none`

The package contains 119 unique Actors, 570 Actor states and 1,353,734 Actor primitives across 72 units. All Actor state and
metadata retention gates equal `1.0`; removed Actors and hidden-target fields equal `0`. Runtime model loading and
hazard-conditioned Actor existence remain disabled. Eight package files occupy 15,671,545 bytes. All 6/6 gates pass;
wall/RSS/GPU=`13.9384s/0.8299GiB/false`.

This is a compiler capability, not physical repair. P7 binds this canonical package and the P5 score rows without changing the
fixed 50% budget or any Actor/hazard contract.
