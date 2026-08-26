# P11 Bounded Collision-Critic Freeze

Date: 2026-08-27  
Branch: `research/worldsim-v6.4-native-uq`  
Task/Hypothesis: `WS-V64-P11-BOUNDED-COLLISION-CRITIC-01 / WS-V64-H-P11-001`

## Why this bounded migration

P10R4 removed the unresolved fixed-denominator relative lock, but it did not establish physical collision or closed-loop
authority. The first P11 stage therefore does not train a large neural world model and does not install another simulator.

The migration uses:

- Waymax's fixed counterfactual rollout and overlap/progress/kinematic metric separation;
- nuPlan's collision, progress, stuck and comfort contracts, including progress as an anti-trivial multiplier;
- InterFuser's use of an interpretable intermediate world state to constrain an action safe set;
- PlanT only as object/action-level planning context, not as a CARLA dependency.

## Frozen experiment

Training uses only the already-consumed P6R confirmation cohort. The downstream P10R4 action-collision labels remain unread at
freeze time. Each of 96 evaluation cases has exactly 13 actions: four progress prefixes crossed with lateral offsets
`-1.5/0/+1.5 m`, plus stop. The horizon is 2 seconds and the collision proxy is overlap between a 1.5 m ego corridor and the
target actor swept envelope.

The three arms use the same ten target-free scalar features and the same linear logistic critic:

1. `real_only`: four zero-lateral logged-route prefixes per training case;
2. `real_plus_naive_generated`: real actions plus every nonzero-lateral generated action;
3. `real_plus_unc_verified`: real actions plus the lowest-risk half of generated actions per case, ranked only by frozen M1
   selected-route mean risk.

The model is fitted before any evaluation action label is generated. The decision threshold is 0.5. There is no action-lattice,
architecture, threshold, seed or test sweep.

## Metrics and gates

Primary: selected-policy collision false-safe count. Unsafe-action recall, safe-action precision, Brier score, ECE, progress,
stuck, comfort and reward are reported.

Only three gates are used:

- verified selected-policy false-safe count is no worse than both comparators;
- verified mean progress ratio is at least 0.50;
- verified stuck rate is at most 0.20.

If Real-only is equally good on the primary metric, the result is explicitly `no_increment_over_real_only`; it is not renamed a
positive result. All-brake behavior fails the progress/stuck gates.

## Authority boundary

This is an exact downstream actor-envelope collision-proxy audit, not physical collision, population, planning, closed-loop, or
safety evidence. No large NWM, RL policy, hash, checksum, fingerprint, bootstrap gate, significance gate, smoke suite, or
regression matrix is added. The single formal run is `20260827T033000Z__bounded-collision-critic-s0-r1`.
