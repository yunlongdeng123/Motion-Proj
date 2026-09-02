# P13 Defer-to-Query Composite World Result

Date: 2026-09-02

Canonical run:

```text
run://worldsim_v7/WS-V7-P13-DEFER-TO-QUERY-COMPOSITE-01/20260903T023000Z__defer-to-query-s0-r1
```

## Result

The frozen exact join contains 523 Actors from 20 consumed AV2 logs. Every policy retains 100% of Actor and hazard semantics;
repair coverage describes which Actors receive the compiled surface, while every abstained Actor keeps its original query.

- Always-repair gains 0.08838 m mean Chamfer but introduces visible failure on 37.28% of all Actors.
- P4 repairs 77.25% (93.66% of hazardous Actors), gains 0.08311 m in the composite world, and reduces population introduced
  visible failure to 28.11%.
- P6-C repairs 83.94% (99.30% hazardous), gains 0.08895 m, and has 30.78% introduced visible failure. It is on this AV2
  frontier but remains rejected as the global selector because fresh nuScenes ranking reverses.
- P11 P4-and-provenance repairs 23.71%, has 4.78% introduced visible failure, but composite mean gain is -0.00088 m. It is
  strictly dominated by query-only (zero introduced failure, zero gain).
- P12 visibility-only is itself dominated by P4-and-visibility. The latter repairs 7.46%, has 0.76% introduced visible
  failure, and gains only 0.00122 m; it is a conservative frontier point, not a useful general authority.

The repair-policy two-axis frontier is P4, P6-C, and P4-and-visibility. Including the no-repair fallback also retains
query-only. Always-repair, P11 provenance, and visibility-only are dominated in population visible-risk/composite-gain space.

## Decision

Register `V7-F23`: lower conditional visible risk on selected Actors does not imply that an accept-or-defer system improves
the fallback world. In particular, P11's provenance selector is worse than doing nothing on mean Chamfer while still adding
visible failures. P12's dual selector remains a valid low-risk frontier point but produces negligible geometric gain and fails
its already frozen coverage/hazard authority gates.

No policy is promoted or retuned. This is consumed-development evidence, not a Waymo result, formal Pareto guarantee,
collision/planning evaluation, or deployment-safety certificate.
