# P5-B Frozen Physical--Reliability Interface Result

Date: 2026-09-02

Canonical run:
`run://worldsim_v7/WS-V7-P5B-FROZEN-PHYSICAL-RELIABILITY-INTERFACE-01/20260902T183000Z__frozen-physical-reliability-interface-s0-r1`

## Verdict

`descriptive_interface_completed`

On the 88 exact-match P4 test Actors (14 scenes, 39,285 retained V6.7 rows), frozen P4 selected only 5 Actors. All 23 Actors
whose compiled Chamfer was worse than query-only Chamfer were rejected; consequently `selected_and_harmful=0`. The selected
Actors contributed 1,781 V6.7 rows with zero occupancy false-safe and zero decision flips. This supports a narrow empirical
guardrail: within this matched source subset, the P4 decision did not authorize a known geometrically harmful repair or coincide
with a retained discrete occupancy failure.

## The non-obvious boundary

P4 selection concentrated on motion-difficult Actors rather than motion-easy Actors:

| Frozen P4 group | Actors | mean cost | q90 cost | mean state error | q90 state error | false-safe | decision flip |
|---|---:|---:|---:|---:|---:|---:|---:|
| selected | 5 | 0.1225 | 0.2833 | 0.6960 m | 1.5772 m | 0 / 1,781 | 0 / 1,781 |
| abstained | 83 | 0.0464 | 0.1016 | 0.2704 m | 0.6932 m | 7 / 37,504 | 47 / 37,504 |

Selected-vs-abstained ratios are `2.64x` mean cost, `2.79x` q90 cost, `2.57x` mean state error, and `2.28x` q90 state error.
For the selected Actors, q90 state error rises from `0.617 m` at `0.8 s` to `4.246 m` at `3.0 s`. Thus the actors for which
physical repair is most confidently allowed can still have the widest long-horizon motion uncertainty.

## Interpretation

Physical validity and trajectory reliability are complementary authorities. P4 can decide whether a geometric repair is
supported; it cannot grant downstream trajectory/control authority. The paper-facing inference graph must keep the P4 physical
repair gate upstream of, but distinct from, C3's multi-horizon reliability authority.

This is a descriptive co-occurrence result on retained source rows. P346 was not executed, no model or threshold changed, and the
zero discrete failures in five selected Actors is not a statistical or formal safety guarantee.
