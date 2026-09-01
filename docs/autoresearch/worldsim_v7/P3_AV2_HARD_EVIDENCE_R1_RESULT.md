# WorldSim V7 P3 — AV2 Hard Evidence r1

## Status

- run: `run://worldsim_v7/WS-V7-P3-AV2-HARD-EVIDENCE-01/20260902T133000Z__hard-evidence-s0-r1`
- verdict: `rejected_zero_shot_av2_hard_physical_evidence`
- gates: quantitative `6/8`, qualitative `6/8`, visual selection `3/3`
- coverage: 30/30 logs; 8 fixed main cases; 30 fixed supplement cases

The total P3 claim is rejected. Passing axes remain descriptive evidence and cannot be promoted to a supported P3 result.

## Hard evidence

| Role | Depth error (m) | Ray termination | Zero-level error (m) | Mean jitter (m) | Chamfer (m) |
|---|---:|---:|---:|---:|---:|
| Quantitative | .206731 → .145237 | .569444 → .668462 | .140003 → .042477 | .042402 | .254369 → .168063 |
| Qualitative | .216718 → .152650 | .538437 → .656728 | .145743 → .043710 | .040629 | .267530 → .173570 |

Actor/ID/lifecycle, trajectory, speed, acceleration, TTC, and hazard-event state are immutable across compilation; all measured
shifts are zero and retention is one. This is an interface guarantee, not evidence that upstream perception or prediction is correct.

## Rejected axes and diagnosis

Free-space violation after compilation is .677673/.649254 for quantitative/qualitative; ghost connected-component ratio is
.454247/.456496. r1 incorrectly asks whether any compiled surface is spatially close to an injected ghost. This discards the
ghost primitive's ray provenance and therefore counts legitimate neighbouring Actor surface as if it terminated the same LiDAR ray.

The recovery follows the common boundary in NeuRAD, SplatAD, and LiDAR-RT: retain the primitive's ray origin, direction, observed
hit, and aligned compiled output. A residual is an output in the same beam tube that terminates before the observed hit beyond the
frozen tolerance. `UNKNOWN` emits no collision surface and cannot borrow an unrelated neighbour. r2 keeps all cohorts, actions,
thresholds, gates, visual cases, and the six unaffected metrics unchanged.

## Visual and resource boundary

The 30 point/surface/ray panels follow qualitative-log order and lexical eligible-Actor order, not metric or appearance ranking.
They are P3-A evidence; RGB, camera depth, and video remain P3-B work. Runtime is 99.389 seconds, peak GPU memory 0.06924 GiB,
peak RSS 1.2050 GiB, and the run occupies 5.2 MiB.
