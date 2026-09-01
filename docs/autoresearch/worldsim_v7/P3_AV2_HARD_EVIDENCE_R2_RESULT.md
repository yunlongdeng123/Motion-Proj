# WorldSim V7 P3 — Ray-Correct Hard Evidence r2

## Verdict

- run: `run://worldsim_v7/WS-V7-P3-AV2-HARD-EVIDENCE-01/20260902T140000Z__hard-evidence-s0-r2`
- verdict: `rejected_zero_shot_av2_hard_physical_evidence`
- gates: quantitative `6/8`, qualitative `6/8`, visuals `3/3`

r2 repairs the r1 metric-provenance defect and therefore provides the first valid per-primitive ray read. It confirms that the
nearest-canonical PROJECT operator is not line-of-sight safe.

## Scientific failure

| Role | Free-space violation after | Ghost component ratio | Mean early depth (m) |
|---|---:|---:|---:|
| Quantitative | .849111 | .916598 | .091783 |
| Qualitative | .828770 | .900591 | .088809 |

The output can be close to a canonical surfel in Euclidean 3D while lying before the observed termination on the matched beam.
The P2 surface/Chamfer improvement is real but insufficient for a full physical-consistency claim.

## Frozen repair candidate

`WS-V7-H-P3-002` replaces only the PROJECT output location. A primitive with direct LiDAR hit provenance projects to that same
observed termination; a primitive without a matched hit becomes `UNKNOWN` and emits no collision surface. The cohort, actions,
target isolation, gates, thresholds, cases, and all non-PROJECT paths stay fixed. r2 remains rejected regardless of r3 outcome.
