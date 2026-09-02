# P15 Fresh Hazard-by-Action Attribution Result

## Canonicals

- raw CUDA attribution: `run://worldsim_v7/WS-V7-P15-FRESH-HAZARD-ACTION-ATTRIBUTION-01/20260903T043000Z__fresh-hazard-action-raw-s0-r1`
- frozen selector join: `run://worldsim_v7/WS-V7-P15-FRESH-HAZARD-ACTION-AUDIT-01/20260903T044500Z__fresh-hazard-action-audit-s0-r1`

Both runs are `done`. Raw attribution covers 20 logs, 523 Actors, 1,435,391 target rays, and uses `46.70 s`, `0.0996 GiB` peak GPU memory, and `1.186 GiB` peak RSS on one RTX 3090. The join takes `0.021 s` CPU time. No training, fitting, calibration, threshold change, policy search, action change, or cohort deletion occurs.

## Fresh aggregate mechanism

The compiled surface introduces 19,354 early target-ray returns (`1.348%`) and 280,889 new target hits, or `14.51` new hits per new early return. COMPLETE accounts for 18,328/19,354 (`94.70%`) new early returns and 280,717/280,889 (`99.94%`) new hits. In contrast, KEEP accounts for 487/509 (`95.68%`) emitted-surface contradictions; COMPLETE accounts for only 22/509 (`4.32%`).

Thus the two failure metrics expose distinct mechanisms. New target-ray early returns mostly trace to inserted COMPLETE surface, while output-to-target surface contradictions mostly trace to retained KEEP geometry. Neither metric can replace the other.

## Hazard versus clear

| Scope | Hazard / clear new-early rate | H/C ratio | Hazard / clear COMPLETE share of new early | Hazard / clear hit-to-early ratio | Hazard / clear KEEP share of contradictions |
|---|---:|---:|---:|---:|---:|
| always repair | 1.912 / .980% | 1.951x | 98.52 / 89.82% | 13.47 / 15.84 | 95.45 / 96.18% |
| P4 selected | 1.922 / .968% | 1.985x | 98.67 / 92.98% | 13.39 / 16.71 | 95.63 / 96.19% |
| P6-C selected | 1.913 / .957% | 1.999x | 98.53 / 92.61% | 13.48 / 16.73 | 95.43 / 95.16% |

Hazardous Actors have about twice the new-early target-ray rate of clear Actors. Their COMPLETE yield is also worse: always-repair obtains `13.67` COMPLETE new hits per COMPLETE new early versus `17.62` on clear Actors. This explains why the hazard stratum can provide high aggregate geometry gain while retaining a larger visible-failure burden.

P4 selects 133/142 hazardous Actors and its hazardous new-early rate is `1.0055x` the always-repair hazardous rate. P6-C selects 141/142 and is `1.0010x` always. The selectors therefore do not suppress the completion-driven hazardous-ray mechanism; this is consistent with P14's finding that their hazard-stratum selected visible risk does not improve. P4-abstained has only nine hazardous Actors and is retained descriptively, not used for a confident comparison.

## PROJECT caveat

PROJECT has zero emitted provenance points because the frozen compiler writes PROJECT to an observed LiDAR hit, concatenates KEEP first, and voxel-deduplicates. A shared-voxel projected hit is therefore labeled KEEP. This is deterministic provenance collapse, not evidence that PROJECT causes zero harm or contributes nothing.

## Verdict

`descriptive_fresh_hazard_action_mechanism_boundary`. P15 strengthens hard 3D evidence by localizing two ray/surface failure channels and connects them to the P14 hazard burden. It does not contradict the existing paper: P3-D already scoped attribution as descriptive, and the V7 claim never equates immutable hazard state with hazard visibility safety. No `V7-F24` is registered.

The 20-log cohort was fresh for P3-C/P6-C but already consumed before P15, so this is a mechanism follow-up rather than another blind confirmation. Nearest-output provenance is not a counterfactual action ablation, and none of these rates is a collision, planning, closed-loop, or road-safety guarantee.
