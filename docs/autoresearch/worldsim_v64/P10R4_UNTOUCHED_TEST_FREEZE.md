# P10R4 Untouched-Test Fixed-Denominator Freeze

Date: 2026-08-27  
Hypothesis: `WS-V64-H-P10R4-001`  
Selection seed: `4`  
Test quality read at freeze: `false`

## Rationale and migrated protocol

P10R3 diagnosed a variable-denominator reversal after P10R2 confirmation was already read, so it cannot independently confirm
relative improvement. Waymo Occupancy Flow's fixed ego-centric cell evaluation supports keeping a common opportunity
denominator. The NeurIPS 2021 Outstanding Paper and `google-research/rliable` support publishing individual results and
probability of improvement for paired comparisons. ICLR 2024 Conformal Risk Control is not migrated as a gate because its
core contract is monotone expected loss rather than this frozen empirical tail contrast.

References:

- `https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/occupancy_flow_metrics.proto`
- `https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf`
- `https://github.com/google-research/rliable`
- `https://openreview.net/pdf?id=33XGfHLtZg`

## Metadata-only cohort

The same frozen 700-scene IR-WM train temporal metadata and minimum 40-sample rule are used. Every `scene-NNNN` already
referenced in project configs, autoresearch docs, and the three ledgers is excluded. A shared `random.Random(4)` shuffles
sorted candidates in the fixed order night, rain, construction, vulnerable/transit and takes two per stratum. Selection reads
only scene name, description, sample count, temporal membership, processed index, and exclusion membership.

| stratum | scenes | processed indices | descriptions |
| --- | --- | --- | --- |
| night | scene-1084, scene-1081 | 824, 821 | Night, scooter, peds, bus stop... / Night, rain, big street... |
| rain | scene-0462, scene-0820 | 375, 636 | Rain, cross intersection... / Rain, parking lots, peds... |
| construction | scene-0534, scene-0598 | 424, 478 | forklift, construction zone... / Rain, construction, roundabout |
| vulnerable/transit | scene-0527, scene-0668 | 417, 522 | pedestrian and dog... / Parked bus, crane... |

Sequential candidate pool sizes are `55/110/79/377`; all eight scenes have `40` or `41` temporal samples. The existing
storage partition key remains `fresh_confirmation` only for runner compatibility; its semantic role is `untouched_test`.

## Exact-once contract

Each scene uses the unchanged 12 targets `17..182` at step 15, for 96 cases. M0 coverages, M1 route cap `0.40`, frozen MLP,
2-second/1.5-metre corridor, route-eligible denominator, and worst10/96 are frozen. The only gates are:

1. absolute mean total-coverage delta `<=1e-6`;
2. `M1-M0` fixed-denominator worst10 CVaR `<=0`;
3. `M1-M0` pooled fixed-denominator density `<=0`.

Per-case M1-lower/equal/higher counts and half-tie paired improvement probability are descriptive only. There is no bootstrap,
significance gate, parameter sweep, refit, second test, hash, checksum, fingerprint, smoke suite, or regression matrix. A pass
can resolve V64-F25 only for this exact untouched empirical cohort and unlock bounded P11 design work; it cannot rewrite P10R2,
create a population guarantee, or support collision/planning/safety claims.

## I/O and resource execution

Required metadata members are collected in one `sample_data.json` traversal. A raw-only archive producer fills the recoverable
temporary batch while one scene-ready feeder exclusively owns preprocessing and immediately dispatches at most two single-scene
native workers to RTX 3090. A later reuse finalizer records completed canonical scenes and removes temporary raw. Projected
peak additional disk is about 14 GiB from 27 GiB free; single-GPU native peak remains about 4.2 GiB, so multi-GPU is not needed.

Canonical IDs:

- raw-only prep: `20260827T021000Z__test-raw-only-s4-r1`;
- native prefix: `20260827T021500Z__test-native`;
- native aggregate: `20260827T023000Z__native-aggregate-s4-r1`;
- evidence: `20260827T023500Z__test-evidence-s4-r1`;
- final exact-once: `20260827T025000Z__exact-once-fixed-denominator-s4-r1`.
