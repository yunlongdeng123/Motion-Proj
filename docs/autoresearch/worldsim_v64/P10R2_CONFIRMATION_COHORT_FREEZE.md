# P10R2 Fresh Confirmation Cohort Freeze

Date: 2026-08-27  
Task: `WS-V64-P10R2-CONFIRMATION-SIDECAR-01`  
Hypothesis: `WS-V64-H-P10R2-002`  
Selection seed: `3`  
Confirmation quality read at freeze: `false`

## Metadata-only selection

The pool is the frozen 700-scene IR-WM train temporal metadata. Selection requires at least 40 nuScenes samples and excludes
all 124 distinct `scene-NNNN` identifiers present in current project configs, autoresearch documents, and the three research
ledgers. A shared `random.Random(3)` shuffles sorted candidates in the fixed order night, rain, construction, and
vulnerable/transit, taking two unused scenes per stratum. Matching uses token-level description terms; no substring `ped`
matching is permitted.

| stratum | scenes | processed indices | metadata descriptions |
| --- | --- | --- | --- |
| night | scene-1020, scene-1016 | 778, 774 | Night, peds, bus, bus stop / Night, peds, bus stop, bump, ped crossing crosswalk, bus |
| rain | scene-0596, scene-0590 | 476, 470 | Rain, wait/start at intersection, peds, truck, dog / Rain, traffic cones, intersection, bendy bus |
| construction | scene-0006, scene-0472 | 5, 383 | construction worker/vehicle / Rain, construction, wheel barrows |
| vulnerable/transit | scene-0070, scene-0371 | 67, 288 | peds, bus, cyclist / peds, bus, scooter, bicycles |

All eight names are direct keys in `nuscenes_temporal_infos_train.pkl`; each has 40 or 41 samples. The candidate pool sizes
at sequential selection were 60 night, 114 rain, 82 construction, and 406 vulnerable/transit. Only scene name, description,
sample count, temporal membership, processed index, and exclusion membership were read. Occupancy, hidden-FREE, evidence,
UQ, risk scores, target quality, and route conflict were not read.

Each scene uses the unchanged 12 targets `17,32,47,62,77,92,107,122,137,152,167,182`, giving 96 cases. The cohort cannot
be replaced after target evidence is generated.
