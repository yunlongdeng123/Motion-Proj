# P4C direct conditional compiler freeze

Date: 2026-08-26

P6R established a frozen full-native risk ranking at 40% global coverage. The already-consumed independent calibration result also exposed a simple, prespecified next version: all three 50% failures were rain, while night, construction, and vulnerable-transit had zero failures at 50%. P4C therefore freezes one conditional coverage map, not a sweep: rain remains at 0.40 and the other three strata use 0.50. C0 is the frozen global 0.40 comparator. The MLP, risk order, hidden-FREE conflict threshold 0.05, and case unit do not change.

The formal calibration replay must show at least 0.05 mean realized-coverage uplift, zero M0 failures overall and per stratum, and no more failures than C0. This replay uses the already-consumed eight calibration scenes and cannot create a fresh claim. It only freezes the candidate for a later new confirmation cohort.

## New confirmation cohort frozen before quality read

The new cohort was selected with seed 2 from nuScenes train metadata only, after conservatively excluding every `scene-NNNN` referenced by project configs, autoresearch documents, and the three research ledgers. Each scene has at least 40 samples. The same ordered keyword strata and two scenes per stratum are retained:

| stratum | scenes | processed indices |
| --- | --- | --- |
| night | scene-0992, scene-1101 | 751, 841 |
| rain | scene-0454, scene-1102 | 368, 842 |
| construction | scene-0876, scene-0895 | 664, 681 |
| vulnerable_transit | scene-0321, scene-0276 | 253, 222 |

Only name, description, sample count, and exclusion membership were read. Occupancy, hidden-FREE, model scores, and target quality remain unread. Confirmation freezes C0 and M0 together; M0 must add at least 0.05 realized coverage while staying within at most 4/96 failures overall and 1/24 per stratum. No model refit, alternative mapping, second confirmation, hash, checksum, fingerprint, coverage sweep, smoke matrix, or regression matrix is allowed.
