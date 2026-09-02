# P23 Fresh AV2 Literal First-Return Confirmation Freeze

## Motivation

P20 and P22 show that target-nearest attribution understates minimum-positive-depth first-return exposure on source and
already consumed AV2 surfaces. P22 cannot serve as an independent external confirmation. CVPR 2024/2025 evidential occupancy
work motivates raw-ray evaluation and an explicit UNKNOWN state, so P23 freezes one disjoint exact-once measurement read.

## Frozen evidence contract

- Task: `WS-V7-P23-FRESH-AV2-FIRST-RETURN-CONFIRMATION-01`.
- Run: `20260903T183000Z__fresh-av2-literal-first-return-s0-r1`.
- Cohort: the ten metadata-only logs in `av2_evicomp_fresh_cohort_v1.json`, frozen before P22 and still quality-unread.
- Surfaces: unchanged P2 four-action compiler with observed-LiDAR-hit PROJECT.
- Proxy: Euclidean-nearest output to each target return, then frozen beam/depth test.
- Literal operator: minimum positive projected depth over every surface point within the same `0.20 m` beam tube.
- Depth tolerance: `0.20 m` for both operators; query and compiled surfaces use identical operators.
- Runtime: one RTX 3090, logs processed sequentially; no training or second run.

## Core confirmation

The result confirms H-V7-P23 only if literal new-early rate exceeds the proxy rate for both all Actors and the frozen hazardous
stratum. KEEP/PROJECT/COMPLETE attribution and hit/early ratios are descriptive and do not add gates. A failed gate records
`V7-F30`; a pass leaves the next failure ID unchanged.

## Prohibited adaptation

No model, checkpoint, threshold, tolerance, surface action, deletion policy, cohort member, or failed-log decision may change
after this freeze. P23 cannot promote P16, P17, P17R, P19, or any selector. It is fresh evidence only for metric undercounting and
compiler-action provenance, not for fitted transfer, collision freedom, planning, closed-loop behavior, or road safety.
