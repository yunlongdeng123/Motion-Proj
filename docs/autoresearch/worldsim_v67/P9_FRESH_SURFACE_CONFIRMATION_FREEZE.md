# V6.7 P9 fresh surface confirmation freeze

- Date: 2026-08-28
- Hypothesis: `WS-V67-H-P9-001`

Before any P9 processing or quality read, repository-mentioned scenes and existing processed directories were excluded from the
nuScenes trainval scene list. Archive band 4 had 79 eligible scenes. Deterministic positions `round(k*(79-1)/7)`, `k=1..6`,
select:

| position | processed index | scene |
| ---: | ---: | --- |
| 11 | 265 | scene-0348 |
| 22 | 277 | scene-0360 |
| 33 | 290 | scene-0373 |
| 45 | 304 | scene-0388 |
| 56 | 315 | scene-0399 |
| 67 | 328 | scene-0414 |

Only archive shard 4 is scanned first. A missing-member fallback may scan the remaining shards for these exact scenes; it may
not replace a scene. Six scenes × 12 frozen targets produce 72 units. Preparation, evidence and native sidecars read no model
score or surface quality. Native GPU jobs are launched as individual processed scenes become ready so archive/preprocess I/O and
GPU inference overlap on the single RTX 3090.

After inputs are complete, the exact frozen P5-P8 chain is repeated: V6.6 head without refit, Actor package, 50% L0 actions and
P4R inward-ray repair with the unchanged nine gates. No threshold, radius, budget, model or rule sweep is allowed. A pass can
support one fresh empirical confirmation, not a population guarantee, RL readiness, planning, policy or safety.
