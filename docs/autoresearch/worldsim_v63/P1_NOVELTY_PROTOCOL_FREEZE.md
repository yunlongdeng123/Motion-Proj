# WorldSim V6.3 P1 Novelty and Protocol Freeze

- Task: `WS-V63-P1-SCOPE-NOVELTY-01`
- Hypothesis: `WS-V63-H-P1-001`
- Status: `done / novelty gate passed`
- Quality read: none
- Failure-ledger delta: none

## Primary-source audit

The audit used papers or official repositories only.

| Source | Established capability | Boundary retained for SurfNCC |
|---|---|---|
| [RELIOcc, IJCAI 2025](https://www.ijcai.org/proceedings/2025/0220.pdf) | hybrid voxel uncertainty learning and offline uncertainty-aware calibration | no proposal-surface hidden-FREE tail objective or Physical State admission |
| [OCCUQ, official repository](https://github.com/ika-rwth-aachen/OCCUQ) | dense-head uncertainty module plus feature-level GMM for epistemic/OOD uncertainty | requires native head/features; voxel UQ is not case-level false-safe control |
| [alpha-OCC](https://arxiv.org/abs/2406.11021) | uncertainty propagation and hierarchical conformal prediction sets for occupancy | class-set coverage does not certify proposal-surface collision safety |
| [QueryOcc](https://arxiv.org/abs/2511.17221) | continuous 4D queries supervised from adjacent-frame rays | justifies native spatiotemporal query supervision, not surface admission or tail control |
| [EvOcc, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Kalble_EvOcc_Accurate_Semantic_Occupancy_for_Automated_Driving_Using_Evidence_Theory_CVPR_2025_paper.html) | evidential mapping explicitly models unobserved and conflicting measurements | uncertainty/contradiction modeling alone does not establish positive OCC authority |
| [Conformal Risk Control, ICLR 2024](https://research.google/pubs/conformal-risk-control/) | finite-sample control of monotone expected losses | the loss, exchangeability unit and anti-trivial coverage still have to be task-native |
| [Non-Exchangeable CRC](https://arxiv.org/abs/2310.01262) | relevance-weighted control under drift/non-exchangeability | retained as sensitivity analysis only; primary remains scene-disjoint exchangeable case-level calibration |
| [Conformal Semantic Segmentation, CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/SAIAD/html/Mossina_Conformal_Semantic_Image_Segmentation_Post-hoc_Quantification_of_Predictive_Uncertainty_CVPRW_2024_paper.html) | structured prediction sets calibrated on independent labeled images | confirms that millions of correlated voxels cannot be treated as independent cases |
| [Point Transformer, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Point_Transformer_ICCV_2021_paper.html) | local point-cloud self-attention | architecture component only; no novelty claim |
| [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html) | permutation-invariant set interaction and attention pooling | motivates patch/proposal tokens; architecture component only |
| [Rockafellar–Uryasev CVaR](https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf) | direct optimization of tail loss | risk primitive only; no novelty claim |
| [Visibility-aware surface reconstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC4897344/) | models free space from visibility to constrain surface reconstruction | supports explicit ray/FREE features; does not solve learned world admission |

No audited source combines native occupancy features, proposal-native surface topology, exact sensor constraints, patch/proposal
hidden-FREE tail optimization, positive OCC authority and independent case-level admission calibration for a driving-world compiler.
The novelty gate therefore passes only for that combination. Native logits, Point/Set Transformer, CVaR, hard projection,
UNKNOWN and conformal calibration are explicitly not standalone contributions.

## Frozen data cohorts

All scenes are nuScenes trainval assets already processed at 10 Hz. Selection used only V4 metadata: official split, location,
description-derived time/weather, actor count, sensor completeness and frame count. No occupancy/proposal/model quality was read.
Each scene uses targets `[17,32,47,62,77,92,107,122,137,152,167,182]`.

- Tier D: unchanged V6.2 six scenes; train=`0071,0317,0862,1012`, selection=`0450,1089`.
- Tier C (72 cases): `0919,0634,1062,0626,0015,0552`.
- Tier H (36 cases, sealed): `0924,0906,1072`.
- Tier T (48 cases, sealed exact-once): `0519,0554,0911,0966`.
- Tier L: legacy `0048,0242`, retrospective only.

The cohorts are pairwise scene-disjoint. Tier C covers Boston/Singapore, dry/rain, day/dusk/night, general urban/intersection/
roundabout/road segment and low/high actor counts. H/T quality stays sealed until the respective gate is unlocked.

## Frozen native and surface representation

IR-WM remains frozen. Each target stores full native `200x200x16x17` FP16 logits and `200x200x256` FP16 current BEV latent,
plus source-valid, native coordinate, argmax, entropy, top1-top2 margin, temporal frame identity and source-grid identity. The official
capability audit found no additional flow/3D-query tensor with an equally direct stable extraction contract, so V6.3 v1 freezes only
logits and current BEV latent. No class prototype or fabricated confidence is legal.

The target grid remains `origin=(-20,-30,-3)m`, extent `(40,30,5)m`, voxel `0.2m`. Candidate volumes are method-visible and frozen:
native occupied components intersected with route/static regions plus deterministic current/swept actor OBB support. A surface is the
6-connected boundary of one candidate component; topology cannot create occupied nodes or change the volume. Patch construction is
deterministic geodesic breadth-first partitioning seeded by lexicographically smallest unassigned boundary voxels, with minimum 64,
target 512 and maximum 2048 points. Sub-64 remnants merge with the adjacent patch sharing the most 6-neighbor edges; components below
64 points are retained as one small patch and reported, not deleted.

Per point the frozen feature schema is: 17 native logits; 256 BEV latent; entropy/margin/source-valid; hard evidence five-state;
signed distance to observed FREE/OCC; local xyz; normal; ray direction and normalized hit order; four temporal support counts; actor
current/swept/canonical/lifecycle features. Coordinates and geometry derive from declared proposals and evidence only.

## Frozen model and objectives

SurfNCC v1 uses a two-layer point MLP, two deterministic 6-neighbor residual aggregation blocks, a 256-D patch token, two
Transformer encoder layers over patch tokens, and one proposal token. Heads predict three-state point probabilities, hidden-FREE risk,
positive OCC authority, patch risk and proposal risk. Mixed precision is FP16; only SurfNCC trains.

Point risk is frozen as
`clip(P(O)*q_HF + 0.25*(1-q_AUTH), 0, 1)`. Patch risk is worst-10% CVaR (`alpha=0.90`); proposal risk is maximum patch CVaR.
Safe-OCC miss risk is maximum patch CVaR of `1-P(O)`. The case score is maximum proposal risk. Exact projection priority remains
`contradiction -> UNKNOWN; observed FREE -> FREE; observed OCC -> OCC; outside lifecycle -> UNKNOWN; otherwise learned`.
Low authority (`q_AUTH < 0.5`) becomes UNKNOWN before case admission.

Loss weights are frozen: state `1.0`, hidden-FREE tail `1.0`, safe-OCC retention `0.75`, matched proposal rank `0.25`, local surface
consistency `0.10`, authority `0.50`. Ranking margin is `0.10`. Train-only structural evidence dropout selects exactly one applicable
mask family uniformly per proposal (ray bundle, spatial block, temporal window, surface patch, plus actor support for actor proposals),
removing 25% of its visible support with seed `20260824`; target labels remain supervision only.

Training uses AdamW `2e-4`, weight decay `1e-2`, FP16, point microbatch `8192`, gradient accumulation `4`, maximum `12` epochs,
minimum `4`, patience `3`, primary seed `0`. Selection is lexicographic: zero hard violations, minimum proposal/surface tail surrogate,
safe-OCC retention, accepted surface yield, then target accuracy. One bounded capacity recovery is legal only under the plan's joint
underfit conditions.

## Frozen calibration and gates

Calibration unit is one target case, never a voxel or patch. For threshold `lambda`, a case is accepted only if it has no hard reject,
its maximum proposal risk is at most `lambda`, and its emitted occupied surfaces pass authority/state requirements. Threshold grid is
`0.000..1.000` in steps of `0.025`, ordered from most conservative to permissive. Tier C uses fixed-sequence one-sided exact-binomial
testing, stopping at the first threshold whose 95% upper bound for `ACCEPT and FREEConflict>0.05` exceeds target risk `epsilon=0.05`;
the previous passing threshold with maximum coverage is frozen. This preserves the familywise confidence under the preregistered
ordered family. Group thresholds require at least 24 calibration cases in that stratum; otherwise the global threshold applies.

Anti-trivial gates for every promotable arm: safe-OCC retention `>=0.60`, source-valid UNKNOWN `<=0.60`, accepted surface area not
below matched Native B2, accepted case coverage `>=0.10`, and nonzero actor plus static coverage. P6 requires two selection scenes,
zero hard violations, B3 surface-tail improvement over B2 `>=2%`, B5 improvement over the better of B3/B4 `>=2%`, and M0 proposal
false-safe surrogate improvement over B5 `>=2%`. Failure of B3 closes surface architecture; failure of B5 closes tail risk.

P7 additionally requires 72 calibration cases, the exact-binomial upper bound `<=0.05`, nonzero policy and all anti-trivial gates.
P8 uses the plan's unchanged legacy gate (`>=5/28`, zero false-safe, R10 `3/3`, actor/static gains, area `>=12%`, worst conflict
`<=0.05`, retention `>=0.60`, UNKNOWN `<=0.60`). Any legacy false-safe keeps P9 locked. P9/P10 require zero empirical false-safe,
valid 95% risk report, accepted coverage `>=0.10`, retention `>=0.60`, at least two supporting scenes and no catastrophic actor/static
regression. Exact-once test remains one consumed attempt.

## Resource freeze

Full native logits plus BEV latent require about 42.2 MB per unit in FP16; all 232 planned D/L/C/H/T units require about 9.8 GB before
registries and surface arrays, within the observed 65 GiB free disk. The frozen IR-WM forward previously peaked near 4.13 GiB; SurfNCC
uses memory-mapped sidecars, CPU preprocessing, patch chunking, microbatching and gradient accumulation. Resolution, ROI, scenes,
surface points, latent width, temporal window, CVaR alpha and gates cannot be reduced to fit hardware.

Next task: `WS-V63-P2-NATIVE-SIDECAR-01`.
