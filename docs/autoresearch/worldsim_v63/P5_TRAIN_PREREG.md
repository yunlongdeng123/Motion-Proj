# WorldSim V6.3 P5 SurfNCC Training Preregistration

- Task: `WS-V63-P5-SURFNCC-TRAIN-01`
- Hypothesis: `WS-V63-H-P5-001`
- Status: `implementation staged / execution locked behind P4 capacity`
- Seed: `0` (the only primary seed)

## Pre-execution structural audit and migration

A method-only read of the first 40 completed P3 units found one surface above 8192 points in every unit (`40/40`, maximum observed
`173488` points). Chunk-local proposal tokens were therefore rejected as a systematic loss of proposal interaction, not accepted as a
rare approximation. The migration follows the hierarchical set principle in [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html)
and the latent bottleneck separation in [Perceiver, ICML 2021](https://proceedings.mlr.press/v139/jaegle21a.html): bounded point chunks
produce deterministic patch tokens, and the much smaller complete patch set is the unit consumed by proposal attention. A focused
modular-forward equivalence audit covered all 12 model outputs and returned maximum absolute difference `0.0` on the unsplit path.
A separate packing-semantic audit returned identical full/split patch-local directed-edge counts (`4/4`) and retained the frozen
safe/unsafe unit ranking pair `[(0,1)]` even though the proposals occurred in different chunks.
Selection uses the same unit boundary: pairs are formed separately inside each complete scene/frame unit and the nonempty unit losses
are averaged. A cross-unit-only safe/unsafe synthetic returned zero pairs, so checkpoint ordering cannot mine across cases.
Failure=`V63-F14 resolved_preexecution`.

Masking follows [Masked Autoencoders, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html):
masked evidence is removed from every encoder-visible channel while the original semantic reconstruction target remains supervision.
Authority is not a semantic target; it is derived from visible support and is therefore recomputed after masking. Finally, selection joins
all hidden-FREE values before complete-proposal CVaR because [mini-batch risk functionals can be biased](https://arxiv.org/abs/2301.11724);
the stochastic training surrogate
remains explicitly bounded and is not presented as an exact full-batch CVaR optimizer.
[Attention-based Deep MIL, ICML 2018](https://proceedings.mlr.press/v80/ilse18a.html) further motivates assigning an auxiliary
bag/proposal head one permutation-invariant complete-bag aggregation target rather than incompatible per-chunk targets; V6.3 retains its
already frozen maximum-patch-risk aggregation instead of importing a new learned pooling rule.

A final loss-by-loss audit found that complete proposal tokens alone were insufficient if ranking pairs were still selected only among
proposals co-occurring in a point chunk. The frozen actor/static and nearest-full-point-count one-to-one matching is now constructed once
from the complete unit metadata. Current detached patch tokens for the whole unit feed differentiable proposal attention and the proposal
risk head once per unit, so every authorized pair is represented independently of packing. [Cross-Batch Memory, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)
supports the diagnosis that minibatches can omit informative pairs, but its stale queue is not migrated because the complete unit patch
set already fits memory. No queue, momentum, margin, weight or new hyperparameter is introduced. Failure=`V63-F12
resolved_preexecution`.

Point chunks must also be graph-invariant. The earlier surface-wide edge builder silently lost cross-chunk edges while retaining the same
edge when adjacent patches happened to co-occur. The two local 6-neighbor blocks are therefore bound to each complete deterministic
patch (maximum 2048 points and never split), and complete-proposal patch attention owns cross-patch interaction. This follows the
local-patch-token hierarchy of Point-BERT and avoids the induced-subgraph edge-loss bias identified by GraphSAINT without adding random
sampling, halo width or another tunable parameter. Failure=`V63-F13 resolved_preexecution`.

## Denominator and batching

Training uses all 48 targets from `scene-0071/0317/0862/1012`; selection uses all 24 targets from scene-disjoint
`scene-0450/1089`. Complete deterministic patches are packed up to 8192 points without deleting tiny or large surfaces. Every packed
point chunk is first pooled into complete patch tokens; all patch tokens are then reassembled by semantic global patch/proposal identity
before the two frozen patch-attention layers and exactly one shared learned proposal token run over each complete proposal. A proposal
larger than one microbatch is split only at patch boundaries; point decoding remains bounded, while primary proposal risk is reconstructed
as the maximum patch CVaR across all chunks. During training a no-graph patch-token prepass is refreshed after every optimizer update and
the current differentiable chunk replaces its cached tokens before global proposal attention. This preserves full proposal context without
retaining every point graph. The proposal head remains auxiliary and never decides admission. An unsafe proposal contains at least one
method-hidden target-FREE point; a safe proposal has target-OCC
support and no such hidden-FREE point. Every chunk inherits its complete proposal's actor/static stratum, safe/unsafe label and full point count;
chunk-local counts are used only when reassembling point metrics. This is the plan-authorized patch chunking recovery, not a denominator
or geometry change.

## Train-only structural dropout

One applicable family is sampled uniformly once per complete proposal and epoch, before any patch chunking, with RNG seed 20260824 and
support fraction 0.25. Every chunk then consumes the same complete-proposal selector. Saved semantic
selectors are bundle IDs, patch IDs, dropped sweep columns, spatial seed plus L-infinity radius/count, or actor unit-point indices. Ray,
spatial, patch and observed-actor masks remove hard evidence, signed-distance, temporal-count and observed-actor channels at affected
points while retaining native logits/BEV. Temporal-window masks recompute hard state and four support counts from remaining sweep
columns. Evidence-derived authority input bits and supervision are recomputed from the support that remains visible: masked observed-OCC
and temporal-OCC bits cannot leak through the auxiliary channel, while independent Actor current/swept and closure support remain legal.
The semantic target state remains the original reconstruction supervision. Target state/contradiction/behind-hit never enters forward
features or masks.

## Loss and matching

State CE excludes contradiction or hard-evidence/target mismatch points. Hidden-FREE and safe-OCC losses average alpha-0.90 CVaR per
proposal in each packed batch; this is the frozen memory-bounded stochastic training surrogate, while checkpoint selection joins every
hidden-FREE point from all chunks before computing exact complete-proposal CVaR. Hidden-FREE and authority binary heads use AMP-safe
BCE-with-logits. Local probability TV plus patch/
proposal-head consistency forms the surface term. The proposal head is matched to the permutation-invariant maximum of patch-risk-head
scores over the complete proposal, never to a chunk-local maximum; patch heads remain anchored to their point-risk CVaR. Matched ranking
pairs unsafe and safe proposals within actor/static stratum by nearest full-proposal point count, one-to-one, with frozen margin 0.10.
Matching is computed over the complete unit, not proposals that happen to share a point chunk. Its once-per-unit train-time differentiable
surrogate runs the proposal attention/risk head over the complete detached patch-token cache; selection and reporting use reconstructed
maximum patch CVaR rather than the auxiliary head. Selection also matches within each scene/frame unit and averages nonempty unit
surrogates; it never pairs proposals from different cases. Loss weights remain P1:
`1.0/1.0/0.75/0.25/0.10/0.50`.

## Optimizer and selection

AdamW uses learning rate `2e-4`, weight decay `1e-2`, FP16, accumulation 4, maximum 12 epochs, minimum 4 and patience 3. Selection
first preserves hard observed FREE/OCC and contradiction projection, then converts only unconstrained learned OCC with authority below
0.50 to UNKNOWN. Coverage, UNKNOWN and secondary accuracy all use this same final decision. Selection reconstructs chunked proposal risk
by maximum and uses this lexicographic tuple:

1. hard projection violations;
2. mean unsafe-proposal hidden-FREE tail plus matched rank surrogate;
3. safe-OCC retention deficit below 0.60;
4. emitted-OCC coverage deficit below 0.10;
5. negative retention, negative coverage, then negative target accuracy.

Thus ordinary weighted loss or target accuracy cannot overrule the frozen surface-tail objective; retention and coverage remain mandatory
promotion gates even though they follow tail risk in the P1-frozen checkpoint ordering. Selection labels come only from the two
authorized Tier-D selection scenes. The best checkpoint and all semantic dropout selectors are saved without hashes, checksums or
fingerprints. P5 produces a trained candidate but no promotion conclusion; only P6 matched AB can advance it. Tier C/H/T remain sealed.

P5 inherits the P4 numerical recovery before any training: FP16 GradScaler initial scale `1024`, deterministic algorithms, math SDPA
only, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` bound before torch import. This fixes execution numerics without changing the frozen
optimizer, model, data, loss or selection protocol. P5 remains locked until P4 r3 passes.
