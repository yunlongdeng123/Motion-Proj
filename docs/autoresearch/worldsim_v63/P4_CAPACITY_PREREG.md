# WorldSim V6.3 P4 SurfNCC Capacity Preregistration

- Task: `WS-V63-P4-CAPACITY-01`
- Hypothesis: `WS-V63-H-P4-002`
- Status: `r3 capacity passed / terminal`
- Quality conclusion: forbidden

Temporary synthetic interface history: r1 was rejected by PyTorch autocast at probability-form BCE before real data; the official AMP
recovery exposes hidden-FREE/authority logits and uses BCE-with-logits while preserving sigmoid inference. Synthetic r2 completed finite
forward/backward with proposal-token gradient. Failure=`V63-F08 resolved`; that synthetic history contained no real P4 run.

The shared model also accepts packed patch tokens with one learned token per proposal for later full-denominator P5 throughput. A
synthetic packed API typo was recovered via official `torch.nonzero`; failure=`V63-F09 resolved`.

H-P4-001 was withdrawn before real execution after a method-only read of the first 40 P3 units found one surface above the 8192-point
microbatch in every unit (maximum observed `173488` points; large-surface patch set maximum `417`). Its first-chunk-only probe would have
validated point memory but not the frozen complete-proposal token contract. H-P4-002 keeps the same units, model, optimizer, steps,
thresholds and resources, but gathers every complete patch token for each selected proposal before running the two patch-attention layers
and unique proposal token. This pre-execution recovery is `V63-F10`; no target quality, P4 run or GPU result was read.

The packing audit also found that surface-wide edges were retained only when adjacent patches happened to share a point chunk. To make
the local encoder invariant to packing, both deterministic 6-neighbor blocks now operate inside each complete frozen patch; patches are
never split and are bounded by 2048 points, while complete-proposal patch attention carries cross-patch interaction. GraphSAINT's
induced-subgraph analysis motivates rejecting packing-dependent missing edges, and Point-BERT supports the local-patch-token hierarchy.
The focused packing-semantic audit returned identical full/split patch-local directed-edge counts (`4/4`). No random sampler, halo
radius, topology change or new hyperparameter is introduced. Failure=`V63-F13 resolved_preexecution`.

P4 uses exactly one train unit (`scene-0071/f017`) and one scene-disjoint selection unit (`scene-0450/f017`). Within each unit it
chooses the largest static and largest actor proposal by point count only, then takes complete deterministic patches up to the frozen
8192-point microbatch for point encoding. All chunks of the selected proposal contribute patch tokens to its global context; a
differentiable current chunk replaces its cached tokens before point decoding. This is a geometry/resource choice, not quality selection.
Two AdamW optimizer steps with four-way gradient
accumulation are sufficient to exercise forward, backward, checkpoint and reload; they do not select a model for P5.

The input is 311D and retains all 17 native logits and all 256 BEV channels. SurfNCC uses a two-layer point MLP, two exact patch-local
6-neighbor residual aggregation blocks, mean patch tokens, two 8-head patch Transformer layers and one learned proposal token at hidden width 256.
Heads produce projected three-state probabilities, hidden-FREE probability, positive authority, patch risk and proposal risk. Patch risk
is frozen empirical upper-tail CVaR at alpha 0.90; proposal admission risk remains maximum patch CVaR. V6.2 closed-form projection is
applied in every forward and must have zero violations. Auxiliary proposal-head consistency uses the permutation-invariant maximum of
patch-risk-head scores over the complete proposal, rather than assigning one global head conflicting chunk-local maxima.

Train-only structural dropout uses RNG seed 20260824 and selects one applicable family uniformly once per complete proposal. Every point
chunk consumes the same semantic selector. Ray dropout samples 25%
of unique ray bundles; spatial dropout takes the L-infinity-nearest 25% block around one seeded surface point; temporal dropout removes
one of three sweep columns and deterministically recomputes visible FREE/OCC/UNKNOWN/contradiction; patch dropout removes 25% of patch
IDs; actor dropout removes 25% of observed actor-hit support while retaining identity/current/swept lifecycle features. Masked points
lose method hard evidence and distance features, while target fields remain supervision only. No mask is shared with selection or later
calibration. VideoMAE and MaST-Pre motivate preserving structured temporal masks, but their objectives and high mask ratios are not
migrated.

For ray, spatial, patch and actor-observation masks, the affected points also lose temporal-count and observed-actor support features;
otherwise those channels would leak the evidence that the hard-state channel claims was removed. Native logits/BEV remain visible as
the frozen learned prior. Temporal-window masking instead recomputes the remaining-sweep counts and hard state explicitly. Evidence-
derived authority input bits and authority supervision are recomputed from remaining visible support; original semantic target state
remains reconstruction supervision.

The probe passes only if losses/gradients are finite, hidden-FREE/CVaR and proposal-token gradients are nonzero, hard violations are
zero, checkpoint reload and repeated forward are exact, both scenes execute, and peak allocated GPU memory is at most 22 GiB. It reports
only capability, tensor counts, throughput and resources—no accuracy, false-safe or promotion conclusion. Calibration, confirmation and
exact-once test remain sealed; no hashes, checksums or fingerprints are created.

The CVaR-gradient gate directly evaluates the VJP from `proposal_cvar.mean()` to the state, hidden-FREE and authority heads with
`torch.autograd.grad`; it does not infer CVaR connectivity from a total gradient that also contains BCE. A focused synthetic returned
finite nonzero gradients for all three heads. This corrects the measurement of the existing gate without adding a gate or denominator.
Failure=`V63-F15 resolved_preexecution`.

## H-P4-002 execution recovery

R1=`20260825T045854Z__capacity-h002-s0-r1` exercised both complete train proposals and both complete selection proposals in 11.181
seconds with 0.196070 GiB peak allocation. Losses and outputs were finite, direct proposal-CVaR gradients reached all three heads,
proposal-token gradient was nonzero, hard violations were zero and checkpoint reload succeeded. It failed only because total FP16
gradients contained nonfinite values and repeated/reloaded CUDA attention forwards differed by `9.059906e-6` from the exact-zero gate.

The sole bounded recovery follows PyTorch's AMP and reproducibility guidance: retain FP16 but set the GradScaler initial scale to `1024`,
disable flash and memory-efficient SDPA, enable math SDPA and deterministic algorithms. It does not alter model parameters, data,
dropout, losses, optimizer settings, two steps, accumulation, pass gates or resource ceiling. R1 remains immutable and contains no
quality conclusion. Launch-time failure state=`V63-F17 active_recovery_ready`; terminal state=`resolved`.

R2=`20260825T050400Z__capacity-h002-s0-r2` reached the first CUDA math-attention operation, where PyTorch correctly refused a
deterministic cuBLAS matrix multiply because `CUBLAS_WORKSPACE_CONFIG` had not been set before process startup. It ended before any
optimizer step, capacity summary or quality read and therefore did not exercise the bounded AMP/math-SDPA recovery. NVIDIA cuBLAS and
PyTorch document `:4096:8` as a deterministic workspace option; its approximately 24 MiB overhead remains negligible against the frozen
22 GiB ceiling. R3 binds `CUBLAS_WORKSPACE_CONFIG=:4096:8` in the launcher and before torch import in the runner, while retaining every
R1 recovery choice and every scientific gate. This is the same bounded recovery's first executable attempt, not a new recovery arm.
R2 remains immutable and empty. Launch-time failure state=`V63-F18 active_recovery_ready`; terminal state=`resolved`.

## Terminal result

Canonical R3=`run://worldsim_v63/WS-V63-P4-CAPACITY-01/20260825T051200Z__capacity-h002-s0-r3` passed in 11.863 seconds.
It executed two complete train proposals and two complete selection proposals across 16 chunks each; the largest complete proposal had
117663 points and 263 patches. Peak allocation was 0.256589 GiB. Loss and unscaled gradients were finite, direct CVaR gradients reached
the state, hidden-FREE and authority heads, proposal-token gradient was nonzero, hard violations were zero, checkpoint reload succeeded,
and both repeated and reloaded forward differences were exactly zero. AMP scale remained 1024 and the cuBLAS workspace was `:4096:8`.
No quality, calibration, confirmation or exact-once test data was read. H-P4-002 is supported as a capacity claim only; F17/F18 are
resolved and the preregistered P5 training stage is unlocked.
