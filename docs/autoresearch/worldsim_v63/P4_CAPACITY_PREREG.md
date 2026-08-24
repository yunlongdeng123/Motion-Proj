# WorldSim V6.3 P4 SurfNCC Capacity Preregistration

- Task: `WS-V63-P4-CAPACITY-01`
- Hypothesis: `WS-V63-H-P4-001`
- Status: `implementation staged / execution locked until P3 formal pass`
- Quality conclusion: forbidden

Temporary synthetic interface history: r1 was rejected by PyTorch autocast at probability-form BCE before real data; the official AMP
recovery exposes hidden-FREE/authority logits and uses BCE-with-logits while preserving sigmoid inference. Synthetic r2 completed finite
forward/backward with proposal-token gradient. Failure=`V63-F08 resolved`; no formal P4 run exists yet.

The shared model also accepts packed patch tokens with one learned token per proposal for later full-denominator P5 throughput. P4 still
passes one proposal per batch. A synthetic packed API typo was recovered via official `torch.nonzero`; failure=`V63-F09 resolved`.

P4 uses exactly one train unit (`scene-0071/f017`) and one scene-disjoint selection unit (`scene-0450/f017`). Within each unit it
chooses the largest static and largest actor proposal by point count only, then takes complete deterministic patches up to the frozen
8192-point microbatch. This is a geometry/resource choice, not quality selection. Two AdamW optimizer steps with four-way gradient
accumulation are sufficient to exercise forward, backward, checkpoint and reload; they do not select a model for P5.

The input is 311D and retains all 17 native logits and all 256 BEV channels. SurfNCC uses a two-layer point MLP, two exact 6-neighbor
residual aggregation blocks, mean patch tokens, two 8-head patch Transformer layers and one learned proposal token at hidden width 256.
Heads produce projected three-state probabilities, hidden-FREE probability, positive authority, patch risk and proposal risk. Patch risk
is frozen empirical upper-tail CVaR at alpha 0.90; proposal admission risk remains maximum patch CVaR. V6.2 closed-form projection is
applied in every forward and must have zero violations.

Train-only structural dropout uses RNG seed 20260824 and selects one applicable family uniformly per proposal. Ray dropout samples 25%
of unique ray bundles; spatial dropout takes the L-infinity-nearest 25% block around one seeded surface point; temporal dropout removes
one of three sweep columns and deterministically recomputes visible FREE/OCC/UNKNOWN/contradiction; patch dropout removes 25% of patch
IDs; actor dropout removes 25% of observed actor-hit support while retaining identity/current/swept lifecycle features. Masked points
lose method hard evidence and distance features, while target fields remain supervision only. No mask is shared with selection or later
calibration. VideoMAE and MaST-Pre motivate preserving structured temporal masks, but their objectives and high mask ratios are not
migrated.

For ray, spatial, patch and actor-observation masks, the affected points also lose temporal-count and observed-actor support features;
otherwise those channels would leak the evidence that the hard-state channel claims was removed. Native logits/BEV remain visible as
the frozen learned prior. Temporal-window masking instead recomputes the remaining-sweep counts and hard state explicitly.

The probe passes only if losses/gradients are finite, hidden-FREE/CVaR and proposal-token gradients are nonzero, hard violations are
zero, checkpoint reload and repeated forward are exact, both scenes execute, and peak allocated GPU memory is at most 22 GiB. It reports
only capability, tensor counts, throughput and resources—no accuracy, false-safe or promotion conclusion. Calibration, confirmation and
exact-once test remain sealed; no hashes, checksums or fingerprints are created.
