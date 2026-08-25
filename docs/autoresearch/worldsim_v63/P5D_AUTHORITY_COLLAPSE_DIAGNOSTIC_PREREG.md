# WorldSim V6.3 P5D Positive-Authority Collapse Diagnostic Preregistration

- Task: `WS-V63-P5D-AUTHORITY-COLLAPSE-DIAGNOSTIC-01`
- Hypothesis: `WS-V63-H-P5D-001`
- Trigger: the terminal P5 best training-objective checkpoint has `SafeOCCRetention=0`
- Status: `prepared; execution locked until the P5 terminal artifact confirms the trigger`

## Scientific correction and scope

P5 capability completion and SurfNCC promotion are distinct. A checkpoint may finish finite training, retain zero hard violations and minimize
the frozen lexicographic training objective while still being scientifically unusable. In particular, a checkpoint with zero safe-OCC
retention is named only the **best training-objective checkpoint**. It is not a best SurfNCC candidate and cannot unlock P6. Low false-safe
risk obtained by sending both dangerous and safe surfaces to UNKNOWN has no Physical State Compiler utility.

P5D is a no-update mechanism diagnosis on the frozen P5 checkpoint. It reads only all 48 Tier-D training units from
`scene-0071/0317/0862/1012`; it does not read the two P5 selection scenes again and does not read P6, calibration, confirmation, H or T.
The original, unmasked train-unit view is used because the question is whether the deployed decision path can recover positive OCC authority
on examples already inside the training support. Structural dropout selectors are not resampled. There is no optimizer, checkpoint selection,
threshold change, gate relaxation, new seed, additional epoch, larger model, CVaR-alpha change or hard-projection change.

## Frozen group distributions

Every point satisfying `method=UNKNOWN && !contradiction` is assigned to exactly one diagnostic group by its original training target:

1. `safe_occ`: target OCC;
2. `hidden_free`: target FREE;
3. `unknown`: target UNKNOWN.

For each group, the formal run accumulates a 1000-bin bounded distribution and mean/std/min/P01/P05/P25/P50/P75/P95/P99/max for:

- predicted `q_AUTH`;
- raw network `P(OCC)` before hard projection;
- `P(OCC)` after hard projection;
- point risk;
- patch CVaR mapped back to its points;
- proposal CVaR mapped back to its points.

It also records FREE/OCC/UNKNOWN counts at three decision stages: raw network argmax, post-hard-projection argmax and post-authority-veto
decision. The report includes the number of projected OCC decisions removed solely by `q_AUTH<0.50`, authority-target prevalence and a binned
safe-OCC-greater-than-hidden-FREE AUC. A single six-panel distribution figure is required; this is mechanism evidence, not a quality gate.

## Frozen gradient probe

Direct gradients are measured without parameter updates on the four training units at target frame 17, one per training scene. Every packed
batch contributes raw and frozen-weighted L2 gradient norms for the whole model and for trunk/state/hidden-FREE/authority/risk heads. The
components are:

- the exact P5 hidden-FREE training term;
- its direct tail and auxiliary BCE parts separately;
- the exact safe-OCC retention term;
- the exact authority term.

The probe also reports the tail-versus-retention gradient cosine. It reuses the P5 complete-patch/proposal context path and the same FP16,
math-SDPA and deterministic cuBLAS numerical contract. Target 17 is fixed before reading diagnostic values; no quality-driven unit choice is
allowed.

## Interpretation table

The diagnostic produces evidence, not a new promotion score:

- `representation/supervision collapse`: safe-OCC and hidden-FREE remain substantially overlapping in both `q_AUTH` and raw `P(OCC)`, with
  no useful separation before policy composition;
- `risk/authority composition failure`: raw or post-projection safe-OCC decisions are present, but the authority veto removes them;
- `objective optimization collapse`: the raw network/post-projection path already emits essentially no safe OCC while the frozen loss still
  supplies a nonzero positive-authority/retention signal, especially when tail gradients dominate or oppose retention gradients.

The interpretation uses the complete distribution and stage transition evidence rather than adding another scalar pass threshold. If the
evidence supports objective collapse, the only authorized next training hypothesis is a separately preregistered constrained optimizer:
minimize surface tail risk subject to the unchanged safe-OCC retention, coverage and exact hard-violation constraints. Simple loss reweighting
is not authorized. P6 remains locked until a later checkpoint satisfies the original promotion gates.

## Literature migration

[SelectiveNet (ICML 2019)](https://proceedings.mlr.press/v97/geifman19a.html) treats coverage as an explicit selective-prediction requirement
rather than an accuracy footnote. [Two-Player Games for Efficient Non-Convex Constrained Optimization (ALT 2019)](https://proceedings.mlr.press/v98/cotter19a.html)
provides proxy-Lagrangian treatment for differentiable proxies with original nondifferentiable rate constraints, while
[Cotter et al. (ICML 2019)](https://proceedings.mlr.press/v97/cotter19b.html) separates model optimization from constraint enforcement on an
independent data split. The maintained design examples in
[TensorFlow Constrained Optimization](https://github.com/google-research/tensorflow_constrained_optimization) and the PyTorch-native
[Cooper library](https://github.com/cooper-org/cooper) confirm that retention/coverage rates can be represented as constraints. P5D imports
no new optimization dependency; these sources constrain the next hypothesis only after the frozen mechanism evidence is available.

No artifact hash, checksum or fingerprint is added.
