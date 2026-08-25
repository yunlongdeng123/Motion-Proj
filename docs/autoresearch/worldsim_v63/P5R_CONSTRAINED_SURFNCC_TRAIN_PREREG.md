# WorldSim V6.3 P5R Constrained SurfNCC Recovery Preregistration

- Task: `WS-V63-P5R-CONSTRAINED-SURFNCC-TRAIN-01`
- Hypothesis: `WS-V63-H-P5R-001`
- Trigger: P5D supports raw-network objective collapse and rejects authority-veto composition as the primary mechanism
- Status: `protocol preregistered; implementation pending`
- Seed: `0` (the only primary seed)

## Claim boundary

P5R asks one question: can the unchanged SurfNCC representation recover a nondegenerate Physical State candidate when safe-OCC retention,
emitted-OCC coverage and source-valid non-UNKNOWN coverage are optimized as constraints rather than weighted penalties? It does not claim a
new architecture, hard solver, calibration procedure or deployment result. P6, calibration, confirmation and exact-once test remain locked.

The starting model is the P5 epoch-3 best training-objective checkpoint, loaded model-only with a fresh AdamW optimizer. This is a recovery
from the measured collapse, not a second random initialization. The 311D features, 256D width, two neighbor blocks, two patch-attention
layers, one proposal token, complete-proposal context, structural-dropout policy, train/selection scenes, target frames, FP16, learning rate,
weight decay, accumulation, maximum/minimum epochs, patience, CVaR alpha, authority threshold and hard projection remain unchanged.

## Proxy primal-dual objective

The primal base objective is the frozen P5 state, hidden-FREE tail, complete-unit ranking, surface consistency and evidence-authority terms.
The old weighted safe-OCC retention penalty is removed from the base sum and represented only as a constraint. For method-UNKNOWN,
non-contradictory points, the differentiable final OCC proxy is:

```text
soft_OCC = P_projected(OCC) * q_AUTH
```

Observed hard-OCC points retain projected `P(OCC)=1` without an authority veto. Three frozen proxy constraints are optimized:

```text
mean(soft_OCC | safe-OCC) >= 0.60
mean(soft emitted OCC over all points) >= 0.10
mean(soft non-UNKNOWN over all points) >= 0.40
```

The third constraint is exactly the P1 `source-valid UNKNOWN<=0.60` anti-triviality contract expressed as coverage. Hard violations remain
identically zero through the existing exact projection and are never relaxed or dualized.

Three nonnegative multipliers start at zero. The model player minimizes the base objective plus multiplier-weighted differentiable proxy
violations. After each original four-batch gradient-accumulation step, the constraint player performs projected ascent with frozen step size
`0.01` using the corresponding original discrete rates from those same batches: post-authority safe-OCC retention, emitted-OCC coverage and
non-UNKNOWN coverage. Multipliers are not clipped above and are saved per epoch. There is no loss-weight, dual-rate or penalty sweep.

This follows [Two-Player Games for Efficient Non-Convex Constrained Optimization (ALT 2019)](https://proceedings.mlr.press/v98/cotter19a.html):
the model player may use differentiable proxy constraints while the constraint player enforces original nondifferentiable rates.
[Training Well-Generalizing Classifiers for Data-Dependent Constraints (ICML 2019)](https://proceedings.mlr.press/v97/cotter19b.html)
motivates keeping model optimization on the four train scenes and exact checkpoint feasibility on the two scene-disjoint selection scenes.
[SelectiveNet (ICML 2019)](https://proceedings.mlr.press/v97/geifman19a.html) supports treating coverage as a first-class selective-prediction
contract. The implementation is local PyTorch and imports no new optimization package.

## Candidate selection and stopping

Every epoch reports the unchanged full 24-unit selection metrics. A checkpoint is a SurfNCC candidate only when all are true:

```text
hard violations = 0
safe-OCC retention >= 0.60
emitted-OCC coverage >= 0.10
source-valid UNKNOWN <= 0.60
```

Among feasible checkpoints, selection minimizes hidden-FREE tail plus matched rank surrogate and then prefers higher retention, coverage and
secondary accuracy. Before any feasible checkpoint exists, a separate `best progress` checkpoint minimizes the ordered exact gate deficits
before tail risk; it is never named or exported as a candidate. Patience follows progress until feasibility exists and follows feasible
tail-risk improvement afterward. Maximum 12, minimum 4 and patience 3 stay unchanged. Terminal reporting keeps training capability and
candidate promotion as separate booleans.

If no feasible checkpoint exists, P5R stops without a seed/epoch/model/threshold/CVaR/dual-rate sweep and P6 stays locked. If a feasible
checkpoint exists, only that candidate can enter the original P6 matched AB. No artifact hash, checksum or fingerprint is added.
