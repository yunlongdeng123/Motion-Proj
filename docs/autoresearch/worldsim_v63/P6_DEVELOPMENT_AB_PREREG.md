# WorldSim V6.3 P6 Fresh Development Matched-AB Preregistration

- Task: `WS-V63-P6-DEVELOPMENT-AB-01`
- Hypotheses: `WS-V63-H-P6-001`, `WS-V63-H-P6-002`, `WS-V63-H-P6-003`
- Trigger: P5R froze a genuinely feasible epoch-6 SurfNCC candidate
- Status: `protocol frozen; B0/B1/B2 baseline complete; B3 training unlocked; B4/B5/M0 locked`
- Seed: `0` for every trained surface ablation

## Claim boundary

P6 asks three sequential mechanism questions on the existing two-scene, 24-unit Tier-D selection denominator: whether an independently
trained surface encoder improves the frozen Native B2 pointwise baseline; whether CVaR training improves a common exact surface-tail metric
over mean/max training; and whether the frozen M0 authority policy reduces a proposal false-safe surrogate beyond B5. It does not fit a
threshold, read Tier C/H/T or legacy quality, claim calibration, or change the P5R candidate.

## Why the surface arms are independently trained

Applying mean, max and CVaR only as post-hoc summaries of one M0 output would leave B3/B4/B5 state decisions identical. For one fixed
nonnegative point-risk distribution, mean is no greater than upper-tail CVaR, which is no greater than max; the frozen requirement that B5
improve over the better of B3/B4 would therefore be structurally unachievable. B3, B4 and B5 are instead matched loss ablations. Each starts
from the same P5 epoch-3 model weights only, uses a fresh AdamW optimizer, seed 0, the same 48 train and 24 selection units, model width,
FP16, structural dropout, hard projection, training horizon and three primal-dual anti-triviality constraints. The only scientific changes
are the train-time hidden-FREE aggregator (`mean`, `max`, or worst-10% `CVaR`) and disabling authority loss/veto for B3--B5. M0 is not
retrained: it is the frozen P5R epoch-6 candidate.

This follows the controlled-objective ablation used by [UnO (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Agro_UnO_Unsupervised_Occupancy_Fields_for_Perception_and_Forecasting_CVPR_2024_paper.pdf),
which holds architecture fixed and retrains each loss variant, rather than swapping metrics after training. The existing P5D/P5R evidence
requires constraints rather than returning to a scalar retention weight. No extra seed, capacity probe or aggregator sweep is added.

## Frozen arms and order

The only legal order is `B0 -> B1 -> B2 -> B3 -> B4 -> B5 -> M0`.

- B0: native IR-WM argmax on surface points, no hard projection.
- B1: B0 plus the frozen contradiction/FREE/OCC/lifecycle hard projection.
- B2: frozen V6.2 CPSC-Lite checkpoint with true V6.3 native logits/BEV and hard projection.
- B3: independently trained SurfNCC Surface-Mean, no authority decision.
- B4: independently trained SurfNCC Surface-Max comparator, no authority decision.
- B5: independently trained SurfNCC Surface-CVaR, no authority decision.
- M0: frozen P5R SurfNCC-CVaR plus positive OCC authority.

B0--B2 are evaluated first. B3 is then trained and evaluated. If B3 fails its frozen stage gate, the surface family closes and B4/B5/M0
are not evaluated as P6 arms. Only after B3 passes may B4 then B5 be trained. If B5 fails, the tail-risk family closes and M0 is not promoted.
Only after B5 passes is the already frozen M0 evaluated against it.

## Common metrics and the P7 boundary

[SelectiveNet (ICML 2019)](https://proceedings.mlr.press/v97/geifman19a.html) separates target-coverage threshold calibration onto an
independent validation set. [AUC-based Selective Classification (AISTATS 2023)](https://proceedings.mlr.press/v206/pugnana23a.html)
likewise defines empirical risk and coverage on the accepted set. P6 therefore does not choose `lambda` from its selection results; P7 owns
the frozen score grid and threshold fitting.

For P6 anti-triviality only, a target unit is structurally accepted when the arm emits at least one final OCC surface point after its hard
projection and, for M0 only, authority veto. Accepted surface area is the count of final emitted OCC surface points on the unchanged P3
surface registry. Actor/static coverage counts accepted proposals in each stratum. These are the exact P5/P5R emission semantics and do not
use a fitted case-risk threshold.

Every learned arm is also scored with the same evaluation metric, independent of its training loss: for each geometric surface, compute the
worst-10% empirical CVaR of OCC confidence on hidden-FREE points, take the maximum surface risk per proposal, and average over proposals with
hidden-FREE support. The proposal false-safe surrogate is the same upper-tail aggregation of final emitted-OCC confidence per proposal;
M0 sets learned OCC confidence to zero where authority veto makes the final state UNKNOWN. Metrics are reported both pooled and separately
for `scene-0450` and `scene-1089`.

## Gates and stopping

Every promotable surface arm must have zero hard violations, safe-OCC retention at least 0.60, source-valid UNKNOWN at most 0.60, accepted
case coverage at least 0.10, accepted surface area no lower than matched B2, and at least one accepted actor and one accepted static proposal.
Both selection scenes must independently support the corresponding stage comparison.

- H-P6-001 is supported only if feasible B3 lowers the common surface FREE-conflict CVaR by at least 2% versus B2 on both scenes.
- H-P6-002 is supported only if feasible B5 lowers the same common metric by at least 2% versus the better feasible B3/B4 comparator on both
  scenes.
- H-P6-003 is supported only if feasible M0 lowers the common proposal false-safe surrogate by at least 2% versus B5 on both scenes.

Secondary target accuracy is reported but cannot override hard/anti-triviality or risk gates. A lower-risk checkpoint that misses any gate
is not a candidate. Failure closes the corresponding family under the plan stop rules; there is no threshold, gate, epoch, seed, model-size,
CVaR-alpha, dual-rate or authority-threshold recovery inside P6.

Implementation binding: `configs/worldsim_v63/p6_development_ab_v1.yaml`. P7, calibration quality, legacy28, confirmation and exact-once
test remain locked.
