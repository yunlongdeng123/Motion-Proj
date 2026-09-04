# WorldSim V7.1 M46 — train--deploy aligned oriented support

## Intervention

M45 showed that GT-supervised oblate child kernels add 3.87pp hit recall but miss hazard early safety by 0.034pp because M11
was optimized for earliest hard intersection, not the M39 categorical CDF. M46 initializes M11 and trains only its normal and
normal-thickness output rows through the deployed categorical measure. M8 centers/tangent scales and M35/M38 authority remain
immutable.

For native GT rays, the loss is `-log P(d >= d_gt-0.20m) - log P(|d-d_gt|<=0.20m)` under oriented categorical Gaussian
energy. The original GT local-normal, boundary, and 0.02m thickness supervision is retained at weight 0.25. Thus the train
target itself constrains physical free and hit intervals; no post-hoc decision changes.

## Frozen protocol

- 593 train / 66 exposed holdout Actors, seed 71146, 4 epochs, 32 train / 64 deploy samples.
- M11 initialization; point encoder, slot embeddings, and shared hidden head frozen. Gradients reach only normal/thickness
  output rows; centers, tangent scales, anchor kernel, both F/O/U heads, pose, and CDF median are frozen.
- No hazard/image/motion input, event/loss/thickness/normal/bin/median/seed sweep, filtering, deletion, or M43 partial read.
- Decisions versus M39: all early delta ≤ 0; hazard and clear early deltas ≤ 0; all hit delta ≥ -1pp.

If any decision fails, register `V71-F45` and close oriented-support training rather than tuning. Even if supported, M46 cannot
replace the already running M43 candidate and needs a future untouched external cohort.
