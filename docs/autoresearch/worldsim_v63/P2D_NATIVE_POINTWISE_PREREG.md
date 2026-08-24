# WorldSim V6.3 P2D Native Pointwise Diagnostic

- Task: `WS-V63-P2D-NATIVE-POINTWISE-DIAGNOSTIC-01`
- Hypothesis: `WS-V63-H-P2D-001`
- Status: `formal preregistered`

This is the one retrospective root-cause diagnostic required by the plan. The V6.2 P5 best CPSC-Lite checkpoint is frozen and receives
true per-cell P2 native logits and BEV latent after deterministic native-to-target-grid mapping. It is not retrained. B0 and the exact
legacy28 denominator/evaluator are unchanged; B1, native unprojected B3 and projected Native B2 are evaluated. Method decisions and
candidates are written before legacy O_eval is opened.

The original P6 gate is unchanged: at least 5/28 accepts, zero false-safe, R10 3/3, at least one new actor and static/disocclusion,
accepted mask area at least 12%, mean/worst accepted FREE conflict at most 0.05, safe-OCC retention at least 0.50, source-valid
UNKNOWN at most 0.50, and zero hard violations. There is one formal run, no threshold/model/seed sweep and no capacity probe.

If Native B2 passes, prototype transfer was the dominant legacy root cause and B2 becomes a strong baseline; SurfNCC must still show
fresh gain. If it fails, pointwise structure/risk remains a root cause and SurfNCC continues. P2D cannot unlock confirmation or serve as
a fresh generalization claim.

## Result

Canonical:
`run://worldsim_v63/WS-V63-P2D-NATIVE-POINTWISE-DIAGNOSTIC-01/20260824T145924Z__native-pointwise-s0-r1`.

Native B2 was rejected: 4/28 accepted and all four were false-safe; accepted mask area was 0.094024, R10 retention 2/3, actor
gain zero, accepted FREE conflict mean/worst 0.045783/0.092105, source-valid UNKNOWN 0.639211, safe-OCC retention 1.0 and hard
violations 0/939206. The accepted set exactly matches the four scene-0242 missing-route-support cases retained by the prototype runs.
Native features are therefore not sufficient for the frozen pointwise/mean-query architecture. Failure ledger delta: `V63-F02 active`.

The post-result primary-source audit retained the P1-frozen migration: deterministic surface neighborhoods follow the efficiency
lesson of [Point Transformer V3](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf),
while explicit FREE support follows [visibility-aware surface reconstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC4897344/).
No P2D recovery or protocol change is authorized; P3 is unlocked.
