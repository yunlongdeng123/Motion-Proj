# V6.7 P20 listwise action compiler freeze

P20 trains on four consumed development domains: P10V, P10X, P9 and V65 P2. Each action receives observable qmean, qstd,
eight corridor quantiles, progress, absolute lateral offset and visited support. A `32/16` head emits a case-centered residual
bounded to `±0.02` around qmean.

The loss combines equal-domain Huber, pairwise ordering and a differentiable soft-rank approximation to the bottom-quartile
selected target cost. The selected fraction remains 0.25. No model/normalization target from the confirmation cohort is read until
the compiler artifact and `model_frozen.json` exist.

Confirmation uses V67 P1 scenes `0030/0055/0453/0501/1046/1085`, whose action targets are untouched by P15--P20. Gates are
selected-cost reduction >=0.35, delta over qmean >=0.02, pairwise >=0.70 and at least five non-increasing scenes. One run only;
no architecture, residual, temperature, loss, fraction or gate sweep.
