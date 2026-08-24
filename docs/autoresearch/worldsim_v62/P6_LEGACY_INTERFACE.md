# P6 legacy28 artifact interface and matched-arm freeze

## Blocker observation

The P6 plan names a frozen V6.1 IR-WM prior sidecar, but the canonical ME3R artifacts persist only `class_label[200,200,16]`,
`occupied_mask`, grid metadata and poses. They do not contain the 17-class logits or 256D BEV features consumed by the frozen P5
model. Re-running IR-WM is explicitly forbidden. Per-voxel uncertainty and latent features therefore cannot be recovered exactly.

The plan also places grouped conformal calibration at P8, after P6/P7. B2 requires a calibrated max-softmax/entropy threshold, B4
requires a separately trained no-evidence-dropout checkpoint, and full M0 requires the later grouped conformal state. None exists at
P6. These arms must not be fabricated or tuned on legacy O_eval.

## Primary-source migration

- [ProtoSeg, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Rethinking_Semantic_Segmentation_A_Prototype_View_CVPR_2022_paper.html)
  shows a nonparametric alternative based on mean training-pixel features for each class. This supports a deterministic class-prototype
  bridge when only class identity survives.
- [When Does Label Smoothing Help?, NeurIPS 2019](https://arxiv.org/abs/1906.02629) shows that hard/smoothed labels discard
  similarity information carried by teacher logits. The bridge is therefore explicitly lossy and cannot be presented as recovered
  uncertainty.
- [Conformal Inference is (almost) Free for Neural Networks Trained with Early Stopping, ICML 2023](https://proceedings.mlr.press/v202/liang23i.html)
  explains that ordinary early-stopped models still require independent holdout calibration for conformal guarantees. P6 does not use
  legacy O_eval to create a conformal threshold; M0 remains deferred to P8.

## Frozen bridge

For each of the 17 IR-WM semantic classes, compute query-weighted mean raw logits and mean BEV features using only the four-scene P5
training split. At legacy inference, the frozen argmax class indexes its prototype. Source-invalid cells remain prior UNKNOWN. Method
FREE/OCC, coordinates and native actor evidence are then supplied exactly as P5 features. The P5 checkpoint is not retrained.

All 17 classes have training support; query counts are:

```text
2065365,29,2880,65833,146046,19431,16408,36308,691,
101474,38543,849891,6392,136424,217090,243489,605441
```

A single read-only distortion audit on the 24 P5 selection units used no selection target for fitting. Full-feature versus prototype
prediction agreement was `0.896898`. Prototype hidden-FREE false-OCC=`0.399349` versus full=`0.384568` and projection-only=
`0.453707`; safe-OCC retention=`0.872897`, target accuracy=`0.452581`, predicted UNKNOWN=`0.221945`, hard violations=`0`.
The bridge remains useful but weaker than the original sidecar, so P6 conclusions are bounded to this artifact-limited transfer.

## Runnable matched arms

| Arm | P6 realization |
|---|---|
| B0 | Replay canonical V6.1 IR-WM argmax method artifact; no inference. |
| B1 | Argmax prior plus exact O_method FREE/OCC hard clamp. |
| B2 | Unavailable: logits absent and Tier-C threshold calibration locked. |
| B3 | Frozen P5 evidential base probabilities before projection, via the class-prototype bridge. |
| B4 | Unavailable: no separately trained no-evidence-dropout checkpoint. B1 remains the projection-only control. |
| B5 | Frozen P5 evidence-dropout/evidential head plus exact projection, without conformal. Primary P6 arm. |
| M0 | Deferred: grouped conformal is P8 after fresh development and calibration. |

The P6 primary gate remains the plan's legacy mechanism gate. Anti-triviality adds the plan's oracle accepted-surface safe-OCC
retention `>=0.50` and a source-valid UNKNOWN upper bound `<=0.50`; it does not add a threshold sweep. Method decisions and candidate
geometry are written before O_eval is loaded. The run does not read confirmation/test and does not create hashes, checksums,
fingerprints or content-addressed artifacts.

## Decision

`V62-F05` is resolved for an artifact-bounded P6 execution. Resolution does not claim that prototype features equal the missing native
features. The full B2/B4/M0 ablation table remains unavailable/deferred for the reasons above. Execute one formal P6 run from a clean
commit; do not add another bridge or model sweep.
