# P6R full-native selective MLP freeze

Date: 2026-08-26

P6 rejected the frozen PCA-16 linear U3 at every registered coverage. This recovery is a new model version, not a threshold rescue.

## Split freeze

- Development training: the 16 scenes whose target quality P6 already consumed.
- Independent calibration: the former eight confirmation scenes (`1079,1097,1106,0576,0067,0258,1083,0738`); their target quality remains unread at this freeze.
- New confirmation: description-only selection from the remaining IR-WM train-temporal pool, requiring at least 40 samples and excluding every prior V6.1--V6.4 scene. A shared `random.Random(1)` shuffles sorted candidates in the frozen order night, rain, construction, vulnerable-transit, selecting two per stratum.

| stratum | scenes | processed indices |
| --- | --- | --- |
| night | scene-1023, scene-1105 | 781, 844 |
| rain | scene-0903, scene-0451 | 689, 365 |
| construction | scene-0981, scene-0537 | 743, 427 |
| vulnerable_transit | scene-0789, scene-0157 | 610, 117 |

The selection used only scene name, description, sample count, and exclusion membership. It did not read Occupancy, hidden-FREE, UQ, model scores, or target quality.

## Model freeze

- Input: all 273 native dimensions (17 logits + 256 BEV); no PCA and no scene identity.
- Network: `273 -> 128 -> 64 -> 1`, GELU, dropout 0.10.
- Loss: focal BCE, gamma 2.0, alpha 0.75.
- AdamW, lr 1e-3, weight decay 1e-4, 20 epochs, batch 16,384, seed 0.
- Sampling: at most 49,152 native boundary points per development scene, evenly capped over 12 cases.
- No width, loss, seed, epoch, learning-rate, or sampling sweep; development AUROC is descriptive and not a gate.

After the model artifact is frozen, the former confirmation eight may be read once for the unchanged case calibration protocol: candidate coverages 0.05--0.50, conflict threshold 0.05, target case risk 0.05, confidence 0.95, and Bonferroni one-sided Clopper--Pearson selection. The new confirmation eight remain unread unless calibration selects positive coverage.

No hash, checksum, fingerprint, broad smoke suite, or regression matrix is added.
