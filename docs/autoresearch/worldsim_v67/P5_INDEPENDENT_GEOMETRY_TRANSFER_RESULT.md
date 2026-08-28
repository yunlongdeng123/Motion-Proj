# V6.7 P5 independent geometry transfer result

- Canonical run: `run://worldsim_v67/WS-V67-P5-INDEPENDENT-GEOMETRY-TRANSFER-01/20260828T110336Z__independent-transfer-s0-r1`
- Verdict: `supported_task_untouched_legacy_geometry_transfer`
- Failure delta: `none`

The frozen V6.6 head was evaluated without model/normalization refit or a threshold on the independent V65 P2 legacy cohort.

- units / Actor-units: `72 / 570`
- conflict / clean Actor-units: `312 / 258`
- head AUROC / AUPRC: `0.665176 / 0.676612`
- improvement over deterministic: `+0.165176 / +0.129244`
- scenes above chance: `6/6`
- gates: `4/4`
- wall / peak GPU / RSS: `10.1148s / 0.02359GiB / 0.9341GiB`

The q0 comparator is stronger on this cohort: AUROC/AUPRC=`0.695177/0.706467`, so the head deltas over q0 are
`-0.030001/-0.029854`. This was not a frozen P5 gate and does not invalidate the registered evidence-vs-deterministic transfer,
but it prohibits a cross-cohort claim that the learned head dominates q0. The physical P6-P8 chain remains necessary.
