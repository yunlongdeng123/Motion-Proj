# V6.7 P1 frozen geometry-head transfer result

Canonical run:
`run://worldsim_v67/WS-V67-P1-GEOMETRY-TRANSFER-01/20260828T104342Z__geometry-transfer-s0-r1`.

The frozen V6.6 8-feature/2x32 Actor-local head was applied once to the V6.7 task-untouched legacy cohort without model,
normalization or threshold refit.

- source units / eligible Actor-units: `72 / 517`;
- conflict / clean: `295 / 222`;
- transfer AUROC/AUPRC: `0.710521 / 0.730703`;
- improvement over deterministic: `+0.210521 / +0.160103`;
- q0 AUROC/AUPRC: `0.688762 / 0.725925`;
- above-chance scenes: `6/6`;
- wall / peak GPU / RSS: `10.84445s / 0.02359GiB / 0.93161GiB`.

All four gates pass. This supports fixed-budget local REPAIR/ABSTAIN ranking on a task-untouched but globally consumed legacy
cohort. It does not support Actor deletion, fresh/population generalization or physical repair by itself.
