# P5 CPSC-Lite formal training closeout

## Identity

- task: `WS-V62-P5-CPSC-LITE-TRAIN-01`
- hypothesis: `WS-V62-H-P5-001`
- canonical: `run://worldsim_v62/WS-V62-P5-CPSC-LITE-TRAIN-01/20260824T092636Z__cpsc-lite-train-s0-r1`
- source at launch: `dd6ff70`
- config: `configs/worldsim_v62/p5_cpsc_lite_v1.yaml`
- mode/seed: `formal / 0`

Identity intentionally uses logical paths, semantic config, task/run and Git. No hash, checksum, fingerprint or content-addressed
identity was added.

## Frozen input and feature boundary

Training used the 48 P2/P4 units from `scene-0071/0317/0862/1012`; selection used all 24 units from the scene-disjoint
`scene-0450/1089` split. Inputs were frozen IR-WM prior logits/BEV features, method-visible evidence, query coordinates and
actor support. Target evidence was supervision only. Query type, dropout evidence and target evidence were never model features.
IR-WM was not resident, and legacy O_eval, confirmation and exact-once test were not read.

## Execution and resources

- parameters: `608,366`; prior/query feature dimensions: `278/13`
- epochs: `9`; optimizer steps: `1,512`; early stop: frozen `min_epochs=4, patience=3`
- best epoch: `5`; best selection objective: `2.0991646573`
- objective trace: `2.27343, 2.20886, 2.14915, 2.13588, 2.15396, 2.09916, 2.18353, 2.14426, 2.21686`
- FP16 peak allocated GPU memory: `0.3724217 GiB`; wall: `341.660 s`
- BEST/FINAL model bytes: `2,450,018 / 2,450,068`

The frozen resource ceiling was passed with substantial margin. No extra seed, smoke or regression matrix was added.

## Best selection result

| Metric | Learned CPSC-Lite | Projection-only | Delta |
|---|---:|---:|---:|
| hidden-FREE false-OCC | 0.384568 | 0.453707 | -0.069139 (-15.24% relative) |
| safe-OCC retention | 0.901058 | 0.900680 | +0.000378 |
| target accuracy | 0.483756 | 0.356765 | +0.126991 |
| predicted UNKNOWN | 0.247579 | 0.087798 | +0.159782 |
| unconstrained UNKNOWN | 0.469596 | 0.166530 | +0.303066 |
| hard violations | 0 / 1,286,134 | 0 | 0 |

Learned predicted fractions were FREE/OCC/UNKNOWN=`0.331365/0.421056/0.247579`. The model therefore did not collapse to
all-UNKNOWN. The hidden-FREE improvement also did not come from reduced safe-OCC retention. Best-checkpoint losses were total=
`2.099165`, query=`0.534274`, evidential=`0.905703`, hidden-FREE=`0.296833`, safe-OCC=`0.417295`, actor=`1.029491`,
prior-preserve=`0.852470`.

## Decision

The frozen P5 hypothesis passes its development criterion: training is finite and resource-bounded, exact projections remain
unviolated, held-out hidden-FREE risk improves over projection-only, safe occupied support is retained, and selective UNKNOWN is
nontrivial rather than total abstention. This is not a legacy false-safe or scene-generalization claim.

P5 is closed with `failure_ledger_delta=none`. The next task is the single frozen legacy28 matched mechanism benchmark
`WS-V62-P6-LEGACY28-ME-01`; it reuses the P5 best model and existing V6.1 artifacts without rerunning IR-WM.
