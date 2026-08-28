# V6.7 P9 fresh input pipeline result

- Preparation: `run://worldsim_v67/WS-V67-P9-FRESH-PREPARATION-01/20260828T111500Z__fresh-prep-s0-r1`
- Native aggregate: `run://worldsim_v67/WS-V67-P9-FRESH-NATIVE-SIDECAR-01/20260828T114200Z__fresh-native-aggregate-s0-r1`
- Evidence: `run://worldsim_v67/WS-V67-P9-FRESH-EVIDENCE-01/20260828T114300Z__fresh-evidence-s0-r1`

Preparation extracted 10,727 members from shard 4 and produced all six new scenes in `1,145.914s`; quality read remained
false. Native inference ran scene-ready on GPU while later scenes were still preprocessing: 72/72 targets, 3,317,884,577 bytes,
maximum peak `4.13145GiB`, no target/calibration/confirmation content read and no repeated inference. Evidence produced 72/72
units and 86,874,060 bytes in `132.283s`, with zero source-role overlap and no score/quality read.

`V67-F02 resolved_pre_quality_entry_contract` groups the native launcher-only failures: missing task parent for disk-usage
query, unexpanded legacy base config and CUDA device index `1` on a single-index-0 host. The retained successful scene-0348 run
is r3; all other scenes are r1. No failed attempt loaded a native target or emitted a scientific metric.

P10 now applies the frozen V6.6 head once. No input audit or regression matrix is added.
