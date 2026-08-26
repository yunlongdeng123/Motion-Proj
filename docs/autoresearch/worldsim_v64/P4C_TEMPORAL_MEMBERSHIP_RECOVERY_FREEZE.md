# P4C temporal-metadata membership recovery freeze

Date: 2026-08-26

The v1 blind feeder completed seven native scenes. The eighth, `scene-0276`, completed DriveStudio preprocessing but failed before any native output or target-quality read because the frozen IR-WM train temporal pickle has no `infos["scene-0276"]`. The official BEVFormer/IR-WM pipeline consumes generated temporal train/val info files rather than the undivided nuScenes scene table; v1 selection checked the latter but omitted membership in the former.

V64-F18 is an input-membership failure, not a model or policy result. The recovery keeps the seven successful scenes, all target frames, C0/M0 coverages, model, gates, and run denominator unchanged. It adds temporal-pickle membership to the metadata-only selection contract and replaces only the invalid vulnerable-transit scene.

The seed-2 shuffled vulnerable fallback order was reconstructed from commit `4813438`, before P4C cohort text entered the used-scene ledger. The immediate next string candidate `scene-0572` is a false substring match (`ped` occurs inside `flipped`) and is not a vulnerable/transit description. The first temporal member satisfying token-level vulnerable keywords is `scene-0813(631)`, whose description explicitly contains pedestrians. No occupancy, hidden-FREE, model score, or target quality was read.

The replacement prep is frozen as `20260826T164500Z__replacement-prep-s0-r1` using exact temporary path `/root/autodl-tmp/tmp/worldsim_v64_p4c_replacement_raw_batch`; native is `20260826T165000Z__confirmation-native-scene-0813-s0-r1`. The corrected aggregate/evidence/exact-once IDs remain `170000Z/171500Z/173000Z`. No valid scene is recomputed and `scene-0276` cannot enter any aggregate or score.
