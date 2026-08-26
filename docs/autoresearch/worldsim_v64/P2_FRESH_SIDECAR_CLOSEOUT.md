# WorldSim V6.4 Fresh Native Sidecar Closeout

- Task: `WS-V64-P2-FRESH-NATIVE-SIDECAR-01`
- Hypothesis: `WS-V64-H-P2-001`
- Status: `done / supported capability`
- Canonical: `run://worldsim_v64/WS-V64-P2-FRESH-NATIVE-SIDECAR-01/20260826T082600Z__fresh-native-s0-r3`
- Seed: `0`

## Result

r3 在独立 run leaf 完整物化 6 个 fresh scene、每 scene 12 个 target，共 72/72 units：

- fit：`scene-0139, scene-0230, scene-0255, scene-0994`；
- evaluation：`scene-0359, scene-0998`；
- 每 unit：真实 IR-WM `200×200×16×17` native logits、`200×200×256` BEV latent，以及
  argmax、entropy、margin、source-valid；
- `all_native_features_complete=true`、`prototype_used=false`；
- `target_evidence_read=false`、`calibration_quality_read=false`、`confirmation_content_read=false`、
  `exact_once_test_read=false`。

## Resource

- wall：`172.2085 s`；
- output：`3,317,884,573 bytes`（run leaf约`3.1 GiB`）；
- maximum worker peak GPU：`4.1314 GiB`；
- two-worker peak sum upper bound：`8.2628 GiB`；
- 当前单 RTX 3090 足够，多卡未使用也不需要。

两个新 evaluation scene index `276/756`在正式运行前由本机已有 raw nuScenes 通过官方 DriveStudio preprocessing
物化，各含`196 LiDAR + 1,176 images`；未下载额外 blob，未读取模型质量。

## Claim boundary

本里程碑只证明 fresh native feature sidecar 可以在冻结单卡预算内完整生成，不证明 U2 优于 U0，也不形成
authority、calibration、conditional coverage 或下游 compiler claim。r2 blocked partial 原样保留，其12个 scene-0230
units没有复用到r3。下一步必须在拟合或读取 fresh evaluation quality 前，冻结 fresh evidence/UQ evaluator；保持
PCA-16/GMM-4/seed0，不做参数 sweep。

formal run failure delta=`none`；pre-quality recovery=`V64-F05`；post-run reader recovery=`V64-F06`。
