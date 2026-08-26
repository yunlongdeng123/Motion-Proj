# WorldSim V6.4 最小原生 UQ 机制冻结

- Task: `WS-V64-P3-NATIVE-UQ-01`
- Hypothesis: `WS-V64-H-P3-001`
- Status: `preregistered / implementation staged`
- Seed: `0`
- Quality read: `none after this freeze`

按用户最新指令，本轮不机械铺开完整 V6.4 阶段和测试矩阵，直接验证最关键的问题：原生 IR-WM feature-density
uncertainty 是否比 softmax entropy/margin 更能识别 hidden FREE。

## Scope

本轮复用 V6.3 的 72-unit sidecar/surface corpus，只作 retrospective mechanism diagnostic：

- fit scenes: `scene-0071, scene-0317, scene-0862, scene-1012`；
- evaluation scenes: `scene-0450, scene-1089`；
- evaluation target 不参与 scaler/PCA/GMM 拟合；
- 旧 scene 不升级为 V6.4 fresh claim，也不解锁 calibration/confirmation/test。

## Frozen comparison

- U0: `1-max probability`、normalized entropy、`1-top1/top2 margin`；
- U2: native `17D logits + 256D BEV`，train-only standardization，PCA-16，按 native FREE/OCC geometry
  分组的 4-component diagonal GMM；
- target: 方法时刻为 UNKNOWN、无 contradiction 的曲面点中，隐藏 target 为 FREE；
- metric: point-level AUROC/AUPRC/FPR@95TPR、risk-coverage；逐 scene 与 pooled 同时报告。

GMM 是 OCCUQ-style feature-density mechanism migration，不是 OCCUQ 全系统复现。当前不训练 aleatoric head，不引入
Surface encoder、scene ID、LoRA、阈值 sweep 或新 gate。若 U2 在两个 scene 都无增量，先诊断表示/条件化而不是堆更多
测试；若有稳定信号，下一步直接建立 fresh sidecar cohort。

## Resource

只使用 CPU 和已有 mmap sidecar；拟合每个 scene 等额抽样 50,000 点，selection 评估读取完整 eligible denominator。
不占用 GPU，不需要多卡。
