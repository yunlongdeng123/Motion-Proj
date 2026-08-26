# WorldSim V6.4 Native UQ Retrospective Closeout

- Task: `WS-V64-P3-NATIVE-UQ-01`
- Hypothesis: `WS-V64-H-P3-001`
- Status: `supported_retrospective / fresh validation required`
- Canonical run:
  `run://worldsim_v64/WS-V64-P3-NATIVE-UQ-01/20260826T080200Z__uq-retrospective-s0-r1`
- Scope: V6.3 retrospective mechanism set only

## Result

训练侧仅用四个 V6.3 scene 等额抽取的 200,000 个点拟合 PCA/GMM；两个 evaluation scene 的 3,169,645 个 eligible
曲面点完整进入评分。

| Metric | Best U0 | U2 feature density | Delta |
|---|---:|---:|---:|
| pooled AUROC | 0.497324 | 0.550470 | +0.053146 |
| pooled AUPRC | 0.059739 | 0.076027 | +0.016288 |
| pooled FPR@95TPR | 0.968577 | 0.942892 | -0.025685 |

逐 scene：

- `scene-0450`：U2 AUROC/AUPRC=`0.580307/0.077317`，最佳 U0=`0.494365/0.056813`；
- `scene-1089`：U2 AUROC/AUPRC=`0.530461/0.076841`，最佳 U0=`0.456648/0.059851`。

U2 在两个 scene 的 AUROC 与 AUPRC 都高于 U0。pooled 50% coverage 下 hidden-FREE risk 从总体 prevalence
`0.060847`降到`0.052620`；U0 inverse-margin 同 coverage 为`0.059330`。FPR@95TPR 仍然很高，且
`scene-1089`的 U2 FPR@95TPR 没有改善，因此当前信号只支持“值得 fresh 验证”，不支持 authority/calibration claim。

## Resource

- GPU: none
- Wall: `49.964 s`
- Peak RSS: `1.044 GiB`

当前无多卡需求。

## Decision

`WS-V64-H-P3-001`记为`supported_retrospective`。下一步直接建立小型 fresh development cohort，并复用同一
PCA/GMM 配置；不基于本结果调整 PCA 维数、GMM component、seed 或阈值，也不解锁 LoRA/calibration。
