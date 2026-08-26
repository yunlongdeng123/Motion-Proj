# WorldSim V6.4 Fresh Evidence and UQ Freeze

- Evidence task: `WS-V64-P2E-FRESH-EVIDENCE-01`
- Surface task: `WS-V64-P2S-FRESH-SURFACE-CORPUS-01`
- UQ task: `WS-V64-P4-FRESH-UQ-01`
- Hypothesis: `WS-V64-H-P4-001`
- Status: `preregistered / target quality unread`
- Seed: `0`

## Direct path

不展开完整V6.4 compiler，直接复用已验证的V6.2 evidence和V6.3 surface materializer，把r3完整native sidecar转换为
与retrospective实验相同语义的method-visible surface denominator。先跑72-unit evidence，再跑72-unit surface corpus，
随后只运行一次保持不变的U0/U2比较。

surface只依赖完整method/target evidence grid，因此关闭未被下游消费的100k/query抽样；这避免无关query quota门和磁盘，
不改变surface denominator、method/target sweep或hidden-FREE标签。

fit固定为`scene-0139,scene-0230,scene-0255,scene-0994`；evaluation固定为
`scene-0359,scene-0998`。evaluation target只在P4评分时作为标签，不进入StandardScaler、PCA或GMM拟合。

## Frozen model and decision

U2完全沿用retrospective设置：`17D logits + 256D BEV -> StandardScaler -> PCA-16 ->`按native FREE/OCC geometry分组的
`GMM-4 diagonal density`，seed0；U0仍是max-probability、entropy、inverse-margin三者。禁止切换PCA维数、GMM数、
seed、scene、target、surface或阈值。

唯一晋级判定：

- U2 pooled AUROC相对最佳U0绝对提升至少`0.02`；
- 两个evaluation scene的U2 AUROC都严格优于各自最佳U0（support=`2/2`）。

AUPRC、FPR@95TPR和risk-coverage完整报告但不追加门。通过只支持fresh feature-density uncertainty机制；失败则关闭当前
PCA/GMM表示，不做同数据参数sweep。无论结果如何都不解锁authority、calibration、conditional threshold、LoRA或下游。

## Resources and run identities

- evidence r1：`20260826T084000Z__fresh-evidence-s0-r1`；CPU、最多2 workers；
- surface r1：`20260826T084500Z__fresh-surface-s0-r1`；CPU、最多2 workers；
- UQ r1：`20260826T085000Z__fresh-uq-s0-r1`；CPU；
- 当前磁盘余量约56 GiB，单3090不参与这三步，故不需要多卡。

不新增hash/checksum/fingerprint，不增加smoke/regression。代码只泛化既有task identity和native partition lookup；提交并push
本freeze后直接读取fresh target evidence。
