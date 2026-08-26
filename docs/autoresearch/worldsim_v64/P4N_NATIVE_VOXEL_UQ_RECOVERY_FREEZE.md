# WorldSim V6.4 Native-Voxel UQ Recovery Freeze

- Task: `WS-V64-P4N-FRESH-NATIVE-VOXEL-UQ-01`
- Hypothesis: `WS-V64-H-P4N-001`
- Status: `global-boundary GMM recovery preregistered / evaluation scores unread`
- Blocked fit run: `20260826T090000Z__fresh-native-voxel-uq-s0-r1`
- Recovery run: `20260826T091500Z__fresh-native-voxel-uq-s0-r2`

## Resource-triggered recovery

继承的V6.3 surface compiler历史72-unit wall=`47,568.47 s`、最大unit=`3,334.28 s`；fresh surface r1运行约4分钟仍为
`0/72 units`，只写出4 KiB negative-test文件。该compiler同时构造signed distance、patch、normal、actor/proposal registry，
这些字段均不被当前U0/U2使用。r1已按精确PGID停止并原样保留，不删除、不覆盖；没有UQ score或gate被读取。

检索后迁移OCCUQ（ICRA 2025）的native voxel-level feature GMM语义：直接在IR-WM `200×200×16`网格上形成唯一体素
denominator。CuPy/cuCIM GPU EDT可加速旧surface路径，但仍保留无用全栈，因此不采用。

## Frozen denominator and model

每target固定：

1. proposal occupied = native argmax occupied ∪ method observed occupied；
2. 用6-neighbor binary erosion取其边界；
3. 只保留位于target evidence ROI、method=UNKNOWN且非contradiction的唯一native voxel；
4. hidden-FREE标签为同一native中心的target evidence=FREE。

fit/evaluation scene、12 targets、seed0、50k fit points/scene、`17D logits+256D BEV`、PCA-16、GMM-4 diagonal、三种U0
以及两条gate全部沿用原freeze：pooled AUROC gain `>=0.02`且scene support=`2/2`。evaluation标签不进入拟合；不做sweep。

这次恢复发生在任何U0/U2 score读取前，但target evidence已物化；初始只允许预注册native-voxel r1。r1随后触发下述
fit-interface失败，且仍未读evaluation score，因此只追加一次显式r2恢复。r2无论结果如何，不得回到surface、换EDT实现、
换denominator或参数救结果。CPU-only，单3090与多卡均不需要。

## Fit-interface recovery

native r1在四个fit scene完成采样后，occupied-boundary denominator内预测FREE组只有`43`点，少于原双geometry-group
GMM-4的最低`80`点，在GMM拟合前停止；r1只有`resolved.yaml/status.json`共8 KiB，没有模型和evaluation score。

OCCUQ官方实现按真实voxel类拟合密度，并在推理时跨类密度边缘化，而非要求每个待评region同时具备足量预测类。当前region
已冻结为occupied boundary，合法恢复是对其整体拟合一个boundary-global diagonal GMM-4；PCA-16、组件数4、seed0、features、
scenes、denominator与gate不变。v1/r1保持blocked，v2只允许r2一次；不得降最低样本数、复制43点或把evaluation混入fit。
