# WorldSim V6.4 Native-Voxel UQ Recovery Freeze

- Task: `WS-V64-P4N-FRESH-NATIVE-VOXEL-UQ-01`
- Hypothesis: `WS-V64-H-P4N-001`
- Status: `preregistered / UQ scores unread`
- Run: `20260826T090000Z__fresh-native-voxel-uq-s0-r1`

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

这次恢复发生在任何U0/U2 score读取前，但target evidence已物化；因此只允许这一个预注册native-voxel r1。无论结果如何，
不得回到surface r1、换EDT实现、换denominator或参数救结果。CPU-only，单3090与多卡均不需要。
