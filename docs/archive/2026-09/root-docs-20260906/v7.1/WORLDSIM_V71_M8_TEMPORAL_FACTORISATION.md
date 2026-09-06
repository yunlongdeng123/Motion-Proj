# WorldSim V7.1 M8：形状—轨迹因子化与逐帧 GT 覆盖

## 科学问题

M7 已证明 supervision-native set completion 能在开发集同时改善 literal first return、Chamfer 与 hit recall，但它的
set loss把所有 held-out 帧合成一个 target。即使每帧已经进入 Actor canonical frame，稠密帧仍可能支配监督；聚合改善也
不能证明同一 canonical surface 对每个时刻都成立。

M8 只回答动静解耦问题，不重新解决语义、外观或行为风险：

- geometry authority：build-only evidence → 单一 Actor-canonical surface；
- motion authority：冻结的 GT trajectory/pose → canonical-to-world 刚体变换；
- static world authority：独立背景表示，M8 不写入；
- hazard/appearance/velocity/timestamp：不进入 geometry head。

## 一手资料与迁移边界

- DrivingGaussian（CVPR 2024）把大尺度静态背景与逐对象 dynamic Gaussian graph分开组合；
- 4D-GS（CVPR 2024）维护 canonical Gaussians并预测时刻形变，但论文也指出无额外监督时 static/dynamic splitting困难，
  且颜色/opacity变化可能伪装不合理的形状变化；
- DeSiRe-GS（CVPR 2025）在图像运动先验之外加入 static-velocity、LiDAR depth和temporal cross-view geometry losses。

当前项目已有3D Actor identity/pose监督，因此不迁移2D motion-mask discovery，也不引入可学习4D deformation field。仅迁移
“时刻间几何约束必须进入训练目标”这一原则：从同一 actor-canonical GT 中恢复帧组，对每帧 endpoint coverage 等权。

## M8 冻结目标

对 Actor (i) 的 M7 surface (S_i) 和第 (t) 个 held-out 帧的 canonical first-hit endpoints (P_{it})：

\[
L_{\mathrm{frame}} = \frac{1}{|T_i|}\sum_{t\in T_i}
\frac{1}{|P_{it}|}\sum_{p\in P_{it}}\min_{s\in S_i}\|p-s\|_2.
\]

这是单向 target-to-surface coverage：单帧只观察部分表面，因此不使用逐帧 symmetric Chamfer去惩罚该帧不可见但由其他帧
支持的合法表面。原M7 union-set Chamfer、local plane/scale、literal first-return与FREE-before-hit全部保留。

帧组直接由已有 `target_sensor_origins` 的重复行恢复；每个 endpoint与其真实sensor origin同源，不新增数据或标签。M8从
canonical M7初始化，只进行一个固定seed、6轮Stage-P fine-tune；`L_frame`按冻结M7 reference归一化并并入geometry任务，
geometry/physics冲突继续使用同一PCGrad定义。

## 冻结评价

- development split与M7相同且明确有历史预训练暴露；
- primary：原五项physical contract仍全部通过；
- mechanism：frame-balanced endpoint distance相对冻结M7严格下降；
- moving audit：轨迹首点到任一点最大位移 `>0.5m` 为moving，否则为quasi-static，仅分层报告，不作调参/gate；
- Actor identity、trajectory、extent、hazard必须100%保留；部署surface仍为anchors加全部4-child；
- 不读取正在运行的M5/M7 AV2 partial physical metrics；若M8通过，才以相同事前20-log cohort冻结一次外测。

## 明确禁止

- 不把velocity、timestamp、hazard或trajectory坐标喂给geometry head；
- 不引入image feature、SH/opacity、motion mask或可学习canonical-to-world deformation；
- 不按动/静类别训练两个shape head；
- 不做branch factor、loss weight、seed或位移阈值扫描；
- 不用推理mask/filter/delete制造逐帧改善。

