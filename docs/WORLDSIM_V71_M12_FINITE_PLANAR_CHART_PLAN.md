# WorldSim V7.1 M12 — Finite Planar Surface Chart Plan

状态：`frozen / development-only`  
日期：2026-09-04  
分支：`research/worldsim-v7.1-learned-evidential-surface`

## 1. 科学问题

M9 sphere通过缩半径降低early却损失hit；M10 oblate ellipsoid仍以hit换early；M11将训练/部署统一为解析ellipsoid后，
hit保持但early不改善。三次正交诊断说明问题已不在loss或renderer approximation，而在物理语义：Gaussian density的有限
`1σ`体积不是观测表面的唯一边界，也不应直接作为collision solid。

M12把问题拆开：M8的GT-supervised centers/radii继续承担几何覆盖；新normal head只定义局部surface chart；物理首返回是
ray与有限zero-thickness disc的解析交点。假设该边界能消除ellipsoid法向厚度带来的提前相交，同时保留切平面coverage。

## 2. 顶会依据与迁移范围

- Geometry Field Splatting with Gaussian Surfels（CVPR 2025）：https://openaccess.thecvf.com/content/CVPR2025/html/Jiang_Geometry_Field_Splatting_with_Gaussian_Surfels_CVPR_2025_paper.html
- MAtCha Gaussians（CVPR 2025）：https://openaccess.thecvf.com/content/CVPR2025/html/Guedon_MAtCha_Gaussians_Atlas_of_Charts_for_High-Quality_Geometry_and_Photorealism_CVPR_2025_paper.html
- 2D Gaussian Splatting（SIGGRAPH 2024）：https://surfsplatting.github.io/

迁移“surface primitive / geometry field / chart”而非体density的原则；不引入RGB appearance、mesh atlas refinement、
monocular prior或后处理mesh extraction。

## 3. 表示与输入

每个predicted child是`(c,n,s)`：

- `c,s`逐值冻结canonical M8；
- `n=normalize(n_parent + 0.5*tanh(delta_n))`，`delta_n`由build-only actor features与branch slot预测；
- chart集合与M8 children一一对应，immutable anchors作为半径`0.02m`的小disc；
- 无trajectory、velocity、time、hazard、image输入；rigid trajectory仍只读地把canonical chart搬到world。

## 4. GT supervision

从train target endpoints为每个child构造8NN local plane：

- `1-|n·n_gt|`法向损失；
- `|(c-t_gt)·n|` point-to-plane损失；
- `relu(||P_tangent(c-t_gt)||/s-1)` in-radius coverage；
- 不监督GT背后空间，不生成inside伪标签。

同一chart再进入解析ray physics。对`o+t d`先求plane root
`t=n·(c-o)/(n·d)`，仅保留`t>0`且交点切向距离`<=s`的primitive，所有charts取最小正根。first loss对GT depth，
FREE loss处罚`target_depth-0.20m`之前的交点。无交点使用有限fallback并由local-plane supervision恢复。

## 5. 优化

- canonical M8初始化；seed=`71114`；6 epochs；batch=4；
- local chart geometry与exact disc first/free使用symmetric PCGrad；
- center/radius永不从新head读取，因此point与temporal表征逐值保持M8；
- 单次开发run；不扫描radius multiplier、loss、seed、epoch、depth/lateral tolerance。

## 6. 验收与边界

在既有66-Actor development holdout：

1. canonical M8 point五项合同保持；
2. hazardous exact chart early相对parent-normal M8 chart下降`>=5%`；
3. all chart hit delta `>=-1pp`；
4. Actor/hazard retention=`100%`；所有charts保留。

任一失败登记`V71-F16`并关闭finite-chart family，不做第二seed或radius恢复。通过才冻结fresh AV2 external evaluator。
即使通过，claim只限“由GT监督的可见surface first-return”，不声称watertight occupancy、内部/外部符号或实体碰撞体积。
