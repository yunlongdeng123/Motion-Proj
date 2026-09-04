# WorldSim V7.1 M16 — Full-Ray Native Query Field Plan

状态：`FROZEN`（2026-09-04）  
任务：`WS-V71-M16-FULL-RAY-DIRECT-QUERY-01`  
假设：`WS-V71-H-M16-FULL-RAY-NATIVE-SUPERVISION`

## 1. Why this is a new level

M10--M15在plane、Gaussian support、disc、ball和one-sided cell间迁移，始终由有限M8 primitives的union承担完整
first-return surface。M15虽把hazard early降低35.11%，hit仍下降13.54pp；继续调primitive support只会重复precision/
coverage二选一。

M16先改变GT construction：像QueryOcc的官方实现一样，直接沿raw-LiDAR ray构造positive/negative native 3D queries；
不是在输出后过滤surface。ShelfOcc关于native metric 3D target优先于render-only supervision的结论与此一致。GaussRender只作为
对照：其render loss与3D supervision并用，因此本实验不把2D render consistency当物理证明。

## 2. Representation and supervision

- M8 children、scale与local latent只作冻结build evidence；
- 每个3D query读取nearest-4 child latent、相对坐标、oriented normal coordinate与scale；
- local decoder直接预测有量纲signed scalar，学习式attention选择相关children；无plane/ball/cylinder base field；
- 每条GT target ray在Actor AABB内采8个FREE points，从entry均匀覆盖至`hit-0.10m`；label为到hit的正距离并截断
  `0.50m`；
- target hit label=`0`，return后`0.05/0.10m`为负距离；保留GT normal probes；
- geometry与FREE/behind physics经PCGrad联合训练；所有标签来自GT ray/target geometry；
- 部署直接取同一signed scalar在AABB内首个positive-to-nonpositive crossing，无UNKNOWN、opacity阈值、primitive filter。

## 3. Frozen protocol

- seed=`71118`，epochs=`6`，actors=`593 train + 66 development holdout`；
- free samples/ray=`8`，field clip=`0.50m`，neighbors=`4`，其余优化设置继承M15；
- M8 point五门必须保持；field hazard early相对M8降低`>=5%`，field all hit delta `>=-1pp`；
- 禁止query-count、clip、capacity、loss、seed、epoch或阈值扫描；development失败登记`V71-F21`；
- 不读Source Final或未完成AV2 aggregate。

## 4. Claim boundary

通过只说明“完整GT ray query监督 + local continuous decoder”在当前development cohort同时改善early并保持hit。必须等待冻结
AV2聚合才能主张跨域泛化；本实验不解决image fusion、dynamic/static routing或轨迹物理。

## Sources

- [QueryOcc official implementation](https://github.com/zenseact/queryocc)
- [ShelfOcc, CVPR 2026](https://arxiv.org/abs/2511.15396)
- [GaussRender official implementation, ICCV 2025](https://github.com/valeoai/GaussRender)
