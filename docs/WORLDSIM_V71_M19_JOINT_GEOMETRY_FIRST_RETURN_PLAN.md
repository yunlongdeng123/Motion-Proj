# WorldSim V7.1 M19：监督原生的联合几何—首交场

## 研究问题

M7/M8分别证明了actor-canonical set completion与逐帧GT coverage可以改善显式表面；M18证明了GT LiDAR
one-hot first-return监督可以改善共享3D query score的ray-level首交。两者仍是串联训练：M18冻结M8 children，
因此不能证明物理首交梯度能够约束completed anchors且不破坏已学几何。

M19只回答这一耦合问题：在持续保留原生3D几何监督时，让同一GT first-return proper loss反向传播到completed
Gaussian anchors，是否能同时保住M8 geometry并保住M18 ray physics。

## 文献迁移边界

- [Gau-Occ, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Lv_Gau-Occ_Geometry-Completed_Gaussians_for_Multi-Modal_3D_Occupancy_Prediction_CVPR_2026_paper.html)
  支持completion-first再联合优化，但其completed cloud只是soft initialization，公开材料未说明geometry loss是否持续；
- [GaussRender, ICCV 2025](https://github.com/valeoai/GaussRender)把projective depth/semantic loss作为训练期模块，
  并明确与标准3D supervision同时使用；
- [ShelfOcc, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Boeder_ShelfOcc_Native_3D_Supervision_beyond_LiDAR_for_Vision-Based_Occupancy_Estimation_CVPR_2026_paper.html)
  强调动态驾驶场景需先构造metric-consistent native-3D supervision；
- [GaussianFormer-2, CVPR 2025](https://github.com/huang-yh/GaussianFormer)把Gaussian解释为共享3D occupied-region
  distribution，而不是每条ray独立的后处理结果。

M19迁移“原生3D监督与ray projection监督同时存在”这一结构，不复现图像backbone、occupancy mIoU或scene-level
Gaussian模型。

## 冻结表示与GT

- surface head：canonical M8 `GaussianSeedExpansionMLP`，初始化后可学习；observed anchors保持不可学习；
- field head：canonical M18 `RayTerminationLogitField`，初始化后可学习；underlying logit只依赖actor-canonical query与
  build-only local evidence，不输入当前target ray direction；
- geometry GT：held-out actor-canonical endpoints的symmetric Chamfer、local point-to-plane/scale，以及逐帧等权
  endpoint-to-surface coverage；
- physics GT：Actor AABB整ray固定32 bins上的唯一LiDAR first-return bin，categorical CE加固定expected-depth L1；
- deployment：显式surface始终是immutable observed anchors + all children；物理首交始终读取同一field softmax CDF
  median。无UNKNOWN mask、surface filter、threshold search或failed-Actor deletion。

每个训练step先从surface head得到当前children/scales，再把它们直接送入field；categorical loss因此可回传到center/scale。
geometry与physics作为两个GT任务，用已有symmetric PCGrad处理共享参数冲突，但不修改任一target。

## 解耦边界

- geometry/physics在损失和账本中分别报告，只在共享3D anchors处耦合；
- trajectory只作为canonical-to-world只读刚体authority，不输入两个head；static world完全独立；
- image、semantic、velocity、time、hazard label和ray-drop均不进入模型；
- 本实验不回答外观融合、行为预测、完整occupancy或真实道路安全。

## 单次实验与判定

- task=`WS-V71-M19-JOINT-GEOMETRY-FIRST-RETURN-01`；seed=`71121`；从冻结M8/M18初始化；
- 593 train Actors，固定4轮，32 train bins；不扫描loss、bin、threshold、epoch、seed或branch factor；
- development仍是历史暴露的66 Actors，只能作mechanism evidence；
- 最小判定：Actor/hazard retention=100%；相对冻结M8的Chamfer恶化不超过1mm；field hazard early相对baseline降低
  至少5%；field all hit相对baseline下降不超过1pp；逐帧coverage和point early/hit完整报告但不增加门控；
- 若失败，登记`V71-F23`并关闭joint-anchor路线，不用调权重恢复；若通过，在不读取现有AV2 partial quality的前提下
  冻结同一20-log fresh cohort外测。

