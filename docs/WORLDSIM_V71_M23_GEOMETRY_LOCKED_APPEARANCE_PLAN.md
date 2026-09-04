# WorldSim V7.1 M23：Geometry-Locked Appearance Carrier

## 问题与边界

M8/M21 已把 Actor geometry 与 ray physics 绑定到 actor-canonical GT supervision，M22 已确认该物理场仅由只读
SE(3) trajectory 搬运。剩余接口问题不是“让图像再修几何”，而是图像学习的 appearance 如何进入统一 Gaussian
资产而不破坏已通过的 3D geometry。

Gau-Occ 的 image/LiDAR fused FFN仍共同预测center、scale、rotation与semantic，公开材料没有geometry stop-gradient或
持续anchor/free-space loss，因此不能直接迁移为geometry-preserving方案。M23采用更窄的attribute-carrier范式：

- physical carrier：冻结M8 anchors/children与M8 scales；observed anchors固定`0.08m`；
- visual source：冻结StreetGS Actor-owned `_features_dc/_features_rest/_opacities`；
- association：只在同一Actor canonical frame内，为每个physical center查找最近appearance Gaussian；
- copied attributes：只复制SH与opacity logit；绝不复制/更新appearance center、scale或rotation；
- motion/static：继续使用M22只读trajectory；Background不进入sidecar。

该sidecar回答“怎样加入image-trained information而不改变3D geometry”，不主张图像提高几何，也不取代真实rendering
quality评价。

## 一手资料

- [Feature 3DGS（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Feature_3DGS_Supercharging_3D_Gaussian_Splatting_to_Enable_Distilled_Feature_CVPR_2024_paper.html)
  展示在3D Gaussians上承载独立2D-distilled feature field；
- [Neural Shell Texture Splatting（ICCV 2025）](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Neural_Shell_Texture_Splatting_More_Details_and_Fewer_Primitives_ICCV_2025_paper.html)
  明确把surface geometry与texture field sampler分解；
- Gau-Occ（CVPR 2026）作为反边界：completed geometry仅为soft initialization，融合后仍可被image/occupancy gradient移动。

M23不复现语言/语义蒸馏，也不训练纹理场；只迁移“视觉属性是独立通道”的结构。

## 一次性实现与报告

- cohort：与M22相同的`scene-0230` 12个identity-matched Actors，不筛选；
- compute：GPU分块同Actor最近邻；只在canonical坐标做association；
- artifact：`GEOMETRY_LOCKED_APPEARANCE_SIDECAR.npz`，包含M8 centers/scales、复制的SH/opacity、Actor offsets/
  identity；另保留StreetGS原rendered image作为只读appearance reference；
- descriptive：physical surface literal first-return/Chamfer/hit、attribute assignment distance分布、Actor/hazard counts；
- minimal decisions：12个identity matches、每个physical Gaussian都获得同Actorvisual attribute、Actor/hazard retention 100%；
- no training、no checkpoint write、no image-to-geometry gradient、no render filter、no parameter/threshold sweep。

## Claim 边界

M23若通过，只支持一个工程与表示命题：image-trained attributes能附着到GT-supervised physical carrier而不改其坐标或
scale。它不证明photorealism、novel-view quality、semantic accuracy或跨域泛化；这些属性也不进入M21 first-return判定。

