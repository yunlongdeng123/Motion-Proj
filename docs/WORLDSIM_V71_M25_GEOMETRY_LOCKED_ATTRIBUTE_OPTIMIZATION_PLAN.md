# WorldSim V7.1 M25：Geometry-Locked Attribute Optimization

## 问题

M24证明M23的309个physical carriers可进入真实StreetGS rasterizer，但最近邻复制SH/opacity在冻结视角使Actor-footprint
PSNR下降11.37dB。M25只回答一个更窄的问题：在physical support完全不动时，多视角图像监督能否把SH/opacity训练到
held-out外观改善；它不再测试几何、first-return或AV2泛化。

GeoSplatting指出结构化Gaussian的高频外观受primitive density限制，并以deferred shading补偿；SC-GS把稀疏运动控制与
稠密appearance分开；Geometry Field Splatting则说明geometry field与颜色kernel可以在可微渲染中分工。M25先测最低成本
的attribute-only上界，若仍不足则关闭one-carrier路线，而不是移动GT-supervised geometry救图。

## 冻结表示与视图

- physical carrier：M23/M24同一Actor的309个M8 centers/scales；identity canonical rotations；
- trajectory：StreetGS Actor 12的只读逐帧SE(3)；Background、其他Actors与相机保持checkpoint原值；
- trainable：只允许目标Actor 309行的SH DC、SH rest与opacity logits；其他RigidNodes行通过梯度mask保持不动；
- forbidden：center/scale/rotation/trajectory/Background更新、primitive增删、image-to-physical-query路径；
- 可见帧并集`[0,84]`已由冻结extrinsics/intrinsics与GT Actor center确定；在M25 quality读取前固定8个train与6个
  held-out frame-camera pairs，M24已暴露的frame42只进入train；不按PSNR筛view。

## 训练目标

每个view先从冻结checkpoint渲染original与actor-hidden。二者差异只定义appearance ROI，不进入physical query。优化图像为
真实dataset RGB：

\[
\mathcal L_{app}=\operatorname{SmoothL1}_{\Omega_A}(I_{carrier},I_{GT})
+0.05\operatorname{L1}_{\bar\Omega_A}(I_{carrier},I_{hidden})
+10^{-3}\lVert\sigma(o)-\sigma(o_0)\rVert_1.
\]

第一项学习目标Actor外观，第二项限制carrier向原Actor footprint外泄漏，第三项只抑制opacity消失捷径。三项都只更新
appearance attributes；M8 geometry仍只来自先前训练内GT set/plane/scale/frame supervision。

固定seed71123、320 steps、单视角循环，不扫lr/steps/loss weight。输出只包含轻量attribute sidecar、逐view图像/rows与
summary，不写StreetGS checkpoint，不生成hash/checksum/fingerprint。

## 判读

- pooled held-out footprint PSNR以original StreetGS、M23-nearest初始化和M25-final并列报告；
- 唯一学习信号判定是final held-out PSNR是否高于initial；不设画质通过阈值；
- 无论改善与否都报告相对original的剩余gap。若改善但仍显著落后，结论是attribute training有效但309-primitives容量不足；
  若不改善则attribute-only路线直接关闭；
- 结果不得影响M8/M21 physical选择，M21 AV2 partial quality继续不读。

## 参考

- Geometry Field Splatting with Gaussian Surfels, CVPR 2025；
- GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering, ICCV 2025；
- SC-GS: Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes, CVPR 2024。
