# WorldSim V7.1 M22：SE(3) 等变 Actor 物理场与动静组合

## 科学问题

M8 已经从监督起点把问题拆开：build-only evidence 预测一个 Actor-canonical surface，逐帧 GT endpoint coverage
直接进入训练目标；时间、速度、hazard、图像和 trajectory 不进入 geometry head。M21 又把同一 canonical centers/scales
解释为 decoder-free Gaussian energy。尚缺的硬证据是部署组合：物理场是否只随刚体轨迹搬运，而不是靠每帧形变、appearance
或静态背景补偿。

M22 不重新训练，也不主张新的几何增益。它只回答动静解耦的部署问题：

- Actor geometry/physics authority：冻结 M8 的 canonical anchors、children 与 scales；
- motion authority：冻结 StreetGS `RigidNodes.instances_trans/quats`；
- static authority：StreetGS `Background` 独立且不读写其参数；
- appearance authority：RigidNodes 的 SH/opacity 仅属于渲染，不进入任何物理 energy query。

## 一手资料与迁移边界

- [DrivingGaussian（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_DrivingGaussian_Composite_Gaussian_Splatting_for_Surrounding_Dynamic_Autonomous_Driving_Scenes_CVPR_2024_paper.html)
  将静态背景与逐对象 dynamic Gaussian graph 分开后再组合；
- [4D-GS（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html)
  用 canonical representation 到时刻空间的映射组织动态场；
- [DeSiRe-GS（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Peng_DeSiRe-GS_4D_Street_Gaussians_for_Static-Dynamic_Decomposition_and_Surface_Reconstruction_CVPR_2025_paper.html)
  表明动静分解需要几何与跨时刻约束，不能只依赖 appearance motion cue；
- [SC-GS（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_SC-GS_Sparse-Controlled_Gaussian_Splatting_for_Editable_Dynamic_Scenes_CVPR_2024_paper.html)
  用稀疏控制节点与局部刚性约束 dense Gaussians。

当前数据已有 Actor identity、canonical 3D GT 与刚体 pose authority，因此不迁移 image motion mask、learned deformation
field 或 ARAP optimizer；只迁移“canonical object 与时变变换分层组合”的结构。

## 冻结表示

对 Actor \(i\) 的 M8 canonical centers \(c_{ij}\)、isotropic scales \(s_{ij}\) 和 StreetGS 只读刚体变换
\(T_{it}=(R_{it},t_{it})\)：

\[
e_i^{\mathrm{actor}}(x)=\log\sum_j\exp\left(-\frac{\lVert x-c_{ij}\rVert^2}{2s_{ij}^2}\right),
\qquad
e_{it}^{\mathrm{world}}(x)=e_i^{\mathrm{actor}}\!\left(R_{it}^{\top}(x-t_{it})\right).
\]

实现等价地把 centers 与 query 同时应用 \(T_{it}\)。刚体变换保持距离，所以同一 energy 在 Actor/world frame
必须等变；scale 不随时间变化。observed anchors 继续固定 `0.08m`，children 使用冻结 M8 supervised scale。

## 一次性审计

- cohort：`scene-0230` 中现有 identity bridge 的 12 个匹配 Actor，不换 Actor、不按运动幅度筛选；
- trajectory：对每个 Actor 使用 `instances_fv` 的全部有效帧做运动统计，从中固定取首/中/末最多三帧做数值等变审计；
- query：每个 Actor 从 canonical centers 等距抽取最多 64 个 query；距离审计最多 32 个 centers；
- 最小判定：12 个 identity match、至少一个 moving Actor、energy 最大绝对残差 `<=1e-4`、pairwise-distance
  最大残差 `<=1e-5m`；这些是实现正确性的数值容差，不是科学指标扫描；
- 输出：逐 Actor trajectory/composition rows、aggregate summary 与只读 authority 声明；不生成新 checkpoint。

## 明确禁止

- 不把 trajectory、timestamp、velocity、hazard、image、SH 或 opacity 输入 M8 geometry/energy；
- 不学习 per-frame deformation，不按动态/静态训练两个 shape head；
- 不修改 StreetGS checkpoint、trajectory 或 Background，不渲染后再 filter；
- 不把 M22 的等变恒等式包装成新的几何/跨域性能；M21 fresh AV2 仍独立等待 20/20 后判定。

