# Gau-Occ 监督与几何锚点审计

日期：2026-09-04  
用途：为 WorldSim V7.1 的 supervision-first M6 冻结可迁移机制与不可声称边界。

## 一手资料与代码边界

- 主文：CVPR 2026, *Gau-Occ: Geometry-Completed Gaussians for Multi-Modal 3D Occupancy Prediction*，§3.1--3.4、Eq. (1)--(17)：<https://openaccess.thecvf.com/content/CVPR2026/html/Lv_Gau-Occ_Geometry-Completed_Gaussians_for_Multi-Modal_3D_Occupancy_Prediction_CVPR_2026_paper.html>
- 补充材料 §7：<https://openaccess.thecvf.com/content/CVPR2026/supplemental/Lv_Gau-Occ_Geometry-Completed_Gaussians_CVPR_2026_supplemental.pdf>
- 截至审计日，CVF Related Material 只有 PDF、supplement 和 arXiv，未列代码；作者名与标题的 GitHub 精确检索也没有可核验的官方仓库。因此本文不虚构 commit、config、函数名或未公开的梯度路径。

## 核验事实

### Completion target

LCD 输入当前稀疏 LiDAR 扫描 `P`，target `T` 是连续 `K=20` 个 sweeps 经 ego-motion alignment 后的累积点云，不是 occupancy voxel GT。主文没有描述 object-motion compensation；直接迁移到动态 Actor 会产生拖影风险。

Eq. (4) 使用局部加噪 `T_j^(t)=T_j+sqrt(1-alpha_bar_t) epsilon`，没有普通 DDPM 对 clean target 的 `sqrt(alpha_bar_t)` 全局缩放；Eq. (5) 为条件 noise-MSE。补充材料固定 LCD 预训练 20 epochs、`T=1000`、线性 beta `3e-5 -> 7e-3`、DPM-Solver 50 步。论文未说明初始噪声点、输出点数与动态点处理细节。

### Completed geometry 与 Gaussian anchors

完成点云 `P'` 只提供初始化和 LiDAR descriptor：70% density selection + 30% random coverage；nuScenes/KITTI-360 分别使用 25,600/40,000 Gaussians；每轴初始 scale 随机取 `U[0.20,1.00]`。最终 FFN 从 LiDAR/image fusion feature 同时预测 center residual、scale、rotation 和 semantic vector。论文没有 completed-point-to-center Chamfer、point-to-plane、anchor drift、free-space 或 surface consistency loss，也没有独立 opacity。

因此 `P'` 对 Gaussian geometry 是 soft initialization prior，不是训练全过程中的硬约束。

### Image injection

GAF 将 Gaussian center 投影到多相机特征图，用 LiDAR descriptor 预测局部 2D offsets，再经 Geo-VLAD、FiLM 与 cross-attention 聚合图像 token。该机制改善几何对齐的语义采样，但 fused feature 最终仍可共同修改 `delta center / scale / rotation / semantics`；没有 geometry stop-gradient、位置残差界或持续 surface loss，不能证明图像不会破坏 3D geometry。

Eq. (9) 将 `pix_{i,v}` 定义为 `Pi_v(mu_i)`，Eq. (15) 又以二者距离构造 reprojection weight；按印刷公式该权重恒为 1。无官方代码时不推断作者未写出的实现。

### End-to-end 状态

补充材料只明确 LCD 先预训练 20 epochs，随后与 Gau-Occ 做 joint optimization；主文最终监督为 occupancy CE + Lovasz。公开材料没有明确 LCD 是否解冻、联合阶段是否保留 diffusion loss、图像 backbone 是否冻结，以及 occupancy gradient 是否穿过 completion sampling。故只能写“两阶段后联合优化”，不能声称端到端梯度的具体路径。

## 对 WorldSim V7.1 的迁移决策

1. 借鉴 target-first，不照搬 diffusion：现有 593 个 Actor 训练样本不足以合理复现 1000-step scene diffuser，首个 M6 使用确定性 conditional anchor relocation。
2. 动态 Actor target 必须由每帧 GT box/pose 变换到 actor-canonical frame后累积。现有 nuScenes/AV2 compiler 已对每帧点和 sensor origin执行该变换；禁止退回只做 ego alignment。
3. M6 首先只攻 geometry：模型只看 build evidence；target/future LiDAR 只构造训练标签。为 completion candidate 构造 target surface match和局部 target scale，直接监督最终 center；同时保留 symmetric Chamfer、literal first-hit/free-space。observed anchors 不进入可学习 head。
4. UNKNOWN 不作为部署硬删除动作。首轮不接 image、semantic、motion/hazard head。
5. 若 geometry 通过，图像只更新 semantic descriptor，对 center/scale/rotation stop-gradient；若未来开放视觉 geometry residual，必须持续保留 geometry/free-space supervision 并限制残差。
6. 下游 occupancy 联合训练必须持续保留 geometry/free-space loss，不能退化为只优化 CE/Lovasz 后再用 filter 修复。

## 冻结的 M6 最小实现

- base：冻结 canonical M5 relocation，作为已通过 development 的几何起点；
- train target：原始 completion candidate 到 actor-canonical target endpoints 的确定性最近表面匹配；匹配点局部 8-NN 中位距离提供 isotropic Gaussian scale target；
- learned output：build-only feature 条件的 bounded 3D center residual + log scale；最终 center 是 M5 center 加 residual；
- Stage G：直接 center Huber + target-local point-to-plane + log-scale Huber + symmetric Chamfer；
- Stage P：上述 GT geometry loss继续回传，再加入 differentiable literal first-return；observed anchors 始终原样并入最终 surface；
- 禁止：UNKNOWN mask、image/semantic/hazard输入、只在部署阶段执行的几何 filter、diffusion、loss/seed sweep。

该实现回答“显式 GT center supervision 能否在不牺牲 first-return 的情况下强化 M5”，不同时回答 appearance、motion prediction 或完整 occupancy。
